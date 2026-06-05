from tree_sitter import Language, Parser
import os
import uuid
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from sentence_transformers import SentenceTransformer
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_qdrant import QdrantVectorStore
import ollama


# ── Setup ────────────────────────────────────────────────────────────────────
PROJECT_ROOT   = r"C:\Users\Erwin\Desktop\rag_system"
CSHARP_GRAMMAR = os.path.join(PROJECT_ROOT, "tree-sitter-c-sharp")
BUILD_PATH     = os.path.join(PROJECT_ROOT, "build", "csharp.dll")

if os.path.exists(BUILD_PATH):
    try:
        os.remove(BUILD_PATH)
    except Exception:
        pass

os.makedirs(os.path.dirname(BUILD_PATH), exist_ok=True)
Language.build_library(BUILD_PATH, [CSHARP_GRAMMAR])

CSHARP_LANGUAGE = Language(BUILD_PATH, "c_sharp")
parser = Parser()
parser.set_language(CSHARP_LANGUAGE)

# ── Helpers ──────────────────────────────────────────────────────────────────
def node_text(node, source: bytes) -> str:
    return source[node.start_byte:node.end_byte].decode("utf-8")

def get_child_by_type(node, child_type: str):
    for child in node.children:
        if child.type == child_type:
            return child
    return None

# Cleaned up formatting helper: removed dependencies/usings loops entirely
def format_chunk_text(text, file_path, class_name=None, method_name=None):
    header_lines = [f"// FILE: {file_path}"]
        
    if class_name:
        header_lines.append(f"// CLASS: {class_name}")
    if method_name:
        header_lines.append(f"// METHOD: {method_name}")
        
    header = "\n".join(header_lines) + "\n\n"
    return header + text

# ── Extractors ───────────────────────────────────────────────────────────────
def get_csharp_usings_and_block(tree, source: bytes):
    using_list = []
    min_byte, max_byte = float('inf'), 0

    def traverse(node):
        nonlocal min_byte, max_byte
        if node.type == "using_directive":
            if node.start_byte < min_byte: min_byte = node.start_byte
            if node.end_byte > max_byte:   max_byte = node.end_byte
            
            raw_statement = source[node.start_byte:node.end_byte].decode("utf-8").strip()
            if raw_statement not in using_list:
                using_list.append(raw_statement)
        for child in node.children:
            traverse(child)

    traverse(tree.root_node)
    if min_byte == float('inf'): return [], None
    return using_list, source[int(min_byte):int(max_byte)].decode("utf-8")

def process_csharp_class(class_node, source: bytes, file_path: str, usings=None) -> list:
    usings = usings or []
    method_chunks = []
    
    name_node = get_child_by_type(class_node, "identifier")
    class_name = node_text(name_node, source) if name_node else "UnknownContainer"
    
    # 1. Prepare the Class Chunk text without passing dependencies to the formatter
    class_raw_text = node_text(class_node, source)
    class_formatted_text = format_chunk_text(class_raw_text, file_path, class_name=class_name)
    
    class_chunk = {
        "type":       "interface" if class_node.type == "interface_declaration" else "class",
        "class_name": class_name,
        "language": "c-sharp",
        "file_path":  file_path,
        "usings":     usings, # Saved in database metadata array only
        "text":       class_formatted_text,
        "start_line": class_node.start_point,
        "end_line":   class_node.end_point
    }
    
    if class_node.type == "interface_declaration":
        return [class_chunk]

    # 2. Extract Methods & Constructors
    dec_list = get_child_by_type(class_node, "declaration_list")
    if dec_list:
        for child in dec_list.children:
            if child.type in ("method_declaration", "constructor_declaration"):
                name_node = get_child_by_type(child, "identifier")
                member_name = node_text(name_node, source) if name_node else "unknown_member"
                chunk_type = "constructor" if child.type == "constructor_declaration" else "method"

                method_start = child.start_point[0] + 1
                method_end = child.end_point[0] + 1
                
                method_chunks.append({
                    "type":        chunk_type,
                    "class_name":  class_name,
                    "method_name": member_name,
                    "language": "c-sharp",
                    "file_path":   file_path,
                    "usings":      usings, # Saved in database metadata array only
                    "text":        format_chunk_text(node_text(child, source), file_path, class_name=class_name, method_name=member_name),
                    "start_line":  method_start,
                    "end_line":    method_end
                })

    if len(method_chunks) == 1:
        return method_chunks

    return [class_chunk] + method_chunks


# ── Main Entrypoint ──────────────────────────────────────────────────────────
def chunk_csharp_file(file_path: str) -> list:
    with open(file_path, "rb") as f:
        source = f.read()

    tree = parser.parse(source)
    chunks = []
    
    # 1. Gather all top-level `using` statements into their single standalone chunk
    usings, raw_using_block = get_csharp_usings_and_block(tree, source)
    if raw_using_block:
        chunks.append({
            "type":      "imports",
            "language": 'c-sharp',
            "file_path": file_path,
            "text":      f"// FILE: {file_path}\n// USINGS\n\n{raw_using_block}",
            "usings":    usings,
            "start_line": 1,
            "end_line": raw_using_block.count('\n') + 1
        })
        
    # 2. Structural tree walker
    def find_code_blocks(node):
        if node.type in ("class_declaration", "interface_declaration", "struct_declaration", "record_declaration"):
            chunks.extend(process_csharp_class(node, source, file_path, usings=usings))
            return

        elif node.type in ("namespace_declaration", "file_scoped_namespace_declaration"):
            for child in node.children:
                find_code_blocks(child)
            return
            
        for child in node.children:
            find_code_blocks(child)

    find_code_blocks(tree.root_node)
    return chunks

class CodebaseRAGPipeline:
    def __init__(self, llm_instance, collection_name="codebase_rag"):
        self.qdrant_client = QdrantClient(url="http://localhost:6333")
        self.collection_name = collection_name
        self.llm = llm_instance
        self.embeddings_model = OllamaEmbeddings(model="nomic-embed-text")
        self._ensure_collection_exists()

    def _ensure_collection_exists(self):
        existing_names = [c.name for c in self.qdrant_client.get_collections().collections]
        if self.collection_name not in existing_names:
            self.qdrant_client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=768, distance=Distance.COSINE)
            )

    def clear_collection(self):
        self.qdrant_client.delete_collection(self.collection_name)
        self._ensure_collection_exists()
        print(f"Collection '{self.collection_name}' cleared and recreated.")

    def store_chunks(self, chunks: list, metadata_list: list = None):  # ← ADD THIS METHOD
        points = []
        for idx, text in enumerate(chunks):
            payload = metadata_list[idx] if metadata_list else {}
            payload["page_content"] = text

            vector = self.embeddings_model.embed_query(f"search_document: {text}")

            point = PointStruct(
                id=str(uuid.uuid4()),
                vector=vector,
                payload=payload
            )
            points.append(point)

        self.qdrant_client.upsert(collection_name=self.collection_name, points=points)
        print(f"Stored {len(points)} chunks into '{self.collection_name}'")

    def retriever(self, question: str, limit: int = 5) -> str:
        prefixed_query = f"search_query: {question}"
        query_vector = self.embeddings_model.embed_query(prefixed_query)

        search_results = self.qdrant_client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            limit=limit
        )

        formatted_blocks = []
        for idx, hit in enumerate(search_results.points):
            payload = hit.payload if hit.payload else {}
            block = (
                f"--- Code Chunk {idx + 1} ---\n"
                f"File: {payload.get('file_path', 'Unknown File')}\n"
                f"Content:\n{payload.get('page_content', 'No content.')}\n"  # ← fixed key
            )
            formatted_blocks.append(block)

        return "\n".join(formatted_blocks)

    def ask(self, question: str, generation: bool=False) -> str:
        

        template = """You are an expert C# code assistant. You can both explain existing code and generate new code.
Operate in one of two modes based on the question:

**EXPLANATION MODE** (when asked to explain, describe, or trace existing code):
- Reference specific class names, method names, or file paths when relevant
- Walk through the logic step by step
- Synthesize information across multiple chunks if needed
- If the answer is not present in the context, say "This information is not available in the provided code chunks."
- Do NOT hallucinate method names, variables, or behavior not shown in the context

**GENERATION MODE** (when asked to create, implement, add, or modify code):
- Use the provided code chunks as context to match the existing code style, patterns, and architecture
- Generate clean, complete C# code that integrates naturally with the existing codebase
- Follow the same naming conventions, design patterns, and structure seen in the context
- Add brief inline comments explaining key decisions
- If the request is ambiguous, state your assumptions before generating

Rules that apply to BOTH modes:
- Display only the answer and nothing else
- Do not include any preamble or closing remarks

Context (relevant code chunks):
{context}

Question: {question}

Answer:"""

        limit = 8 if generation else 5

        chain = (
            {"context": lambda q: self.retriever(q, limit=limit), "question": RunnablePassthrough()}
            | ChatPromptTemplate.from_template(template)
            | self.llm
            | StrOutputParser()
        )

        return chain.invoke(question)



if __name__ == "__main__":
    local_llm = ChatOllama(model="qwen2.5-custom:latest")
    rag_pipeline = CodebaseRAGPipeline(llm_instance=local_llm)

    rag_pipeline.clear_collection()  # ← wipe before indexing

    chunks = chunk_csharp_file(r"C:\Users\Erwin\Desktop\rag_system\codebase_rag\files\dummy_framework.cs")
    texts = [c["text"] for c in chunks]
    metadata = [{k: v for k, v in c.items() if k != "text"} for c in chunks]
    rag_pipeline.store_chunks(texts, metadata)

    user_query = "Add a DeleteOrder method across all necessary classes and interfaces"
    ai_response = rag_pipeline.ask(user_query, generation=True)
    print(f"\nAI Response:\n{ai_response}")

    # try:
    #     # 2. Run your query logic
    #     user_query = "Add a DeleteOrder method to the codebase. Implement it across all necessary classes and interfaces."
    #     ai_response = rag_pipeline.ask(user_query)
    #     print(f"\nAI Response:\n{ai_response}")
        
    # finally:
    #     # 3. Force clean closure of database files before the script exits
    #     print("\nClosing Qdrant database connections safely...")
    #     rag_pipeline.qdrant_client.close()
        
