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
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage
from langchain.agents import create_agent
import re
import subprocess
import tempfile


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


# FIX #1 & #2: Removed the erroneous `for file_path in folder_path` loop.
# `file_path` is now a plain string parameter, used directly.
def process_csharp_class(class_node, source: bytes, file_path: str, usings=None) -> list:
    usings = usings or []
    method_chunks = []

    name_node = get_child_by_type(class_node, "identifier")
    class_name = node_text(name_node, source) if name_node else "UnknownContainer"

    # 1. Prepare the Class Chunk
    class_raw_text = node_text(class_node, source)
    class_formatted_text = format_chunk_text(class_raw_text, file_path, class_name=class_name)

    class_chunk = {
        "type":       "interface" if class_node.type == "interface_declaration" else "class",
        "class_name": class_name,
        "language":   "c-sharp",
        "file_path":  file_path,
        "usings":     usings,
        "text":       class_formatted_text,
        "start_line": class_node.start_point[0] + 1,
        "end_line":   class_node.end_point[0] + 1,
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
                method_end   = child.end_point[0] + 1

                method_chunks.append({
                    "type":        chunk_type,
                    "class_name":  class_name,
                    "method_name": member_name,
                    "language":    "c-sharp",
                    "file_path":   file_path,
                    "usings":      usings,
                    "text":        format_chunk_text(
                                       node_text(child, source),
                                       file_path,
                                       class_name=class_name,
                                       method_name=member_name
                                   ),
                    "start_line":  method_start,
                    "end_line":    method_end
                })

    # FIX #3: Always return class_chunk + method_chunks.
    # The old `if len(method_chunks) == 1: return method_chunks` dropped the
    # class chunk when a class had exactly one method — now removed.
    return [class_chunk] + method_chunks


# ── Main Entrypoint ──────────────────────────────────────────────────────────

# FIX #4: Processes a single .cs file. The old version looped over the string
# characters of the path and returned inside the loop (early exit on first char).
def chunk_csharp_file(file_path: str) -> list:
    with open(file_path, "rb") as f:
        source = f.read()

    tree = parser.parse(source)
    chunks = []

    # 1. Gather all top-level `using` statements
    usings, raw_using_block = get_csharp_usings_and_block(tree, source)
    if raw_using_block:
        chunks.append({
            "type":       "imports",
            "language":   "c-sharp",
            "file_path":  file_path,
            "text":       f"// FILE: {file_path}\n// USINGS\n\n{raw_using_block}",
            "usings":     usings,
            "start_line": 1,
            "end_line":   raw_using_block.count('\n') + 1
        })

    # 2. Structural tree walker
    def find_code_blocks(node):
        if node.type in ("class_declaration", "interface_declaration",
                         "struct_declaration", "record_declaration"):
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


# NEW: Walks a directory and chunks every .cs file found.
def chunk_csharp_directory(dir_path: str) -> list:
    all_chunks = []
    for filename in os.listdir(dir_path):
        if filename.endswith(".cs"):
            file_path = os.path.join(dir_path, filename)
            all_chunks.extend(chunk_csharp_file(file_path))
    return all_chunks


# ── RAG Pipeline ─────────────────────────────────────────────────────────────
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
        info = self.qdrant_client.get_collection(self.collection_name)
        print("Points after clear:", info.points_count)

    def store_chunks(self, chunks: list, metadata_list: list = None):
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
                f"Lines: {payload.get('start_line', '?')}–{payload.get('end_line', '?')}\n"
                f"Content:\n{payload.get('page_content', 'No content.')}\n"
            )
            formatted_blocks.append(block)

        return "\n".join(formatted_blocks)

    def ask(self, question: str, generation: bool = False) -> str:
        limit = 8 if generation else 5

        context = self.retriever(question, limit=limit)
        print(f"\n--- Retrieved Context ---\n{context}\n")

        template = """You are an expert C# software engineer.

    Use the provided context as your primary reference. You may infer standard C# patterns to complete the implementation.

    Context:
    {context}

    Task:
    {question}

    Generate a git patch for the required changes.
    The line numbers in the context are the REAL line numbers in the file. Use them directly in @@ hunk headers.

    Output format:

    <git_patch>
    --- a/<filepath>
    +++ b/<filepath>
    @@ -old_start,old_count +new_start,new_count @@
    [3 context lines before change]
    -[removed line]
    +[added line]
    [3 context lines after change]
    </git_patch>

    Rules:
    - Use forward slashes in file paths (C:/Users/...)
    - Only modify what the task requires
    - Hunk line counts must be exact — wrong counts will break the patch
    - Preserve all indentation and spacing
    - Multiple files = multiple diff blocks inside one <git_patch>
    - New file: use --- /dev/null and +++ b/<filepath>
    - If context is insufficient to generate a valid patch, reply with: INSUFFICIENT_CONTEXT"""

        prompt = ChatPromptTemplate.from_template(template)
        chain = prompt | self.llm | StrOutputParser()

        return chain.invoke({"context": context, "question": question})


def extract_patch(llm_response: str) -> str | None:
    # Try <git_patch> tags first
    match = re.search(r"<git_patch>(.*?)</git_patch>", llm_response, re.DOTALL)
    if match:
        return match.group(1).strip()

    # Fallback: ```git_patch or ```diff code fences
    match = re.search(r"```(?:git_patch|diff)(.*?)```", llm_response, re.DOTALL)
    if match:
        return match.group(1).strip()

    return None

def validate_patch(patch_content: str) -> dict:
    lines = patch_content.splitlines()
    errors = []
    in_hunk = False

    for i, line in enumerate(lines, 1):
        if line.startswith("@@"):
            in_hunk = True
        elif line.startswith(("---", "+++")):
            continue
        elif in_hunk and not line.startswith((" ", "+", "-", "\\")):
            errors.append(f"Line {i}: invalid prefix — '{line[:40]}'")

    return {"valid": len(errors) == 0, "errors": errors}

    return {"valid": len(errors) == 0, "errors": errors}

def fix_patch(patch_content: str) -> str:
    lines = patch_content.splitlines()
    fixed = []
    in_hunk = False
    hunk_pattern = re.compile(r"^(@@ -)(\d+)(,\d+)? (\+)(\d+)(,\d+)? (@@.*)")
    pending_hunk = None
    hunk_body = []

    def flush_hunk():
        if pending_hunk is None:
            return []
        old_start, new_start, suffix = pending_hunk
        old_count = sum(1 for l in hunk_body if l.startswith(" ") or l.startswith("-"))
        new_count = sum(1 for l in hunk_body if l.startswith(" ") or l.startswith("+"))
        header = f"@@ -{old_start},{old_count} +{new_start},{new_count} {suffix}"
        return [header] + hunk_body

    for line in lines:
        if line.startswith("diff --git"):
            continue

        m = hunk_pattern.match(line)
        if m:
            fixed.extend(flush_hunk())
            hunk_body = []
            in_hunk = True
            pending_hunk = (m.group(2), m.group(5), m.group(7))
        elif in_hunk:
            if line == "":
                hunk_body.append(" ")
            elif line[0] not in ("+", "-", " ", "\\"):
                hunk_body.append(" " + line)  # force context prefix if missing
            else:
                hunk_body.append(line)
        else:
            fixed.append(line)

    fixed.extend(flush_hunk())
    return "\n".join(fixed) + "\n"  # patch must end with newline

def apply_patch(patch_content: str, repo_path: str) -> dict:
    with tempfile.NamedTemporaryFile(mode="w", suffix=".patch", delete=False, encoding="utf-8") as f:
        f.write(patch_content)
        patch_file = f.name

    try:
        check = subprocess.run(
            ["git", "apply", "--check", "--whitespace=fix", patch_file],
            cwd=repo_path, capture_output=True, text=True
        )
        if check.returncode != 0:
            return {"success": False, "error": check.stderr}

        result = subprocess.run(
            ["git", "apply", "--whitespace=fix", patch_file],
            cwd=repo_path, capture_output=True, text=True
        )
        return {
            "success": result.returncode == 0,
            "output": result.stdout,
            "error": result.stderr or None
        }
    finally:
        os.unlink(patch_file)

# ── Entry Point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    local_llm = ChatOllama(model="qwen2.5-coder:14b")
    rag_pipeline = CodebaseRAGPipeline(llm_instance=local_llm)

    rag_pipeline.clear_collection()

    chunks = chunk_csharp_directory(r"C:\Users\Erwin\Desktop\rag_system\codebase_rag\DummyApplication")
    texts    = [c["text"] for c in chunks]
    metadata = [{k: v for k, v in c.items() if k != "text"} for c in chunks]
    rag_pipeline.store_chunks(texts, metadata)

    user_query = "Add a method to EmployeeService that returns all employees with a configured email address."
    repo_path = r"C:\Users\Erwin\Desktop\rag_system\codebase_rag\DummyApplication"

    ai_response = rag_pipeline.ask(user_query, generation=True)
    print(f"\n--- LLM Response ---\n{ai_response}\n")

    patch = extract_patch(ai_response)

    if patch:
        patch = fix_patch(patch)  # auto-fix empty context lines
        print(f"--- Extracted Patch ---\n{patch}\n")

        validation = validate_patch(patch)
        if not validation["valid"]:
            print("Patch validation failed:")
            for err in validation["errors"]:
                print(f"  - {err}")
        else:
            result = apply_patch(patch, repo_path)
            if result["success"]:
                print("Patch applied successfully")
            else:
                print(f"Patch failed:\n{result['error']}")
    else:
        print("No patch found — model may have returned INSUFFICIENT_CONTEXT")