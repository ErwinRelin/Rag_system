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
import json
import tree_sitter_c_sharp as tscsharp
from tree_sitter import Language, Parser

# ── Setup ────────────────────────────────────────────────────────────────────


CS_LANGUAGE = Language(tscsharp.language())

parser = Parser(CS_LANGUAGE)

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

def find_nodes(node, node_type):
    results = []

    def walk(n):
        if n.type == node_type:
            results.append(n)

        for child in n.children:
            walk(child)

    walk(node)
    return results

def build_type_map_for_directory(dir_path):
    """
    Pass 1:
    Maps instance variables to their declared types.

    Example:
        private readonly EmployeeRepository _repository;

    becomes:

        {
            "_repository": "EmployeeRepository"
        }
    """

    type_map = {}

    field_pattern = (
        r'(?:public|private|protected|internal)\s+'
        r'(?:readonly\s+)?'
        r'([\w<>\s]+?)\s+'
        r'([_a-zA-Z]\w*)\s*'
        r'(?:=[\s\S]*?)?;'
    )

    for filename in os.listdir(dir_path):

        if not filename.endswith(".cs"):
            continue

        file_path = os.path.join(dir_path, filename)

        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            source = f.read()

        matches = re.findall(field_pattern, source)

        for csharp_type, variable_name in matches:

            type_map[variable_name.strip()] = (
                csharp_type
                .replace("?", "")
                .strip()
            )

    return type_map

def build_type_map_for_directory(dir_path):
    """
    Pass 1:
    Maps instance variables to their declared types.

    Example:
        private readonly EmployeeRepository _repository;

    becomes:

        {
            "_repository": "EmployeeRepository"
        }
    """

    type_map = {}

    field_pattern = (
        r'(?:public|private|protected|internal)\s+'
        r'(?:readonly\s+)?'
        r'([\w<>\s]+?)\s+'
        r'([_a-zA-Z]\w*)\s*'
        r'(?:=[\s\S]*?)?;'
    )

    for filename in os.listdir(dir_path):

        if not filename.endswith(".cs"):
            continue

        file_path = os.path.join(dir_path, filename)

        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            source = f.read()

        matches = re.findall(field_pattern, source)

        for csharp_type, variable_name in matches:

            type_map[variable_name.strip()] = (
                csharp_type
                .replace("?", "")
                .strip()
            )

    return type_map

def extract_method_calls(method_node, source, type_map):

    calls = set()

    def get_method_name(expr):

        if expr is None:
            return None

        text = expr.text.decode("utf8").strip()

        if "." in text:
            return text.split(".")[-1]

        return text

    def walk(node):

        if node.type == "invocation_expression":

            function_node = node.child_by_field_name("function")

            if function_node:

                raw_text = function_node.text.decode("utf8").strip()

                method_name = get_method_name(function_node)

                if not method_name:
                    return

                if "." in raw_text:

                    component = raw_text.split(".")[0]

                    resolved_class = type_map.get(
                        component,
                        component
                    )

                    calls.add(
                        f"{resolved_class}.{method_name}"
                    )

                else:

                    calls.add(method_name)

        for child in node.children:
            walk(child)

    walk(method_node)

    return sorted(calls)

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
def process_csharp_class(class_node, source: bytes, file_path: str, usings=None,type_map = None) -> list:
    usings = usings or []
    method_chunks = []
    type_map = type_map or {}

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

                called_methods = extract_method_calls(child, source, type_map)

                method_chunks.append({
                    "type": chunk_type,
                    "class_name": class_name,
                    "method_name": member_name,
                    "called_methods": called_methods,
                    "language": "c-sharp",
                    "file_path": file_path,
                    "usings": usings,
                    "text": format_chunk_text(
                        node_text(child, source),
                        file_path,
                        class_name=class_name,
                        method_name=member_name
                    ),
                    "start_line": method_start,
                    "end_line": method_end,
                    })

    # FIX #3: Always return class_chunk + method_chunks.
    # The old `if len(method_chunks) == 1: return method_chunks` dropped the
    # class chunk when a class had exactly one method — now removed.
    return [class_chunk] + method_chunks


# ── Main Entrypoint ──────────────────────────────────────────────────────────

# FIX #4: Processes a single .cs file. The old version looped over the string
# characters of the path and returned inside the loop (early exit on first char).
def chunk_csharp_file(file_path: str, type_map = None) -> list:

    type_map = type_map or {}

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
            "end_line":   raw_using_block.count('\n') + 1,

        })

    # 2. Structural tree walker
    def find_code_blocks(node):
        if node.type in ("class_declaration", "interface_declaration",
                         "struct_declaration", "record_declaration"):
            chunks.extend(process_csharp_class(node, source, file_path, usings=usings, type_map=type_map))
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
def chunk_csharp_directory(dir_path: str):

    type_map = build_type_map_for_directory(
        dir_path
    )

    all_chunks = []

    for filename in os.listdir(dir_path):

        if filename.endswith(".cs"):

            file_path = os.path.join(
                dir_path,
                filename
            )

            all_chunks.extend(
                chunk_csharp_file(
                    file_path,
                    type_map=type_map
                )
            )

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
        template = template = """You are an expert C# code generation assistant.

Use only the provided context.

Return only valid JSON.

Context:
{context}

Question:
{question}

Schema:

{{
  "status": "SUCCESS | INSUFFICIENT_CONTEXT",
  "changes": [
    {{
      "file_path": "",
      "target_scope": "",
      "operation": "",
      "anchor": "",
      "new_code": ""
    }}
  ]
}}

File Path Rules:

- Use absolute file paths.
- Use forward slashes (/) only.
- Never use backslashes (\).

Example:
C:/Users/Erwin/Desktop/rag_system/codebase_rag/DummyApplication/EmployeeService.cs

target_scope must contain the actual class or method name.

Examples:
- Employee
- EmployeeService
- Program
- Main
- Employee.DisplayInfo

Never use generic values:
- CLASS
- METHOD
- FUNCTION
- CODE

Valid operations:
- APPEND_TO_CLASS
- INSERT_AFTER
- INSERT_BEFORE
- REPLACE_METHOD
- CREATE_FILE

Rules:
- Use existing files unless CREATE_FILE.
- Use specific target scopes.
- Return only JSON."""

        limit = 8 if generation else 5

        chain = (
            {"context": lambda q: self.retriever(q, limit=limit), "question": RunnablePassthrough()}
            | ChatPromptTemplate.from_template(template)
            | self.llm
            | StrOutputParser()
        )

        return chain.invoke(question)

def apply_codebase_changes_with_logging(json_data):
    data = json.loads(json_data)
    
    if data.get("status") != "SUCCESS":
        print(f"❌ Aborting: JSON status is '{data.get('status')}', not 'SUCCESS'.")
        return

    print(f"🔍 Found {len(data['changes'])} total task(s) to process.")

    for idx, change in enumerate(data["changes"], 1):
        file_path = change["file_path"]
        operation = change["operation"]
        target_scope = change["target_scope"]
        anchor = change["anchor"]
        new_code = change["new_code"]
        
        print(f"\n--- Task {idx}: {operation} on {target_scope} ---")
        print(f"📂 Checking path: {file_path}")

        if not os.path.exists(file_path):
            print(f"❌ Error: Python cannot locate this file on disk! Check folder path permissions.")
            continue

        with open(file_path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        modified_lines = []
        insert_index = -1
        
        # --- PROCESSING APPEND_TO_CLASS ---
        if operation == "APPEND_TO_CLASS":
            found_class = False
            brace_count = 0
            
            for line_idx, line in enumerate(lines):
                if f"class {target_scope}" in line:
                    found_class = True
                if found_class:
                    brace_count += line.count("{")
                    brace_count -= line.count("}")
                    if brace_count == 0 and "}" in line:
                        insert_index = line_idx
                        break
            
            if insert_index != -1:
                clean_code = new_code.strip()
                padded_code = f"        {clean_code}\n"
                lines.insert(insert_index, padded_code)
                modified_lines = lines
                print(f"✅ Matched class boundary! Ready to insert at line {insert_index + 1}.")
            else:
                print(f"❌ Error: Could not parse brace structure for class '{target_scope}'.")
                modified_lines = lines

        # --- PROCESSING INSERT_AFTER METHOD BOUNDARY ---
        elif operation == "INSERT_AFTER":
            found_anchor = False
            brace_count = 0
            method_started = False

            for line_idx, line in enumerate(lines):
                if anchor in line:
                    found_anchor = True
                
                if found_anchor:
                    if "{" in line:
                        brace_count += line.count("{")
                        method_started = True
                    if "}" in line:
                        brace_count -= line.count("}")
                    
                    if method_started and brace_count == 0:
                        insert_index = line_idx + 1
                        break
            
            if insert_index != -1:
                clean_method = new_code.strip('\n')
                lines.insert(insert_index, f"\n{clean_method}\n")
                modified_lines = lines
                print(f"✅ Found closing brace of method! Ready to insert at line {insert_index + 1}.")
            else:
                print(f"❌ Error: Could not locate method signature string: '{anchor}'")
                modified_lines = lines

        # Write to temp file and overwrite safely
        if insert_index != -1:
            with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", delete=False) as temp_file:
                temp_path = temp_file.name
                temp_file.writelines(modified_lines)

            try:
                os.replace(temp_path, file_path)
                print(f"🎉 Successfully modified and updated file: {os.path.basename(file_path)}")
            except Exception as e:
                if os.path.exists(temp_path):
                    os.unlink(temp_path)
                print(f"❌ File Swap Failure: {e}")

def verify_dotnet_syntax():
    print("\n Starting .NET multi-project syntax verification...")
    
    # This is the exact PowerShell script compressed into a single-line execution string
    ps_script = (
        "Get-ChildItem -Recurse -Filter *.csproj | ForEach-Object { "
        "Write-Host 'Checking Syntax:' $_.Name; "
        "dotnet build $_.FullName --no-incremental /p:BuildProjectReferences=false /v:quiet "
        "}"
    )
    
    # We call powershell.exe and pass the script securely via the -Command argument
    result = subprocess.run(
        ["powershell.exe", "-Command", ps_script],
        capture_output=True,
        text=True
    )
    
    # Print the terminal outputs so you can see any compilation syntax errors
    if result.stdout:
        print(result.stdout)
    if result.stderr:
        print(f"System/Shell Errors:\n{result.stderr}")
        
    # Check if the execution context encountered errors
    if "error CS" in result.stdout or result.returncode != 0:
        print("❌ Syntax verification failed! Errors found in the codebase.")
        return False
    else:
        print("✅ Syntax verification passed! All projects are clean.")
        return True


# def extract_patch(llm_response: str) -> str | None:
#     # Try <git_patch> tags first
#     match = re.search(r"<git_patch>(.*?)</git_patch>", llm_response, re.DOTALL)
#     if match:
#         return match.group(1).strip()

#     # Fallback: ```git_patch or ```diff code fences
#     match = re.search(r"```(?:git_patch|diff)(.*?)```", llm_response, re.DOTALL)
#     if match:
#         return match.group(1).strip()

#     return None

# def validate_patch(patch_content: str) -> dict:
#     lines = patch_content.splitlines()
#     errors = []
#     in_hunk = False

#     for i, line in enumerate(lines, 1):
#         if line.startswith("@@"):
#             in_hunk = True
#         elif line.startswith(("---", "+++")):
#             continue
#         elif in_hunk and not line.startswith((" ", "+", "-", "\\")):
#             errors.append(f"Line {i}: invalid prefix — '{line[:40]}'")

#     return {"valid": len(errors) == 0, "errors": errors}

#     return {"valid": len(errors) == 0, "errors": errors}

# def fix_patch(patch_content: str) -> str:
#     lines = patch_content.splitlines()
#     fixed = []
#     in_hunk = False
#     hunk_pattern = re.compile(r"^(@@ -)(\d+)(,\d+)? (\+)(\d+)(,\d+)? (@@.*)")
#     pending_hunk = None
#     hunk_body = []

#     def flush_hunk():
#         if pending_hunk is None:
#             return []
#         old_start, new_start, suffix = pending_hunk
#         old_count = sum(1 for l in hunk_body if l.startswith(" ") or l.startswith("-"))
#         new_count = sum(1 for l in hunk_body if l.startswith(" ") or l.startswith("+"))
#         header = f"@@ -{old_start},{old_count} +{new_start},{new_count} {suffix}"
#         return [header] + hunk_body

#     for line in lines:
#         if line.startswith("diff --git"):
#             continue

#         m = hunk_pattern.match(line)
#         if m:
#             fixed.extend(flush_hunk())
#             hunk_body = []
#             in_hunk = True
#             pending_hunk = (m.group(2), m.group(5), m.group(7))
#         elif in_hunk:
#             if line == "":
#                 hunk_body.append(" ")
#             elif line[0] not in ("+", "-", " ", "\\"):
#                 hunk_body.append(" " + line)  # force context prefix if missing
#             else:
#                 hunk_body.append(line)
#         else:
#             fixed.append(line)

#     fixed.extend(flush_hunk())
#     return "\n".join(fixed) + "\n"  # patch must end with newline

# def apply_patch(patch_content: str, repo_path: str) -> dict:
#     with tempfile.NamedTemporaryFile(mode="w", suffix=".patch", delete=False, encoding="utf-8") as f:
#         f.write(patch_content)
#         patch_file = f.name

#     try:
#         check = subprocess.run(
#             ["git", "apply", "--check", "--whitespace=fix", patch_file],
#             cwd=repo_path, capture_output=True, text=True
#         )
#         if check.returncode != 0:
#             return {"success": False, "error": check.stderr}

#         result = subprocess.run(
#             ["git", "apply", "--whitespace=fix", patch_file],
#             cwd=repo_path, capture_output=True, text=True
#         )
#         return {
#             "success": result.returncode == 0,
#             "output": result.stdout,
#             "error": result.stderr or None
#         }
#     finally:
#         os.unlink(patch_file)

# ── Entry Point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    local_llm = ChatOllama(model="devstral-small-2:24b")
    rag_pipeline = CodebaseRAGPipeline(llm_instance=local_llm)

    rag_pipeline.clear_collection()

    chunks = chunk_csharp_directory(r"C:\Users\Erwin\Desktop\rag_system\codebase_rag\DummyApplication")
    texts    = [c["text"] for c in chunks]
    metadata = [{k: v for k, v in c.items() if k != "text"} for c in chunks]
    print("\n========== GENERATED CHUNKS ==========\n")

    for i, chunk in enumerate(chunks, start=1):

        print(f"\n----- CHUNK {i} -----")

        for key, value in chunk.items():

            if key == "text":
                print(f"\nTEXT:\n{value}")

            else:
                print(f"{key}: {value}")

        print("\n-----------------------")
    rag_pipeline.store_chunks(texts, metadata)

    user_query = "Add employee hire date support and display it in audit reports."
    repo_path = r"C:\Users\Erwin\Desktop\rag_system\codebase_rag\DummyApplication"

    ai_response = rag_pipeline.ask(user_query, generation=True)
    print(f"\n--- LLM Response ---\n{ai_response}\n")

    # patch = extract_patch(ai_response)

    # if patch:
    #     patch = fix_patch(patch)  # auto-fix empty context lines
    #     print(f"--- Extracted Patch ---\n{patch}\n")

    #     validation = validate_patch(patch)
    #     if not validation["valid"]:
    #         print("Patch validation failed:")
    #         for err in validation["errors"]:
    #             print(f"  - {err}")
    #     else:
    #         result = apply_patch(patch, repo_path)
    #         if result["success"]:
    #             print("Patch applied successfully")
    #         else:
    #             print(f"Patch failed:\n{result['error']}")
    # else:
    #     print("No patch found — model may have returned INSUFFICIENT_CONTEXT")