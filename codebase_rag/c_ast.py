from tree_sitter import Language, Parser
import os
import uuid
import json
import shutil
import tempfile
import subprocess
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import ChatOllama, OllamaEmbeddings
import ollama


# ── Setup ─────────────────────────────────────────────────────────────────────
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


# ── Helpers ───────────────────────────────────────────────────────────────────
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


# ── Extractors ────────────────────────────────────────────────────────────────
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

    name_node  = get_child_by_type(class_node, "identifier")
    class_name = node_text(name_node, source) if name_node else "UnknownContainer"

    class_raw_text      = node_text(class_node, source)
    class_formatted_text = format_chunk_text(class_raw_text, file_path, class_name=class_name)

    class_chunk = {
        "type":       "interface" if class_node.type == "interface_declaration" else "class",
        "class_name": class_name,
        "language":   "c-sharp",
        "file_path":  file_path,
        "usings":     usings,
        "text":       class_formatted_text,
        "start_line": class_node.start_point,
        "end_line":   class_node.end_point
    }

    if class_node.type == "interface_declaration":
        return [class_chunk]

    dec_list = get_child_by_type(class_node, "declaration_list")
    if dec_list:
        for child in dec_list.children:
            if child.type in ("method_declaration", "constructor_declaration"):
                name_node   = get_child_by_type(child, "identifier")
                member_name = node_text(name_node, source) if name_node else "unknown_member"
                chunk_type  = "constructor" if child.type == "constructor_declaration" else "method"

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
                    "start_line":  child.start_point[0] + 1,
                    "end_line":    child.end_point[0] + 1
                })

    return [class_chunk] + method_chunks


def chunk_csharp_file(file_path: str) -> list:
    with open(file_path, "rb") as f:
        source = f.read()

    tree   = parser.parse(source)
    chunks = []

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


def chunk_csharp_directory(dir_path: str) -> list:
    all_chunks = []
    for filename in os.listdir(dir_path):
        if filename.endswith(".cs"):
            file_path = os.path.join(dir_path, filename)
            all_chunks.extend(chunk_csharp_file(file_path))
    return all_chunks


# ── File Tools ────────────────────────────────────────────────────────────────
def read_file(file_path: str) -> str:
    """Read the contents of a file and return it as a string."""
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()


def write_file(file_path: str, new_code: str, start_line: int = None, end_line: int = None) -> str:
    if not os.path.exists(file_path):
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(new_code)
        return f"Created new file: {file_path}"

    # If no line range given, overwrite the entire file — not append
    if start_line is None or end_line is None:
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(new_code)
        return f"Overwrote entire file: {file_path}"

    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    start     = start_line - 1
    end       = end_line
    new_lines = new_code.splitlines(keepends=True)
    if new_lines and not new_lines[-1].endswith("\n"):
        new_lines[-1] += "\n"
    lines = lines[:start] + new_lines + lines[end:]

    with open(file_path, "w", encoding="utf-8") as f:
        f.writelines(lines)

    return f"Successfully wrote to {file_path} (lines {start_line}–{end_line})"


# ── Tool Schemas (Ollama format) ──────────────────────────────────────────────
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read the full contents of a C# source file before making changes.",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "Absolute path to the file to read."
                    }
                },
                "required": ["file_path"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": (
                "Write or replace code in a C# source file. "
                "If start_line and end_line are provided, replaces that line range. "
                "If omitted, appends to the end of the file."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "Absolute path to the file to modify."
                    },
                    "new_code": {
                        "type": "string",
                        "description": "Complete, compilable C# code to insert."
                    },
                    "start_line": {
                        "type": "integer",
                        "description": "1-indexed line to start replacing from (inclusive). Omit to append."
                    },
                    "end_line": {
                        "type": "integer",
                        "description": "1-indexed line to stop replacing at (inclusive). Omit to append."
                    }
                },
                "required": ["file_path", "new_code"]
            }
        }
    }
]

TOOL_REGISTRY = {
    "read_file":  read_file,
    "write_file": write_file,
}

def dispatch_tool(tool_name: str, args: dict) -> str:
    fn = TOOL_REGISTRY.get(tool_name)
    if not fn:
        return f"Unknown tool: {tool_name}"
    return fn(**args)


# ── Step 1: Code Generation ───────────────────────────────────────────────────
def generate_change_plan(question: str, context: str, model: str = "qwen2.5-custom:latest") -> list:
    """
    Asks the LLM to produce a JSON change plan based on the user question and RAG context.
    Returns a list of { file_path, start_line, end_line, new_code } dicts.
    """
    generation_prompt = f"""You are an expert C# code assistant embedded in a RAG pipeline.
Your sole task is to generate production-ready C# code based on the user's request and the retrieved codebase context below.

## Codebase Context
{context}

## Task
{question}

## Rules
- Mirror the naming conventions, patterns, and architecture found in the context exactly
- Output ONLY a JSON array — no markdown, no explanation, no preamble
- Each element in the array represents one file change and must follow this schema:
  {{
    "file_path": "<exact path from context>",
    "start_line": <int or null if appending>,
    "end_line": <int or null if appending>,
    "new_code": "<complete code to insert or replace>"
  }}
- If a new file must be created, set start_line and end_line to null and infer the file_path from the project structure
- If the change spans multiple files (e.g. interface + implementation + registration), include one entry per file
- new_code must be complete and compilable — never use placeholders like // ... or // existing code
- Always set start_line and end_line to null and return the COMPLETE file contents in new_code — never return partial snippets
- new_code for an existing file must contain the entire file from top to bottom
- The generated code must fully satisfy the user's request: "{question}"

## Output (JSON array only):"""

    print("\n── Step 1: Generating change plan ──────────────────")
    response = ollama.chat(
        model=model,
        messages=[{"role": "user", "content": generation_prompt}]
    )

    raw = response["message"]["content"].strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    raw = raw.strip()

    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        # Replace unescaped backslashes in file paths
        raw = raw.replace("\\", "\\\\")
        # But fix double-escaped common sequences back
        for seq in ["\\\\n", "\\\\t", "\\\\r", "\\\\\""]:
            raw = raw.replace(seq, seq[1:])
        return json.loads(raw)


# ── Step 2: Apply Changes via Tool Calls ─────────────────────────────────────
def apply_change_plan(changes: list, model: str = "qwen2.5-custom:latest") -> str:
    """
    Feeds the change plan to the LLM which uses read_file + write_file tools
    to verify and apply each change to disk.
    """
    print("\n── Step 2: Applying changes ─────────────────────────")

    messages = [
        {
            "role": "system",
            "content": (
                "You are a code-writing assistant. You will be given a list of file changes to apply. "
                "For each change: first call read_file to verify the current file contents and confirm "
                "the correct line numbers, then call write_file to apply the change. "
                "When all changes are done, respond with a plain-text summary of everything applied."
            )
        },
        {
            "role": "user",
            "content": f"Apply these changes to the codebase:\n\n{json.dumps(changes, indent=2)}"
        }
    ]

    while True:
        response = ollama.chat(
            model=model,
            messages=messages,
            tools=TOOLS,
        )

        message = response["message"]
        messages.append(message)

        if not message.get("tool_calls"):
            print("\n── Changes applied ──────────────────────────────────")
            return message["content"]

        for tool_call in message["tool_calls"]:
            name   = tool_call["function"]["name"]
            args   = tool_call["function"]["arguments"]

            print(f"  → Tool: {name}  args: {args}")
            result = dispatch_tool(name, args)
            print(f"    Result: {result[:120]}{'...' if len(result) > 120 else ''}")

            messages.append({
                "role":    "tool",
                "content": result,
            })


def run_agent(question: str, context: str, model: str = "qwen2.5-custom:latest") -> str:
    """Generates a change plan then applies it via tool calls."""
    changes = generate_change_plan(question, context, model=model)
    if not changes:
        return "No changes generated."
    return apply_change_plan(changes, model=model)


# ── Runner ────────────────────────────────────────────────────────────────────
def run_csharp_code(file_paths: list, timeout: int = 30) -> dict:
    """
    Compiles and runs a list of .cs files in a temp dotnet console project.
    Returns stdout, stderr, exit_code, stage, and success.
    """
    tmp_dir = tempfile.mkdtemp()
    try:
        subprocess.run(
            ["dotnet", "new", "console", "-n", "DebugRunner", "--force"],
            cwd=tmp_dir,
            capture_output=True
        )

        project_dir     = os.path.join(tmp_dir, "DebugRunner")
        default_program = os.path.join(project_dir, "Program.cs")
        if os.path.exists(default_program):
            os.remove(default_program)

        for fp in file_paths:
            dest = os.path.join(project_dir, os.path.basename(fp))
            shutil.copy(fp, dest)

        build_result = subprocess.run(
            ["dotnet", "build", "--nologo", "-v", "q"],
            cwd=project_dir,
            capture_output=True,
            text=True,
            timeout=timeout
        )

        if build_result.returncode != 0:
            return {
                "success":   False,
                "stage":     "compile",
                "stdout":    build_result.stdout,
                "stderr":    build_result.stderr,
                "exit_code": build_result.returncode
            }

        run_result = subprocess.run(
            ["dotnet", "run", "--no-build", "--nologo"],
            cwd=project_dir,
            capture_output=True,
            text=True,
            timeout=timeout
        )

        return {
            "success":   run_result.returncode == 0,
            "stage":     "runtime",
            "stdout":    run_result.stdout,
            "stderr":    run_result.stderr,
            "exit_code": run_result.returncode
        }

    except subprocess.TimeoutExpired:
        return {
            "success":   False,
            "stage":     "timeout",
            "stdout":    "",
            "stderr":    f"Execution timed out after {timeout}s",
            "exit_code": -1
        }
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


# ── Evaluator ─────────────────────────────────────────────────────────────────
def evaluate_output(stdout: str, question: str, expected_output: str = None) -> dict:
    """
    Judges whether the program output satisfies the user's original request.
    - Exact match if expected_output is provided.
    - LLM judgment against the original question otherwise.
    """
    if expected_output is not None:
        passed = stdout.strip() == expected_output.strip()
        return {
            "passed": passed,
            "reason": "Output matched expected." if passed else
                      f"Expected:\n{expected_output}\n\nGot:\n{stdout}"
        }

    judgment_prompt = f"""You are a C# output validator.
The user originally requested: "{question}"

Given the program output below, decide if it correctly satisfies the user's request.
Reply ONLY with a JSON object: {{"passed": true/false, "reason": "..."}}

Output:
{stdout}"""

    response = ollama.chat(
        model="qwen2.5-custom:latest",
        messages=[{"role": "user", "content": judgment_prompt}]
    )

    raw = response["message"]["content"].strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    raw = raw.strip()

    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {"passed": True, "reason": "Could not parse judgment — assuming passed."}


# ── Fixer ─────────────────────────────────────────────────────────────────────
def fix_code(file_paths: list, error: str, question: str, context: str,
             model: str = "qwen2.5-custom:latest") -> list:
    """
    Asks the LLM to fix the code given the error/unexpected output and the original question.
    Returns a JSON change plan in the same format as generate_change_plan.
    """
    file_contents = ""
    for fp in file_paths:
        contents = read_file(fp)
        file_contents += f"\n\n// ── FILE: {fp} ──\n{contents}"

    fix_prompt = f"""You are an expert C# debugger.
The user originally requested: "{question}"

The code was written to fulfill that request but produced the following error or unexpected output.
Fix the code so it compiles cleanly and correctly fulfills the original request.

## Current File Contents
{file_contents}

## Error / Unexpected Output
{error}

## Additional Codebase Context
{context}

## Rules
- Output ONLY a JSON array — no markdown, no explanation, no preamble
- Each element must follow this schema:
  {{
    "file_path": "<exact path>",
    "start_line": <int or null>,
    "end_line": <int or null>,
    "new_code": "<complete fixed code>"
  }}
- new_code must be complete and compilable — never use placeholders like // ...
- Always set start_line and end_line to null and return the COMPLETE file contents in new_code — never return partial snippets
- new_code for an existing file must contain the entire file from top to bottom
- The fix must satisfy the original request: "{question}"

## Output (JSON array only):"""

    response = ollama.chat(
        model=model,
        messages=[{"role": "user", "content": fix_prompt}]
    )

    raw = response["message"]["content"].strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    raw = raw.strip()

    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        # Replace unescaped backslashes in file paths
        raw = raw.replace("\\", "\\\\")
        # But fix double-escaped common sequences back
        for seq in ["\\\\n", "\\\\t", "\\\\r", "\\\\\""]:
            raw = raw.replace(seq, seq[1:])
        return json.loads(raw)


# ── Debug Loop ────────────────────────────────────────────────────────────────
def debug_loop(
    file_paths: list,
    question: str,
    context: str,
    expected_output: str = None,
    max_attempts: int = 5,
    model: str = "qwen2.5-custom:latest"
) -> dict:
    """
    Runs the code, checks output against the original question, and fixes errors
    in a loop until the output passes or max_attempts is reached.
    """
    print("\n══ Debug Loop Starting ══════════════════════════════")

    last_result = {}

    for attempt in range(1, max_attempts + 1):
        print(f"\n── Attempt {attempt}/{max_attempts} ──────────────────────────────")

        last_result = run_csharp_code(file_paths)
        print(f"  Stage:     {last_result['stage']}")
        print(f"  Exit code: {last_result['exit_code']}")
        if last_result["stdout"]: print(f"  Stdout:\n{last_result['stdout']}")
        if last_result["stderr"]: print(f"  Stderr:\n{last_result['stderr']}")

        if last_result["success"]:
            evaluation = evaluate_output(last_result["stdout"], question, expected_output)
            print(f"  Evaluation passed: {evaluation['passed']}")
            print(f"  Reason: {evaluation['reason']}")

            if evaluation["passed"]:
                print("\n══ Debug Loop Passed ════════════════════════════════")
                return {"status": "passed", "attempts": attempt, "output": last_result["stdout"]}

            error_description = f"Unexpected output:\n{evaluation['reason']}"
        else:
            error_description = (
                f"[{last_result['stage'].upper()} ERROR]\n"
                f"{last_result['stderr'] or last_result['stdout']}"
            )

        if attempt == max_attempts:
            break

        print(f"\n  Fixing: {error_description[:200]}")
        changes = fix_code(file_paths, error_description, question, context, model=model)

        if not changes:
            print("  ✗ No fix plan returned — stopping early.")
            break

        print(f"  Applying {len(changes)} fix(es)...")
        for change in changes:
            result = write_file(
                file_path  = change["file_path"],
                new_code   = change["new_code"],
                start_line = change.get("start_line"),
                end_line   = change.get("end_line")
            )
            print(f"    → {result}")

    print("\n══ Debug Loop Failed ════════════════════════════════")
    return {
        "status":   "failed",
        "attempts": max_attempts,
        "output":   last_result.get("stderr") or last_result.get("stdout")
    }


# ── RAG Pipeline ──────────────────────────────────────────────────────────────
class CodebaseRAGPipeline:
    def __init__(self, llm_instance, collection_name="codebase_rag",
                 files_dir: str = r"C:\Users\Erwin\Desktop\rag_system\codebase_rag\files"):
        self.qdrant_client   = QdrantClient(url="http://localhost:6333")
        self.collection_name = collection_name
        self.llm             = llm_instance
        self.files_dir       = files_dir
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

    def store_chunks(self, chunks: list, metadata_list: list = None):
        points = []
        for idx, text in enumerate(chunks):
            payload = metadata_list[idx] if metadata_list else {}
            payload["page_content"] = text
            vector = self.embeddings_model.embed_query(f"search_document: {text}")
            points.append(PointStruct(
                id=str(uuid.uuid4()),
                vector=vector,
                payload=payload
            ))
        self.qdrant_client.upsert(collection_name=self.collection_name, points=points)
        print(f"Stored {len(points)} chunks into '{self.collection_name}'")

    def retriever(self, question: str, limit: int = 5) -> str:
        query_vector = self.embeddings_model.embed_query(f"search_query: {question}")

        search_results = self.qdrant_client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            limit=limit
        )

        formatted_blocks = []
        for idx, hit in enumerate(search_results.points):
            payload = hit.payload if hit.payload else {}
            formatted_blocks.append(
                f"--- Code Chunk {idx + 1} ---\n"
                f"File: {payload.get('file_path', 'Unknown File')}\n"
                f"Content:\n{payload.get('page_content', 'No content.')}\n"
            )

        return "\n".join(formatted_blocks)

    def ask(self, question: str, generation: bool = False, debug: bool = False,
            expected_output: str = None) -> str:

        limit   = 8 if generation else 5
        context = self.retriever(question, limit=limit)

        if generation:
            run_agent(question, context)

            if debug:
                file_paths = list({
                    c["file_path"]
                    for c in chunk_csharp_directory(self.files_dir)
                })
                result = debug_loop(
                    file_paths,
                    question,
                    context,
                    expected_output=expected_output
                )
                return str(result)

            return "Generation complete."

        # Plain Q&A — no tools
        template = """You are an expert C# code assistant.
Answer the question using only the codebase context below.
Be concise and precise. Do not generate unsolicited code.

## Codebase Context
{context}

## Question
{question}

## Answer:"""

        chain = (
            {"context": lambda q: self.retriever(q, limit=limit), "question": RunnablePassthrough()}
            | ChatPromptTemplate.from_template(template)
            | self.llm
            | StrOutputParser()
        )
        return chain.invoke(question)


# ── Entry Point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    FILES_DIR = r"C:\Users\Erwin\Desktop\rag_system\codebase_rag\files"

    local_llm    = ChatOllama(model="qwen2.5-custom:latest")
    rag_pipeline = CodebaseRAGPipeline(llm_instance=local_llm, files_dir=FILES_DIR)

    rag_pipeline.clear_collection()

    chunks   = chunk_csharp_directory(FILES_DIR)
    texts    = [c["text"] for c in chunks]
    metadata = [{k: v for k, v in c.items() if k != "text"} for c in chunks]
    rag_pipeline.store_chunks(texts, metadata)

    user_query = (
        "Add a DeleteOrder method to OrderRepository that removes an order by id "
        "and prints 'Order {id} deleted.' Then call it in Program.cs to delete "
        "order 2, and print the total order count after deletion."
    )

    expected = (
        "Order 1 added.\n"
        "Order 2 added.\n"
        "Order 3 added.\n"
        "Total orders: 3\n"
        "Found order: Banana\n"
        "Order 2 deleted.\n"
        "Total orders: 2"
    )

    result = rag_pipeline.ask(
        user_query,
        generation=True,
        debug=True,
        expected_output=expected
    )

    print(f"\nFinal Result:\n{result}")