from tree_sitter import Language, Parser
import os
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
import sys 

sys.setrecursionlimit(5000)

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
                
                method_chunks.append({
                    "type":        chunk_type,
                    "class_name":  class_name,
                    "method_name": member_name,
                    "file_path":   file_path,
                    "usings":      usings, # Saved in database metadata array only
                    "text":        format_chunk_text(node_text(child, source), file_path, class_name=class_name, method_name=member_name),
                    "start_line":  child.start_point,
                    "end_line":    child.end_point
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
            "file_path": file_path,
            "text":      f"// FILE: {file_path}\n// USINGS\n\n{raw_using_block}",
            "usings":    usings
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


# ── Quick Test Run execution ──────────────────────────────────────────────────
if __name__ == "__main__":
    test_file = os.path.join(PROJECT_ROOT, "classrag", "dummy_framework.cs")
    
    if os.path.exists(test_file):
        results = chunk_csharp_file(test_file)
        print(f"\n--- Total Generated Chunks: {len(results)} ---")
        for idx, c in enumerate(results):
            print(f"\n[Chunk {idx}] Type: {c['type']} | Lines: {c.get('start_line', 0)}-{c.get('end_line', 0)}")
            print(c['text'])
