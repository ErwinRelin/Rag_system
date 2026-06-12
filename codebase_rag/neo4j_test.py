import os
import re
from dotenv import load_dotenv
from neo4j import GraphDatabase
from neo4j.exceptions import DriverError, Neo4jError
import tree_sitter_c_sharp as tscsharp
from tree_sitter import Language, Parser

# --- 1. SETUP ENVIRONMENT AND PARSER ---
load_dotenv()

CS_LANGUAGE = Language(tscsharp.language())
parser = Parser(CS_LANGUAGE)

# Generic system filters to remove basic infrastructure methods 
# that are not class declarations (e.g. built-in Console actions)
SYSTEM_FILTERS = ["Console.WriteLine", "_logs.Clear", "_logs.Add", "Local Context.Log"]


# --- 2. NEO4J GRAPH CONTROLLER ---
class Neo4jApp:
    def __init__(self):
        self.uri = os.getenv("NEO4J_URI", "neo4j://localhost:7687")
        username = os.getenv("NEO4J_USERNAME", "neo4j")
        password = os.getenv("NEO4J_PASSWORD", "12345678")
        
        if not password:
            raise ValueError("Critical Error: NEO4J_PASSWORD variable is empty or missing.")
            
        self.auth = (username, password)
        self.driver = None

    def connect(self):
        try:
            self.driver = GraphDatabase.driver(self.uri, auth=self.auth)
            self.driver.verify_connectivity()
            print("Successfully connected to Neo4j database.")
        except DriverError as e:
            print(f"Driver failed to connect to {self.uri}: {e}")
            raise

    def close(self):
        if self.driver:
            self.driver.close()
            print("Neo4j driver connection closed.")

    def clear_database(self):
        """Wipes the database clean before running the whitelisted ingestion."""
        query = "MATCH (n) DETACH DELETE n"
        try:
            print("\n🧹 Wiping Neo4j database clean for a fresh codebase scan...")
            self.driver.execute_query(query, database_="neo4j")
            print("✨ Database is empty and ready!")
        except Neo4jError as e:
            print(f"Failed to clear database: {e}")

    def add_dependency_edge(self, source_class: str, source_method: str, target_class: str, target_method: str):
        """Creates Class and Method structural nodes and links them with a CALLS edge."""
        if not all([source_class, source_method, target_class, target_method]):
            return

        query = """
        MERGE (sc:Class {name: $source_class})
        MERGE (tc:Class {name: $target_class})
        
        MERGE (sm:Method {name: $source_method, parent_class: $source_class})
        MERGE (tm:Method {name: $target_method, parent_class: $target_class})
        
        MERGE (sc)-[:CONTAINS]->(sm)
        MERGE (tc)-[:CONTAINS]->(tm)
        
        MERGE (sm)-[r:CALLS]->(tm)
        RETURN sc.name
        """
        try:
            self.driver.execute_query(
                query,
                source_class=str(source_class).strip(),
                source_method=str(source_method).strip(),
                target_class=str(target_class).strip(),
                target_method=str(target_method).strip(),
                database_="neo4j"
            )
            print(f"   ➔ Ingested Edge: {source_class}.{source_method}() CALLS {target_class}.{target_method}()")
        except Neo4jError as e:
            print(f"Failed to write relationship to Neo4j: {e}")


# --- 3. STREAMING AST TRAVERSAL WITH WHITELIST GATE ---
def parse_and_stream_to_db(node, db_app, current_class="Unknown", current_method="Unknown", global_map=None, valid_classes=None):
    if node is None:
        return
    if global_map is None: global_map = {}
    if valid_classes is None: valid_classes = set()

    if node.type == "class_declaration":
        for child in node.children:
            if child.type == "identifier":
                current_class = child.text.decode("utf8").strip()
                break

    elif node.type == "method_declaration":
        for child in node.children:
            if child.type == "identifier":
                current_method = child.text.decode("utf8").strip()
                break

    elif node.type == "invocation_expression":
        function_node = node.child_by_field_name("function")
        if function_node:
            call_text = function_node.text.decode("utf8").strip()
            
            if "." in call_text:
                called_component, called_method = call_text.rsplit(".", 1)
            else:
                called_component = "Local Context"
                called_method = call_text

            cleaned_component = re.sub(r'\s+', ' ', called_component.replace("?", "")).strip()
            cleaned_method = re.sub(r'\s+', ' ', called_method).strip()

            if f"{cleaned_component}.{cleaned_method}" in SYSTEM_FILTERS or cleaned_method == "nameof":
                return

            # Resolve variable types using Pass 1 map definitions
            if cleaned_component == "Local Context":
                resolved_component = current_class
            elif cleaned_component in global_map:
                resolved_component = global_map[cleaned_component]
            else:
                resolved_component = cleaned_component

            # WHITELIST FILTER GATE: Only allow ingestion if the resolved class
            # belongs to your verified project class collection.
            if resolved_component in valid_classes:
                db_app.add_dependency_edge(
                    source_class=current_class,
                    source_method=current_method,
                    target_class=resolved_component,
                    target_method=cleaned_method
                )

    for child in node.children:
        parse_and_stream_to_db(child, db_app, current_class, current_method, global_map, valid_classes)

# --- 4. C# TYPE MAP RESOLUTION & WHITELIST SCANNER ---
def scan_file_for_types(source_code, global_map, valid_classes):
    """
    Pass 1 Indexer: Maps field components textually to true C# types 
    and registers all declared class definitions into an application whitelist.
    """
    # 1. Map object instance variables to declared types
    field_pattern = r'(?:public|private|protected|internal)\s+(?:readonly\s+)?([\w<>\s]+?)\s+([_a-zA-Z]\w*)\s*(?:=[\s\S]*?)?;'
    for csharp_type, var_name in re.findall(field_pattern, source_code):
        global_map[var_name.strip()] = csharp_type.replace("?", "").strip()
    
    # Context variable helpers for local collection iteration logic 
    global_map["emp"] = "Employee"
    global_map["dept"] = "Department"
    global_map["top"] = "Employee"

    # 2. Extract explicit Class string names to automatically create the Whitelist registry
    class_pattern = r'\bclass\s+([_a-zA-Z]\w*)'
    for class_name in re.findall(class_pattern, source_code):
        valid_classes.add(class_name.strip())


# --- 5. SYSTEM COORDINATOR AND DIRECTORY RUNNER ---
def process_entire_codebase_whitelist(root_directory, db_app):
    global_type_catalog = {}
    verified_project_components = set()

    # Clear existing nodes out of database before running fresh scans
    db_app.clear_database()

    # === PASS 1: REGEX TEXT ANALYSIS TO DISCOVER LOCAL APP WORKSPACE SCHEMAS ===
    print("🔍 Pass 1: Dynamically discovering and whitelisting application classes...")
    for root, _, files in os.walk(root_directory):
        for file in files:
            if file.endswith(".cs") and "obj" not in root and "bin" not in root:
                file_path = os.path.join(root, file)
                with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                    code_content = f.read()
                    scan_file_for_types(code_content, global_type_catalog, verified_project_components)

    print(f"🎯 Whitelist Compiled! Found {len(verified_project_components)} target project components.")
    # Debug helper line to see what is allowed:
    # print(f"Allowed Classes: {verified_project_components}")

    # === PASS 2: AST STRUCTURE EXTRACTION GATEWAYS ===
    print("\n🚀 Pass 2: Parsing syntax trees and streaming whitelisted application calls...")
    for root, _, files in os.walk(root_directory):
        for file in files:
            if file.endswith(".cs") and "obj" not in root and "bin" not in root:
                file_path = os.path.join(root, file)
                
                with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                    code_content = f.read()
                    
                source_bytes = bytes(code_content, "utf8")
                tree = parser.parse(source_bytes)
                
                parse_and_stream_to_db(
                    tree.root_node, 
                    db_app=db_app, 
                    global_map=global_type_catalog, 
                    valid_classes=verified_project_components
                )


# --- 6. MAIN APPLICATION EXECUTION BLOCK ---
if __name__ == "__main__":
    # Point this path parameter straight to your codebase workspace directory container
    TARGET_CODEBASE = r"C:\Users\Erwin\Desktop\rag_system\codebase_rag"
    
    db = Neo4jApp()
    try:
        db.connect()
        
        # Start automated discovery and parsing sequence
        process_entire_codebase_whitelist(TARGET_CODEBASE, db_app=db)
        
        print("\n✨ Whitelisting operation complete! Open Neo4j browser to view your clean graph.")
        
    finally:
        db.close()
