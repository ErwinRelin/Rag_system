import os
import uuid
import json
import warnings
import subprocess
from collections import defaultdict

warnings.filterwarnings("ignore", category=FutureWarning)

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import ChatOllama, OllamaEmbeddings
from neo4j import GraphDatabase


# ==============================================================================
# SECTION 1: DATA EXTRACTION ENGINE
# ==============================================================================
def section_1_extract_chunks_from_roslyn_json(json_file_path: str) -> dict:
    """Reads Roslyn json files and slices physical C# file ranges into chunks."""
    try:
        with open(json_file_path, "r", encoding="utf-8-sig") as f:
            edges = json.load(f)
    except Exception as e:
        print(f"❌ Failed to load '{json_file_path}': {e}")
        return {}

    if not edges:
        return {}

    groups = defaultdict(list)
    for edge in edges:
        key = (edge["file_path"], edge["chunk_start_line"], edge["chunk_end_line"])
        groups[key].append(edge)

    chunks_package = {}

    for (file_path, start_line, end_line), group_edges in groups.items():
        code_text = ""
        using_header = ""
        try:
            with open(file_path, "r", encoding="utf-8-sig", errors="ignore") as f:
                all_lines = f.readlines()
            
            code_text = "".join(all_lines[start_line - 1 : end_line])
            using_directives = [
                line.strip() for line in all_lines
                if line.strip().startswith("using ") and line.strip().endswith(";")
            ]
            using_header = "\n".join(using_directives)
        except FileNotFoundError:
            code_text = f"// Source file not found: {file_path}"

        rel_lines = [
            f"// {e['source_class']}.{e['source_method']} --[{e['relation_type']}]--> {e['target_class']}.{e['target_method']}"
            for e in group_edges
        ]
        relationship_block = "\n".join(rel_lines)

        first = group_edges[0]

        text_content = (
            f"// FILE: {file_path}\n// CLASS: {first['source_class']}\n// METHOD: {first['source_method']}\n// LINES: {start_line}–{end_line}\n\n"
            f"{relationship_block}\n\n"
        )
        if using_header:
            text_content += f"// Active Namespace Dependencies:\n{using_header}\n\n"
        text_content += code_text.strip()

        graph_edges = [
            {"source": f"{e['source_class']}.{e['source_method']}", "target": f"{e['target_class']}.{e['target_method']}", "relation": e["relation_type"]}
            for e in group_edges
        ]

        metadata = {
            "file_path":  file_path,
            "class_name": first["source_class"],
            "method_name": first["source_method"],
            "start_line": start_line,
            "end_line":   end_line,
            "is_partial": first["source_is_partial"],
            "is_sealed":  first["source_is_sealed"],
            "graph_edges": json.dumps(graph_edges),
        }

        chunk_id = f"{file_path}::{start_line}::{end_line}"
        chunks_package[chunk_id] = {"text_content": text_content, "metadata": metadata}

    return chunks_package


# ==============================================================================
# SECTION 2: NEO4J GRAPH CONTROLLER
# ==============================================================================
def section_2_populate_neo4j_graph(json_file_path: str, password: str):
    """Loads compilation mappings directly into your local Neo4j Instance database."""
    driver = GraphDatabase.driver("bolt://localhost:7687", auth=("neo4j", password))
    
    with driver.session() as session:
        session.run("MATCH (n) DETACH DELETE n")
        print("Sub-system 2: 🧹 Neo4j schema space purged clean.")

    with open(json_file_path, "r", encoding="utf-8-sig") as f:
        edges = json.load(f)

    with driver.session() as session:
        for edge in edges:
            session.run(
                """
                MERGE (src:Method {id: $src_id})
                SET src.name = $src_method, src.class = $src_class, src.file = $file_path, 
                    src.is_partial = $src_partial, src.is_sealed = $src_sealed
                
                MERGE (tgt:Method {id: $tgt_id})
                SET tgt.name = $tgt_method, tgt.class = $tgt_class,
                    tgt.is_partial = $tgt_partial, tgt.is_sealed = $tgt_sealed
                
                WITH src, tgt
                MERGE (src)-[r:INVOKES {type: $rel_type}]->(tgt)
                """,
                src_id=f"{edge['file_path']}::{edge['source_class']}.{edge['source_method']}",
                src_method=edge['source_method'],
                src_class=edge['source_class'],
                file_path=edge['file_path'],
                src_partial=edge['source_is_partial'],
                src_sealed=edge['source_is_sealed'],
                
                tgt_id=f"{edge['file_path']}::{edge['target_class']}.{edge['target_method']}",
                tgt_method=edge['target_method'],
                tgt_class=edge['target_class'],
                tgt_partial=edge['target_is_partial'],
                tgt_sealed=edge['target_is_sealed'],
                rel_type=edge['relation_type']
            )
    driver.close()
    print(f"Sub-system 2: 📊 Successfully loaded links into Neo4j Graph.")


# ==============================================================================
# SECTION 3: .NET COMPILER SYNTAX VERIFIER
# ==============================================================================
def section_3_verify_dotnet_syntax() -> bool:
    """Invokes PowerShell commands to verify that code builds with zero compiler flags errors."""
    print("Sub-system 3: \nStarting .NET syntax verification...")
    ps_script = (
        "Get-ChildItem -Recurse -Filter *.csproj | ForEach-Object { "
        "Write-Host 'Checking Syntax:' $_.Name; "
        "dotnet build $_.FullName --no-incremental /p:BuildProjectReferences=false /v:quiet "
        "}"
    )

    result = subprocess.run(["powershell.exe", "-Command", ps_script], capture_output=True, text=True)

    if result.stdout: 
        print(result.stdout)
    if result.stderr: 
        print(f"System/Shell Errors:\n{result.stderr}")

    if "error CS" in result.stdout or result.returncode != 0:
        print("Sub-system 3: ❌ Syntax verification failed.")
        return False

    print("Sub-system 3: ✅ Syntax verification passed.")
    return True


# ==============================================================================
# SECTION 4: AGENT PIPELINE & MASTER EXECUTION
# ==============================================================================
class CodebaseRAGPipeline:
    def __init__(self, llm_instance, collection_name="codebase_rag", password="your_password"):
        self.qdrant_client = QdrantClient(path="./qdrant_storage_db")
        self.collection_name = collection_name
        self.llm = llm_instance
        self.embeddings_model = OllamaEmbeddings(model="nomic-embed-text")
        self.neo4j_password = password
        self._ensure_collection_exists()

    def _ensure_collection_exists(self):
        existing_names = [c.name for c in self.qdrant_client.get_collections().collections]
        if self.collection_name not in existing_names:
            self.qdrant_client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=768, distance=Distance.COSINE)
            )

    def ingest_codebase_to_qdrant(self, json_file_path: str):
        chunks_package = section_1_extract_chunks_from_roslyn_json(json_file_path)
        if not chunks_package:
            return

        self.qdrant_client.delete_collection(self.collection_name)
        self._ensure_collection_exists()

        chunk_ids = list(chunks_package.keys())
        texts_to_embed = [chunks_package[cid]["text_content"] for cid in chunk_ids]

        print(f"🧠 Generating Nomic embeddings for {len(texts_to_embed)} code chunks...")
        embeddings = self.embeddings_model.embed_documents(texts_to_embed)

        points = []
        for index, cid in enumerate(chunk_ids):
            raw_text = chunks_package[cid]["text_content"]
            metadata = dict(chunks_package[cid]["metadata"])
            if "graph_edges" in metadata and isinstance(metadata["graph_edges"], str):
                metadata["graph_edges"] = json.loads(metadata["graph_edges"])

            payload = {"document_text": raw_text, **metadata}
            points.append(PointStruct(id=str(uuid.uuid4()), vector=embeddings[index], payload=payload))

        self.qdrant_client.upsert(collection_name=self.collection_name, points=points)
        print(f"✅ Successfully stored elements inside Qdrant Store.")

    def retriever(self, question: str, limit: int = 3) -> str:
        """Crawl vector scores from Qdrant, then extract relational hops from Neo4j."""
        query_vector = self.embeddings_model.embed_query(f"search_query: {question}")
        search_results = self.qdrant_client.query_points(collection_name=self.collection_name, query=query_vector, limit=1)
        
        if not search_results.points:
            return "No text records found."

        hit = search_results.points[0]
        payload = hit.payload
        matched_node_id = f"{payload['file_path']}::{payload['class_name']}.{payload['method_name']}"
        context_blocks = [f"--- PRIMARY CODE SELECTION MATCH ---\n{payload['document_text']}"]
        
        neo4j_driver = GraphDatabase.driver("bolt://localhost:7687", auth=("neo4j", self.neo4j_password))
        cypher_query = "MATCH (src:Method {id: $node_id})-[r:INVOKES]->(tgt:Method) RETURN tgt.name AS target_method, r.type AS relation LIMIT $limit"
        
        with neo4j_driver.session() as session:
            result = session.run(cypher_query, node_id=matched_node_id, limit=limit)
            for record in result:
                # FIXED: Implemented type-strict Filter classes for Qdrant compatibility
                qdrant_filter = Filter(
                    must=[
                        FieldCondition(
                            key="method_name",
                            match=MatchValue(value=record["target_method"])
                        )
                    ]
                )
                
                # Scroll returns a tuple: (points_list, next_page_offset)
                points_list, _ = self.qdrant_client.scroll(
                    collection_name=self.collection_name,
                    scroll_filter=qdrant_filter,
                    limit=1
                )
                
                # FIXED: Safely verify list content extraction mapping
                if points_list and len(points_list) > 0:
                    connected_point = points_list[0]
                    context_blocks.append(f"--- GRAPH INFERRED DEPENDENCY ({record['relation']}) ---\n{connected_point.payload['document_text']}")
        
        neo4j_driver.close()
        return "\n\n".join(context_blocks)

    def ask(self, question: str, generation: bool = False) -> str:
        template = """You are an elite, type-safe automated C# refactoring engine. Your task is to generate structural code modifications based on the codebase context and requested objective.

### STRUCTURAL INTER-DEPENDENCY CONTEXT:
{context}

### REFACTORING TARGET OBJECTIVE:
{question}

### MANDATORY REFLECTION RULES:
1. COMPLETE SCOPES ONLY: The "new_code" field must contain the ENTIRE method block or class block requested. Never abbreviate code with comments like "// ... rest of code unchanged".
2. STRING ESCAPING: The final output must be pure, valid JSON. You MUST escape internal double quotes inside the C# code as \\" and newlines as \\n.
3. ABSOLUTE PATH CONFORMITY: Use absolute paths exactly as they appear in the file headers. Use forward slashes (/) only. Never use backslashes (\\).
4. OPERATIONAL ACTIONS: Use "REPLACE_METHOD" if editing an existing method, "APPEND_TO_CLASS" for adding fields/methods to an existing class, or "CREATE_FILE" for new files.

### ONE-SHOT REFERENCE EXAMPLE:
If the user objective was to "Add unique Id tracking to the Employee class and update Main", the expected output format would look exactly like this:

{{
  "status": "SUCCESS",
  "changes": [
    {{
      "file_path": "C:/Project/Employee.cs",
      "target_scope": "DummyApplication.Employee",
      "operation": "APPEND_TO_CLASS",
      "new_code": "public Guid UniqueId {{ get; set; }} = Guid.NewGuid();"
    }},
    {{
      "file_path": "C:/Project/Program.cs",
      "target_scope": "DummyApplication.Program.Main",
      "operation": "REPLACE_METHOD",
      "new_code": "static void Main(string[] args)\\n{{\\n    var emp = new Employee();\\n    Console.WriteLine(\$\\"Employee tracking started: {{emp.UniqueId}}\\");\\n}}"
    }}
  ]
}}

### OUTPUT FORMAT INSTRUCTION:
Return ONLY the raw JSON document. Do not wrap it in markdown block quotes (do not use ```json). Do not append any introduction or closing text. Begin with '{{' and end with '}}'.

JSON Response:"""


        limit = 8 if generation else 5
        chain = ({"context": lambda q: self.retriever(q, limit=limit), "question": RunnablePassthrough()} | ChatPromptTemplate.from_template(template) | self.llm | StrOutputParser())
        return chain.invoke(question)


if __name__ == "__main__":
    # --- CONFIGURATION BASE PARAMETERS ---
    NEO4J_PASSWORD = "###"
    JSON_SOURCE = "call_graph.json"

    # 🚨 RUN SECTION 2: MAP LINK GRAPHS TO NEO4J
    section_2_populate_neo4j_graph(JSON_SOURCE, NEO4J_PASSWORD)

    # 🚨 RUN SECTION 4: INITIALIZE AI EMBEDDINGS AGENT STACK
    local_llm = ChatOllama(model="qwen2.5-custom:latest", temperature=0.05)
    rag_pipeline = CodebaseRAGPipeline(llm_instance=local_llm, password=NEO4J_PASSWORD)
    rag_pipeline.ingest_codebase_to_qdrant(JSON_SOURCE)

    # FIRE CONVERSATIONAL EXECUTION
    user_query = "Add employee hire date support and display it in audit reports."
    ai_response = rag_pipeline.ask(user_query, generation=True)
    print(f"\n--- LLM Response ---\n{ai_response}\n")

    # 🚨 RUN SECTION 3: CHECK COMPILER INTEGRITY STATS via dotnet build
    section_3_verify_dotnet_syntax()

    # SAFE SYSTEM TEARDOWN
    rag_pipeline.qdrant_client.close()
    print("✨ Execution thread complete.")
