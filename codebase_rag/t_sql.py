import json
import requests
from typing import List, Dict, Optional
import numpy as np
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import (
    VectorParams, Distance, PointStruct, Filter, FieldCondition,
    MatchValue, Range, PayloadSchemaType, SearchParams
)


# ═══════════════════════════════════════════════════════════════
# QDRANT INDEXER
# ═══════════════════════════════════════════════════════════════

class SqlChunkIndexer:
    def __init__(
        self,
        collection_name: str = "sql_chunks",
        model_name: str = "nomic-ai/nomic-embed-text-v1",
        qdrant_host: str = "localhost",
        qdrant_port: int = 6333,
        vector_size: int = 768
    ):
        self.collection_name = collection_name
        self.model = SentenceTransformer(model_name)
        self.client = QdrantClient(host=qdrant_host, port=qdrant_port)
        self.vector_size = vector_size

    def create_collection(self):
        self.client.recreate_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(size=self.vector_size, distance=Distance.COSINE)
        )
        for field in ["metadata.classification", "metadata.capability", 
                      "metadata.stage", "metadata.traits"]:
            self.client.create_payload_index(
                collection_name=self.collection_name,
                field_name=field,
                field_schema=PayloadSchemaType.KEYWORD
            )
        for field in ["metadata.businessScore", "metadata.reportingScore"]:
            self.client.create_payload_index(
                collection_name=self.collection_name,
                field_name=field,
                field_schema=PayloadSchemaType.FLOAT
            )
        print(f"Collection '{self.collection_name}' ready.")

    def load_chunks(self, path: str) -> List[Dict]:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def build_embedding_text(self, chunk: Dict) -> str:
        parts = [chunk.get("embeddingText", "")]
        for s in chunk.get("sections", []):
            summary = s.get("summary", "")
            if summary:
                parts.append(f"[{s.get('sectionType', '')}] {summary}")
        return "\n".join(parts).strip()

    def build_payload(self, chunk: Dict) -> Dict:
        return {
            "id": chunk["id"],
            "embeddingText": chunk.get("embeddingText", ""),
            "summary": chunk.get("metadata", {}).get("summary", ""),
            "metadata": {
                "classification": chunk.get("metadata", {}).get("classification"),
                "traits": chunk.get("metadata", {}).get("traits", []),
                "businessScore": chunk.get("metadata", {}).get("businessScore"),
                "reportingScore": chunk.get("metadata", {}).get("reportingScore"),
                "capability": chunk.get("metadata", {}).get("capability"),
                "stage": chunk.get("metadata", {}).get("stage"),
                "reads": chunk.get("metadata", {}).get("reads", []),
                "writes": chunk.get("metadata", {}).get("writes", []),
                "dependencies": chunk.get("metadata", {}).get("dependencies", []),
            },
            "sections": {
                s.get("sectionType", ""): {
                    "purpose": s.get("purpose", ""),
                    "summary": s.get("summary", ""),
                    "sqlText": s.get("sqlText", ""),
                    "startLine": s.get("startLine"),
                    "endLine": s.get("endLine"),
                }
                for s in chunk.get("sections", [])
            }
        }

    def index_chunks(self, chunks_path: str, batch_size: int = 100):
        chunks = self.load_chunks(chunks_path)
        print(f"Indexing {len(chunks)} chunks...")
        for start in range(0, len(chunks), batch_size):
            batch = chunks[start:start + batch_size]
            texts = [self.build_embedding_text(c) for c in batch]
            vectors = self.model.encode(texts, normalize_embeddings=True)
            points = [
                PointStruct(
                    id=abs(hash(c["id"])) % (2**63 - 1),
                    vector=v.tolist(),
                    payload=self.build_payload(c)
                )
                for c, v in zip(batch, vectors)
            ]
            self.client.upsert(collection_name=self.collection_name, points=points)
            print(f"  {min(start + batch_size, len(chunks))}/{len(chunks)}")
        print("Done.")

    def search(
        self,
        query: str,
        top_k: int = 5,
        classification: Optional[str] = None,
        capability: Optional[str] = None,
        min_business_score: Optional[float] = None,
        max_reporting_score: Optional[float] = None,
        traits: Optional[List[str]] = None,
    ) -> List[Dict]:
        must = []
        if classification:
            must.append(FieldCondition(key="metadata.classification", match=MatchValue(value=classification)))
        if capability:
            must.append(FieldCondition(key="metadata.capability", match=MatchValue(value=capability)))
        if min_business_score is not None:
            must.append(FieldCondition(key="metadata.businessScore", range=Range(gte=min_business_score)))
        if max_reporting_score is not None:
            must.append(FieldCondition(key="metadata.reportingScore", range=Range(lte=max_reporting_score)))
        if traits:
            for t in traits:
                must.append(FieldCondition(key="metadata.traits", match=MatchValue(value=t)))

        query_filter = Filter(must=must) if must else None
        query_vector = self.model.encode(query, normalize_embeddings=True).tolist()

        results = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            query_filter=query_filter,
            limit=top_k,
            search_params=SearchParams(hnsw_ef=128)
        )

        formatted = []
        for r in results.points:
            formatted.append({
                "id": r.payload["id"],
                "score": r.score,
                "summary": r.payload.get("summary", ""),
                "metadata": r.payload.get("metadata", {}),
                "sections": r.payload.get("sections", {}),
            })
        return formatted


# ═══════════════════════════════════════════════════════════════
# SECTION SCORER
# ═══════════════════════════════════════════════════════════════

class SectionScorer:
    def __init__(self, model: SentenceTransformer):
        self.model = model

    def score_sections(self, query: str, sections: Dict) -> Dict[str, float]:
        if not sections:
            return {}
        section_names = list(sections.keys())
        summaries = [
            sections[name].get("summary", "") or sections[name].get("purpose", "") or name
            for name in section_names
        ]
        query_vec = self.model.encode(query, normalize_embeddings=True)
        summary_vecs = self.model.encode(summaries, normalize_embeddings=True)
        return {
            name: round(float(np.dot(query_vec, summary_vecs[i])), 4)
            for i, name in enumerate(section_names)
        }


# ═══════════════════════════════════════════════════════════════
# SMART SECTION SELECTOR
# ═══════════════════════════════════════════════════════════════

class SmartSectionSelector:
    IMPORTANCE_WEIGHTS = {
        "HEADER": 1.3,
        "VALIDATION": 1.2,
        "BUSINESS_PROCESS": 1.1,
        "OUTPUT": 1.0,
        "DATA_RETRIEVAL": 1.0,
        "AUDIT": 0.8,
        "TRANSACTION": 0.7,
        "ERROR_HANDLING": 0.6,
        "INITIALIZATION": 0.5,
        "CLEANUP": 0.4,
    }

    def __init__(self, model: SentenceTransformer):
        self.scorer = SectionScorer(model)

    def select(self, chunk: Dict, query: str, max_sections: int = 5, min_score: float = 0.15) -> Dict:
        sections = chunk.get("sections", {})
        base_scores = self.scorer.score_sections(query, sections)

        weighted = {
            name: round(score * self.IMPORTANCE_WEIGHTS.get(name, 1.0), 4)
            for name, score in base_scores.items()
        }

        available = {
            name: weighted[name]
            for name in sections.keys()
            if weighted.get(name, 0) >= min_score
        }

        if "HEADER" in sections and "HEADER" not in available:
            available["HEADER"] = max(weighted.get("HEADER", 0), 0.1)

        ranked = sorted(available.items(), key=lambda x: x[1], reverse=True)
        selected = dict(ranked[:max_sections])

        return {
            "id": chunk["id"],
            "score": chunk.get("score"),
            "summary": chunk.get("summary", ""),
            "metadata": chunk.get("metadata", {}),
            "selected_sections": list(selected.keys()),
            "section_scores": selected,
            "sections": {name: sections[name] for name in selected.keys()},
        }


# ═══════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ═══════════════════════════════════════════════════════════════

class SqlRagPipeline:
    def __init__(
        self,
        collection_name: str = "sql_chunks",
        model_name: str = "nomic-ai/nomic-embed-text-v1",
        model: Optional[SentenceTransformer] = None,
        qdrant_host: str = "localhost",
        qdrant_port: int = 6333,
        max_context_tokens: int = 4000,
    ):
        self.collection_name = collection_name
        self.model_name = model_name
        self.qdrant_host = qdrant_host
        self.qdrant_port = qdrant_port
        self.max_context_tokens = max_context_tokens

        # Safely initialize the model inside the block
        if model is None:
            self.model = SentenceTransformer(self.model_name, trust_remote_code=True)
        else:
            self.model = model

        self.indexer = SqlChunkIndexer(
            collection_name=collection_name,
            model_name=model_name,
            qdrant_host=qdrant_host,
            qdrant_port=qdrant_port,
        )
        self.selector = SmartSectionSelector(model=self.indexer.model)
        self.max_context_tokens = max_context_tokens

    def index(self, chunks_path: str):
        self.indexer.create_collection()
        self.indexer.index_chunks(chunks_path)

    def query(self, user_query: str, top_k: int = 6,
              classification: Optional[str] = None,
              capability: Optional[str] = None) -> Dict:
        results = self.indexer.search(
            query=user_query, top_k=top_k,
            classification=classification, capability=capability,
        )
        return {
            "results": results,
            "top_procedures": [r["id"] for r in results[:top_k]],
            "top_scores": [round(r["score"], 3) for r in results[:top_k]],
        }


# ═══════════════════════════════════════════════════════════════
# ANSWER GENERATOR
# ═══════════════════════════════════════════════════════════════

class AnswerGenerator:
    INTENT_DESCRIPTIONS = """
You are an intent classifier for a SQL codebase assistant. Classify the user's query into exactly one of these intents:

1. GENERATE — User wants you to WRITE or CREATE new SQL code. They are asking for a new procedure, modification to existing code, or a new implementation. 
   KEY INDICATORS: "write", "create", "generate", "build", "make", "code to", "implement", "change X to Y", "add feature", "modify procedure"
   Examples: "Write me a procedure to transfer employees", "Change the payroll to batch process", "Add validation for email domains", "Create a new report"

2. EXPLAIN — User wants to UNDERSTAND existing code. They are NOT asking for new code to be written.
   KEY INDICATORS: "what does", "how does", "explain", "describe", "what is", "tell me about", "understand"
   Examples: "What does sp_AddNewEmployee do?", "Explain the leave workflow", "How does payroll work?"

3. ANALYZE — User wants to EVALUATE or COMPARE existing code. They are NOT asking for new code.
   KEY INDICATORS: "compare", "analyze", "review", "evaluate", "pros and cons", "best practice", "difference between"
   Examples: "Compare leave vs attendance validation", "Is this approach efficient?"

4. DEBUG — User is experiencing a PROBLEM with existing code.
   KEY INDICATORS: "error", "bug", "fix", "not working", "failing", "issue", "problem", "wrong"
   Examples: "Why does ProcessPayroll throw an error?", "The headcount is wrong after termination"

IMPORTANT: If the query asks you to WRITE, CREATE, CHANGE, MODIFY, or IMPLEMENT code, it is GENERATE — even if it also describes existing behavior.

Return ONLY one word: GENERATE, EXPLAIN, ANALYZE, or DEBUG.
"""

    PROMPTS = {
        "EXPLAIN": """You are a SQL expert explaining existing code.

RULES (follow strictly):
1. ONLY reference code shown in the RETRIEVED PROCEDURES section below.
2. NEVER invent, suggest, or generate new SQL code, procedures, or modifications.
3. If the retrieved code doesn't fully answer the question, say so clearly.
4. Explain in plain language what the existing code does, step by step.

RETRIEVED CONTEXT:
{context}

USER QUESTION: {query}

Explain the existing code that answers this question:""",

        "ANALYZE": """You are a SQL code analyst reviewing existing procedures.

RULES (follow strictly):
1. ONLY analyze code shown in the RETRIEVED PROCEDURES section below.
2. NEVER invent, suggest, or generate new SQL code or modifications.
3. If asked to compare, only compare procedures that are present in the context.
4. Identify patterns, potential issues, and business rules in the existing code.

RETRIEVED CONTEXT:
{context}

USER QUESTION: {query}

Analyze the existing code based on this question:""",

        "DEBUG": """You are a SQL debugging assistant helping fix issues in existing code.

RULES (follow strictly):
1. FIRST, examine the RETRIEVED PROCEDURES for the source of the issue.
2. Point to specific lines or sections that may be causing the problem.
3. If you suggest a fix, clearly mark it as SUGGESTED FIX and explain why.
4. If the retrieved code doesn't contain the source of the error, say so.

RETRIEVED CONTEXT:
{context}

USER QUESTION: {query}

Debug this issue using the existing code:""",

        "GENERATE": """You are a SQL developer writing stored procedures.

RULES:
1. Reference the RETRIEVED PROCEDURES for patterns, conventions, and existing tables.
2. Follow the same patterns (error handling, transaction management, audit logging) used by existing procedures.
3. ONLY use tables and columns that exist in the retrieved procedures.
4. Match the naming conventions and parameter patterns of the existing codebase.

RETRIEVED CONTEXT (existing procedures for reference):
{context}

USER QUESTION: {query}

Generate the requested SQL code following the patterns shown above:""",
    }

    def __init__(self, pipeline: SqlRagPipeline):
        self.pipeline = pipeline

    def detect_intent(self, query: str, intent_model: str = "qwen2.5:3b") -> str:
        # Wrap the instruction strictly and end with a direct structural cue
        prompt = f"{self.INTENT_DESCRIPTIONS}\n\nUser query: {query}\n\nClassification: "
        
        try:
            response = requests.post(
                "http://localhost:11434/api/generate",
                json={
                    "model": intent_model,
                    "prompt": prompt,
                    "stream": False,
                    # Increased num_predict to 30 so conversational filler won't truncate the actual answer
                    "options": {"temperature": 0.0, "num_predict": 30}, 
                },
                timeout=15
            )
            if response.status_code == 200:
                raw = response.json().get("response", "").strip()
                print(f"  [DEBUG] Intent LLM raw response: '{raw}'")
                intent = raw.upper()
                
                # Check for keywords anywhere in the model's text response
                for valid in ["GENERATE", "EXPLAIN", "ANALYZE", "DEBUG"]:
                    if valid in intent:
                        return valid
        except Exception as e:
            print(f"  [DEBUG] Intent LLM error: {e}")
        
        # If it genuinely fails, a safe backup strategy is to look for core action words in the string yourself
        query_upper = query.upper()
        if any(w in query_upper for w in ["WRITE", "CREATE", "GENERATE", "MODIFY", "CHANGE"]):
            return "GENERATE"
        if any(w in query_upper for w in ["EXPLAIN", "WHAT DOES", "HOW DOES", "UNDERSTAND"]):
            return "EXPLAIN"
            
        return "GENERATE"

    def ask(
        self,
        user_query: str,
        top_k: int = 5,
        classification: Optional[str] = None,
        capability: Optional[str] = None,
        llm_model: str = "qwen2.5-coder:14b",
        intent_model: str = "qwen2.5:3b",
        temperature: float = 0.1,
        max_tokens: int = 1500,
        stream: bool = False,
    ) -> Dict:
        # Step 1: Detect intent
        intent = self.detect_intent(user_query, intent_model)
        
        # Dynamic top_k adjustment: cast a wider net if comparing code structures
        actual_top_k = 6 if intent == "ANALYZE" else top_k

        # Step 2: Search
        results = self.pipeline.indexer.search(
            query=user_query, top_k=actual_top_k,
            classification=classification, capability=capability,
        )
        
        # Step 3: Build context (Passing the intent variable here!)
        context = self._build_context(results, user_query, intent=intent)
        
        # Step 4: Build prompt
        prompt = self.PROMPTS[intent].format(context=context, query=user_query)
        
        # Step 5: Call Ollama
        gen_temp = 0.3 if intent == "GENERATE" else temperature
        
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": llm_model,
                "prompt": prompt,
                "stream": stream,
                "options": {"temperature": gen_temp, "num_predict": max_tokens},
            },
            timeout=300
        )
        
        answer = ""
        if response.status_code == 200:
            if stream:
                answer = response.text
            else:
                answer = response.json().get("response", "")
        
        return {
            "intent": intent,
            "answer": answer,
            "prompt": prompt,
            "procedures_retrieved": [r["id"] for r in results[:actual_top_k]],
            "scores": [round(r["score"], 3) for r in results[:actual_top_k]],
            "model": llm_model,
        }

    def ask_streaming(self, user_query: str, top_k: int = 5,
                      classification: Optional[str] = None,
                      capability: Optional[str] = None,
                      llm_model: str = "qwen2.5-coder:14b",
                      intent_model: str = "qwen2.5:3b"):
        intent = self.detect_intent(user_query, intent_model)
        
        # Dynamic top_k adjustment for streaming mode as well
        actual_top_k = 6 if intent == "ANALYZE" else top_k

        results = self.pipeline.indexer.search(
            query=user_query, top_k=actual_top_k,
            classification=classification, capability=capability,
        )
        
        # Passing the intent variable here too!
        context = self._build_context(results, user_query, intent=intent)
        prompt = self.PROMPTS[intent].format(context=context, query=user_query)
        
        gen_temp = 0.3 if intent == "GENERATE" else 0.1
        
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": llm_model,
                "prompt": prompt,
                "stream": True,
                "options": {"temperature": gen_temp},
            },
            stream=True,
            timeout=300
        )
        
        for line in response.iter_lines():
            if line:
                data = json.loads(line)
                yield data.get("response", "")

    def _build_context(self, results: List[Dict], query: str, intent: str = "EXPLAIN") -> str:
        parts = []
        total_chars = 0
        
        # 1. If analyzing/comparing, read deeper into top_k (up to 6) instead of choking at 3
        max_chunks = 6 if intent == "ANALYZE" else 3
        # 2. If comparing, restrict sections per file to save space so they all fit
        max_sections_per_chunk = 2 if intent == "ANALYZE" else 4
        
        for chunk in results[:max_chunks]:
            filtered = self.pipeline.selector.select(chunk, query, max_sections=max_sections_per_chunk)
            if not filtered["selected_sections"]:
                continue
            
            chunk_text = self._format_chunk(filtered)
            
            # Expanded character window slightly to accommodate multiple files
            if total_chars + len(chunk_text) > 12000:
                break
            
            parts.append(chunk_text)
            total_chars += len(chunk_text)
        
        return "\n\n".join(parts) if parts else "[No relevant procedures found.]"

    def _format_chunk(self, filtered: Dict) -> str:
        lines = []
        metadata = filtered.get("metadata", {})
        
        lines.append(f"PROCEDURE: {filtered['id']}")
        lines.append(f"TYPE: {metadata.get('classification', '')}")
        lines.append(f"CAPABILITY: {metadata.get('capability', '')}")
        lines.append(f"STAGE: {metadata.get('stage', '')}")
        
        if filtered.get("summary"):
            lines.append(f"SUMMARY: {filtered['summary']}")
        
        lines.append(f"SECTIONS INCLUDED: {', '.join(filtered['selected_sections'])}")
        lines.append("")
        
        for stype in filtered["selected_sections"]:
            section = filtered["sections"].get(stype, {})
            if not section:
                continue
            sql = section.get("sqlText", "")
            if len(sql) > 2500:
                sql = sql[:2500] + "\n-- [truncated]"
            lines.append(f"-- [{stype}] {section.get('purpose', '')}")
            lines.append(sql)
            lines.append("")
        
        return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════
# USAGE
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    pipeline = SqlRagPipeline(
        collection_name="hrms_procedures",
        max_context_tokens=3000,
    )

    # ── Index (run once) ──────────────────────────────────────
    pipeline.index(r"C:\Users\Erwin\Desktop\rag_system\codebase_rag\t-sql\shared_data\semantic_chunks.json")

    generator = AnswerGenerator(pipeline)

    # ── Interactive mode ──────────────────────────────────────
    print("\n" + "="*60)
    print("  SQL RAG ASSISTANT")
    print("  Type 'quit' to exit, 'stream' for streaming mode")
    print("="*60)

    while True:
        query = input("\nYou: ").strip()
        if not query:
            continue
        if query.lower() == 'quit':
            break

        if query.lower().startswith("stream "):
            query = query[7:]
            print(f"\n[Streaming] ", end="", flush=True)
            for token in generator.ask_streaming(query, top_k=6):
                print(token, end="", flush=True)
            print()
        else:
            result = generator.ask(query, top_k=6)
            print(f"\n[Intent: {result['intent']}]")
            print(f"[Procedures: {result['procedures_retrieved']}]")
            print(f"[Scores: {result['scores']}]")
            print(f"\n{result['answer']}")