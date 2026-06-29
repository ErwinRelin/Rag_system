import json
from typing import List, Dict, Optional
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import (
    VectorParams, Distance, PointStruct, Filter, FieldCondition,
    MatchValue, Range, PayloadSchemaType, SearchParams
)


class SqlChunkIndexer:
    """
    Embeds semantic chunks and stores them in Qdrant.
    Retrieval returns relevant sections — not the full SQL.
    """
    
    def __init__(
        self,
        collection_name: str = "sql_chunks",
        model_name: str = "all-MiniLM-L6-v2",
        qdrant_host: str = "localhost",
        qdrant_port: int = 6333,
        vector_size: int = 384
    ):
        self.collection_name = collection_name
        self.model = SentenceTransformer(model_name)
        self.client = QdrantClient(host=qdrant_host, port=qdrant_port)
        self.vector_size = vector_size

    def create_collection(self):
        """Create collection with indexes for filterable metadata."""
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
        """Only embed semantic description — never SQL."""
        parts = [chunk.get("embeddingText", "")]
        for s in chunk.get("sections", []):
            if s.get("summary"):
                parts.append(f"[{s['type']}] {s['summary']}")
        return "\n".join(parts).strip()

    def build_payload(self, chunk: Dict) -> Dict:
        """Payload: metadata + section summaries + SQL indexed by section type."""
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
                s.get("sectionType", s.get("type", "")): {
                    "purpose": s.get("purpose", ""),
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

    # ═══════════════════════════════════════════════════════════
    # RETRIEVAL
    # ═══════════════════════════════════════════════════════════

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
        """Search and return chunks with metadata + section map."""
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
# SECTION SELECTOR
# ═══════════════════════════════════════════════════════════════

class SectionSelector:
    """
    Selects only the relevant sections from a retrieved chunk.
    Passes them to a validator for final filtering.
    """
    
    # Which sections are needed for different query intents
    INTENT_SECTIONS = {
        "what_is":       ["HEADER", "VALIDATION", "OUTPUT"],
        "how_to":        ["HEADER", "VALIDATION", "DATA_MODIFICATION", "AUDIT", "TRANSACTION", "OUTPUT"],
        "validate":      ["VALIDATION", "BUSINESS_RULES"],
        "modify":        ["DATA_MODIFICATION", "TRANSACTION", "AUDIT"],
        "error":         ["ERROR_HANDLING", "VALIDATION"],
        "report":        ["HEADER", "DATA_RETRIEVAL", "OUTPUT"],
        "audit":         ["AUDIT", "DATA_MODIFICATION"],
        "full":          None,  # None = all sections
    }
    
    def select(self, chunk: Dict, intent: str = "how_to", max_sections: int = 4) -> Dict:
        """
        Select relevant sections based on query intent.
        
        Returns a filtered chunk with only the needed sections.
        """
        sections = chunk.get("sections", {})
        allowed = self.INTENT_SECTIONS.get(intent)
        
        if allowed is None:  # "full" intent
            selected = sections
        else:
            selected = {
                k: v for k, v in sections.items()
                if k in allowed
            }
        
        # Limit to most important sections
        priority_order = ["HEADER", "VALIDATION", "BUSINESS_RULES", 
                         "DATA_MODIFICATION", "TRANSACTION", "AUDIT", 
                         "OUTPUT", "DATA_RETRIEVAL", "ERROR_HANDLING",
                         "INITIALIZATION", "CLEANUP"]
        
        ordered = sorted(
            selected.items(),
            key=lambda x: priority_order.index(x[0]) if x[0] in priority_order else 999
        )
        
        selected = dict(ordered[:max_sections])
        
        return {
            "id": chunk["id"],
            "score": chunk.get("score"),
            "summary": chunk.get("summary", ""),
            "metadata": chunk.get("metadata", {}),
            "selected_sections": list(selected.keys()),
            "sections": selected,
        }


# ═══════════════════════════════════════════════════════════════
# VALIDATOR
# ═══════════════════════════════════════════════════════════════

class SectionValidator:
    """
    Validates that selected sections are complete and relevant.
    Strips noise before passing to the LLM.
    """
    
    def __init__(self, max_tokens_per_section: int = 2000):
        self.max_tokens = max_tokens_per_section
    
    def validate(self, filtered_chunk: Dict) -> Dict:
        """
        Validate and clean selected sections.
        
        - Removes empty sections
        - Truncates overly long sections
        - Ensures HEADER is present for business operations
        - Adds context about missing sections
        """
        sections = filtered_chunk.get("sections", {})
        metadata = filtered_chunk.get("metadata", {})
        
        # Remove empty sections
        sections = {
            k: v for k, v in sections.items()
            if v.get("sqlText", "").strip()
        }
        
        # Truncate long sections
        for key in sections:
            sql = sections[key]["sqlText"]
            if len(sql) > self.max_tokens * 4:  # rough char estimate
                sections[key]["sqlText"] = sql[:self.max_tokens * 4] + "\n-- [TRUNCATED]"
                sections[key]["truncated"] = True
        
        # Ensure HEADER exists for business operations
        if metadata.get("classification") == "BUSINESS_OPERATION" and "HEADER" not in sections:
            sections["_missing_header"] = {
                "purpose": "Header was not selected",
                "sqlText": f"-- Procedure: {filtered_chunk['id']}",
            }
        
        # Flag what was excluded
        all_section_types = {"HEADER", "VALIDATION", "BUSINESS_RULES", "DATA_MODIFICATION",
                            "TRANSACTION", "AUDIT", "OUTPUT", "DATA_RETRIEVAL",
                            "ERROR_HANDLING", "INITIALIZATION", "CLEANUP"}
        missing = all_section_types - set(sections.keys()) - {"_missing_header"}
        
        return {
            **filtered_chunk,
            "sections": sections,
            "warnings": {
                "truncated": [k for k, v in sections.items() if v.get("truncated")],
                "excluded_sections": list(missing),
            }
        }
    
    def to_llm_context(self, validated: Dict) -> str:
        """
        Convert validated sections into a context block for the LLM.
        """
        parts = []
        
        parts.append(f"PROCEDURE: {validated['id']}")
        parts.append(f"SUMMARY: {validated.get('summary', '')}")
        parts.append(f"SELECTED SECTIONS: {', '.join(validated.get('selected_sections', []))}")
        
        warnings = validated.get("warnings", {})
        if warnings.get("excluded_sections"):
            parts.append(f"NOTE: Sections not included: {', '.join(warnings['excluded_sections'])}")
        
        parts.append("\n--- SQL SECTIONS ---\n")
        
        for section_type, section_data in validated.get("sections", {}).items():
            parts.append(f"\n-- [{section_type}] {section_data.get('purpose', '')}")
            parts.append(section_data["sqlText"])
            if section_data.get("truncated"):
                parts.append("-- [This section was truncated due to length]")
        
        return "\n".join(parts)


# ═══════════════════════════════════════════════════════════════
# USAGE
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    indexer = SqlChunkIndexer(collection_name="hrms_procedures")
    indexer.create_collection()
    indexer.index_chunks("semantic_chunks.json")
    
    selector = SectionSelector()
    validator = SectionValidator(max_tokens_per_section=1500)
    
    # ── Example 1: "How do I add a new employee?" ─────────────
    results = indexer.search(
        query="How do I hire a new employee with validation?",
        top_k=3,
        classification="BUSINESS_OPERATION"
    )
    
    for r in results:
        # Select only the sections needed to understand "how to" do this
        filtered = selector.select(r, intent="how_to", max_sections=5)
        
        # Validate and clean
        validated = validator.validate(filtered)
        
        # Get LLM-ready context
        context = validator.to_llm_context(validated)
        print(f"\n{'='*60}")
        print(context[:1500])
    
    # ── Example 2: "What validations exist?" ──────────────────
    results = indexer.search(
        query="What business rules validate employee data?",
        top_k=3,
        traits=["VALIDATED"]
    )
    
    for r in results:
        filtered = selector.select(r, intent="validate", max_sections=3)
        validated = validator.validate(filtered)
        # Only validation sections are returned
        print(f"\n{r['id']}: {list(validated['sections'].keys())}")
    
    # ── Example 3: "Show me all reports" ──────────────────────
    results = indexer.search(
        query="dashboards and attendance reports",
        top_k=5,
        classification="REPORT"
    )
    
    for r in results:
        filtered = selector.select(r, intent="report", max_sections=3)
        validated = validator.validate(filtered)
        context = validator.to_llm_context(validated)
        print(f"\n{r['id']} (score: {r['score']:.3f})")
        print(f"  Sections: {validated['selected_sections']}")