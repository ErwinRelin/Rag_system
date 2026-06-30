import json
import numpy as np
from typing import List, Dict, Optional,Tuple
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
        
        # Now each section has its own unique summary
        for s in chunk.get("sections", []):
            section_summary = s.get("summary", "")
            if section_summary:
                parts.append(f"[{s.get('sectionType', s.get('type', ''))}] {section_summary}")
        
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
        "how_to":        ["HEADER", "VALIDATION", "DATA_MODIFICATION", "AUDIT", "OUTPUT"],
        "validate":      ["VALIDATION", "BUSINESS_RULES"],
        "modify":        ["DATA_MODIFICATION", "TRANSACTION", "AUDIT"],
        "error":         ["ERROR_HANDLING", "VALIDATION"],
        "report":        ["HEADER", "DATA_RETRIEVAL", "OUTPUT"],
        "audit":         ["AUDIT", "DATA_MODIFICATION"],
        "full":          None,  # None = all sections
    }
    
    def select(self, chunk: Dict, intent: str = "how_to", max_sections: int = 4) -> Dict:
        sections = chunk.get("sections", {})
        allowed = self.INTENT_SECTIONS.get(intent)
        
        if allowed is None:
            selected = dict(sections)  # copy all
        else:
            selected = {
                k: v for k, v in sections.items()
                if k in allowed
            }
        
        # Priority order for sorting
        priority_order = ["HEADER", "VALIDATION", "BUSINESS_RULES", 
                        "DATA_MODIFICATION", "TRANSACTION", "AUDIT", 
                        "OUTPUT", "DATA_RETRIEVAL", "ERROR_HANDLING",
                        "INITIALIZATION", "CLEANUP"]
        
        ordered = sorted(
            selected.items(),
            key=lambda x: priority_order.index(x[0]) if x[0] in priority_order else 999
        )
        
        # Take only top max_sections AND create a NEW filtered dict
        filtered_sections = dict(ordered[:max_sections])
        
        return {
            "id": chunk["id"],
            "score": chunk.get("score"),
            "summary": chunk.get("summary", ""),
            "metadata": chunk.get("metadata", {}),
            "selected_sections": list(filtered_sections.keys()),
            "sections": filtered_sections,  # ← ONLY the selected sections, not all
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
    
    def validate(self, filtered_chunk: Dict, intent: str = "how_to") -> Dict:
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
            if len(sql) > self.max_tokens * 4:
                sections[key]["sqlText"] = sql[:self.max_tokens * 4] + "\n-- [TRUNCATED]"
                sections[key]["truncated"] = True

        # Only warn about missing HEADER for intents that need it
        needs_header = intent in ("how_to", "what_is", "modify", "full")
        has_header = any(k.upper() == "HEADER" for k in sections.keys())
        
        if metadata.get("classification") == "BUSINESS_OPERATION" and needs_header and not has_header:
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

class SectionScorer:
    def __init__(self, model: SentenceTransformer):
        self.model = model
    
    def score_sections(self, query: str, sections: Dict) -> Dict[str, float]:
        """Score each section by comparing query to the section's unique summary."""
        if not sections:
            return {}
        
        section_names = list(sections.keys())
        section_summaries = []
        for name in section_names:
            section = sections[name]
            # Prefer the business summary, fall back to purpose, then section name
            summary = section.get("summary", "") or section.get("purpose", "") or name
            section_summaries.append(summary)
        
        query_vector = self.model.encode(query, normalize_embeddings=True)
        summary_vectors = self.model.encode(section_summaries, normalize_embeddings=True)
        
        scores = {}
        for i, name in enumerate(section_names):
            similarity = float(np.dot(query_vector, summary_vectors[i]))
            scores[name] = round(similarity, 4)

        
        return scores
    
    def select_sections(
        self,
        chunk: Dict,
        query: str,
        max_sections: int = 5,
        min_score: float = 0.2
    ) -> Dict:
        """
        Select the most relevant sections from a chunk based on query similarity.
        
        Args:
            chunk: The retrieved chunk with all sections
            query: The user's original query
            max_sections: Maximum number of sections to return
            min_score: Minimum relevance score to include a section
        """
        sections = chunk.get("sections", {})
        
        # Score each available section type
        scores = self.score_sections(query)
        
        # Filter to only sections that exist in this chunk AND meet min score
        available_scores = {
            name: scores.get(name, 0)
            for name in sections.keys()
            if scores.get(name, 0) >= min_score
        }
        
        # Always include HEADER if it exists and score is borderline
        if "HEADER" in sections and "HEADER" not in available_scores:
            header_score = scores.get("HEADER", 0)
            if header_score >= 0.1:  # lower threshold for HEADER
                available_scores["HEADER"] = header_score
        
        # Sort by score descending
        ranked = sorted(available_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Take top N
        selected = dict(ranked[:max_sections])
        
        # Build filtered sections
        filtered_sections = {
            name: sections[name]
            for name in selected.keys()
        }
        
        return {
            "id": chunk["id"],
            "score": chunk.get("score"),
            "summary": chunk.get("summary", ""),
            "metadata": chunk.get("metadata", {}),
            "selected_sections": list(filtered_sections.keys()),
            "section_scores": selected,
            "sections": filtered_sections,
        }


class SmartSectionSelector:

    IMPORTANCE_WEIGHTS = {
        "HEADER": 0.75,
        "VALIDATION": 1.2,
        "BUSINESS_RULES": 1.1,
        "DATA_MODIFICATION": 1.0,
        "OUTPUT": 1.0,
        "DATA_RETRIEVAL": 1.0,
        "AUDIT": 0.65,
        "TRANSACTION": 0.7,
        "ERROR_HANDLING": 0.6,
        "INITIALIZATION": 0.5,
        "CLEANUP": 0.4,
    }

    def __init__(self, model: SentenceTransformer):
        self.model = model
        self.scorer = SectionScorer(model)
    
    def select(self, chunk: Dict, query: str, max_sections: int = 5, min_score: float = 0.15) -> Dict:
        sections = chunk.get("sections", {})
        
        # Score the actual sections in this chunk using their unique summaries
        base_scores = self.scorer.score_sections(query, sections)  # ← pass sections
        
        # Apply importance weighting
        weighted_scores = {}
        for name, score in base_scores.items():
            weight = self.IMPORTANCE_WEIGHTS.get(name, 1.0)
            weighted_scores[name] = round(score * weight, 4)
        
        # Filter to available sections above min score
        available = {
            name: weighted_scores.get(name, 0)
            for name in sections.keys()
            if weighted_scores.get(name, 0) >= min_score
        }
        
        # Always include HEADER if it exists
        if "HEADER" in sections and "HEADER" not in available:
            available["HEADER"] = max(weighted_scores.get("HEADER", 0), 0.1)
        
        # Rank and select
        ranked = sorted(available.items(), key=lambda x: x[1], reverse=True)
        selected = dict(ranked[:max_sections])
        
        filtered_sections = {name: sections[name] for name in selected.keys()}
        
        return {
            "id": chunk["id"],
            "score": chunk.get("score"),
            "summary": chunk.get("summary", ""),
            "metadata": chunk.get("metadata", {}),
            "selected_sections": list(filtered_sections.keys()),
            "section_scores": selected,
            "all_scores": weighted_scores,
            "sections": filtered_sections,
        }


# ═══════════════════════════════════════════════════════════════
# UPDATED USAGE
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    indexer = SqlChunkIndexer(collection_name="hrms_procedures")
    indexer.create_collection()
    indexer.index_chunks(r"C:\Users\Erwin\Desktop\rag_system\codebase_rag\t-sql\shared_data\semantic_chunks.json")
    
    # Initialize smart selector with the same model
    selector = SmartSectionSelector(model=indexer.model)
    validator = SectionValidator(max_tokens_per_section=1500)
    
    # ── Example 1: "How do I add a new employee?" ─────────────
    query1 = "How do I hire a new employee with validation?"
    results = indexer.search(query=query1, top_k=3, classification="BUSINESS_OPERATION")
    
    for r in results:
        print(f"\n{'='*60}")
        print(f"Procedure: {r['id']} (score: {r['score']:.3f})")
        
        # Smart selection based on query
        filtered = selector.select(r, query=query1, max_sections=5)
        
        print(f"Section scores: {filtered['section_scores']}")
        print(f"Selected: {filtered['selected_sections']}")
        
        validated = validator.validate(filtered, intent="auto")
        context = validator.to_llm_context(validated)
        print(context[:800])
    
    # ── Example 2: "What validations exist?" ──────────────────
    query2 = "What business rules validate employee data before insertion?"
    results = indexer.search(query=query2, top_k=3, traits=["VALIDATED"])
    
    for r in results:
        filtered = selector.select(r, query=query2, max_sections=3)
        print(f"\n{r['id']}: scores={filtered['section_scores']}")
        print(f"  Selected: {filtered['selected_sections']}")
    
    # ── Example 3: "How is payroll calculated?" ───────────────
    query3 = "How is payroll calculated and processed in batches?"
    results = indexer.search(query=query3, top_k=3)
    
    for r in results:
        filtered = selector.select(r, query=query3, max_sections=5)
        print(f"\n{r['id']} (score: {r['score']:.3f})")
        print(f"  All scores: {filtered['all_scores']}")
        print(f"  Selected: {filtered['selected_sections']}")