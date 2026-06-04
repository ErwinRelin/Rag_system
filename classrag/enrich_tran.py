# enrichment.py
import os
import argparse
from pathlib import Path
from operator import itemgetter
from langchain_core.runnables import RunnableLambda
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import ChatOllama
from langchain_chroma import Chroma
from langchain_experimental.open_clip import OpenCLIPEmbeddings
from deep_translator import GoogleTranslator
import chromadb
import uuid
import logging
from langchain_core.tracers.context import collect_runs
import traceback
from datetime import datetime
import json
import config

# import your tanglish translator
from tanglish import english_to_tanglish


os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = config.MY_API_KEY

# ── models ─────────────────────────────────────────────────────────────────────

llm = ChatOllama(model="qwen2.5vl:7custom", temperature=0)
embedding_fn = OpenCLIPEmbeddings()

logging.basicConfig(level=logging.INFO, filename=r"C:\Users\Erwin\Desktop\rag_system\classrag\ultimate_llm_system.json", format="%(asctime)s - %(message)s")

ENRICHMENT_PROMPT = """You are a technical writer. Using ONLY the BRD context below, rewrite the issue description into a single clear, specific sentence that identifies the exact component, the broken behavior, and the expected behavior.

Context: {context}
Issue Description: {issue_description}
Expected Outcome: {expected_result}

Output one sentence only. No headings, no bullet points, no explanation.
"""



# ── context retrieval ──────────────────────────────────────────────────────────
class enrichment_transalation():
    def make_context_fn(self, retriever):

        def context_from_inputs(inputs) -> str:

            query = (
                inputs.get("issue_description","") + " " +
                inputs.get("expected_result","")
                if isinstance(inputs, dict)
                else str(inputs)
            )

            docs = retriever.invoke(query)

            print("\n===== RETRIEVED =====")

            for i,d in enumerate(docs):
                print(f"\n--- CHUNK {i+1} ---")
                print(d.page_content[:700])

            print("\n====================")

        return context_from_inputs


    def build_collection_from_file(self, file_path: str) -> Chroma:
        """Ingest a single document into a temporary Chroma collection."""
        from unstructured.partition.auto import partition
        from unstructured.chunking.title import chunk_by_title
        import base64, tempfile, os, io
        from PIL import Image

        collection = Chroma(
            collection_name=f"enrich_{uuid.uuid4()}",
            embedding_function=embedding_fn,
        )

        elements = partition(
            filename=file_path,
            strategy="auto",
            extract_images_in_pdf=False,
            extract_image_block_types=["Image", "Table"],
            extract_image_block_to_payload=True,
        )
        chunks = chunk_by_title(elements, max_characters=1500, combine_text_under_n_chars=700, overlap=300)

        text_chunks, image_uris = [], []
        seen = set()

        for chunk in chunks:
            if chunk.text.strip():
                text_chunks.append(chunk.text)
            for orig in getattr(chunk.metadata, "orig_elements", None) or []:
                if orig.category == "Image":
                    b64 = getattr(orig.metadata, "image_base64", None)
                    if b64 and b64 not in seen:
                        seen.add(b64)
                        img_bytes = base64.b64decode(b64)
                        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
                            Image.open(io.BytesIO(img_bytes)).save(f.name)
                            image_uris.append(f.name)

        if text_chunks:
            collection.add_texts(text_chunks, ids=[str(uuid.uuid4()) for _ in text_chunks])

        return collection


    # ── enrichment ─────────────────────────────────────────────────────────────────

    def build_enrichment_chain(self, context_fn):
        # Using itemgetter or explicit lambdas ensures LangChain treats this dict as a RunnableMap
        
        
        return (
            
            {
                "context": RunnableLambda(context_fn),
                "issue_description": RunnableLambda(lambda x: x.get("issue_description")),
                "expected_result": RunnableLambda(lambda x: x.get("expected_result")),
            }

            | ChatPromptTemplate.from_template(ENRICHMENT_PROMPT)
            | llm
            | StrOutputParser()
        )


    def enrich_issue(self, context_fn, issue_description: str, expected_result: str = "") -> str:
        chain = self.build_enrichment_chain(context_fn)

        

        response = ""
        error_occurred = None
            
        with collect_runs() as cb:
            try:
                response = chain.invoke({
                    "issue_description": issue_description,
                    "expected_result": expected_result,
                })
            except Exception as e:
                # Capture system crashes or API timeout details
                error_occurred = {
                    "error_type": type(e).__name__,
                    "error_message": str(e),
                    "traceback": traceback.format_exc()
                }
                response = "An internal execution error occurred."
            
            # Extract every piece of underlying metadata from LangSmith
            if cb.traced_runs:
                run = cb.traced_runs[0]
                
                # 1. Calculate ultra-precise latencies
                start_time = run.start_time
                end_time = run.end_time if run.end_time else datetime.now()
                total_duration_ms = (end_time - start_time).total_seconds() * 1000

                usage = getattr(run, "usage", {}) or {}
                prompt_tokens = usage.get("prompt_tokens", 0) or 0
                completion_tokens = usage.get("completion_tokens", 0) or 0
                
                # 2. Build the ultimate structured data payload
                ultimate_log = {
                    "trace_identity": {
                        "run_id": str(run.id),
                        "project_name": os.environ.get("LANGCHAIN_PROJECT", "default"),
                        "run_name": run.name,
                        "execution_status": "FAILED" if error_occurred else "SUCCESS"
                    },
                    "timestamps": {
                        "started_at": start_time.isoformat(),
                        "ended_at": end_time.isoformat(),
                        "total_duration_ms": round(total_duration_ms, 2)
                    },
                    "raw_inputs": {
                        "variables_passed": {
                            "issue_description": issue_description,
                            "expected_result": expected_result
                        },
                        "langsmith_captured_inputs": run.inputs
                    },
                    "raw_outputs": {
                        "final_string_response": response,
                        "langsmith_captured_outputs": run.outputs
                    },
                    
                    "usage_metrics": {
                        "token_counts": prompt_tokens + completion_tokens,
                        "prompt_tokens": prompt_tokens,
                        "completion_tokens": completion_tokens,
                        "extra_metadata": run.extra or {}
                    },
                    "errors": error_occurred
                }
                
                # Write everything as a single line of minified JSON
                logging.info(json.dumps(ultimate_log))
                    
            return response

    # ── translation ────────────────────────────────────────────────────────────────

    def translate_to_tanglish(self, text: str) -> str:
        return english_to_tanglish(text)


    # ── combined pipeline ──────────────────────────────────────────────────────────

    def run_enrichment_pipeline(
        self,
        context_fn,
        issue_description: str,
        expected_result: str= "",
    ) -> dict:
        """
        Called by the classification file — accepts a pre-built context_fn.
        Returns enriched issue + both translations.
        """
        enriched = self.enrich_issue(context_fn, issue_description, expected_result)
        return {
            "enriched_issue": enriched,
            "tanglish": self.translate_to_tanglish(enriched),
        }


    def run_standalone(self, file_path: str, issue_description: str, expected_result: str = "") -> dict:
        """
        Called when used as a standalone script — builds its own collection from a file.
        """
        collection = self.build_collection_from_file(file_path)
        retriever = collection.as_retriever(search_type="similarity", search_kwargs={"k": 5})
        context_fn = self.make_context_fn(retriever)
        return self.run_enrichment_pipeline(context_fn, issue_description, expected_result)


# ── CLI entry point ────────────────────────────────────────────────────────────

if __name__ == "__main__":

    enrich_translation = enrichment_transalation()

    result = (enrich_translation.run_standalone(
        file_path=r"C:\Users\Erwin\Desktop\rag_system\classrag\screen\Real Estate Property Management\Tenant Portal\Tenant Portal.pdf",
        issue_description="The 'Pay Rent' button is red, which users are confusing with an error message."
        ))

    print(f"\nEnriched : {result['enriched_issue']}")
    print(f"Tanglish : {result['tanglish']}")