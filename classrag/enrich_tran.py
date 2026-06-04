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
from langchain_core.callbacks import BaseCallbackHandler
import chromadb
import uuid
import logging
from langchain_core.tracers.context import collect_runs
import traceback
from datetime import datetime, timezone
import json
import config

# import your tanglish translator
from tanglish import english_to_tanglish


# os.environ["LANGCHAIN_TRACING_V2"] = "true"
# os.environ["LANGCHAIN_API_KEY"] = config.MY_API_KEY

# ── models ─────────────────────────────────────────────────────────────────────

llm = ChatOllama(model="gemma3:12b", temperature=0)
embedding_fn = OpenCLIPEmbeddings()

ENRICHMENT_PROMPT = """
You are a senior Business Analyst creating developer-ready defect descriptions.

Using ONLY the BRD context, issue description, and expected outcome, rewrite the issue as a single concise defect statement.

Rules:
- Describe the current incorrect behavior, not the desired behavior.
- Identify the affected component, workflow, field, API, or UI element when possible.
- Include relevant business or system impact if directly related to the issue.
- Use BRD terminology where applicable.
- Preserve all meaningful details from the issue and expected outcome.
- Do not invent new defects, side effects, fields, or requirements.
- Do not write acceptance criteria.
- Do not use phrases such as:
  - "should"
  - "must"
  - "needs to"
  - "expected to"
- Write in defect format:
  [Component] + [Incorrect Behavior] + [Impact]

Context:
{context}

Issue Description:
{issue_description}

Expected Outcome:
{expected_result}

Output one sentence only.
"""

OUTPUT_LOG_PATH = r"C:\Users\Erwin\Desktop\ultimate_llm_system.json"

# class UltimateLoggingHandler(BaseCallbackHandler):
#     """A clean event listener that hooks directly into the LLM lifecycle."""
    
#     def __init__(self, user_id: str, custom_context: str):
#         super().__init__()
#         self.user_id = user_id
#         self.custom_context = custom_context
#         self.start_time = None

#     def on_llm_start(self, serialized, prompts, **kwargs):
#         """Triggers exactly when the prompt leaves your computer."""
#         self.start_time = datetime.now(timezone.utc)

#     def on_llm_end(self, response, **kwargs):
#         """Triggers exactly when the model outputs text."""
#         end_time = datetime.now(timezone.utc)
#         duration_ms = (end_time - self.start_time).total_seconds() * 1000 if self.start_time else 0
        
#         # 1. Safely extract standard generation text and run IDs
#         generation_info = response.generations[0][0]
#         final_text = generation_info.text
#         run_id = kwargs.get("run_id", "unknown-id")
        
#         # 2. Extract Token counts safely from LangChain response structures
#         llm_output = response.llm_output or {}
#         token_usage = llm_output.get("token_usage", {}) or {}
        
#         # 3. Create a clean structured payload map
#         full_telemetry = {
#             "identity": {
#                 "run_id": str(run_id),
#                 "user_id": self.user_id
#             },
#             "performance": {
#                 "started_at": self.start_time.isoformat() if self.start_time else None,
#                 "ended_at": end_time.isoformat(),
#                 "duration_ms": round(duration_ms, 2)
#             },
#             "data": {
#                 "passed_context": self.custom_context,
#                 "final_output": final_text,
#                 "generation_metadata": generation_info.generation_info
#             },
#             "usage": {
#                 "prompt_tokens": token_usage.get("prompt_tokens", 0),
#                 "completion_tokens": token_usage.get("completion_tokens", 0),
#                 "total_tokens": token_usage.get("total_tokens", 0)
#             }
#         }
        
#         # 4. Instantly append directly to your desktop file line-by-line
#         with open(OUTPUT_LOG_PATH, "a", encoding="utf-8") as f:
#             f.write(json.dumps(full_telemetry) + "\n")
#         print(f"✅ Callback Success! Telemetry written directly to: {OUTPUT_LOG_PATH}")

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
            
            return "\n\n".join(d.page_content for d in docs)

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



    def on_llm_error(self, error, **kwargs):
        """Triggers automatically if the model crashes or times out."""
        error_payload = {
            "status": "CRASHED",
            "error_message": str(error),
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        with open(OUTPUT_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(error_payload) + "\n")


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

        # logger_callback = UltimateLoggingHandler(user_id="Erwin_Local", custom_context=context_fn)

        response = chain.invoke({
            "issue_description": issue_description,
            "expected_result": expected_result},
            # config={"callbacks": [logger_callback]}
        )
                    
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
        file_path=r"C:\Users\Erwin\Desktop\rag_system\classrag\screen\Real Estate Property Management\Contractor View\Contractor View.pdf",
        issue_description="Offline contractor update overwrites higher priority assignment changes after reconnection."
        ))

    print(f"\nEnriched : {result['enriched_issue']}")
    print(f"Tanglish : {result['tanglish']}")