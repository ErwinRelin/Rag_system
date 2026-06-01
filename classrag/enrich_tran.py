# enrichment.py

import argparse
from pathlib import Path
from operator import itemgetter

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import ChatOllama
from langchain_chroma import Chroma
from langchain_experimental.open_clip import OpenCLIPEmbeddings
from deep_translator import GoogleTranslator
import chromadb
import uuid

# import your tanglish translator
from tanglish import english_to_tanglish

# ── models ─────────────────────────────────────────────────────────────────────

llm = ChatOllama(model="qwen2.5vl:7custom", temperature=0)
embedding_fn = OpenCLIPEmbeddings()

ENRICHMENT_PROMPT = """You are a technical writer. Using ONLY the BRD context below, rewrite the issue description into a single clear, specific sentence that identifies the exact component, the broken behavior, and the expected behavior.

Context: {context}
Issue Description: {issue_description}
Expected Outcome: {expected_result}

Output one sentence only. No headings, no bullet points, no explanation.
"""


# ── context retrieval ──────────────────────────────────────────────────────────

def make_context_fn(retriever):
    def context_from_inputs(inputs) -> str:
        query = (
            inputs.get("issue_description", "") + " " + inputs.get("expected_result", "")
            if isinstance(inputs, dict)
            else str(inputs)
        )
        raw = embedding_fn.embed_query(query)
        query_embedding = raw[0] if isinstance(raw[0], list) else raw
        docs = retriever.vectorstore.similarity_search_by_vector(query_embedding, k=5)
        return "\n\n".join(d.page_content for d in docs)
    return context_from_inputs


def build_collection_from_file(file_path: str) -> Chroma:
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
        extract_images_in_pdf=True,
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
    if image_uris:
        try:
            collection.add_images(uris=image_uris, ids=[str(uuid.uuid4()) for _ in image_uris])
        finally:
            for path in image_uris:
                try:
                    os.remove(path)
                except OSError:
                    pass

    return collection


# ── enrichment ─────────────────────────────────────────────────────────────────

def build_enrichment_chain(context_fn):
    return (
        {
            "context": context_fn,
            "issue_description": itemgetter("issue_description"),
            "expected_result": itemgetter("expected_result"),
        }
        | ChatPromptTemplate.from_template(ENRICHMENT_PROMPT)
        | llm
        | StrOutputParser()
    )


def enrich_issue(context_fn, issue_description: str, expected_result: str) -> str:
    chain = build_enrichment_chain(context_fn)
    return chain.invoke({
        "issue_description": issue_description,
        "expected_result": expected_result,
    })


# ── translation ────────────────────────────────────────────────────────────────

def translate_to_tanglish(text: str) -> str:
    return english_to_tanglish(text)


# ── combined pipeline ──────────────────────────────────────────────────────────

def run_enrichment_pipeline(
    context_fn,
    issue_description: str,
    expected_result: str,
) -> dict:
    """
    Called by the classification file — accepts a pre-built context_fn.
    Returns enriched issue + both translations.
    """
    enriched = enrich_issue(context_fn, issue_description, expected_result)
    return {
        "enriched_issue": enriched,
        "tanglish": translate_to_tanglish(enriched),
    }


def run_standalone(file_path: str, issue_description: str, expected_result: str) -> dict:
    """
    Called when used as a standalone script — builds its own collection from a file.
    """
    collection = build_collection_from_file(file_path)
    retriever = collection.as_retriever(search_type="similarity", search_kwargs={"k": 5})
    context_fn = make_context_fn(retriever)
    return run_enrichment_pipeline(context_fn, issue_description, expected_result)


# ── CLI entry point ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    file_path         = r"C:\Users\Erwin\Desktop\rag_system\classrag\screen\Real Estate Property Management\Tenant Portal\Tenant Portal.pdf"
    issue_description = "Login button does nothing after clicking"
    expected_result   = "User should be redirected to the dashboard"

    result = run_standalone(file_path, issue_description, expected_result)

    print(f"\nEnriched : {result['enriched_issue']}")
    print(f"Tanglish : {result['tanglish']}")