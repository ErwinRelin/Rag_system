import pandas as pd
from pathlib import Path
import os
import base64
import tempfile
from PIL import Image as Image
import io
from operator import itemgetter
from unstructured.partition.auto import partition
from unstructured.chunking.title import chunk_by_title
from langchain_experimental.open_clip import OpenCLIPEmbeddings
from langchain_core.runnables import RunnablePassthrough
from langchain_chroma import Chroma
import uuid
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import ChatOllama
from enrich_tran import run_enrichment_pipeline, make_context_fn

# ── constants ──────────────────────────────────────────────────────────────────

FOLDER_PATH = Path(r"C:\Users\Erwin\Desktop\rag_system\classrag\screen\Real Estate Property Management")
EXCEL_PATH  = r"screen\Real Estate Property Management\issues1.xlsx"
SHEET_NAME  = "Sheet1"

llm         = ChatOllama(model="qwen2.5vl:7custom", temperature=0)
embedding_fn = OpenCLIPEmbeddings()

CLASSIFICATION_TEMPLATE = """
You are a precise assistant. Answer ONLY using the context below.
If the answer is not present, say "I don't know based on the provided document."

Reply with ONLY the category label and a one-sentence reason. Format:
Category: <UI Issue | API Issue | Cannot Be Decided>
Reason: <one sentence>

Classify the issue into EXACTLY one of these three categories:
UI Issue          — the problem is in the frontend / visual layer (layout, rendering, styling, navigation, form validation, UI component behaviour)
API Issue         — the problem is in the backend / data layer (wrong response, missing data, incorrect status code, authentication, slow response, data not persisted)
Cannot Be Decided — the context does not contain enough information to determine the root cause

Given:
- context from the documents: {context}
- Issue description: {enriched_issue}
"""


# ── helpers (defined once, outside the loop) ───────────────────────────────────

def parse_result(text: str) -> dict:
    lines = {
        k.strip(): v.strip()
        for line in text.strip().splitlines()
        if ": " in line
        for k, v in [line.split(": ", 1)]
    }
    category = lines.get("Category", "Cannot Be Decided")
    if category not in {"UI Issue", "API Issue", "Cannot Be Decided"}:
        category = "Cannot Be Decided"
    return {"category": category, "reason": lines.get("Reason", "")}


def build_classification_chain(context_fn):
    return (
        {
            "context": context_fn,
            "enriched_issue": RunnablePassthrough(),
        }
        | ChatPromptTemplate.from_template(CLASSIFICATION_TEMPLATE)
        | llm
        | StrOutputParser()
        | parse_result
    )


# ── ingestion ──────────────────────────────────────────────────────────────────

def ingest_screen(screen_folder: Path) -> Chroma:
    collection = Chroma(
        collection_name=f"my_chunk_{uuid.uuid4()}",
        embedding_function=embedding_fn,
    )
    os.makedirs("image_dir", exist_ok=True)

    for entry in os.listdir(screen_folder):
        full_path = str(screen_folder / entry)

        elements = partition(
            filename=full_path,
            strategy="auto",
            extract_images_in_pdf=True,
            extract_image_block_types=["Image", "Table"],
            extract_image_block_to_payload=True,
        )
        chunks = chunk_by_title(
            elements,
            max_characters=1500,
            combine_text_under_n_chars=700,
            overlap=300,
            multipage_sections=False,
        )

        text_chunks, image_chunks, seen_images = [], [], set()
        for chunk in chunks:
            if chunk.text.strip():
                text_chunks.append(chunk.text)
            for orig in getattr(chunk.metadata, "orig_elements", None) or []:
                if orig.category == "Image":
                    b64 = getattr(orig.metadata, "image_base64", None)
                    if b64 and b64 not in seen_images:
                        seen_images.add(b64)
                        image_chunks.append(b64)

        image_uris = []
        for b64 in image_chunks:
            img_bytes = base64.b64decode(b64)
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
                Image.open(io.BytesIO(img_bytes)).save(f.name)
                image_uris.append(f.name)

        if text_chunks:
            collection.add_texts(text_chunks, ids=[str(uuid.uuid4()) for _ in text_chunks])
        try:
            if image_uris:
                collection.add_images(uris=image_uris, ids=[str(uuid.uuid4()) for _ in image_uris])
        finally:
            for path in image_uris:
                try:
                    os.remove(path)
                except OSError:
                    pass

    return collection


# ── per-row processing ─────────────────────────────────────────────────────────

def process_row(row, context_fn) -> dict:
    issue_description = row["Issue Description"]
    expected_result   = row["Expected Outcome"]

    print(f"  issue: {issue_description}")

    # enrich + translate via enrich_tran.py
    enrichment_result = run_enrichment_pipeline(
        context_fn=context_fn,
        issue_description=issue_description,
        expected_result=expected_result,
    )

    enriched_issue = enrichment_result["enriched_issue"]
    print(f"  enriched: {enriched_issue}")

    # classify
    chain  = build_classification_chain(context_fn)
    result = chain.invoke(enriched_issue)
    print(f"  category: {result['category']} — {result['reason']}\n")

    return {
        "Enriched Issue": enriched_issue,
        "Tamil Issue":    enrichment_result["tamil"],
        "Tanglish Issue": enrichment_result["tanglish"],
        "Classification": result["category"],
        "Reason":         result["reason"],
    }


# ── output ─────────────────────────────────────────────────────────────────────

def save_results(results: list[dict]) -> None:
    final_df = pd.DataFrame(results)
    with pd.ExcelWriter(EXCEL_PATH, engine="openpyxl", mode="a", if_sheet_exists="overlay") as writer:
        start_col = writer.sheets[SHEET_NAME].max_column - 1  # fix off-by-one
        final_df.to_excel(writer, sheet_name=SHEET_NAME, startcol=start_col, header=True, index=False)
    print(f"Saved to {EXCEL_PATH}")


# ── main ───────────────────────────────────────────────────────────────────────

def main():
    df          = pd.read_excel(EXCEL_PATH)
    all_results = []

    for screen_name in df["Screen Name"].unique():
        print(f"\nscreen: {screen_name}")

        collection   = ingest_screen(FOLDER_PATH / screen_name)
        retriever    = collection.as_retriever(search_type="similarity", search_kwargs={"k": 5})
        context_fn   = make_context_fn(retriever, embedding_fn)  # from enrich_tran

        for _, row in df[df["Screen Name"] == screen_name].iterrows():
            all_results.append(process_row(row, context_fn))

    save_results(all_results)


if __name__ == "__main__":
    main()