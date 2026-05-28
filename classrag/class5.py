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
import chromadb
import uuid
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import ChatOllama
from langchain_community.embeddings import HuggingFaceEmbeddings
from deep_translator import GoogleTranslator

# llm1 = ChatOllama(model="qwen2.5-custom:latest", temperture = 0)
llm2 = ChatOllama(model="qwen2.5vl:7custom", temperature = 0)

all_results = []

chroma_client = chromadb.Client()
embedding_fn = OpenCLIPEmbeddings()
text_embedding_fn = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

collection = Chroma(
    collection_name="my_chunk",
    embedding_function = embedding_fn,
    client = chroma_client
    )

folder_path = Path(r"C:\Users\Erwin\Desktop\rag_system\classrag\screen\Real Estate Property Management")

# Read the excel file
df = pd.read_excel('screen\Real Estate Property Management\issues1.xlsx')


# Get column names as a list
screen_names = df['Screen Name'].unique()


for screen_name in screen_names:
    screen_folder = folder_path / screen_name

    # ✅ Build ONE collection per screen name (load all its files first)
    collection = Chroma(
        collection_name=f"my_chunk_{uuid.uuid4()}",
        embedding_function=embedding_fn
    )

    output_dir = "image_dir"
    os.makedirs(output_dir, exist_ok=True)

    # ✅ Ingest ALL files for this screen into the collection FIRST
    for entry in os.listdir(screen_folder):
        full_path = os.path.join(folder_path, screen_name, entry)

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
            multipage_sections = False
        )

        text_chunks = []
        image_chunks = []

        seen_images = set()

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
        for img_base64 in image_chunks:
            img_bytes = base64.b64decode(img_base64)
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp_file:
                temp_path = temp_file.name
            with Image.open(io.BytesIO(img_bytes)) as img:
                img.save(temp_path)
            image_uris.append(temp_path)

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

    matched_rows = df[df['Screen Name'] == screen_name]
    for _, row in matched_rows.iterrows():
        issue_description = row['Issue Description']
        expected_result = row['Expected Outcome']

        print(f"screen_name: {screen_name}")
        print(f"issue_description: {issue_description}")
        print(f"expected_result: {expected_result}")

        def make_context_fn(retriever, embedding_fn):
            def context_from_inputs(inputs) -> str:
                if isinstance(inputs, dict):
                    query = inputs.get("issue_description", "") + " " + inputs.get("expected_result", "")
                else:
                    query = str(inputs)

                # OpenCLIP wraps text embeddings in an extra list — unwrap it
                raw = embedding_fn.embed_query(query)
                if isinstance(raw[0], list):
                    query_embedding = raw[0]   # unwrap one level [[...]] -> [...]
                else:
                    query_embedding = raw      # already flat [...]

                docs = retriever.vectorstore.similarity_search_by_vector(
                    query_embedding, k=5
                )
                return "\n\n".join(d.page_content for d in docs)
            return context_from_inputs

        retriever = collection.as_retriever(search_type="similarity", search_kwargs={"k": 5})
        context_from_inputs = make_context_fn(retriever, embedding_fn)

        prompt = """You are a technical writer. Using ONLY the BRD context below, rewrite the issue description into a single clear, specific sentence that identifies the exact component, the broken behavior, and the expected behavior.

        Context: {context}
        Issue Description: {issue_description}
        Expected Outcome: {expected_result}

        Output one sentence only. No headings, no bullet points, no explanation.
        """

        enrichment_chain = (
            {
                "context": context_from_inputs,
                "issue_description": itemgetter("issue_description"),
                "expected_result": itemgetter("expected_result"),  # also passed to prompt
            }
            | ChatPromptTemplate.from_template(prompt)
            | llm2
            | StrOutputParser()
        )

        template = """
        You are a precise assistant. Answer ONLY using the context below. 
        If the answer is not present, say "I don't know based on the provided document."

        Reply with ONLY the category label and a one-sentence reason. Format:
        Category: <UI Issue | API Issue | Cannot Be Decided>
        Reason: <one sentence>
        
        Classify the issue into EXACTLY one of these three categories:
        UI Issue      — the problem is in the frontend / visual layer (layout, rendering, styling, navigation, form validation, UI component behaviour)
        API Issue     — the problem is in the backend / data layer (wrong response, missing data, incorrect status code, authentication, slow response, data not persisted)
        Not enough details given — the context does not contain enough information to determine the root cause

        Given:
        - context from the documents: {context}
        - Issue description: {enriched_issue}
        """

        def parse_result(text: str) -> dict:
            lines = {k.strip(): v.strip() for line in text.strip().splitlines()
                    if ": " in line for k, v in [line.split(": ", 1)]}
            category = lines.get("Category", "Cannot Be Decided")
            valid = {"UI Issue", "API Issue", "Cannot Be Decided"}
            if category not in valid:
                category = "Cannot Be Decided"
            return {"category": category, "reason": lines.get("Reason", "")}

        chain = (
            {"context": context_from_inputs,
             "enriched_issue": RunnablePassthrough()}
            | ChatPromptTemplate.from_template(template)
            | llm2
            | StrOutputParser()
            | parse_result
        )

        enriched_issue = enrichment_chain.invoke({
            "issue_description": issue_description,
            "expected_result": expected_result,
        })

        print(f"enriched_issue: {enriched_issue}")
        transalated_issue = GoogleTranslator(source='auto', target='ta').translate(enriched_issue)

        result = chain.invoke(enriched_issue)

        print(f"result: {result}\n")

        all_results.append({
            "Enriched Issue": enriched_issue,
            "Tamil Issue": transalated_issue,
            "Classification": result["category"],
            "Reason": result["reason"],
        })

final_df = pd.DataFrame(all_results)

with pd.ExcelWriter('screen\Real Estate Property Management\issues1.xlsx', engine="openpyxl", mode="a", if_sheet_exists="overlay") as writer:
    start_column = writer.sheets["sheet1"].max_column
    final_df.to_excel(writer, sheet_name="sheet1", startcol=start_column, header=True, index=False)

