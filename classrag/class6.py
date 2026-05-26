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
from langchain_chroma import Chroma
import chromadb
import uuid
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import ChatOllama
from pdf2image import convert_from_path

# llm1 = ChatOllama(model="qwen2.5-custom:latest", temperture = 0)
llm2 = ChatOllama(model="gemma4:custom", temperature = 0)

chroma_client = chromadb.Client()
embedding_fn = OpenCLIPEmbeddings()

collection = Chroma(
    collection_name="my_chunk",
    embedding_function = embedding_fn
    )

folder_path = Path(r"C:\Users\Erwin\Desktop\rag_system\classrag\screen\Real Estate Property Management")

# Read the excel file
df = pd.read_excel('screen\Real Estate Property Management\excel5.xls')

# Get column names as a list
screen_names = df['Screen Name'].unique()  # ✅ ['Tenant Portal', 'Listing Manager', ...]

for screen_name in screen_names:
    screen_folder = folder_path / screen_name

    # ✅ Build ONE collection per screen name (load all its files first)
    collection = Chroma(
        collection_name=f"my_chunk_{str(uuid.uuid4())[:8]}",
        embedding_function=embedding_fn,
        client = chroma_client
    )

    # ✅ Ingest ALL files for this screen into the collection FIRST
    for entry in os.listdir(screen_folder):
        full_path = os.path.join(folder_path, screen_name, entry)

        print(full_path)

        images = convert_from_path(full_path)

        for i, image in enumerate(images):
            # 1. Define a clean file path for each page
            image_path = f'page_{screen_name}_{i}.jpg'
            image.save(image_path, 'JPEG')

            # 2. Pass the file path inside a list to uris (NOT images)
            collection.add_images(
                uris=[image_path]
            )


    #     output_dir = "image_dir"
    #     os.makedirs(output_dir, exist_ok=True)

    #     elements = partition(
    #         filename=full_path,
    #         strategy="hi_res",
    #         extract_images_in_pdf=True,
    #         extract_image_block_types=["Image", "Table"],
    #         extract_image_block_output_dir=output_dir,
    #         extract_image_block_to_payload=True,
    #     )

    #     chunks = chunk_by_title(
    #         elements,
    #         max_characters=2200,
    #         combine_text_under_n_chars=800,
    #         overlap=300
    #     )

    #     text_chunks = []
    #     image_chunks = []

    #     for chunk in chunks:
    #         if chunk.text.strip():
    #             text_chunks.append(chunk.text)
    #         if hasattr(chunk.metadata, "orig_elements") and chunk.metadata.orig_elements:
    #             for orig in chunk.metadata.orig_elements:
    #                 if orig.category == "Image":
    #                     if hasattr(orig.metadata, "image_base64") and orig.metadata.image_base64:
    #                         image_chunks.append(orig.metadata.image_base64)

    #     image_uris = []
    #     for img_base64 in image_chunks:
    #         img_bytes = base64.b64decode(img_base64)
    #         with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp_file:
    #             temp_path = temp_file.name
    #         with Image.open(io.BytesIO(img_bytes)) as img:
    #             img.save(temp_path)
    #         image_uris.append(temp_path)

    #     if text_chunks:
    #         collection.add_texts(text_chunks, ids=[str(uuid.uuid4()) for _ in text_chunks])
    #     if image_uris:
    #         collection.add_images(uris=image_uris, ids=[str(uuid.uuid4()) for _ in image_uris])
    #         for path in image_uris:
    #             try:
    #                 os.remove(path)
    #             except Exception as e:
    #                 print(f"Delete failed: {e}")

    retriever = collection.as_retriever(search_type="similarity", search_kwargs={"k": 4})

    matched_rows = df[df['Screen Name'] == screen_name]
    for _, row in matched_rows.iterrows():
        issue_description = row['Issue Description']
        expected_result = row['Expected Result']

        print(f"screen_name: {screen_name}")
        print(f"issue_description: {issue_description}")
        print(f"expected_result: {expected_result}")

        template = """
        You are a strict classifier. You must respond in EXACTLY this format and nothing else:
        Category: <UI Issue | API Issue | Cannot Be Decided>
        Reason: <one sentence>

        Rules:
        - Do NOT invent new categories
        - Do NOT add extra text before or after
        - Choose EXACTLY one from: UI Issue, API Issue, Cannot Be Decided

        Category definitions:
        UI Issue     — problem is in the frontend/visual layer ONLY 
                    (layout, rendering, styling, navigation, 
                    form validation, UI component behaviour)

        API Issue    — problem is in the backend/data layer 
                    (wrong response, missing/incorrect data, 
                    authentication, data not persisted,
                    CALCULATION ERRORS, business logic errors,
                    incorrect computed values, formula mistakes)

        Cannot Be Decided — context does not have enough information 
                            to determine root cause

        Respond now in EXACTLY this format:
        Category: 
        Reason: 

        Context from documents:
        {context}

        Issue Description: {issue_description}
        Expected Result: {expected_result}
        """

        chain = (
            {"context": itemgetter("issue_description") | retriever,
             "issue_description": itemgetter("issue_description"),
             "expected_result": itemgetter("expected_result")}
            | ChatPromptTemplate.from_template(template)
            | llm2
            | StrOutputParser()
        )

        result = chain.invoke({
            "issue_description": issue_description,
            "expected_result": expected_result
        })
        print(f"google: {result}\n")

chroma_client.reset()

