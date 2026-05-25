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

# llm1 = ChatOllama(model="qwen2.5-custom:latest", temperture = 0)
llm2 = ChatOllama(model="gemma4:e2b", temperature = 0)

chroma_client = chromadb.Client()
embedding_fn = OpenCLIPEmbeddings()

collection = Chroma(
    collection_name="my_chunk",
    embedding_function = embedding_fn
    )

folder_path = Path(r"C:\Users\Erwin\Desktop\rag_system\classrag\screen\Real Estate Property Management")

# Read the excel file
df = pd.read_excel('screen\Real Estate Property Management\main5.xls')

# Get column names as a list
screen_names = df['Screen Name'][0:]
for screen_name in screen_names:

    collection = Chroma(
        collection_name=f"my_chunk_{uuid.uuid4()}",
        embedding_function=embedding_fn
    )

    condition = df['Screen Name'] == screen_name
    expected_result = df.loc[condition, 'Expected Result'].values[0]
    issue_description = df.loc[condition, 'Issue Description'].values[0]

    print(f"screen_name: {screen_name}")
    print(f"issue_description: {issue_description}")
    print(f"expected_result: {expected_result}")

    screen_folder = folder_path / screen_name

    for entry in os.listdir(screen_folder):
        full_path = os.path.join(folder_path, screen_name, entry)

        output_dir = "image_dir"
        os.makedirs(output_dir, exist_ok=True)  # ensure folder exists

        # Step 1: Partition without chunking_strategy
        elements = partition(
            filename=full_path,
            strategy="hi_res",
            extract_images_in_pdf=True,
            extract_image_block_types=["Image", "Table"],
            extract_image_block_output_dir=output_dir,
            extract_image_block_to_payload=True,
        )

        # Step 2: Check images on raw elements BEFORE chunking
        for el in elements:
            if el.category == "Image":
                print("✅ Image element found!")
                if hasattr(el.metadata, "image_base64") and el.metadata.image_base64:
                    print(el.metadata.image_base64[:50])
                else:
                    print("⚠️ image_base64 is empty — check if hi_res strategy is working")

        # Step 3: Chunk separately
        chunks = chunk_by_title(
            elements,
            max_characters=2200,
            combine_text_under_n_chars=800,
            overlap=300
        )

        text_chunks = []
        image_chunks = []

        for chunk in chunks:
            # Always collect text
            if chunk.text.strip():
                text_chunks.append(chunk.text)

            # Dig into orig_elements for images
            if hasattr(chunk.metadata, "orig_elements") and chunk.metadata.orig_elements:
                for orig in chunk.metadata.orig_elements:
                    if orig.category == "Image":
                        if hasattr(orig.metadata, "image_base64") and orig.metadata.image_base64:
                            image_chunks.append(orig.metadata.image_base64)

        print(f"Text chunks : {len(text_chunks)}")
        print(f"Image chunks: {len(image_chunks)}")

        image_uris = []

        for img_base64 in image_chunks:
            # Decode base64 → bytes → save as temp PNG file
            img_bytes = base64.b64decode(img_base64)
            img = Image.open(io.BytesIO(img_bytes))
            
            temp_file = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
            img.save(temp_file.name)
            image_uris.append(temp_file.name)

        print(f"Saved {len(image_uris)} images to temp files")

        # Add text chunks to Chroma
        if text_chunks:
            collection.add_texts(
                text_chunks,
                ids=[str(uuid.uuid4()) for _ in text_chunks]
            )

        if image_uris:
            collection.add_images(
                uris=image_uris,
                ids=[str(uuid.uuid4()) for _ in image_uris]
            )

            for path in image_uris:
                try:
                    os.remove(path)
                except Exception as e:
                    print(f"Delete failed: {e}")

    retriever = collection.as_retriever(
        search_type = "similarity",
        search_kwargs = {"k": 8}
    )

    # template = """
    # You are a precise assistant. Answer ONLY using the context below. 
    # If the answer is not present, say "I don't know based on the provided document."

    # Reply with ONLY the category label and a one-sentence reason. Format:
    # Category: <UI Issue | API Issue | Cannot Be Decided>
    # Reason: <one sentence>
    
    # Classify the issue into EXACTLY one of these three categories:
    # UI Issue      — the problem is in the frontend / visual layer (layout, rendering, styling, navigation, form validation, UI component behaviour)
    # API Issue     — the problem is in the backend / data layer (wrong response, missing data, incorrect status code, authentication, slow response, data not persisted)
    # Not enough details given — the context does not contain enough information to determine the root cause

    # Given:
    # - {context}
    # - {issue_description}
    # - {expected_result}
    # """

    # chain = (
    #     {"context": itemgetter("issue_description") | retriever, "issue_description": itemgetter("issue_description"), "expected_result": itemgetter("expected_result")}
    #     | ChatPromptTemplate.from_template(template)
    #     | llm1
    #     | StrOutputParser()
    # )

    # result = chain.invoke({
    #     "issue_description": issue_description,
    #     "expected_result": expected_result
    #     })
    # print(f"qwen: {result}")

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
    - {context}
    - {issue_description}
    - {expected_result}
    """

    chain = (
        {"context": itemgetter("issue_description") | retriever, "issue_description": itemgetter("issue_description"), "expected_result": itemgetter("expected_result")}
        | ChatPromptTemplate.from_template(template)
        | llm2
        | StrOutputParser()
    )

    result = chain.invoke({
        "issue_description": issue_description,
        "expected_result": expected_result
        })
    
    print(f"google: {result}")

