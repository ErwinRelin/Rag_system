import pandas as pd
import os
from unstructured.partition.auto import partition
from unstructured.chunking.title import chunk_by_title
from langchain_experimental.open_clip import OpenCLIPEmbeddings
from langchain_chroma import Chroma
import chromadb
import uuid
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import ChatOllama

llm = ChatOllama(model="qwen2.5-custom:latest", temperture = 0)

chroma_client = chromadb.Client()
embedding_fn = OpenCLIPEmbeddings()

collection = Chroma(
    collection_name="my_chunk",
    embedding_function = embedding_fn
    )

folder_path = "Tenant Portal"

# Read the excel file
df = pd.read_excel('screen\Real Estate Property Management\main5.xls')

# Get column names as a list
screen_name = df['Screen Name'][0]

# Replace 'Your_Column_Name' with the actual name of that column
condition = df['Screen Name'] == screen_name
expected_result = df.loc[condition, 'Expected Result'].values[0]
issue_description = df.loc[condition, 'Issue Description'].values[0]

print(f"issue_description: {issue_description}")
print(f"expected_result: {expected_result}")


for entry in os.listdir(screen_name):
    full_path = os.path.join(folder_path, entry)
    print(full_path)

    output_dir = "image_dir"

    chunks = partition(
        filename = full_path,
        strategy = "hi_res",
        extract_images_in_pdf = True,
        extract_image_block_types = ["Image", "Tables"],
        extract_image_block_output_dir = output_dir,
        extract_image_block_to_payload = True,
        include_original_elements = True,

        chunking_strategy= "by_title",
        max_characters = 1000,
        combine_text_under_n_chars = 500,
        overlap = 100
    )

    for chunk in chunks:
        if chunk.category == "Image":
            print("Image chunk safely captured!")
            print(chunk.metadata.image_base64[:50])

    chunk_strings = [chunk.text for chunk in chunks]

    collection.add_texts (
        chunk_strings,
        ids=[str(uuid.uuid4()) for _ in chunk_strings]
    )

retriever = collection.as_retriever(
    search_type = "similarity",
    search_kwargs = {"k": 5}
)

docs = retriever.invoke(issue_description)
print(docs)

template = """You are a precise assistant. Answer ONLY using the context below. 
If the answer is not present, say "I don't know based on the provided document."
Given:
- {issue_description}
- {context}

Classify the issue into EXACTLY one of these three categories:
  UI Issue      — the problem is in the frontend / visual layer (layout, rendering, styling, navigation, form validation, UI component behaviour)
  API Issue     — the problem is in the backend / data layer (wrong response, missing data, incorrect status code, authentication, slow response, data not persisted)
  Not enough details given — the context does not contain enough information to determine the root cause

Reply with ONLY the category label and a one-sentence reason. Format:
  Category: <UI Issue | API Issue | Cannot Be Decided>
  Reason: <one sentence>"""

chain = (
    {"context": retriever, "issue_description": RunnablePassthrough()}
    | ChatPromptTemplate.from_template(template)
    | llm
    | StrOutputParser()
)

print(chain.invoke(issue_description))

