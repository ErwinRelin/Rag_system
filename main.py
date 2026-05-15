from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from factory import RAGHandlerFactory
from langchain_ollama import ChatOllama, OllamaEmbeddings
import sys
import asyncio
import shutil
import os

if sys.platform == 'win32':
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:8000", "http://localhost:8000"],
    allow_credentials=True,
    allow_methods=["*"],    
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="static"), name="static")

llm = ChatOllama(model="qwen2.5-custom:latest", temperature=0)
embeddings = OllamaEmbeddings(model="nomic-embed-text:latest")

UPLOAD_DIR = "tmp/rag_uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.get("/")
def root():
    return FileResponse("static/index.html")


@app.post("/ask")
async def ask(
    file: UploadFile = File(...),
    question: str = Form(...)
):
    print(f"Received file: {file.filename}, size hint: {file.size}")
    print(f"Question: {question}")
    # Save uploaded file temporarily
    filepath = os.path.join(UPLOAD_DIR, file.filename)

    with open(filepath, "wb") as f:
        shutil.copyfileobj(file.file, f)
    handler = RAGHandlerFactory.get_handler(filepath, llm, embeddings)
    handler.load(filepath)
    response = handler.answer(question)
    return {"answer": response}
