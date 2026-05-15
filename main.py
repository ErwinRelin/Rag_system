from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from factory import RAGHandlerFactory
from langchain_ollama import ChatOllama, OllamaEmbeddings
import sys
import asyncio

if sys.platform == 'win32':
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://127.0.0.1:5500",
        "http://localhost:5500",
        "http://127.0.0.1:8000",
        "http://localhost:8000"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="static"), name="static")

llm = ChatOllama(model="qwen2.5-custom:latest", temperature=0)
embeddings = OllamaEmbeddings(model="nomic-embed-text:latest")

class FileQuestionRequest(BaseModel):
    filename: str
    question: str

@app.get("/")
def root():
    return FileResponse("static/index.html")

@app.post("/ask")
async def ask(data: FileQuestionRequest):
    filename = data.filename
    question = data.question

    print(f"Target File: {filename}")
    print(f"Question: {question}")

    handler = RAGHandlerFactory.get_handler(filename, llm, embeddings)
    handler.load(filename)
    response = handler.answer(question)
    return {"answer": response}