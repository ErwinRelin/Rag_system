from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from enrich_tran import enrichment_transalation
from langchain_ollama import ChatOllama, OllamaEmbeddings
import sys
import asyncio
import shutil
import os

if sys.platform == 'win32':
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

app = FastAPI()
enrichment_service = enrichment_transalation()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],    
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="static"), name="static")

UPLOAD_DIR = r"C:\Users\Erwin\Desktop\server_enrich"
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

    response = enrichment_service.run_standalone(
        file_path=filepath, 
        issue_description=question
    )
    
    print(response)
    return {"answer": response["enriched_issue"], "tanglish": response["tanglish"]}