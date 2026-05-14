
from sam import rag 
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # In production, replace with your domain
    allow_methods=["*"],
    allow_headers=["*"],
)

class FileQuestionRequest(BaseModel):
    filename: str
    question: str

@app.post("/ask")
async def ask(data: FileQuestionRequest):

    user_filename = data.filename
    user_question = data.question

    print(f"Target File: {user_filename}")
    print(f"Question: {user_question}")

    answer = rag(user_filename, user_question)
    return {"answer": answer}