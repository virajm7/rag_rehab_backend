# app/main.py
import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from app.rag import RAG

load_dotenv()
app = FastAPI(title="FAQ RAG Bot")

# Allow everything (temporary for development & Render hosting)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],       # Allow all origins
    allow_credentials=True,
    allow_methods=["*"],       # Allow all HTTP methods
    allow_headers=["*"],       # Allow all headers
)

# Config from .env
KNOWLEDGE_PATH = os.environ.get("KNOWLEDGE_PATH", "app/data/knowledge.txt")
PINECONE_INDEX = os.environ.get("PINECONE_INDEX_NAME", "faqbot-index")
OPENROUTER_MODEL = os.environ.get(
    "OPENROUTER_MODEL", "meta-llama/llama-3.1-8b-instruct:free"
)

# Instantiate RAG engine
rag = RAG(
    knowledge_path=KNOWLEDGE_PATH,
    pinecone_index_name=PINECONE_INDEX,
    openrouter_model=OPENROUTER_MODEL,
)

@app.get("/")
async def root():
    return {"status": "ok", "message": "FAQ RAG backend awake"}

class ChatRequest(BaseModel):
    message: str

@app.post("/chat")
async def chat(req: ChatRequest):
    question = req.message
    result = rag.ask(question)
    return {
        "reply": result.get("answer", ""),
        "source": result.get("source", None),
    }
