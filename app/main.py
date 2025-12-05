# app/main.py
import os
from fastapi import FastAPI
from pydantic import BaseModel
from dotenv import load_dotenv
from app.rag import RAG

load_dotenv()
app = FastAPI(title="FAQ RAG Bot")

# (Optional) read config from env – RAG itself already uses these defaults
KNOWLEDGE_PATH = os.environ.get("KNOWLEDGE_PATH", "app/data/knowledge.txt")
PINECONE_INDEX = os.environ.get("PINECONE_INDEX_NAME", "faqbot-index")
OPENROUTER_MODEL = os.environ.get("OPENROUTER_MODEL", "meta-llama/llama-3.1-8b-instruct:free")

# instantiate RAG (this will load and index the doc on startup)
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
    resp = rag.ask(question)
    return {
        "reply": resp.get("answer", ""),
        "source": resp.get("source", None),
    }
