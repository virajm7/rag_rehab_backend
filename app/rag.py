# app/rag.py
import os
import json
from typing import List, Dict, Optional

from dotenv import load_dotenv
import requests

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore

from pinecone import Pinecone, ServerlessSpec
from app.utils import load_text_or_pdf

# Load environment variables from .env
load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
if not PINECONE_API_KEY:
    raise RuntimeError("PINECONE_API_KEY is not set. Please add it to your .env file.")

PINECONE_INDEX = os.environ.get("PINECONE_INDEX_NAME", "faqbot-index")

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
if not OPENROUTER_API_KEY:
    raise RuntimeError("OPENROUTER_API_KEY is not set. Please add it to your .env file.")

OPENROUTER_MODEL = os.environ.get(
    "OPENROUTER_MODEL", "meta-llama/llama-3.1-8b-instruct:free"
)
KNOWLEDGE_PATH = os.environ.get("KNOWLEDGE_PATH", "app/data/knowledge.txt")
MAX_CHUNK_SIZE = int(os.environ.get("MAX_CHUNK_SIZE", 800))
CHUNK_OVERLAP = int(os.environ.get("CHUNK_OVERLAP", 100))

# Use HuggingFace embedding model (all-MiniLM-L6-v2 => 384-dim)
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


class RAG:
    def __init__(
        self,
        knowledge_path: str = KNOWLEDGE_PATH,
        pinecone_index_name: str = PINECONE_INDEX,
        openrouter_model: str = OPENROUTER_MODEL,
    ):
        self.knowledge_path = knowledge_path
        self.pinecone_index_name = pinecone_index_name
        self.openrouter_model = openrouter_model

        print("\n🔹 Using OpenRouter model:", self.openrouter_model)

        # Initialize Pinecone serverless
        self.pc = Pinecone(api_key=PINECONE_API_KEY)

        # Create serverless index if it doesn't exist
        existing_indexes = [idx["name"] for idx in self.pc.list_indexes().indexes]
        if self.pinecone_index_name not in existing_indexes:
            print("🆕 Creating Pinecone index...")
            self.pc.create_index(
                name=self.pinecone_index_name,
                dimension=384,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1"),
            )

        self.index = self.pc.Index(self.pinecone_index_name)

        # Embeddings
        self.embedder = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)

        # Pinecone Vector Store
        self.vectorstore = PineconeVectorStore(
            index_name=self.pinecone_index_name,
            embedding=self.embedder,
            pinecone_api_key=PINECONE_API_KEY,
        )

        self._ensure_docs_indexed()

    # ------------------------------------------------------------------
    # Indexing
    # ------------------------------------------------------------------
    def _ensure_docs_indexed(self):
        """If the Pinecone index is empty, load knowledge.txt and index it."""
        stats = self.index.describe_index_stats()
        vcount = stats.get("total_vector_count", 0)

        if vcount == 0:
            print("📥 Index empty — loading knowledge file...")
            text = load_text_or_pdf(self.knowledge_path)
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=MAX_CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
            )
            docs = splitter.split_text(text)

            self.vectorstore.add_texts(
                texts=docs,
                metadatas=[{"source": os.path.basename(self.knowledge_path)}]
                * len(docs),
                ids=[f"doc-{i}" for i in range(len(docs))],
            )

            print(f"✔ Indexed {len(docs)} chunks into Pinecone!")

    # ------------------------------------------------------------------
    # Rule-based answers (no OpenRouter call)
    # ------------------------------------------------------------------
    def _rule_based_answer(self, q: str) -> Optional[str]:
        q = q.lower().strip()

        # Normalize a bit
        q_clean = q.rstrip("?.!").strip()

        # Very common questions
        if "clinic name" in q_clean or "name of clinic" in q_clean or q_clean == "name":
            return "The clinic is called Reviva Rehab Clinic."

        if "where" in q_clean or "location" in q_clean or "address" in q_clean:
            return (
                "Reviva Rehab Clinic is located at #21 Wellness Avenue, "
                "JP Nagar, Bengaluru, India."
            )

        if "phone" in q_clean or "contact" in q_clean or "number" in q_clean:
            return (
                "You can contact Reviva Rehab Clinic at +91 98765 00000 "
                "or email contact@revivarehab.in."
            )

        if "email" in q_clean:
            return "The clinic email is contact@revivarehab.in."

        if "website" in q_clean or "site" in q_clean:
            return "Our website is www.revivarehab.in."

        if "time" in q_clean or "timing" in q_clean or "hours" in q_clean or "open" in q_clean or "close" in q_clean:
            return (
                "Reviva Rehab Clinic is open Monday to Saturday "
                "from 09:00 AM to 08:00 PM and closed on Sundays."
            )

        # Services / treatments / therapies
        service_keywords = [
            "service",
            "services",
            "treatment",
            "treatments",
            "therapy",
            "therapies",
            "rehab",
            "offer",
            "provide",
        ]
        if any(k in q_clean for k in service_keywords):
            return (
                "Reviva Rehab Clinic offers physiotherapy, post-surgery rehabilitation, "
                "neurological rehabilitation, sports injury rehabilitation, pediatric therapy, "
                "spine care therapy, pain management, geriatric rehabilitation, and home visit "
                "physiotherapy on appointment."
            )

        # If nothing rule-based matches
        return None

    # ------------------------------------------------------------------
    # Main ask method
    # ------------------------------------------------------------------
    def ask(self, question: str, top_k: int = 4) -> Dict:
        qnorm = question.lower().strip().rstrip("?.!")

        # 0) Try rule-based answer first (fast, no API call)
        rb = self._rule_based_answer(qnorm)
        if rb is not None:
            return {"answer": rb, "source": []}

        # 1) Normal RAG flow: search vectorstore
        query_result = self.vectorstore.similarity_search_with_score(
            question, k=top_k
        )

        contexts: List[Dict] = [
            {"text": doc.page_content, "meta": doc.metadata, "score": float(score)}
            for doc, score in query_result
        ]

        prompt = self._build_prompt(question, contexts)

        # 2) Try OpenRouter; if it fails, use graceful fallback
        try:
            answer = self._call_openrouter(prompt)
        except Exception as e:
            print("⚠️ OpenRouter error, using fallback:", e)
            answer = self._fallback_answer(question, contexts, e)

        return {"answer": answer, "source": contexts}

    # ------------------------------------------------------------------
    # Prompt construction
    # ------------------------------------------------------------------
    def _build_prompt(self, question: str, ctx: List[Dict]) -> str:
        ctx_texts = "\n\n---\n\n".join(x["text"] for x in ctx)

        return f"""
You are a friendly assistant for Reviva Rehab Clinic, a physiotherapy and rehabilitation center.

RULES:
- Use ONLY the information from the context below.
- If the answer is not found in the context, reply exactly: "I don't know based on the provided documents."
- Keep answers short and clear (1–3 sentences).
- Do NOT show metadata, scores, or file names. Just speak like a normal human assistant.

CONTEXT:
{ctx_texts}

QUESTION: {question}

Answer:
""".strip()

    # ------------------------------------------------------------------
    # Fallback logic (when OpenRouter fails – rate limit, 404, etc.)
    # ------------------------------------------------------------------
    def _fallback_answer(
        self, question: str, contexts: List[Dict], error: Exception
    ) -> str:
        q = question.lower()

        # Reuse our rule-based logic as a backup too
        rb = self._rule_based_answer(q)
        if rb is not None:
            return rb

        # If we reach here, we truly don't know
        return (
            "I'm sorry, I don't know based on the provided documents. "
            "Please contact the clinic directly for more detailed information."
        )

    # ------------------------------------------------------------------
    # OpenRouter call
    # ------------------------------------------------------------------
    def _call_openrouter(self, prompt: str) -> str:
        url = "https://openrouter.ai/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json",
        }
        body = {
            "model": self.openrouter_model,
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt},
            ],
            "max_tokens": 256,
            "temperature": 0.0,
        }

        resp = requests.post(url, headers=headers, json=body, timeout=30)

        # If status is not OK, raise a clean error so fallback can take over
        if not resp.ok:
            # Avoid dumping giant HTML; just give a short snippet
            text_snippet = resp.text[:300].replace("\n", " ")
            raise RuntimeError(
                f"OpenRouter HTTP {resp.status_code}: {text_snippet}"
            )

        try:
            data = resp.json()
        except json.JSONDecodeError:
            raise RuntimeError(
                f"OpenRouter returned non-JSON response (first 200 chars): {resp.text[:200]}"
            )

        if "choices" in data and data["choices"]:
            return data["choices"][0]["message"]["content"].strip()

        # If structure unexpected, raise so fallback is used
        raise RuntimeError(f"OpenRouter unexpected response: {data}")
