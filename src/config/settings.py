# src/config/settings.py
import os
from functools import lru_cache
from pathlib import Path
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_pinecone import PineconeVectorStore

# Load environment variables from .env file
env_path = Path(__file__).resolve().parents[2] / ".env"
load_dotenv(env_path)

# ── Environment variables ───────────────────────────────────────────────────
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
if not PINECONE_API_KEY:
    raise ValueError("PINECONE_API_KEY not found in environment variables")

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY not found in environment variables")

PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "csea-assistant")  # Read from .env, fallback to default

DOCS_FOLDER = "./documents"
FEEDBACK_FILE = "feedback.json"

# ── Embedding model ─────────────────────────────────────────────────────────
@lru_cache(maxsize=1)
def get_embeddings():
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# ── LLM ─────────────────────────────────────────────────────────────────────
def get_llm():
    return ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0.0,
        groq_api_key=GROQ_API_KEY
    )

# ── Pinecone vector store & retriever ───────────────────────────────────────
def get_vectorstore():
    # No 'pinecone_api_key' arg needed anymore - it reads from env var automatically
    return PineconeVectorStore.from_existing_index(
        index_name=PINECONE_INDEX_NAME,
        embedding=get_embeddings()
    )

def get_retriever():
    vectorstore = get_vectorstore()
    return vectorstore.as_retriever(search_kwargs={"k": 15})