import os
from functools import lru_cache
from pathlib import Path
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_pinecone import PineconeVectorStore

# 1. LOAD ENV (Single source of truth for the entire app)
env_path = Path(__file__).resolve().parents[2] / ".env"
load_dotenv(env_path)

TESSERACT_CMD = os.getenv("TESSERACT_CMD", r"C:\Program Files\Tesseract-OCR\tesseract.exe")

# 2. VALIDATION (Fail fast at startup)
PINECONE_API_KEY  = os.getenv("PINECONE_API_KEY")
if not PINECONE_API_KEY: raise ValueError("PINECONE_API_KEY missing. Check .env file.")

GROQ_API_KEY      = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY: raise ValueError("GROQ_API_KEY missing. Check .env file.")

SUPABASE_URL      = os.getenv("SUPABASE_URL")
if not SUPABASE_URL: raise ValueError("SUPABASE_URL missing. Check .env file.")

SUPABASE_KEY      = os.getenv("SUPABASE_KEY")
if not SUPABASE_KEY: raise ValueError("SUPABASE_KEY missing. Check .env file.")

# 3. CONFIG CONSTANTS
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "csea-assistant")
GROQ_MODEL          = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

# Absolute pathing prevents "folder not found" errors during deployment
DOCS_FOLDER = Path(__file__).resolve().parents[2] / "documents"

# 4. CACHED FACTORIES
@lru_cache(maxsize=1)
def get_embeddings() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

@lru_cache(maxsize=1)
def get_llm(temperature: float = 0.0) -> ChatGroq:
    """Pass temperature=0.0 for the Critic, and higher for conversational generation if needed."""
    return ChatGroq(model=GROQ_MODEL, temperature=temperature, groq_api_key=GROQ_API_KEY)

# NO LRU CACHE HERE: Forces a fresh connection so Pinecone doesn't time out during your defense!
def get_vectorstore() -> PineconeVectorStore:
    return PineconeVectorStore.from_existing_index(
        index_name=PINECONE_INDEX_NAME,
        embedding=get_embeddings()
    )

def get_retriever(k: int = 5):
    """Centralized retriever call to prevent mismatched 'k' values."""
    return get_vectorstore().as_retriever(search_kwargs={"k": k})