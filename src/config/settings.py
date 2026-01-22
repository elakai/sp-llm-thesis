# src/config/settings.py
import os
from functools import lru_cache
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_chroma import Chroma

# ── Constants ────────────────────────────────────────────────────────────────
DB_PATH = "./chroma_db"
DOCS_FOLDER = "./documents"
FEEDBACK_FILE = "feedback.json"

# IMPORTANT: Use environment variable in production / thesis demo
# For now keeping hardcoded as in original
GROQ_API_KEY = "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"

# ── Models & Factories ───────────────────────────────────────────────────────
@lru_cache(maxsize=1)
def get_embeddings():
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

def get_llm():
    return ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0.1,
        groq_api_key=GROQ_API_KEY
    )

# Do NOT initialize Chroma here globally – it can cause issues in some environments
# We initialize it only when needed (in retrieval / ingestion)
def get_db():
    embeddings = get_embeddings()
    return Chroma(persist_directory=DB_PATH, embedding_function=embeddings)

def get_retriever():
    db = get_db()
    return db.as_retriever(search_kwargs={"k": 15})