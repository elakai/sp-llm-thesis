import os
from functools import lru_cache
from pathlib import Path
from dotenv import load_dotenv # pip install python-dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_pinecone import PineconeVectorStore

# ─────────────────────────────────────────────────────────────────────────────
# 1. LOAD ENVIRONMENT VARIABLES (The Upgrade)
# ─────────────────────────────────────────────────────────────────────────────
# This automatically finds the .env file in your project root
env_path = Path(__file__).resolve().parents[2] / ".env"
load_dotenv(env_path)

# ─────────────────────────────────────────────────────────────────────────────
# 2. VALIDATION
# ─────────────────────────────────────────────────────────────────────────────
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
if not PINECONE_API_KEY:
    raise ValueError("PINECONE_API_KEY not found. Check your .env file.")

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY not found. Check your .env file.")

# Use the env var if it exists, otherwise default to "csea-assistant"
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "csea-assistant")

DOCS_FOLDER = "./documents"
FEEDBACK_FILE = "feedback.json"

# ─────────────────────────────────────────────────────────────────────────────
# 3. EMBEDDING MODEL
# ─────────────────────────────────────────────────────────────────────────────
@lru_cache(maxsize=1)
def get_embeddings():
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# ─────────────────────────────────────────────────────────────────────────────
# 4. LLM
# ─────────────────────────────────────────────────────────────────────────────
def get_llm():
    return ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0.0,
        groq_api_key=GROQ_API_KEY
    )

# ─────────────────────────────────────────────────────────────────────────────
# 5. PINECONE STORE
# ─────────────────────────────────────────────────────────────────────────────
def get_vectorstore():
    return PineconeVectorStore.from_existing_index(
        index_name=PINECONE_INDEX_NAME,
        embedding=get_embeddings()
    )

def get_retriever():
    vectorstore = get_vectorstore()
    return vectorstore.as_retriever(search_kwargs={"k": 15})