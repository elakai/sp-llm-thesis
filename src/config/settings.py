import os
import streamlit as st
from pathlib import Path
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_pinecone import PineconeVectorStore

# 1. LOAD ENV (Single source of truth for the entire app)
env_path = Path(__file__).resolve().parents[2] / ".env"
load_dotenv(env_path)

# Support both local development and Streamlit Cloud deployment
if os.getenv("STREAMLIT_RUNTIME_EXISTS"):
    # Streamlit Cloud
    TESSERACT_CMD = os.getenv("TESSERACT_CMD", "/usr/bin/tesseract")
else:
    # Local Windows development
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
@st.cache_resource
def get_embeddings() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

@st.cache_resource
def get_generator_llm() -> ChatGroq:
    """LLM for main answer generation. Lives in server RAM permanently."""
    return ChatGroq(model=GROQ_MODEL, temperature=0.1, groq_api_key=GROQ_API_KEY)

@st.cache_resource
def get_critic_llm() -> ChatGroq:
    """LLM for answer verification (Critic persona). Zero temperature for deterministic checking."""
    return ChatGroq(model=GROQ_MODEL, temperature=0.0, groq_api_key=GROQ_API_KEY)

# Cached with 5-minute TTL: connection stays alive across queries but refreshes
# automatically to prevent Pinecone idle timeouts.
@st.cache_resource(ttl=300)
def get_vectorstore() -> PineconeVectorStore:
    return PineconeVectorStore.from_existing_index(
        index_name=PINECONE_INDEX_NAME,
        embedding=get_embeddings()
    )

def get_retriever(k: int = 5):
    """Centralized retriever call to prevent mismatched 'k' values."""
    return get_vectorstore().as_retriever(search_kwargs={"k": k})