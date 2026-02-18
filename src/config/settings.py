import os
from functools import lru_cache
from pathlib import Path
from dotenv import load_dotenv
# This import allows the "Reader" to run inside your app (Unlimited)
from langchain_huggingface import HuggingFaceEmbeddings 
from langchain_groq import ChatGroq
from langchain_pinecone import PineconeVectorStore

# 1. LOAD ENV
env_path = Path(__file__).resolve().parents[2] / ".env"
load_dotenv(env_path)

# 2. VALIDATION
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
if not PINECONE_API_KEY:
    raise ValueError("PINECONE_API_KEY missing. Check .env file.")

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY missing. Check .env file.")

PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "csea-assistant")
DOCS_FOLDER = "./documents"

# 3. THE UNLIMITED READER (Embeddings)
# This runs on your server/computer CPU. It does NOT use the Groq API.
# It is "Local" to the app, meaning it deploys WITH your app.
@lru_cache(maxsize=1)
def get_embeddings():
    # 'all-MiniLM-L6-v2' is the industry standard for fast, efficient RAG.
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# 4. THE CLOUD BRAIN (LLM)
# We use Groq only for answering, not for reading.
def get_llm():
    return ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0.0,
        groq_api_key=GROQ_API_KEY
    )

# 5. DATABASE CONNECTION
def get_vectorstore():
    # Connects your unlimited reader to Pinecone
    return PineconeVectorStore.from_existing_index(
        index_name=PINECONE_INDEX_NAME,
        embedding=get_embeddings()
    )

def get_retriever():
    vectorstore = get_vectorstore()
    return vectorstore.as_retriever(search_kwargs={"k": 5})