"""Diagnose what content is actually in Pinecone for curriculum/grading queries."""
import os, sys
from pathlib import Path
from dotenv import load_dotenv

project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))
load_dotenv(project_root / ".env")

# Suppress streamlit warnings
os.environ["STREAMLIT_RUNTIME_EXISTS"] = "false"

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = PineconeVectorStore(
    index_name=os.getenv("PINECONE_INDEX_NAME"),
    embedding=embeddings
)
retriever = vectorstore.as_retriever(search_kwargs={"k": 15})

queries = [
    "2nd year subjects BS Mathematics curriculum",
    "BS Civil Engineering curriculum subjects",
    "grading system Ateneo de Naga University",
    "delayed examination process",
    "curriculum BS CE year 2 semester 1 subjects",
]

for q in queries:
    print(f"\n{'='*80}")
    print(f"QUERY: {q}")
    print(f"{'='*80}")
    docs = retriever.invoke(q)
    print(f"Retrieved {len(docs)} chunks\n")
    for i, doc in enumerate(docs[:5]):  # Top 5 only
        src = doc.metadata.get("source", "?")
        page = doc.metadata.get("page", "?")
        doc_type = doc.metadata.get("type", "?")
        cat = doc.metadata.get("category", "?")
        content_preview = doc.page_content[:300].replace("\n", " | ")
        print(f"  [{i+1}] source={src}  page={page}  type={doc_type}  cat={cat}")
        print(f"      {content_preview}")
        print()
