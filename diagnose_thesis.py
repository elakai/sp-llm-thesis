"""Quick diagnostic: what chunks come back for thesis queries?"""
import os, sys
sys.path.insert(0, os.path.dirname(__file__))
from dotenv import load_dotenv
load_dotenv()

from src.config.settings import get_embeddings, get_retriever
from sentence_transformers import CrossEncoder

emb = get_embeddings()
retriever = get_retriever(k=15)
reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

query = "what are some thesis or practicum report in civil engineering program"

print(f"\nQuery: {query}\n")
docs = retriever.invoke(query)

print(f"Retrieved {len(docs)} chunks. Sources:\n")
for i, doc in enumerate(docs):
    src = doc.metadata.get('source', '?')
    cat = doc.metadata.get('category', '?')
    preview = doc.page_content[:100].replace('\n', ' ')
    print(f"  [{i+1}] cat={cat} | {src}")
    print(f"       {preview}...\n")

# Rerank
pairs = [(query, doc.page_content) for doc in docs]
scores = reranker.predict(pairs)
sorted_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)

print(f"\n{'='*70}")
print(f"After CrossEncoder reranking (top 8 sent to LLM):\n")
for rank, idx in enumerate(sorted_idx[:8]):
    src = docs[idx].metadata.get('source', '?')[:50]
    cat = docs[idx].metadata.get('category', '?')
    score = scores[idx]
    preview = docs[idx].page_content[:80].replace('\n', ' ')
    print(f"  {rank+1}. [{score:>7.2f}] cat={cat} | {src}")
    print(f"             {preview}...\n")

# Also: what thesis chunks exist in Pinecone?
print(f"\n{'='*70}")
print("Checking Pinecone for ALL thesis-category chunks...\n")

from pinecone import Pinecone as PC
pc = PC(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index("csea-assistant")

# Query with a thesis-specific vector
thesis_query = "civil engineering thesis manuscript abstract"
vec = emb.embed_query(thesis_query)
results = index.query(vector=vec, top_k=30, include_metadata=True, filter={"category": "thesis"})

print(f"Found {len(results['matches'])} thesis-category vectors:\n")
for m in results['matches']:
    src = m['metadata'].get('source', '?')
    score = m['score']
    print(f"  [{score:.4f}] {src}")
