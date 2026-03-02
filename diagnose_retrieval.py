"""
Diagnostic script: traces the full retrieval pipeline for an org-structure query.
Run from the project root with: python diagnose_retrieval.py
"""
import os, sys
sys.path.insert(0, os.path.dirname(__file__))

from dotenv import load_dotenv
load_dotenv()

# ── Step 1: Check if the document exists in Pinecone ────────────────────────
print("\n" + "="*70)
print("STEP 1 — Checking Pinecone for org-structure vectors")
print("="*70)

from pinecone import Pinecone as PineconeClient
from src.config.settings import PINECONE_INDEX_NAME, get_embeddings

pc = PineconeClient(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index(PINECONE_INDEX_NAME)
stats = index.describe_index_stats()
print(f"Total vectors in Pinecone: {stats.total_vector_count}")

# Search for org-structure content using a direct Pinecone query
emb_model = get_embeddings()
test_query = "who is the dean of CSEA"
query_vector = emb_model.embed_query(test_query)

print(f"\nQuerying Pinecone with: '{test_query}'")
print(f"Embedding dimension: {len(query_vector)}")

results = index.query(vector=query_vector, top_k=20, include_metadata=True)

print(f"\nPinecone returned {len(results['matches'])} matches:\n")

org_keywords = ["dean", "chairperson", "faculty", "department", "soriano", "ojeda", "casimiro", "organizational"]
found_org = False

for i, match in enumerate(results['matches']):
    score = match['score']
    source = match['metadata'].get('source', 'unknown')
    text_preview = match['metadata'].get('text', '')[:200] if 'text' in match['metadata'] else '[no text in metadata]'
    
    # Check if this chunk looks like org structure content
    is_org = any(kw in source.lower() or kw in text_preview.lower() for kw in org_keywords)
    if is_org:
        found_org = True
    
    marker = " <<<< ORG STRUCTURE" if is_org else ""
    print(f"  [{i+1}] Score: {score:.4f} | Source: {source}{marker}")
    if i < 5:  # Show preview for top 5
        print(f"       Preview: {text_preview[:150]}...")
    print()

if not found_org:
    print("⚠️  NO org-structure chunks found in top 20 results!")
    print("    This means either:")
    print("    1. The DOCX was never successfully ingested into Pinecone")
    print("    2. The embedding similarity is too low to surface it")
    print()
    
    # Let's also check the manifest
    print("Checking Supabase manifest for the file...")
    try:
        from src.core.feedback import supabase
        resp = supabase.table("manifest").select("*").execute()
        print(f"  Manifest has {len(resp.data)} entries:")
        for row in resp.data:
            print(f"    - {row['filename']} ({row['chunks']} chunks, status: {row['status']})")
    except Exception as e:
        print(f"  Could not read manifest: {e}")


# ── Step 2: Simulate the retrieval pipeline ─────────────────────────────────
print("\n" + "="*70)
print("STEP 2 — Simulating retrieval pipeline (LangChain retriever)")
print("="*70)

from src.config.settings import get_retriever
retriever = get_retriever(k=15)

queries_to_test = [
    "who is the dean of CSEA",
    "list the ECE department faculty",
    "who is the chairperson of civil engineering",
    "organizational structure of CSEA",
]

for q in queries_to_test:
    docs = retriever.invoke(q)
    print(f"\nQuery: '{q}'")
    print(f"  Retrieved {len(docs)} chunks")
    if docs:
        for j, doc in enumerate(docs[:3]):
            src = doc.metadata.get('source', '?')
            preview = doc.page_content[:120].replace('\n', ' ')
            print(f"    [{j+1}] {src}: {preview}...")
    else:
        print("    ⚠️ ZERO chunks returned!")


# ── Step 3: CrossEncoder reranking scores ────────────────────────────────────
print("\n" + "="*70)
print("STEP 3 — CrossEncoder reranking scores")
print("="*70)

from sentence_transformers import CrossEncoder
reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

test_q = "who is the dean of CSEA"
docs = retriever.invoke(test_q)

if docs:
    pairs = [(test_q, doc.page_content) for doc in docs]
    scores = reranker.predict(pairs)
    
    sorted_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    
    print(f"\nQuery: '{test_q}'")
    print(f"{'Rank':<6} {'Score':<10} {'Source':<40} {'Preview'}")
    print("-" * 100)
    
    for rank, idx in enumerate(sorted_indices):
        score = scores[idx]
        src = docs[idx].metadata.get('source', '?')[:38]
        preview = docs[idx].page_content[:60].replace('\n', ' ')
        
        # Flag confidence tier
        if score < -13.0:
            tier = "ABORT"
        elif score < -5.0:
            tier = "CRITIC"
        else:
            tier = "TRUST"
        
        print(f"  {rank+1:<4} {score:<10.4f} {src:<40} {preview}...  [{tier}]")
    
    top_score = scores[sorted_indices[0]]
    print(f"\n  ★ Top score: {top_score:.4f}")
    print(f"  ★ LOW_CONFIDENCE_THRESHOLD:  -13.0  (below = abort, no LLM call)")
    print(f"  ★ HIGH_CONFIDENCE_THRESHOLD: -5.0   (below = critic check)")
    
    if top_score < -13.0:
        print(f"\n  ❌ DIAGNOSIS: Top score ({top_score:.2f}) is BELOW -13.0 → system ABORTS before calling LLM!")
        print(f"     This is why the LLM 'doesn't read' the content — it never sees it.")
    elif top_score < -5.0:
        print(f"\n  ⚠️  DIAGNOSIS: Top score ({top_score:.2f}) triggers CRITIC mode.")
        print(f"     The Critic might be rejecting the answer.")
    else:
        print(f"\n  ✅ Top score ({top_score:.2f}) is in TRUST range. Content should reach the LLM.")
else:
    print(f"\n  ❌ No documents retrieved for '{test_q}' — nothing to rerank!")


print("\n" + "="*70)
print("DIAGNOSIS COMPLETE")
print("="*70)
