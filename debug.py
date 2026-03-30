# Run this as a one-off script, not in the app
# Save as debug_chunks.py in the project root and run: python debug_chunks.py

import os
from pinecone import Pinecone
from src.config.settings import PINECONE_INDEX_NAME

pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))
index = pc.Index(PINECONE_INDEX_NAME)

# Query specifically for organizations
results = index.query(
    vector=[0.1] * 384,
    top_k=100,
    include_metadata=True,
    filter={"source": {"$in": ["organizations.md", "organizations"]}}
)

print(f"Total chunks found for organizations.md: {len(results['matches'])}")
for i, match in enumerate(results['matches']):
    meta = match['metadata']
    text_preview = meta.get('text', meta.get('page_content', ''))[:150]
    print(f"\n[{i+1}] source={meta.get('source')} | doc_type={meta.get('doc_type', 'MISSING')} | chunk_index={meta.get('chunk_index', '?')}")
    print(f"     preview: {text_preview!r}")