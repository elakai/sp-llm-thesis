# Run this as a one-off script, not in the app
# Save as debug_chunks.py in the project root and run: python debug_chunks.py

import os
from pinecone import Pinecone
from src.config.settings import PINECONE_INDEX_NAME

pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))
index = pc.Index(PINECONE_INDEX_NAME)

# Add to debug.py and run
results = index.query(
    vector=[0.1] * 384,
    top_k=100,
    include_metadata=True,
    filter={"source": {"$in": ["campus directory.md", "campus_directory.md", "directory.md", "main campus directory.md"]}}
)

print(f"Total chunks: {len(results['matches'])}")
for i, match in enumerate(results['matches']):
    meta = match['metadata']
    print(f"\n[{i+1}] source={meta.get('source')} | doc_type={meta.get('doc_type', 'MISSING')} | chunk_index={meta.get('chunk_index')}")
    print(f"     preview: {meta.get('text', meta.get('page_content', ''))[:120]!r}")