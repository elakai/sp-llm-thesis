# src/core/retrieval.py
from datetime import datetime
from typing import List, Dict
from src.config.settings import get_llm, get_vectorstore
from sentence_transformers import CrossEncoder
from rank_bm25 import BM25Okapi
from langchain_core.documents import Document

# Reranker: cross-encoder (free, local, high-quality relevance scoring)
reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

def rewrite_query(query: str) -> str:
    """
    Rewrite query to improve retrieval quality (optional but helpful).
    """
    llm = get_llm()
    prompt = f"""Rewrite this user query to be more detailed and precise for searching a university student handbook or policy documents.
Include relevant keywords like 'Ateneo de Naga University', 'CSEA', 'student handbook', 'dress code', 'typhoon signal', 'pregnancy exemption', 'grading system', 'awards'.
Keep it factual and natural.

Original: {query}

Rewritten:"""
    try:
        rewritten = llm.invoke(prompt).content.strip()
        return rewritten or query
    except Exception:
        return query

def hybrid_rerank(query: str, docs: List[Document]) -> List[Document]:
    """
    Simple hybrid: BM25 keyword scoring + semantic position boost.
    """
    if not docs:
        return []

    tokenized_docs = [doc.page_content.split() for doc in docs]
    bm25 = BM25Okapi(tokenized_docs)
    tokenized_query = query.lower().split()
    bm25_scores = bm25.get_scores(tokenized_query)

    ranked = []
    for i, doc in enumerate(docs):
        score = bm25_scores[i] + (len(docs) - i) * 0.05
        ranked.append((score, doc))

    ranked.sort(reverse=True, key=lambda x: x[0])
    return [doc for _, doc in ranked[:15]]

def prefer_latest_per_source(docs: List[Document]) -> List[Document]:
    """
    Group chunks by source filename (similar/revised documents).
    Keep only the newest upload_timestamp per group.
    This ensures retrieval prefers the most recent uploaded version of similar documents.
    """
    grouped: Dict[str, List[Document]] = {}
    for doc in docs:
        source = doc.metadata.get("source", "unknown")
        grouped.setdefault(source, []).append(doc)

    latest_docs = []
    for source, group in grouped.items():
        # Sort by upload_timestamp descending, keep only the newest
        sorted_group = sorted(
            group,
            key=lambda d: d.metadata.get("upload_timestamp", 0),
            reverse=True
        )
        if sorted_group:
            latest_docs.append(sorted_group[0])  # only the most recent per source

    print(f"After preferring latest per source: {len(latest_docs)} / {len(docs)} chunks kept")
    return latest_docs

def generate_response(query: str) -> str:
    """
    RAG pipeline with improvements:
    - Query rewriting
    - Semantic retrieval (no type filter)
    - Prefer latest version per source (your main request)
    - Hybrid BM25 boost
    - Cross-encoder rerank
    - Final recency sort
    - Tuned prompt with Markdown
    """
    vectorstore = get_vectorstore()

    # Tuned retrieval: 25 candidates
    retriever = vectorstore.as_retriever(search_kwargs={"k": 25})  # no filter

    rewritten_query = rewrite_query(query)
    print(f"Rewritten query: {rewritten_query}")

    semantic_results = retriever.invoke(rewritten_query)
    print(f"Retrieved {len(semantic_results)} semantic chunks")

    # NEW: Prefer most recent upload per source (for similar/revised documents)
    latest_per_source = prefer_latest_per_source(semantic_results)

    # Hybrid boost
    hybrid_results = hybrid_rerank(rewritten_query, latest_per_source)

    # Cross-encoder rerank
    if hybrid_results:
        pairs = [(rewritten_query, doc.page_content) for doc in hybrid_results]
        scores = reranker.predict(pairs)
        sorted_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        top_reranked = [hybrid_results[i] for i in sorted_indices[:10]]
    else:
        top_reranked = []

    # Final sort by upload_timestamp (newest first)
    sorted_results = sorted(
        top_reranked,
        key=lambda d: d.metadata.get("upload_timestamp", 0),
        reverse=True
    )

    context = "\n\n".join([doc.page_content for doc in sorted_results])
    print(f"Final context length: {len(context)} chars")

    # Tuned prompt: less strict, encourages partial answers, strong citation rule
    prompt = f"""You are the official CSEA Information Assistant at Ateneo de Naga University.
Use the provided context to answer as completely and helpfully as possible. Be accurate, professional, and detailed.
If the context has partial or related information, use it and explain.
If absolutely no relevant information, say: "I'm sorry, this information is not available in the current documents."

Cite sources when possible: [Source: filename, upload_date]

**IMPORTANT FORMATTING RULES:**
- Always use proper Markdown formatting in your responses:
  - **Bold** important terms
  - *Italics* for emphasis
  - Use bullet points or numbered lists for multiple items
  - Use headings like ## or ### for sections
  - Use code blocks ``` for examples or quotes
  - Use tables when comparing things (e.g. grading scale)
  - Include inline links if relevant [text](url)

Context:
{context}

Question: {query}
Answer (use Markdown):"""

    # Call LLM with tuned temperature = 0.0 for maximum factuality
    llm = get_llm()  # make sure get_llm() uses temperature=0.0
    response = llm.invoke(prompt).content

    return response