# src/core/retrieval.py
from datetime import datetime
from typing import List
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
        score = bm25_scores[i] + (len(docs) - i) * 0.05  # boost top semantic results
        ranked.append((score, doc))

    ranked.sort(reverse=True, key=lambda x: x[0])
    return [doc for _, doc in ranked[:15]]  # top 15 after hybrid

def generate_response(query: str) -> str:
    """
    RAG pipeline with improvements:
    - Query rewriting
    - Semantic retrieval + type filter
    - Hybrid BM25 boost
    - Cross-encoder reranking (main improvement)
    - Recency sort
    - Tuned prompt & low temperature
    """
    vectorstore = get_vectorstore()

    # Tuned retrieval: 25 candidates (good balance for reranking)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 25})

    rewritten_query = rewrite_query(query)
    print(f"Rewritten query: {rewritten_query}")

    semantic_results = retriever.invoke(rewritten_query)
    print(f"Retrieved {len(semantic_results)} semantic chunks")

    # Hybrid boost
    hybrid_results = hybrid_rerank(rewritten_query, semantic_results)

    # Cross-encoder reranking (core improvement)
    if hybrid_results:
        pairs = [(rewritten_query, doc.page_content) for doc in hybrid_results]
        scores = reranker.predict(pairs)
        sorted_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        top_reranked = [hybrid_results[i] for i in sorted_indices[:8]]  # Tuned: top 8 after rerank
    else:
        top_reranked = []

    # Sort by upload_timestamp descending (newest first)
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
**IMPORTANT FORMATTING RULES:**
- Always use proper Markdown formatting in your responses:
  - **Bold** important terms
  - *Italics* for emphasis
  - Use bullet points or numbered lists for multiple items
  - Use headings like ## or ### for sections
  - Use code blocks ``` for examples or quotes
  - Use tables when comparing things (e.g. grading scale)
  - Include inline links if relevant [text](url)
- If the context lacks information, say: "I'm sorry, this information is not available in the current documents."
- Cite sources when possible: [Source: filename, upload_date]

Context:
{context}

Question: {query}
Answer (use Markdown):"""

    # Call LLM with tuned temperature = 0.0 for maximum factuality
    llm = get_llm()  # make sure get_llm() uses temperature=0.0
    response = llm.invoke(prompt).content

    return response