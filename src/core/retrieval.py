# src/core/retrieval.py
from typing import List, Dict, Any, Tuple
from langchain_core.documents import Document
from sentence_transformers import CrossEncoder
from rank_bm25 import BM25Okapi
from src.config.settings import get_llm, get_vectorstore

# ─────────────────────────────────────────────────────────────────────────────
# Global Reranker 
# ─────────────────────────────────────────────────────────────────────────────
reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')


# ─────────────────────────────────────────────────────────────────────────────
# Helper Functions
# ─────────────────────────────────────────────────────────────────────────────

def format_chat_history(messages: List[Dict[str, str]]) -> str:
    """
    Converts Streamlit's session state messages into a string for the LLM prompt.
    Keeps the last 3 turns (6 messages) to maintain context without overflowing.
    """
    formatted_history = []
    # Skip the very first greeting message if it exists/is irrelevant
    history_to_process = messages[1:] if len(messages) > 1 else []

    for msg in history_to_process[-6:]:  # Only keep last 6 messages
        role = "User" if msg["role"] == "user" else "Assistant"
        # Sanitize content to prevent prompt injection issues
        content = msg["content"].replace("{", "{{").replace("}", "}}")
        formatted_history.append(f"{role}: {content}")
    
    return "\n".join(formatted_history) if formatted_history else "No previous context."

def rewrite_query(query: str) -> str:
    """
    Uses LLM to rewrite the query for better vector matching.
    """
    llm = get_llm()
    prompt = f"""Rewrite this user query to be more detailed and precise for searching a university student handbook, policy documents, curriculum, thesis manuscripts, laboratory manuals, or organizational charts.
Include relevant keywords like 'Ateneo de Naga University', 'CSEA', 'student handbook', 'dress code', 'typhoon signal', 'pregnancy exemption', 'grading system', 'awards'.
Keep it factual and natural. Do not answer the question, just rewrite it.

Original: {query}

Rewritten:"""
    try:
        rewritten = llm.invoke(prompt).content.strip()
        return rewritten or query
    except Exception:
        return query

def hybrid_rerank(query: str, docs: List[Document]) -> List[Document]:
    """
    Combines BM25 (keyword match) with Semantic Search results.
    """
    if not docs:
        return []

    tokenized_docs = [doc.page_content.split() for doc in docs]
    bm25 = BM25Okapi(tokenized_docs)
    tokenized_query = query.lower().split()
    bm25_scores = bm25.get_scores(tokenized_query)

    ranked = []
    for i, doc in enumerate(docs):
        # Weighted score: 70% BM25 + 30% Semantic Rank (simulated by position boost)
        # Since 'docs' are already sorted by semantic similarity, 'i' is the rank.
        position_score = (len(docs) - i) * 0.05 
        final_score = bm25_scores[i] + position_score
        ranked.append((final_score, doc))

    ranked.sort(reverse=True, key=lambda x: x[0])
    return [doc for _, doc in ranked[:15]]

def prefer_latest_per_source(docs: List[Document]) -> List[Document]:
    """
    Groups by filename and keeps ALL chunks that belong to the latest version 
    (highest upload_timestamp) of that file.
    """
    if not docs:
        return []

    # 1. Group all chunks by their source filename
    grouped: Dict[str, List[Document]] = {}
    for doc in docs:
        source = doc.metadata.get("source", "unknown")
        grouped.setdefault(source, []).append(doc)

    filtered_docs = []
    
    for source, group in grouped.items():
        # 2. Find the newest timestamp among all chunks for this specific file
        # (Handle cases where timestamp might be missing by defaulting to 0)
        latest_timestamp = max(
            (d.metadata.get("upload_timestamp", 0) for d in group), 
            default=0
        )
        
        # 3. Keep ONLY the chunks that match this latest timestamp
        # This removes old versions but keeps all relevant pages of the new version
        current_version_chunks = [
            d for d in group 
            if d.metadata.get("upload_timestamp", 0) == latest_timestamp
        ]
        
        filtered_docs.extend(current_version_chunks)

    print(f"Refined {len(docs)} chunks down to {len(filtered_docs)} (removed old versions)")
    return filtered_docs
# ─────────────────────────────────────────────────────────────────────────────
# Main Retrieval Function
# ─────────────────────────────────────────────────────────────────────────────

def generate_response(query: str, chat_history_list: List[Dict[str, str]] = []) -> str:
    """
    Full RAG Pipeline:
    1. Query Rewriting
    2. Vector Search (Pinecone)
    3. Filtering (Latest per source)
    4. Hybrid Reranking (BM25)
    5. Cross-Encoder Reranking (Accuracy)
    6. LLM Generation
    """
    vectorstore = get_vectorstore()

    # 1. Rewrite Query
    rewritten_query = rewrite_query(query)
    print(f"Original: {query} -> Rewritten: {rewritten_query}")

    # 2. Semantic Search (Broad Phase)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 60})
    semantic_results = retriever.invoke(rewritten_query)

    # 3. Recency Filter
    latest_per_source = prefer_latest_per_source(semantic_results)

    # 4. Hybrid Rerank (Keyword Boost)
    hybrid_results = hybrid_rerank(rewritten_query, latest_per_source)

    # 5. Cross-Encoder Rerank (Precision Phase)
    if hybrid_results:
        pairs = [(rewritten_query, doc.page_content) for doc in hybrid_results]
        scores = reranker.predict(pairs)
        
        # Sort by Cross-Encoder score
        sorted_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        top_reranked = [hybrid_results[i] for i in sorted_indices[:10]]
    else:
        top_reranked = []

    # Final Context Construction
    # Sort final results by date again to ensure context flow isn't jarring? 
    # Or keep by relevance? Let's keep by relevance (top_reranked order).
    context = "\n\n".join([doc.page_content for doc in top_reranked])
    
    # Format History
    history_text = format_chat_history(chat_history_list)

    # 6. LLM Generation
    prompt = f"""You are the official CSEA Information Assistant at Ateneo de Naga University.
Use the provided context to answer as completely and helpfully as possible. Be accurate, professional, and detailed.
If the context has partial or related information, use it and explain.
If absolutely no relevant information, say: "I'm sorry, this information is not available in the current documents."

**Formatting Rules:**
- **Bold** important terms and headings
- Use bullet points for lists
- Cite sources inline if possible, e.g., [Source: Student Handbook 2024]

Conversation History:
{history_text}

Context from Documents:
{context}

Current Question: {query}

Answer (use Markdown):"""

    llm = get_llm()
    response = llm.invoke(prompt).content

    return response