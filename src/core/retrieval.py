import time
from typing import List, Dict, Any, Tuple
from langchain_core.documents import Document
from sentence_transformers import CrossEncoder
from rank_bm25 import BM25Okapi
from src.config.settings import get_llm, get_vectorstore

# ─────────────────────────────────────────────────────────────────────────────
# Global Reranker (Load once)
# ─────────────────────────────────────────────────────────────────────────────
reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

# ─────────────────────────────────────────────────────────────────────────────
# Helper Functions
# ─────────────────────────────────────────────────────────────────────────────

def format_chat_history(messages: List[Dict[str, str]]) -> str:
    """
    Converts Streamlit's session state messages into a string for the LLM prompt.
    Truncates history to last 4 turns to avoid token overflow.
    """
    formatted_history = []
    # Skip the greeting; keep last 4 exchanges (8 messages) max
    history_to_process = messages[1:] if len(messages) > 1 else []
    
    for msg in history_to_process[-8:]: 
        role = "User" if msg["role"] == "user" else "Assistant"
        content = msg["content"].replace("{", "{{").replace("}", "}}")
        formatted_history.append(f"{role}: {content}")
    
    return "\n".join(formatted_history) if formatted_history else "No previous context."

def rewrite_query(query: str) -> str:
    """
    Uses LLM to rewrite the query. Returns original if LLM fails.
    """
    llm = get_llm()
    prompt = f"Rewrite this query for a university handbook search: {query}"
    try:
        return llm.invoke(prompt).content.strip() or query
    except Exception:
        return query

def hybrid_rerank(query: str, docs: List[Document]) -> List[Document]:
    """
    Combines BM25 (keyword match) with Semantic Search results.
    """
    if not docs:
        return []

    try:
        tokenized_docs = [doc.page_content.split() for doc in docs]
        bm25 = BM25Okapi(tokenized_docs)
        tokenized_query = query.lower().split()
        bm25_scores = bm25.get_scores(tokenized_query)

        ranked = []
        for i, doc in enumerate(docs):
            # Weighted score: 70% BM25 + 30% Semantic Rank
            position_score = (len(docs) - i) * 0.05 
            final_score = bm25_scores[i] + position_score
            ranked.append((final_score, doc))

        ranked.sort(reverse=True, key=lambda x: x[0])
        return [doc for _, doc in ranked[:15]]
    except Exception as e:
        print(f"Hybrid rerank failed: {e}")
        return docs[:10] # Fallback to raw semantic results

def prefer_latest_per_source(docs: List[Document]) -> List[Document]:
    """
    Groups by filename and keeps ALL chunks that belong to the latest version.
    """
    if not docs: return []

    grouped: Dict[str, List[Document]] = {}
    for doc in docs:
        source = doc.metadata.get("source", "unknown")
        grouped.setdefault(source, []).append(doc)

    filtered_docs = []
    for source, group in grouped.items():
        latest_timestamp = max((d.metadata.get("upload_timestamp", 0) for d in group), default=0)
        current_version_chunks = [d for d in group if d.metadata.get("upload_timestamp", 0) == latest_timestamp]
        filtered_docs.extend(current_version_chunks)

    return filtered_docs

# ─────────────────────────────────────────────────────────────────────────────
# Main Retrieval Function
# ─────────────────────────────────────────────────────────────────────────────

def generate_response(query: str, chat_history_list: List[Dict[str, str]] = []) -> str:
    vectorstore = get_vectorstore()

    # 1. Broad Search with Error Handling
    try:
        rewritten_query = rewrite_query(query)
        retriever = vectorstore.as_retriever(search_kwargs={"k": 40})
        semantic_results = retriever.invoke(rewritten_query)
    except Exception as e:
        return f"⚠️ **Connection Error:** I couldn't search the handbook database. Please try again in a moment.\n\n*(Error: {str(e)})*"

    # 🛡️ EDGE CASE: NO RESULTS FOUND
    if not semantic_results:
        return "I couldn't find any information about that in the uploaded documents. Please try rephrasing your question or ask about a different topic."

    # 2. Filter & Hybrid Rerank
    latest_per_source = prefer_latest_per_source(semantic_results)
    hybrid_results = hybrid_rerank(rewritten_query, latest_per_source)

    # 3. Cross-Encoder Rerank
    if hybrid_results:
        try:
            top_candidates = hybrid_results[:20]
            pairs = [(rewritten_query, doc.page_content) for doc in top_candidates]
            scores = reranker.predict(pairs)
            
            # Sort by score
            sorted_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
            
            # 🛡️ EDGE CASE: LOW RELEVANCE FILTER
            # If the best match has a negative score (common with CrossEncoders for bad matches), stop.
            # Adjust -5.0 based on your model; typically < -8 means completely irrelevant.
            if scores[sorted_indices[0]] < -8.0:
                return "I searched the handbook, but couldn't find a relevant answer to your specific question."

            top_reranked = [top_candidates[i] for i in sorted_indices[:8]] # Keep top 8 strictly
        except Exception:
            top_reranked = hybrid_results[:5] # Fallback if reranker fails
    else:
        top_reranked = []

    # 4. Build Context (Truncate to ~3000 chars to be safe)
    context_pieces = []
    total_chars = 0
    limit = 3500 
    
    for doc in top_reranked:
        source = doc.metadata.get("source", "Unknown Document")
        page = doc.metadata.get("page", "?")
        content = doc.page_content.replace("\n", " ") 
        
        entry = f"[[Source: {source} | Page: {page}]]\n{content}"
        
        if total_chars + len(entry) > limit:
            break # Stop adding if we hit the limit
            
        context_pieces.append(entry)
        total_chars += len(entry)

    context = "\n\n".join(context_pieces)
    history_text = format_chat_history(chat_history_list)

    # 5. Generate with Retry Logic
    prompt = f"""You are the official CSEA Information Assistant at Ateneo de Naga University.
Answer based ONLY on the provided context. If the answer isn't there, say so.

**Citation Rules:**
- Use the provided source tags [[Source: ... | Page: ...]] at the end of sentences.
- Do NOT make up page numbers.

Conversation History:
{history_text}

Context:
{context}

Question: {query}

Answer (in Markdown):"""

    llm = get_llm()
    
    # Simple Retry Logic for API Flukes
    max_retries = 2
    for attempt in range(max_retries):
        try:
            response = llm.invoke(prompt).content
            return response
        except Exception as e:
            if attempt == max_retries - 1:
                return f"⚠️ **API Error:** I'm having trouble connecting to the brain module right now. Please try again. ({str(e)})"
            time.sleep(1) # Wait 1s before retry