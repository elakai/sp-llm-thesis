#retrieval.py
import time
from typing import List, Dict, Any, Tuple
from langchain_core.documents import Document
from sentence_transformers import CrossEncoder
from rank_bm25 import BM25Okapi
from src.config.settings import get_llm, get_vectorstore
from src.core.router import route_query  
import random 
from src.core.decomposition import decompose_query

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

def generate_response(query: str, chat_history_list: List[Dict[str, str]] = []): # Removed -> str type hint
    """
    Generator function that yields chunks of text for streaming.
    """
    
    # 🚀 STEP 1: SMART ROUTER
    intent = route_query(query)
    
    # PATH A: GREETING (Simulate streaming for consistency)
    if intent == "greeting":
        greetings = [
            "Hello! I am AXIsstant. How can I help you with the CSEA Handbook today?",
            "Hi there! I'm ready to answer your questions about university policies.",
            "Greetings! Feel free to ask me about uniforms, grading, or curriculum."
        ]
        response = random.choice(greetings)
        # Yield words one by one to simulate typing
        for word in response.split():
            yield word + " "
            time.sleep(0.05)
        return

    # PATH B: OFF-TOPIC
    if intent == "off_topic":
        msg = "I am designed to answer questions about the CSEA Student Handbook... I cannot help with general topics."
        for word in msg.split():
            yield word + " "
            time.sleep(0.05)
        return

    # PATH C: SEARCH (The Agentic Pipeline)
    vectorstore = get_vectorstore()
    retriever = vectorstore.as_retriever(search_kwargs={"k": 10}) # Reduce k since we might run multiple searches

    # 🚀 STEP 2: DECOMPOSE (The New Logic)
    # We ask the "Brain" if this needs to be split
    sub_queries = decompose_query(query)
    
    # If it actually decomposed (more than 1 query), let the user know!
    if len(sub_queries) > 1:
        yield f"🔄 **Complex Query Detected:** I'm splitting this into {len(sub_queries)} searches...\n\n"
        for i, sub_q in enumerate(sub_queries):
            yield f"* 🔎 Searching: *'{sub_q}'*...\n"
            time.sleep(0.1)
    
    # 🚀 STEP 3: MULTI-STEP RETRIEVAL
    all_docs = []
    
    for sub_q in sub_queries:
        try:
            # We skip 'rewrite_query' here because decompose usually simplifies it enough
            docs = retriever.invoke(sub_q)
            all_docs.extend(docs)
        except Exception:
            continue

    # 🛡️ EDGE CASE: NO RESULTS
    if not all_docs:
        yield "I couldn't find any information about that in the uploaded documents."
        return

    # 🚀 STEP 4: DEDUPLICATION
    # We might find the same document twice. Let's remove duplicates based on page content.
    unique_docs = {}
    for doc in all_docs:
        # Use page number + source as a unique key
        key = f"{doc.metadata.get('source')}_{doc.metadata.get('page')}"
        if key not in unique_docs:
            unique_docs[key] = doc
    
    semantic_results = list(unique_docs.values())

    # 2. Filter & Rerank (Keep your existing logic here!)
    latest_per_source = prefer_latest_per_source(semantic_results)
    hybrid_results = hybrid_rerank(rewritten_query, latest_per_source)

    # 3. Cross-Encoder (Keep your existing logic here!)
    if hybrid_results:
        try:
            top_candidates = hybrid_results[:20]
            pairs = [(rewritten_query, doc.page_content) for doc in top_candidates]
            scores = reranker.predict(pairs)
            sorted_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
            
            if scores[sorted_indices[0]] < -8.0:
                yield "I searched the handbook, but couldn't find a relevant answer to your specific question."
                return

            top_reranked = [top_candidates[i] for i in sorted_indices[:8]]
        except:
            top_reranked = hybrid_results[:5]
    else:
        top_reranked = []

    # 4. Build Context
    context_pieces = []
    total_chars = 0
    limit = 3500 
    
    for doc in top_reranked:
        source = doc.metadata.get("source", "Unknown")
        page_raw = doc.metadata.get("page", "?")
        try:
            page = int(float(page_raw))
        except:
            page = page_raw

        content = doc.page_content.replace("\n", " ") 
        entry = f"[[Source: {source} | Page: {page}]]\n{content}"
        
        if total_chars + len(entry) > limit:
            break 
        context_pieces.append(entry)
        total_chars += len(entry)

    context = "\n\n".join(context_pieces)
    history_text = format_chat_history(chat_history_list)

    # 5. Generate Response (STREAMING MODE)
    prompt = f"""You are AXIsstant, the official Academic AI...
(Keep your same Prompt text from before here)
Context: {context}
History: {history_text}
Question: {query}
Answer:"""

    llm = get_llm()
    
    # 🚀 KEY CHANGE: Use .stream() instead of .invoke()
    try:
        for chunk in llm.stream(prompt):
            yield chunk.content
    except Exception as e:
        yield f"⚠️ **API Error:** {str(e)}"