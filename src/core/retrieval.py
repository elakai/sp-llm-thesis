import time
import random 
import streamlit as st
import concurrent.futures
from typing import List, Dict, Any, Tuple
from langchain_core.documents import Document
from sentence_transformers import CrossEncoder
from rank_bm25 import BM25Okapi
from src.config.settings import get_llm, get_vectorstore
from src.core.router import route_query  
from src.core.decomposition import decompose_query
from src.core.guardrails import verify_answer

# ─────────────────────────────────────────────────────────────────────────────
# Global Reranker (Load once)
# ─────────────────────────────────────────────────────────────────────────────
# Using L-6 for balance between speed and accuracy
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
    """Uses LLM to rewrite the query. Returns original if LLM fails."""
    # SKIP rewrite for very short queries to save time
    if len(query.split()) < 5:
        return query
        
    llm = get_llm()
    prompt = f"Rewrite this query for a university handbook search: {query}"
    try:
        return llm.invoke(prompt).content.strip() or query
    except Exception:
        return query

def hybrid_rerank(query: str, docs: List[Document]) -> List[Document]:
    """Combines BM25 (keyword match) with Semantic Search results."""
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
    """Groups by filename and keeps ALL chunks that belong to the latest version."""
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

def generate_response(query: str, chat_history_list: List[Dict[str, str]] = []):
    """
    Generator function that yields chunks of text for streaming.
    """
    
    # 🚀 STEP 1: SMART ROUTER
    intent = route_query(query)
    
    if intent == "greeting":
        greetings = [
            "Hello! I am AXIsstant. How can I help you with the CSEA Handbook today?",
            "Hi there! I'm ready to answer your questions about university policies.",
            "Greetings! Feel free to ask me about uniforms, grading, or curriculum."
        ]
        response = random.choice(greetings)
        for word in response.split():
            yield word + " "
            time.sleep(0.02)
        return

    if intent == "off_topic":
        msg = "I am designed to answer questions about the CSEA Student Handbook... I cannot help with general topics."
        for word in msg.split():
            yield word + " "
            time.sleep(0.02)
        return

    # 🚀 STEP 2: FAST TRACK DECOMPOSITION
    # If query is short (< 15 words) and simple, SKIP decomposition.
    is_complex = len(query.split()) > 15 or " and " in query or "?" in query.split()[-1] and len(query.split()) > 10
    
    sub_queries = [query]
    if is_complex:
        try:
            sub_queries = decompose_query(query)
            if len(sub_queries) > 1:
                # Log to terminal (hidden from chat UI)
                print(f"\n🔄 **Complex Query Detected:** Splitting into {len(sub_queries)} searches...")
        except:
            sub_queries = [query]

    # 🚀 STEP 3: PARALLEL RETRIEVAL
    vectorstore = get_vectorstore()
    retriever = vectorstore.as_retriever(search_kwargs={"k": 8}) # Reduced K for speed
    all_docs = []

    # Run searches in parallel threads
    with concurrent.futures.ThreadPoolExecutor() as executor:
        results = list(executor.map(retriever.invoke, sub_queries))
    
    for res in results:
        all_docs.extend(res)

    if not all_docs:
        yield "I couldn't find any information about that in the uploaded documents."
        return

    # 🚀 STEP 4: DEDUPLICATION & RERANKING
    unique_docs = {}
    for doc in all_docs:
        key = f"{doc.metadata.get('source')}_{doc.metadata.get('page')}"
        if key not in unique_docs:
            unique_docs[key] = doc
    
    semantic_results = list(unique_docs.values())

    # Filter Latest
    latest_per_source = prefer_latest_per_source(semantic_results)
    
    # Rewrite query for better matching (only if necessary)
    rewritten_query = rewrite_query(query) 

    # Hybrid Rerank
    hybrid_results = hybrid_rerank(rewritten_query, latest_per_source)

    # Cross-Encoder (The Heavy Lifter)
    # OPTIMIZATION: Only rerank top 10 candidates instead of 20
    if hybrid_results:
        try:
            top_candidates = hybrid_results[:10] 
            pairs = [(rewritten_query, doc.page_content) for doc in top_candidates]
            scores = reranker.predict(pairs)
            sorted_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
            
            if scores[sorted_indices[0]] < -8.0:
                yield "I searched the handbook, but couldn't find a relevant answer to your specific question."
                return

            top_reranked = [top_candidates[i] for i in sorted_indices[:6]] # Keep top 6
        except:
            top_reranked = hybrid_results[:5]
    else:
        top_reranked = []

    # 4. Build Context
    context_pieces = []
    total_chars = 0
    limit = 3000 # Reduced limit slightly for speed
    
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
    st.session_state["last_retrieved_context"] = context


# 5. Generate Response (Enhanced Spacing Prompt)
    prompt = f"""You are AXIsstant, the official Academic AI of Ateneo de Naga University. 
Provide a clear, highly-readable response based ONLY on the context.

**STRICT MARKDOWN RULES:**
- Use `###` for headers. You MUST put TWO blank lines before every header.
- Every bullet point `*` MUST be on its own line.
- Put a BLANK LINE between every paragraph.
- Use **Bold** for grades, rules, or key terms.

**Context:**
{context}

**History:**
{history_text}

**Question:** {query}

**Answer:**"""

    llm = get_llm()

    try:
        # A. Generate the FULL answer silently first
        full_answer = llm.invoke(prompt).content
        
        # B. Verify it (Guardrails)
        verified_answer = verify_answer(query, context, full_answer)
        
        # 🚀 C. THE FORMAT-FIXER (The Nuclear Option)
        # We manually force double newlines before headers and bullets 
        # to ensure Streamlit renders them as a clean list.
        formatted_answer = verified_answer.replace("### ", "\n\n### ").replace("* ", "\n\n* ")
        # Ensure citations start on a new line
        formatted_answer = formatted_answer.replace("(Source:", "\n\n(Source:")

        # D. Now Stream the Verified & Formatted Answer
        # We use .split(" ") instead of .split() to preserve those newlines
        for word in formatted_answer.split(" "):
            yield word + " "
            time.sleep(0.01) 
            
    except Exception as e:
        yield f"⚠️ **API Error:** {str(e)}"