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
reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

# ─────────────────────────────────────────────────────────────────────────────
# Helper Functions
# ─────────────────────────────────────────────────────────────────────────────

def format_chat_history(messages: List[Dict[str, str]]) -> str:
    """Converts Streamlit's session state messages into a string."""
    formatted_history = []
    history_to_process = messages[1:] if len(messages) > 1 else []
    
    for msg in history_to_process[-8:]: 
        role = "User" if msg["role"] == "user" else "Assistant"
        # Escape curly braces for f-string safety
        content = msg["content"].replace("{", "{{").replace("}", "}}")
        formatted_history.append(f"{role}: {content}")
    
    return "\n".join(formatted_history) if formatted_history else "No previous context."

def rewrite_query(query: str) -> str:
    """Uses LLM to rewrite the query. Returns original if short or fails."""
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
            # Weighted score: Boost keyword matches (BM25) to handle typos better
            position_score = (len(docs) - i) * 0.05 
            final_score = bm25_scores[i] + position_score
            ranked.append((final_score, doc))

        ranked.sort(reverse=True, key=lambda x: x[0])
        return [doc for _, doc in ranked[:15]]
    except Exception as e:
        print(f"Hybrid rerank failed: {e}")
        return docs[:10]

def prefer_latest_per_source(docs: List[Document]) -> List[Document]:
    """Groups by filename and keeps chunks from the latest version."""
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
    start_time = time.time()
    retrieval_start = time.time()
    
    # 🚀 STEP 1: SMART ROUTER
    intent = route_query(query)
    
    if intent == "greeting":
        greetings = ["Hello! I am AXIsstant. How can I help you today?", "Hi! I'm ready to answer handbook questions."]
        response = random.choice(greetings)
        for word in response.split():
            yield word + " "
            time.sleep(0.02)
        return

    if intent == "off_topic":
        msg = "I am designed for CSEA Student Handbook questions only."
        for word in msg.split():
            yield word + " "
            time.sleep(0.02)
        return

    # 🚀 STEP 2: DECOMPOSITION
    is_complex = len(query.split()) > 15 or " and " in query
    sub_queries = [query]
    if is_complex:
        try:
            sub_queries = decompose_query(query)
        except:
            sub_queries = [query]

    # 🚀 STEP 3: PARALLEL RETRIEVAL
    vectorstore = get_vectorstore()
    # Increased 'k' to 12 to ensure better recall for reranking
    retriever = vectorstore.as_retriever(search_kwargs={"k": 12})
    all_docs = []

    with concurrent.futures.ThreadPoolExecutor() as executor:
        results = list(executor.map(retriever.invoke, sub_queries))
    
    for res in results:
        all_docs.extend(res)

    if not all_docs:
        yield "I couldn't find any information about that in the documents."
        return

    # 🚀 STEP 4: DEDUPLICATION & RERANKING
    # Deduplicate by content to prevent identical chunks from eating up context space
    unique_docs_map = {hash(d.page_content): d for d in all_docs}
    latest_per_source = prefer_latest_per_source(list(unique_docs_map.values()))
    
    rewritten_query = rewrite_query(query) 
    hybrid_results = hybrid_rerank(rewritten_query, latest_per_source)

    if hybrid_results:
        try:
            top_candidates = hybrid_results[:12] 
            # Use original query for reranking to catch raw keyword similarities
            pairs = [(query, doc.page_content) for doc in top_candidates]
            scores = reranker.predict(pairs)
            sorted_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
            
            # 🛡️ RELAXED THRESHOLD: Changed from -8.0 to -15.0 to handle typos
            if scores[sorted_indices[0]] < -15.0:
                yield "I found some documents, but they didn't contain a specific answer to your question. I don't want to guess and give you wrong information."
                return

            top_reranked = [top_candidates[i] for i in sorted_indices[:6]]
        except Exception as e:
            print(f"Reranking failed: {e}")
            top_reranked = hybrid_results[:6]
    else:
        top_reranked = []

    # 🚀 STEP 5: BUILD CONTEXT
    context_pieces = []
    for doc in top_reranked:
        source = doc.metadata.get("source", "Unknown")
        page = doc.metadata.get("page", "?")
        content = doc.page_content.replace("\n", " ") 
        context_pieces.append(f"[[Source: {source} | Page: {page}]]\n{content}")

    context = "\n\n".join(context_pieces)
    history_text = format_chat_history(chat_history_list)
    
    st.session_state["last_retrieved_context"] = context
    retrieval_time = time.time() - retrieval_start

    # 🚀 STEP 6: GENERATE
    gen_start = time.time()
    prompt = f"""You are AXIsstant, the official Academic AI of Ateneo de Naga University. 
Provide a clear, highly-readable response based ONLY on the context. If the information isn't in the context, say you don't know.

**STRICT MARKDOWN RULES:**
1. Use '###' for headers and ALWAYS put a blank line above them.
2. Every bullet point must start on a new line.
3. Use double asterisks **like this** for bolding.

**Context:**
{context}

**History:**
{history_text}

**Question:** {query}

**Answer:**"""

    llm = get_llm()

    try:
        full_answer = llm.invoke(prompt).content
        verified_answer = verify_answer(query, context, full_answer)

        # Post-process for UI readability
        formatted_answer = verified_answer.replace("### ", "\n\n### ").replace("* ", "\n* ")

        gen_time = time.time() - gen_start 
        total_time = time.time() - start_time

        # Update metrics for your logging
        st.session_state["performance_metrics"] = {
            "retrieval_latency": retrieval_time,
            "generation_latency": gen_time,
            "total_latency": total_time
        }
        
        for word in formatted_answer.split(" "):
            yield word + " "
            time.sleep(0.01) 
            
    except Exception as e:
        yield f"⚠️ **API Error:** {str(e)}"