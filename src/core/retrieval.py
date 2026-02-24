import time
import random 
import numpy as np
import streamlit as st
import concurrent.futures
from typing import List, Dict
from langchain_core.documents import Document
from sentence_transformers import CrossEncoder
from rank_bm25 import BM25Okapi

from src.core.guardrails import verify_answer
from src.config.settings import get_llm, get_vectorstore, get_embeddings, get_retriever
from src.core.router import route_query   
from src.core.decomposition import decompose_query
from src.config.constants import RETRIEVAL_K, RERANKER_TOP_K, DECOMPOSE_TRIGGERS, CONFIDENCE_THRESHOLD
from src.config.logging_config import logger

# ─────────────────────────────────────────────────────────────────────────────
# GLOBALS & CACHE
# ─────────────────────────────────────────────────────────────────────────────
reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

@st.cache_resource
def get_semantic_cache() -> list:
    return []

GLOBAL_CACHE = get_semantic_cache()

# ─────────────────────────────────────────────────────────────────────────────
# 1. CONVERSATIONAL MEMORY & CACHING FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────
def contextualize_query(query: str, chat_history_list: List[Dict[str, str]]) -> str:
    history_to_process = chat_history_list[1:] if len(chat_history_list) > 1 else []
    if not history_to_process: return query
        
    history_text = format_chat_history(chat_history_list)
    prompt = f"""Given the following chat history and the user's latest question, formulate a standalone question that can be understood without the chat history.
    Do NOT answer the question. Just reformulate it if needed. If it doesn't need reformulating, return it exactly as is.

    Chat History:
    {history_text}

    Latest Question: {query}
    Standalone Question:"""
    
    try:
        llm = get_llm()
        standalone_query = llm.invoke(prompt).content.strip()
        logger.info(f"Memory Rewrite: '{query}' -> '{standalone_query}'")
        return standalone_query
    except Exception as e:
        logger.warning(f"Contextualize Error: {e}")
        return query

def check_semantic_cache(query: str, threshold: float = 0.95) -> str:
    if not GLOBAL_CACHE: return None
    try:
        emb_model = get_embeddings()
        query_emb = np.array(emb_model.embed_query(query))
        for item in GLOBAL_CACHE:
            cached_emb = item["embedding"]
            sim = np.dot(query_emb, cached_emb) / (np.linalg.norm(query_emb) * np.linalg.norm(cached_emb))
            if sim >= threshold:
                logger.info(f"Semantic Cache HIT! (Similarity: {sim:.2f})")
                return item["response"]
    except Exception as e:
        logger.error(f"Cache Error: {e}")
    return None

def add_to_cache(query: str, response: str):
    try:
        emb_model = get_embeddings()
        query_emb = np.array(emb_model.embed_query(query))
        GLOBAL_CACHE.append({"embedding": query_emb, "response": response})
        if len(GLOBAL_CACHE) > 50: GLOBAL_CACHE.pop(0)
    except:
        pass

def invalidate_cache():
    """Wipes the semantic cache object entirely."""
    GLOBAL_CACHE.clear() # Mutates the existing list object
    logger.info("🧹 Semantic cache invalidated. AI will now pull fresh data from Pinecone.")

# ─────────────────────────────────────────────────────────────────────────────
# 2. HELPER FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────
def format_chat_history(messages: List[Dict[str, str]]) -> str:
    formatted_history = []
    history_to_process = messages[1:] if len(messages) > 1 else []
    for msg in history_to_process[-6:]: 
        role = "User" if msg["role"] == "user" else "Assistant"
        content = msg["content"].replace("{", "{{").replace("}", "}}")
        formatted_history.append(f"{role}: {content}")
    return "\n".join(formatted_history) if formatted_history else "No previous context."

def rewrite_query(query: str) -> str:
    if len(query.split()) < 4: return query
    try:
        llm = get_llm()
        prompt = f"Extract the core keywords for a vector search from this student question: {query}"
        return llm.invoke(prompt).content.strip() or query
    except Exception:
        return query

def hybrid_rerank(query: str, docs: List[Document]) -> List[Document]:
    if not docs: return []
    try:
        tokenized_docs = [doc.page_content.split() for doc in docs]
        bm25 = BM25Okapi(tokenized_docs)
        tokenized_query = query.lower().split()
        bm25_scores = bm25.get_scores(tokenized_query)

        ranked = []
        for i, doc in enumerate(docs):
            position_score = (len(docs) - i) * 0.05 
            final_score = bm25_scores[i] + position_score
            ranked.append((final_score, doc))

        ranked.sort(reverse=True, key=lambda x: x[0])
        return [doc for _, doc in ranked[:RETRIEVAL_K]]
    except Exception as e:
        logger.warning(f"Hybrid rerank failed: {e}")
        return docs[:RETRIEVAL_K]

def prefer_latest_per_source(docs: List[Document]) -> List[Document]:
    if not docs: return []
    grouped: Dict[str, List[Document]] = {}
    for doc in docs:
        source = doc.metadata.get("source", "unknown")
        grouped.setdefault(source, []).append(doc)

    filtered_docs = []
    for source, group in grouped.items():
        latest_timestamp = max((d.metadata.get("uploaded_at", 0) for d in group), default=0)
        current_version_chunks = [d for d in group if d.metadata.get("uploaded_at", 0) == latest_timestamp]
        filtered_docs.extend(current_version_chunks)
    return filtered_docs

# ─────────────────────────────────────────────────────────────────────────────
# 3. MAIN GENERATOR PIPELINE
# ─────────────────────────────────────────────────────────────────────────────
def generate_response(query: str, chat_history_list: List[Dict[str, str]] = []):
    start_time = time.time()
    
    # 🚀 STEP 0: CONVERSATIONAL MEMORY
    standalone_query = contextualize_query(query, chat_history_list)
    
    # 🚀 STEP 1: CACHE
    cached_answer = check_semantic_cache(standalone_query)
    if cached_answer:
        for word in cached_answer.split(" "):
            yield word + " "
            time.sleep(0.01)
        st.session_state["performance_metrics"] = {
            "retrieval_latency": 0.0,
            "generation_latency": time.time() - start_time,
            "total_latency": time.time() - start_time
        }
        return

    retrieval_start = time.time()
    
    # 🚀 STEP 2: SMART ROUTER & METADATA DETECTION
    try:
        # UNPACKING 4 VALUES: intent, program_filters, content_type, category_filter
        intent, program_filters, content_type, category_filter = route_query(standalone_query)
    except Exception as e:
        logger.warning(f"Router fallback triggered: {e}")
        intent, program_filters, content_type, category_filter = "search", None, "all", None
        
    if intent == "greeting":
        greetings = ["Hello! I am AXIsstant. How can I help you with the CSEA Handbook?", "Hi! I'm ready to answer your questions."]
        for word in random.choice(greetings).split():
            yield word + " "
            time.sleep(0.02)
        return

    if intent == "off_topic":
        msg = "I am designed for CSEA Student Handbook questions only."
        for word in msg.split():
            yield word + " "
            time.sleep(0.02)
        return

    # 🚀 STEP 3: DECOMPOSITION 
    is_complex = any(trigger in standalone_query.lower() for trigger in DECOMPOSE_TRIGGERS)
    sub_queries = [standalone_query]
    if is_complex:
        try:
            sub_queries = decompose_query(standalone_query)
        except:
            pass

    # 🚀 STEP 4: PARALLEL RETRIEVAL WITH FILTERS
    search_kwargs = {"k": RETRIEVAL_K}
    pinecone_filter = {}
    
    # Existing program/source filters
    if program_filters:
        pinecone_filter["source"] = {"$in": program_filters}
    
    # Existing content type (table vs text) filters
    if content_type == "table":
        pinecone_filter["type"] = {"$eq": "table"}

    # --- NEW: Category/Subfolder filter ---
    if category_filter:
        pinecone_filter["category"] = {"$eq": category_filter}
        logger.info(f"🎯 Category Filter Applied: {category_filter}")

    if pinecone_filter:
        search_kwargs["filter"] = pinecone_filter

    retriever = get_retriever(k=RETRIEVAL_K)
    retriever.search_kwargs = search_kwargs 
    all_docs = []

    with concurrent.futures.ThreadPoolExecutor() as executor:
        results = list(executor.map(retriever.invoke, sub_queries))
    for res in results: all_docs.extend(res)

    if not all_docs:
        yield "I checked the handbook, but I couldn't find any information about that."
        return

    # 🚀 STEP 5: DEDUPLICATION & RERANKING
    unique_docs_map = {hash(d.page_content): d for d in all_docs}
    latest_per_source = prefer_latest_per_source(list(unique_docs_map.values()))
    
    rewritten_query = rewrite_query(standalone_query) 
    hybrid_results = hybrid_rerank(rewritten_query, latest_per_source)

    if hybrid_results:
        try:
            pairs = [(standalone_query, doc.page_content) for doc in hybrid_results]
            scores = reranker.predict(pairs)
            sorted_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
            
            if scores[sorted_indices[0]] < -10.0:
                yield "I found some documents, but they didn't seem relevant to your specific question."
                return

            top_reranked = [hybrid_results[i] for i in sorted_indices[:RERANKER_TOP_K]] 
        except Exception as e:
            logger.error(f"Reranking failed: {e}")
            top_reranked = hybrid_results[:RERANKER_TOP_K]
    else:
        top_reranked = []

    # 🚀 STEP 6: BUILD CONTEXT
    context_pieces = [f"[[Source: {doc.metadata.get('source', 'Unknown')}]]\n{doc.page_content.replace(chr(10), ' ')}" for doc in top_reranked]
    context = "\n\n".join(context_pieces)
    st.session_state["last_retrieved_context"] = context
    retrieval_time = time.time() - retrieval_start

    # 🚀 STEP 7: GENERATE RESPONSE
    gen_start = time.time()
    prompt = f"""You are AXIsstant, the official Academic AI of Ateneo de Naga University. 
Your goal is to provide accurate answers based ONLY on the context provided, while being approachable and conversational.

### TONE & FORMATTING RULES (YOU MUST FOLLOW THESE):
1. **BE APPROACHABLE & HELPFUL**:
   - Always provide a brief, friendly explanation or introduction before presenting raw data. 
   - Speak naturally to the student. Be supportive, but professional.
   - Do NOT just output a table with no context. 

2. **USE TABLES FOR DATA**: 
   - When presenting a **Curriculum**, **Grading System**, **Schedule**, or **Faculty List**, you MUST format the core data as a Markdown Table.
   - Example flow: "Here is the grading system used by the university..." followed by the table.

3. **CLEAN UP LISTS**:
   - Use standard Markdown bullets (`- Name`).
   - If listing people, Bold their names: `- **Dr. John Doe** - Dean`

4. **STRICTLY FACTUAL**:
   - If the specific semester, year, or data is missing from the context, state clearly: "I have the handbook, but I cannot find that specific information in my records."
   - Do not hallucinate courses, grades, or rules.

**Context:**
{context}

**Chat History:**
{format_chat_history(chat_history_list)}

**Question:** {standalone_query}

**Answer:**"""

    llm = get_llm()

    try:
        draft_response = llm.invoke(prompt).content
        top_score = scores[sorted_indices[0]] if 'scores' in locals() and len(scores) > 0 else 0
        
        # CONDITIONAL CRITIC
        if top_score < CONFIDENCE_THRESHOLD:
            logger.info(f"Reranker score low ({top_score:.2f}). Critic Persona analyzing draft...")
            final_verified_response = verify_answer(standalone_query, context, draft_response)
        else:
            logger.info(f"High confidence ({top_score:.2f}). Bypassing Critic.")
            final_verified_response = draft_response

        for word in final_verified_response.split(" "):
            yield word + " "
            time.sleep(0.01) 

        st.session_state["performance_metrics"] = {
            "retrieval_latency": retrieval_time,
            "generation_latency": time.time() - gen_start,
            "total_latency": time.time() - start_time
        }
        add_to_cache(standalone_query, final_verified_response)
            
    except Exception as e:
        logger.error(f"API Error in Generation: {e}")
        yield f"⚠️ **API Error:** {str(e)}"