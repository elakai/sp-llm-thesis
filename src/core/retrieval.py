import time
import numpy as np
import streamlit as st
import concurrent.futures
from typing import List, Dict
from langchain_core.documents import Document
from sentence_transformers import CrossEncoder
from rank_bm25 import BM25Okapi
import re
from tenacity import retry, stop_after_attempt, wait_exponential

from src.core.guardrails import verify_answer, validate_query, redact_pii
from src.config.settings import get_generator_llm, get_vectorstore, get_embeddings, get_retriever
from src.core.router import route_query   
from src.core.decomposition import decompose_query
from src.config.constants import (
    RETRIEVAL_K, 
    RERANKER_TOP_K, 
    DECOMPOSE_TRIGGERS, 
    RETRIEVAL_K_MAP,
    LOW_CONFIDENCE_THRESHOLD,
    HIGH_CONFIDENCE_THRESHOLD,
    STREAM_DELAY
)
from src.config.logging_config import logger

# ─────────────────────────────────────────────────────────────────────────────
# GLOBALS & CACHE
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource
def get_reranker() -> CrossEncoder:
    """CrossEncoder model lives in server RAM permanently."""
    return CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

@st.cache_resource
def get_semantic_cache() -> list:
    return []

GLOBAL_CACHE = get_semantic_cache()

# Words that signal the query references prior conversation and needs rewriting
_CONTEXT_TRIGGERS = re.compile(
    r'\b(it|its|they|them|their|this|that|these|those|the same|'
    r'above|previous|earlier|last|mentioned|said|again|also|more)\b',
    re.IGNORECASE
)

# ─────────────────────────────────────────────────────────────────────────────
# 1. CONVERSATIONAL MEMORY & CACHING FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────
def contextualize_query(query: str, chat_history_list: List[Dict[str, str]]) -> str:
    history_to_process = chat_history_list[1:] if len(chat_history_list) > 1 else []
    if not history_to_process: return query
    
    # Skip the LLM call if the query doesn't reference prior conversation
    if not _CONTEXT_TRIGGERS.search(query):
        logger.info(f"Context skip: No pronouns/references detected in '{query}'")
        return query
        
    history_text = format_chat_history(chat_history_list)
    prompt = f"""Given the following chat history and the user's latest question, formulate a standalone question that can be understood without the chat history.
    Do NOT answer the question. Just reformulate it if needed. If it doesn't need reformulating, return it exactly as is.

    Chat History:
    {history_text}

    Latest Question: {query}
    Standalone Question:"""
    
    try:
        llm = get_generator_llm()
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
def generate_response(query: str, chat_history_list: List[Dict[str, str]] = None):
    if chat_history_list is None:
        chat_history_list = []
    start_time = time.time()
    top_score = 0.0
    
    # 🚀 STEP 0: VALIDATION & CONVERSATIONAL MEMORY
    is_valid, clean_query = validate_query(query) # Call from guardrails.py
    if not is_valid:
        yield clean_query
        return
    
    safe_query = redact_pii(clean_query) # Call from guardrails.py
    standalone_query = contextualize_query(safe_query, chat_history_list)
    
    # 🚀 STEP 1: CACHE
    cached_answer = check_semantic_cache(standalone_query)
    if cached_answer:
        for word in cached_answer.split(" "):
            yield word + " "
            time.sleep(0.01)
        return

    retrieval_start = time.time()
    
    # 🚀 STEP 2: SMART ROUTER & METADATA DETECTION
    try:
        intent, program_filters, content_type, category_filter = route_query(standalone_query)
    except Exception as e:
        logger.warning(f"Router fallback triggered: {e}")
        intent, program_filters, content_type, category_filter = "search", None, "all", None

    # Handle greetings and off-topic early
    if intent in ["greeting", "off_topic"]:
        msg = "Hello! I am AXIsstant..." if intent == "greeting" else "I am designed for CSEA questions only."
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

    # 🚀 STEP 4: PARALLEL RETRIEVAL WITH DYNAMIC K
    # Fetch K based on intent from constants.py, default to 5
    dynamic_k = RETRIEVAL_K_MAP.get(intent, 5) 
    search_kwargs = {"k": dynamic_k}
    
    # Initialize metadata filter dictionary
    pinecone_filter = {} 

    # Apply Program/Source filters (e.g., specific curriculum files)
    if program_filters:
        pinecone_filter["source"] = {"$in": program_filters}
    
    # Apply Content Type filters (e.g., tables vs text)
    if content_type == "table":
        pinecone_filter["type"] = {"$eq": "table"}

    # Apply Category/Subfolder filters (e.g., memos, thesis, laboratory)
    if category_filter:
        pinecone_filter["category"] = {"$eq": category_filter}
        logger.info(f"🎯 Category Filter Applied: {category_filter}")

    # Attach filters to search arguments if any exist
    if pinecone_filter:
        search_kwargs["filter"] = pinecone_filter

    # Configure the retriever with dynamic settings
    retriever = get_retriever(k=dynamic_k)
    retriever.search_kwargs = search_kwargs 
    
    all_docs = []

    # ⚡ Execute Retrieval — skip thread pool overhead for single queries
    if len(sub_queries) == 1:
        all_docs = retriever.invoke(sub_queries[0])
    else:
        with concurrent.futures.ThreadPoolExecutor() as executor:
            results = list(executor.map(retriever.invoke, sub_queries))
        for res in results:
            all_docs.extend(res)

    # 🛑 Early exit if no information is found
    if not all_docs:
        logger.warning(f"⚠️ Vector Search returned 0 results for: {standalone_query}")
        yield "I checked the handbook, but I couldn't find any information about that."
        return

    logger.info(f"📂 Retrieval Success: Found {len(all_docs)} raw chunks using K={dynamic_k}")

    # 🚀 STEP 5: DEDUPLICATION & RERANKING
    unique_docs_map = {hash(d.page_content): d for d in all_docs}
    latest_per_source = prefer_latest_per_source(list(unique_docs_map.values()))
    
    hybrid_results = hybrid_rerank(standalone_query, latest_per_source)

    if hybrid_results:
        try:
            pairs = [(standalone_query, doc.page_content) for doc in hybrid_results]
            scores = get_reranker().predict(pairs)
            sorted_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
            
            top_score = float(scores[sorted_indices[0]])
            
            # 🛑 REMOVED the hard `if top_score < -10.0:` block here. 
            # Step 7's Three-Tier logic will handle the confidence check.

            top_reranked = [hybrid_results[i] for i in sorted_indices[:RERANKER_TOP_K]] 
        except Exception as e:
            logger.error(f"Reranking failed: {e}")
            top_reranked = hybrid_results[:RERANKER_TOP_K]
    else:
        top_reranked = []

    # 🐛 TEMPORARY DEBUG LOGGING: Watch this in your terminal!
    logger.info(f"📊 DEBUG | Query: '{standalone_query}'")
    logger.info(f"📊 DEBUG | Docs Retrieved: {len(all_docs)} -> After Rerank: {len(top_reranked)}")
    logger.info(f"📊 DEBUG | Top Score: {top_score:.2f} (Low Cutoff: {LOW_CONFIDENCE_THRESHOLD}, High Cutoff: {HIGH_CONFIDENCE_THRESHOLD})")

    # 🚀 STEP 6: BUILD CONTEXT
    context_pieces = [f"[[Source: {doc.metadata.get('source', 'Unknown')}]]\n{doc.page_content.replace(chr(10), ' ')}" for doc in top_reranked]
    context = "\n\n".join(context_pieces)
    st.session_state["last_retrieved_context"] = context
    retrieval_time = time.time() - retrieval_start

  # 🚀 STEP 7: THREE-TIER CONFIDENCE & GENERATION
    gen_start = time.time()
    
    # Define the Prompt (Instruction-Heavy for formatting)
    prompt = f"""You are AXIsstant, the official Academic AI of Ateneo de Naga University. 
Your goal is to provide accurate answers based ONLY on the context provided, while being approachable and conversational.

### TONE & FORMATTING RULES (YOU MUST FOLLOW THESE):
0. **LANGUAGE**: Always respond in English regardless of what language the user used to ask the question. If the user asks in Filipino, still answer in English.

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

4. **STRICTLY FACTUAL**: Answer using ONLY what is in the context. Never give general academic advice (like "study hard" or "attend classes") as a substitute for missing information. If the context contains thesis abstracts, list the relevant ones. If the context genuinely has nothing related, say: 'The retrieved documents do not contain this information.'

**Context:**
{context}

**Chat History:**
{format_chat_history(chat_history_list)}

**Question:** {standalone_query}

**Answer:**"""

    try:
        # Tier 1: Retrieval is too weak — Exit early to prevent hallucination
        if top_score < LOW_CONFIDENCE_THRESHOLD:
            logger.warning(f"🔇 Low Retrieval Score ({top_score:.2f}). Aborting generation.")
            yield "I checked the handbook but couldn't find enough specific information to answer that confidently. Please consult the CSEA Department Chair."
            return

        # Tier 2 & 3: Retrieval is sufficient — Invoke LLM with Retry Logic
        llm = get_generator_llm()
        draft_response = get_llm_response(llm, prompt).content
        
        if top_score < HIGH_CONFIDENCE_THRESHOLD:
            # Tier 2: Moderate confidence — Trigger Critic to verify against context
            logger.info(f"🔍 Moderate Confidence ({top_score:.2f}). Triggering Critic Persona...")
            final_verified_response = verify_answer(standalone_query, context, draft_response)
        else:
            # Tier 3: High confidence — Trust the draft
            logger.info(f"✨ High Confidence ({top_score:.2f}). Bypassing Critic.")
            final_verified_response = draft_response

        # 🚀 STEP 8: METRICS, CACHE & STREAMING
        
        # We record metrics BEFORE streaming so they are saved even if the user disconnects
        st.session_state["performance_metrics"] = {
            "retrieval_latency": retrieval_time,
            "generation_latency": time.time() - gen_start,
            "total_latency": time.time() - start_time,
            "confidence_score": float(top_score)
        }
        add_to_cache(standalone_query, final_verified_response)

        if not final_verified_response:
            logger.error("final_verified_response is None. Falling back to draft.")
            final_verified_response = draft_response
        
        # Final Streaming Loop with Fallback
        try:
            for word in final_verified_response.split(" "):
                yield word + " "
                time.sleep(STREAM_DELAY) # Optimized delay from constants
        except GeneratorExit:
            # User navigated away; cleanup handled by Python GC
            return
        except Exception as e:
            logger.error(f"Streaming interruption: {e}")
            yield f"\n\n⚠️ *Stream interrupted. Displaying full response:* \n{final_verified_response}"
            
    except Exception as e:
        logger.error(f"❌ Generation Pipeline Failed: {e}")
        yield f"⚠️ **AXIsstant is having trouble connecting to its brain.** (Error: {str(e)})"

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def get_llm_response(llm, prompt):
    """Reliable wrapper for LLM calls with exponential backoff."""
    return llm.invoke(prompt)