import time
import random 
import numpy as np
import streamlit as st
import concurrent.futures
from typing import List, Dict, Any, Tuple
from langchain_core.documents import Document
from sentence_transformers import CrossEncoder
from rank_bm25 import BM25Okapi
from src.config.settings import get_llm, get_vectorstore, get_embeddings
from src.core.router import route_query   
from src.core.decomposition import decompose_query

# ─────────────────────────────────────────────────────────────────────────────
# 0. GLOBALS: RERANKER & CACHE
# ─────────────────────────────────────────────────────────────────────────────
# We use a distinct model for reranking to ensure high accuracy
reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

# Global In-Memory Semantic Cache
@st.cache_resource
def get_semantic_cache() -> list:
    return []

GLOBAL_CACHE = get_semantic_cache()

# ─────────────────────────────────────────────────────────────────────────────
# 1. NEW: CONVERSATIONAL MEMORY & CACHING FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────
def contextualize_query(query: str, chat_history_list: List[Dict[str, str]]) -> str:
    """
    Rewrites the user's query into a standalone query using the chat history.
    Example: "Who teaches it?" -> "Who teaches CENG424?"
    """
    history_to_process = chat_history_list[1:] if len(chat_history_list) > 1 else []
    if not history_to_process:
        return query
        
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
        print(f"🧠 Memory Rewrite: '{query}' ➔ '{standalone_query}'")
        return standalone_query
    except Exception as e:
        print(f"⚠️ Contextualize Error: {e}")
        return query

def check_semantic_cache(query: str, threshold: float = 0.95) -> str:
    """Checks if a highly similar query was answered recently."""
    if not GLOBAL_CACHE: return None
    
    try:
        emb_model = get_embeddings()
        query_emb = np.array(emb_model.embed_query(query))
        
        for item in GLOBAL_CACHE:
            cached_emb = item["embedding"]
            sim = np.dot(query_emb, cached_emb) / (np.linalg.norm(query_emb) * np.linalg.norm(cached_emb))
            if sim >= threshold:
                print(f"⚡ Semantic Cache HIT! (Similarity: {sim:.2f})")
                return item["response"]
    except Exception as e:
        print(f"Cache Error: {e}")
    return None

def add_to_cache(query: str, response: str):
    """Saves the query and generated response to the local cache."""
    try:
        emb_model = get_embeddings()
        query_emb = np.array(emb_model.embed_query(query))
        GLOBAL_CACHE.append({
            "embedding": query_emb, 
            "response": response
        })
        if len(GLOBAL_CACHE) > 50:
            GLOBAL_CACHE.pop(0)
    except:
        pass

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
        return [doc for _, doc in ranked[:15]]
    except Exception as e:
        print(f"Hybrid rerank failed: {e}")
        return docs[:10]

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
    
    # 🚀 STEP 0: CONVERSATIONAL MEMORY (Query Rewrite)
    standalone_query = contextualize_query(query, chat_history_list)
    
    # 🚀 STEP 1: CHECK SEMANTIC CACHE
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
        intent, program_filters = route_query(standalone_query)
    except:
        intent, program_filters = "search", None 

    if intent == "greeting":
        greetings = ["Hello! I am AXIsstant. How can I help you with the CSEA Handbook?", "Hi! I'm ready to answer your questions."]
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

    # 🚀 STEP 3: DECOMPOSITION 
    is_complex = len(standalone_query.split()) > 15 or " and " in standalone_query
    sub_queries = [standalone_query]
    if is_complex:
        try:
            sub_queries = decompose_query(standalone_query)
        except:
            pass

    # 🚀 STEP 4: PARALLEL RETRIEVAL WITH FILTERS
    vectorstore = get_vectorstore()
    
    search_kwargs = {"k": 15}
    if program_filters:
        search_kwargs["filter"] = {"source": {"$in": program_filters}}
        print(f"🎯 Pinecone Filter Applied: {program_filters}")

    retriever = vectorstore.as_retriever(search_kwargs=search_kwargs)
    all_docs = []

    with concurrent.futures.ThreadPoolExecutor() as executor:
        results = list(executor.map(retriever.invoke, sub_queries))
    
    for res in results:
        all_docs.extend(res)

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
            top_candidates = hybrid_results[:10] 
            pairs = [(standalone_query, doc.page_content) for doc in top_candidates]
            scores = reranker.predict(pairs)
            sorted_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
            
            if scores[sorted_indices[0]] < -10.0:
                yield "I found some documents, but they didn't seem relevant to your specific question."
                return

            top_reranked = [top_candidates[i] for i in sorted_indices[:6]] 
        except Exception as e:
            print(f"Reranking failed: {e}")
            top_reranked = hybrid_results[:6]
    else:
        top_reranked = []

    # 🚀 STEP 6: BUILD CONTEXT
    context_pieces = []
    for doc in top_reranked:
        source = doc.metadata.get("source", "Unknown")
        content = doc.page_content.replace("\n", " ") 
        context_pieces.append(f"[[Source: {source}]]\n{content}")

    context = "\n\n".join(context_pieces)
    history_text = format_chat_history(chat_history_list)
    
    st.session_state["last_retrieved_context"] = context
    retrieval_time = time.time() - retrieval_start

    # 🚀 STEP 7: GENERATE RESPONSE
    gen_start = time.time()
    prompt = f"""You are AXIsstant, the official Academic AI of Ateneo de Naga University. 
Your goal is to provide accurate, strictly formatted answers based ONLY on the context provided.

### STRICT FORMATTING RULES (YOU MUST FOLLOW THESE):
1. **USE TABLES FOR DATA**: 
   - If the user asks for a **Curriculum**, **Schedule**, **List of Grades**, or **Faculty List**, you MUST output a Markdown Table.
   - Example format:
     | Course Code | Course Title | Units | Prerequisite |
     |:------------|:-------------|:------|:-------------|
     | MATH101     | Calculus 1   | 3     | None         |

2. **CLEAN UP LISTS**:
   - Never start a line with a loose asterisk like `*Name`. 
   - Use standard Markdown bullets: `- Name`.
   - If listing people, Bold their names: `- **Dr. John Doe** - Dean`

3. **NO FLUFF**:
   - Do NOT say "Based on the provided context..." or "The document says...".
   - Just give the answer directly.

4. **MISSING INFO**:
   - If the specific semester or year is missing from the context, state clearly: "I have the curriculum for [Available Years], but [Requested Year] is missing from my records."
   - Do not hallucinate courses.

**Context:**
{context}

**Chat History:**
{history_text}

**Question:** {standalone_query}

**Answer:**"""

    llm = get_llm()

    try:
        full_response_buffer = ""
        for chunk in llm.stream(prompt):
            content = chunk.content
            if content:
                full_response_buffer += content
                yield content 
                time.sleep(0.005) 

        gen_time = time.time() - gen_start 
        total_time = time.time() - start_time

        st.session_state["performance_metrics"] = {
            "retrieval_latency": retrieval_time,
            "generation_latency": gen_time,
            "total_latency": total_time
        }
        
        # Save to semantic cache for future identical questions
        add_to_cache(standalone_query, full_response_buffer)
            
    except Exception as e:
        yield f"⚠️ **API Error:** {str(e)}"