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
# Note: I removed verify_answer temporarily to ensure it doesn't break the table formatting
# from src.core.guardrails import verify_answer 

# ─────────────────────────────────────────────────────────────────────────────
# Global Reranker (Load once)
# ─────────────────────────────────────────────────────────────────────────────
# We use a distinct model for reranking to ensure high accuracy
reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

# ─────────────────────────────────────────────────────────────────────────────
# Helper Functions
# ─────────────────────────────────────────────────────────────────────────────

def format_chat_history(messages: List[Dict[str, str]]) -> str:
    """Converts Streamlit's session state messages into a string."""
    formatted_history = []
    # Skip the first message if it's a system prompt or empty
    history_to_process = messages[1:] if len(messages) > 1 else []
    
    for msg in history_to_process[-6:]: # Keep last 6 turns for context
        role = "User" if msg["role"] == "user" else "Assistant"
        content = msg["content"].replace("{", "{{").replace("}", "}}")
        formatted_history.append(f"{role}: {content}")
    
    return "\n".join(formatted_history) if formatted_history else "No previous context."

def rewrite_query(query: str) -> str:
    """Uses LLM to rewrite the query. Returns original if short or fails."""
    if len(query.split()) < 4: return query
        
    try:
        llm = get_llm()
        # Simple prompt to make the query search-engine friendly
        prompt = f"Extract the core keywords for a vector search from this student question: {query}"
        return llm.invoke(prompt).content.strip() or query
    except Exception:
        return query

def hybrid_rerank(query: str, docs: List[Document]) -> List[Document]:
    """Combines BM25 (keyword match) with Semantic Search results."""
    if not docs: return []

    try:
        # BM25 scores (Keyword exact match)
        tokenized_docs = [doc.page_content.split() for doc in docs]
        bm25 = BM25Okapi(tokenized_docs)
        tokenized_query = query.lower().split()
        bm25_scores = bm25.get_scores(tokenized_query)

        ranked = []
        for i, doc in enumerate(docs):
            # Hybrid Score = BM25 Score + (Vector Rank Weight)
            # We give a slight boost to documents that appeared earlier in the Vector Search
            position_score = (len(docs) - i) * 0.05 
            final_score = bm25_scores[i] + position_score
            ranked.append((final_score, doc))

        ranked.sort(reverse=True, key=lambda x: x[0])
        return [doc for _, doc in ranked[:15]] # Return top 15 candidates
    except Exception as e:
        print(f"Hybrid rerank failed: {e}")
        return docs[:10]

def prefer_latest_per_source(docs: List[Document]) -> List[Document]:
    """Deduplicates chunks, preferring the most recent upload."""
    if not docs: return []

    grouped: Dict[str, List[Document]] = {}
    for doc in docs:
        source = doc.metadata.get("source", "unknown")
        grouped.setdefault(source, []).append(doc)

    filtered_docs = []
    for source, group in grouped.items():
        # Find the latest timestamp for this specific file
        latest_timestamp = max((d.metadata.get("uploaded_at", 0) for d in group), default=0)
        # Keep only chunks from that version
        current_version_chunks = [d for d in group if d.metadata.get("uploaded_at", 0) == latest_timestamp]
        filtered_docs.extend(current_version_chunks)

    return filtered_docs

# ─────────────────────────────────────────────────────────────────────────────
# Main Retrieval Function
# ─────────────────────────────────────────────────────────────────────────────

def generate_response(query: str, chat_history_list: List[Dict[str, str]] = []):
    start_time = time.time()
    retrieval_start = time.time()
    
    # 🚀 STEP 1: SMART ROUTER
    # Identify if the user is just saying "Hi" or asking a real question
    try:
        intent = route_query(query)
    except:
        intent = "query" # Fallback

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

    # 🚀 STEP 2: DECOMPOSITION (Break down complex questions)
    is_complex = len(query.split()) > 15 or " and " in query
    sub_queries = [query]
    if is_complex:
        try:
            sub_queries = decompose_query(query)
        except:
            pass

    # 🚀 STEP 3: PARALLEL RETRIEVAL
    vectorstore = get_vectorstore()
    retriever = vectorstore.as_retriever(search_kwargs={"k": 15}) # Fetch more docs initially
    all_docs = []

    with concurrent.futures.ThreadPoolExecutor() as executor:
        results = list(executor.map(retriever.invoke, sub_queries))
    
    for res in results:
        all_docs.extend(res)

    if not all_docs:
        yield "I checked the handbook, but I couldn't find any information about that."
        return

    # 🚀 STEP 4: DEDUPLICATION & RERANKING
    # Remove duplicates
    unique_docs_map = {hash(d.page_content): d for d in all_docs}
    latest_per_source = prefer_latest_per_source(list(unique_docs_map.values()))
    
    # Rerank Logic
    rewritten_query = rewrite_query(query) 
    hybrid_results = hybrid_rerank(rewritten_query, latest_per_source)

    if hybrid_results:
        try:
            top_candidates = hybrid_results[:10] 
            # Cross-Encoder Rerank (The "Golden" Reranker)
            pairs = [(query, doc.page_content) for doc in top_candidates]
            scores = reranker.predict(pairs)
            sorted_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
            
            # Filter low relevance (Threshold: -10.0 allows broader matches)
            if scores[sorted_indices[0]] < -10.0:
                yield "I found some documents, but they didn't seem relevant to your specific question."
                return

            top_reranked = [top_candidates[i] for i in sorted_indices[:6]] # Keep top 6
        except Exception as e:
            print(f"Reranking failed: {e}")
            top_reranked = hybrid_results[:6]
    else:
        top_reranked = []

    # 🚀 STEP 5: BUILD CONTEXT
    context_pieces = []
    for doc in top_reranked:
        source = doc.metadata.get("source", "Unknown")
        content = doc.page_content.replace("\n", " ") 
        context_pieces.append(f"[[Source: {source}]]\n{content}")

    context = "\n\n".join(context_pieces)
    history_text = format_chat_history(chat_history_list)
    
    st.session_state["last_retrieved_context"] = context
    retrieval_time = time.time() - retrieval_start

    # 🚀 STEP 6: GENERATE (THE FORMATTING SNOB)
    gen_start = time.time()
    
    # --- THIS IS THE NEW PROMPT THAT FIXES YOUR FORMATTING ---
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

**Question:** {query}

**Answer:**"""

    llm = get_llm()

    try:
        # We stream the response directly from the LLM to the UI
        full_response_buffer = ""
        for chunk in llm.stream(prompt):
            content = chunk.content
            if content:
                full_response_buffer += content
                yield content 
                # Small sleep ensures the UI renders smoother tables
                time.sleep(0.005) 

        gen_time = time.time() - gen_start 
        total_time = time.time() - start_time

        # Update metrics
        st.session_state["performance_metrics"] = {
            "retrieval_latency": retrieval_time,
            "generation_latency": gen_time,
            "total_latency": total_time
        }
            
    except Exception as e:
        yield f"⚠️ **API Error:** {str(e)}"