import json
import time
import numpy as np
import streamlit as st
import concurrent.futures
from typing import List, Dict
from langchain_core.documents import Document
from rank_bm25 import BM25Okapi
import re
from tenacity import retry, stop_after_attempt, wait_exponential

from src.core.guardrails import verify_answer, validate_query, redact_pii
from src.config.settings import get_generator_llm, get_vectorstore, get_embeddings, get_retriever
from src.core.router import route_query, get_dynamic_k
from src.core.decomposition import decompose_query
from src.config.constants import (
    RETRIEVAL_K,
    RERANKER_TOP_K,
    DECOMPOSE_TRIGGERS,
    LOW_CONFIDENCE_THRESHOLD,
    HIGH_CONFIDENCE_THRESHOLD,
    HIGH_CONFIDENCE_MARGIN,
    STREAM_DELAY,
    POSITIONAL_SCORE_WEIGHT,
    SEMANTIC_CACHE_THRESHOLD,
)
from src.config.logging_config import logger

# ─────────────────────────────────────────────────────────────────────────────
# GLOBALS & CACHE
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading ranking model...")
def get_reranker():
    """CrossEncoder model lives in server RAM permanently."""
    from sentence_transformers import CrossEncoder
    return CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

@st.cache_resource(show_spinner=False)
def get_semantic_cache() -> list:
    return []

GLOBAL_CACHE = get_semantic_cache()

# Words that signal the query references prior conversation and needs rewriting
_CONTEXT_TRIGGERS = re.compile(
    r'\b(it|its|they|them|their|this|that|these|those|the same|'
    r'above|previous|earlier|last|mentioned|said|again|also|more|'
    r'how about|what about|and the|the other|besides|aside from)\b',
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

def check_semantic_cache(query: str, threshold: float = SEMANTIC_CACHE_THRESHOLD) -> str:
    """
    Checks the in-process semantic cache for a response to a near-identical
    prior query.  Uses cosine similarity between all-MiniLM-L6-v2 embeddings.

    Args:
        query:     The (already contextualized) standalone query string.
        threshold: Minimum cosine similarity for a cache hit.  Defaults to
                   ``SEMANTIC_CACHE_THRESHOLD`` (0.88).  Set higher for stricter
                   matching; lower values risk returning off-topic cached answers.

    Returns:
        The cached response string on a hit, or ``None`` on a miss.
    """
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
    """
    Embeds ``query`` and stores the (embedding, response) pair in ``GLOBAL_CACHE``.
    Uses a fixed-size FIFO eviction policy (max 50 entries) to stay within
    Streamlit Cloud RAM limits.
    Failures are logged as warnings and do not interrupt the response stream.
    """
    try:
        emb_model = get_embeddings()
        query_emb = np.array(emb_model.embed_query(query))
        GLOBAL_CACHE.append({"embedding": query_emb, "response": response})
        if len(GLOBAL_CACHE) > 50: GLOBAL_CACHE.pop(0)
    except Exception as e:
        logger.warning(f"Cache write failed: {e}")

def invalidate_cache():
    """Wipes the semantic cache object entirely."""
    GLOBAL_CACHE.clear() # Mutates the existing list object
    logger.info("🧹 Semantic cache invalidated. AI will now pull fresh data from Pinecone.")

# ─────────────────────────────────────────────────────────────────────────────
# 2. HELPER FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────
def format_chat_history(messages: List[Dict[str, str]]) -> str:
    """
    Formats the last 6 non-system messages into a plain-text "User: ... \nAssistant: ..."
    block for injection into the LLM prompt.  Curly braces in message content
    are escaped to ``{{`` / ``}}`` to prevent Python f-string substitution errors.

    Args:
        messages: Full conversation list including the system prompt at index 0.

    Returns:
        A newline-joined string of the last 6 turns, or "No previous context."
        if the conversation contains only the system prompt.
    """
    formatted_history = []
    history_to_process = messages[1:] if len(messages) > 1 else []
    for msg in history_to_process[-6:]: 
        role = "User" if msg["role"] == "user" else "Assistant"
        content = msg["content"].replace("{", "{{").replace("}", "}}")
        formatted_history.append(f"{role}: {content}")
    return "\n".join(formatted_history) if formatted_history else "No previous context."


def _tokenize(text: str) -> list:
    """Strip punctuation before BM25 tokenization so 'requirement.' matches 'requirement'."""
    return re.sub(r'[^\w\s]', ' ', text.lower()).split()


def hybrid_rerank(query: str, docs: List[Document]) -> List[Document]:
    """
    Re-ranks documents using BM25 combined with a positional bias score.

    The positional score (POSITIONAL_SCORE_WEIGHT * remaining position) encodes
    a mild preference for Pinecone’s original ANN rank, on the assumption that
    semantic similarity already provides a reasonable prior ordering.  The
    CrossEncoder in Step 5 provides the definitive final ranking.

    Args:
        query: The user’s standalone query string.
        docs:  Candidate documents from Pinecone retrieval.

    Returns:
        Top RETRIEVAL_K documents sorted by combined BM25 + positional score.
    """
    if not docs: return []
    try:
        tokenized_docs = [_tokenize(doc.page_content) for doc in docs]
        bm25 = BM25Okapi(tokenized_docs)
        tokenized_query = _tokenize(query)
        bm25_scores = bm25.get_scores(tokenized_query)

        ranked = []
        for i, doc in enumerate(docs):
            position_score = (len(docs) - i) * POSITIONAL_SCORE_WEIGHT
            final_score = bm25_scores[i] + position_score
            ranked.append((final_score, doc))

        ranked.sort(reverse=True, key=lambda x: x[0])
        return [doc for _, doc in ranked[:RETRIEVAL_K]]
    except Exception as e:
        logger.warning(f"Hybrid rerank failed: {e}")
        return docs[:RETRIEVAL_K]

def enforce_source_diversity(docs: List[Document], max_per_source: int = 3) -> List[Document]:
    """
    Limits chunks from any single source to max_per_source before CrossEncoder scoring.
    Ensures cross-document queries retrieve content from multiple files.
    Preserves BM25 rank ordering within each source group.
    """
    source_counts: Dict[str, int] = {}
    diverse_docs = []
    for doc in docs:
        source = doc.metadata.get("source", "unknown")
        count = source_counts.get(source, 0)
        if count < max_per_source:
            diverse_docs.append(doc)
            source_counts[source] = count + 1
    return diverse_docs


def _contains_markdown_table(text: str) -> bool:
    """Returns True if the response contains at least one Markdown table row."""
    return any(
        '|' in line and line.strip().startswith('|')
        for line in text.strip().split('\n')
    )


def prefer_latest_per_source(docs: List[Document]) -> List[Document]:
    """
    Filters documents so only chunks from the most recently ingested version
    of each source file are kept.

    Comparison is done via the ``uploaded_at`` Unix timestamp injected during
    ingestion.  If two ingestions of the same file share the same timestamp
    (edge case), all chunks are retained.  Documents lacking ``uploaded_at``
    default to 0 and are superseded by any version that has the field set.
    """
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
    """
    Main RAG generator pipeline.  A Streamlit generator function (uses ``yield``).

    Pipeline stages:
        0. Query validation and PII redaction (guardrails)
        1. Semantic cache check
        2. Intent routing (greeting / off_topic / search)
        3. Query decomposition for multi-topic questions
        4. Parallel Pinecone retrieval with dynamic K
        5. BM25 hybrid rerank + CrossEncoder rerank
        6. Context assembly
        7. Three-tier confidence gating:
               Tier 1 (score < LOW_CONFIDENCE_THRESHOLD)  → refuse to answer
               Tier 2 (moderate confidence)               → Critic LLM verifies draft
               Tier 3 (score ≥ HIGH_CONFIDENCE_THRESHOLD) → stream draft directly
        8. Metrics recording, cache population, and word-by-word streaming

    Args:
        query:             Raw user query string from the Streamlit text input.
        chat_history_list: List of {"role": str, "content": str} dicts representing
                           the full conversation so far (including the system prompt
                           at index 0 if present).

    Yields:
        str: Individual words (space-appended) for streaming display.
    """
    if chat_history_list is None:
        chat_history_list = []
    start_time = time.time()
    top_score = float("-inf")
    
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
    
    # 🚀 STEP 2: INTENT DETECTION
    try:
        intent, _, _, _ = route_query(standalone_query)
    except Exception as e:
        logger.warning(f"Router fallback triggered: {e}")
        intent = "search"

    # Handle greetings and off-topic early
    if intent in ["greeting", "off_topic"]:
        if intent == "greeting":
            msg = "Hey! I'm AXIsstant, your CSEA academic assistant. Ask me anything about your curriculum, lab guidelines, or school policies!"
        else:
            msg = "That's a bit outside my expertise — I'm mainly built for CSEA-related questions. Try asking me about your curriculum, guidelines, or academic policies!"
        
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
    dynamic_k = get_dynamic_k(standalone_query)
    retriever = get_retriever(k=dynamic_k)
    
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
        yield "Hmm, I couldn't find anything about that in the documents I have. Try asking your department chair!"
        return

    logger.info(f"📂 Retrieval Success: Found {len(all_docs)} raw chunks using K={dynamic_k}")

    # 🚀 STEP 5: DEDUPLICATION & RERANKING
    unique_docs_map = {hash(d.page_content): d for d in all_docs}
    latest_per_source = prefer_latest_per_source(list(unique_docs_map.values()))
    
    hybrid_results = hybrid_rerank(standalone_query, latest_per_source)
    # Detect curriculum queries and allow more chunks per source
    is_curriculum_query = any(
        kw in standalone_query.lower()
        for kw in ['curriculum', 'subject', 'course', 'year', 'semester', 'units', 'prerequisite', 'schedule']
    )
    max_chunks = 8 if is_curriculum_query else 3
    hybrid_results = enforce_source_diversity(hybrid_results, max_per_source=max_chunks)

    top_score = float("-inf")
    second_score = float("-inf")

    if hybrid_results:
        try:
            pairs = [(standalone_query, doc.page_content) for doc in hybrid_results]
            scores = get_reranker().predict(pairs)
            sorted_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
            
            top_score = float(scores[sorted_indices[0]])
            if len(sorted_indices) > 1:
                second_score = float(scores[sorted_indices[1]])
            
            # 🛑 REMOVED the hard `if top_score < -10.0:` block here. 
            # Step 7's Three-Tier logic will handle the confidence check.

            top_reranked = [hybrid_results[i] for i in sorted_indices[:RERANKER_TOP_K]] 
        except Exception as e:
            logger.error(f"Reranking failed: {e}")
            top_reranked = hybrid_results[:RERANKER_TOP_K]
    else:
        top_reranked = []
    
    logger.info(f"📊 Query: '{standalone_query}'")
    logger.info(f"📊 Docs retrieved: {len(all_docs)} → after rerank: {len(top_reranked)}")
    logger.info(
        f"📊 Top score: {top_score:.2f} | Second: {second_score:.2f} | "
        f"Margin: {(top_score - second_score):.2f} "
        f"(Cutoffs — low: {LOW_CONFIDENCE_THRESHOLD}, high: {HIGH_CONFIDENCE_THRESHOLD}, "
        f"margin: {HIGH_CONFIDENCE_MARGIN})"
    )
    logger.info(f"📄 Top chunks: {[(doc.metadata.get('year','?'), doc.metadata.get('semester','?')) for doc in top_reranked]}")

    # 🚀 STEP 6: BUILD CONTEXT
    context_pieces = [f"[[Source: {doc.metadata.get('source', 'Unknown')}]\n{doc.page_content}" for doc in top_reranked]
    context = "\n\n".join(context_pieces)
    st.session_state["last_retrieved_context"] = context
    # Log which chunk indices are being retrieved for debugging
    logger.info(f"📄 Context sources: {[doc.metadata.get('source','?') + ' chunk_' + str(doc.metadata.get('chunk_index','?')) for doc in top_reranked]}")
    retrieval_time = time.time() - retrieval_start

    # 🚀 STEP 7: THREE-TIER CONFIDENCE & GENERATION
    gen_start = time.time()
    
    # Define the Prompt (Instruction-Heavy for formatting)
    prompt = f"""You are AXIsstant, the official Academic AI of Ateneo de Naga University.
Answer the student's question using ONLY the context below. Be friendly but direct.

### RULES (FOLLOW STRICTLY):

1. **TONE**: Write like a friendly, approachable upperclassman helping a 
   classmate — casual but still accurate. Use natural conversational 
   language. Contractions are fine ("you'll", "it's", "here's"). 
   Never sound like a formal document or a customer service bot.
   Do NOT start with "Great question!", "Good question!", or any 
   sycophantic opener. Just talk like a normal person would.

2. **NO FILLER**: Do NOT say "To provide you with..." or "I'll need to refer to..." or "Let me check the handbook for you..." or any variation.

3. **LANGUAGE**: Always respond in English unless the student writes in Filipino, in which case respond in Filipino. 

4. **USE TABLES FOR STRUCTURED DATA**: When the context contains curriculum subjects, grading scales, schedules, or faculty lists, reproduce the ACTUAL data in a Markdown table. Include specific course codes, titles, units, and prerequisites. **SHOW EVERY ROW** — never truncate or skip rows.

5. **CLEAN UP LISTS**: Use `- **Name** - Role` for people.

6. **STRICTLY FACTUAL**: Use ONLY what is in the context. Do NOT pad with general advice. If the context genuinely lacks the answer, say: 'I couldn't find that in the available documents. You might want to check with your respective department chair directly!'

7. **CURRICULUM QUERIES**: When asked about subjects for a specific year, present ALL semesters for that year (1st semester, 2nd semester, and intersession if applicable) together in one response.

8. **BE CONCISE**: Get to the point quickly. One natural opener if it 
   helps the flow, then the answer. No repetition, no padding, no 
   sign-offs like "I hope this helps!" or "Feel free to ask more!".

9. **LISTS**: When listing multiple items, always put each item on its own 
   line with a blank line before the list starts. Never write list items 
   inline like "1. Item 2. Item 3. Item".

10. **ANALYTICAL QUERIES**: If asked to find the course with the most/least prerequisites, compare courses, or rank anything — go through ALL courses visible in the context, count carefully, and give a definitive answer with the course code and title. Show your reasoning as a small table if helpful.

11. **MULTIPLE TABLES**: When presenting multiple tables (e.g. different class periods), give each table a clear bold label above it like **1-hour class period** and put a blank line between each table. Never run tables together without labels.

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
            yield "I don't have enough info to answer that confidently — best to check with your department chair directly."
            return

        # Tier 2 & 3: Retrieval is sufficient — Invoke LLM with Retry Logic
        llm = get_generator_llm()
        draft_response = get_llm_response(llm, prompt).content
        
        score_margin = top_score - second_score if second_score != float("-inf") else top_score
        high_confidence = (
            top_score >= HIGH_CONFIDENCE_THRESHOLD
            and score_margin >= HIGH_CONFIDENCE_MARGIN
        )

        if high_confidence:
            # Tier 3: High confidence — Trust the draft
            logger.info(
                f"✨ High Confidence ({top_score:.2f}, margin {score_margin:.2f}). Bypassing Critic."
            )
            final_verified_response = draft_response
        else:
            # Tier 2: Moderate confidence — Trigger Critic to verify against context
            logger.info(
                f"🔍 Moderate Confidence ({top_score:.2f}, margin {score_margin:.2f}). "
                "Triggering Critic Persona..."
            )
            final_verified_response = verify_answer(standalone_query, context, draft_response)

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
        
        # ── NEW/UPDATED: Clean up Markdown Tables & Lists ──
        final_verified_response = fix_markdown_tables(final_verified_response)
        final_verified_response = re.sub(r'\s+(\d+\.\s)', r'\n\1', final_verified_response)
        
        # ── Generate suggested follow-up questions ──
        suggested_questions = generate_suggested_questions(
            standalone_query, final_verified_response, context
        )
        if suggested_questions:
            suggestions_md = "\n\n---\n**You might also want to ask:**\n" + \
                "\n".join(f"- {q}" for q in suggested_questions)
            final_verified_response += suggestions_md
        
        # Final Streaming Loop with Fallback
        # Table-aware: yield entire response at once when a Markdown table is present
        # so Streamlit never renders a partial/broken table mid-stream.
        # For plain text, stream word-by-word for a natural feel.
        try:
            if _contains_markdown_table(final_verified_response):
                yield final_verified_response
            else:
                # Stream word by word so the warm opener lands first
                words = final_verified_response.split(" ")
                for word in words:
                    yield word + " "
                    time.sleep(STREAM_DELAY) 
                    
        except GeneratorExit:
            # User navigated away; cleanup handled by Python GC
            return
        except Exception as e:
            logger.error(f"Streaming interruption: {e}")
            yield f"\n\n⚠️ *Stream interrupted. Displaying full response:* \n{final_verified_response}"
            
    except Exception as e:
        logger.error(f"❌ Generation Pipeline Failed: {e}")
        yield "Something went wrong on my end. Give it another try in a bit!"

def fix_markdown_tables(text: str) -> str:
    lines = text.split('\n')
    fixed = []
    i = 0
    while i < len(lines):
        line = lines[i]
        
        # Add blank line before a table row if previous line is not blank/separator
        if (line.strip().startswith('|') and 
            fixed and 
            fixed[-1].strip() and 
            not fixed[-1].strip().startswith('|')):
            fixed.append('')
            
        fixed.append(line)
        
        # Inject missing separator row after header
        if (line.strip().startswith('|') and 
            i + 1 < len(lines) and 
            lines[i + 1].strip().startswith('|') and 
            '---' not in lines[i + 1]):
            
            col_count = line.count('|') - 1
            if col_count > 0:
                separator = '|' + '---|' * col_count
                fixed.append(separator)
                
        # Add blank line after a table ends (next line is not a table row)
        if (line.strip().startswith('|') and 
            i + 1 < len(lines) and 
            lines[i + 1].strip() and 
            not lines[i + 1].strip().startswith('|')):
            fixed.append('')
            
        i += 1
        
    return '\n'.join(fixed)


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def get_llm_response(llm, prompt):
    """Reliable wrapper for LLM calls with exponential backoff."""
    return llm.invoke(prompt)

def generate_suggested_questions(query: str, response: str, context: str) -> list[str]:
    """
    Generates 3 related follow-up questions based on the query, response, and context.
    Returns a list of question strings, or empty list on failure.
    """
    try:
        llm = get_generator_llm()
        prompt = f"""Based on this conversation, suggest exactly 3 short follow-up questions 
the user might want to ask next. Return ONLY the 3 questions as a JSON array of strings.
No explanations, no numbering, no extra text. Just the JSON array.

Example output: ["Question 1?", "Question 2?", "Question 3?"]

User asked: {query}
Assistant answered: {response[:300]}
Available topic context: {context[:500]}

Output (JSON array only):"""
        
        result = llm.invoke(prompt).content.strip()
        # Strip markdown code fences if present
        result = re.sub(r'^```json|^```|```$', '', result.strip(), flags=re.MULTILINE).strip()
        questions = json.loads(result)
        if isinstance(questions, list):
            return [q for q in questions if isinstance(q, str)][:3]
    except Exception as e:
        logger.warning(f"Suggested questions failed: {e}")
    return []