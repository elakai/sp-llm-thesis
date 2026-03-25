import json
import time
import numpy as np
import streamlit as st
import concurrent.futures
from typing import List, Dict
import re
from tenacity import retry, stop_after_attempt, wait_exponential

from src.core.guardrails import verify_answer, validate_query, redact_pii
from src.config.settings import get_generator_llm, get_embeddings, get_retriever
from src.core.router import route_query, get_dynamic_k
from src.core.decomposition import decompose_query
from src.config.constants import (
    RERANKER_TOP_K,
    DECOMPOSE_TRIGGERS,
    LOW_CONFIDENCE_THRESHOLD,
    HIGH_CONFIDENCE_THRESHOLD,
    HIGH_CONFIDENCE_MARGIN,
    STREAM_DELAY,
    SEMANTIC_CACHE_THRESHOLD,
)
from src.config.logging_config import logger

# ── NEW IMPORTS FROM REFACTOR ──
from src.core.reranking import (
    hybrid_rerank, enforce_source_diversity, filter_to_program,
    filter_to_people_docs, boost_people_list_docs, rank_people_list_docs,
    prefer_latest_per_source
)
from src.core.response_formatting import (
    fix_markdown_tables, format_raw_links, remove_speculative_sentences,
    build_source_certainty_note, fallback_questions, build_no_answer_response,
    _contains_markdown_table, _contains_speculation
)

# ─────────────────────────────────────────────────────────────────────────────
# GLOBALS & CACHE
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading ranking model...")
def get_reranker():
    from sentence_transformers import CrossEncoder
    return CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

@st.cache_resource(show_spinner=False)
def get_semantic_cache() -> list:
    return []

GLOBAL_CACHE = get_semantic_cache()
_REWRITE_CACHE = {} 

_CONTEXT_TRIGGERS = re.compile(
    r'\b(it|its|they|them|their|this|that|these|those|the same|'
    r'above|previous|earlier|last|mentioned|said|again|'
    r'how about|what about|and the|the other|besides|aside from)\b',
    re.IGNORECASE
)

# ─────────────────────────────────────────────────────────────────────────────
# 1. CONVERSATIONAL MEMORY & CACHING FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────
def format_chat_history(messages: List[Dict[str, str]]) -> str:
    formatted_history = []
    history_to_process = messages[1:] if len(messages) > 1 else []
    for msg in history_to_process[-6:]: 
        role = "User" if msg["role"] == "user" else "Assistant"
        content = msg["content"].replace("{", "{{").replace("}", "}}")
        formatted_history.append(f"{role}: {content}")
    return "\n".join(formatted_history) if formatted_history else "No previous context."

def contextualize_query(query: str, chat_history_list: List[Dict[str, str]]) -> str:
    history_to_process = chat_history_list[1:] if len(chat_history_list) > 1 else []
    if not history_to_process: return query
    if not _CONTEXT_TRIGGERS.search(query): return query
        
    history_text = format_chat_history(chat_history_list)
    session_id = st.session_state.get("session_id", "default")
    cache_key = (session_id, query, history_text[-300:]) 
    if cache_key in _REWRITE_CACHE: return _REWRITE_CACHE[cache_key]
        
    prompt = f"""Given the following chat history and the user's latest question, formulate a standalone question that can be understood without the chat history.
    Do NOT answer the question. Just reformulate it if needed. If it doesn't need reformulating, return it exactly as is.

    Chat History:
    {history_text}

    Latest Question: {query}
    Standalone Question:"""
    
    try:
        llm = get_generator_llm()
        standalone_query = llm.invoke(prompt).content.strip()
        _REWRITE_CACHE[cache_key] = standalone_query
        if len(_REWRITE_CACHE) > 100: _REWRITE_CACHE.pop(next(iter(_REWRITE_CACHE))) 
        return standalone_query
    except Exception as e:
        logger.warning(f"Contextualize Error: {e}")
        return query

def check_semantic_cache(query: str, threshold: float = SEMANTIC_CACHE_THRESHOLD) -> str:
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
    except Exception as e:
        logger.warning(f"Cache write failed: {e}")

def invalidate_cache():
    GLOBAL_CACHE.clear() 
    logger.info("🧹 Semantic cache invalidated.")


# ─────────────────────────────────────────────────────────────────────────────
# 2. INTENT DETECTION & QUERY BUILDING
# ─────────────────────────────────────────────────────────────────────────────
def _detect_query_intent(query: str) -> str:
    q_lower = (query or "").lower()
    tokens = set(re.findall(r"[a-z0-9']+", q_lower))

    def has_any(words): return any(word in tokens for word in words)
    def has_phrases(phrases): return any(phrase in q_lower for phrase in phrases)

    if has_any({"where", "room", "building", "located", "floor", "lab", "office", "directory"}): return "location"
    if has_any({"curriculum", "course", "subject", "semester", "units", "prerequisite"}): return "curriculum"
    if has_any({"who", "faculty", "chair", "dean", "professor", "staff", "instructor"}) or has_phrases(["department chair", "faculty list"]): return "people"
    if has_any({"download", "link", "pdf", "form", "access"}) or has_phrases(["google form"]): return "download"
    if has_any({"policy", "rule", "guideline", "procedure", "manual"}) or has_phrases(["dress code"]): return "policy"
    return "general"

def _is_listing_query(query: str) -> bool:
    q = (query or "").lower()
    return any(trigger in q for trigger in ["list", "enumerate", "show", "all ", "who are", "names", "members"])

def _is_incomplete_query(query: str) -> bool:
    q_lower = (query or "").strip().lower()
    if re.search(r'\b[A-Z]{1,3}\d{3}\b', query.upper()): return False
        
    tokens = re.findall(r"[a-zA-Z0-9']+", q_lower)
    if len(tokens) <= 2: return True

    starters = {"what", "where", "who", "when", "why", "how", "which"}
    verbs = {"is", "are", "can", "do", "does", "did", "show", "list", "find", "tell"}
    return len(tokens) <= 4 and not any(t in starters or t in verbs for t in tokens)

def _build_incomplete_query_variants(query: str, chat_history_list: List[Dict[str, str]]) -> List[str]:
    base = (query or "").strip()
    if not base: return []

    intent = _detect_query_intent(base)
    variants = [base]

    if intent == "location": variants.append(f"{base} location building room")
    elif intent == "curriculum": variants.append(f"{base} curriculum course prerequisite")
    elif intent == "people": variants.append(f"{base} faculty staff department chair professor")
    elif intent == "download": variants.append(f"{base} official link pdf")
    elif intent == "policy": variants.append(f"{base} policy guideline rule")

    for msg in reversed(chat_history_list):
        if msg.get("role") == "user" and (msg.get("content") or "").strip().lower() != base.lower():
            variants.append(f"{msg.get('content').strip()} {base}")
            break

    deduped, seen = [], set()
    for v in variants:
        norm = v.strip().lower()
        if norm and norm not in seen:
            deduped.append(v.strip())
            seen.add(norm)
    return deduped[:2] 

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def get_llm_response(llm, prompt):
    return llm.invoke(prompt)


# ─────────────────────────────────────────────────────────────────────────────
# 3. MAIN GENERATOR PIPELINE
# ─────────────────────────────────────────────────────────────────────────────
def generate_response(query: str, chat_history_list: List[Dict[str, str]] = None):
    if chat_history_list is None: chat_history_list = []
    start_time = time.time()
    
    is_valid, clean_query = validate_query(query) 
    if not is_valid:
        yield clean_query
        return
    
    safe_query = redact_pii(clean_query) 
    standalone_query = contextualize_query(safe_query, chat_history_list)
    is_incomplete_input = _is_incomplete_query(standalone_query)
    
    if not is_incomplete_input and not _is_listing_query(standalone_query):
        cached_answer = check_semantic_cache(standalone_query)
        if cached_answer:
            words = cached_answer.split(" ")
            for i in range(0, len(words), 3):
                yield " ".join(words[i:i+3]) + " "
                time.sleep(0.01)
            return

    retrieval_start = time.time()
    
    try: intent, _, _, _ = route_query(standalone_query)
    except Exception as e:
        logger.warning(f"Router fallback triggered: {e}")
        intent = "search"

    if intent in ["greeting", "off_topic"]:
        if intent == "greeting":
            msg = ("Hey! I'm AXIsstant, the academic assistant specifically built "
                   "for CSEA students and faculty at Ateneo de Naga University. "
                   "I can help with curriculum info, lab guidelines, school policies, "
                   "room locations, and more. What do you need?")
        else:
            msg = ("That's outside what I can help with — I'm specifically built "
                   "for CSEA at Ateneo de Naga University. Try asking about "
                   "your curriculum, lab guidelines, or school policies!")
        
        for word in msg.split():
            yield word + " "
            time.sleep(0.02)
        return

    is_complex = (not is_incomplete_input) and any(trigger in standalone_query.lower() for trigger in DECOMPOSE_TRIGGERS)
    sub_queries = [standalone_query]
    if is_complex:
        try: sub_queries = decompose_query(standalone_query)
        except: pass

    if is_incomplete_input:
        for variant in _build_incomplete_query_variants(standalone_query, chat_history_list):
            if variant not in sub_queries:
                sub_queries.append(variant)

    dynamic_k = max(get_dynamic_k(standalone_query), 20) if is_incomplete_input else get_dynamic_k(standalone_query)
    retriever = get_retriever(k=dynamic_k)
    
    all_docs = []
    if len(sub_queries) == 1:
        all_docs = retriever.invoke(sub_queries[0])
    else:
        with concurrent.futures.ThreadPoolExecutor() as executor:
            results = list(executor.map(retriever.invoke, sub_queries))
        for res in results: all_docs.extend(res)

    if not all_docs:
        yield build_no_answer_response(standalone_query)
        return

    logger.info(f"📂 Retrieval Success: Found {len(all_docs)} raw chunks using K={dynamic_k}")

    unique_docs_map = {hash(d.page_content): d for d in all_docs}
    latest_per_source = prefer_latest_per_source(list(unique_docs_map.values()))
    
    is_curriculum_query = any(
        kw in standalone_query.lower() for kw in [
            'curriculum', 'subject', 'course', 'year', 'semester', 'units',
            'prerequisite', 'schedule', 'ojt', 'practicum', 'internship',
            'immersion', 'operating systems', 'elective', 'track'
        ]
    )
    is_analytical_query = any(kw in standalone_query.lower() for kw in ['most', 'least', 'highest', 'lowest', 'compare', 'which course', 'how many courses', 'most prerequisites', 'hardest', 'rank', 'full', 'entire', 'complete', 'all subjects'])
    is_download_query = any(kw in standalone_query.lower() for kw in ['download', 'link', 'pdf', 'get the', 'access', 'where can i get', 'where can i download'])

    top_score, second_score = float("-inf"), float("-inf")
    hybrid_results = hybrid_rerank(standalone_query, latest_per_source)

    if _detect_query_intent(standalone_query) == "people":
        people_pool = filter_to_people_docs(latest_per_source, standalone_query)
        people_pool = boost_people_list_docs(standalone_query, people_pool, dynamic_k)
        ranked_people = rank_people_list_docs(people_pool, standalone_query)
        top_reranked = ranked_people[:max(RERANKER_TOP_K, 16)]
        if top_reranked: top_score = max(top_score, 5.0)

    elif is_analytical_query and is_curriculum_query:
        all_program_docs = filter_to_program(latest_per_source, standalone_query)
        big_retriever = get_retriever(k=50)
        extra_filtered = filter_to_program(prefer_latest_per_source(big_retriever.invoke(standalone_query)), standalone_query)
        combined = {hash(d.page_content): d for d in all_program_docs + extra_filtered}
        top_reranked = list(combined.values())
        if top_reranked: top_score = 10.0 
        
    else:
        max_chunks = 8 if is_curriculum_query else 3
        hybrid_results = enforce_source_diversity(hybrid_results, max_per_source=max_chunks)

        if hybrid_results:
            try:
                pairs = [(standalone_query, doc.page_content) for doc in hybrid_results]
                scores = get_reranker().predict(pairs)
                sorted_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
                
                top_score = float(scores[sorted_indices[0]])
                if len(sorted_indices) > 1: second_score = float(scores[sorted_indices[1]])
                
                top_reranked = [hybrid_results[i] for i in sorted_indices[:RERANKER_TOP_K]] 
            except Exception as e:
                logger.error(f"Reranking failed: {e}")
                top_reranked = hybrid_results[:RERANKER_TOP_K]
        else:
            top_reranked = []
            
    if is_download_query and top_reranked is not None:
        link_retriever = get_retriever(k=20)
        link_filtered = prefer_latest_per_source(link_retriever.invoke("official curriculum PDF download link"))
        existing_hashes = {hash(d.page_content) for d in top_reranked}
        for doc in link_filtered:
            if hash(doc.page_content) not in existing_hashes:
                top_reranked.append(doc)
                existing_hashes.add(hash(doc.page_content))
        if top_reranked: top_score = max(top_score, 5.0)

    is_facility_query = any(
        kw in standalone_query.lower() for kw in ['room', 'building', 'floor', 'lab', 'laboratory', 'office', 'located', 'where is', 'where are', 'nasaan', 'campus', 'facility', 'location of']
    )

    if is_facility_query:
        already_has_directory = any('directory' in doc.metadata.get('source', '').lower() or 'campus' in doc.metadata.get('source', '').lower() for doc in top_reranked)
        if not already_has_directory:
            building_code_match = re.search(r'\b([A-Z]{1,3})\d{3}\b', standalone_query.upper())
            if building_code_match:
                directory_query = f"{building_code_match.group(1)} building rooms directory"
            else:
                directory_query = "campus building directory rooms"

            directory_filtered = prefer_latest_per_source(get_retriever(k=8).invoke(directory_query))
            existing_hashes = {hash(d.page_content) for d in top_reranked}
            for doc in directory_filtered:
                if hash(doc.page_content) not in existing_hashes:
                    top_reranked.append(doc)
                    existing_hashes.add(hash(doc.page_content))
            if top_reranked: top_score = max(top_score, 5.0)
    
    logger.info(f"📊 Top score: {top_score:.2f} | Second: {second_score:.2f}")
    
    context_pieces = [f"[[Source: {doc.metadata.get('source', 'Unknown')}]\n{doc.page_content}" for doc in top_reranked]
    context = "\n\n".join(context_pieces)
    st.session_state["last_retrieved_context"] = context
    
    retrieval_time = time.time() - retrieval_start
    gen_start = time.time()
    
    prompt = f"""You are AXIsstant, the friendly and helpful Academic AI assistant of Ateneo de Naga University's College of Science, Engineering, and Architecture (CSEA). You help students and faculty with academic questions in a warm, conversational tone.

Answer the question using ONLY the context below.

### RULES (FOLLOW STRICTLY):
1. **TONE**: Write like a friendly, approachable upperclassman helping a classmate — casual but still accurate. Never sound like a formal document or a customer service bot. Do NOT start with "Great question!", "Good question!", "So,", "Kuya", "So here's", or "So to answer". Make every response different.
2. **NO FILLER**: Do NOT say "To provide you with..." or "I'll need to refer to...".
3. **LANGUAGE**: Always respond in English unless the student writes in Filipino, in which case respond in Filipino.
4. **USE TABLES FOR STRUCTURED DATA**: When the context contains curriculum subjects, grading scales, schedules, or faculty lists, reproduce the ACTUAL data in a Markdown table. **SHOW EVERY ROW** — never truncate or skip rows. Do NOT include rows where every cell contains only dashes (---).
5. **CLEAN UP LISTS**: Use `- **Name** - Role` for people. 
6. **STRICTLY FACTUAL**: Use ONLY what is in the context. If the context genuinely lacks the answer, say: 'I couldn't find that in the available documents. You might want to check with your respective department chair directly!'
7. **CURRICULUM QUERIES**: When asked about subjects for a specific year, present ALL semesters for that year.
8. **BE CONCISE**: Get to the point quickly.
9. **LISTS**: When listing multiple items, always put each item on its own line with a blank line before the list starts.
10. **ANALYTICAL QUERIES**: If asked to find the course with the most/least prerequisites, compare courses, or rank anything — count carefully, and give a definitive answer. 
11. **MULTIPLE TABLES**: Give each table a clear bold label above it.
12. **LINKS**: Never paste raw URLs. Always format links as descriptive markdown like [Download the official curriculum here](url).
13. **ROOM CODES**: When the user asks about a room like "D412", "AL112", or "EB111", look up the building prefix in the campus directory in the context. "D" = Fr. Francis Dolan SJ Building, "AL" = Godofredo Alingal SJ Building, "EB" = Engineering Building. Always decode the building name, floor number, and room number for the user. Only use acronyms confirmed in the context.
14. **NO SPECULATION**: Never guess or infer missing details. Do not use speculative words like "likely", "possibly", "probably", "might", or "could be".
15. **PREREQUISITES**: When showing curriculum subjects, ALWAYS include the prerequisite column in the table. If a subject has no prerequisite, write "None" in that cell.

**Context:**
{context}

**Chat History:**
{format_chat_history(chat_history_list)}

**Question:** {standalone_query}

**Answer:**
Provide your answer above, then on a new line add exactly this block:

SUGGESTED_QUESTIONS:
1. [first follow-up question, under 12 words, answerable from the context above]
2. [second follow-up question, under 12 words, answerable from the context above]
3. [third follow-up question, under 12 words, answerable from the context above]

Hard rules for suggested questions:
- Only ask about things explicitly present in the context above
- Do not introduce new course codes, names, or policies not mentioned
- Do not repeat the user's original question
- Questions must be different topics from each other
"""

    try:
        is_protected_query = (
            is_curriculum_query or is_facility_query or is_analytical_query or is_download_query or is_incomplete_input
        )

        if top_score < LOW_CONFIDENCE_THRESHOLD and not is_protected_query:
            logger.warning(f"🔇 Low Retrieval Score ({top_score:.2f}). Aborting generation.")
            yield build_no_answer_response(standalone_query)
            return

        llm = get_generator_llm()
        draft_raw = get_llm_response(llm, prompt).content.strip()
        
        SUGGESTION_SPLIT = re.compile(r'\n+SUGGESTED_QUESTIONS:\s*\n((?:\d+\..+\n?){1,3})', re.IGNORECASE)
        suggestion_match = SUGGESTION_SPLIT.search(draft_raw)
        suggested_questions = []

        if suggestion_match:
            draft_response = draft_raw[:suggestion_match.start()].strip()
            raw_lines = suggestion_match.group(1).strip().split('\n')
            for line in raw_lines:
                q = re.sub(r'^\d+\.\s*', '', line).strip()
                if q and len(q) > 8:
                    if not q.endswith('?'): q += '?'
                    suggested_questions.append(q)
        else:
            draft_response = draft_raw.strip()

        score_margin = top_score - second_score if second_score != float("-inf") else top_score
        high_confidence = (top_score >= 1.5 and score_margin >= 0.4)

        if high_confidence:
            logger.info(f"✨ High Confidence ({top_score:.2f}, margin {score_margin:.2f}). Bypassing Critic.")
            final_verified_response = draft_response
        else:
            logger.info(f"🔍 Moderate Confidence ({top_score:.2f}, margin {score_margin:.2f}). Triggering Critic Persona...")
            final_verified_response = verify_answer(standalone_query, context, draft_response)

        st.session_state["performance_metrics"] = {
            "retrieval_latency": retrieval_time,
            "generation_latency": time.time() - gen_start,
            "total_latency": time.time() - start_time,
            "confidence_score": float(top_score)
        }

        if not final_verified_response: final_verified_response = draft_response

        if _contains_speculation(final_verified_response):
            cleaned_non_speculative = remove_speculative_sentences(final_verified_response)
            final_verified_response = cleaned_non_speculative if cleaned_non_speculative else "I couldn't find an explicit answer for that detail in the retrieved documents."

        final_verified_response = fix_markdown_tables(final_verified_response)
        final_verified_response = re.sub(r'\s+(\d+\.\s)', r'\n\1', final_verified_response)
        final_verified_response = re.sub(r'(?<!\n)\s{2,}-\s+\*\*', r'\n- **', final_verified_response)
        final_verified_response = format_raw_links(final_verified_response)

        clean_response_for_cache = final_verified_response
        add_to_cache(standalone_query, clean_response_for_cache)

        source_list = [doc.metadata.get('source', 'Unknown') for doc in top_reranked]
        certainty_note = build_source_certainty_note(top_score, score_margin, source_list)
        final_verified_response += f"\n\n{certainty_note}"

        if not suggested_questions:
            suggested_questions = fallback_questions(final_verified_response + "\n" + context, standalone_query, max_items=3)
            
        if suggested_questions:
            suggestions_md = "\n\n---\n**You might also want to ask:**\n" + "\n".join(f"- {q}" for q in suggested_questions[:3])
            final_verified_response += suggestions_md
        
        try:
            if _contains_markdown_table(final_verified_response):
                yield final_verified_response
            else:
                chunk_size = 40
                for i in range(0, len(final_verified_response), chunk_size):
                    yield final_verified_response[i:i+chunk_size]
                    time.sleep(STREAM_DELAY) 
        except GeneratorExit:
            return
        except Exception as e:
            logger.error(f"Streaming interruption: {e}")
            yield f"\n\n⚠️ *Stream interrupted. Displaying full response:* \n{final_verified_response}"
            
    except Exception as e:
        logger.error(f"❌ Generation Pipeline Failed: {e}")
        yield "Something went wrong on my end. Give it another try in a bit!"