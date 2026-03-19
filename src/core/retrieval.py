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

# [FIX 7]: Simple in-memory cache to prevent redundant contextualize API calls
_REWRITE_CACHE = {}

# [FIX 6]: Removed "also" and "more" to prevent unnecessary rewrites
_CONTEXT_TRIGGERS = re.compile(
    r'\b(it|its|they|them|their|this|that|these|those|the same|'
    r'above|previous|earlier|last|mentioned|said|again|'
    r'how about|what about|and the|the other|besides|aside from)\b',
    re.IGNORECASE
)

# ─────────────────────────────────────────────────────────────────────────────
# 1. CONVERSATIONAL MEMORY & CACHING FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────
def contextualize_query(query: str, chat_history_list: List[Dict[str, str]]) -> str:
    history_to_process = chat_history_list[1:] if len(chat_history_list) > 1 else []
    if not history_to_process: return query
    
    if not _CONTEXT_TRIGGERS.search(query):
        logger.info(f"Context skip: No pronouns/references detected in '{query}'")
        return query
        
    history_text = format_chat_history(chat_history_list)
    
    # Use a lightweight cache to prevent identical follow-ups from hitting the LLM
    cache_key = (query, history_text[-300:]) 
    if cache_key in _REWRITE_CACHE:
        return _REWRITE_CACHE[cache_key]
        
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
        
        _REWRITE_CACHE[cache_key] = standalone_query
        if len(_REWRITE_CACHE) > 100:
            _REWRITE_CACHE.pop(next(iter(_REWRITE_CACHE))) # Keep cache bounded
            
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

def _tokenize(text: str) -> list:
    return re.sub(r'[^\w\s]', ' ', text.lower()).split()

def hybrid_rerank(query: str, docs: List[Document]) -> List[Document]:
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
    source_counts: Dict[str, int] = {}
    diverse_docs = []
    for doc in docs:
        source = doc.metadata.get("source", "unknown")
        count = source_counts.get(source, 0)
        if count < max_per_source:
            diverse_docs.append(doc)
            source_counts[source] = count + 1
    return diverse_docs

def filter_to_program(docs: List[Document], query: str) -> List[Document]:
    PROGRAM_KEYWORDS = {
        'computer engineering': 'cpe',
        'cpe': 'cpe',
        'civil engineering': 'ce',
        'bs ce': 'ce',
        'electronics engineering': 'ece',
        'bs ece': 'ece',
        'architecture': 'arch',
        'bs arch': 'arch',
        'biology': 'bio',
        'bs bio': 'bio',
        'mathematics': 'math',
        'bs math': 'math',
        'environmental management': 'em',
        'bs em': 'em',
    }
    q = query.lower()
    matched_program = None
    for keyword, code in PROGRAM_KEYWORDS.items():
        if keyword in q:
            matched_program = code
            break

    if not matched_program:
        return docs

    return [
        d for d in docs
        if matched_program in d.metadata.get("source", "").lower()
    ]

def filter_to_people_docs(docs: List[Document], query: str) -> List[Document]:
    if not docs:
        return []

    q = (query or "").lower()
    people_triggers = [
        "professor", "faculty", "instructor", "teacher", "staff", "chair", "dean",
        "department chair", "chairperson"
    ]
    if not any(trigger in q for trigger in people_triggers):
        return docs

    content_keywords = [
        "faculty", "professor", "instructor", "teacher", "staff", "chair",
        "department", "office", "personnel", "full-time", "part-time"
    ]
    source_keywords = ["faculty", "organizational", "org", "structure", "staff", "personnel"]

    filtered = []
    for doc in docs:
        content = (doc.page_content or "").lower()
        source = (doc.metadata.get("source") or "").lower()
        if any(k in content for k in content_keywords) or any(k in source for k in source_keywords):
            filtered.append(doc)

    return filtered if filtered else docs

def _contains_markdown_table(text: str) -> bool:
    return any(
        '|' in line and line.strip().startswith('|')
        for line in text.strip().split('\n')
    )

# [FIX 1]: Narrowed to facts, excluded recommendations like "might" or "could be"
def _contains_speculation(text: str) -> bool:
    return bool(re.search(r"\b(likely|possibly|probably|appears to be|seems to be)\b", text, re.IGNORECASE))

def _remove_speculative_sentences(text: str) -> str:
    if not text or not text.strip():
        return text
    parts = re.split(r'(?<=[.!?])\s+', text.strip())
    kept = []
    for sentence in parts:
        if not sentence.strip():
            continue
        if _contains_speculation(sentence):
            continue
        kept.append(sentence.strip())
    return " ".join(kept).strip()

# [FIX 2]: Strip out messy internal extensions and underscores for public display
def _build_source_certainty_note(top_score: float, score_margin: float, sources: list[str]) -> str:
    unique_sources = list({(s or "Unknown").strip() for s in sources})

    if top_score >= HIGH_CONFIDENCE_THRESHOLD and score_margin >= HIGH_CONFIDENCE_MARGIN:
        level = "High"
    elif top_score >= LOW_CONFIDENCE_THRESHOLD:
        level = "Medium"
    else:
        level = "Low"

    def _clean_source_name(s: str) -> str:
        s = re.sub(r'[-_]', ' ', s)           
        s = re.sub(r'\.(pdf|md|txt|docx|csv|xlsx)$', '', s, flags=re.IGNORECASE)  
        return s.strip().title()

    clean_names = [_clean_source_name(s) for s in unique_sources[:2]]
    source_preview = ", ".join(clean_names) if clean_names else "retrieved documents"
    return f"> **Source certainty:** {level} — based on {len(unique_sources)} document(s): *{source_preview}*"

def _detect_query_intent(query: str) -> str:
    q_lower = (query or "").lower()
    tokens = set(re.findall(r"[a-z0-9']+", q_lower))

    def has_any_words(words: set[str]) -> bool:
        return any(word in tokens for word in words)

    def has_any_phrases(phrases: list[str]) -> bool:
        return any(phrase in q_lower for phrase in phrases)

    if has_any_words({"where", "room", "building", "located", "floor", "lab", "office", "directory"}):
        return "location"
    if has_any_words({"curriculum", "course", "subject", "semester", "units", "prerequisite"}):
        return "curriculum"
    if has_any_words({"who", "faculty", "chair", "dean", "professor", "staff", "instructor"}):
        return "people"
    if has_any_words({"download", "link", "pdf", "form", "access"}) or has_any_phrases(["google form", "download link"]):
        return "download"
    if has_any_words({"policy", "rule", "guideline", "procedure", "manual"}) or has_any_phrases(["dress code"]):
        return "policy"
    return "general"

def _is_no_answer_response(text: str) -> bool:
    if not text:
        return False
    patterns = [
        r"i couldn't find that in the available documents",
        r"i don't have enough info to answer that confidently",
        r"not explicitly stated in the retrieved documents",
        r"best to check with your department chair",
    ]
    lowered = text.lower()
    return any(re.search(p, lowered) for p in patterns)

# [FIX 4]: Room codes no longer get falsely flagged as "incomplete"
def _is_incomplete_query(query: str) -> bool:
    q_lower = (query or "").strip().lower()
    
    if re.search(r'\b[A-Z]{1,3}\d{3}\b', query.upper()):
        return False
        
    tokens = re.findall(r"[a-zA-Z0-9']+", q_lower)
    if len(tokens) <= 2:
        return True

    question_starters = {"what", "where", "who", "when", "why", "how", "which"}
    helper_verbs = {"is", "are", "can", "do", "does", "did", "show", "list", "find", "tell", "explain"}
    has_structure = any(t in question_starters or t in helper_verbs for t in tokens)

    return len(tokens) <= 4 and not has_structure

def _get_previous_user_query(chat_history_list: List[Dict[str, str]], current_query: str) -> str:
    if not chat_history_list:
        return ""
    current_norm = (current_query or "").strip().lower()
    for msg in reversed(chat_history_list):
        if msg.get("role") != "user":
            continue
        content = (msg.get("content") or "").strip()
        if not content:
            continue
        if content.lower() == current_norm:
            continue
        return content
    return ""

def _build_incomplete_query_variants(query: str, chat_history_list: List[Dict[str, str]]) -> List[str]:
    base = (query or "").strip()
    if not base:
        return []

    intent = _detect_query_intent(base)
    variants: List[str] = [base]

    if intent == "location":
        variants.append(f"{base} location building room directory")
    elif intent == "curriculum":
        variants.append(f"{base} curriculum course semester prerequisite")
    elif intent == "people":
        variants.append(f"{base} faculty staff role department professor instructor teacher")
    elif intent == "download":
        variants.append(f"{base} official link pdf form")
    elif intent == "policy":
        variants.append(f"{base} policy guideline rule procedure")

    previous_user_query = _get_previous_user_query(chat_history_list, base)
    if previous_user_query:
        variants.append(f"{previous_user_query} {base}")

    deduped = []
    seen = set()
    for variant in variants:
        norm = variant.strip().lower()
        if norm and norm not in seen:
            deduped.append(variant.strip())
            seen.add(norm)
    return deduped[:5]

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
    top_score = float("-inf")
    
    is_valid, clean_query = validate_query(query) 
    if not is_valid:
        yield clean_query
        return
    
    safe_query = redact_pii(clean_query) 
    standalone_query = contextualize_query(safe_query, chat_history_list)
    is_incomplete_input = _is_incomplete_query(standalone_query)
    
    if not is_incomplete_input:
        cached_answer = check_semantic_cache(standalone_query)
        if cached_answer:
            words = cached_answer.split(" ")
            chunk_size = 3
            for i in range(0, len(words), chunk_size):
                yield " ".join(words[i:i+chunk_size]) + " "
                time.sleep(0.01)
            return
    else:
        logger.info(f"Incomplete query detected: '{standalone_query}'. Skipping semantic cache for fresh closest-match retrieval.")

    retrieval_start = time.time()
    
    try:
        intent, _, _, _ = route_query(standalone_query)
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
        try:
            sub_queries = decompose_query(standalone_query)
        except:
            pass

    if is_incomplete_input:
        for variant in _build_incomplete_query_variants(standalone_query, chat_history_list):
            if variant not in sub_queries:
                sub_queries.append(variant)

    dynamic_k = get_dynamic_k(standalone_query)
    if is_incomplete_input:
        dynamic_k = max(dynamic_k, 20)
    retriever = get_retriever(k=dynamic_k)
    
    all_docs = []

    if len(sub_queries) == 1:
        all_docs = retriever.invoke(sub_queries[0])
    else:
        with concurrent.futures.ThreadPoolExecutor() as executor:
            results = list(executor.map(retriever.invoke, sub_queries))
        for res in results:
            all_docs.extend(res)

    if not all_docs:
        if is_incomplete_input:
            broad_queries = _build_incomplete_query_variants(standalone_query, chat_history_list)
            broad_retriever = get_retriever(k=max(dynamic_k, 30))
            with concurrent.futures.ThreadPoolExecutor() as executor:
                fallback_results = list(executor.map(broad_retriever.invoke, broad_queries))
            for res in fallback_results:
                all_docs.extend(res)

        logger.warning(f"⚠️ Vector Search returned 0 results for: {standalone_query}")
        if all_docs:
            logger.info(f"🔎 Incomplete-query fallback recovered {len(all_docs)} chunks using broad closest-match retrieval.")
        else:
            yield "Hmm, I couldn't find anything about that in the documents I have. Try asking your department chair!"
            return

    if not all_docs:
        yield "Hmm, I couldn't find anything about that in the documents I have. Try asking your department chair!"
        return

    logger.info(f"📂 Retrieval Success: Found {len(all_docs)} raw chunks using K={dynamic_k}")

    unique_docs_map = {hash(d.page_content): d for d in all_docs}
    latest_per_source = prefer_latest_per_source(list(unique_docs_map.values()))
    
    hybrid_results = hybrid_rerank(standalone_query, latest_per_source)
    if _detect_query_intent(standalone_query) == "people":
        hybrid_results = filter_to_people_docs(hybrid_results, standalone_query)
    
    is_curriculum_query = any(
        kw in standalone_query.lower()
        for kw in [
            'curriculum', 'subject', 'course', 'year', 'semester', 'units',
            'prerequisite', 'schedule', 'ojt', 'practicum', 'internship',
            'immersion', 'faculty', 'full-time', 'part-time', 'instructor',
            'professor', 'chairperson', 'chair', 'dean', 'department',
            'operating systems', 'elective', 'track'
        ]
    )
    
    ANALYTICAL_TRIGGERS = [
        'most', 'least', 'highest', 'lowest', 'compare', 'which course',
        'how many courses', 'most prerequisites', 'hardest', 'rank'
    ]
    is_analytical_query = any(kw in standalone_query.lower() for kw in ANALYTICAL_TRIGGERS)

    is_download_query = any(
        kw in standalone_query.lower()
        for kw in ['download', 'link', 'pdf', 'get the', 'access', 'where can i get', 'where can i download']
    )

    top_score = float("-inf")
    second_score = float("-inf")

    if is_analytical_query and is_curriculum_query:
        all_program_docs = filter_to_program(
            prefer_latest_per_source(list(unique_docs_map.values())),
            standalone_query
        )
        big_retriever = get_retriever(k=50)
        extra_docs = big_retriever.invoke(standalone_query)
        extra_filtered = filter_to_program(
            prefer_latest_per_source(extra_docs),
            standalone_query
        )
        combined = {hash(d.page_content): d for d in all_program_docs + extra_filtered}
        top_reranked = list(combined.values())
        
        if top_reranked:
            top_score = 10.0 
            
        logger.info(f"🔬 Analytical curriculum query — using {len(top_reranked)} full program chunks")
        
    else:
        # [FIX 6]: Ensure people intents get a broader slice of the context chunks
        query_intent = _detect_query_intent(standalone_query)
        if query_intent == "people":
            max_chunks = 6
        elif is_curriculum_query:
            max_chunks = 8
        else:
            max_chunks = 3
            
        hybrid_results = enforce_source_diversity(hybrid_results, max_per_source=max_chunks)

        if hybrid_results:
            try:
                pairs = [(standalone_query, doc.page_content) for doc in hybrid_results]
                scores = get_reranker().predict(pairs)
                sorted_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
                
                top_score = float(scores[sorted_indices[0]])
                if len(sorted_indices) > 1:
                    second_score = float(scores[sorted_indices[1]])
                
                top_reranked = [hybrid_results[i] for i in sorted_indices[:RERANKER_TOP_K]] 
            except Exception as e:
                logger.error(f"Reranking failed: {e}")
                top_reranked = hybrid_results[:RERANKER_TOP_K]
        else:
            top_reranked = []
            
    if is_download_query and top_reranked is not None:
        link_retriever = get_retriever(k=20)
        link_docs = link_retriever.invoke("official curriculum PDF download link")
        link_filtered = prefer_latest_per_source(link_docs)
        existing_hashes = {hash(d.page_content) for d in top_reranked}
        for doc in link_filtered:
            if hash(doc.page_content) not in existing_hashes:
                top_reranked.append(doc)
                existing_hashes.add(hash(doc.page_content))
        if top_reranked:
            top_score = max(top_score, 5.0)
        logger.info(f"🔗 Download query — injected {len(link_filtered)} link chunks")

    is_facility_query = any(
        kw in standalone_query.lower()
        for kw in [
            'room', 'building', 'floor', 'lab', 'laboratory', 
            'office', 'located', 'where is', 'where are', 'nasaan', 
            'campus', 'facility', 'location of'
        ]
    )

    if is_facility_query:
        already_has_directory = any(
            'directory' in doc.metadata.get('source', '').lower() or
            'campus' in doc.metadata.get('source', '').lower()
            for doc in top_reranked
        )
        if not already_has_directory:
            building_code_match = re.search(r'\b([A-Z]{1,3})\d{3}\b', standalone_query.upper())
            if building_code_match:
                building_code = building_code_match.group(1)
                directory_query = f"{building_code} building rooms directory"
            else:
                directory_query = "campus building directory rooms"

            directory_retriever = get_retriever(k=8)
            directory_docs = directory_retriever.invoke(directory_query)
            directory_filtered = prefer_latest_per_source(directory_docs)
            existing_hashes = {hash(d.page_content) for d in top_reranked}
            for doc in directory_filtered:
                if hash(doc.page_content) not in existing_hashes:
                    top_reranked.append(doc)
                    existing_hashes.add(hash(doc.page_content))
            if top_reranked:
                top_score = max(top_score, 5.0)
            logger.info(f"🏢 Facility query — building code: '{building_code_match.group(1) if building_code_match else 'generic'}', injected {len(directory_filtered)} chunks")
    
    logger.info(f"📊 Query: '{standalone_query}'")
    logger.info(f"📊 Docs retrieved: {len(all_docs)} → after rerank: {len(top_reranked)}")
    logger.info(
        f"📊 Top score: {top_score:.2f} | Second: {second_score:.2f} | "
        f"Margin: {(top_score - second_score):.2f} "
        f"(Cutoffs — low: {LOW_CONFIDENCE_THRESHOLD}, high: {HIGH_CONFIDENCE_THRESHOLD}, "
        f"margin: {HIGH_CONFIDENCE_MARGIN})"
    )
    
    logger.info(f"📄 Final context sources ({len(top_reranked)} chunks): {[doc.metadata.get('source','?') for doc in top_reranked]}")

    context_pieces = [f"[[Source: {doc.metadata.get('source', 'Unknown')}]\n{doc.page_content}" for doc in top_reranked]
    context = "\n\n".join(context_pieces)
    st.session_state["last_retrieved_context"] = context
    
    retrieval_time = time.time() - retrieval_start
    gen_start = time.time()
    
    # ── FIX: SINGLE-SHOT SUGGESTIONS IN PROMPT ──
    prompt = f"""You are AXIsstant, the friendly and helpful Academic AI assistant of Ateneo de Naga University's College of Science, Engineering, and Architecture (CSEA). You help students and faculty with academic questions in a warm, conversational tone — like a knowledgeable kuya or ate who actually wants to help.

Answer the question using ONLY the context below.

### RULES (FOLLOW STRICTLY):

1. **TONE**: Write like a friendly, approachable upperclassman helping a 
   classmate — casual but still accurate. Use natural conversational 
   language. Contractions are fine ("you'll", "it's", "here's"). 
   Never sound like a formal document or a customer service bot.
   Do NOT start with "Great question!", "Good question!", or any 
   sycophantic opener. Just talk like a normal person would. Do NOT start with "So,", "Kuya", "So here's", "So to answer", or any 
   sentence that begins with the word "So". Make every response different — do NOT use a template. Avoid repeating sentence starters across different responses. 
   Do NOT use the same opening phrase more than once in a 5-response span.

2. **NO FILLER**: Do NOT say "To provide you with..." or "I'll need to refer to..." or "Let me check the handbook for you..." or any variation.

3. **LANGUAGE**: Always respond in English unless the student writes in Filipino, in which case respond in Filipino. 

4. **USE TABLES FOR STRUCTURED DATA**: When the context contains curriculum 
   subjects, grading scales, schedules, or faculty lists, reproduce the ACTUAL 
   data in a Markdown table. Include specific course codes, titles, units, and 
   prerequisites. **SHOW EVERY ROW** — never truncate or skip rows. 
   If the source data has rows where every cell is "---", skip those rows 
   entirely — they are visual dividers in the original document and must NOT 
   appear in your markdown table output.

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

12. **LINKS**: Never paste raw URLs. Always format links as descriptive 
    markdown like [Download the official curriculum here](url) or 
    [Fill out the form here](url). The link text should describe what 
    the link does, not the URL itself.

13. **ROOM CODES**: When the user asks about a room like "D412", "AL112", 
    or "EB111", look up the building prefix in the campus directory in 
    the context. "D" = Fr. Francis Dolan SJ Building, "AL" = Godofredo 
    Alingal SJ Building, "EB" = Engineering Building, "AR" = Fr. Pedro 
    Arrupe SJ Building, "P" = Fr. John Phelan SJ Building, "S" = Fr. 
    Pedro Santos SJ Building, "B" = Fr. Francis Burns SJ Building, 
    "CC" = Covered Courts, "RB" = Fr. Raul Bonoan SJ Building.
    Always decode the building name, floor number, and room number for 
    the user. Only use acronyms confirmed in the context.

14. **NO SPECULATION**: Never guess or infer missing details. Do not use speculative words like "likely", "possibly", "probably", "might", or "could be". If the context does not explicitly state a detail, clearly say it is not explicitly stated in the retrieved documents.

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
            is_curriculum_query or 
            is_facility_query or 
            is_analytical_query or
            is_download_query or
            is_incomplete_input
        )

        if top_score < LOW_CONFIDENCE_THRESHOLD and not is_protected_query:
            logger.warning(f"🔇 Low Retrieval Score ({top_score:.2f}). Aborting generation.")
            yield "I don't have enough info to answer that confidently — best to check with your department chair directly."
            return

        llm = get_generator_llm()
        draft_raw = get_llm_response(llm, prompt).content
        
        # ── EXTRACT SUGGESTIONS (BEFORE CRITIC SEES IT) ──
        SUGGESTION_SPLIT = re.compile(
            r'\n+SUGGESTED_QUESTIONS:\s*\n((?:\d+\..+\n?){1,3})',
            re.IGNORECASE
        )
        suggestion_match = SUGGESTION_SPLIT.search(draft_raw)
        suggested_questions = []

        if suggestion_match:
            draft_response = draft_raw[:suggestion_match.start()].strip()
            raw_lines = suggestion_match.group(1).strip().split('\n')
            for line in raw_lines:
                q = re.sub(r'^\d+\.\s*', '', line).strip()
                if q and len(q) > 8:
                    if not q.endswith('?'):
                        q += '?'
                    suggested_questions.append(q)
        else:
            draft_response = draft_raw.strip()

        # ── CRITIC / CONFIDENCE GATE ──
        score_margin = top_score - second_score if second_score != float("-inf") else top_score
        high_confidence = (
            top_score >= HIGH_CONFIDENCE_THRESHOLD
            and score_margin >= HIGH_CONFIDENCE_MARGIN
        )

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

        if not final_verified_response:
            logger.error("final_verified_response is None. Falling back to draft.")
            final_verified_response = draft_response

        if _contains_speculation(final_verified_response):
            cleaned_non_speculative = _remove_speculative_sentences(final_verified_response)
            if cleaned_non_speculative:
                final_verified_response = cleaned_non_speculative
            else:
                final_verified_response = "I couldn't find an explicit answer for that detail in the retrieved documents."

        final_verified_response = fix_markdown_tables(final_verified_response)
        final_verified_response = re.sub(r'\s+(\d+\.\s)', r'\n\1', final_verified_response)
        final_verified_response = format_raw_links(final_verified_response)

        # ── CACHING (FIX 3): Cache the clean response before metadata/suggestions ──
        clean_response_for_cache = final_verified_response
        add_to_cache(standalone_query, clean_response_for_cache)

        # ── APPEND METADATA & SUGGESTIONS FOR STREAMING ──
        source_list = [doc.metadata.get('source', 'Unknown') for doc in top_reranked]
        certainty_note = _build_source_certainty_note(top_score, score_margin, source_list)
        final_verified_response += f"\n\n{certainty_note}"
        
        if suggested_questions:
            suggestions_md = "\n\n---\n**You might also want to ask:**\n" + \
                "\n".join(f"- {q}" for q in suggested_questions[:3])
            final_verified_response += suggestions_md
        
        # ── STREAMING ──
        try:
            if _contains_markdown_table(final_verified_response):
                yield final_verified_response
            else:
                words = final_verified_response.split(" ")
                chunk_size = 3
                for i in range(0, len(words), chunk_size):
                    yield " ".join(words[i:i+chunk_size]) + " "
                    time.sleep(STREAM_DELAY) 
                    
        except GeneratorExit:
            return
        except Exception as e:
            logger.error(f"Streaming interruption: {e}")
            yield f"\n\n⚠️ *Stream interrupted. Displaying full response:* \n{final_verified_response}"
            
    except Exception as e:
        logger.error(f"❌ Generation Pipeline Failed: {e}")
        yield "Something went wrong on my end. Give it another try in a bit!"


def fix_markdown_tables(text: str) -> str:
    if '|' not in text:
        return text
    
    # [FIX 5]: Removed the stray indent before the comment
    # ── NEW: Remove decorative dash-only rows from LLM output ──
    # These come from source documents that use --- as visual dividers.
    # In markdown tables they render as broken separator rows.
    def _strip_decorative_dash_rows(t: str) -> str:
    cleaned = []
    for line in t.split('\n'):
        stripped = line.strip()
        if stripped.startswith('|') and stripped.endswith('|'):
            cells = [c.strip() for c in stripped.strip('|').split('|')]
            # Skip rows where every cell is ONLY dashes (1 or more) or empty
            # But preserve real separator rows (those between header and data)
            all_dashes = all(re.fullmatch(r'-+', cell) for cell in cells if cell)
            has_empty = any(cell == '' for cell in cells)
            if all_dashes and not has_empty:
                continue  # decorative row — skip it
        cleaned.append(line)
    return '\n'.join(cleaned)

    text = _strip_decorative_dash_rows(text)
        
    lines = text.split('\n')
    fixed = []
    i = 0
    while i < len(lines):
        line = lines[i]
        next_line = lines[i + 1] if i + 1 < len(lines) else ''

        if (line.strip().startswith('|') and
                fixed and
                fixed[-1].strip() and
                not fixed[-1].strip().startswith('|')):
            fixed.append('')

        fixed.append(line)

        next_next_line = lines[i + 2] if i + 2 < len(lines) else ''
        is_header_candidate = (
            line.strip().startswith('|') and
            next_line.strip().startswith('|') and
            '---' not in next_line and
            '---' not in line and
            not any('---' in f for f in fixed[-3:])
        )
        if is_header_candidate:
            col_count = line.count('|') - 1
            if col_count > 0:
                fixed.append('|' + '---|' * col_count)

        if (line.strip().startswith('|') and
                next_line.strip() and
                not next_line.strip().startswith('|')):
            fixed.append('')

        i += 1
    return '\n'.join(fixed)


def format_raw_links(text: str) -> str:
    raw_url_pattern = re.compile(
        r'(?<!\()'
        r'(https?://[^\s\)\]\,\"\']+)'
    )
    def replace_url(match):
        url = match.group(1)
        full_text = text
        pos = match.start()
        preceding = full_text[max(0, pos-50):pos]
        if re.search(r'\[[^\]]*\]\($', preceding):
            return url 
            
        if 'supabase' in url or 'storage' in url:
            label = 'Download here'
        elif 'form' in url or 'docs.google' in url:
            label = 'Access the form here'
        elif 'drive.google' in url:
            label = 'View document here'
        else:
            label = 'View link here'
        return f'[{label}]({url})'
    return raw_url_pattern.sub(replace_url, text)


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def get_llm_response(llm, prompt):
    return llm.invoke(prompt)