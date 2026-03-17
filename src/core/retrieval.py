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

def _contains_markdown_table(text: str) -> bool:
    return any(
        '|' in line and line.strip().startswith('|')
        for line in text.strip().split('\n')
    )


def _contains_speculation(text: str) -> bool:
    return bool(re.search(r"\b(likely|possibly|probably|maybe|might|could be|appears to be|seems to be)\b", text, re.IGNORECASE))


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


def _build_source_certainty_note(top_score: float, score_margin: float, sources: list[str]) -> str:
    unique_sources = []
    seen = set()
    for source in sources:
        s = (source or "Unknown").strip()
        if s not in seen:
            unique_sources.append(s)
            seen.add(s)

    if top_score >= HIGH_CONFIDENCE_THRESHOLD and score_margin >= HIGH_CONFIDENCE_MARGIN:
        level = "High"
    elif top_score >= LOW_CONFIDENCE_THRESHOLD:
        level = "Medium"
    else:
        level = "Low"

    source_preview = ", ".join(unique_sources[:2]) if unique_sources else "retrieved documents"
    return f"> **Source certainty:** {level} ({len(unique_sources)} source file(s); based on: {source_preview})"


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


def _generate_recovery_questions(query: str) -> list[str]:
    intent = _detect_query_intent(query)
    if intent == "location":
        return [
            "Can I share a room code (like EB213) so you can check it directly?",
            "Can you search by building name instead of the lab name?",
            "Can you list nearby rooms from the same directory section?",
        ]
    if intent == "curriculum":
        return [
            "Can you focus on one program and year level first?",
            "Can you show the full semester list or just one course?",
            "Can you check prerequisites for a specific course code?",
        ]
    if intent == "people":
        return [
            "Can you check faculty, staff, or chairperson information?",
            "Can you match the exact person name in the records?",
            "Can you search by department instead of person name?",
        ]
    if intent == "download":
        return [
            "Can you check which document this download link is for?",
            "Can you share the official PDF link or form link?",
            "Can you look for the latest posted version only?",
        ]
    if intent == "policy":
        return [
            "Can you check a specific policy area first?",
            "Can you show the rule details, exceptions, or procedures?",
            "Can you point me to the exact policy line and section title?",
        ]
    return [
        "Can you narrow this using a more specific keyword?",
        "Can you filter this by department or document type?",
        "Can you point me to the exact source line and section for this?",
    ]


def _is_incomplete_query(query: str) -> bool:
    q_lower = (query or "").strip().lower()
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
        variants.append(f"{base} faculty staff role department")
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
        max_chunks = 8 if is_curriculum_query else 3
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

    # ── FIX: DIRECTORY EXTRACTION LOGIC ──
    if is_facility_query:
        already_has_directory = any(
            'directory' in doc.metadata.get('source', '').lower() or
            'campus' in doc.metadata.get('source', '').lower()
            for doc in top_reranked
        )
        if not already_has_directory:
            # Extract building code from query (e.g. "D412" → "D", "AL112" → "AL", "EB111" → "EB")
            building_code_match = re.search(r'\b([A-Z]{1,3})\d{3}\b', standalone_query.upper())
            if building_code_match:
                building_code = building_code_match.group(1)
                # Search specifically for that building code + "rooms" or "building"
                directory_query = f"{building_code} building rooms directory"
            else:
                directory_query = "campus building directory rooms"

            directory_retriever = get_retriever(k=8)  # was k=5, increase to cast wider net
            directory_docs = directory_retriever.invoke(directory_query)
            directory_filtered = prefer_latest_per_source(directory_docs)
            existing_hashes = {hash(d.page_content) for d in top_reranked}
            for doc in directory_filtered:
                if hash(doc.page_content) not in existing_hashes:
                    top_reranked.append(doc)
                    existing_hashes.add(hash(doc.page_content))
            if top_reranked:
                top_score = max(top_score, 5.0)  # prevent Tier 1 kill
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
    
    # ── FIX: UPDATED RULE 13 WITH EXPLICIT CODES ──
    prompt = f"""You are AXIsstant, the friendly and helpful Academic AI assistant of Ateneo de Naga University's College of Science, Engineering, and Architecture (CSEA). You help students and faculty with academic questions in a warm, conversational tone — like a knowledgeable kuya or ate who actually wants to help.

Answer the question using ONLY the context below.

### RULES (FOLLOW STRICTLY):

1. **TONE**: Write like a friendly, approachable upperclassman helping a 
   classmate — casual but still accurate. Use natural conversational 
   language. Contractions are fine ("you'll", "it's", "here's"). 
   Never sound like a formal document or a customer service bot.
   Do NOT start with "Great question!", "Good question!", or any 
   sycophantic opener. Just talk like a normal person would. Do NOT start with "So,", "So here's", "So to answer", or any 
   sentence that begins with the word "So". Start directly with 
   a greeting, and the answer.

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

**Answer:**"""

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
        draft_response = get_llm_response(llm, prompt).content
        
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

        # Hard block speculative language that can mislead users.
        if _contains_speculation(final_verified_response):
            cleaned_non_speculative = _remove_speculative_sentences(final_verified_response)
            if cleaned_non_speculative:
                final_verified_response = cleaned_non_speculative
            else:
                final_verified_response = "I couldn't find an explicit answer for that detail in the retrieved documents."

        final_verified_response = fix_markdown_tables(final_verified_response)
        final_verified_response = re.sub(r'\s+(\d+\.\s)', r'\n\1', final_verified_response)
        final_verified_response = format_raw_links(final_verified_response)

        source_list = [doc.metadata.get('source', 'Unknown') for doc in top_reranked]
        score_margin = top_score - second_score if second_score != float("-inf") else top_score
        certainty_note = _build_source_certainty_note(top_score, score_margin, source_list)
        final_verified_response = f"{final_verified_response}\n\n{certainty_note}"
        
        gen_time = time.time() - gen_start
        if _is_no_answer_response(final_verified_response):
            suggested_questions = _generate_recovery_questions(standalone_query)
        else:
            suggested_questions = generate_suggested_questions(
                standalone_query, final_verified_response, context,
                skip_llm=(gen_time > 8.0)
            )
        
        if suggested_questions:
            suggestions_md = "\n\n---\n**You might also want to ask:**\n" + \
                "\n".join(f"- {q}" for q in suggested_questions)
            final_verified_response += suggestions_md
            
        clean_response_for_cache = re.sub(
            r'\n\n---\n\*\*You might also want to ask:\*\*\n(?:- .+\n?)*',
            '',
            final_verified_response
        ).strip()
        add_to_cache(standalone_query, clean_response_for_cache)
        
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

def generate_suggested_questions(query: str, response: str, context: str, skip_llm: bool = False) -> list[str]:
    STRONG_STOPWORDS = {
        "what", "which", "where", "when", "why", "how", "this", "that", "those", "these",
        "from", "with", "about", "into", "within", "there", "their", "your", "have", "has",
        "does", "doing", "can", "could", "would", "should", "will", "just", "same", "topic",
        "context", "specific", "say", "says", "details", "detail", "entry", "source", "document",
    }

    def _token_set(text: str) -> set[str]:
        return {
            tok for tok in re.findall(r"[a-zA-Z0-9']+", (text or "").lower())
            if len(tok) >= 3
        }

    def _detect_followup_intent(q: str) -> str:
        return _detect_query_intent(q)

    def _extract_query_focus(original_query: str) -> str:
        q = (original_query or "").strip()
        tokens = re.findall(r"[A-Za-z0-9']+", q)
        if not tokens:
            return "this topic"

        ignore = {
            "what", "where", "when", "why", "how", "which", "who",
            "this", "that", "these", "those",
            "is", "are", "was", "were", "do", "does", "did", "can", "could", "would", "should",
            "the", "a", "an", "about", "for", "to", "of", "in", "on", "at",
            "i", "me", "my", "we", "us", "our", "you", "your", "please",
            "explain", "show", "tell", "share", "check", "find", "list",
            "check", "latest", "entry", "entries", "topic", "exact", "line", "source", "section",
        }

        generic_focus_terms = {
            "handbook", "chapter", "manual", "document", "documents", "info", "information", "details",
            "detail", "record", "records", "update", "updates", "requirement", "requirements",
        }

        kept = [t for t in tokens if t.lower() not in ignore]
        meaningful = [t for t in kept if t.lower() not in generic_focus_terms]
        if meaningful:
            kept = meaningful

        if not kept:
            kept = tokens

        if all(t.lower() in ignore or t.lower() in generic_focus_terms for t in kept):
            return "this topic"

        focus = " ".join(kept[:3]).strip()
        return focus if focus else "this topic"

    def _choose_best_anchor(query_focus: str, anchors: list[str]) -> str:
        if query_focus and query_focus != "this topic":
            return query_focus

        preferred = []
        for anchor in anchors:
            a = (anchor or "").strip()
            lower = a.lower()
            if not a:
                continue
            if len(a.split()) < 2:
                continue
            if len(a) > 45:
                continue
            if any(x in lower for x in ["chapter", "handbook", ".pdf", "source"]):
                continue
            preferred.append(a)

        if preferred:
            return preferred[0]
        if anchors:
            return anchors[0]
        return "this topic"

    def _intent_fallback_templates(intent: str, anchors: list[str], source_codes: set[str], query_focus: str) -> list[str]:
        anchor = _choose_best_anchor(query_focus, anchors)
        code = sorted(source_codes)[0] if source_codes else None

        if intent == "location":
            qs = [
                f"Can you point me to the source entry that mentions {anchor}?",
                f"Where is {anchor} located (building and room)?",
                "Can you list nearby room or lab entries in the same building section?",
            ]
        elif intent == "curriculum":
            qs = [
                f"Can you check the prerequisite details for {code or anchor}?",
                f"Where can I find {code or anchor} across semesters?",
                "Can you list related courses in the same curriculum section?",
            ]
        elif intent == "people":
            qs = [
                f"Who is listed for {anchor} and what role do they have?",
                f"Who else is listed with {anchor} in the same section?",
                "Can you point me to the source line for this personnel detail?",
            ]
        elif intent == "download":
            qs = [
                "Can you point me to the exact source with the official download link?",
                f"What document is that link for in the same source as {anchor}?",
                "Are there related forms or files in that same source section?",
            ]
        elif intent == "policy":
            qs = [
                f"What condition or exception is stated for {anchor}?",
                "Can you show the exact procedure steps in the same source section?",
                "Can you point me to the source line for this rule?",
            ]
        else:
            qs = [
                f"Can you point me to the exact source line about {anchor}?",
                f"Can you show me related details listed with {anchor} in the same section?",
                f"Where can I find the latest handbook update for {anchor}?",
            ]

        return qs

    def _context_fill_templates(intent: str, anchor: str, source_codes: set[str]) -> list[str]:
        code = sorted(source_codes)[0] if source_codes else None
        if intent == "location":
            return [
                f"What nearby entry is listed closest to {anchor} in the same section?",
                f"Is there a floor or landmark detail listed with {anchor}?",
                f"Can you point me to the source line for {anchor}'s location details?",
            ]
        if intent == "curriculum":
            return [
                f"Which semester entry is closest to {code or anchor} in the same table?",
                f"What prerequisite note is listed alongside {code or anchor}?",
                f"Can you point me to the source line for {code or anchor}'s curriculum detail?",
            ]
        if intent == "people":
            return [
                f"Which section lists {anchor} with role details?",
                f"What related personnel detail appears near {anchor}?",
                f"Can you point me to the exact source line for {anchor}'s role?",
            ]
        if intent == "download":
            return [
                f"Which section lists the official file linked with {anchor}?",
                f"Is there a newer version note for the {anchor} document?",
                "Can you point me to the exact source line with the active download link?",
            ]
        if intent == "policy":
            return [
                f"Which section title contains the rule related to {anchor}?",
                f"What exception note is listed with {anchor} in the same section?",
                f"Can you point me to the source line for {anchor}'s procedure detail?",
            ]
        return [
            f"Where in the handbook is {anchor} mentioned?",
            f"Can you show me the update note listed with {anchor} in that section?",
            f"Which chapter should I check first for the latest {anchor} details?",
        ]

    def _meaningful_tokens(text: str) -> set[str]:
        return {t for t in _token_set(text) if t not in STRONG_STOPWORDS}

    def _is_query_aligned(q: str, original_query: str) -> bool:
        q_tokens = _meaningful_tokens(q)
        query_tokens = _meaningful_tokens(original_query)
        if not query_tokens:
            return True
        return len(q_tokens.intersection(query_tokens)) >= 1

    def _is_intent_aligned(q: str, intent: str) -> bool:
        q_lower = (q or "").lower()
        if intent == "location":
            if any(x in q_lower for x in ["chapter", "handbook", "manual"]):
                return False
            return any(x in q_lower for x in ["where", "room", "building", "floor", "location", "located", "directory", "nearby"])
        if intent == "curriculum":
            return any(x in q_lower for x in ["course", "subject", "semester", "units", "prerequisite", "curriculum"])
        if intent == "people":
            return any(x in q_lower for x in ["who", "faculty", "staff", "role", "instructor", "chair", "dean", "professor"])
        if intent == "download":
            return any(x in q_lower for x in ["link", "download", "pdf", "file", "form", "access"])
        if intent == "policy":
            return any(x in q_lower for x in ["rule", "policy", "guideline", "procedure", "exception", "condition"])
        return True

    def _jaccard(a: set[str], b: set[str]) -> float:
        if not a or not b:
            return 0.0
        return len(a.intersection(b)) / max(1, len(a.union(b)))

    def _normalize_question(q: str) -> str:
        cleaned = re.sub(r"\b(fr|mr|ms|mrs|dr)\.\s+", "", q or "", flags=re.IGNORECASE)
        cleaned = re.sub(r"[^a-zA-Z0-9\s]", " ", cleaned.lower())
        cleaned = re.sub(r"\b(is|are|was|were|in|the|a|an|to|of|for|on|at|about|directory|campus|adnu)\b", " ", cleaned)
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        return cleaned

    def _is_redundant_candidate(q: str, existing: list[str]) -> bool:
        q_norm = _normalize_question(q)
        q_tokens = _token_set(q_norm)
        if not q_norm or len(q_tokens) < 2:
            return True
        for prev in existing:
            prev_norm = _normalize_question(prev)
            if q_norm == prev_norm:
                return True
            if _jaccard(q_tokens, _token_set(prev_norm)) >= 0.62:
                return True
        return False

    def _is_shallow_rephrase(q: str, original_query: str) -> bool:
        q_lower = (q or "").strip().lower()
        yes_no_starters = (
            "is ", "are ", "was ", "were ", "do ", "does ", "did ",
            "can ", "could ", "will ", "would ", "has ", "have ", "had "
        )
        if _normalize_question(q_lower) == _normalize_question(original_query):
            return True

        similarity = _jaccard(_token_set(q_lower), _token_set(original_query))
        if similarity >= 0.70:
            return True

        if not q_lower.startswith(yes_no_starters):
            return False
        return similarity >= 0.45

    def _remove_question_lead(q: str) -> str:
        q_clean = (q or "").strip().lower()
        q_clean = q_clean.rstrip("? ")
        q_clean = re.sub(
            r"^(what|which|who|where|when|why|how)\s+(is|are|was|were|does|do|did|can|could|will|would|has|have|had)\s+",
            "",
            q_clean,
        )
        q_clean = re.sub(r"^(what|which|who|where|when|why|how)\s+", "", q_clean)
        q_clean = re.sub(r"^(tell me|explain|describe|list)\s+", "", q_clean)
        return re.sub(r"\s+", " ", q_clean).strip()

    def _already_answered_by_response(q: str, response_text: str) -> bool:
        if not response_text.strip():
            return False

        q_core = _remove_question_lead(q)
        q_tokens = _token_set(q_core)
        resp_tokens = _token_set(response_text)

        if len(q_tokens) < 2:
            return True

        coverage = len(q_tokens.intersection(resp_tokens)) / max(1, len(q_tokens))
        if coverage >= 0.8:
            return True

        response_lines = [line.strip() for line in response_text.splitlines() if line.strip()]
        q_norm_tokens = _token_set(_normalize_question(q_core))
        if not q_norm_tokens:
            return False

        for line in response_lines:
            line_tokens = _token_set(_normalize_question(line))
            if not line_tokens:
                continue
            if _jaccard(q_norm_tokens, line_tokens) >= 0.62:
                return True

        return False

    def _adds_related_context_value(q: str, response_text: str, context_text: str) -> bool:
        q_tokens = _meaningful_tokens(q)
        if len(q_tokens) < 2:
            return False

        resp_tokens = _meaningful_tokens(response_text)
        ctx_tokens = _meaningful_tokens(context_text)
        ctx_extra = ctx_tokens - resp_tokens

        # Must stay within current message topic
        topic_overlap = len(q_tokens.intersection(resp_tokens))
        if topic_overlap < 1:
            return False

        # Must add at least one related detail from retrieved context not already explicit in response
        if ctx_extra:
            return len(q_tokens.intersection(ctx_extra)) >= 1

        # If no extra context terms exist, require stronger overlap with response for strict relevance
        return len(q_tokens.intersection(resp_tokens)) >= 2

    def _extract_anchor_phrases(source_text: str) -> list[str]:
        anchors = []

        def _normalize_anchor_phrase(value: str) -> str:
            cleaned = re.sub(r"\s+", " ", (value or "")).strip()
            cleaned = re.sub(r"^(the|a|an)\s+", "", cleaned, flags=re.IGNORECASE)
            cleaned = re.sub(r"\s+(the|a|an)$", "", cleaned, flags=re.IGNORECASE)
            return cleaned.strip()

        source_names = re.findall(r"\[\[Source:\s*([^\]]+)\]", source_text)
        for src in source_names:
            name = re.sub(r"\.(pdf|md|txt|docx?)$", "", src.strip(), flags=re.IGNORECASE)
            name = re.sub(r"[_-]+", " ", name).strip()
            if 4 <= len(name) <= 70:
                anchors.append(name)

        for code in re.findall(r"\b[A-Z]{2,5}\d{3}\b", source_text.upper()):
            anchors.append(code)

        title_like = re.findall(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,4}\b", source_text)
        for phrase in title_like:
            cleaned = _normalize_anchor_phrase(phrase)
            if 6 <= len(cleaned) <= 50:
                anchors.append(cleaned)

        deduped = []
        seen = set()
        for item in anchors:
            key = item.lower()
            if key not in seen:
                deduped.append(item)
                seen.add(key)
            if len(deduped) == 15:
                break
        return deduped

    def _extract_tokens(text: str) -> set[str]:
        return {
            tok for tok in re.findall(r"[a-zA-Z0-9']+", text.lower())
            if len(tok) >= 4
        }

    def _extract_course_codes(text: str) -> set[str]:
        return set(re.findall(r"\b[A-Z]{2,5}\d{3}\b", text.upper()))

    def _cleanup_question(q: str) -> str:
        cleaned = re.sub(r"\s+", " ", str(q or "")).strip()
        if not cleaned:
            return ""
        if not cleaned.endswith("?"):
            cleaned += "?"
        return cleaned

    def _contains_anchor(q: str, anchors: list[str]) -> bool:
        q_lower = q.lower()
        return any(anchor.lower() in q_lower for anchor in anchors)

    def _is_answerable_question(
        q: str,
        source_tokens: set[str],
        source_codes: set[str],
        anchors: list[str],
        original_query: str,
        response_text: str,
        context_text: str,
        intent: str,
    ) -> bool:
        if len(q) < 12 or len(q) > 140:
            return False

        if _is_shallow_rephrase(q, original_query):
            return False

        if not _is_query_aligned(q, original_query):
            return False

        if not _is_intent_aligned(q, intent):
            return False

        q_codes = _extract_course_codes(q)
        if q_codes and not q_codes.issubset(source_codes):
            return False

        if anchors and not _contains_anchor(q, anchors):
            return False

        if not _adds_related_context_value(q, response_text, context_text):
            return False

        q_tokens = _extract_tokens(q)
        overlap = q_tokens.intersection(source_tokens)
        return len(overlap) >= 3

    def _fallback_questions(source_text: str, anchors: list[str], source_codes: set[str], query_focus: str) -> list[str]:
        intent = _detect_followup_intent(query)
        fallbacks = _intent_fallback_templates(intent, anchors, source_codes, query_focus)

        if not fallbacks and source_text.strip():
            fallbacks = _intent_fallback_templates("general", anchors, source_codes, query_focus)

        deduped = []
        seen = set()
        for q in fallbacks:
            normalized = q.lower().strip()
            if normalized not in seen:
                deduped.append(q)
                seen.add(normalized)
            if len(deduped) == 3:
                break
        return deduped

    source_text = context if context.strip() else f"{response}\n{context}"
    strict_context = f"{response}\n{context}" if context.strip() else response
    anchors = _extract_anchor_phrases(source_text)
    source_tokens = _extract_tokens(source_text)
    source_codes = _extract_course_codes(source_text)
    followup_intent = _detect_followup_intent(query)
    query_focus = _extract_query_focus(query)
    best_anchor = _choose_best_anchor(query_focus, anchors)
    
    if skip_llm:
        quick_candidates = _fallback_questions(source_text, anchors, source_codes, query_focus)
        quick_validated: list[str] = []
        for q in quick_candidates:
            q_clean = _cleanup_question(q)
            if not q_clean:
                continue
            if _is_redundant_candidate(q_clean, quick_validated):
                continue
            if _is_shallow_rephrase(q_clean, query):
                continue
            if not _is_query_aligned(q_clean, query):
                continue
            if not _is_intent_aligned(q_clean, followup_intent):
                continue
            quick_validated.append(q_clean)

        if len(quick_validated) < 2:
            for q in _generate_recovery_questions(query):
                q_clean = _cleanup_question(q)
                if not q_clean:
                    continue
                if _is_redundant_candidate(q_clean, quick_validated):
                    continue
                if _is_shallow_rephrase(q_clean, query):
                    continue
                if not _is_intent_aligned(q_clean, followup_intent):
                    continue
                quick_validated.append(q_clean)
                if len(quick_validated) >= 3:
                    break

        if len(quick_validated) < 3:
            for q in _context_fill_templates(followup_intent, best_anchor, source_codes):
                q_clean = _cleanup_question(q)
                if not q_clean:
                    continue
                if _is_redundant_candidate(q_clean, quick_validated):
                    continue
                if _is_shallow_rephrase(q_clean, query):
                    continue
                if _already_answered_by_response(q_clean, response):
                    continue
                if not _is_intent_aligned(q_clean, followup_intent):
                    continue
                quick_validated.append(q_clean)
                if len(quick_validated) >= 3:
                    break

        return quick_validated[:3]

    try:
        llm = get_generator_llm()
        anchor_text = ", ".join(anchors[:10]) if anchors else "(none)"
        intent_rules = {
            "location": "Ask about explicit room/building/floor entries, neighboring entries, or source line evidence.",
            "curriculum": "Ask about prerequisites, semester placement, or closely related courses in the same section.",
            "people": "Ask about exact roles, associated personnel, or source evidence lines.",
            "download": "Ask about exact file/link target and related source-listed documents.",
            "policy": "Ask about explicit conditions, exceptions, steps, or enforcement details in the same source.",
            "general": "Ask for source-backed supporting detail, conditions, and related details from the same section.",
        }
        prompt = f"""Create exactly 3 short follow-up questions the assistant can answer
USING ONLY the provided response and context.

Hard constraints:
- Do not introduce new facts, entities, course codes, policies, or assumptions.
- Keep each question specific to details already present.
- Each question must include at least one exact anchor term from this list: {anchor_text}
    - Make each question go one level deeper (details, conditions, comparisons, or source evidence).
    - Do NOT produce yes/no rephrasings of the same fact from the answer.
    - Make all 3 questions clearly different from each other.
- Follow this intent policy for this query: {intent_rules.get(followup_intent, intent_rules['general'])}
- Keep each question under 16 words.
- Return ONLY a JSON array of 3 strings.
- No markdown, no numbering, no extra text.

Example output: ["Question 1?", "Question 2?", "Question 3?"]

User asked: {query}
Assistant answered: {response[:300]}
Available topic context: {context[:500]}

Output (JSON array only):"""
        
        result = llm.invoke(prompt).content.strip()
        result = re.sub(r'^```json|^```|```$', '', result.strip(), flags=re.MULTILINE).strip()
        raw_questions = json.loads(result)

        validated: list[str] = []
        seen = set()

        if isinstance(raw_questions, list):
            for q in raw_questions:
                cleaned = _cleanup_question(q)
                if not cleaned:
                    continue
                key = cleaned.lower()
                if key in seen:
                    continue
                if _is_redundant_candidate(cleaned, validated):
                    continue
                if _already_answered_by_response(cleaned, response):
                    continue
                if _is_answerable_question(
                    cleaned,
                    source_tokens,
                    source_codes,
                    anchors,
                    query,
                    response,
                    strict_context,
                    followup_intent,
                ):
                    validated.append(cleaned)
                    seen.add(key)
                if len(validated) == 3:
                    break

        if len(validated) < 3:
            for q in _fallback_questions(source_text, anchors, source_codes, query_focus):
                key = q.lower()
                if key not in seen:
                    if _is_redundant_candidate(q, validated):
                        continue
                    if _is_shallow_rephrase(q, query):
                        continue
                    if _already_answered_by_response(q, response):
                        continue
                    if not _adds_related_context_value(q, response, strict_context):
                        continue
                    if not _is_query_aligned(q, query):
                        continue
                    if not _is_intent_aligned(q, followup_intent):
                        continue
                    validated.append(q)
                    seen.add(key)
                if len(validated) == 3:
                    break

        if len(validated) < 2:
            rescue_candidates = _fallback_questions(source_text, anchors, source_codes, query_focus)
            rescue_candidates.extend(_generate_recovery_questions(query))
            for q in rescue_candidates:
                cleaned = _cleanup_question(q)
                if not cleaned:
                    continue
                key = cleaned.lower()
                if key in seen:
                    continue
                if _is_redundant_candidate(cleaned, validated):
                    continue
                if _is_shallow_rephrase(cleaned, query):
                    continue
                if not _is_intent_aligned(cleaned, followup_intent):
                    continue
                if not _is_query_aligned(cleaned, query):
                    continue
                validated.append(cleaned)
                seen.add(key)
                if len(validated) >= 3:
                    break

        if len(validated) < 3:
            for q in _context_fill_templates(followup_intent, best_anchor, source_codes):
                cleaned = _cleanup_question(q)
                if not cleaned:
                    continue
                key = cleaned.lower()
                if key in seen:
                    continue
                if _is_redundant_candidate(cleaned, validated):
                    continue
                if _is_shallow_rephrase(cleaned, query):
                    continue
                if _already_answered_by_response(cleaned, response):
                    continue
                if not _is_intent_aligned(cleaned, followup_intent):
                    continue
                validated.append(cleaned)
                seen.add(key)
                if len(validated) >= 3:
                    break

        return validated[:3]
    except Exception as e:
        logger.warning(f"Suggested questions failed: {e}")
    safe_fallback = _fallback_questions(source_text, anchors, source_codes, query_focus)
    if len(safe_fallback) < 2:
        safe_fallback.extend(_generate_recovery_questions(query))
    return safe_fallback[:3]