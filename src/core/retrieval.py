import json
import time
import numpy as np
import streamlit as st
import concurrent.futures
from typing import List, Dict, Set
from langchain_core.documents import Document
from rank_bm25 import BM25Okapi
import re
from tenacity import retry, stop_after_attempt, wait_exponential

from src.core.semantics import (
    tokenize, detect_query_intent, is_listing_query, is_people_list_query,
    is_curriculum_list_query, is_incomplete_query, build_incomplete_query_variants,
    is_custodian_lookup_query, is_custodian_list_query, normalize_lab_aliases,
    normalize_course_codes, expand_query_semantics
)
from src.core.guardrails import verify_answer, validate_query, redact_pii
from src.config.settings import get_generator_llm, get_embeddings, get_retriever
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
    DOCS_FOLDER,
    CAMPUS_MAP_URL,
)
from src.config.logging_config import logger

from src.core.reranking import (
    hybrid_rerank, enforce_source_diversity, filter_to_program,
    filter_to_people_docs, boost_people_list_docs, rank_people_list_docs,
    prefer_latest_per_source
)
from src.core.response_formatting import (
    fix_markdown_tables, format_raw_links, remove_speculative_sentences,
    build_source_certainty_note, fallback_questions, build_no_answer_response,
    is_no_answer_response, _contains_markdown_table, _contains_speculation
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

_CANONICALIZE_TRIGGERS = re.compile(
    r'\b(do you know|do u know|did you know|can you tell me|could you tell me|would you tell me|'
    r'i want to know|may i ask|do you happen to know)\b',
    re.IGNORECASE,
)

# ─────────────────────────────────────────────────────────────────────────────
# 1. CONVERSATIONAL MEMORY & CACHING FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────
def format_chat_history(messages: List[Dict[str, str]]) -> str:
    formatted_history = []
    history_to_process = messages[1:] if len(messages) > 1 else []
    
    for msg in history_to_process[-4:]: 
        role = "User" if msg["role"] == "user" else "Assistant"
        content = msg["content"].replace("{", "{{").replace("}", "}}")
        
        if role == "Assistant" and "|" in content and "---" in content:
            table_start = content.find("|")
            content = content[:table_start] + "\n... [Previous table truncated to preserve memory]"
            
        if len(content) > 500 and role == "Assistant":
            content = content[:500] + "..."

        formatted_history.append(f"{role}: {content}")
        
    return "\n".join(formatted_history) if formatted_history else "No previous context."

def contextualize_query(query: str, chat_history_list: List[Dict[str, str]]) -> str:
    history_to_process = chat_history_list[1:] if len(chat_history_list) > 1 else []
    has_history = bool(history_to_process)
    followup_context_trigger = has_history and bool(_CONTEXT_TRIGGERS.search(query))
    canonicalize_trigger = bool(_CANONICALIZE_TRIGGERS.search(query))
    if not (followup_context_trigger or canonicalize_trigger):
        return query
        
    history_text = format_chat_history(chat_history_list) if has_history else "No previous context."
    session_id = st.session_state.get("session_id", "default")
    cache_key = (session_id, query, history_text[-300:]) 
    if cache_key in _REWRITE_CACHE: return _REWRITE_CACHE[cache_key]
        
    prompt = f"""You are a query normalizer for an academic RAG assistant.
Rewrite the latest question into ONE canonical standalone question so semantically similar questions map to the same retrieval intent.

Rules:
- Do NOT answer the question.
- Preserve the original meaning, entities, course codes, and constraints.
- Remove conversational wrappers/fillers (e.g., "do you know", "did you know", "can you tell me").
- Resolve references using chat history when available.
- Normalize phrasing for consistency:
  * People/role lists -> "Who are the ...?"
  * Range/value fact queries -> "What is the ... range?"
  * Other factual queries -> use a direct factual question form.
- Return exactly one question and nothing else.

Examples:
- "did you know the chairpersons" -> "Who are the chairpersons in CSEA?"
- "do you know the range of CFRS" -> "What is the CFRS range?"

    Chat History:
    {history_text}

    Latest Question: {query}
    Canonical Standalone Question:"""
    
    try:
        llm = get_generator_llm()
        standalone_query = llm.invoke(prompt).content.strip()
        standalone_query = re.sub(r"^(?:canonical\s+)?standalone\s+question\s*:\s*", "", standalone_query, flags=re.IGNORECASE).strip()
        standalone_query = standalone_query.strip('"\'')
        standalone_query = re.sub(r"\s+", " ", standalone_query)
        if standalone_query and not standalone_query.endswith("?"):
            standalone_query += "?"
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
# 2. LOCAL RETRIEVAL HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _boost_curriculum_list_docs(query: str, docs: List[Document], base_k: int) -> List[Document]:
    if not is_curriculum_list_query(query):
        return docs

    boosted_queries = [
        f"{query} complete curriculum all semesters all subjects",
        f"{query} list all course codes titles units prerequisites",
        "complete curriculum 1st semester 2nd semester intersession subjects",
    ]

    boosted_docs = list(docs)
    curriculum_retriever = get_retriever(k=max(base_k, 50))
    try:
        with concurrent.futures.ThreadPoolExecutor() as executor:
            results = list(executor.map(curriculum_retriever.invoke, boosted_queries))
        for res in results:
            boosted_docs.extend(res)
    except Exception as e:
        logger.warning(f"Curriculum-list retrieval boost failed: {e}")

    boosted_docs = filter_to_program(boosted_docs, query)

    seen = set()
    deduped = []
    for doc in boosted_docs:
        key = hash(doc.page_content)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(doc)
    return deduped

def _boost_people_list_docs(query: str, docs: List[Document], base_k: int) -> List[Document]:
    if not is_people_list_query(query):
        return docs

    boosted_queries = [
        f"{query} full list of faculty members",
        f"{query} list of department chairs chairpersons",
        "CSEA faculty list full-time part-time instructors department chair dean",
        "Ateneo de Naga CSEA organizational structure faculty staff personnel chairpersons",
    ]

    boosted_docs = list(docs)
    people_retriever = get_retriever(k=max(base_k, 40))
    try:
        with concurrent.futures.ThreadPoolExecutor() as executor:
            results = list(executor.map(people_retriever.invoke, boosted_queries))
        for res in results:
            boosted_docs.extend(res)
    except Exception as e:
        logger.warning(f"People-list retrieval boost failed: {e}")

    seen = set()
    deduped = []
    for doc in boosted_docs:
        key = hash(doc.page_content)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(doc)
    return deduped

def _rank_people_list_docs(docs: List[Document], query: str) -> List[Document]:
    if not docs:
        return []

    q = (query or "").lower()
    ask_chairpersons = any(term in q for term in ["chair", "chairperson", "chairpersons", "department chair"])

    def _score(doc: Document) -> float:
        source = (doc.metadata.get("source") or "").lower()
        content = (doc.page_content or "").lower()
        score = 0.0

        if any(key in source for key in ["organizational", "faculty", "staff"]):
            score += 4.0
        if "csea" in source or "csea" in content:
            score += 2.0
        if any(key in content for key in ["faculty", "instructor", "professor", "staff", "department", "dean"]):
            score += 2.0
        if ask_chairpersons and any(key in content for key in ["chairperson", "department chair", "chair"]):
            score += 2.0

        if any(key in content for key in [
            "committee chairperson", "shall be appointed by the president", "vphe", "committee"
        ]):
            score -= 2.5

        score += min(len(content) / 5000.0, 1.5)
        return score

    return sorted(docs, key=_score, reverse=True)

# ── NEW: VECTOR DB CUSTODIAN LOADER ──
def _load_custodian_roster_from_vectordb() -> List[tuple[str, str]]:
    """Pulls the lab directory from the Vector DB chunks instead of a local file."""
    try:
        # 1. Ask the Vector DB specifically for the directory chunks
        roster_retriever = get_retriever(k=20)
        docs = roster_retriever.invoke("Custodian Laboratory Alias Room directory list")
        
        # 2. Stitch the scattered chunks back together into one giant text block
        text = "\n".join([doc.page_content for doc in docs])
        
        # 3. Run your custom data-cleaning regex over the stitched chunks!
        pattern = re.compile(
            r"^-\s*Custodian:\s*(.*?)\s*\|\s*Laboratory:\s*(.*?)\s*(?:\|\s*Alias:\s*(.*?)\s*)?(?:\|\s*Room:\s*(.*?)\s*)?$",
            re.IGNORECASE | re.MULTILINE,
        )
        roster: List[tuple[str, str]] = []
        seen = set()
        for name, laboratory, alias, _room in pattern.findall(text):
            custodian = re.sub(r"\s+", " ", (name or "").strip())
            lab = re.sub(r"\s+", " ", (laboratory or "").strip())
            alias_clean = re.sub(r"\s+", " ", (alias or "").strip())
            if not custodian or not lab:
                continue
            
            if alias_clean and alias_clean.lower() not in lab.lower():
                lab = f"{lab} ({alias_clean})"
                
            key = (custodian.lower(), lab.lower())
            if key in seen:
                continue
            seen.add(key)
            roster.append((custodian, lab))
        return roster
    except Exception as e:
        logger.warning(f"Failed to read custodian roster from Vector DB: {e}")
        return []
# ─────────────────────────────────────

def _format_custodian_roster_response(roster: List[tuple[str, str]]) -> str:
    lines = ["Here are all custodians and their assigned laboratories:", ""]
    for custodian, laboratory in roster:
        lines.append(f"- **{custodian}** - {laboratory}")
    return "\n".join(lines).strip()
# ─────────────────────────────────────────────────

def _extract_person_name_candidates(context: str) -> List[str]:
    if not context:
        return []

    blocked_words = {
        "bachelor", "science", "engineering", "program", "title", "description",
        "revised", "curriculum", "first", "second", "third", "fourth", "semester",
        "laboratory", "lecture", "ateneo", "naga", "university", "college", "source",
        "unknown", "methods", "research", "national", "service", "training", "physical",
        "activities", "technology", "society", "introduction", "catholic", "faith",
    }

    pattern = re.compile(
        r"\b(?:Dr\.?|Engr\.?|Prof\.?|Ar\.?)?\s*([A-Z][A-Za-z]+(?:\s+[A-Z]\.)?(?:\s+[A-Z][A-Za-z]+){1,3})\b"
    )

    candidates = []
    seen = set()
    for raw_name in pattern.findall(context):
        parts = [part.strip(".,") for part in raw_name.split() if part.strip(".,")]
        if sum(1 for part in parts if len(part) > 1) < 2:
            continue
        if any(any(ch.isdigit() for ch in part) for part in parts):
            continue
        if any(part.lower() in blocked_words for part in parts):
            continue

        normalized_parts = []
        for part in parts:
            if len(part) == 1:
                normalized_parts.append(part.upper() + ".")
            elif part.isupper():
                normalized_parts.append(part.title())
            else:
                normalized_parts.append(part)

        normalized = " ".join(normalized_parts)
        key = normalized.lower()
        if key in seen:
            continue
        seen.add(key)
        candidates.append(normalized)

    return candidates


def _extract_person_query_tokens(query: str) -> List[str]:
    q = (query or "").lower()
    if not re.search(
        r"\b(prof|professor|dr|engr|sir|maam|mr|ms|mrs|who is|sino si|teach|teaches|qualification|credentials)\b",
        q,
    ):
        return []

    words = re.findall(r"[a-zA-Z]+", q)
    if not words:
        return []

    titles = {"prof", "professor", "dr", "engr", "sir", "maam", "mr", "ms", "mrs", "ar"}
    stopwords = {
        "what", "who", "is", "are", "the", "for", "of", "in", "on", "to", "about",
        "tell", "me", "does", "do", "teach", "teaches", "teaching", "load", "loads",
        "credential", "credentials", "qualification", "qualifications", "department",
        "professor", "prof", "dr", "engr", "sir", "maam", "mr", "ms", "mrs", "ar",
    }

    preferred = []
    for idx, word in enumerate(words):
        if word in titles:
            for offset in (1, 2):
                next_idx = idx + offset
                if next_idx < len(words):
                    candidate = words[next_idx]
                    if len(candidate) >= 3 and candidate not in stopwords:
                        preferred.append(candidate)

    candidates = preferred + [w for w in words if len(w) >= 3 and w not in stopwords]
    deduped = []
    seen = set()
    for token in candidates:
        if token in seen:
            continue
        seen.add(token)
        deduped.append(token)
    return deduped


def _extract_person_name_query_parts(query: str) -> List[str]:
    if not query:
        return []

    words = re.findall(r"\b[a-zA-Z]{2,}\b", query.lower())
    stopwords = {
        "who", "what", "when", "where", "which", "how", "is", "are", "the", "a", "an",
        "tell", "me", "about", "prof", "professor", "dr", "engr", "sir", "maam", "mr", "ms", "mrs",
        "faculty", "staff", "teacher", "instructor", "dean", "chair", "chairperson", "department",
        "course", "curriculum", "subject", "subjects", "program", "room", "building", "lab", "laboratory",
        "do", "you", "u", "know", "did", "can", "could", "would", "happen", "want", "ask", "please", "pls", "po", "sino",
    }

    parts = []
    seen = set()
    for word in words:
        if word in stopwords or word in seen:
            continue
        seen.add(word)
        parts.append(word)
    return parts


def _count_name_part_matches(text: str, query_parts: List[str]) -> int:
    if not text or not query_parts:
        return 0
    text_lower = text.lower()
    return sum(1 for part in query_parts if re.search(rf"\b{re.escape(part)}\b", text_lower))


def _score_people_chunk_for_debug(doc: Document, query: str) -> float:
    source = (doc.metadata.get("source") or "").lower()
    content = (doc.page_content or "").lower()
    q = (query or "").lower()
    ask_chairpersons = any(term in q for term in ["chair", "chairperson", "chairpersons", "department chair"])

    score = 0.0
    if any(key in source for key in ["organizational", "faculty", "staff"]):
        score += 4.0
    if "csea" in source or "csea" in content:
        score += 2.0
    if any(key in content for key in ["faculty", "instructor", "professor", "staff"]):
        score += 2.0
    if ask_chairpersons and any(key in content for key in ["chairperson", "department chair"]):
        score += 2.0
    if "committee chairperson" in content:
        score -= 2.5

    return score + min(len(content) / 5000.0, 1.5)


def _log_people_rerank_debug(query: str, ranked_docs: List[Document], name_parts: List[str], limit: int = 6):
    if not ranked_docs:
        logger.info("[PeopleDebug] No ranked people chunks available.")
        return

    logger.info(
        f"[PeopleDebug] Query='{query}' | name_parts={name_parts} | candidates={len(ranked_docs)} | top_logged={min(limit, len(ranked_docs))}"
    )

    for idx, doc in enumerate(ranked_docs[:limit], start=1):
        content = doc.page_content or ""
        source = doc.metadata.get("source", "Unknown")
        base_score = _score_people_chunk_for_debug(doc, query)
        name_matches = _count_name_part_matches(content, name_parts) if len(name_parts) >= 1 else 0
        name_boost = 25.0 if name_matches >= 1 else 0.0
        final_score = base_score + name_boost
        preview = re.sub(r"\s+", " ", content).strip()[:120]
        logger.info(
            f"[PeopleDebug] #{idx} source='{source}' base={base_score:.2f} boost={name_boost:.2f} final={final_score:.2f} matches={name_matches} preview='{preview}'"
        )


def _build_people_disambiguation_message(query: str, context: str) -> str:
    tokens = _extract_person_query_tokens(query)
    if not tokens:
        return ""

    names = _extract_person_name_candidates(context)
    if len(names) < 2:
        return ""

    for token in tokens:
        matches = sorted({
            name for name in names
            if re.search(rf"\b{re.escape(token)}\b", name.lower())
        })
        if len(matches) > 1:
            options = ", ".join(matches[:4])
            return (
                f"I found multiple faculty members matching \"{token.title()}\": {options}. "
                "Please specify the full name so I can give the correct teaching load or credentials."
            )

    return ""


def _is_prerequisite_policy_query(query: str) -> bool:
    q = (query or "").lower()
    if not q:
        return False

    prerequisite_terms = ["prerequisite", "pre-requisite", "prereq", "required before", "kailangan"]
    fail_terms = ["fail", "failed", "failing", "bumagsak", "bagsak", "did not pass", "didn't pass"]
    progression_terms = [
        "can i take", "still take", "allowed to take", "pwede", "pwede ba", "take",
        "enroll", "proceed", "advance", "next course", "subsequent course",
        "ojt", "practicum", "internship",
    ]

    mentioned_course_codes = re.findall(r"\b[a-z]{2,5}\s*-?\s*\d{3}\b", q)

    has_prerequisite = any(term in q for term in prerequisite_terms) or len(mentioned_course_codes) >= 2
    has_fail_signal = any(term in q for term in fail_terms)
    has_progression_signal = any(term in q for term in progression_terms)

    return has_prerequisite and has_fail_signal and has_progression_signal


_FACTOID_STOPWORDS: Set[str] = {
    "what", "which", "who", "when", "where", "how", "is", "are", "was", "were",
    "the", "a", "an", "for", "of", "in", "on", "to", "and", "or", "from", "with",
    "course", "program", "student", "students", "adnu", "ateneo", "naga", "university",
}

_PROGRAM_HINTS = [
    "computer engineering",
    "civil engineering",
    "electronics engineering",
    "environmental management",
    "mathematics",
    "biology",
    "architecture",
]


def _extract_course_codes(text: str) -> List[str]:
    if not text:
        return []
    raw_codes = re.findall(r"\b[A-Za-z]{2,5}[-\s]*\d{3}\b", text)
    return [re.sub(r"[-\s]", "", code).lower() for code in raw_codes]


def _extract_factoid_keywords(query: str) -> List[str]:
    words = re.findall(r"[a-zA-Z]{3,}", (query or "").lower())
    deduped = []
    seen = set()
    for word in words:
        if word in _FACTOID_STOPWORDS or word in seen:
            continue
        seen.add(word)
        deduped.append(word)
    return deduped


def _prune_factoid_context_docs(query: str, docs: List[Document], max_docs: int = 4) -> List[Document]:
    """Keeps the most query-relevant chunks for short factoid queries."""
    if not docs:
        return []

    query_lower = (query or "").lower()
    query_codes = _extract_course_codes(query)
    query_keywords = _extract_factoid_keywords(query)
    query_program_hints = [hint for hint in _PROGRAM_HINTS if hint in query_lower]

    scored_docs = []
    for idx, doc in enumerate(docs):
        content = doc.page_content or ""
        content_lower = content.lower()
        content_norm = content_lower.replace(" ", "").replace("-", "")
        source_lower = (doc.metadata.get("source", "") or "").lower()

        score = 0.0
        if query_codes and any(code in content_norm for code in query_codes):
            score += 7.0

        for hint in query_program_hints:
            if hint in content_lower:
                score += 4.0
            elif "curriculum" in content_lower or "academic programs" in source_lower:
                score -= 1.5

        if query_program_hints:
            mismatched_programs = [hint for hint in _PROGRAM_HINTS if hint in content_lower and hint not in query_program_hints]
            if mismatched_programs:
                score -= 4.0 * len(mismatched_programs)

        overlap = sum(1 for keyword in query_keywords if keyword in content_lower)
        score += overlap * 1.5

        if "department" in query_lower and "department" in content_lower:
            score += 2.0
        if "admission" in query_lower and "admission" in content_lower:
            score += 2.0
        if "basis" in query_lower and ("based on cmo" in content_lower or "cmo" in content_lower):
            score += 2.0
        if "chapter" in query_lower and "chapter" in content_lower:
            score += 1.5
        if "unit" in query_lower and "unit" in content_lower:
            score += 1.0
        if "academic programs" in source_lower and any(k in query_lower for k in ["department", "admission", "program"]):
            score += 1.0

        # Keep slight positional preference from earlier ranking.
        score += max(0.0, 0.35 - (idx * 0.03))
        scored_docs.append((score, doc))

    scored_docs.sort(key=lambda item: item[0], reverse=True)

    top_doc_score = scored_docs[0][0]
    min_score_to_keep = max(2.0, top_doc_score - 2.5)
    pruned = [doc for score, doc in scored_docs if score >= min_score_to_keep][:max_docs]
    if pruned:
        return pruned

    return [doc for _, doc in scored_docs[:max_docs]]


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def get_llm_response(llm, prompt):
    return llm.invoke(prompt)

# ─────────────────────────────────────────────────────────────────────────────
# 3. MAIN GENERATOR PIPELINE
# ─────────────────────────────────────────────────────────────────────────────
def generate_response(query: str, chat_history_list: List[Dict[str, str]] = None):
    if chat_history_list is None: chat_history_list = []
    start_time = time.time()
    st.session_state["last_response_metadata"] = {
        "source_certainty": "",
        "suggested_questions": [],
    }
    
    is_valid, clean_query = validate_query(query) 
    if not is_valid:
        yield clean_query
        return
    
    safe_query = redact_pii(clean_query) 
    standalone_query = contextualize_query(safe_query, chat_history_list)

    # ── THESIS FIX: DETERMINISTIC ACRONYM INJECTION ──
    # Do not let the LLM guess what CSEA means. Force the exact spelling.
    standalone_query = re.sub(r'\bcsea\b', 'College of Science Engineering and Architecture', standalone_query, flags=re.IGNORECASE)
    standalone_query = re.sub(r'\badnu\b', 'Ateneo de Naga University', standalone_query, flags=re.IGNORECASE)
    # ── THESIS FIX: IMPLICIT CSEA CONTEXT ──
    # If the user generically asks for the dean or chair, implicitly inject CSEA 
    # so the Cross-Encoder knows exactly who they are talking about.
    if re.search(r'\b(dean|deans|chair|chairs|chairperson|chairpersons|dept chair| dept)\b', standalone_query, re.IGNORECASE) and 'CSEA' not in standalone_query:
        standalone_query += " CSEA"
    # ── THESIS FIX: IMPLICIT ROOM CONTEXT ──
    # If the user asks for a workshop or lab but forgets to type "room", inject it.
    # This guarantees a high vector match with headers like "Electronics Workshop Room".
    if re.search(r'\b(workshop|lab|laboratory)\b', standalone_query, re.IGNORECASE) and 'room' not in standalone_query.lower():
        standalone_query += " room"

    # ── THESIS FIX: IMPLICIT CALENDAR CONTEXT ──
    # Catch "when is" OR specific ADNU events and general holiday terms
    calendar_triggers = r'\b(when is|when are|schedule|date of|pintakasi|traslacion|holy week|undas|intrams|enrollment|midterm|midterms|prefi|final|finals|peñafrancia|penafrancia|festival|holiday|vacation|break)\b'
    if re.search(calendar_triggers, standalone_query, re.IGNORECASE) and 'calendar' not in standalone_query.lower():
        standalone_query += " calendar 2025 2026"
    # ─────────────────────────────────────────────────

# ── DIRECT ROUTING FOR EXTERNAL TOOLS (Estimator) ──
    estimator_keywords = ['estimator', 'tuition', 'fee', 'fees', 'cost', 'payment', 'price', 'magkano']
    if any(kw in standalone_query.lower() for kw in estimator_keywords):
        msg = "To estimate your tuition fees and other school assessments, please use the official ADNU School Fee Estimator here: [https://www.adnu.edu.ph/school-fee-estimator/]"
        for word in msg.split():
            yield word + " "
            time.sleep(0.02)
        return

    # ── DIRECT ROUTING FOR TIME & DATE ──
    date_keywords = ['what date', 'date today', 'current date', 'what day is it', 'what time is it', 'current time']
    if any(kw in standalone_query.lower() for kw in date_keywords) and 'enrollment' not in standalone_query.lower() and 'midterm' not in standalone_query.lower():
        from datetime import datetime
        now_str = datetime.now().strftime("%A, %B %d, %Y")
        msg = f"Today is {now_str}! If you need to know about specific academic dates like midterms or enrollment, just ask me about the calendar."
        for word in msg.split():
            yield word + " "
            time.sleep(0.02)
        return
        
    # ── PAASCU & ACCREDITATION BOOST ──
    if "paascu" in standalone_query.lower() or "accreditation" in standalone_query.lower():
        standalone_query += " PAASCU accreditation status level standard"
    # ────────────────────────────────────────────────────────

    # ── DYNAMIC FALLBACK GENERATOR ──
    def _generate_dynamic_fallback(user_q: str) -> str:
        fb_prompt = f"""You are AXIsstant, a friendly upperclassman CSEA assistant. 
The user asked: "{user_q}"
You do not have the official documentation to answer this, or it is a subjective/opinion-based question. 
Respond warmly and conversationally in 2-3 sentences. Acknowledge what they asked naturally (e.g., "As much as I'd love to tell you who the best teacher is..." or "I wish I knew the answer to that..."). Explain that your brain is only loaded with official university and college-approved documents, and offer to help them with those topics instead."""
        try:
            return get_generator_llm().invoke(fb_prompt).content.strip()
        except:
            return "I don't have the official info for that. However, if you have any more questions, I will try my best to answer!"
    # ─────────────────────────────────────
    
    standalone_query = normalize_course_codes(standalone_query)
    standalone_query = normalize_lab_aliases(standalone_query)
    is_prerequisite_policy_query = _is_prerequisite_policy_query(standalone_query)
    
    is_incomplete_input = is_incomplete_query(standalone_query)
    
    if (
        not is_incomplete_input
        and not is_listing_query(standalone_query)
        and not is_custodian_lookup_query(standalone_query)
        and not is_prerequisite_policy_query
    ):
        cached_answer = check_semantic_cache(standalone_query)
        if cached_answer:
            words = cached_answer.split(" ")
            for i in range(0, len(words), 3):
                yield " ".join(words[i:i+3]) + " "
                time.sleep(0.01)
            return
    else:
        if is_prerequisite_policy_query:
            logger.info(f"Prerequisite policy query detected: '{standalone_query}'. Skipping semantic cache for deterministic policy-safe response.")
        else:
            logger.info(f"Incomplete query detected: '{standalone_query}'. Skipping semantic cache for fresh closest-match retrieval.")

    if is_listing_query(standalone_query):
        logger.info(f"Listing query detected: '{standalone_query}'. Skipping semantic cache for complete list retrieval.")

    retrieval_start = time.time()
    
    try: intent, _, _, _ = route_query(standalone_query)
    except Exception as e:
        logger.warning(f"Router fallback triggered: {e}")
        intent = "search"

    # ── THESIS FIX: FORCE PEOPLE INTENT ──
    # If they mention any of these words, force the pipeline to treat it as a faculty search
    # so it triggers your custom ranking and penalties!
    people_keywords = ['chairperson', 'chairpersons', 'chair', 'chairs', 'dean', 'deans', 'faculty', 'teacher', 'professor', 'mam', 'maam', 'sir', 'prof', 'custodian']
    people_phrase_trigger = re.search(r"\b(who is|who are|sino si|tell me about)\b", standalone_query.lower())
    if detect_query_intent(standalone_query) == "people" or any(kw in standalone_query.lower() for kw in people_keywords) or people_phrase_trigger:
        intent = "people"
    # ───────────────────────────────────────

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

# ── THESIS FEATURE: SEMANTIC QUERY EXPANSION ──
    # 1. Translate Taglish/Slang into formal academic terms
# STEP 1: Expand query FIRST (before k is determined)
    if intent == "people":
        expanded_query = standalone_query
        logger.info("Bypassing semantic expansion for people intent to protect literal name strings.")
    else:
        expanded_query = expand_query_semantics(standalone_query)
    effective_query = expanded_query if expanded_query != standalone_query else standalone_query

    sub_queries = [standalone_query]
    
    # If the LLM expanded the query, add the formal version to our search net
    if expanded_query != standalone_query:
        sub_queries.append(expanded_query)
        
    # 2. Handle Complex/Comparative Queries (e.g. "difference between BSCS and BSCpE")
    is_complex = (not is_incomplete_input) and any(trigger in standalone_query.lower() for trigger in DECOMPOSE_TRIGGERS)
    if is_complex:
        try: 
            sub_queries.extend(decompose_query(standalone_query))
        except: pass

    # 3. Handle Follow-up Queries (e.g. "what are the prerequisites?")
    if is_incomplete_input:
        for variant in build_incomplete_query_variants(standalone_query, chat_history_list):
            if variant not in sub_queries:
                sub_queries.append(variant)
                
    # Add capitalization variants so retrieval is robust to user casing style.
    case_variants = [
        standalone_query,
        standalone_query.lower(),
        standalone_query.title(),
    ]
    for variant in case_variants:
        v = (variant or "").strip()
        if v and v not in sub_queries:
            sub_queries.append(v)

    # Add a retrieval bait query so people lookups reliably pull org-structure chunks.
    if intent == "people":
        bait_query = f"{standalone_query} CSEA faculty directory org structure"
        if bait_query not in sub_queries:
            sub_queries.append(bait_query)

    # Ensure uniqueness to save Pinecone calls
    sub_queries = list(dict.fromkeys(sub_queries))
    # ──────────────────────────────────────────────

    longest_variant = max(sub_queries, key=len)
    base_k = get_dynamic_k(longest_variant)
    
    has_course_code = bool(re.search(r'\b[A-Za-z]{2,5}\d{3}\b', standalone_query.upper()))
    has_specific_target = has_course_code or any(kw in standalone_query.lower() for kw in ['intersession', 'summer', 'prerequisite', 'elective'])



    def _extract_curriculum_version_label(context: str) -> str:
        """Extracts the clearest curriculum version label from retrieved context."""
        if not context:
            return ""

        text = re.sub(r"\r\n?", "\n", context)
        revised_match = re.search(
            r"((?:Revised|Newly\s+Revised|Updated)\s+Curriculum[^\n]*(?:\n(?:SY|AY)[^\n]*)?)",
            text,
            re.IGNORECASE,
        )
        if revised_match:
            version = revised_match.group(1)
            version = re.sub(r"^\s*#+\s*", "", version, flags=re.MULTILINE)
            version = re.sub(r"\s+", " ", version).strip(" -")
            return version

        sy_match = re.search(r"\b(?:SY|AY)\s*\d{4}\s*[-/]\s*\d{4}\b", text, re.IGNORECASE)
        if sy_match:
            return sy_match.group(0).strip()

        return ""


    def _extract_curriculum_note(context: str) -> str:
        """Extracts a short curriculum blurb from rationale or program-description sections."""
        if not context:
            return ""

        def _sanitize_paragraph(paragraph: str) -> str:
            paragraph = re.sub(r"\[\[Source:[^\]]+\]\]", "", paragraph)
            paragraph = re.sub(r"\s+", " ", paragraph).strip()
            if not paragraph or len(paragraph) < 60:
                return ""
            if paragraph.startswith("|") or paragraph.startswith("-") or paragraph.startswith("#"):
                return ""
            if re.match(r"^(?:SY|AY)\b", paragraph, re.IGNORECASE):
                return ""
            sentences = re.split(r"(?<=[.!?])\s+", paragraph)
            return " ".join(sentences[:2]).strip()

        text = re.sub(r"\r\n?", "\n", context)
        heading_patterns = [
            r"##\s*Program Title and Description",
            r"##\s*Program Description",
            r"##\s*Rationale(?: of the Revision)?",
            r"##\s*Rationale",
        ]

        for heading_pattern in heading_patterns:
            heading_match = re.search(heading_pattern, text, re.IGNORECASE)
            if not heading_match:
                continue
            tail = text[heading_match.end():]
            for paragraph in re.split(r"\n\s*\n", tail):
                cleaned = _sanitize_paragraph(paragraph)
                if cleaned:
                    return cleaned
                if paragraph.strip().startswith("##"):
                    break

        for paragraph in re.split(r"\n\s*\n", text):
            cleaned = _sanitize_paragraph(paragraph)
            if cleaned and any(term in cleaned.lower() for term in ["curriculum", "program", "engineering"]):
                return cleaned

        return ""


    # ── MASSIVE HAYSTACK TO BEAT VECTOR DILUTION ──
    is_curr_search = has_course_code or any(kw in standalone_query.lower() for kw in ['curriculum', 'subject', 'course', 'prerequisite', 'program', 'cpe', 'ece'])
    if is_curr_search or intent == "people":
        dynamic_k = max(base_k, 150)
    else:
        dynamic_k = max(base_k, 30) if (is_incomplete_input or has_specific_target) else base_k

    if has_course_code:
        code_match = re.search(r'\b[A-Z]{2,5}\d{3}\b', standalone_query.upper())
        if code_match:
            bait_query = f"{code_match.group(0)} complete curriculum syllabus course subjects prerequisites units laboratory"
            if bait_query not in sub_queries:
                sub_queries.append(bait_query)
            
            if code_match.group(0) not in sub_queries:
                sub_queries.append(code_match.group(0))

    retriever = get_retriever(k=dynamic_k)
    
    all_docs = []
    if len(sub_queries) == 1:
        all_docs = retriever.invoke(sub_queries[0])
    else:
        with concurrent.futures.ThreadPoolExecutor() as executor:
            results = list(executor.map(retriever.invoke, sub_queries))
        for res in results: all_docs.extend(res)

    if not all_docs:
        fallback = _generate_dynamic_fallback(standalone_query)
        chunk_size = 40
        for i in range(0, len(fallback), chunk_size):
            yield fallback[i:i+chunk_size]
            time.sleep(STREAM_DELAY)
        return

    logger.info(f"📂 Retrieval Success: Found {len(all_docs)} raw chunks using K={dynamic_k}")

    unique_docs_map = {d.page_content: d for d in all_docs}
    latest_per_source = prefer_latest_per_source(list(unique_docs_map.values()))
    
    is_curriculum_query = has_course_code or any(
        kw in standalone_query.lower() for kw in [
            'curriculum', 'subject', 'course', 'year', 'semester', 'units',
            'prerequisite', 'schedule', 'ojt', 'practicum', 'internship',
            'immersion', 'operating systems', 'elective', 'track',
            'program', 'cpe', 'ece', 'bscpe', 'bsece', 'ce', 'bsece', 
            'bsbio', 'bio', 'arch', 'bsarch', 'math', 'bsmath', 'em', 'bsem'
        ]
    )

    is_curriculum_output_query = is_curriculum_query and (
        is_curriculum_list_query(standalone_query) or any(
            kw in standalone_query.lower() for kw in [
                'curriculum', 'prospectus', 'subject', 'subjects', 'semester', 'year',
                '1st year', '2nd year', '3rd year', '4th year'
            ]
        )
    )
    
    is_analytical_query = any(kw in standalone_query.lower() for kw in ['most', 'least', 'highest', 'lowest', 'compare', 'which course', 'how many', 'most prerequisites', 'hardest', 'rank', 'full', 'entire', 'complete', 'all subjects', 'sum', 'total', 'count'])
    
    is_download_query = any(kw in standalone_query.lower() for kw in ['download', 'link', 'pdf', 'get the', 'access', 'where can i get', 'where can i download'])
    
    is_prerequisite_query = any(kw in standalone_query.lower() for kw in ['prerequisite', 'pre-requisite', 'prereq', 'required before', 'failed', 'can i take', 'allowed to take', 'kailangan'])

    is_facility_query = any(
        kw in standalone_query.lower() for kw in ['room', 'building', 'floor', 'lab', 'laboratory', 'office', 'located', 'where is', 'where', 'where are', 'nasaan', 'saan', 'campus', 'facility', 'location of']
    )

    is_listing_style_query = (
        is_listing_query(standalone_query)
        or is_curriculum_list_query(standalone_query)
        or is_people_list_query(standalone_query)
    )

    is_concise_factoid_query = (
        not is_listing_style_query
        and not is_curriculum_output_query
        and not is_analytical_query
        and not is_facility_query
        and not is_download_query
        and re.search(r"^(what|which|who|when|where|how many|how much|is|are|does|do|can)\b", standalone_query.lower()) is not None
    )

    top_score, second_score = float("-inf"), float("-inf")
    hybrid_results = hybrid_rerank(expanded_query, latest_per_source)

    # ── THE BM25 RESCUE OPERATION ──
    if has_course_code:
        course_codes = [re.sub(r'[-\s]', '', c).lower() for c in re.findall(r'\b[A-Za-z]{2,5}\d{3}\b', standalone_query)]
        for doc in latest_per_source:
            content_norm = doc.page_content.lower().replace(" ", "").replace("-", "")
            if any(code in content_norm for code in course_codes):
                if not any(d.page_content == doc.page_content for d in hybrid_results):
                    hybrid_results.append(doc)
    # ────────────────────────────────

    if intent == "people":
        people_pool = filter_to_people_docs(latest_per_source, standalone_query)
        people_pool = boost_people_list_docs(standalone_query, people_pool, dynamic_k)
        ranked_people = rank_people_list_docs(people_pool, standalone_query)
        people_name_parts = _extract_person_name_query_parts(standalone_query)

        # ── THESIS FIX: NAME RESCUE OVERRIDE ──
        # For specific-person queries, scan the full raw haystack and force matching chunks in.
        if len(people_name_parts) >= 1:
            ranked_contents = {doc.page_content for doc in ranked_people}
            for doc in latest_per_source:
                if _count_name_part_matches(doc.page_content or "", people_name_parts) >= 1:
                    if doc.page_content not in ranked_contents:
                        ranked_people.insert(0, doc)
                        ranked_contents.add(doc.page_content)

        scored_people = []
        for doc in ranked_people:
            base_people_score = _score_people_chunk_for_debug(doc, standalone_query)
            name_matches = _count_name_part_matches(doc.page_content or "", people_name_parts) if len(people_name_parts) >= 1 else 0
            name_boost = 25.0 if name_matches >= 1 else 0.0
            final_people_score = base_people_score + name_boost
            scored_people.append((final_people_score, base_people_score, name_matches, doc))

        scored_people.sort(key=lambda row: (row[2], row[0], row[1]), reverse=True)
        ranked_people = [row[3] for row in scored_people]
        top_reranked = ranked_people[:max(RERANKER_TOP_K, 16)]
        _log_people_rerank_debug(standalone_query, ranked_people, people_name_parts, limit=6)

        if scored_people:
            top_score = max(top_score, 5.0, float(scored_people[0][0]))
            if len(scored_people) > 1:
                second_score = max(second_score, float(scored_people[1][0]))

    elif is_analytical_query and is_curriculum_query:
        all_program_docs = filter_to_program(latest_per_source, standalone_query)
        big_retriever = get_retriever(k=50)
        extra_filtered = filter_to_program(prefer_latest_per_source(big_retriever.invoke(standalone_query)), standalone_query)
        combined = {hash(d.page_content): d for d in all_program_docs + extra_filtered}
        top_reranked = list(combined.values())
        if top_reranked: top_score = 10.0 
        
    else:
        query_intent = detect_query_intent(standalone_query)
        # ── HOLISTIC STRUCTURAL ROUTING ──
        # If the user asks for a comprehensive list (orgs, policies, directories)
        if is_listing_style_query:
            max_chunks = 25  # Open the floodgates to read the whole document
        # If it's a general domain query
        elif is_curriculum_query or query_intent == "people":
            max_chunks = 12           
        # If it's a highly specific factoid (prevent hallucination)
        else:
            max_chunks = 5
            # ─────────────────────────────────
            
        hybrid_results = enforce_source_diversity(hybrid_results, max_per_source=max_chunks)

        if hybrid_results:
            try:
                # ── THESIS FIX: SCORE AGAINST ORIGINAL QUERY, NOT EXPANDED ──
                # Expanded queries are for Vector DB recall. 
                # The Cross-Encoder MUST grade precision against the user's exact keywords (like "CSEA").
                pairs = [(standalone_query, doc.page_content) for doc in hybrid_results]
                
                scores = list(get_reranker().predict(pairs))
                
                # ── THE CROSS-ENCODER SAFETY NET ──
                raw_codes = re.findall(r'\b[A-Za-z]{2,5}[-\s]*\d{3}\b', standalone_query)
                course_codes = [re.sub(r'[-\s]', '', c).lower() for c in raw_codes]
                people_name_parts = _extract_person_name_query_parts(standalone_query)
                
                subject_keywords = [w for w in re.findall(r'\b[a-zA-Z]{4,}\b', standalone_query.lower()) if w not in {"what", "when", "where", "which", "who", "how", "course", "code", "subject", "for", "and", "the"}]

                for i, doc in enumerate(hybrid_results):
                    content_lower = doc.page_content.lower()
                    content_norm = content_lower.replace(" ", "").replace("-", "")
                    
                    if course_codes and any(code in content_norm for code in course_codes):
                        scores[i] += 50.0
                    elif '|' in content_lower and '---' in content_lower:
                        match_count = sum(1 for kw in subject_keywords if kw in content_lower)
                        if match_count > 0:
                            scores[i] += (match_count * 15.0) 

                    if len(people_name_parts) >= 1:
                        name_matches = _count_name_part_matches(content_lower, people_name_parts)
                        if name_matches >= 1:
                            scores[i] += 25.0
                            logger.info(f"🚀 Fuzzy name match boost (+25) for {people_name_parts} with {name_matches} matched parts.")
                # ─────────────────────────────────────────

                sorted_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
                
                top_score = float(scores[sorted_indices[0]])
                if len(sorted_indices) > 1: second_score = float(scores[sorted_indices[1]])
                
                # ── HOLISTIC FIX: SCORE-BASED GUILLOTINE FOR LISTS ──
                if is_listing_style_query:
                    # Keep chunks that are actually relevant (score > -2.5) to drop the random garbage.
                    # This prevents Context Dilution so the LLM doesn't get lazy reading massive texts.
                    valid_indices = [i for i in sorted_indices if scores[i] > -2.5]
                    if valid_indices:
                        top_reranked = [hybrid_results[i] for i in valid_indices]
                    else:
                        top_reranked = [hybrid_results[i] for i in sorted_indices[:5]] # Fallback
                else:
                    # For standard factoids, slice off the noise as usual.
                    top_reranked = [hybrid_results[i] for i in sorted_indices[:RERANKER_TOP_K]] 
                # ──────────────────────────────────────────────────────────────

            except Exception as e:
                logger.error(f"Reranking failed: {e}")
                top_reranked = hybrid_results[:RERANKER_TOP_K]
        else:
            top_reranked = []

# ── CUSTODIAN DIRECTORY INJECTION ──
    if is_custodian_lookup_query(standalone_query) or is_custodian_list_query(standalone_query):
        # We now call the Vector DB version!
        roster = _load_custodian_roster_from_vectordb()
        if roster:
            roster_text = _format_custodian_roster_response(roster)
            roster_doc = Document(page_content=roster_text, metadata={"source": "Lab Directory"})
            
            existing_contents = {d.page_content for d in top_reranked}
            if roster_doc.page_content not in existing_contents:
                top_reranked.insert(0, roster_doc)
                top_score = max(top_score, 15.0)
    # ────────────────────────────────────────

    # ── DYNAMIC PREREQUISITE INJECTION ──
    if is_prerequisite_query:
        stop_words = r'\b(what|is|the|for|of|in|bs|cpe|ece|ce|arch|em|bio|math|prerequisite|pre-requisite|prereq|subject|course|required|before)\b'
        subject_terms = re.sub(stop_words, '', standalone_query.lower()).strip()
        
        if len(subject_terms) > 3:
            prereq_retriever = get_retriever(k=15)
            prereq_docs = prereq_retriever.invoke(f"{subject_terms} prerequisite curriculum")
            prereq_filtered = prefer_latest_per_source(prereq_docs)
            
            existing_contents = {d.page_content for d in top_reranked}
            for doc in prereq_filtered:
                if doc.page_content not in existing_contents:
                    top_reranked.append(doc)
                    existing_contents.add(doc.page_content)
                    
            if top_reranked: top_score = max(top_score, 5.0)
            
    if is_download_query and top_reranked is not None:
        link_retriever = get_retriever(k=20)
        link_filtered = prefer_latest_per_source(link_retriever.invoke("official curriculum PDF download link"))
        existing_contents = {d.page_content for d in top_reranked}
        for doc in link_filtered:
            if doc.page_content not in existing_contents:
                top_reranked.append(doc)
                existing_contents.add(doc.page_content)
        if top_reranked: top_score = max(top_score, 5.0)

    if is_facility_query:
        already_has_directory = any('directory' in doc.metadata.get('source', '').lower() or 'campus' in doc.metadata.get('source', '').lower() for doc in top_reranked)
        if not already_has_directory:
            building_code_match = re.search(r'\b([A-Z]{1,3})\s*\d{3}\b', standalone_query.upper())
            if building_code_match:
                directory_query = f"{building_code_match.group(1)} building rooms directory"
            else:
                directory_query = "campus building directory rooms"

            directory_filtered = prefer_latest_per_source(get_retriever(k=8).invoke(directory_query))
            existing_contents = {d.page_content for d in top_reranked}
            for doc in directory_filtered:
                if doc.page_content not in existing_contents:
                    top_reranked.append(doc)
                    existing_contents.add(doc.page_content)
            if top_reranked: top_score = max(top_score, 5.0)

    if is_concise_factoid_query and top_reranked:
        top_reranked = _prune_factoid_context_docs(standalone_query, top_reranked, max_docs=4)
    
    logger.info(f"📊 Top score: {top_score:.2f} | Second: {second_score:.2f}")
    
    context_pieces = [f"[[Source: {doc.metadata.get('source', 'Unknown')}]\n{doc.page_content}" for doc in top_reranked]
    context = "\n\n".join(context_pieces)
    st.session_state["last_retrieved_context"] = context

    people_disambiguation_message = ""
    if intent == "people":
        people_disambiguation_message = _build_people_disambiguation_message(standalone_query, context)

    if people_disambiguation_message:
        source_list = [doc.metadata.get('source', 'Unknown') for doc in top_reranked]
        score_margin = top_score - second_score if second_score != float("-inf") else top_score
        certainty_note = build_source_certainty_note(top_score, score_margin, source_list)

        final_people_response = f"{people_disambiguation_message}\n\n{certainty_note}"
        st.session_state["last_response_metadata"] = {
            "source_certainty": certainty_note,
            "suggested_questions": [],
        }

        chunk_size = 40
        for i in range(0, len(final_people_response), chunk_size):
            yield final_people_response[i:i+chunk_size]
            time.sleep(STREAM_DELAY)
        return
    
    retrieval_time = time.time() - retrieval_start
    gen_start = time.time()

    # ── THESIS FIX: TIME-AWARENESS ──
    from datetime import datetime
    current_date = datetime.now().strftime("%B %d, %Y")
    # ────────────────────────────────
    
    prompt = f"""You are AXIsstant, the friendly and helpful Academic AI assistant of Ateneo de Naga University's College of Science, Engineering, and Architecture (CSEA). You help students and faculty with academic questions in a warm, conversational tone.

Answer the question using ONLY the context below.

### RULES (FOLLOW STRICTLY):
1. **TONE**: Write like a friendly, approachable upperclassman helping a classmate — casual but still accurate. Never sound like a formal document or a customer service bot. Do NOT start with "Great question!", "Good question!", "So,", "Kuya", "So here's", or "So to answer". Make every response different.
2. **NO FILLER**: Do NOT say "To provide you with..." or "I'll need to refer to...".
3. **LANGUAGE**: Always respond in English unless the student writes in Filipino, in which case respond in Filipino.
4. **USE TABLES FOR STRUCTURED DATA**: When the context contains curriculum subjects, grading scales, schedules, or faculty lists, reproduce the ACTUAL data in a Markdown table. **SHOW EVERY ROW** — never truncate or skip rows. Do NOT include rows where every cell contains only dashes (---).
5. **RESPECT MARKDOWN HEADERS**: The context uses markdown headers (## and ###). If a user asks for a specific category (like "CSEA orgs" or "CSEA"), ONLY list the items explicitly found underneath that specific matching header. Do NOT include items from other headers like "## EXTRACURRICULAR".
6. **CONCISE LISTING**: When the user asks for a "list" or "all" items, provide the NAME of every single item found in that section. For each item, provide only a 1-sentence summary of its purpose. Do NOT copy the full paragraphs from the context. This ensures a complete but readable response.
7. **EXHAUST ALL ITEMS**: When listing items from a section, include every item from that section and never skip entries. Keep each item concise according to Rule 6.
8. **CLEAN UP LISTS**: Use `- **Name** - Role` for people.
9. **STRICTLY FACTUAL & DYNAMIC FALLBACKS**: Use ONLY the provided context. If the answer is missing, or if the question is subjective (e.g., "who is the best teacher"), DO NOT use a robotic fallback. Respond conversationally and warmly based on their exact question (e.g., "As much as I'd love to tell you who the best prof is, I only have access to official documents..."). Suggest they ask a classmate or their department chair, and offer to help with curriculum/policies instead.
10. **CURRICULUM QUERIES**: When asked about subjects for a specific year, present ALL semesters for that year.
11. ELABORATE NATURALLY: Don't just give a one-sentence answer. If the context provides details (like what a thesis is about, or the specific goals of a program), include 1-2 extra sentences of explanation to be truly helpful.
12. **LISTS**: When listing multiple items, always put each item on its own line with a blank line before the list starts.
13. **ANALYTICAL QUERIES**: If asked to find the course with the most/least prerequisites, compare courses, or rank anything — count carefully, and give a definitive answer. 
14. **MULTIPLE TABLES**: Give each table a clear bold label above it.
15. **LINKS**: Never paste raw URLs. Always format links as descriptive markdown like [Download the official curriculum here](url).
16. **ROOM CODES**: When the user asks about a room, look up the building prefix in the campus directory in the context. Always decode the building name.
17. **NO SPECULATION OR ASSUMPTIONS**: NEVER guess, infer, or use phrases like "let's assume" or "assuming that". If a user's question requires variables that are missing from the context (e.g., specific class hours, exact unit loads), you MUST refuse to calculate it and explicitly state what missing information is needed to answer them.
18. **PREREQUISITES**: When showing curriculum subjects, ALWAYS include the prerequisite column in the table. If a subject has no prerequisite, write "None" in that cell.
19. **VAGUE COURSE QUERIES**: If the user just asks "What is [Course Code]?" or "[Course Code]", do not fail. Reply with a short sentence containing the Course Title, Credit Units, and Prerequisites.
20. **TIME AWARENESS**: Today's date is {current_date}. If the user asks for the current date or time, answer them directly. If a user asks when a recurring event is (like midterms, finals, or enrollment), look at today's date and ONLY provide the dates for the current or upcoming semester. Do NOT list dates from past semesters unless explicitly asked.
21. **CURRICULUM VERSION DISCLAIMER**: If the query is about curriculum, subjects, or prospectus, ALWAYS begin with this format: "⚠️ **Disclaimer:** This information is based on the **[version found in context]**. If you belong to an older batch, your curriculum may be different. Please verify with your department." Then add a short 1-2 sentence curriculum note using rationale/program description from context before listing subject details.
22. **FAILED PREREQUISITE POLICY**: For ANY course with a prerequisite (including OJT/practicum/internship), if a student fails the prerequisite subject, they cannot take the subsequent course yet. They must first retake and pass the failed prerequisite subject in the semester it is offered. Do NOT use unrelated policies (like absences or reduced loads) for this.
23. **DIRECT FACT ANSWERS**: If the user asks for a single fact (e.g., department, chapter, basis, units, course title, requirement), answer in 1-2 concise sentences and put the exact fact in the first sentence.

### FACULTY & CREDENTIALS RULES ###
1. Fuzzy Name Matching: You MUST be highly forgiving with faculty names. Users will frequently omit titles and middle names. For example, if a user asks about "Juan Dela Cruz", you MUST confidently match them to "Engr. Juan P. Dela Cruz" if they exist in the context.
2. Comprehensive Extraction: Once you match a faculty member, you MUST look at the text to identify their Department and Role. Your answer MUST clearly state their exact full name, title (if any), department, role, and the degree listed next to their name. Do NOT claim information is missing if it is physically in the text.
3. Disambiguation: If the user asks for a last name that belongs to multiple faculty members, you MUST stop and ask the user to clarify which specific professor they mean. Do not guess.
4. Degrees vs. Subjects Taught: If a name is listed next to a degree, that is the degree they HOLD, not the subject they teach. NEVER claim a professor teaches the degree they graduated with.
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
        # Update this boolean check in generate_response (around source: 334)
        # ── THESIS FIX: PROTECT CALENDARS FROM DROPPING ──
        is_calendar_query = 'calendar' in standalone_query.lower()
        
        is_protected_query = (
            is_curriculum_query or is_facility_query or is_analytical_query or 
            is_download_query or is_incomplete_input or is_prerequisite_query or
            has_course_code or intent == "history" or is_calendar_query  # <-- Added calendar here!
        )
        # ─────────────────────────────────────────────────

        # if top_score < LOW_CONFIDENCE_THRESHOLD and not is_protected_query:
        #     logger.warning(f"🔇 Low Retrieval Score ({top_score:.2f}). Aborting generation.")
        #     fallback = _generate_dynamic_fallback(standalone_query)
        #     chunk_size = 40
        #     for i in range(0, len(fallback), chunk_size):
        #         yield fallback[i:i+chunk_size]
        #         time.sleep(STREAM_DELAY)
        #     return

        # ── THESIS FIX: REMOVED THE ARTIFICIAL CHOKEHOLD ──
        # Let the LLM read the context instead of killing it based on a math score.
        if top_score < LOW_CONFIDENCE_THRESHOLD and not is_protected_query:
            logger.warning(f"⚠️ Low Retrieval Score ({top_score:.2f}), but passing to LLM to verify context anyway.")
        # ──────────────────────────────────────────────────

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
        
        # ── HOLISTIC ARCHITECTURAL FIX: INTENT-AWARE CONFIDENCE ──
        # Factoid queries require strict math (high score, high margin) to pass.
        # But for LIST queries, chunks are just fragments of the whole. 
        # CE scores will be naturally low and margins will be tiny. 
        if is_listing_query(standalone_query) or is_curriculum_list_query(standalone_query) or is_people_list_query(standalone_query):
            # If we know the user wants a list, and we found chunks, trust the retrieval and bypass the Critic.
            high_confidence = len(top_reranked) > 0
        else:
            # Standard factoid logic
            high_confidence = (top_score >= 1.5 and score_margin >= 0.4)
        # ─────────────────────────────────────────────────────────

        if high_confidence:
            logger.info(f"✨ High Confidence Routing. Bypassing Critic.")
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

        # ── DETERMINISTIC ANTI-SPECULATION KILL SWITCH ──
        # Made strictly contextual so it ignores phrases like "assume leadership roles"
        speculation_triggers = r'\b(let\'s assume|assuming that you|i will assume|i am assuming|hypothetically speaking)\b'
        if re.search(speculation_triggers, final_verified_response, re.IGNORECASE):
            logger.warning("🛡️ Python Kill Switch Triggered: LLM attempted to speculate.")
            final_verified_response = "I don't have enough specific information (like your exact unit load or class hours) to calculate that accurately. Please check your syllabus or ask your instructor directly to avoid any academic penalties!"
        
        
        # ── INTERCEPT NO-ANSWER SCENARIOS FOR BETTER TIPS ──
        if _contains_speculation(final_verified_response):
            cleaned_non_speculative = remove_speculative_sentences(final_verified_response)
            final_verified_response = cleaned_non_speculative if cleaned_non_speculative else "I couldn't find an explicit answer for that detail in the retrieved documents."

        if is_prerequisite_policy_query:
            final_verified_response = (
                "If you fail a prerequisite subject, you cannot take the subsequent course that requires it, including OJT/practicum/internship. "
                "You need to retake and pass the failed prerequisite first during the semester it is offered before enrolling in that next course."
            )

        final_verified_response = fix_markdown_tables(final_verified_response)
        final_verified_response = re.sub(r'\s+(\d+\.\s)', r'\n\1', final_verified_response)
        final_verified_response = re.sub(r'(?<!\n)\s{2,}-\s+\*\*', r'\n- **', final_verified_response)
        final_verified_response = format_raw_links(final_verified_response)

        if is_curriculum_output_query:
            version_label = _extract_curriculum_version_label(context) or "the latest curriculum document in the retrieved context"
            curriculum_disclaimer = (
                f"**Disclaimer:** This information is based on the **{version_label}** curriculum. "
                "If you belong to an older batch, your curriculum may be different. Please verify with your department."
            )
            curriculum_note = _extract_curriculum_note(context)
            note_block = f"**Curriculum note:** {curriculum_note}" if curriculum_note else ""

            if not final_verified_response.lstrip().startswith("⚠️ **Disclaimer:**"):
                prefix_parts = [curriculum_disclaimer]
                if note_block:
                    prefix_parts.append(note_block)
                final_verified_response = f"{'\\n\\n'.join(prefix_parts)}\\n\\n{final_verified_response}"
            elif note_block and "**Curriculum note:**" not in final_verified_response:
                final_verified_response = f"{final_verified_response}\\n\\n{note_block}"

        if is_facility_query and CAMPUS_MAP_URL and CAMPUS_MAP_URL not in final_verified_response:
            campus_map_note = (
                "For easier navigation, you can also check the "
                f"[ADNU campus map]({CAMPUS_MAP_URL})."
            )
            final_verified_response = f"{final_verified_response}\n\n{campus_map_note}"

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

        st.session_state["last_response_metadata"] = {
            "source_certainty": certainty_note,
            "suggested_questions": suggested_questions[:3],
        }
        
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
        st.session_state["last_response_metadata"] = {
            "source_certainty": "",
            "suggested_questions": [],
        }
        yield "Something went wrong on my end. Give it another try in a bit!"