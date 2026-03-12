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
    
    cached_answer = check_semantic_cache(standalone_query)
    if cached_answer:
        words = cached_answer.split(" ")
        chunk_size = 3
        for i in range(0, len(words), chunk_size):
            yield " ".join(words[i:i+chunk_size]) + " "
            time.sleep(0.01)
        return

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

    is_complex = any(trigger in standalone_query.lower() for trigger in DECOMPOSE_TRIGGERS)
    sub_queries = [standalone_query]
    if is_complex:
        try:
            sub_queries = decompose_query(standalone_query)
        except:
            pass

    dynamic_k = get_dynamic_k(standalone_query)
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
        logger.warning(f"⚠️ Vector Search returned 0 results for: {standalone_query}")
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
            is_download_query
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

        final_verified_response = fix_markdown_tables(final_verified_response)
        final_verified_response = re.sub(r'\s+(\d+\.\s)', r'\n\1', final_verified_response)
        final_verified_response = format_raw_links(final_verified_response)
        
        gen_time = time.time() - gen_start
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

    def _is_answerable_question(q: str, source_tokens: set[str], source_codes: set[str]) -> bool:
        if len(q) < 12 or len(q) > 140:
            return False

        q_codes = _extract_course_codes(q)
        if q_codes and not q_codes.issubset(source_codes):
            return False

        q_tokens = _extract_tokens(q)
        overlap = q_tokens.intersection(source_tokens)
        return len(overlap) >= 2

    def _fallback_questions(source_text: str) -> list[str]:
        source_lower = source_text.lower()
        fallbacks: list[str] = []

        if "prereq" in source_lower or "pre-requisite" in source_lower or "prerequisite" in source_lower:
            fallbacks.append("What are the exact prerequisites mentioned here?")
        if _extract_course_codes(source_text):
            fallbacks.append("Can you list all course codes mentioned in this answer?")
        if "semester" in source_lower or "curriculum" in source_lower or "course" in source_lower:
            fallbacks.append("Can you summarize this by semester or category?")

        fallbacks.extend([
            "Can you summarize that in 3 short bullet points?",
            "Which part of the available documents supports this answer?",
            "What is the key takeaway I should remember from this?",
        ])

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

    source_text = f"{response}\n{context}"
    
    if skip_llm:
        return _fallback_questions(source_text)

    try:
        llm = get_generator_llm()
        prompt = f"""Create exactly 3 short follow-up questions the assistant can answer
USING ONLY the provided response and context.

Hard constraints:
- Do not introduce new facts, entities, course codes, policies, or assumptions.
- Keep each question specific to details already present.
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

        source_tokens = _extract_tokens(source_text)
        source_codes = _extract_course_codes(source_text)

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
                if _is_answerable_question(cleaned, source_tokens, source_codes):
                    validated.append(cleaned)
                    seen.add(key)
                if len(validated) == 3:
                    break

        if len(validated) < 3:
            for q in _fallback_questions(source_text):
                key = q.lower()
                if key not in seen:
                    validated.append(q)
                    seen.add(key)
                if len(validated) == 3:
                    break

        return validated[:3]
    except Exception as e:
        logger.warning(f"Suggested questions failed: {e}")
    return [
        "Can you summarize that in 3 short bullet points?",
        "Which part of the available documents supports this answer?",
        "What is the key takeaway I should remember from this?",
    ]