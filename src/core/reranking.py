import re
from typing import List
from collections import defaultdict
from langchain_core.documents import Document
from rank_bm25 import BM25Okapi
from src.config.constants import POSITIONAL_SCORE_WEIGHT, RETRIEVAL_K
from src.config.settings import get_retriever
from src.config.logging_config import logger

def _tokenize(text: str) -> list:
    return re.sub(r'[^\w\s]', ' ', text.lower()).split()

def hybrid_rerank(query: str, docs: List[Document]) -> List[Document]:
    if not docs: return []
    try:
        tokenized_docs = [_tokenize(doc.page_content) for doc in docs]
        bm25 = BM25Okapi(tokenized_docs)
        bm25_scores = bm25.get_scores(_tokenize(query))

        ranked = []
        for i, doc in enumerate(docs):
            position_score = (len(docs) - i) * POSITIONAL_SCORE_WEIGHT
            ranked.append((bm25_scores[i] + position_score, doc))

        ranked.sort(reverse=True, key=lambda x: x[0])
        return [doc for _, doc in ranked[:RETRIEVAL_K]]
    except Exception as e:
        logger.warning(f"Hybrid rerank failed: {e}")
        return docs[:RETRIEVAL_K]

def enforce_source_diversity(docs: List[Document], max_per_source: int = 3) -> List[Document]:
    source_counts = defaultdict(int)
    diverse_docs = []
    for doc in docs:
        src = doc.metadata.get("source", "unknown")
        if source_counts[src] < max_per_source:
            diverse_docs.append(doc)
            source_counts[src] += 1
    return diverse_docs

def filter_to_program(docs: List[Document], query: str) -> List[Document]:
    PROGRAM_KEYWORDS = {
        'computer engineering': 'cpe', 'cpe': 'cpe', 'civil engineering': 'ce', 'bs ce': 'ce',
        'electronics engineering': 'ece', 'bs ece': 'ece', 'architecture': 'arch', 'bs arch': 'arch',
        'biology': 'bio', 'bs bio': 'bio', 'mathematics': 'math', 'bs math': 'math',
        'environmental management': 'em', 'bs em': 'em',
    }
    q = query.lower()
    matched_program = next((code for kw, code in PROGRAM_KEYWORDS.items() if kw in q), None)
    if not matched_program: return docs
    return [d for d in docs if matched_program in d.metadata.get("source", "").lower()]

def filter_to_people_docs(docs: List[Document], query: str) -> List[Document]:
    if not docs: return []
    q = (query or "").lower()
    if not any(trigger in q for trigger in ["professor", "faculty", "instructor", "teacher", "staff", "chair", "dean", "chairperson"]):
        return docs

    content_kws = ["faculty", "professor", "instructor", "teacher", "staff", "chair", "department", "office", "personnel", "full-time", "part-time"]
    source_kws = ["faculty", "organizational", "org", "structure", "staff", "personnel"]

    filtered = []
    for doc in docs:
        content = (doc.page_content or "").lower()
        source = (doc.metadata.get("source") or "").lower()
        if any(k in content for k in content_kws) or any(k in source for k in source_kws):
            filtered.append(doc)
    return filtered if filtered else docs

def boost_people_list_docs(query: str, docs: List[Document], base_k: int) -> List[Document]:
    boosted_docs = list(docs)
    people_retriever = get_retriever(k=max(base_k, 25))
    try:
        boosted_docs.extend(people_retriever.invoke("Ateneo de Naga CSEA faculty list and organizational structure department chairs"))
    except Exception as e:
        logger.warning(f"People-list retrieval boost failed: {e}")

    seen, deduped = set(), []
    for doc in boosted_docs:
        key = hash(doc.page_content)
        if key not in seen:
            seen.add(key)
            deduped.append(doc)
    return deduped

def rank_people_list_docs(docs: List[Document], query: str) -> List[Document]:
    if not docs: return []
    q = (query or "").lower()
    ask_chairs = any(t in q for t in ["chair", "chairperson"])

    def _score(doc: Document) -> float:
        score = 0.0
        src, cnt = (doc.metadata.get("source") or "").lower(), (doc.page_content or "").lower()
        if any(key in src for key in ["organizational", "faculty", "staff"]): score += 4.0
        if "csea" in src or "csea" in cnt: score += 2.0
        if any(key in cnt for key in ["faculty", "instructor", "professor", "staff"]): score += 2.0
        if ask_chairs and any(key in cnt for key in ["chairperson", "department chair"]): score += 2.0
        if "committee chairperson" in cnt: score -= 2.5
        return score + min(len(cnt) / 5000.0, 1.5)

    return sorted(docs, key=_score, reverse=True)

def prefer_latest_per_source(docs: List[Document]) -> List[Document]:
    if not docs: return []
    grouped = defaultdict(list)
    for doc in docs: grouped[doc.metadata.get("source", "unknown")].append(doc)

    filtered = []
    for group in grouped.values():
        latest = max((d.metadata.get("uploaded_at", 0) for d in group), default=0)
        filtered.extend([d for d in group if d.metadata.get("uploaded_at", 0) == latest])
    return filtered
