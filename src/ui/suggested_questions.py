import json
import re
from typing import List
import streamlit as st

from src.config.settings import get_generator_llm


def _normalize(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", (text or "").lower()).strip()


def _token_set(text: str) -> set:
    normalized = _normalize(text)
    return set(normalized.split()) if normalized else set()


def _token_overlap_ratio(a: str, b: str) -> float:
    a_tokens = _token_set(a)
    b_tokens = _token_set(b)
    if not a_tokens or not b_tokens:
        return 0.0
    overlap = a_tokens & b_tokens
    union = a_tokens | b_tokens
    return len(overlap) / len(union) if union else 0.0


def _parse_fallback_lines(text: str) -> List[str]:
    if not text:
        return []
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    items: List[str] = []
    for line in lines:
        if line.startswith("-"):
            items.append(line.lstrip("- ").strip())
        else:
            numbered = re.sub(r"^\d+\.[\s]+", "", line)
            if numbered != line:
                items.append(numbered.strip())
    return items


def _dedupe_questions(questions: List[str], user_query: str, max_questions: int) -> List[str]:
    seen = set()
    norm_user = _normalize(user_query)
    user_token_count = len(_token_set(user_query))
    cleaned: List[str] = []
    for question in questions:
        q = " ".join((question or "").split()).strip()
        if not q:
            continue
        if len(q.split()) > 12:
            continue
        if not q.endswith("?"):
            q += "?"
        if user_token_count >= 3 and _token_overlap_ratio(q, user_query) < 0.12:
            continue
        norm_q = _normalize(q)
        if not norm_q or norm_q == norm_user:
            continue
        if norm_q in seen:
            continue
        q_tokens = _token_set(q)
        is_redundant = False
        for kept in cleaned:
            kept_tokens = _token_set(kept)
            if not q_tokens or not kept_tokens:
                continue
            overlap = q_tokens & kept_tokens
            union = q_tokens | kept_tokens
            if union and (len(overlap) / len(union)) >= 0.6:
                is_redundant = True
                break
        if is_redundant:
            continue
        seen.add(norm_q)
        cleaned.append(q)
        if len(cleaned) >= max_questions:
            break
    return cleaned


def _needs_achievement_award_followup(user_query: str) -> bool:
    normalized = _normalize(user_query)
    return "achievement award" in normalized or "achievement awards" in normalized


def generate_suggested_questions(user_query: str, context: str, max_questions: int = 3) -> List[str]:
    if not user_query or not context or len(context) < 80:
        return []

    max_questions = min(max_questions, 3)

    prompt = f"""You create suggested follow-up questions for a school knowledge-base assistant.
Use ONLY the provided context. Each question must be answerable using the context alone.
Each question must be related to the user's question but explore other related topics.
Do NOT repeat the user's question. Do NOT ask about anything not present in the context.
Keep each question concise (max 12 words).
If you cannot find enough grounded options, return fewer questions.
Return a JSON array of strings and nothing else.

User question: "{user_query}"

Context:
{context}
"""

    try:
        llm = get_generator_llm()
        response = llm.invoke(prompt)
        raw = response.content.strip() if response and response.content else ""
    except Exception:
        return []

    questions: List[str] = []
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, list):
            questions = [str(item).strip() for item in parsed if isinstance(item, str)]
    except Exception:
        questions = _parse_fallback_lines(raw)

    if _needs_achievement_award_followup(user_query):
        questions.insert(0, "What are other achievement awards?")

    return _dedupe_questions(questions, user_query, max_questions)


def render_suggested_questions(questions: List[str], key_prefix: str) -> None:
    if not questions:
        return
    st.markdown("<div class='suggested-title'>You might also ask</div>", unsafe_allow_html=True)
    cols = st.columns(len(questions))
    for idx, question in enumerate(questions):
        with cols[idx]:
            if st.button(question, key=f"{key_prefix}_{idx}"):
                st.session_state["queued_query"] = question
                st.rerun()
