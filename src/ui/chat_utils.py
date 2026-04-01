import base64
import html
import re
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Dict, List, Tuple

import streamlit as st

_PHT = timezone(timedelta(hours=8))

_SUGGESTION_BLOCK_PATTERN = re.compile(
    r"\n\n---\n\*\*You might also want to ask:\*\*\n(?:- .+\n?)*"
)
_SUGGESTION_LIST_PATTERN = re.compile(
    r"\n+---\n\*\*You might also want to ask:\*\*\n((?:- .+\n?)+)",
    re.IGNORECASE,
)
_SOURCE_CERTAINTY_PATTERN = re.compile(
    r"\n?>\s*\*\*Source certainty:\*\*.*?(?=\n\n---|\Z)",
    re.IGNORECASE | re.DOTALL,
)


def now_pht() -> str:
    return datetime.now(_PHT).strftime("%I:%M %p")


def get_logo_base64() -> str:
    logo_path = Path("assets/logo.png")
    if logo_path.exists():
        with open(logo_path, "rb") as file_obj:
            return base64.b64encode(file_obj.read()).decode()
    return ""


def get_previous_user_query(messages: List[Dict[str, Any]], assistant_idx: int) -> str:
    for cursor in range(assistant_idx - 1, -1, -1):
        if messages[cursor].get("role") == "user":
            return messages[cursor].get("content", "")
    return ""


def strip_suggestions(text: str) -> str:
    if not text:
        return text
    return _SUGGESTION_BLOCK_PATTERN.sub("", text).strip()


def extract_source_certainty(text: str) -> Tuple[str, str]:
    if not text:
        return text, ""

    match = _SOURCE_CERTAINTY_PATTERN.search(text)
    if not match:
        return text.strip(), ""

    source_block = match.group(0).strip()
    source_plain = re.sub(r"^\s*>\s?", "", source_block, flags=re.MULTILINE).strip()
    source_plain = re.sub(r"\*\*(.*?)\*\*", r"\1", source_plain)
    source_plain = re.sub(r"\*(.*?)\*", r"\1", source_plain)

    cleaned = (text[:match.start()] + text[match.end():]).strip()
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned, source_plain


def extract_suggestions(text: str) -> List[str]:
    if not text:
        return []

    suggestion_match = _SUGGESTION_LIST_PATTERN.search(text)
    if not suggestion_match:
        return []

    suggestions = []
    for line in suggestion_match.group(1).strip().split("\n"):
        line = line.strip()
        if line.startswith("- "):
            suggestion = re.sub(r"^- ", "", line).strip()
            if suggestion:
                suggestions.append(suggestion)
    return suggestions


def get_last_response_metadata() -> Tuple[str, List[str]]:
    metadata = st.session_state.get("last_response_metadata") or {}
    source_certainty = (metadata.get("source_certainty") or "").strip()
    source_certainty = re.sub(r"^\s*>\s?", "", source_certainty).strip()

    raw_suggestions = metadata.get("suggested_questions") or []
    suggestions: List[str] = []
    if isinstance(raw_suggestions, list):
        for item in raw_suggestions:
            if isinstance(item, str):
                question = item.strip()
                if question:
                    suggestions.append(question)

    return source_certainty, suggestions


def render_message_meta(source_certainty: str, timestamp: str = "") -> None:
    if not source_certainty and not timestamp:
        return

    certainty_html = ""
    if source_certainty:
        source_certainty = re.sub(r"\*\*(.*?)\*\*", r"\1", source_certainty)
        source_certainty = source_certainty.replace("*", "")
        count_match = re.search(r"based on\s+(\d+)\s+document", source_certainty, re.IGNORECASE)
        badge_text = count_match.group(1) if count_match else "i"
        tooltip = html.escape(source_certainty)
        certainty_html = (
            f"<div class='source-certainty-wrap'>"
            f"<span class='source-certainty-btn'>{badge_text}</span>"
            f"<div class='source-certainty-tooltip'>{tooltip}</div>"
            f"</div>"
        )

    timestamp_html = ""
    if timestamp:
        timestamp_html = f"<span class='message-timestamp-inline'>{html.escape(timestamp)}</span>"

    st.markdown(f"<div class='message-meta-row'>{certainty_html}{timestamp_html}</div>", unsafe_allow_html=True)
