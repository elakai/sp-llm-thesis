import re
import html
import streamlit as st
import base64
from datetime import datetime, timezone, timedelta
from pathlib import Path

from src.core.retrieval import generate_response
from src.ui.suggested_questions import render_suggested_questions
from src.core.feedback import log_conversation

_PHT = timezone(timedelta(hours=8))

def _now_pht() -> str:
    return datetime.now(_PHT).strftime("%I:%M %p")


def _get_logo_base64() -> str:
    logo_path = Path("assets/logo.png")
    if logo_path.exists():
        with open(logo_path, "rb") as f:
            return base64.b64encode(f.read()).decode()
    return ""


def _get_previous_user_query(messages, assistant_idx: int) -> str:
    for cursor in range(assistant_idx - 1, -1, -1):
        if messages[cursor].get("role") == "user":
            return messages[cursor].get("content", "")
    return ""

def _strip_suggestions(text: str) -> str:
    if not text:
        return text
    return re.sub(
        r'\n\n---\n\*\*You might also want to ask:\*\*\n(?:- .+\n?)*',
        '',
        text
    ).strip()

def _extract_source_certainty(text: str) -> tuple[str, str]:
    if not text:
        return text, ""

    pattern = re.compile(
        r'\n?>\s*\*\*Source certainty:\*\*.*?(?=\n\n---|\Z)',
        re.IGNORECASE | re.DOTALL,
    )
    match = pattern.search(text)
    if not match:
        return text.strip(), ""

    source_block = match.group(0).strip()
    source_plain = re.sub(r'^\s*>\s?', '', source_block, flags=re.MULTILINE).strip()
    source_plain = re.sub(r'\*\*(.*?)\*\*', r'\1', source_plain)
    source_plain = re.sub(r'\*(.*?)\*', r'\1', source_plain)

    cleaned = (text[:match.start()] + text[match.end():]).strip()
    cleaned = re.sub(r'\n{3,}', '\n\n', cleaned)
    return cleaned, source_plain

def _render_message_meta(source_certainty: str, timestamp: str = ""):
    if not source_certainty and not timestamp:
        return

    certainty_html = ""
    if source_certainty:
        source_certainty = re.sub(r'\*\*(.*?)\*\*', r'\1', source_certainty)
        source_certainty = source_certainty.replace('*', '')
        count_match = re.search(r'based on\s+(\d+)\s+document', source_certainty, re.IGNORECASE)
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


# ─────────────────────────────────────────────────────────────────────────────
# GUEST CHAT VIEW
# ─────────────────────────────────────────────────────────────────────────────
def render_guest_chat_view():
    """Renders the guest chat interface with 5-query limit."""
    
    # Initialize guest query counter if not exists
    if "guest_query_count" not in st.session_state:
        st.session_state.guest_query_count = 0

    logo_base64 = _get_logo_base64()
    if logo_base64:
        st.markdown(
            f"""
            <style>
            .stApp::before {{
                content: "" !important;
                position: fixed;
                top: 0;
                left: 0;
                right: 0;
                bottom: 0;
                background:
                    radial-gradient(ellipse at 20% 80%, rgba(255, 143, 31, 0.05) 0%, transparent 50%),
                    radial-gradient(ellipse at 80% 20%, rgba(255, 251, 173, 0.03) 0%, transparent 40%),
                    url("data:image/png;base64,{logo_base64}");
                background-repeat: no-repeat, no-repeat, no-repeat;
                background-position: center, center, center;
                background-size: auto, auto, min(52vw, 504px);
                opacity: 0.14;
                pointer-events: none;
                z-index: 0;
            }}
            </style>
            """,
            unsafe_allow_html=True,
        )
    
    st.markdown("<h1 style='color: #FFAF47; font-weight: bold;'>Your CSEA Information Assistant</h1>", unsafe_allow_html=True)

    if not st.session_state.messages:
        st.info("Welcome to AXIstant Guest Mode! Try asking: 'What is the grading system?' or Ateneo de Naga's Dress Code.\n\n"
                "You only have **10** free guest queries")

    # Initialize feedback tracking if not exists
    if "message_feedback" not in st.session_state:
        st.session_state.message_feedback = {}

    # Display existing messages
    for idx, message in enumerate(st.session_state.messages):
        role = message.get("role", "")
        is_assistant = (role == "assistant")
        avatar = "assets/logo.png" if is_assistant else None
        
        with st.chat_message(role, avatar=avatar):
            content = _strip_suggestions(message["content"])
            source_certainty = (message.get("source_certainty") or "").strip()
            content, content_source_certainty = _extract_source_certainty(content)
            if not source_certainty and content_source_certainty:
                source_certainty = content_source_certainty
            st.markdown(content)
            if is_assistant:
                _render_message_meta(source_certainty, message.get("timestamp", ""))
            elif "timestamp" in message:
                st.markdown(f"<span style='font-size: 0.8em; color: gray;'>{message['timestamp']}</span>", unsafe_allow_html=True)

            if is_assistant and message.get("suggestions"):
                render_suggested_questions(message["suggestions"], key_prefix=f"suggest_{idx}")

        # Simplified feedback for guests - no database saving
        if role == "assistant":
            st.markdown("<div class='eval-prompt'>Was this answer helpful?</div>", unsafe_allow_html=True)
            feedback_col1, feedback_col2 = st.columns(2)

            with feedback_col1:
                if st.button(
                    "Helpful",
                    key=f"eval_helpful_{idx}",
                    type="primary" if st.session_state.message_feedback.get(idx) == "helpful" else "secondary",
                ):
                    if st.session_state.message_feedback.get(idx) == "helpful":
                        st.session_state.message_feedback[idx] = None
                        st.toast("Feedback removed")
                    else:
                        st.session_state.message_feedback[idx] = "helpful"
                        st.toast("Thanks for your feedback!", icon="✅")
                    st.rerun()

            with feedback_col2:
                if st.button(
                    "Not helpful",
                    key=f"eval_not_helpful_{idx}",
                    type="primary" if st.session_state.message_feedback.get(idx) == "not_helpful" else "secondary",
                ):
                    if st.session_state.message_feedback.get(idx) == "not_helpful":
                        st.session_state.message_feedback[idx] = None
                        st.toast("Feedback removed")
                    else:
                        st.session_state.message_feedback[idx] = "not_helpful"
                        st.toast("We'll improve this answer.", icon="📝")
                    st.rerun()

            current_feedback = st.session_state.message_feedback.get(idx)
            if current_feedback == "helpful":
                st.markdown("<div class='feedback-indicator helpful'><span>✓</span> You marked this response as helpful</div>", unsafe_allow_html=True)
            elif current_feedback == "not_helpful":
                st.markdown("<div class='feedback-indicator not-helpful'><span>✗</span> You marked this response as not helpful</div>", unsafe_allow_html=True)

    # Process queued suggestion after the history loop so it renders as
    # a separate chat turn, not inside the previous assistant message container.
    queued_query = st.session_state.pop("queued_query", None)
    if queued_query:
        _process_guest_query(queued_query)
        return

    # Handle new user input
    if query := st.chat_input("Use full official names for subjects and rooms; avoid abbreviations; be specific."):
        _process_guest_query(query)


def _process_guest_query(query: str):
    """Process guest user query with 5-query limit."""
    
    # Check if guest hit the limit
    if st.session_state.guest_query_count >= 10:
        with st.chat_message("assistant", avatar="assets/logo.png"):
            st.error(
                "**Limit Reached!**\n\n"
                "You have used all 10 free guest queries. "
                "Please sign in with your **@gbox.adnu.edu.ph** account to continue using AXIstant with unlimited queries!"
            )
        st.stop()
        return
    
    # Increment counter
    st.session_state.guest_query_count += 1
    
    user_ts = _now_pht()
    st.session_state.messages.append({"role": "user", "content": query, "timestamp": user_ts})
    
    with st.chat_message("user"):
        st.markdown(query)
        st.markdown(f"<span style='font-size: 0.8em; color: gray;'>{user_ts}</span>", unsafe_allow_html=True)
    
    with st.chat_message("assistant", avatar="assets/logo.png"):
        extracted_source_certainty = ""
        suggestions = []
        clean_response = ""
        current_context = ""
        performance_metrics = {}
        try:
            with st.spinner("Thinking..."):
                stream = generate_response(
                    query=query,
                    chat_history_list=st.session_state.messages
                )
                response_placeholder = st.empty()
                full_response = ""
                for chunk in stream:
                    full_response += chunk
                    stream_no_source, _ = _extract_source_certainty(full_response)
                    if '|' in stream_no_source:
                        response_placeholder.markdown(stream_no_source, unsafe_allow_html=True)
                    else:
                        response_placeholder.markdown(stream_no_source + "▌", unsafe_allow_html=True)
                rendered_response = _strip_suggestions(full_response)
                rendered_response_no_source, source_certainty = _extract_source_certainty(rendered_response)
                response_placeholder.markdown(rendered_response_no_source, unsafe_allow_html=True)
                _render_message_meta(source_certainty)
                
            clean_response = _strip_suggestions(full_response)
            clean_no_source, extracted_source_certainty = _extract_source_certainty(clean_response)
            clean_response = clean_no_source

            current_context = st.session_state.get("last_retrieved_context", "")
            performance_metrics = st.session_state.get("performance_metrics", {})

            SUGGESTION_PATTERN = re.compile(
                r'\n+---\n\*\*You might also want to ask:\*\*\n((?:- .+\n?)+)',
                re.IGNORECASE
            )
            suggestion_match = SUGGESTION_PATTERN.search(full_response)
            if suggestion_match:
                suggestions = [
                    re.sub(r'^- ', '', line).strip()
                    for line in suggestion_match.group(1).strip().split('\n')
                    if line.strip().startswith('- ')
                ]
            else:
                suggestions = []
            
        except Exception as e:
            clean_response = f"⚠️ Backend Error: {str(e)}"
            st.error(clean_response)
            suggestions = []

        # Always log guest queries so they appear in admin live feedback logs,
        # even when generation fails or returns guarded responses.
        guest_user_label = st.session_state.get("user_id") or "Guest"
        log_conversation(
            query=query,
            response=clean_response,
            user_email=guest_user_label,
            session_id=st.session_state.get("session_id"),
            context=current_context,
            metrics=performance_metrics,
            force_log=True,
            is_guest=True
        )
        
        asst_ts = _now_pht()
        # Save the clean assistant response into session state
        st.session_state.messages.append({
            "role": "assistant", 
            "content": clean_response, 
            "timestamp": asst_ts,
            "suggestions": suggestions,
            "source_certainty": extracted_source_certainty,
        })
        
        # Rerun to cleanly display from session state
        st.rerun()
