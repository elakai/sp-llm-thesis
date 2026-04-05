import uuid
import streamlit as st

from src.core.retrieval import generate_response
from src.ui.suggested_questions import render_suggested_questions
from src.core.feedback import log_conversation, save_feedback
from src.config.constants import GUEST_QUERY_LIMIT
from src.config.logging_config import logger
from src.ui.chat_utils import (
    extract_source_certainty as _extract_source_certainty,
    extract_suggestions as _extract_suggestions,
    get_last_response_metadata as _get_last_response_metadata,
    get_logo_base64 as _get_logo_base64,
    prepare_message_content as _prepare_message_content,
    get_previous_user_query as _get_previous_user_query,
    now_pht as _now_pht,
    render_message_meta as _render_message_meta,
    stream_response_with_throttle as _stream_response_with_throttle,
    strip_suggestions as _strip_suggestions,
)


# ─────────────────────────────────────────────────────────────────────────────
# GUEST CHAT VIEW
# ─────────────────────────────────────────────────────────────────────────────
def render_guest_chat_view():
    """Renders the guest chat interface with a fixed guest query limit."""
    
    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())

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
        st.info(
            "Welcome to AXIstant Guest Mode! Try asking: 'What is the grading system?' "
            "or Ateneo de Naga's Dress Code.\n\n"
            f"You only have **{GUEST_QUERY_LIMIT}** free guest queries"
        )

    # Initialize feedback tracking if not exists
    if "message_feedback" not in st.session_state:
        st.session_state.message_feedback = {}

    # Display existing messages
    for idx, message in enumerate(st.session_state.messages):
        role = message.get("role", "")
        is_assistant = (role == "assistant")
        avatar = "assets/logo.png" if is_assistant else None
        
        with st.chat_message(role, avatar=avatar):
            content, source_certainty = _prepare_message_content(
                message.get("content", ""),
                message.get("source_certainty", ""),
            )
            st.markdown(content)
            if is_assistant:
                _render_message_meta(source_certainty, message.get("timestamp", ""))
            elif "timestamp" in message:
                st.markdown(f"<span style='font-size: 0.8em; color: gray;'>{message['timestamp']}</span>", unsafe_allow_html=True)

            if is_assistant and message.get("suggestions"):
                render_suggested_questions(message["suggestions"], key_prefix=f"suggest_{idx}")

        # Feedback UI and Database Saving
        if role == "assistant":
            current_feedback = st.session_state.message_feedback.get(idx, None)
            query = _get_previous_user_query(st.session_state.messages, idx)
            
            # ROOT CAUSE FIX 1: Hardcode "Guest" to guarantee a database match
            guest_user_label = "Guest"
            current_session_id = st.session_state.get("session_id")

            st.markdown("<div class='eval-prompt'>Was this answer helpful?</div>", unsafe_allow_html=True)
            feedback_col1, feedback_col2 = st.columns(2)

            with feedback_col1:
                if st.button(
                    "Helpful",
                    key=f"eval_helpful_{idx}",
                    type="primary" if current_feedback == "helpful" else "secondary",
                    help="Click again to remove" if current_feedback == "helpful" else None,
                ):
                    if current_feedback == "helpful":
                        st.session_state.message_feedback[idx] = None
                        save_feedback(
                            query,
                            message["content"],
                            None,
                            guest_user_label,
                            current_session_id,
                            log_id=message.get("log_id"),
                            created_at=message.get("log_created_at"),
                        )
                        st.toast("Feedback removed")
                    else:
                        st.session_state.message_feedback[idx] = "helpful"
                        save_feedback(
                            query,
                            message["content"],
                            "helpful",
                            guest_user_label,
                            current_session_id,
                            log_id=message.get("log_id"),
                            created_at=message.get("log_created_at"),
                        )
                        st.toast("Thanks for your feedback!", icon="✅")
                    st.rerun()

            with feedback_col2:
                if st.button(
                    "Not helpful",
                    key=f"eval_not_helpful_{idx}",
                    type="primary" if current_feedback == "not_helpful" else "secondary",
                    help="Click again to remove" if current_feedback == "not_helpful" else None,
                ):
                    if current_feedback == "not_helpful":
                        st.session_state.message_feedback[idx] = None
                        save_feedback(
                            query,
                            message["content"],
                            None,
                            guest_user_label,
                            current_session_id,
                            log_id=message.get("log_id"),
                            created_at=message.get("log_created_at"),
                        )
                        st.toast("Feedback removed")
                    else:
                        st.session_state.message_feedback[idx] = "not_helpful"
                        save_feedback(
                            query,
                            message["content"],
                            "not_helpful",
                            guest_user_label,
                            current_session_id,
                            log_id=message.get("log_id"),
                            created_at=message.get("log_created_at"),
                        )
                        st.toast("We'll improve this answer.", icon="📝")
                    st.rerun()

            if current_feedback == "helpful":
                st.markdown("<div class='feedback-indicator helpful'><span>✓</span> You marked this response as helpful</div>", unsafe_allow_html=True)
            elif current_feedback == "not_helpful":
                st.markdown("<div class='feedback-indicator not-helpful'><span>✗</span> You marked this response as not helpful</div>", unsafe_allow_html=True)

    queued_query = st.session_state.pop("queued_query", None)
    if queued_query:
        _process_guest_query(queued_query)
        return

    # Handle new user input
    if query := st.chat_input("Use full official names for subjects and rooms; avoid abbreviations; be specific."):
        _process_guest_query(query)


def _process_guest_query(query: str):
    """Process a guest query while enforcing a fixed free-query limit."""
    
    # Check if guest hit the limit
    if st.session_state.guest_query_count >= GUEST_QUERY_LIMIT:
        with st.chat_message("assistant", avatar="assets/logo.png"):
            st.error(
                "**Limit Reached!**\n\n"
                f"You have used all {GUEST_QUERY_LIMIT} free guest queries. "
                "Please sign in with your **@gbox.adnu.edu.ph** account to continue using AXIstant with unlimited queries!"
            )
        st.stop()
        return
    
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
        generation_succeeded = False
        try:
            with st.spinner("Thinking..."):
                stream = generate_response(
                    query=query,
                    chat_history_list=st.session_state.messages
                )
                response_placeholder = st.empty()
                full_response = _stream_response_with_throttle(stream, response_placeholder)
                rendered_response = _strip_suggestions(full_response)
                rendered_response_no_source, parsed_source_certainty = _extract_source_certainty(rendered_response)
                metadata_source_certainty, _ = _get_last_response_metadata()
                source_certainty = metadata_source_certainty or parsed_source_certainty
                response_placeholder.markdown(rendered_response_no_source)
                _render_message_meta(source_certainty)
                
            clean_response = _strip_suggestions(full_response)
            clean_no_source, parsed_source_certainty = _extract_source_certainty(clean_response)
            clean_response = clean_no_source

            metadata_source_certainty, metadata_suggestions = _get_last_response_metadata()
            extracted_source_certainty = metadata_source_certainty or parsed_source_certainty

            current_context = st.session_state.get("last_retrieved_context", "")
            performance_metrics = st.session_state.get("performance_metrics", {})
            suggestions = metadata_suggestions or _extract_suggestions(full_response)
            generation_succeeded = True

        except Exception as e:
            logger.error(f"Guest response generation failed: {e}")
            clean_response = "I ran into a backend error while generating that answer. Please try again."
            st.error(f"⚠️ {clean_response}")
            suggestions = []

        guest_user_label = "Guest"

        if generation_succeeded:
            st.session_state.guest_query_count += 1

        log_metadata = log_conversation(
            query=query,
            response=clean_response,
            user_email=guest_user_label,
            session_id=st.session_state.get("session_id"),
            context=current_context,
            metrics=performance_metrics,
            force_log=True,
            is_guest=True
        ) or {}
        
        asst_ts = _now_pht()
        # Save the clean assistant response into session state
        st.session_state.messages.append({
            "role": "assistant", 
            "content": clean_response, 
            "timestamp": asst_ts,
            "suggestions": suggestions,
            "source_certainty": extracted_source_certainty,
            "log_created_at": log_metadata.get("created_at"),
            "log_id": log_metadata.get("id"),
        })
        
        # Rerun to cleanly display from session state
        st.rerun()
