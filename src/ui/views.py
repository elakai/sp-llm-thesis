import streamlit as st
import uuid

from src.core.retrieval import generate_response
from src.ui.suggested_questions import render_suggested_questions
from src.core.feedback import save_feedback, log_conversation, delete_conversation
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
# HISTORY VIEW
# ─────────────────────────────────────────────────────────────────────────────
def render_history_view():
    """Renders the chat history view with conversation list and controls."""
    st.markdown("<h1 style='color: #F0A62D; font-weight: bold;'>Chat History</h1>", unsafe_allow_html=True)
    st.markdown(
        """
        <style>
        /* Scope to widgets keyed like delete_0, delete_1, ... */
        [class*="st-key-delete_"] button {
            background-color: #dc2626 !important;
            color: #ffffff !important;
            border: 1px solid #b91c1c !important;
        }
        [class*="st-key-delete_"] button:hover {
            background-color: #b91c1c !important;
            border-color: #991b1b !important;
            color: #ffffff !important;
        }
        [class*="st-key-delete_"] button:focus {
            box-shadow: 0 0 0 0.2rem rgba(220, 38, 38, 0.35) !important;
        }
        [class*="st-key-delete_all_history"] button {
            background-color: #b91c1c !important;
            color: #ffffff !important;
            border: 1px solid #991b1b !important;
        }
        [class*="st-key-delete_all_history"] button:hover {
            background-color: #991b1b !important;
            border-color: #7f1d1d !important;
            color: #ffffff !important;
        }
        [class*="st-key-confirm_delete_all_btn"] button {
            background-color: #dc2626 !important;
            color: #ffffff !important;
            border: 1px solid #b91c1c !important;
        }
        [class*="st-key-confirm_delete_all_btn"] button:hover {
            background-color: #b91c1c !important;
            border-color: #991b1b !important;
            color: #ffffff !important;
        }
        [class*="st-key-cancel_delete_all"] button {
            background-color: #16a34a !important;
            color: #ffffff !important;
            border: 1px solid #15803d !important;
        }
        [class*="st-key-cancel_delete_all"] button:hover {
            background-color: #15803d !important;
            border-color: #166534 !important;
            color: #ffffff !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("---")

    history_owner = st.session_state.get("email") or st.session_state.get("user_id")
    if "confirm_delete_all_pending" not in st.session_state:
        st.session_state["confirm_delete_all_pending"] = False

    if st.session_state.get("chat_history"):
        if st.button("Delete all conversations", key="delete_all_history", type="secondary"):
            st.session_state["confirm_delete_all_pending"] = True
            st.rerun()

        if st.session_state.get("confirm_delete_all_pending"):
            st.warning("This will permanently delete all saved conversations for your account.")
            confirm_col, cancel_col = st.columns(2)
            with confirm_col:
                if st.button("Confirm delete", key="confirm_delete_all_btn", type="secondary"):
                    delete_failed = False
                    for conv in st.session_state["chat_history"]:
                        if isinstance(conv, dict):
                            session_id = conv.get("session_id")
                        else:
                            session_id = None
                        if session_id:
                            if not delete_conversation(session_id, history_owner):
                                delete_failed = True

                    st.session_state["chat_history"] = []
                    st.session_state["active_convo_idx"] = None
                    st.session_state["messages"] = []
                    st.session_state["confirm_delete_all_pending"] = False
                    if delete_failed:
                        st.toast("Some conversations could not be deleted from the database.", icon="⚠️")
                    st.rerun()
            with cancel_col:
                if st.button("Cancel", key="cancel_delete_all"):
                    st.session_state["confirm_delete_all_pending"] = False
                    st.rerun()

        if st.session_state.get("confirm_delete_all_pending"):
            st.markdown("---")

        if st.session_state.get("confirm_delete_all_pending"):
            return


    
    if not st.session_state.get("chat_history"):
        st.info("📭 No saved conversations yet.")
    else:
        for i, conv in enumerate(reversed(st.session_state["chat_history"])):
            if not conv: continue
            
            # Handle both old list format and new dictionary format
            if isinstance(conv, dict):
                session_id = conv.get("session_id", str(uuid.uuid4()))
                messages = conv.get("messages", [])
            else:
                # Old format: conv is a list of messages
                session_id = str(uuid.uuid4())
                messages = conv
            
            actual_idx = len(st.session_state["chat_history"]) - 1 - i
            first_msg = messages[0]["content"] if messages else "Empty Chat"
            title = first_msg[:50] + "..." if len(first_msg) > 50 else first_msg
            
            with st.expander(f"{title}", expanded=False):
                # Iterate over the messages list
                for msg in messages:
                    role_icon = "🐙" if msg["role"] == "assistant" else "👤"
                    st.markdown(f"**{role_icon} {msg['role'].title()}:** {msg['content']}")
                
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("Continue", key=f"load_{i}"):
                        # Pass only the messages list to the chat window
                        st.session_state["messages"] = [dict(msg) for msg in messages]
                        st.session_state["active_convo_idx"] = actual_idx
                        # Reuse the historical session ID
                        st.session_state["session_id"] = session_id
                        st.session_state["view"] = "chat"
                        st.rerun()
                with col2:
                    if st.button("Delete", key=f"delete_{i}", type="secondary"):
                        # Delete from database first (chat_logs are keyed by user_email)
                        db_deleted = delete_conversation(session_id, history_owner)
                        
                        # Then remove from session state
                        st.session_state["chat_history"].pop(actual_idx)
                        current_idx = st.session_state.get("active_convo_idx")
                        if current_idx is not None:
                            if current_idx == actual_idx:
                                st.session_state["active_convo_idx"] = None
                                st.session_state["messages"] = []
                            elif current_idx > actual_idx:
                                st.session_state["active_convo_idx"] = current_idx - 1
                        if not db_deleted:
                            st.toast("Conversation removed from this view, but database delete could not be confirmed.", icon="⚠️")
                        st.rerun()

# ─────────────────────────────────────────────────────────────────────────────
# CHAT VIEW
# ─────────────────────────────────────────────────────────────────────────────
def render_chat_view():
    """Renders the main chat interface with message history and input."""
    
    # Show welcome toast after login
    if st.session_state.get("show_welcome"):
        st.toast(f"Welcome back, {st.session_state.get('full_name', 'User')}!", icon="👋")
        st.session_state["show_welcome"] = False

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
        st.info("Welcome! Try asking: 'What is the grading system?' or Ateneo de Naga's Dress Code.")

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


        if role == "assistant":
            current_feedback = st.session_state.message_feedback.get(idx, None)
            query = _get_previous_user_query(st.session_state.messages, idx)

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
                            st.session_state.get("email"),
                            st.session_state.get("session_id"),
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
                            st.session_state.get("email"),
                            st.session_state.get("session_id"),
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
                            st.session_state.get("email"),
                            st.session_state.get("session_id"),
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
                            st.session_state.get("email"),
                            st.session_state.get("session_id"),
                            log_id=message.get("log_id"),
                            created_at=message.get("log_created_at"),
                        )
                        st.toast("We'll improve this answer.", icon="📝")
                    st.rerun()

            if current_feedback == "helpful":
                st.markdown("<div class='feedback-indicator helpful'><span>✓</span> You marked this response as helpful</div>", unsafe_allow_html=True)
            elif current_feedback == "not_helpful":
                st.markdown("<div class='feedback-indicator not-helpful'><span>✗</span> You marked this response as not helpful</div>", unsafe_allow_html=True)

    # Process queued suggestion after the history loop so it renders as
    # a separate chat turn, not inside the previous assistant message container.
    queued_query = st.session_state.pop("queued_query", None)
    if queued_query:
        _process_user_query(queued_query)
        return

    # Handle new user input
    if query := st.chat_input("Use full official names for subjects and rooms; avoid abbreviations; be specific."):
        _process_user_query(query)


def _process_user_query(query: str):
    """Process user query, generate response, and update chat state."""

    # ── GUEST MODE LIMITATION (Fallback - guests should use guest_chat.py) ──
    # Check if the user is a guest via is_guest flag
    is_guest_user = st.session_state.get("is_guest")
    if is_guest_user:
        # Initialize guest counter if it doesn't exist
        if "guest_query_count" not in st.session_state:
            st.session_state.guest_query_count = 0
            
        if st.session_state.guest_query_count >= GUEST_QUERY_LIMIT:
            with st.chat_message("assistant", avatar="assets/logo.png"):
                st.error(
                    "**Guest Limit Reached!**\n\n"
                    f"You have used all {GUEST_QUERY_LIMIT} free guest queries. "
                    "Please sign in with your **@gbox.adnu.edu.ph** or **@adnu.edu.ph** account to continue using AXIstant with unlimited queries!"
                )
            return # Stop processing
    # ────────────────────────────────────────────────────────────────────────────
    
    user_ts = _now_pht()
    st.session_state.messages.append({"role": "user", "content": query, "timestamp": user_ts})
    
    with st.chat_message("user"):
        st.markdown(query)
        st.markdown(f"<span style='font-size: 0.8em; color: gray;'>{user_ts}</span>", unsafe_allow_html=True)
    
    with st.chat_message("assistant", avatar="assets/logo.png"):
        extracted_source_certainty = ""
        suggestions = []
        clean_response = ""
        generation_succeeded = False
        log_metadata = {}
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

            log_metadata = log_conversation(
                query=query,
                response=clean_response,
                user_email=st.session_state.get("email", "Guest"),
                session_id=st.session_state.get("session_id"),
                context=current_context,
                metrics=performance_metrics
            ) or {}

        except Exception as e:
            logger.error(f"Response generation failed: {e}")
            clean_response = "I ran into a backend error while generating that answer. Please try again."
            st.error(f"⚠️ {clean_response}")

        if is_guest_user and generation_succeeded:
            st.session_state.guest_query_count += 1
        
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
        
        # Update chat history - handle both new and continued conversations
        if "chat_history" not in st.session_state:
            st.session_state["chat_history"] = []
        
        active_idx = st.session_state.get("active_convo_idx")
        chat_history = st.session_state["chat_history"]
        message_snapshot = [dict(msg) for msg in st.session_state["messages"]]
        
        # Validate active_idx is still valid
        if active_idx is not None and isinstance(active_idx, int) and 0 <= active_idx < len(chat_history):
            # Continuing an existing conversation - preserve its session_id and update messages
            existing = chat_history[active_idx]
            s_id = existing.get("session_id") if isinstance(existing, dict) else st.session_state.get("session_id")
            chat_history[active_idx] = {
                "session_id": s_id,
                "messages": message_snapshot
            }
        else:
            # New conversation - store as dict with session_id so feedback scope works
            chat_history.append({
                "session_id": st.session_state.get("session_id"),
                "messages": message_snapshot
            })
            st.session_state["active_convo_idx"] = len(chat_history) - 1

        # Rerun to cleanly display from session state with feedback buttons and suggestion chips
        st.rerun()
