import re
import streamlit as st
import uuid
import copy
import base64
from datetime import datetime
from pathlib import Path

from src.core.retrieval import generate_response
from src.core.feedback import save_feedback, log_conversation, delete_conversation


def _get_logo_base64() -> str:
    logo_path = Path("assets/logo.png")
    if logo_path.exists():
        with open(logo_path, "rb") as f:
            return base64.b64encode(f.read()).decode()
    return ""

# ─────────────────────────────────────────────────────────────────────────────
# HISTORY VIEW
# ─────────────────────────────────────────────────────────────────────────────
def render_history_view():
    """Renders the chat history view with conversation list and controls."""
    st.markdown("<h1 style='color: #F0A62D; font-weight: bold;'>Chat History</h1>", unsafe_allow_html=True)
    st.markdown("---")
    
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
                    if st.button("📂 Continue", key=f"load_{i}"):
                        # Pass only the messages list to the chat window
                        st.session_state["messages"] = copy.deepcopy(messages)
                        st.session_state["active_convo_idx"] = actual_idx
                        # Reuse the historical session ID
                        st.session_state["session_id"] = session_id
                        st.session_state["view"] = "chat"
                        st.rerun()
                with col2:
                    if st.button("🗑️ Delete", key=f"delete_{i}", type="secondary"):
                        # Delete from database first
                        user_email = st.session_state.get("user_id")
                        delete_conversation(session_id, user_email)
                        
                        # Then remove from session state
                        st.session_state["chat_history"].pop(actual_idx)
                        current_idx = st.session_state.get("active_convo_idx")
                        if current_idx is not None:
                            if current_idx == actual_idx:
                                st.session_state["active_convo_idx"] = None
                                st.session_state["messages"] = []
                            elif current_idx > actual_idx:
                                st.session_state["active_convo_idx"] = current_idx - 1
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
    
    st.markdown("<h1 style='color: #FFAF47; font-weight: bold;'>AXIsstant</h1>", unsafe_allow_html=True)

    if not st.session_state.messages:
        st.info("👋 Welcome! Try asking: 'What is the grading system?' or Ateneo de Naga's Dress Code")

    # Initialize feedback tracking if not exists
    if "message_feedback" not in st.session_state:
        st.session_state.message_feedback = {}

    # Display existing messages
    for idx, message in enumerate(st.session_state.messages):
        role = message.get("role", "")
        is_assistant = (role == "assistant")
        avatar = "assets/logo.png" if is_assistant else None
        
        with st.chat_message(role, avatar=avatar):
            st.markdown(message["content"])
            if "timestamp" in message:
                st.markdown(f"<span style='font-size: 0.8em; color: gray;'>{message['timestamp']}</span>", unsafe_allow_html=True)
            
            # Feedback buttons - ONLY render if role is exactly "assistant"
            if role == "assistant":
                current_feedback = st.session_state.message_feedback.get(idx, None)
                
                col1, _ = st.columns([1, 4])
                with col1:
                    sub1, sub2 = st.columns(2)
                    with sub1:
                        if st.button("👍", key=f"g_{idx}", help="Helpful" if current_feedback != "helpful" else "Click to remove"):
                            query = st.session_state.messages[idx-1]["content"] if idx > 0 else ""
                            if current_feedback == "helpful":
                                st.session_state.message_feedback[idx] = None
                                save_feedback(query, message["content"], "removed", st.session_state.get("email"), st.session_state.get("session_id"))
                                st.toast("Feedback removed")
                            else:
                                st.session_state.message_feedback[idx] = "helpful"
                                save_feedback(query, message["content"], "helpful", st.session_state.get("email"), st.session_state.get("session_id"))
                                st.toast("Thanks for your feedback!", icon="✅")
                            st.rerun()
                    with sub2:
                        if st.button("👎", key=f"b_{idx}", help="Not helpful" if current_feedback != "not_helpful" else "Click to remove"):
                            query = st.session_state.messages[idx-1]["content"] if idx > 0 else ""
                            if current_feedback == "not_helpful":
                                st.session_state.message_feedback[idx] = None
                                save_feedback(query, message["content"], "removed", st.session_state.get("email"), st.session_state.get("session_id"))
                                st.toast("Feedback removed")
                            else:
                                st.session_state.message_feedback[idx] = "not_helpful"
                                save_feedback(query, message["content"], "not_helpful", st.session_state.get("email"), st.session_state.get("session_id"))
                                st.toast("We'll improve this answer.", icon="📝")
                            st.rerun()
                
                # Visual indicator for selected feedback
                if current_feedback == "helpful":
                    st.markdown("""
                    <div class="feedback-indicator helpful">
                        <span>✓</span> You found this helpful
                    </div>
                    """, unsafe_allow_html=True)
                elif current_feedback == "not_helpful":
                    st.markdown("""
                    <div class="feedback-indicator not-helpful">
                        <span>✗</span> Marked for improvement
                    </div>
                    """, unsafe_allow_html=True)

                # ── RENDER SUGGESTED QUESTIONS (ONLY ON THE MOST RECENT MESSAGE) ──
                if message.get("suggestions") and idx == len(st.session_state.messages) - 1:
                    st.markdown("<br>**You might also want to ask:**", unsafe_allow_html=True)
                    for q_idx, q in enumerate(message["suggestions"]):
                        # Render button. If clicked, automatically send it as a query.
                        if st.button(q, key=f"sugg_{idx}_{q_idx}"):
                            _process_user_query(q)

    # Handle new user input
    if query := st.chat_input("Ask AXIsstant about rules, exemptions, or curriculum..."):
        _process_user_query(query)


def _process_user_query(query: str):
    """Process user query, generate response, and update chat state."""
    user_ts = datetime.now().strftime("%I:%M %p")
    st.session_state.messages.append({"role": "user", "content": query, "timestamp": user_ts})
    
    with st.chat_message("user"):
        st.markdown(query)
        st.markdown(f"<span style='font-size: 0.8em; color: gray;'>{user_ts}</span>", unsafe_allow_html=True)
    
    with st.chat_message("assistant", avatar="assets/logo.png"):
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
                    if '|' in full_response:
                        response_placeholder.markdown(full_response, unsafe_allow_html=True)
                    else:
                        response_placeholder.markdown(full_response + "▌", unsafe_allow_html=True)
                response_placeholder.markdown(full_response, unsafe_allow_html=True)
                
            # ── EXTRACT & STRIP SUGGESTED QUESTIONS FROM RAW TEXT ──
            suggested_q_match = re.search(
                r'\*\*You might also want to ask:\*\*\n((?:- .+\n?)+)',
                full_response
            )
            
            extracted_questions = []
            clean_response = full_response
            
            if suggested_q_match:
                # Get the questions
                extracted_questions = re.findall(r'- (.+)', suggested_q_match.group(1))
                # Remove the raw markdown from the text so it doesn't print as a bulleted list
                clean_response = full_response.replace(suggested_q_match.group(0), "").strip()

            current_context = st.session_state.get("last_retrieved_context", "")
            performance_metrics = st.session_state.get("performance_metrics", {})
            
            log_conversation(
                query=query,
                response=clean_response,
                user_email=st.session_state.get("email", "Guest"),
                session_id=st.session_state.get("session_id"),
                context=current_context,
                metrics=performance_metrics
            )

        except Exception as e:
            clean_response = f"⚠️ Backend Error: {str(e)}"
            extracted_questions = []
            st.error(clean_response)
        
        asst_ts = datetime.now().strftime("%I:%M %p")
        # Save the Clean text and the Suggestions array into the dictionary
        st.session_state.messages.append({
            "role": "assistant", 
            "content": clean_response, 
            "timestamp": asst_ts,
            "suggestions": extracted_questions  # <--- Added to state here!
        })
        
        # Update chat history - handle both new and continued conversations
        if "chat_history" not in st.session_state:
            st.session_state["chat_history"] = []
        
        active_idx = st.session_state.get("active_convo_idx")
        chat_history = st.session_state["chat_history"]
        
        # Validate active_idx is still valid
        if active_idx is not None and isinstance(active_idx, int) and 0 <= active_idx < len(chat_history):
            # Continuing an existing conversation - preserve its session_id and update messages
            existing = chat_history[active_idx]
            s_id = existing.get("session_id") if isinstance(existing, dict) else st.session_state.get("session_id")
            chat_history[active_idx] = {
                "session_id": s_id,
                "messages": copy.deepcopy(st.session_state["messages"])
            }
        else:
            # New conversation - store as dict with session_id so feedback scope works
            chat_history.append({
                "session_id": st.session_state.get("session_id"),
                "messages": copy.deepcopy(st.session_state["messages"])
            })
            st.session_state["active_convo_idx"] = len(chat_history) - 1

        # Rerun to cleanly display from session state with feedback buttons and suggestion chips
        st.rerun()