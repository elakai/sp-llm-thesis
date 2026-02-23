import streamlit as st
import uuid
from datetime import datetime

from src.core.retrieval import generate_response
from src.core.feedback import save_feedback, log_conversation

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
            
            actual_idx = len(st.session_state["chat_history"]) - 1 - i
            first_msg = conv[0]["content"] if conv else "Empty Chat"
            title = first_msg[:50] + "..." if len(first_msg) > 50 else first_msg
            
            with st.expander(f"💬 {title}", expanded=False):
                for msg in conv:
                    role_icon = "🐙" if msg["role"] == "assistant" else "👤"
                    st.markdown(f"**{role_icon} {msg['role'].title()}:** {msg['content']}")
                
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("📂 Continue", key=f"load_{i}"):
                        st.session_state["messages"] = conv.copy()
                        st.session_state["active_convo_idx"] = actual_idx
                        st.session_state["session_id"] = str(uuid.uuid4())
                        st.session_state["view"] = "chat"
                        st.rerun()
                with col2:
                    if st.button("🗑️ Delete", key=f"delete_{i}", type="secondary"):
                        st.session_state["chat_history"].pop(actual_idx)
                        if st.session_state.get("active_convo_idx") == actual_idx:
                            st.session_state["active_convo_idx"] = None
                            st.session_state["messages"] = []
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
    
    st.markdown("<h1 style='color: #F0A62D; font-weight: bold;'>AXIsstant</h1>", unsafe_allow_html=True)

    if not st.session_state.messages:
        st.info("👋 Welcome! Try asking: 'What is the grading system?' or Ateneo de Naga's Dress Code")

    # Display existing messages
    for idx, message in enumerate(st.session_state.messages):
        avatar = "assets/kraken_logo.png" if message["role"] == "assistant" else None
        with st.chat_message(message["role"], avatar=avatar):
            st.markdown(message["content"])
            if "timestamp" in message:
                st.markdown(f"<span style='font-size: 0.8em; color: gray;'>{message['timestamp']}</span>", unsafe_allow_html=True)
            # Show feedback buttons for assistant messages
            if message["role"] == "assistant":
                col1, _ = st.columns([1, 4])
                with col1:
                    sub1, sub2 = st.columns(2)
                    with sub1:
                        if st.button("👍", key=f"g_{idx}"):
                            # Get the previous user message as query
                            query = st.session_state.messages[idx-1]["content"] if idx > 0 else ""
                            save_feedback(query, message["content"], "helpful", st.session_state.get("user_id"))
                            st.toast("Feedback recorded!")
                    with sub2:
                        if st.button("👎", key=f"b_{idx}"):
                            query = st.session_state.messages[idx-1]["content"] if idx > 0 else ""
                            save_feedback(query, message["content"], "not_helpful", st.session_state.get("user_id"))
                            st.toast("We'll improve this answer.", icon="📝")

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
    
    with st.chat_message("assistant", avatar="assets/kraken_logo.png"):
        try:
            # Create placeholder for thinking animation
            thinking_placeholder = st.empty()
            thinking_html = """
            <div class="thinking-container">
                <span class="thinking-text">Thinking</span>
                <div class="thinking-dots">
                    <span></span>
                    <span></span>
                    <span></span>
                </div>
            </div>
            """
            thinking_placeholder.markdown(thinking_html, unsafe_allow_html=True)
            
            # Get the response stream
            stream = generate_response(
                query=query,
                chat_history_list=st.session_state.messages
            )
            
            # Wrap stream to clear placeholder on first chunk
            def stream_with_clear():
                first = True
                for chunk in stream:
                    if first:
                        thinking_placeholder.empty()
                        first = False
                    yield chunk
            
            response = st.write_stream(stream_with_clear())

            current_context = st.session_state.get("last_retrieved_context", "")
            performance_metrics = st.session_state.get("performance_metrics", {})
            
            log_conversation(
                query=query,
                response=response,
                user_email=st.session_state.get("user_id", "Guest"),
                session_id=st.session_state.get("session_id"),
                context=current_context,
                metrics=performance_metrics
            )

        except Exception as e:
            response = f"⚠️ Backend Error: {str(e)}"
            st.error(response)
        
        asst_ts = datetime.now().strftime("%I:%M %p")
        st.session_state.messages.append({"role": "assistant", "content": response, "timestamp": asst_ts})
        
        # Update chat history
        if st.session_state.get("active_convo_idx") is not None:
            idx = st.session_state["active_convo_idx"]
            if 0 <= idx < len(st.session_state["chat_history"]):
                st.session_state["chat_history"][idx] = st.session_state["messages"].copy()
        elif len(st.session_state.messages) == 2:
            st.session_state["chat_history"].append(st.session_state["messages"].copy())
            st.session_state["active_convo_idx"] = len(st.session_state["chat_history"]) - 1

        # Rerun to cleanly display from session state with feedback buttons
        st.rerun()
