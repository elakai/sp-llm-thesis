import sys
from pathlib import Path
import uuid
from datetime import datetime

# ─────────────────────────────────────────────────────────────────────────────
# 1. PATH SETUP (CRITICAL: Must be at the top)
# ─────────────────────────────────────────────────────────────────────────────
project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# ─────────────────────────────────────────────────────────────────────────────
# 2. IMPORTS
# ─────────────────────────────────────────────────────────────────────────────
import streamlit as st
from src.ui.components import render_login, render_sidebar, render_main_styles
from src.ui.admin_dashboard import render_admin_view
from src.core.retrieval import generate_response
from src.core.feedback import save_feedback, log_conversation, load_chat_history 
# ─────────────────────────────────────────────────────────────────────────────
# 3. CONFIG & SESSION STATE
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AXIsstant", 
    page_icon="🦅", 
    layout="wide", 
    initial_sidebar_state="expanded"
)

# Initialize all session variables
if "session_id" not in st.session_state: st.session_state["session_id"] = str(uuid.uuid4())
if "authenticated" not in st.session_state: st.session_state["authenticated"] = False
if "messages" not in st.session_state: st.session_state["messages"] = []
if "chat_history_loaded" not in st.session_state: st.session_state["chat_history_loaded"] = False
if "view" not in st.session_state: st.session_state["view"] = "chat"
if "user_id" not in st.session_state: st.session_state["user_id"] = None
if "role" not in st.session_state: st.session_state["role"] = "student"
if "chat_history" not in st.session_state: st.session_state["chat_history"] = []
if "active_convo_idx" not in st.session_state: st.session_state["active_convo_idx"] = None

# ─────────────────────────────────────────────────────────────────────────────
# 4. AUTHENTICATION GATE
# ─────────────────────────────────────────────────────────────────────────────
if not st.session_state["authenticated"]:
    render_login()
    st.stop()

if not st.session_state["chat_history_loaded"]:
    with st.spinner("Loading your past conversations..."):
        user_history = load_chat_history(st.session_state["user_id"])
        if user_history:
            st.session_state["chat_history"] = user_history
        st.session_state["chat_history_loaded"] = True # Mark as done

render_main_styles()
render_sidebar()

# ─────────────────────────────────────────────────────────────────────────────
# 5. VIEW CONTROLLER
# ─────────────────────────────────────────────────────────────────────────────

# --- OPTION A: ADMIN VIEW ---
if st.session_state["view"] == "admin" and st.session_state.get("role") == "admin":
    render_admin_view()

# --- OPTION B: HISTORY VIEW ---
elif st.session_state["view"] == "history":
    st.markdown("<h1 style='color: #F0A62D; font-weight: bold;'>Chat History</h1>", unsafe_allow_html=True)
    st.markdown("---")
    
    if not st.session_state.get("chat_history"):
        st.info("📭 No saved conversations yet. Start chatting and your history will appear here!")
    else:
        # Show history in reverse order (newest first)
        for i, conv in enumerate(reversed(st.session_state["chat_history"])):
            if not conv: continue
            
            # Get first user message as title
            first_msg = conv[0]["content"] if conv else "Empty Chat"
            title = first_msg[:50] + "..." if len(first_msg) > 50 else first_msg
            
            with st.expander(f"💬 {title}", expanded=False):
                for msg in conv:
                    role_icon = "🐙" if msg["role"] == "assistant" else "👤"
                    st.markdown(f"**{role_icon} {msg['role'].title()}:** {msg['content']}")
                
                # Buttons row
                col1, col2 = st.columns(2)
                actual_idx = len(st.session_state["chat_history"]) - 1 - i
                
                with col1:
                    # Button to load this conversation
                    if st.button("📂 Continue", key=f"load_{i}"):
                        st.session_state["messages"] = conv.copy()
                        st.session_state["active_convo_idx"] = actual_idx
                        
                        # 🚀 Generate NEW Session ID when reloading old chats so we don't mix logs
                        st.session_state["session_id"] = str(uuid.uuid4())
                        
                        st.session_state["view"] = "chat"
                        st.rerun()
                
                with col2:
                    # Button to delete this conversation
                    if st.button("🗑️ Delete", key=f"delete_{i}", type="secondary"):
                        st.session_state["chat_history"].pop(actual_idx)
                        # Reset active conversation if we deleted it
                        if st.session_state.get("active_convo_idx") == actual_idx:
                            st.session_state["active_convo_idx"] = None
                            st.session_state["messages"] = []
                        st.rerun()

# --- OPTION C: MAIN CHAT VIEW ---
else:
    st.markdown("<h1 style='color: #F0A62D; font-weight: bold;'>AXIsstant</h1>", unsafe_allow_html=True)

    # Empty State / Quick Start
    if not st.session_state.messages:
        st.info("👋 Welcome! Try asking: 'What is the grading system?' or 'How do I request an overload?'")

    # Display Chat History
    for message in st.session_state.messages:
        avatar = "assets/kraken_logo.png" if message["role"] == "assistant" else None
        with st.chat_message(message["role"], avatar=avatar):
            st.markdown(message["content"])
            if "timestamp" in message:
                st.markdown(f"<span class='timestamp'>{message['timestamp']}</span>", unsafe_allow_html=True)

    # Handle User Input
    if query := st.chat_input("Ask AXIsstant about rules, exemptions, or curriculum..."):
        
        # 1. Append User Message
        timestamp = datetime.now().strftime("%I:%M %p")
        st.session_state.messages.append({"role": "user", "content": query, "timestamp": timestamp})
        with st.chat_message("user"):
            st.markdown(query)
            st.markdown(f"<span class='timestamp'>{timestamp}</span>", unsafe_allow_html=True)
        
        # 2. Generate Response (STREAMING + LOGGING)
        with st.chat_message("assistant", avatar="assets/kraken_logo.png"):
            try:
                # Show thinking status while generating
                with st.status("🧠 Thinking...", expanded=True) as status:
                    st.write("Processing your question...")
                    
                    # Get the generator
                    stream = generate_response(
                        query=query, 
                        chat_history_list=st.session_state.messages
                    )
                    
                    # Collect the stream response
                    response_chunks = []
                    for chunk in stream:
                        response_chunks.append(chunk)
                    response = "".join(response_chunks)
                    
                    status.update(label="✅ Done!", state="complete", expanded=False)
                
                # Display the response
                st.markdown(response)
                current_context = st.session_state.get("last_retrieved_context", "")
                performance_metrics = st.session_state.get("performance_metrics", {})
                
                # 🚀 AUTO-LOG to Supabase (Updated with Performance Data)
                user_email = st.session_state.get("user_id", "Guest")
                session_id = st.session_state.get("session_id")
                
                # Passing metrics allows you to build the performance charts for your thesis
                log_conversation(
                    query=query, 
                    response=response, 
                    user_email=user_email, 
                    session_id=session_id, 
                    context=current_context,
                    metrics=performance_metrics
                )

            except Exception as e:
                response = f"⚠️ Backend Error: {str(e)}"
                st.error(response)
            
            # 3. Update Session State
            response_timestamp = datetime.now().strftime("%I:%M %p")
            st.markdown(f"<span class='timestamp'>{response_timestamp}</span>", unsafe_allow_html=True)
            st.session_state.messages.append({"role": "assistant", "content": response, "timestamp": response_timestamp})
            
            # 4. Update History List (UI persistence)
            if st.session_state.get("active_convo_idx") is not None:
                idx = st.session_state["active_convo_idx"]
                if 0 <= idx < len(st.session_state.get("chat_history", [])):
                    st.session_state["chat_history"][idx] = st.session_state["messages"].copy()
            
            elif len(st.session_state.messages) == 2: 
                 st.session_state["chat_history"].append(st.session_state["messages"].copy())
                 st.session_state["active_convo_idx"] = len(st.session_state["chat_history"]) - 1

            # 5. Feedback UI
            col1, _ = st.columns([1, 4])
            msg_idx = len(st.session_state.messages)
            with col1:
                sub1, sub2 = st.columns(2)
                with sub1:
                    if st.button("👍", key=f"g_{msg_idx}"):
                        save_feedback(query, response, "helpful", st.session_state.get("user_id"))
                        st.toast("Feedback recorded!")
                with sub2:
                    if st.button("👎", key=f"b_{msg_idx}"):
                        save_feedback(query, response, "not_helpful", st.session_state.get("user_id"))
                        st.toast("We'll improve this answer.", icon="📝")