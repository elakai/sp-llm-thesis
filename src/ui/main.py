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

# Load main styles immediately to prevent flash of unstyled content
render_main_styles()

# Initialize ALL your session variables exactly as you had them
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
        st.session_state["chat_history_loaded"] = True

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

# --- OPTION C: MAIN CHAT VIEW ---
else:
    st.markdown("<h1 style='color: #F0A62D; font-weight: bold;'>AXIsstant</h1>", unsafe_allow_html=True)

    if not st.session_state.messages:
        st.info("👋 Welcome! Try asking: 'What is the grading system?' or 'How do I request an overload?'")

    for message in st.session_state.messages:
        avatar = "assets/kraken_logo.png" if message["role"] == "assistant" else None
        with st.chat_message(message["role"], avatar=avatar):
            st.markdown(message["content"])
            if "timestamp" in message:
                st.markdown(f"<span style='font-size: 0.8em; color: gray;'>{message['timestamp']}</span>", unsafe_allow_html=True)

    if query := st.chat_input("Ask AXIsstant about rules, exemptions, or curriculum..."):
        
        user_ts = datetime.now().strftime("%I:%M %p")
        st.session_state.messages.append({"role": "user", "content": query, "timestamp": user_ts})
        with st.chat_message("user"):
            st.markdown(query)
            st.markdown(f"<span style='font-size: 0.8em; color: gray;'>{user_ts}</span>", unsafe_allow_html=True)
        
        with st.chat_message("assistant", avatar="assets/kraken_logo.png"):
            try:
                # 1. Show the "Thinking" status bar while backend starts
                with st.status("🧠 Thinking...", expanded=False) as status:
                    stream = generate_response(
                        query=query,
                        chat_history_list=st.session_state.messages
                    )
                
                # 2. RESTORED STREAMING: Use st.write_stream directly
                response = st.write_stream(stream)
                
                asst_ts = datetime.now().strftime("%I:%M %p")
                st.markdown(f"<span style='font-size: 0.8em; color: gray;'>{asst_ts}</span>", unsafe_allow_html=True)

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
            
            st.session_state.messages.append({"role": "assistant", "content": response, "timestamp": asst_ts})
            
            if st.session_state.get("active_convo_idx") is not None:
                idx = st.session_state["active_convo_idx"]
                if 0 <= idx < len(st.session_state["chat_history"]):
                    st.session_state["chat_history"][idx] = st.session_state["messages"].copy()
            
            elif len(st.session_state.messages) == 2:
                 st.session_state["chat_history"].append(st.session_state["messages"].copy())
                 st.session_state["active_convo_idx"] = len(st.session_state["chat_history"]) - 1

            # Feedback UI
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