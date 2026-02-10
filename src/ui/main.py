import streamlit as st
from src.ui.components import render_login, render_sidebar, render_main_styles
from src.ui.admin_dashboard import render_admin_view
from src.core.retrieval import generate_response
from src.core.feedback import save_feedback

st.set_page_config(page_title="AXIsstant", page_icon="🦅", layout="wide", initial_sidebar_state="expanded")

if "authenticated" not in st.session_state: st.session_state["authenticated"] = False
if "messages" not in st.session_state: st.session_state["messages"] = []
if "view" not in st.session_state: st.session_state["view"] = "chat"
if "user_id" not in st.session_state: st.session_state["user_id"] = None
if "role" not in st.session_state: st.session_state["role"] = "student"
if "chat_history" not in st.session_state: st.session_state["chat_history"] = []  # List of past conversations
if "active_convo_idx" not in st.session_state: st.session_state["active_convo_idx"] = None  # Index of conversation being continued

if not st.session_state["authenticated"]:
    render_login()
    st.stop()

render_main_styles()
render_sidebar()

if st.session_state["view"] == "admin" and st.session_state.get("role") == "admin":
    render_admin_view()
elif st.session_state["view"] == "history":
    # History View - shows past conversations
    st.markdown("<h1 style='color: #F0A62D; font-weight: bold;'>Chat History</h1>", unsafe_allow_html=True)
    st.markdown("---")
    
    if not st.session_state.get("chat_history"):
        st.info("📭 No saved conversations yet. Start chatting and your history will appear here!")
    else:
        for i, conv in enumerate(reversed(st.session_state["chat_history"])):
            # Get first user message as title
            title = conv[0]["content"][:50] + "..." if conv and conv[0]["content"] else f"Conversation {i+1}"
            with st.expander(f"💬 {title}", expanded=False):
                for msg in conv:
                    role_icon = "🐙" if msg["role"] == "assistant" else "👤"
                    st.markdown(f"**{role_icon} {msg['role'].title()}:** {msg['content']}")
                
                # Button to load this conversation
                if st.button("📂 Continue this chat", key=f"load_{i}"):
                    # Calculate actual index (since we reversed the list for display)
                    actual_idx = len(st.session_state["chat_history"]) - 1 - i
                    st.session_state["messages"] = conv.copy()
                    st.session_state["active_convo_idx"] = actual_idx
                    st.session_state["view"] = "chat"
                    st.rerun()
else:
    st.markdown("<h1 style='color: #F0A62D; font-weight: bold;'>AXIsstant</h1>", unsafe_allow_html=True)

    # --- Empty State / Quick Start ---
    if not st.session_state.messages:
        st.info("👋 Welcome! Try asking: 'What is the grading system?' or 'How do I request an overload?'")

    for message in st.session_state.messages:
        avatar = "assets/kraken_logo.png" if message["role"] == "assistant" else None
        with st.chat_message(message["role"], avatar=avatar):
            st.markdown(message["content"])

    if query := st.chat_input("Ask AXIsstant about rules, exemptions, or curriculum..."):
        st.session_state.messages.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.markdown(query)
        
        with st.chat_message("assistant", avatar="assets/kraken_logo.png"):
            with st.spinner("Searching Documents..."):
                try:
                    response = generate_response(query=query, chat_history_list=st.session_state.messages)
                except Exception as e:
                    response = f"⚠️ Backend Error: {str(e)}"
            
            st.markdown(response, unsafe_allow_html=True)
            st.session_state.messages.append({"role": "assistant", "content": response})
            
            # Update history if continuing an existing conversation
            if st.session_state.get("active_convo_idx") is not None:
                idx = st.session_state["active_convo_idx"]
                if 0 <= idx < len(st.session_state.get("chat_history", [])):
                    st.session_state["chat_history"][idx] = st.session_state["messages"].copy()

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