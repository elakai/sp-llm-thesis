import streamlit as st
from src.ui.components import render_login, render_sidebar, render_main_styles
from src.ui.admin_dashboard import render_admin_view
from src.core.retrieval import generate_response
from src.core.feedback import save_feedback

st.set_page_config(page_title="AXIsstant", page_icon="🦅", layout="wide", initial_sidebar_state="expanded")

if "authenticated" not in st.session_state: st.session_state["authenticated"] = False
if "messages" not in st.session_state: st.session_state.messages = []
if "view" not in st.session_state: st.session_state["view"] = "chat"
if "user_id" not in st.session_state: st.session_state["user_id"] = None
if "role" not in st.session_state: st.session_state["role"] = "student"

if not st.session_state["authenticated"]:
    render_login()
    st.stop()

render_main_styles()
render_sidebar()

if st.session_state["view"] == "admin" and st.session_state.get("role") == "admin":
    render_admin_view()
else:
    st.markdown("<h1 style='color: #F0A62D; font-weight: bold;'>AXIsstant</h1>", unsafe_allow_html=True)

    # --- Empty State / Quick Start ---
    if not st.session_state.messages:
        st.info("👋 Welcome! Try asking: 'What are the rules for late enrollment?' or 'How do I request an overload?'")

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if query := st.chat_input("Ask AXIsstant about rules, exemptions, or curriculum..."):
        st.session_state.messages.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.markdown(query)
        
        with st.chat_message("assistant"):
            with st.spinner("Searching official handbook..."):
                try:
                    response = generate_response(query=query, chat_history_list=st.session_state.messages)
                except Exception as e:
                    response = f"⚠️ Backend Error: {str(e)}"
            
            st.markdown(response, unsafe_allow_html=True)
            st.session_state.messages.append({"role": "assistant", "content": response})

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