# src/ui/main.py
import streamlit as st
from src.ui.components import render_login, render_sidebar_nav, render_chat_styles
from src.core.retrieval import generate_response

st.set_page_config(
    page_title="AXIsstant", 
    page_icon="🏛️", 
    layout="wide", 
    initial_sidebar_state="expanded" # This is mandatory to keep it visible
)

if "authenticated" not in st.session_state: st.session_state["authenticated"] = False
if "messages" not in st.session_state: st.session_state.messages = []

if not st.session_state["authenticated"]:
    render_login()
    st.stop()

# --- CHAT UI ---
render_chat_styles()
render_sidebar_nav(st.session_state.get("user_id", "Student"))

# White Header Bar
st.markdown("<h1 style='color: #F0A62D; font-weight: bold;'>AXIsstant</h1>", unsafe_allow_html=True)

# Message Display Loop
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Fixed Chat Logic: Resolves 'unexpected keyword argument'
if query := st.chat_input("Ask AXIsstant..."):
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)
    
    with st.chat_message("assistant"):
        with st.spinner("Searching official handbook..."):
            try:
                # REMOVED chat_history_list to fix the error
                response = generate_response(query) 
            except Exception as e:
                response = f"⚠️ Backend Error: {str(e)}"
            st.markdown(response)
    
    st.session_state.messages.append({"role": "assistant", "content": response})
    st.rerun()