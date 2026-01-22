# src/ui/main.py
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# ────────────────────────────────────────────────
# Now normal imports should work
import streamlit as st
from src.ui.components import render_header, render_admin_panel
from src.core.retrieval import generate_response
from src.core.feedback import save_feedback

st.set_page_config(
    page_title="CSEA Assistant",
    page_icon="Eagle",
    layout="centered"
)

render_header()
render_admin_panel()

# Chat history
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello! I am the CSEA Information Assistant.\n\nAsk me anything!"}
    ]

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if query := st.chat_input("Ask about typhoon uniform rule, pregnancy exemption, dress code, etc."):
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        with st.spinner("Searching official handbook..."):
            response = generate_response(query)

        st.markdown(response)

        col1, col2 = st.columns([1, 1])
        with col1:
            if st.button("Helpful", key=f"good_{len(st.session_state.messages)}"):
                save_feedback(query, response, "good")
                st.success("Thank you!")
        with col2:
            if st.button("Not helpful", key=f"bad_{len(st.session_state.messages)}"):
                save_feedback(query, response, "bad")
                st.error("We'll improve!")

    st.session_state.messages.append({"role": "assistant", "content": response})