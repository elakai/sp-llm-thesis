# src/ui/main.py
import streamlit as st
from src.ui.components import render_login, render_sidebar_nav, render_chat_styles
from src.core.retrieval import generate_response

# MUST BE FIRST Streamlit command
st.set_page_config(
    page_title="CSEA Assistant",
    page_icon="Eagle",
    layout="centered"
)

render_header()
render_admin_panel()

# Placeholder for future login (you'll replace this later)
# For now, assume no user_id (anonymous feedback)
user_id = st.session_state.get("user_id")  # will be None until login is added

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Hello! I am the CSEA Information Assistant.\n\nAsk me anything about CSEA rules, dress code, typhoon guidelines, or exemptions!"
        }
    ]

# Display existing messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User input
if query := st.chat_input("Ask about typhoon uniform rule, pregnancy exemption, dress code, etc."):
    # Add user message to history and display
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    # Generate assistant response
    with st.chat_message("assistant"):
        with st.spinner("Searching official handbook..."):
            try:
                response = generate_response(query)
            except Exception as e:
                response = f"Sorry, something went wrong while generating the answer: {str(e)}"

        st.markdown(response, unsafe_allow_html=True)

        # Feedback buttons
        col1, col2 = st.columns([1, 1])
        with col1:
            good_key = f"good_{len(st.session_state.messages)}"
            if st.button("Helpful", key=good_key):
                success = save_feedback(
                    user_id=user_id,
                    query=query,
                    answer=response,
                    rating="helpful"
                )
                if success:
                    st.success("Thank you for the feedback!")
                else:
                    st.warning("Feedback saved, but there was an issue — we'll still use it!")

        with col2:
            bad_key = f"bad_{len(st.session_state.messages)}"
            if st.button("Not helpful", key=bad_key):
                success = save_feedback(
                    user_id=user_id,
                    query=query,
                    answer=response,
                    rating="not_helpful"
                )
                if success:
                    st.info("Thanks — we'll improve!")
                else:
                    st.warning("Feedback saved, but there was an issue — we'll still use it!")

    # Add assistant message to history
    st.session_state.messages.append({"role": "assistant", "content": response})