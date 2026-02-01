import sys
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
# Path Configuration (Must be at the very top)
# ─────────────────────────────────────────────────────────────────────────────
# Add project root to sys.path so we can import from src.
project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# ─────────────────────────────────────────────────────────────────────────────
# Imports
# ─────────────────────────────────────────────────────────────────────────────
import streamlit as st
from src.ui.components import render_header, render_admin_panel
from src.core.retrieval import generate_response
from src.core.feedback import save_feedback

# ─────────────────────────────────────────────────────────────────────────────
# Streamlit Config (Must be the first Streamlit command)
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CSEA Assistant",
    page_icon="🦅",
    layout="centered"
)

# ─────────────────────────────────────────────────────────────────────────────
# UI Initialization
# ─────────────────────────────────────────────────────────────────────────────
render_header()
render_admin_panel()

# Initialize Session State for Chat History
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Hello! I am the CSEA Information Assistant.\n\nAsk me anything about CSEA rules, dress code, typhoon guidelines, or exemptions!"
        }
    ]

# Initialize Session State for User ID (Placeholder for future login)
if "user_id" not in st.session_state:
    st.session_state.user_id = None  # Currently anonymous

# ─────────────────────────────────────────────────────────────────────────────
# Chat Interface
# ─────────────────────────────────────────────────────────────────────────────

# 1. Display existing chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 2. Handle User Input
if query := st.chat_input("Ask about typhoon uniform rule, pregnancy exemption, dress code, etc."):
    
    # A. Display User Message
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    # B. Generate Assistant Response
    with st.chat_message("assistant"):
        with st.spinner("Searching official handbook..."):
            try:
                # ⚡ CRITICAL FIX: Pass the session history to the retrieval engine
                response = generate_response(
                    query=query, 
                    chat_history_list=st.session_state.messages
                )
            except Exception as e:
                response = f"⚠️ Sorry, I encountered an error: {str(e)}"

        st.markdown(response, unsafe_allow_html=True)
        
        # Add Assistant Message to History
        st.session_state.messages.append({"role": "assistant", "content": response})

        # C. Feedback Mechanism (Buttons appear only after a new response)
        # Using a container to keep buttons compact
        with st.container():
            col1, col2 = st.columns([1, 4]) # Small column for helpful, rest for spacing
            
            # Unique keys for buttons based on message index to prevent duplicate ID errors
            msg_index = len(st.session_state.messages)
            
            with col1:
                # We use a simple popover or columns for thumbs up/down
                sub_col1, sub_col2 = st.columns(2)
                with sub_col1:
                    if st.button("👍", key=f"good_{msg_index}", help="This answer was helpful"):
                        save_feedback(
                            query=query,
                            answer=response,
                            rating="helpful",
                            user_id=st.session_state.user_id
                        )
                        st.toast("Thanks for the feedback!", icon="✅")
                
                with sub_col2:
                    if st.button("👎", key=f"bad_{msg_index}", help="This answer was not helpful"):
                        save_feedback(
                            query=query,
                            answer=response,
                            rating="not_helpful",
                            user_id=st.session_state.user_id
                        )
                        st.toast("We'll improve this answer.", icon="📝")