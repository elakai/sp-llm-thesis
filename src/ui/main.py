import sys
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
# 1. PATH SETUP (MUST BE FIRST)
# ─────────────────────────────────────────────────────────────────────────────
project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# ─────────────────────────────────────────────────────────────────────────────
# 2. IMPORTS
# ─────────────────────────────────────────────────────────────────────────────
import streamlit as st
# We use the NEW components (Login, Sidebar, Styles)
from src.ui.components import render_login, render_sidebar_nav, render_chat_styles
from src.core.retrieval import generate_response
from src.core.feedback import save_feedback

# ─────────────────────────────────────────────────────────────────────────────
# 3. CONFIGURATION (From Front End Push)
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AXIsstant", 
    page_icon="🏛️", 
    layout="wide", 
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────────────────────────────────────
# 4. SESSION STATE & AUTHENTICATION
# ─────────────────────────────────────────────────────────────────────────────
if "authenticated" not in st.session_state:
    st.session_state["authenticated"] = False

if "messages" not in st.session_state:
    st.session_state.messages = []

# AUTH GATE: Stop everything else if not logged in
if not st.session_state["authenticated"]:
    render_login()
    st.stop()  # Prevents the rest of the app from running

# ─────────────────────────────────────────────────────────────────────────────
# 5. MAIN APP UI (Only runs if authenticated)
# ─────────────────────────────────────────────────────────────────────────────
render_chat_styles()

# Sidebar (Pass the user ID to show profile)
current_user = st.session_state.get("user_id", "Student")
render_sidebar_nav(current_user)

# Header
st.markdown("<h1 style='color: #F0A62D; font-weight: bold;'>AXIsstant</h1>", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# 6. CHAT LOGIC
# ─────────────────────────────────────────────────────────────────────────────

# Display History
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Handle Input
if query := st.chat_input("Ask AXIsstant about rules, exemptions, or curriculum..."):
    
    # 1. Display User Message
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)
    
    # 2. Generate Response
    with st.chat_message("assistant"):
        with st.spinner("Searching official handbook..."):
            try:
                # ✅ MERGE FIX: We put 'chat_history_list' BACK because 
                # we fixed retrieval.py to accept it.
                response = generate_response(
                    query=query,
                    chat_history_list=st.session_state.messages
                )
            except Exception as e:
                response = f"⚠️ Backend Error: {str(e)}"
        
        st.markdown(response, unsafe_allow_html=True)
        st.session_state.messages.append({"role": "assistant", "content": response})

        # 3. Feedback Mechanism (From your Main Code - Better UI)
        with st.container():
            col1, col2 = st.columns([1, 4])
            msg_index = len(st.session_state.messages)
            
            with col1:
                sub_col1, sub_col2 = st.columns(2)
                with sub_col1:
                    if st.button("👍", key=f"good_{msg_index}", help="Helpful"):
                        save_feedback(query, response, "helpful", current_user)
                        st.toast("Thanks!", icon="✅")
                
                with sub_col2:
                    if st.button("👎", key=f"bad_{msg_index}", help="Not Helpful"):
                        save_feedback(query, response, "not_helpful", current_user)
                        st.toast("Noted.", icon="📝")