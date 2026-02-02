import sys
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
# 1. PATH SETUP (MUST BE FIRST)
# ─────────────────────────────────────────────────────────────────────────────
# This tells Python where to look for your 'src' folder
project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# ─────────────────────────────────────────────────────────────────────────────
# 2. IMPORTS
# ─────────────────────────────────────────────────────────────────────────────
import streamlit as st
from src.ui.components import render_login, render_sidebar, render_main_styles
from src.ui.admin_dashboard import render_admin_view  # Importing the separate admin file
from src.core.retrieval import generate_response
from src.core.feedback import save_feedback

# ─────────────────────────────────────────────────────────────────────────────
# 3. APP CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AXIsstant", 
    page_icon="🦅", 
    layout="wide", 
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────────────────────────────────────
# 4. SESSION STATE INITIALIZATION
# ─────────────────────────────────────────────────────────────────────────────
if "authenticated" not in st.session_state:
    st.session_state["authenticated"] = False

if "messages" not in st.session_state:
    st.session_state.messages = []

if "view" not in st.session_state:
    st.session_state["view"] = "chat"  # Default view is 'chat'

if "user_id" not in st.session_state:
    st.session_state["user_id"] = None

if "role" not in st.session_state:
    st.session_state["role"] = "student"

# ─────────────────────────────────────────────────────────────────────────────
# 5. AUTHENTICATION GATE
# ─────────────────────────────────────────────────────────────────────────────
# If the user is NOT logged in, show the Login Screen and stop the script.
if not st.session_state["authenticated"]:
    render_login()
    st.stop()

# ─────────────────────────────────────────────────────────────────────────────
# 6. MAIN APP LOGIC (Only runs if authenticated)
# ─────────────────────────────────────────────────────────────────────────────
render_main_styles()  # Apply Global CSS
render_sidebar()      # Show the Navigation Sidebar

# ROUTING: Decide which screen to show based on the 'view' state
if st.session_state["view"] == "admin" and st.session_state.get("role") == "admin":
    # 🅰️ SHOW ADMIN DASHBOARD
    render_admin_view()

else:
    # 🅱️ SHOW CHAT INTERFACE
    st.markdown("<h1 style='color: #F0A62D; font-weight: bold;'>AXIsstant</h1>", unsafe_allow_html=True)

    # 1. Display Chat History
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # 2. Handle User Input
    if query := st.chat_input("Ask AXIsstant about rules, exemptions, or curriculum..."):
        
        # Add User Message to History
        st.session_state.messages.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.markdown(query)
        
        # Generate Assistant Response
        with st.chat_message("assistant"):
            with st.spinner("Searching official handbook..."):
                try:
                    # Call the backend RAG engine
                    response = generate_response(
                        query=query,
                        chat_history_list=st.session_state.messages
                    )
                except Exception as e:
                    response = f"⚠️ Backend Error: {str(e)}"
            
            st.markdown(response, unsafe_allow_html=True)
            st.session_state.messages.append({"role": "assistant", "content": response})

            # 3. Feedback Buttons (Thumbs Up/Down)
            with st.container():
                col1, col2 = st.columns([1, 4])
                msg_index = len(st.session_state.messages)
                current_user = st.session_state.get("user_id")
                
                with col1:
                    sub1, sub2 = st.columns(2)
                    with sub1:
                        if st.button("👍", key=f"good_{msg_index}", help="This helped"):
                            save_feedback(query, response, "helpful", current_user)
                            st.toast("Thanks for the feedback!", icon="✅")
                    with sub2:
                        if st.button("👎", key=f"bad_{msg_index}", help="Not helpful"):
                            save_feedback(query, response, "not_helpful", current_user)
                            st.toast("We'll improve this answer.", icon="📝")