import sys
from pathlib import Path
import uuid

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
from src.ui.views import render_history_view, render_chat_view
from src.core.feedback import load_chat_history

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
    render_history_view()

# --- OPTION C: MAIN CHAT VIEW ---
else:
    render_chat_view()