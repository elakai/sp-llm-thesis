import sys
import time
from pathlib import Path
import uuid
import streamlit as st

_app_start_time = time.time()

# ─────────────────────────────────────────────────────────────────────────────
# 1. CONFIG (CRITICAL: Must be the very first Streamlit command)
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AXIsstant",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────────────────────────────────────
# 2. PATH SETUP 
# ─────────────────────────────────────────────────────────────────────────────
project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# ─────────────────────────────────────────────────────────────────────────────
# 3. IMPORTS (Local modules must be imported AFTER set_page_config)
# ─────────────────────────────────────────────────────────────────────────────
from src.ui.components import render_login, render_sidebar, render_main_styles
from src.ui.admin_dashboard import render_admin_view
from src.ui.views import render_history_view, render_chat_view
from src.core.feedback import load_chat_history
from src.core.auth import supabase as _sb
from src.config.logging_config import logger
from src.config.settings import PINECONE_INDEX_NAME


def check_pinecone_health() -> bool:
    """Lightweight health check — avoids importing heavy ingestion module."""
    try:
        import os
        from pinecone import Pinecone
        pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))
        index = pc.Index(PINECONE_INDEX_NAME)
        stats = index.describe_index_stats()
        logger.info(f"Pinecone healthy: {stats.total_vector_count} vectors indexed")
        return True
    except Exception as e:
        logger.error(f"Pinecone Health Check Failed: {e}")
        return False


# ─────────────────────────────────────────────────────────────────────────────
# 4. SESSION STATE
# ─────────────────────────────────────────────────────────────────────────────
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
if "sidebar_open" not in st.session_state: st.session_state["sidebar_open"] = True
if "db_online" not in st.session_state:
    st.session_state["db_online"] = check_pinecone_health()

sidebar_width = "280px" if st.session_state["sidebar_open"] else "92px"
content_gutter = "3.5rem"
st.markdown(
    f"""
    <style>
    :root {{ --axi-sidebar-width: {sidebar_width}; --axi-content-gutter: {content_gutter}; }}
    </style>
    """,
    unsafe_allow_html=True,
)

# Pre-load heavy ML resources on first cold start
if "app_loaded" not in st.session_state:
    with st.spinner("Starting AXIsstant… please wait"):
        from src.config.settings import get_embeddings, get_generator_llm
        get_embeddings()    # triggers model download/cache on first run
        get_generator_llm()
        st.session_state["app_loaded"] = True
    logger.info(f"App cold start completed in {time.time() - _app_start_time:.1f}s")
# ─────────────────────────────────────────────────────────────────────────────
# 5. AUTHENTICATION GATE
# ─────────────────────────────────────────────────────────────────────────────

if not st.session_state["db_online"]:
    st.error("🚨 Database Connection Error. Please verify your Pinecone API Key and Index status.")
    st.stop()

# Attempt to restore a previous Supabase session (survives page refreshes)
if not st.session_state["authenticated"]:
    try:
        session = _sb.auth.get_session()
        if session and session.user:
            user_id = session.user.id
            try:
                profile = _sb.table("users") \
                    .select("role, full_name") \
                    .eq("id", user_id) \
                    .single() \
                    .execute()
                db_role = profile.data.get("role", "student")
                db_name = profile.data.get("full_name", "Student")
            except Exception:
                db_role, db_name = "student", "Student"

            st.session_state["authenticated"] = True
            st.session_state["user_id"] = user_id
            st.session_state["email"] = session.user.email
            st.session_state["role"] = db_role
            st.session_state["full_name"] = db_name
            logger.info(f"Session restored for {session.user.email}")
    except Exception:
        pass  # No session to restore — show login page

if not st.session_state["authenticated"]:
    render_login()
    st.stop()

if not st.session_state["chat_history_loaded"]:
    user_history = load_chat_history(st.session_state["user_id"])
    if user_history:
        st.session_state["chat_history"] = user_history
        
        # If there are current messages, try to find matching conversation
        current_msgs = st.session_state.get("messages", [])
        if current_msgs and st.session_state.get("active_convo_idx") is None:
            # Look for a conversation that matches current messages
            for i, conv in enumerate(user_history):
                if conv and len(conv) > 0 and len(current_msgs) > 0:
                    # Match by first message content
                    if conv[0].get("content") == current_msgs[0].get("content"):
                        st.session_state["active_convo_idx"] = i
                        break
    st.session_state["chat_history_loaded"] = True

render_sidebar()

# ─────────────────────────────────────────────────────────────────────────────
# 6. VIEW CONTROLLER
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

    