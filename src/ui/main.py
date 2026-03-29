import sys
import time
from pathlib import Path
import uuid
import streamlit as st
import streamlit.components.v1 as st_components # <-- Added this!

_app_start_time = time.time()

# ─────────────────────────────────────────────────────────────────────────────
# 1. CONFIG (CRITICAL: Must be the very first Streamlit command)
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AXIstant",
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
from src.ui.document_management import render_indexed_documents_view
from src.ui.views import render_history_view, render_chat_view
from src.core.feedback import load_chat_history
from src.core.auth import supabase as _sb, create_supabase_client, normalize_role
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
mobile_sidebar_width = "280px" if st.session_state["sidebar_open"] else "0px"
content_gutter = "3.5rem"
st.markdown(
    f"""
    <style>
    :root {{ --axi-sidebar-width: {sidebar_width}; --axi-mobile-sidebar-width: {mobile_sidebar_width}; --axi-content-gutter: {content_gutter}; }}
    </style>
    """,
    unsafe_allow_html=True,
)

# ── FIX: REPLACED SPINNER BLOCK WITH FASTER COLD START ──
if "app_loaded" not in st.session_state:
    from src.config.settings import get_embeddings, get_generator_llm
    from src.core.retrieval import get_reranker
    get_embeddings()
    get_generator_llm()
    get_reranker()
    st.session_state["app_loaded"] = True
    logger.info(f"App cold start completed in {time.time() - _app_start_time:.1f}s")


# ─────────────────────────────────────────────────────────────────────────────
# 4.5. THE JAVASCRIPT HASH CONVERTER (Runs in the browser)
# Streamlit can't read '#'. This script catches the raw tokens, flips '#' to '?', 
# and instantly reloads so Python can read them.
# ─────────────────────────────────────────────────────────────────────────────
st_components.html(
    """
    <script>
    if (window.location.hash && window.location.hash.includes("access_token")) {
        let newUrl = window.location.href.replace("#", "?");
        window.location.replace(newUrl);
    }
    </script>
    """,
    height=0
)

# ─────────────────────────────────────────────────────────────────────────────
# 4.6. THE DIRECT TOKEN CATCHER (Needs NO memory!)
# ─────────────────────────────────────────────────────────────────────────────
if "access_token" in st.query_params and "refresh_token" in st.query_params and not st.session_state.get("authenticated"):
    st.warning("🔄 Tokens detected! Setting up your session... (Please wait)")
    try:
        access_token = st.query_params["access_token"]
        refresh_token = st.query_params["refresh_token"]

        # Create a brand new client and feed it the raw tokens directly
        supabase_client = create_supabase_client()
        session = supabase_client.auth.set_session(access_token, refresh_token)
        
        user = session.user
        role = normalize_role(user.user_metadata.get("role", "student")) 
        full_name = user.user_metadata.get("full_name", user.email.split("@")[0])

        # Log the user in
        st.session_state["authenticated"] = True
        st.session_state["user_id"] = user.id
        st.session_state["email"] = user.email
        st.session_state["role"] = role
        st.session_state["full_name"] = full_name
        st.session_state["show_welcome"] = True
        
        if "session_id" not in st.session_state:
            st.session_state["session_id"] = str(uuid.uuid4())
            
        st.session_state["view"] = "admin" if role == "admin" else "chat"
        
        # Clean up the ugly URL parameters
        st.query_params.clear()
        st.rerun()
        
    except Exception as e:
        st.error(f"🚨 Token authentication failed! Error details: {e}")
        st.stop()


# ─────────────────────────────────────────────────────────────────────────────
# 5. AUTHENTICATION GATE
# ─────────────────────────────────────────────────────────────────────────────

if not st.session_state["db_online"]:
    st.error("🚨 Database Connection Error. Please verify your Pinecone API Key and Index status.")
    st.stop()

if not st.session_state["authenticated"]:
    render_login()
    st.stop()

if not st.session_state["chat_history_loaded"]:
    # chat_logs are stored by user_email, so load with email for refresh persistence.
    history_owner = st.session_state.get("email") or st.session_state.get("user_id")
    user_history = load_chat_history(history_owner)
    if user_history:
        st.session_state["chat_history"] = user_history
        
        # If there are current messages, try to find matching conversation
        current_msgs = st.session_state.get("messages", [])
        if current_msgs and st.session_state.get("active_convo_idx") is None:
            # Look for a conversation that matches current messages
            for i, conv in enumerate(user_history):
                messages = conv.get("messages", []) if isinstance(conv, dict) else conv
                if messages and len(current_msgs) > 0:
                    # Match by first message content
                    if messages[0].get("content") == current_msgs[0].get("content"):
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

# --- OPTION B: INDEXED DOCUMENTS VIEW ---
elif st.session_state["view"] == "indexed_docs" and st.session_state.get("role") == "admin":
    render_indexed_documents_view()

# --- OPTION C: HISTORY VIEW ---
elif st.session_state["view"] == "history":
    render_history_view()

# --- OPTION D: MAIN CHAT VIEW ---
else:
    render_chat_view()