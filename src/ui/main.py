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
from src.core.auth import normalize_role
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
# 5. AUTHENTICATION GATE
# ─────────────────────────────────────────────────────────────────────────────

# 1. RENDER YOUR CUSTOM UI (Tabs, Email, Password, AND Google Button)
# We check both the native Google state AND your manual email/password session state
if not st.user.is_logged_in and not st.session_state.get("authenticated"):
    render_login() # <--- This brings your tabs and CSS back!
    st.stop()

# 2. CATCH THE NATIVE GOOGLE REDIRECT
if st.user.is_logged_in and not st.session_state.get("authenticated"):
    email = st.user.email

    # Enforce the ADNU Domain Lock
    if not (email.endswith("@gbox.adnu.edu.ph") or email.endswith("@adnu.edu.ph")):
        st.error(f"🚨 Access Restricted: {email} is not a valid ADNU Gbox email.")
        if st.button("Log Out and Try Again"):
            st.logout()
        st.stop()

    # Fetch the user's role from Supabase and set session state
    sb = create_supabase_client()
    try:
        profile = sb.table("users").select("role, full_name").eq("email", email).single().execute()
        role = normalize_role(profile.data.get("role"))
        full_name = profile.data.get("full_name")
    except Exception:
        # FIRST TIME LOGIN: Default to student and auto-register them in the DB
        role = "student"
        full_name = st.user.get("name") or email.split("@")[0]
        try:
            sb.table("users").insert({"email": email, "full_name": full_name, "role": role}).execute()
        except Exception as db_e:
            logger.error(f"Could not auto-register user in DB: {db_e}")
    
    # Lock in the session state for the rest of the app
    st.session_state["authenticated"] = True
    st.session_state["email"] = email
    st.session_state["role"] = role
    st.session_state["full_name"] = full_name
    st.session_state["view"] = "admin" if role == "admin" else "chat"
    
    st.rerun()

# ─────────────────────────────────────────────────────────────────────────────

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