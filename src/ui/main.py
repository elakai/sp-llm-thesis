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

# ── FIX: Added create_supabase_client to imports! ──
from src.core.auth import normalize_role, create_supabase_client 

from src.config.logging_config import logger
from src.config.settings import PINECONE_INDEX_NAME

def check_pinecone_health() -> bool:
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
render_main_styles()

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

# Bulletproof st.user lookup (handles Streamlit version quirks)
current_user = getattr(st, "user", getattr(st, "experimental_user", None))
google_email = getattr(current_user, "email", None) if current_user else None
if not google_email and isinstance(current_user, dict):
    google_email = current_user.get("email")

# 1. Check Database Health (Restored)
if not st.session_state["db_online"]:
    st.error("🚨 Database Connection Error. Please verify your Pinecone API Key.")
    st.stop()

# 2. Render Login Form
if not google_email and not st.session_state.get("authenticated"):
    render_login() 
    st.stop()

# 3. Catch the Google Redirect
if google_email and not st.session_state.get("authenticated"):
    
    # Enforce Domain Lock
    if not (google_email.endswith("@gbox.adnu.edu.ph") or google_email.endswith("@adnu.edu.ph")):
        st.error(f"🚨 Access Restricted: {google_email} is not a valid ADNU email.")
        if st.button("Log Out and Try Again"):
            st.logout()
        st.stop()

    # Database Lookup (Now works because import is fixed!)
    sb = create_supabase_client()
    try:
        profile = sb.table("users").select("role, full_name").eq("email", google_email).single().execute()
        role = normalize_role(profile.data.get("role"))
        full_name = profile.data.get("full_name")
    except Exception:
        # First-time user registration
        role = "student"
        google_name = getattr(current_user, "name", None) if current_user else None
        if not google_name and isinstance(current_user, dict):
            google_name = current_user.get("name")
            
        full_name = google_name or google_email.split("@")[0]
        try:
            sb.table("users").insert({"email": google_email, "full_name": full_name, "role": role}).execute()
        except Exception as db_e:
            logger.error(f"Could not auto-register user in DB: {db_e}")
    
    # Lock in Session
    st.session_state["authenticated"] = True
    st.session_state["email"] = google_email
    st.session_state["role"] = role
    st.session_state["full_name"] = full_name
    st.session_state["view"] = "admin" if role == "admin" else "chat"
    
    st.rerun()

# ── RESTORED: CHAT HISTORY LOADER ──
if not st.session_state["chat_history_loaded"]:
    history_owner = st.session_state.get("email") or st.session_state.get("user_id")
    user_history = load_chat_history(history_owner)
    if user_history:
        st.session_state["chat_history"] = user_history
        current_msgs = st.session_state.get("messages", [])
        if current_msgs and st.session_state.get("active_convo_idx") is None:
            for i, conv in enumerate(user_history):
                messages = conv.get("messages", []) if isinstance(conv, dict) else conv
                if messages and len(current_msgs) > 0:
                    if messages[0].get("content") == current_msgs[0].get("content"):
                        st.session_state["active_convo_idx"] = i
                        break
    st.session_state["chat_history_loaded"] = True


# ─────────────────────────────────────────────────────────────────────────────
# 6. VIEW CONTROLLER
# ─────────────────────────────────────────────────────────────────────────────
render_sidebar()

if st.session_state["view"] == "admin" and st.session_state.get("role") == "admin":
    render_admin_view()
elif st.session_state["view"] == "indexed_docs" and st.session_state.get("role") == "admin":
    render_indexed_documents_view()
elif st.session_state["view"] == "history":
    render_history_view()
else:
    render_chat_view()