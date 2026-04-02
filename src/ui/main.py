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
from src.ui.guest_chat import render_guest_chat_view
from src.core.feedback import load_chat_history
from src.core.auth import normalize_role
from src.config.logging_config import logger
from src.config.settings import PINECONE_INDEX_NAME


@st.cache_data(ttl=120, show_spinner=False)
def cached_pinecone_health() -> bool:
    return check_pinecone_health()


@st.cache_resource(show_spinner=False)
def warmup_runtime_resources() -> None:
    from src.config.settings import get_embeddings, get_generator_llm
    from src.core.retrieval import get_reranker

    get_embeddings()
    get_generator_llm()
    get_reranker()


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
if "is_guest" not in st.session_state: st.session_state["is_guest"] = False
if "messages" not in st.session_state: st.session_state["messages"] = []
if "chat_history_loaded" not in st.session_state: st.session_state["chat_history_loaded"] = False
if "view" not in st.session_state: st.session_state["view"] = "chat"
if "user_id" not in st.session_state: st.session_state["user_id"] = None
if "role" not in st.session_state: st.session_state["role"] = "student"
if "chat_history" not in st.session_state: st.session_state["chat_history"] = []
if "active_convo_idx" not in st.session_state: st.session_state["active_convo_idx"] = None
if "sidebar_open" not in st.session_state: st.session_state["sidebar_open"] = True
if "db_online" not in st.session_state:
    st.session_state["db_online"] = cached_pinecone_health()

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

# Warm heavy runtime resources once per server process instead of once per user session.
warmup_runtime_resources()
if "app_loaded" not in st.session_state:
    st.session_state["app_loaded"] = True
    logger.info(f"App cold start completed in {time.time() - _app_start_time:.1f}s")

# ─────────────────────────────────────────────────────────────────────────────
# 5. AUTHENTICATION GATE
# ─────────────────────────────────────────────────────────────────────────────

if not st.session_state["db_online"]:
    st.error("🚨 Database Connection Error. Please verify your Pinecone API Key and Index status.")
    st.stop()

# Keep guest sessions authenticated so login/signup never reappears,
# while still allowing normal view routing (chat/history) like users.
if st.session_state.get("is_guest"):
    st.session_state["authenticated"] = True

if st.session_state.get("authenticated"):
    st.session_state["role"] = normalize_role(st.session_state.get("role"))

if not st.session_state["authenticated"]:
    render_login()
    st.stop()

# Render sidebar for all authenticated users (including guests)
render_sidebar()

role = normalize_role(st.session_state.get("role"))
st.session_state["role"] = role
current_view = st.session_state.get("view", "chat")

# Defense-in-depth role guard for route/view state.
if role != "admin" and current_view in {"admin", "indexed_docs"}:
    st.session_state["view"] = "chat"
    logger.warning(
        "Blocked non-admin user %s from admin view '%s'; rerouted to chat.",
        st.session_state.get("email", "unknown"),
        current_view,
    )
elif role == "admin" and current_view not in {"admin", "indexed_docs", "chat"}:
    st.session_state["view"] = "admin"
    logger.info(
        "Corrected admin user %s from stale view '%s' to admin dashboard.",
        st.session_state.get("email", "unknown"),
        current_view,
    )

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

# ─────────────────────────────────────────────────────────────────────────────
# 6. VIEW CONTROLLER
# ─────────────────────────────────────────────────────────────────────────────

# --- OPTION A: GUEST VIEWS ---
if st.session_state.get("is_guest") and st.session_state["view"] == "history":
    render_history_view()

elif st.session_state.get("is_guest"):
    render_guest_chat_view()

# --- OPTION B: ADMIN VIEW ---
elif st.session_state["view"] == "admin" and st.session_state.get("role") == "admin":
    render_admin_view()

# --- OPTION C: INDEXED DOCUMENTS VIEW ---
elif st.session_state["view"] == "indexed_docs" and st.session_state.get("role") == "admin":
    render_indexed_documents_view()

# --- OPTION D: HISTORY VIEW ---
elif st.session_state["view"] == "history":
    render_history_view()

# --- OPTION E: MAIN CHAT VIEW ---
else:
    render_chat_view()