import sys
import time
from pathlib import Path
import uuid
import streamlit as st

_app_start_time = time.time()

# 1. CONFIG
st.set_page_config(page_title="AXIstant", layout="wide", initial_sidebar_state="expanded")

project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# 2. IMPORTS
from src.ui.components import render_login, render_sidebar, render_main_styles
from src.ui.admin_dashboard import render_admin_view
from src.ui.document_management import render_indexed_documents_view
from src.ui.views import render_history_view, render_chat_view
from src.core.feedback import load_chat_history
from src.core.auth import normalize_role, create_supabase_client 
from src.config.logging_config import logger
from src.config.settings import PINECONE_INDEX_NAME

def _is_valid_adnu_email(email: str) -> bool:
    email = (email or "").lower().strip()
    return email.endswith("@gbox.adnu.edu.ph") or email.endswith("@adnu.edu.ph")

# 3. SESSION STATE
render_main_styles()
if "authenticated" not in st.session_state: st.session_state["authenticated"] = False
if "google_auth_initiated" not in st.session_state: st.session_state["google_auth_initiated"] = False
if "view" not in st.session_state: st.session_state["view"] = "chat"
if "role" not in st.session_state: st.session_state["role"] = "student"
if "chat_history_loaded" not in st.session_state: st.session_state["chat_history_loaded"] = False

# ─────────────────────────────────────────────────────────────────────────────
# 4. AUTHENTICATION GATE
# ─────────────────────────────────────────────────────────────────────────────
google_email = None
try:
    if hasattr(st, "user") and st.user and getattr(st.user, "is_logged_in", False):
        google_email = st.user.email
except Exception:
    pass

# Case 1: Google OAuth returned an email AND user explicitly clicked our button
if google_email and not st.session_state.get("authenticated"):
    if not st.session_state.get("google_auth_initiated"):
        # Live session exists but button wasn't clicked. Clear Google context.
        try:
            st.logout()
        except Exception:
            pass
        render_login()
        st.stop()

    # Reset flag so it doesn't persist across future logouts
    st.session_state["google_auth_initiated"] = False

    if not _is_valid_adnu_email(google_email):
        try:
            st.logout()
        except Exception:
            pass
        st.error(f"🚨 Access Restricted: {google_email} is not a valid ADNU email.")
        st.stop()

    try:
        sb = create_supabase_client()
        profile = sb.table("users").select("role, full_name").eq("email", google_email).single().execute()
        role = normalize_role(profile.data.get("role"))
        full_name = profile.data.get("full_name") or google_email.split("@")[0]
    except Exception:
        role = "student"
        full_name = google_email.split("@")[0]
        try:
            create_supabase_client().table("users").upsert({
                "email": google_email,
                "full_name": full_name,
                "role": role
            }).execute()
        except Exception:
            pass

    st.session_state["authenticated"] = True
    st.session_state["email"] = google_email
    st.session_state["role"] = role
    st.session_state["full_name"] = full_name
    st.session_state["show_welcome"] = True
    st.session_state.setdefault("session_id", str(uuid.uuid4()))
    st.session_state["view"] = "admin" if role == "admin" else "chat"
    # Page continues rendering naturally with authenticated=True

# Case 2: Not authenticated at all — show login
if not st.session_state.get("authenticated"):
    if google_email:
        try:
            st.logout()
        except Exception:
            pass
    render_login()
    st.stop()

# ─────────────────────────────────────────────────────────────────────────────
# 5. RENDER APP
# ─────────────────────────────────────────────────────────────────────────────
render_sidebar()
if st.session_state["view"] == "admin": render_admin_view()
elif st.session_state["view"] == "indexed_docs": render_indexed_documents_view()
elif st.session_state["view"] == "history": render_history_view()
else: render_chat_view()