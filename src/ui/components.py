import streamlit as st
import base64
import sys
import uuid
import copy
from pathlib import Path

# Path Setup
project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.core.auth import login_user, register_user, supabase as _sb

# ─────────────────────────────────────────────────────────────────────────────
# HELPER: CSS LOADER
# ─────────────────────────────────────────────────────────────────────────────
def load_css(file_name):
    """Reads a CSS file from src/ui/styles/ and injects it."""
    css_path = project_root / "src" / "ui" / "styles" / file_name
    try:
        with open(css_path, "r") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    except FileNotFoundError:
        st.error(f"⚠️ CSS file not found: {css_path}")

# ─────────────────────────────────────────────────────────────────────────────
# LOGO HELPER
# ─────────────────────────────────────────────────────────────────────────────
def get_base64_logo():
    logo_path = Path("assets/logo.png") 
    if logo_path.exists():
        with open(logo_path, "rb") as f:
            return base64.b64encode(f.read()).decode()
    return ""

# ─────────────────────────────────────────────────────────────────────────────
# VIEWS
# ─────────────────────────────────────────────────────────────────────────────
def render_login():
    # Load the CSS from file
    load_css("login.css")
    
    logo_base64 = get_base64_logo()
    if logo_base64:
        st.markdown(f'<div class="logo-container"><img src="data:image/png;base64,{logo_base64}" class="logo-image"><div class="logo-title" style="color: #FF950A;">AXIsstant</div></div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="logo-container"><div class="logo-title" style="color: #FF950A;">AXIsstant</div></div>', unsafe_allow_html=True)

    tab1, tab2 = st.tabs(["Login", "Sign Up"])

    with tab1:
        with st.form("login_form"):
            st.markdown("<h3 style='color: white; text-align: center;'>Sign In</h3>", unsafe_allow_html=True)
            email = st.text_input("Email Address")
            password = st.text_input("Password", type="password")
            if st.form_submit_button("Login", use_container_width=True):
                user = login_user(email, password)
                if user == "UNVERIFIED":
                    st.error("Please verify your email before logging in. Check your inbox for the confirmation link.")
                elif user:
                    st.session_state["authenticated"] = True
                    st.session_state["user_id"] = user["id"]
                    st.session_state["email"] = user["email"]
                    st.session_state["role"] = user["role"]
                    st.session_state["full_name"] = user["full_name"]
                    st.session_state["show_welcome"] = True
                    
                    if "session_id" not in st.session_state:
                        st.session_state["session_id"] = str(uuid.uuid4())
                    
                    st.session_state["view"] = "chat"
                    st.rerun()
                else:
                    st.error("Invalid credentials.")

    with tab2:
        with st.form("signup_form"):
            st.markdown("<h3 style='color: white; text-align: center;'>Create Account</h3>", unsafe_allow_html=True)
            new_email = st.text_input("Email Address", key="new_email")
            new_name = st.text_input("Full Name", key="new_name")
            new_pass = st.text_input("Password", type="password", key="new_pass")
            confirm_pass = st.text_input("Confirm Password", type="password", key="confirm_pass")
            if st.form_submit_button("Register", use_container_width=True):
                if len(new_pass) <= 6:
                    st.error("Password must be more than 6 characters.")
                elif new_pass != confirm_pass:
                    st.error("Passwords do not match.")
                else:
                    success, message = register_user(new_email, new_pass, new_name)
                    if success:
                        st.success(message)
                        st.info(f"✅ Account created! We sent a confirmation link to {new_email}. Please verify before logging in.")
                    else:
                        st.error(f"Error: {message}")

def render_sidebar():
    # Note: We don't load main.css here because 'render_main_styles' 
    # is usually called in main.py. But if you want it loaded with the sidebar:
    load_css("main.css")

    with st.sidebar:
        sidebar_open = st.session_state.get("sidebar_open", True)

        toggle_label = "☰" if sidebar_open else "☰"
        if st.button(toggle_label, key="sidebar_toggle", use_container_width=False):
            st.session_state["sidebar_open"] = not sidebar_open
            st.rerun()

        logo_base64 = get_base64_logo()
        if logo_base64 and sidebar_open:
            st.markdown(f"""
                <div class="sidebar-brand-open" style="text-align: center; padding-bottom: 10px;">
                    <img src="data:image/png;base64,{logo_base64}" width="90" style="filter: drop-shadow(0 0 5px #F3B153);">
                    <div style="margin-top: 5px; font-size: 3rem; font-weight: 800; color: #FF950A; letter-spacing: 0.8px;">AXIsstant</div>
                </div>
            """, unsafe_allow_html=True)
        elif logo_base64 and not sidebar_open:
            st.markdown(f"""
                <div class="sidebar-brand-collapsed" style="text-align: center; padding-bottom: 8px;">
                    <img src="data:image/png;base64,{logo_base64}" width="44" style="filter: drop-shadow(0 0 5px #F3B153);">
                </div>
            """, unsafe_allow_html=True)
        else:
            fallback_text = "AXIsstant" if sidebar_open else "AXI"
            st.markdown(f"<div class='sidebar-brand-fallback' style='text-align: center; padding-bottom: 10px; margin-top: 4px; font-size: 1.55rem; font-weight: 800; color: #111111; letter-spacing: 0.5px;'>{fallback_text}</div>", unsafe_allow_html=True)
        st.markdown("---")
        
        # New Chat button
        new_chat_label = "New Chat" if sidebar_open else "✏️"
        if st.button(new_chat_label, use_container_width=True):
            if st.session_state.get("messages") and len(st.session_state["messages"]) > 0:
                if "chat_history" not in st.session_state:
                    st.session_state["chat_history"] = []
                if st.session_state.get("active_convo_idx") is not None:
                    idx = st.session_state["active_convo_idx"]
                    if 0 <= idx < len(st.session_state["chat_history"]):
                        st.session_state["chat_history"][idx] = copy.deepcopy(st.session_state["messages"])
                else:
                    st.session_state["chat_history"].append(copy.deepcopy(st.session_state["messages"]))
            
            st.session_state["messages"] = []
            st.session_state["active_convo_idx"] = None 
            st.session_state["session_id"] = str(uuid.uuid4())
            st.session_state["view"] = "chat"
            st.rerun()
        
        # History button
        history_label = "History" if sidebar_open else "🕘"
        if st.button(history_label, use_container_width=True):
            if st.session_state.get("messages") and len(st.session_state["messages"]) > 0:
                if "chat_history" not in st.session_state:
                    st.session_state["chat_history"] = []
                if st.session_state.get("active_convo_idx") is not None:
                    idx = st.session_state["active_convo_idx"]
                    if 0 <= idx < len(st.session_state["chat_history"]):
                        st.session_state["chat_history"][idx] = copy.deepcopy(st.session_state["messages"])
                else:
                    st.session_state["chat_history"].append(copy.deepcopy(st.session_state["messages"]))
                    st.session_state["active_convo_idx"] = len(st.session_state["chat_history"]) - 1
            st.session_state["view"] = "history"
            st.rerun()
            
        if st.session_state.get("role") == "admin":
            admin_label = "🛠️ **Admin Dashboard**" if sidebar_open and st.session_state.get("view") == "admin" else ("🛠️ Admin Dashboard" if sidebar_open else "🛠️")
            if st.button(admin_label, use_container_width=True):
                st.session_state["view"] = "admin"
                st.rerun()

        st.markdown("---")
        if sidebar_open:
            user_email = st.session_state.get("email", "Guest")
            role = st.session_state.get("role", "Student").upper()
            st.markdown(f"<div class='user-profile'><strong>{role}</strong><br><small>{user_email}</small></div>", unsafe_allow_html=True)

        logout_label = "Logout" if sidebar_open else "🚪"
        if st.button(logout_label, use_container_width=True, type="primary"):
            try:
                _sb.auth.sign_out()
            except Exception:
                pass
            st.session_state.clear()
            st.rerun()

# ─────────────────────────────────────────────────────────────────────────────
# EXPORT HELPERS (Used in main.py)
# ─────────────────────────────────────────────────────────────────────────────
# We can keep this empty function signature for compatibility with main.py
# or assume main.py calls 'render_sidebar' which now loads the CSS.
def render_main_styles():
    load_css("main.css")