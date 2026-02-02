import streamlit as st
import base64
import sys
from pathlib import Path

# Path Setup
project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.core.auth import login_user, register_user

# ─────────────────────────────────────────────────────────────────────────────
# HELPER FUNCTIONS (Logos)
# ─────────────────────────────────────────────────────────────────────────────
def get_base64_logo():
    logo_path = Path("assets/kraken_logo.png") 
    if logo_path.exists():
        with open(logo_path, "rb") as f:
            return base64.b64encode(f.read()).decode()
    return ""

# ─────────────────────────────────────────────────────────────────────────────
# STYLES
# ─────────────────────────────────────────────────────────────────────────────
def render_login_styles():
    st.markdown("""
    <style>
        header, footer { visibility: hidden; }
        .stApp { background: linear-gradient(to bottom, #1F1F1F, #F09F2D) !important; background-attachment: fixed; }
        [data-testid="stVerticalBlock"] { width: 100% !important; margin: 0 auto; }
        .logo-container { display: flex; flex-direction: column; align-items: center; margin-top: 5vh; width: 100%; }
        .logo-image { width: 200px; filter: drop-shadow(0 15px 15px rgba(0,0,0,0.4)); }
        .logo-title { font-size: 3.5rem; color: #fef3c7; font-weight: 700; text-align: center; }
        [data-testid="stForm"] { background-color: #f59e0b !important; border-radius: 20px !important; padding: 40px !important; width: 420px !important; margin: 0 auto !important; }
        div.stButton > button { width: 100% !important; background-color: #1a1a1a !important; color: white !important; font-weight: bold !important; }
        div.stButton > button:hover { background-color: #fef3c7 !important; color: #1a1a1a !important; }
    </style>
    """, unsafe_allow_html=True)

def render_main_styles():
    st.markdown("""
    <style>
        header, footer {visibility: hidden;}
        [data-testid="sidebar-button"], [data-testid="collapsedControl"] { display: none !important; }
        [data-testid="stSidebar"] { min-width: 280px !important; max-width: 280px !important; background-color: #F0A52D; }
        .stApp { background-color: #656565; }
        .user-profile { padding: 12px; background-color: #1a1a1a; border-radius: 12px; color: white; position: fixed; bottom: 20px; width: 260px; }
        
        /* Chat Specifics */
        [data-testid="stChatMessageAssistant"] { background-color: #1a1a1a !important; border-left: 5px solid #F0A52D !important; color: #eeeeee !important; }
        
        /* Admin Dashboard Cards */
        div[data-testid="stMetric"] { background-color: #1a1a1a; padding: 20px; border-radius: 10px; border-left: 5px solid #F0A52D; }
        label[data-testid="stLabel"] { color: #F0A52D !important; font-weight: bold; }
    </style>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# VIEWS
# ─────────────────────────────────────────────────────────────────────────────
def render_login():
    render_login_styles()
    logo_base64 = get_base64_logo()
    if logo_base64:
        st.markdown(f'<div class="logo-container"><img src="data:image/png;base64,{logo_base64}" class="logo-image"><div class="logo-title">AXIsstant</div></div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="logo-container"><div class="logo-title">AXIsstant</div></div>', unsafe_allow_html=True)

    tab1, tab2 = st.tabs(["Login", "Sign Up"])

    with tab1:
        with st.form("login_form"):
            st.markdown("<h3 style='color: white; text-align: center;'>Sign In</h3>", unsafe_allow_html=True)
            email = st.text_input("Email Address")
            password = st.text_input("Password", type="password")
            if st.form_submit_button("Login", use_container_width=True):
                user = login_user(email, password)
                if user:
                    st.session_state["authenticated"] = True
                    st.session_state["user_id"] = user["email"]
                    st.session_state["role"] = user["role"]
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
            if st.form_submit_button("Register", use_container_width=True):
                success, message = register_user(new_email, new_pass, new_name)
                if success: st.success("Account created! Go to Login.")
                else: st.error(f"Error: {message}")

def render_sidebar():
    """Sidebar Navigation Only"""
    with st.sidebar:
        st.title("🏛️ AXIsstant")
        st.markdown("---")
        
        # Navigation Buttons
        if st.button("💬 Chat Interface", use_container_width=True):
            st.session_state["view"] = "chat"
            st.rerun()
            
        # Admin Button (Conditional)
        if st.session_state.get("role") == "admin":
            if st.button("🛠️ Admin Dashboard", use_container_width=True):
                st.session_state["view"] = "admin"
                st.rerun()

        st.markdown("---")
        if st.button("🚪 Logout", use_container_width=True):
            st.session_state.clear()
            st.rerun()
            
        # Footer
        user_email = st.session_state.get("user_id", "Guest")
        role = st.session_state.get("role", "Student").upper()
        st.markdown(f"<div class='user-profile'><strong>{role}</strong><br><small>{user_email}</small></div>", unsafe_allow_html=True)