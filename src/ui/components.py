import streamlit as st
import base64
import sys
from pathlib import Path

# Path Setup
project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.core.auth import login_user, register_user

# Logo

def get_base64_logo():
    logo_path = Path("assets/kraken_logo.png") 
    if logo_path.exists():
        with open(logo_path, "rb") as f:
            return base64.b64encode(f.read()).decode()
    return ""


# STYLES

def render_login_styles():
    st.markdown("""
    <style>
        header, footer { visibility: hidden; }
        .stApp { background: linear-gradient(to bottom, #1F1F1F, #F09F2D) !important; background-attachment: fixed; }
        
        /* Center and scale Logo */
        .logo-container { display: flex; flex-direction: column; align-items: center; margin-top: 5vh; width: 100%; }
        .logo-image { width: 150px; filter: drop-shadow(0 15px 15px rgba(0,0,0,0.4)); transition: transform 0.3s ease; }
        .logo-image:hover { transform: scale(1.05); }
        .logo-title { font-size: 3.0rem; color: #fef3c7; font-weight: 600; text-align: center; margin-bottom: 10px; }
        
        /* Center the Tabs */
        [data-testid="stTabs"] { display: flex; justify-content: center; border: none !important; }
        [data-testid="stTab"] { color: #fef3c7 !important; }
        
        /* Responsive Form */
        [data-testid="stForm"] { 
            background-color: #f59e0b !important; 
            border-radius: 20px !important; 
            padding: 40px !important; 
            max-width: 500px !important; 
            width: 90% !important; 
            margin: 0 auto !important; 
        }
        
        div.stButton > button { width: 100% !important; background-color: #1a1a1a !important; color: white !important; font-weight: bold !important; border-radius: 10px; }
        div.stButton > button:hover { background-color: #fef3c7 !important; color: #1a1a1a !important; border: 1px solid #1a1a1a; }
    </style>
    """, unsafe_allow_html=True)

def render_main_styles():
    st.markdown("""
    <style>
        header, footer {visibility: hidden;}
        [data-testid="sidebar-button"], [data-testid="collapsedControl"] { display: none !important; }
        [data-testid="stSidebar"] { 
                min-width: 280px !important; 
                max-width: 280px !important; 
                background-color: #F3B653;
                margin-right: auto;  }
        
        .stApp { background-color: #0A0A0A; }
        
        .sidebar-top {
            text-align: center;
            width: 100%;
        }
        /* Better Message Bubbles */
        [data-testid="stChatMessage"] { border-radius: 15px; padding: 15px; margin-bottom: 15px; max-width: 80%; }
        [data-testid="stChatMessageAssistant"] { 
            background-color: #1a1a1a !important; 
            border-left: 5px solid #F0A52D !important; 
            color: #eeeeee !important;
            margin-right: auto; 
        }
        
        [data-testid="stChatMessageUser"] { 
            background-color: #3d3d3d !important; 
            border-right: 5px solid #fef3c7 !important; 
            color: white !important;
            margin-left: auto;
        }
        
        /* Increase text size in chat messages */
        [data-testid="stChatMessage"] p,
        [data-testid="stChatMessage"] li,
        [data-testid="stChatMessage"] span {
            font-size: 1.1rem !important;
            line-height: 1.6 !important;
        }

        /* Floating Chat Input Box */
        [data-testid="stChatInput"] { padding: 1.5rem; background-color: transparent !important; }
        [data-testid="stChatInput"] > div { 
            border-radius: 30px !important; 
            border: 1px solid #F0A52D !important; 
            background-color: #1a1a1a !important; 
        }
        
        /* Increase text size in chat input */
        [data-testid="stChatInput"] textarea {
            font-size: 1.1rem !important;
        }

        /* Sidebar Profile */
        .user-profile { padding: 12px; background-color: #1a1a1a; border-radius: 12px; color: white; position: fixed; bottom: 20px; width: 240px; }
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
    with st.sidebar:
        logo_base64 = get_base64_logo()
        if logo_base64:
            st.markdown(f"""
                <div style="text-align: center; padding-bottom: 10px;">
                    <img src="data:image/png;base64,{logo_base64}" width="90" style="filter: drop-shadow(0 0 5px #F3B153);">
                    <h1 style='color: #0A0A0A; margin-top: 10px; font-size: 2.5rem;'>AXIsstant</h1>
                </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"<h1 style='text-align: center; color: #0A0A0A; font-size: 2.5rem;'>AXIsstant</h1>", unsafe_allow_html=True)
        st.markdown("---")
        
        # New Chat button - saves current conversation and starts fresh
        if st.button("➕ New Chat", use_container_width=True):
            # Save/update current conversation to history if it has messages
            if st.session_state.get("messages") and len(st.session_state["messages"]) > 0:
                if "chat_history" not in st.session_state:
                    st.session_state["chat_history"] = []
                
                # If continuing an existing conversation, update it instead of appending
                if st.session_state.get("active_convo_idx") is not None:
                    idx = st.session_state["active_convo_idx"]
                    if 0 <= idx < len(st.session_state["chat_history"]):
                        st.session_state["chat_history"][idx] = st.session_state["messages"].copy()
                else:
                    # New conversation, append to history
                    st.session_state["chat_history"].append(st.session_state["messages"].copy())
            
            st.session_state["messages"] = []
            st.session_state["active_convo_idx"] = None  # Reset active conversation
            st.session_state["view"] = "chat"
            st.rerun()
        
        # History button - shows chat history (saves current conversation first)
        history_label = "📜 **History**" if st.session_state.get("view") == "history" else "📜 History"
        if st.button(history_label, use_container_width=True):
            # Save current conversation before viewing history
            if st.session_state.get("messages") and len(st.session_state["messages"]) > 0:
                if "chat_history" not in st.session_state:
                    st.session_state["chat_history"] = []
                
                if st.session_state.get("active_convo_idx") is not None:
                    idx = st.session_state["active_convo_idx"]
                    if 0 <= idx < len(st.session_state["chat_history"]):
                        st.session_state["chat_history"][idx] = st.session_state["messages"].copy()
                else:
                    # New unsaved conversation
                    st.session_state["chat_history"].append(st.session_state["messages"].copy())
                    st.session_state["active_convo_idx"] = len(st.session_state["chat_history"]) - 1
            
            st.session_state["view"] = "history"
            st.rerun()
            
        if st.session_state.get("role") == "admin":
            admin_label = "🛠️ **Admin Dashboard**" if st.session_state.get("view") == "admin" else "🛠️ Admin Dashboard"
            if st.button(admin_label, use_container_width=True):
                st.session_state["view"] = "admin"
                st.rerun()

        st.markdown("---")
        if st.button("🚪 Logout", use_container_width=True):
            st.session_state.clear()
            st.rerun()
            
        user_email = st.session_state.get("user_id", "Guest")
        role = st.session_state.get("role", "Student").upper()
        st.markdown(f"<div class='user-profile'><strong>{role}</strong><br><small>{user_email}</small></div>", unsafe_allow_html=True)