# src/ui/components.py
import streamlit as st
import base64
from pathlib import Path
from src.config.settings import get_vectorstore
from src.core.ingestion import train_all_pdfs

def get_base64_logo():
    logo_path = Path("assets/kraken_logo.png") 
    if logo_path.exists():
        with open(logo_path, "rb") as f:
            return base64.b64encode(f.read()).decode()
    return ""

def render_login_styles():
    """Keeps your high-quality orange card login design"""
    st.markdown("""
    <style>
        header, footer { visibility: hidden; }
        .stApp { background: linear-gradient(to bottom, #1F1F1F, #F09F2D) !important; background-attachment: fixed; }
        [data-testid="stVerticalBlock"] { display: flex; flex-direction: column; align-items: center; justify-content: center; width: 100% !important; margin: 0 auto; }
        .logo-container { display: flex; flex-direction: column; align-items: center; margin-top: 5vh; width: 100%; }
        .logo-image { width: 200px; filter: drop-shadow(0 15px 15px rgba(0,0,0,0.4)); }
        .logo-title { font-size: 3.5rem; color: #fef3c7; font-weight: 700; text-align: center; }
        [data-testid="stForm"] { background-color: #f59e0b !important; border-radius: 20px !important; padding: 40px !important; width: 420px !important; margin: 0 auto !important; }
        div.stButton > button { width: 100% !important; background-color: #1a1a1a !important; color: white !important; font-weight: bold !important; }
        div.stButton > button:hover { background-color: #fef3c7 !important; color: #1a1a1a !important; }
    </style>
    """, unsafe_allow_html=True)

def render_chat_styles():
    """Restores the original dark charcoal chat theme with locked sidebar"""
    st.markdown("""
    <style>
        header, footer {visibility: hidden;}
        
        /* PERMANENT SIDEBAR LOGIC */
        /* Hides the collapse button (top left) and the re-open button if collapsed */
        [data-testid="sidebar-button"], 
        [data-testid="collapsedControl"] {
            display: none !important;
        }

        /* Ensure the sidebar doesn't have a transition effect that shows the button */
        [data-testid="stSidebar"] {
            min-width: 280px !important;
            max-width: 280px !important;
        }

        /* Rest of your existing styles... */
        .stApp { background-color: #656565; }
        [data-testid="stSidebar"] { background-color: #F0A52D; }
        
        [data-testid="stMetricValue"] {
            font-size: 1.6rem !important;
            color: #000000 !important;
            font-weight: 700 !important;
        }
        
        [data-testid="stChatMessageAssistant"] {
            background-color: #1a1a1a !important;
            border-left: 5px solid #F0A52D !important;
            color: #eeeeee !important;
        }

        .user-profile {
            padding: 12px;
            background-color: #1a1a1a;
            border-radius: 12px;
            color: white;
            position: fixed;
            bottom: 20px;
            width: 260px;
        }
    </style>
    """, unsafe_allow_html=True)

def render_login():
    render_login_styles()
    logo_base64 = get_base64_logo()
    st.markdown(f'<div class="logo-container"><img src="data:image/png;base64,{logo_base64}" class="logo-image"><div class="logo-title">AXIsstant</div></div>', unsafe_allow_html=True)
    with st.form("login_form"):
        st.markdown("<h2 style='color: white; text-align: center;'>Sign in to continue</h2>", unsafe_allow_html=True)
        username = st.text_input("ID Number / Username")
        password = st.text_input("Password", type="password")
        if st.form_submit_button("Login", use_container_width=True):
            if password == "csea_student":
                st.session_state["authenticated"] = True
                st.session_state["user_id"] = username
                st.rerun()

def render_sidebar_nav(user_id):
    """Renders original sidebar with authenticated Admin Panel and stats"""
    with st.sidebar:
        st.title("🏛️ AXIsstant")
        
        with st.expander("🛠️ Admin Panel"):
            pwd = st.text_input("Admin Password", type="password", key="admin_pwd")
            if pwd == "csea2025":
                st.success("Admin Access Granted")
                try:
                    vectorstore = get_vectorstore()
                    stats = vectorstore._index.describe_index_stats()
                    st.metric("Knowledge Chunks", f"{stats.get('total_vector_count', 0):,}")
                except:
                    st.metric("Knowledge Chunks", "0")
                
                if st.button("🚀 TRAIN ALL PDFs", use_container_width=True):
                    with st.spinner("Training..."):
                        train_all_pdfs()
                    st.success("Trained!")
                
                if st.button("⚠️ FULL RESET", use_container_width=True):
                    get_vectorstore().delete(delete_all=True)
                    st.rerun()

        st.markdown("---")
        if st.button("💬 Chats", use_container_width=True): st.session_state["view"] = "chat"
        if st.button("🕒 History", use_container_width=True): st.session_state["view"] = "history"

        st.markdown(f"<div class='user-profile'><strong>{user_id}</strong><br><small>{user_id.lower()}@gbox.adnu.edu.ph</small></div>", unsafe_allow_html=True)