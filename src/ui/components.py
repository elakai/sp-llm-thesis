import streamlit as st
import base64
import sys
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
# 1. PATH SETUP (For asset loading)
# ─────────────────────────────────────────────────────────────────────────────
project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.config.settings import get_vectorstore
from src.core.ingestion import train_all_pdfs

# ─────────────────────────────────────────────────────────────────────────────
# 2. HELPER FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────
def get_base64_logo():
    """Loads the logo from assets/ folder for the login screen."""
    logo_path = Path("assets/kraken_logo.png") 
    if logo_path.exists():
        with open(logo_path, "rb") as f:
            return base64.b64encode(f.read()).decode()
    return ""

# ─────────────────────────────────────────────────────────────────────────────
# 3. STYLING COMPONENTS
# ─────────────────────────────────────────────────────────────────────────────
def render_login_styles():
    """CSS specifically for the Login Screen (Orange Gradient)."""
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
    """CSS for the Main Chat Interface (Dark Charcoal & Orange)."""
    st.markdown("""
    <style>
        header, footer {visibility: hidden;}
        
        /* Hides the collapse button to lock sidebar */
        [data-testid="sidebar-button"], 
        [data-testid="collapsedControl"] { display: none !important; }

        /* Lock sidebar width */
        [data-testid="stSidebar"] { min-width: 280px !important; max-width: 280px !important; background-color: #F0A52D; }

        /* Main Background */
        .stApp { background-color: #656565; }
        
        /* Metric Styles */
        [data-testid="stMetricValue"] { font-size: 1.6rem !important; color: #000000 !important; font-weight: 700 !important; }
        
        /* Chat Bubble Styles */
        [data-testid="stChatMessageAssistant"] { background-color: #1a1a1a !important; border-left: 5px solid #F0A52D !important; color: #eeeeee !important; }

        /* Profile Footer */
        .user-profile { padding: 12px; background-color: #1a1a1a; border-radius: 12px; color: white; position: fixed; bottom: 20px; width: 260px; }
    </style>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# 4. PAGE COMPONENTS
# ─────────────────────────────────────────────────────────────────────────────
def render_login():
    """Renders the Login Form."""
    render_login_styles()
    logo_base64 = get_base64_logo()
    
    # Logo Display
    if logo_base64:
        st.markdown(f'<div class="logo-container"><img src="data:image/png;base64,{logo_base64}" class="logo-image"><div class="logo-title">AXIsstant</div></div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="logo-container"><div class="logo-title">AXIsstant</div></div>', unsafe_allow_html=True)

    # Login Form
    with st.form("login_form"):
        st.markdown("<h2 style='color: white; text-align: center;'>Sign in to continue</h2>", unsafe_allow_html=True)
        username = st.text_input("ID Number / Username")
        password = st.text_input("Password", type="password")
        
        if st.form_submit_button("Login", use_container_width=True):
            # Hardcoded credential for now
            if password == "csea_student":
                st.session_state["authenticated"] = True
                st.session_state["user_id"] = username
                st.rerun()
            else:
                st.error("Incorrect password")

def render_sidebar_nav(user_id):
    """Renders Sidebar + Embedded Admin Panel."""
    with st.sidebar:
        st.title("🏛️ AXIsstant")
        
        # ─── ADMIN PANEL (Hidden in Expander) ───
        with st.expander("🛠️ Admin Panel"):
            pwd = st.text_input("Admin Password", type="password", key="admin_pwd")
            
            if pwd == "csea2025":
                st.success("Access Granted")
                
                # Stats
                try:
                    vectorstore = get_vectorstore()
                    stats = vectorstore._index.describe_index_stats()
                    st.metric("Knowledge Chunks", f"{stats.get('total_vector_count', 0):,}")
                except:
                    st.metric("Knowledge Chunks", "0")
                
                # Actions
                if st.button("🚀 TRAIN ALL PDFs", use_container_width=True):
                    with st.spinner("Training..."):
                        try:
                            train_all_pdfs()
                            st.success("Training Complete!")
                            st.balloons()
                        except Exception as e:
                            st.error(f"Failed: {e}")
                
                if st.button("⚠️ FULL RESET", use_container_width=True):
                    get_vectorstore().delete(delete_all=True)
                    st.success("Database Wiped.")
                    st.rerun()

        st.markdown("---")
        
        # Navigation Buttons (Placeholder for future features)
        if st.button("💬 Chats", use_container_width=True): 
            st.session_state["view"] = "chat"
        
        # User Profile Footer
        st.markdown(f"<div class='user-profile'><strong>{user_id}</strong><br><small>{str(user_id).lower()}@gbox.adnu.edu.ph</small></div>", unsafe_allow_html=True)