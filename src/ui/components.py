import streamlit as st
import streamlit.components.v1 as st_components
import base64
import sys
import uuid
import copy
from pathlib import Path

# Path Setup
project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.core.auth import login_user, register_user, normalize_role, create_supabase_client


def _is_mobile_client() -> bool:
    try:
        headers = getattr(st, "context", None)
        if not headers:
            return False
        user_agent = (st.context.headers.get("user-agent", "") or "").lower()
        mobile_tokens = ("mobile", "android", "iphone", "ipad", "ipod")
        return any(token in user_agent for token in mobile_tokens)
    except Exception:
        return False

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
        st.markdown(f'<div class="logo-container"><img src="data:image/png;base64,{logo_base64}" class="logo-image"><div class="logo-title" style="color: #FF950A;">AXIstant</div></div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="logo-container"><div class="logo-title" style="color: #FF950A;">AXIstant</div></div>', unsafe_allow_html=True)

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
                    role = normalize_role(user.get("role"))
                    st.session_state["authenticated"] = True
                    st.session_state["user_id"] = user["id"]
                    st.session_state["email"] = user["email"]
                    st.session_state["role"] = role
                    st.session_state["full_name"] = user["full_name"]
                    st.session_state["show_welcome"] = True
                    
                    if "session_id" not in st.session_state:
                        st.session_state["session_id"] = str(uuid.uuid4())
                    
                    st.session_state["view"] = "admin" if role == "admin" else "chat"
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

    # ── NATIVE STREAMLIT GOOGLE LOGIN (Fix: Removed Button Wrapper) ──
    st.markdown("<div style='text-align: center; color: #888; margin: 15px 0 10px 0;'>────── OR ──────</div>", unsafe_allow_html=True)
    
    # Calling st.login directly prevents the double-click render failure
    st.login("google")

def render_sidebar():
    load_css("main.css")
    st_components.html(
        """
        <script>
        (function () {
            let rootWindow = window;
            let rootDocument = document;
            try {
                if (window.parent && window.parent !== window && window.parent.document) {
                    rootWindow = window.parent;
                    rootDocument = window.parent.document;
                }
            } catch (err) {
                rootWindow = window;
                rootDocument = document;
            }

            if (rootWindow.__axiMobileTabSidebarAutoCollapseBound) {
                return;
            }
            rootWindow.__axiMobileTabSidebarAutoCollapseBound = true;

            function isMobileViewport() {
                return rootWindow.matchMedia && rootWindow.matchMedia("(max-width: 768px)").matches;
            }

            function isSidebarOpenOnMobile() {
                const styles = rootWindow.getComputedStyle(rootDocument.documentElement);
                const widthValue = (styles.getPropertyValue("--axi-mobile-sidebar-width") || "").trim();
                return widthValue && widthValue !== "0" && widthValue !== "0px";
            }

            function collapseSidebarIfNeeded() {
                if (!isMobileViewport() || !isSidebarOpenOnMobile()) {
                    return;
                }
                const mobileToggle = rootDocument.querySelector('[class*="st-key-mobile_sidebar_toggle"] button');
                if (mobileToggle) {
                    mobileToggle.click();
                }
            }

            rootDocument.addEventListener(
                "click",
                function (event) {
                    const clickedTab = event.target && event.target.closest('[data-baseweb="tab"], [role="tab"]');
                    const clickedSidebarNav = event.target && event.target.closest('[class*="st-key-nav_"]');
                    if (!clickedTab && !clickedSidebarNav) {
                        return;
                    }
                    setTimeout(collapseSidebarIfNeeded, 120);
                },
                true
            );
        })();
        </script>
        """,
        height=0,
    )

    if st.button("☰", key="mobile_sidebar_toggle", use_container_width=False):
        st.session_state["sidebar_open"] = not st.session_state.get("sidebar_open", True)
        st.rerun()

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
                    <div style="margin-top: 5px; font-size: 3rem; font-weight: 800; color: #FF950A; letter-spacing: 0.8px;">AXIstant</div>
                </div>
            """, unsafe_allow_html=True)
        elif logo_base64 and not sidebar_open:
            st.markdown(f"""
                <div class="sidebar-brand-collapsed" style="text-align: center; padding-bottom: 8px;">
                    <img src="data:image/png;base64,{logo_base64}" width="44" style="filter: drop-shadow(0 0 5px #F3B153);">
                </div>
            """, unsafe_allow_html=True)
        else:
            fallback_text = "AXIstant" if sidebar_open else "AXI"
            st.markdown(f"<div class='sidebar-brand-fallback' style='text-align: center; padding-bottom: 10px; margin-top: 4px; font-size: 1.55rem; font-weight: 800; color: #111111; letter-spacing: 0.5px;'>{fallback_text}</div>", unsafe_allow_html=True)
        st.markdown("---")
        mobile_client = _is_mobile_client()
        
        # User Specific Navigation
        if st.session_state.get("role") != "admin":
            
            # New Chat button
            new_chat_label = "New Chat" if sidebar_open else "✏️"
            if st.button(new_chat_label, key="nav_new_chat", use_container_width=True):
                if mobile_client and st.session_state.get("sidebar_open", True):
                    st.session_state["sidebar_open"] = False
                st.session_state["messages"] = []
                st.session_state["active_convo_idx"] = None 
                st.session_state["session_id"] = str(uuid.uuid4())
                st.session_state["view"] = "chat"
                st.rerun()
            
            # History button
            history_label = "History" if sidebar_open else "🕘"
            if st.button(history_label, key="nav_history", use_container_width=True):
                if mobile_client and st.session_state.get("sidebar_open", True):
                    st.session_state["sidebar_open"] = False
                st.session_state["view"] = "history"
                st.rerun()
            
        # Admin Specific Navigation
        if st.session_state.get("role") == "admin":
            
            # Main Dashboard Button
            admin_label = "**Admin Dashboard**" if sidebar_open and st.session_state.get("view") == "admin" else ("Admin Dashboard" if sidebar_open else "🛠️")
            if st.button(admin_label, key="nav_admin_dashboard", use_container_width=True):
                if mobile_client and st.session_state.get("sidebar_open", True):
                    st.session_state["sidebar_open"] = False
                st.session_state["view"] = "admin"
                st.rerun()

            # Chat Console Button
            admin_chat_label = "**Chat Console**" if sidebar_open and st.session_state.get("view") == "chat" else ("Chat Console" if sidebar_open else "💬")
            if st.button(admin_chat_label, key="nav_chat_console", use_container_width=True):
                if mobile_client and st.session_state.get("sidebar_open", True):
                    st.session_state["sidebar_open"] = False
                st.session_state["messages"] = [] 
                st.session_state["active_convo_idx"] = None 
                st.session_state["session_id"] = str(uuid.uuid4())
                st.session_state["view"] = "chat"
                st.rerun()  

            # Indexed Documents Button
            docs_label = "**Document Management**" if sidebar_open and st.session_state.get("view") == "indexed_docs" else ("Document Management" if sidebar_open else "📚")
            if st.button(docs_label, key="nav_indexed_docs", use_container_width=True):
                if mobile_client and st.session_state.get("sidebar_open", True):
                    st.session_state["sidebar_open"] = False
                st.session_state["view"] = "indexed_docs"
                st.rerun()

        st.markdown("---")
        if sidebar_open:
            user_email = st.session_state.get("email", "Guest")
            role = st.session_state.get("role", "Student").upper()
            st.markdown(f"<div class='user-profile'><strong>{role}</strong><br><small>{user_email}</small></div>", unsafe_allow_html=True)

        logout_label = "Logout" if sidebar_open else "🚪"
        if st.button(logout_label, use_container_width=True, type="primary"):
            try:
                create_supabase_client().auth.sign_out()
            except Exception:
                pass
            
            # ── FIX: Force Streamlit to drop the internal Google Cookie ──
            try:
                st.logout()  
            except Exception:
                pass
                
            st.session_state.clear()
            st.rerun()

# ─────────────────────────────────────────────────────────────────────────────
# EXPORT HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def render_main_styles():
    load_css("main.css")