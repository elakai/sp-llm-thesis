# src/ui/components.py
import streamlit as st
from src.config.settings import get_db
from src.core.ingestion import train_all_pdfs

def render_header():
    # NO st.set_page_config() here anymore!
    st.markdown("""
    <style>
        .main {background-color: #f8f9fa;}
        .stChatMessage {margin: 20px 0;}
        .header-title {font-size: 3.2rem; color: #0d47a1; text-align: center; font-weight: bold;}
        .header-subtitle {font-size: 1.4rem; color: #f9a825; text-align: center; margin: 10px 0 40px;}
        .stTextInput > div > div > input {border-radius: 12px; padding: 14px;}
    </style>
    """, unsafe_allow_html=True)

    st.markdown("<h1 class='header-title'>CSEA Assistant</h1>", unsafe_allow_html=True)
    st.markdown("<p class='header-subtitle'>College of Science, Engineering and Architecture<br>Ateneo de Naga University</p>", unsafe_allow_html=True)
    
def render_admin_panel():
    with st.sidebar:
        st.markdown("### Admin Panel")
        pwd = st.text_input("Password", type="password", key="admin_pwd")

        if pwd == "csea2025":
            st.success("Access Granted")

            try:
                count = get_db()._collection.count()
                st.info(f"Knowledge chunks: {count}")
            except Exception:
                st.info("Chunks: 0 (database not initialized yet)")

            if st.button("FULL RESET"):
                get_db().delete_collection()
                st.rerun()

            if st.button("TRAIN ALL PDFs"):
                with st.spinner("Training on documents..."):
                    train_all_pdfs()
                st.balloons()
                st.success("Training completed!")