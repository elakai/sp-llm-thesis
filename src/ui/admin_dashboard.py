import streamlit as st
import os
from src.config.settings import get_vectorstore, DOCS_FOLDER
from src.core.ingestion import train_all_pdfs

def save_uploaded_file(uploaded_file):
    """Helper to save uploaded files to disk."""
    if not os.path.exists(DOCS_FOLDER):
        os.makedirs(DOCS_FOLDER)
    
    file_path = os.path.join(DOCS_FOLDER, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return file_path

def render_admin_view():
    """The Full-Page Admin Dashboard Logic."""
    st.markdown("## 🛠️ Admin Dashboard")
    st.info("Manage the Knowledge Base and System Settings.")

    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("### 📤 Upload Documents")
        st.write("Upload PDFs to add them to the knowledge base. Page numbers are automatically extracted.")
        uploaded_files = st.file_uploader("Drop PDFs here", type=["pdf"], accept_multiple_files=True)
        
        if uploaded_files:
            if st.button("Save & Train", type="primary"):
                progress_bar = st.progress(0)
                status = st.empty()
                
                # 1. Save Files
                for i, file in enumerate(uploaded_files):
                    status.text(f"Saving {file.name}...")
                    save_uploaded_file(file)
                    progress_bar.progress((i + 1) / len(uploaded_files) * 0.3)
                
                # 2. Train Model
                status.text("Indexing into Pinecone (this takes a moment)...")
                try:
                    train_all_pdfs()
                    progress_bar.progress(1.0)
                    st.success(f"Successfully processed {len(uploaded_files)} files!")
                    st.balloons()
                except Exception as e:
                    st.error(f"Training Failed: {str(e)}")

    with col2:
        st.markdown("### 📊 System Stats")
        try:
            vectorstore = get_vectorstore()
            stats = vectorstore._index.describe_index_stats()
            count = stats.get('total_vector_count', 0)
            st.metric(label="Total Knowledge Chunks", value=f"{count:,}")
        except:
            st.metric(label="Total Knowledge Chunks", value="0")
        
        st.markdown("### ⚠️ Danger Zone")
        if st.button("Purge Database", help="Deletes ALL knowledge."):
            get_vectorstore().delete(delete_all=True)
            st.warning("Database has been reset.")
            st.rerun()