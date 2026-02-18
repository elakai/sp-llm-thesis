import streamlit as st
import pandas as pd
import plotly.express as px
import os
from src.config.settings import get_vectorstore
from src.core.ingestion import ingest_all_files as train_all_pdfs
from src.core.auth import supabase

def fetch_eval_metrics():
    """Fetches Ragas scores and feedback for the analytics bar."""
    try:
        response = supabase.table("chat_logs") \
            .select("rating, faithfulness, answer_relevancy") \
            .execute()
        return pd.DataFrame(response.data)
    except Exception:
        return pd.DataFrame()

def render_admin_view():
    st.markdown("## 🛠️ Admin Control Center")
    st.caption("Management tools for knowledge ingestion and AI performance evaluation.")
    st.markdown("---")

    # 1. TOP BAR: Ragas Quality Metrics
    df_eval = fetch_eval_metrics()
    if not df_eval.empty:
        m1, m2, m3, m4 = st.columns(4)
        
        avg_faith = df_eval['faithfulness'].mean() if 'faithfulness' in df_eval else 0.0
        avg_rel = df_eval['answer_relevancy'].mean() if 'answer_relevancy' in df_eval else 0.0
        pos_feedback = (df_eval['rating'] == 'helpful').sum()
        
        m1.metric("Faithfulness", f"{avg_faith:.2f}", help="Score 0-1: Accuracy")
        m2.metric("Relevancy", f"{avg_rel:.2f}", help="Score 0-1: Helpfulness")
        m3.metric("User Likes", pos_feedback)
        m4.metric("Status", "Online")
        st.markdown("---")

    # 2. MIDDLE SECTION: Knowledge & Stats
    left_col, right_col = st.columns([2, 1])

    with left_col:
        st.markdown("### 📤 Document Ingestion")
        st.write("Add new PDFs to the Handbook Knowledge Base.")
        uploaded_files = st.file_uploader("Upload Handbook PDFs", type=["pdf"], accept_multiple_files=True)
        
        if uploaded_files:
            if st.button("🚀 Process & Index Documents", type="primary"):
                with st.status("Ingesting documents...", expanded=True) as status:
                    
                    # 🛠️ STEP 1: SAVE FILES TO DISK (The Missing Link)
                    st.write("📂 Saving uploaded files to 'data/' folder...")
                    
                    # Ensure this matches the folder your ingestion.py looks inside!
                    TARGET_DIR = "documents" 
                    if not os.path.exists(TARGET_DIR):
                        os.makedirs(TARGET_DIR)

                    for uploaded_file in uploaded_files:
                        file_path = os.path.join(TARGET_DIR, uploaded_file.name)
                        with open(file_path, "wb") as f:
                            f.write(uploaded_file.getbuffer())
                    
                    # 🛠️ STEP 2: RUN INGESTION
                    st.write("🧠 Running embedding and Pinecone indexing...")
                    try:
                        train_all_pdfs()
                        status.update(label="Ingestion Complete!", state="complete", expanded=False)
                        st.success(f"Successfully indexed {len(uploaded_files)} documents.")
                        st.balloons()
                    except Exception as e:
                        st.error(f"Ingestion Failed: {e}")

    with right_col:
        st.markdown("### 📊 Database Health")
        try:
            vectorstore = get_vectorstore()
            # Some versions of Pinecone/LangChain use slightly different stats calls
            # Use a generic try/except to prevent UI crashes
            try:
                stats = vectorstore._index.describe_index_stats()
                count = stats.get('total_vector_count', 0)
            except:
                count = "Connected"
            
            st.metric(label="Vector Count", value=f"{count}")
        except:
            st.metric(label="Vector Count", value="Offline")
        
        st.markdown("#### 🗑️ Purge Records")
        if st.button("Wipe Pinecone Index", help="Irreversible: Deletes ALL AI knowledge."):
            try:
                vectorstore = get_vectorstore()
                vectorstore.delete(delete_all=True)
                st.warning("Database Cleared.")
                st.rerun()
            except Exception as e:
                st.error(f"Error clearing DB: {e}")

    # 3. BOTTOM SECTION: Ragas Log Table
    st.markdown("---")
    st.markdown("### 🧪 Detailed Ragas Evaluation Logs")
    if not df_eval.empty:
        eval_table = df_eval.dropna(subset=['faithfulness', 'answer_relevancy'])
        st.dataframe(eval_table.tail(15), use_container_width=True)
    else:
        st.info("No evaluation data found. Run `run_eval.py` to populate these scores.")