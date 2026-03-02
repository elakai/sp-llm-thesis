import streamlit as st
import pandas as pd
import time

from src.config.settings import get_vectorstore
from src.core.auth import supabase
from src.core.memory_ingestion import ingest_uploaded_files
from src.core.ingestion import get_uploaded_files, delete_document, verify_sync
from src.config.constants import VALID_CATEGORIES


def fetch_eval_metrics():
    """Fetches Ragas scores and feedback for the analytics bar."""
    try:
        response = supabase.table("chat_logs") \
            .select("rating, faithfulness, answer_relevancy") \
            .execute()
        return pd.DataFrame(response.data)
    except Exception:
        return pd.DataFrame()


def render_document_management():
    """Upload & Index section — files processed in memory, never stored permanently."""
    st.header("Document Management")

    # ── UPLOAD SECTION ──────────────────────────────────────────
    st.subheader("Upload & Index New Documents")
    st.caption(
        "Files are processed and indexed immediately. "
        "They are never stored permanently — only the vector embeddings are saved to Pinecone."
    )

    category = st.selectbox(
        "Document Category",
        options=list(VALID_CATEGORIES.keys()) + ["general"],
        format_func=lambda x: VALID_CATEGORIES.get(x, "General"),
    )

    uploaded_files = st.file_uploader(
        "Drop files here or click to browse",
        type=["pdf", "docx", "xlsx", "csv", "txt"],
        accept_multiple_files=True,
        help="Supported: PDF, Word, Excel, CSV, Text",
    )

    # Warn about large files
    if uploaded_files:
        for f in uploaded_files:
            if f.size > 50 * 1024 * 1024:  # 50 MB
                st.warning(f"⚠️ {f.name} is over 50 MB. This may take a while to process.")

    if st.button("Upload and Index", type="primary", disabled=not uploaded_files):
        with st.spinner("Processing documents… this may take a minute for large PDFs."):
            progress = st.progress(0, text="Starting…")

            success, message = ingest_uploaded_files(uploaded_files, category)

            progress.progress(100, text="Done!")

            if success:
                st.success(message)
                st.balloons()
            else:
                st.error(message)

    st.divider()

    # ── DOCUMENT LIBRARY ────────────────────────────────────────
    st.subheader("Indexed Documents")

    manifest = get_uploaded_files()

    if not manifest:
        st.info("No documents indexed yet. Upload files above to get started.")
    else:
        for filename, info in manifest.items():
            col1, col2, col3, col4 = st.columns([4, 1, 2, 1])
            col1.write(f"📄 {filename}")
            col2.write(f"{info.get('chunks', 0)} chunks")
            col3.write(info.get("uploaded_at", "Unknown"))

            # Delete with confirmation using session state
            delete_key = f"confirm_delete_{filename}"
            if delete_key not in st.session_state:
                st.session_state[delete_key] = False

            if not st.session_state[delete_key]:
                if col4.button("Delete", key=f"del_{filename}"):
                    st.session_state[delete_key] = True
                    st.rerun()
            else:
                st.warning(f"Delete **{filename}**?")
                confirm_col, cancel_col = st.columns(2)
                if confirm_col.button("Yes, delete", key=f"confirm_{filename}"):
                    with st.spinner(f"Deleting {filename}…"):
                        success, msg = delete_document(filename)
                        if success:
                            st.success(msg)
                        else:
                            st.error(msg)
                    st.session_state[delete_key] = False
                    st.rerun()
                if cancel_col.button("Cancel", key=f"cancel_{filename}"):
                    st.session_state[delete_key] = False
                    st.rerun()

    st.divider()

    # ── SYNC STATUS ─────────────────────────────────────────────
    st.subheader("Pinecone Sync Status")

    if st.button("Verify Sync"):
        with st.spinner("Checking sync status…"):
            sync_result = verify_sync()
            if sync_result:
                st.write("**In Both (healthy):**", len(sync_result.get("in_both", [])))
                st.write("**Manifest Only (ghost entries):**", sync_result.get("manifest_only", []))
                st.write("**Pinecone Only (untracked):**", sync_result.get("pinecone_only", []))
            else:
                st.error("Sync check failed. Check logs.")


def render_admin_view():
    st.markdown("## 🛠️ Admin Control Center")
    st.caption("Manage documents, monitor quality metrics, and inspect database health.")
    st.markdown("---")

    # ── TOP BAR: Quality Metrics ────────────────────────────────
    df_eval = fetch_eval_metrics()
    if not df_eval.empty:
        m1, m2, m3, m4 = st.columns(4)

        avg_faith = df_eval["faithfulness"].mean() if "faithfulness" in df_eval else 0.0
        avg_rel = df_eval["answer_relevancy"].mean() if "answer_relevancy" in df_eval else 0.0
        pos_feedback = (df_eval["rating"] == "helpful").sum()

        m1.metric("Faithfulness", f"{avg_faith:.2f}", help="Score 0-1: Accuracy")
        m2.metric("Relevancy", f"{avg_rel:.2f}", help="Score 0-1: Helpfulness")
        m3.metric("User Likes", pos_feedback)
        m4.metric("Status", "Online")
        st.markdown("---")

    # ── TABS ────────────────────────────────────────────────────
    tab_docs, tab_health, tab_eval = st.tabs(
        ["📤 Document Management", "📊 Database Health", "🧪 Evaluation Logs"]
    )

    with tab_docs:
        render_document_management()

    with tab_health:
        st.subheader("Database Health")
        try:
            vectorstore = get_vectorstore()
            try:
                stats = vectorstore._index.describe_index_stats()
                count = stats.get("total_vector_count", 0)
            except Exception:
                count = "Connected"
            st.metric(label="Vector Count", value=f"{count}")
        except Exception:
            st.metric(label="Vector Count", value="Offline")

        st.markdown("#### 🗑️ Purge Records")
        if st.button("Wipe Pinecone Index", help="Irreversible: Deletes ALL AI knowledge."):
            try:
                vectorstore = get_vectorstore()
                vectorstore.delete(delete_all=True)
                st.warning("Database Cleared.")
                time.sleep(1)
                st.rerun()
            except Exception as e:
                st.error(f"Error clearing DB: {e}")

    with tab_eval:
        st.subheader("Detailed Ragas Evaluation Logs")
        if not df_eval.empty:
            eval_table = df_eval.dropna(subset=["faithfulness", "answer_relevancy"])
            st.dataframe(eval_table.tail(15), use_container_width=True)
        else:
            st.info("No evaluation data found in Supabase.")