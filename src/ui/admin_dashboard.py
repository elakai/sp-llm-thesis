import streamlit as st
import pandas as pd
import time

from src.config.settings import get_vectorstore
from src.core.auth import supabase
from src.core.memory_ingestion import ingest_uploaded_files
from src.core.ingestion import get_uploaded_files, delete_document, verify_sync, purge_all_vectors
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


def fetch_evaluation_runs():
    """Fetches RAGAS evaluation run history from the evaluation_runs table."""
    try:
        response = supabase.table("evaluation_runs") \
            .select("*") \
            .order("run_at", desc=True) \
            .limit(50) \
            .execute()
        return pd.DataFrame(response.data) if response.data else pd.DataFrame()
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
        type=["pdf", "docx", "xlsx", "csv", "txt", "md", "png", "jpg", "jpeg", "tiff", "bmp"],
        accept_multiple_files=True,
        help="Supported: PDF, Word, Excel, CSV, Text, Markdown, Images (PNG/JPG/TIFF — text extracted via OCR)",
    )

    # Warn about large files
    if uploaded_files:
        for f in uploaded_files:
            if f.size > 50 * 1024 * 1024:  # 50 MB
                st.warning(f"⚠️ {f.name} is over 50 MB. This may take a while to process.")

    if st.button("Upload and Index", type="primary", disabled=not uploaded_files):
        total = len(uploaded_files)
        progress = st.progress(0, text=f"Starting… 0 / {total} files")
        status_box = st.empty()
        results = []

        for i, single_file in enumerate(uploaded_files):
            pct = int((i / total) * 100)
            progress.progress(pct, text=f"Processing {single_file.name} ({i + 1} / {total})…")
            status_box.info(f"⏳ Indexing: **{single_file.name}**")

            success, message = ingest_uploaded_files([single_file], category)
            results.append((single_file.name, success, message))

        progress.progress(100, text=f"Done! {total} file(s) processed.")
        status_box.empty()

        all_ok = all(s for _, s, _ in results)
        for fname, success, message in results:
            if success:
                st.success(f"✅ **{fname}**: {message}")
            else:
                st.error(f"❌ **{fname}**: {message}")

        if all_ok:
            st.balloons()

    st.divider()

    # ── DOCUMENT LIBRARY ────────────────────────────────────────
    st.subheader("Indexed Documents")

    manifest = get_uploaded_files()

    if not manifest:
        st.info("No documents indexed yet. Upload files above to get started.")
    else:
        # Initialize selection state
        if "selected_docs" not in st.session_state:
            st.session_state["selected_docs"] = set()

        # Select All / Deselect All controls
        sel_col1, sel_col2, sel_col3 = st.columns([2, 2, 6])
        if sel_col1.button("Select All", use_container_width=True):
            st.session_state["selected_docs"] = set(manifest.keys())
            st.rerun()
        if sel_col2.button("Deselect All", use_container_width=True):
            st.session_state["selected_docs"] = set()
            st.rerun()

        # Document list with checkboxes
        for filename, info in manifest.items():
            col_check, col_name, col_chunks, col_date = st.columns([0.5, 4, 1, 2])
            is_selected = col_check.checkbox(
                "sel",
                value=filename in st.session_state["selected_docs"],
                key=f"chk_{filename}",
                label_visibility="collapsed",
            )
            if is_selected:
                st.session_state["selected_docs"].add(filename)
            else:
                st.session_state["selected_docs"].discard(filename)

            col_name.write(f"📄 {filename}")
            col_chunks.write(f"{info.get('chunks', 0)} chunks")
            col_date.write(info.get("uploaded_at", "Unknown"))

        # Delete Selected button
        selected = st.session_state["selected_docs"] & set(manifest.keys())
        if selected:
            st.warning(f"{len(selected)} document(s) selected")

            if "confirm_bulk_delete" not in st.session_state:
                st.session_state["confirm_bulk_delete"] = False

            if not st.session_state["confirm_bulk_delete"]:
                if st.button(
                    f"🗑️ Delete {len(selected)} Selected",
                    type="primary",
                    use_container_width=True,
                ):
                    st.session_state["confirm_bulk_delete"] = True
                    st.rerun()
            else:
                st.error(
                    f"⚠️ Permanently delete these {len(selected)} document(s)?\n\n"
                    + "\n".join(f"- {f}" for f in sorted(selected))
                )
                c1, c2 = st.columns(2)
                if c1.button("Yes, delete all selected", type="primary"):
                    progress = st.progress(0, text="Deleting…")
                    deleted, failed = 0, 0
                    files_to_delete = list(selected)
                    for i, fname in enumerate(files_to_delete):
                        progress.progress(
                            (i + 1) / len(files_to_delete),
                            text=f"Deleting {fname}…",
                        )
                        ok, msg = delete_document(fname)
                        if ok:
                            deleted += 1
                        else:
                            failed += 1
                            st.error(f"Failed: {fname} — {msg}")
                    progress.progress(1.0, text="Done!")
                    st.success(f"Deleted {deleted} document(s).")
                    if failed:
                        st.warning(f"{failed} deletion(s) failed.")
                    st.session_state["selected_docs"] = set()
                    st.session_state["confirm_bulk_delete"] = False
                    st.rerun()
                if c2.button("Cancel"):
                    st.session_state["confirm_bulk_delete"] = False
                    st.rerun()

    st.divider()

    # ── SYNC STATUS ─────────────────────────────────────────────
    st.subheader("Pinecone Sync Status")

    if st.button("Verify Sync"):
        with st.spinner("Checking sync status…"):
            sync_result = verify_sync()
            if sync_result:
                st.write("**In Both (healthy):**", len(sync_result.get("in_both", [])))
                ghost = sync_result.get("manifest_only", [])
                orphans = sync_result.get("pinecone_only", [])
                if ghost:
                    st.write("**Manifest Only (ghost entries):**", ghost)
                if orphans:
                    st.warning(f"**Pinecone Only (orphaned vectors):** {len(orphans)} source(s)")
                    for src in orphans:
                        st.write(f"  - `{src}`")
                if not ghost and not orphans:
                    st.success("Everything is in sync!")
            else:
                st.error("Sync check failed. Check logs.")

    # ── PURGE ALL ───────────────────────────────────────────────
    st.divider()
    st.subheader("⚠️ Danger Zone")

    if "confirm_purge_all" not in st.session_state:
        st.session_state["confirm_purge_all"] = False

    if not st.session_state["confirm_purge_all"]:
        if st.button("🗑️ Purge ALL Vectors from Pinecone", type="secondary"):
            st.session_state["confirm_purge_all"] = True
            st.rerun()
    else:
        st.error(
            "⚠️ This will DELETE EVERY vector from the Pinecone index "
            "and clear the manifest. You will need to re-upload all documents."
        )
        confirm_text = st.text_input('Type **DELETE ALL** to confirm:', key="purge_confirm_text")
        c1, c2 = st.columns(2)
        if c1.button("Permanently Delete All Vectors", type="primary"):
            if confirm_text == "DELETE ALL":
                with st.spinner("Purging…"):
                    ok, msg = purge_all_vectors()
                if ok:
                    st.success(msg)
                else:
                    st.error(msg)
                st.session_state["confirm_purge_all"] = False
                st.rerun()
            else:
                st.warning('Please type "DELETE ALL" exactly to confirm.')
        if c2.button("Cancel purge"):
            st.session_state["confirm_purge_all"] = False
            st.rerun()


def render_admin_view():
    st.markdown("## 🛠️ Admin Control Center")
    st.caption("Manage documents, monitor quality metrics, and inspect database health.")
    st.markdown("---")

    # ── TOP BAR: Quality Metrics ────────────────────────────────
    df_eval = fetch_eval_metrics()
    df_runs = fetch_evaluation_runs()

    m1, m2, m3, m4 = st.columns(4)

    # Show latest evaluation run metrics if available
    if not df_runs.empty:
        latest = df_runs.iloc[0]
        m1.metric("Faithfulness", f"{latest.get('faithfulness', 0):.2f}", help="Score 0–1: Did the LLM hallucinate?")
        m2.metric("Answer Correctness", f"{latest.get('answer_correctness', 0):.2f}", help="Score 0–1: Did it answer correctly?")
    elif not df_eval.empty:
        avg_faith = df_eval["faithfulness"].mean() if "faithfulness" in df_eval else 0.0
        avg_rel = df_eval["answer_relevancy"].mean() if "answer_relevancy" in df_eval else 0.0
        m1.metric("Faithfulness", f"{avg_faith:.2f}", help="Score 0–1: Accuracy")
        m2.metric("Relevancy", f"{avg_rel:.2f}", help="Score 0–1: Helpfulness")
    else:
        m1.metric("Faithfulness", "N/A")
        m2.metric("Relevancy", "N/A")

    if not df_eval.empty:
        pos_feedback = (df_eval["rating"] == "helpful").sum()
        m3.metric("User Likes", pos_feedback)
    else:
        m3.metric("User Likes", 0)
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
        st.subheader("RAGAS Evaluation History")

        if not df_runs.empty:
            # Show the latest run summary
            latest = df_runs.iloc[0]
            st.markdown("#### Latest Run")
            cols = st.columns(5)
            metric_map = [
                ("Faithfulness", "faithfulness"),
                ("Context Precision", "context_precision"),
                ("Context Recall", "context_recall"),
                ("Answer Correctness", "answer_correctness"),
                ("Answer Relevancy", "answer_relevancy"),
            ]
            for col, (label, key) in zip(cols, metric_map):
                val = latest.get(key)
                col.metric(label, f"{val:.2f}" if pd.notna(val) else "—")

            info_cols = st.columns(3)
            info_cols[0].caption(f"**Mode:** {latest.get('mode', 'N/A')}")
            info_cols[1].caption(f"**Sample Size:** {latest.get('sample_size', 'N/A')}")
            info_cols[2].caption(f"**Run At:** {latest.get('run_at', 'N/A')}")

            st.divider()

            # Full history table
            st.markdown("#### All Evaluation Runs")
            display_cols = [c for c in [
                "run_at", "mode", "sample_size",
                "faithfulness", "context_precision", "context_recall",
                "answer_correctness", "answer_relevancy",
            ] if c in df_runs.columns]
            st.dataframe(
                df_runs[display_cols],
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.info(
                "No evaluation runs found. Run the evaluator to see results here:\n\n"
                "`python -m src.core.evaluate_rag --mode dataset`"
            )

        st.divider()
        st.markdown("#### User Feedback Logs")
        if not df_eval.empty:
            eval_table = df_eval.dropna(subset=["faithfulness", "answer_relevancy"])
            if not eval_table.empty:
                st.dataframe(eval_table.tail(15), use_container_width=True)
            else:
                st.info("No per-query RAGAS scores in chat logs yet.")
        else:
            st.info("No chat log data found.")