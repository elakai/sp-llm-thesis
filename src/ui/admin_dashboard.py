import streamlit as st
import pandas as pd
import time
import os
import re
from collections import Counter

from src.config.settings import get_vectorstore
from src.core.auth import supabase
from src.core.memory_ingestion import ingest_uploaded_files
from src.core.ingestion import get_uploaded_files, delete_document, verify_sync, purge_all_vectors
from src.config.constants import VALID_CATEGORIES


def fetch_eval_metrics():
    """Fetches feedback, usage data, latency metrics, and session mapping for analytics."""
    try:
        response = supabase.table("chat_logs") \
            .select("session_id, rating, query, user_email, created_at, retrieval_latency, generation_latency, total_latency") \
            .execute()
        return pd.DataFrame(response.data)
    except Exception as e:
        st.error(f"🚨 Database Error fetching chat logs: {e}")
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


def inject_admin_styles():
    """Reads custom CSS from the styles folder to beautify tabs, tables, and metric cards."""
    css_path = os.path.join(os.path.dirname(__file__), "styles", "admin.css")
    try:
        with open(css_path, "r") as f:
            css = f.read()
        st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)
    except FileNotFoundError:
        st.warning("⚠️ CSS file not found at src/ui/styles/admin.css. Styles may not load correctly.")


# ── CONSOLIDATED DOCUMENT MANAGEMENT VIEW ──
def render_indexed_documents_view():
    inject_admin_styles()

    st.markdown("""
        <div class="admin-page-header">
            <div class="admin-page-title-row">
                <div class="admin-page-icon">📚</div>
                <h1 class="admin-page-title">Document Management</h1>
            </div>
            <p class="admin-page-subtitle">Manage the Knowledge Base for the CSEA Academic AI. Upload New Files, Manage your Indexed Library, and Perform Advanced Database Operations.</p>
        </div>
    """, unsafe_allow_html=True)

    tab_upload, tab_library, tab_advanced = st.tabs([
        "Upload & Index", 
        "Indexed Library", 
        "Advanced Tools"
    ])

    with tab_upload:
        with st.container(border=True):
            st.markdown("""
                <div style='padding: 20px; background-color: rgba(128,128,128,0.03); border-radius: 10px; border: 1px solid rgba(128,128,128,0.1); margin-bottom: 20px;'>
                    <h3 style='margin: 0; padding: 0; font-size: 1.5rem; font-weight: 700; color: inherit;'>📤 Upload & Index New Documents</h3>
                    <div style='font-size: 0.9rem; opacity: 0.7; margin-top: 5px;'>Files are Processed and Indexed Immediately. They are never Stored Permanently.</div>
                </div>
            """, unsafe_allow_html=True)

            category = st.selectbox(
                "Document Category",
                options=list(VALID_CATEGORIES.keys()) + ["general"],
                format_func=lambda x: VALID_CATEGORIES.get(x, "General"),
            )

            uploaded_files = st.file_uploader(
                "Drop Files Here or Click to Browse",
                type=["pdf", "docx", "xlsx", "csv", "txt", "md", "png", "jpg", "jpeg", "tiff", "bmp"],
                accept_multiple_files=True,
                help="Supported: PDF, Word, Excel, CSV, Text, Markdown, Images (PNG/JPG/TIFF — text extracted via OCR)",
            )

            if uploaded_files:
                for f in uploaded_files:
                    if f.size > 50 * 1024 * 1024:  # 50 MB
                        st.warning(f"Note: **{f.name}** is over 50 MB. This may take a while to process.")

            if st.button("Upload and Index", type="primary", disabled=not uploaded_files, width="stretch"):
                total = len(uploaded_files)
                progress = st.progress(0, text=f"Starting… 0 / {total} files")
                status_box = st.empty()
                results = []

                for i, single_file in enumerate(uploaded_files):
                    pct = int((i / total) * 100)
                    progress.progress(pct, text=f"Processing {single_file.name} ({i + 1} / {total})…")
                    status_box.info(f"Indexing: **{single_file.name}**")

                    success, message = ingest_uploaded_files([single_file], category)
                    results.append((single_file.name, success, message))

                progress.progress(100, text=f"Done! {total} file(s) processed.")
                status_box.empty()

                all_ok = all(s for _, s, _ in results)
                for fname, success, message in results:
                    if success:
                        st.success(f"Success: **{fname}**: {message}")
                    else:
                        st.error(f"Error: **{fname}**: {message}")

    with tab_library:
        with st.container(border=True):
            try:
                vectorstore = get_vectorstore()
                stats = vectorstore._index.describe_index_stats()
                count = stats.get("total_vector_count", 0)
                count_display = f"{count:,}"
            except Exception:
                count_display = "Offline"
                
            st.markdown(f"""
                <div style='display: flex; justify-content: space-between; align-items: center; padding: 20px; background-color: rgba(128,128,128,0.03); border-radius: 10px; border: 1px solid rgba(128,128,128,0.1); margin-bottom: 20px;'>
                    <div>
                        <h3 style='margin: 0; padding: 0; font-size: 1.5rem; font-weight: 700; color: inherit;'>📂 Indexed Documents Library</h3>
                        <div style='font-size: 0.9rem; opacity: 0.7; margin-top: 5px;'>Manage Your Currently Indexed Files and Chunks</div>
                    </div>
                    <div style='text-align: right; background-color: #FF950A; padding: 10px 20px; border-radius: 8px; box-shadow: 0 4px 6px rgba(255, 149, 10, 0.3);'>
                        <div style='font-size: 0.75rem; font-weight: 700; text-transform: uppercase; letter-spacing: 1px; color: rgba(255, 255, 255, 0.9);'>Total Vectors</div>
                        <div style='font-size: 1.8rem; font-weight: 800; color: white; line-height: 1.2;'>{count_display}</div>
                    </div>
                </div>
            """, unsafe_allow_html=True)
            
            manifest = get_uploaded_files()

            if not manifest:
                st.info("No documents indexed yet. Upload files from the 'Upload & Index' tab to get started.")
            else:
                if "selected_docs" not in st.session_state:
                    st.session_state["selected_docs"] = set()
                if "table_key" not in st.session_state:
                    st.session_state["table_key"] = 0

                sorted_manifest = sorted(manifest.items(), key=lambda x: x[0])

                df_data = []
                for filename, info in sorted_manifest:
                    df_data.append({
                        "Select": filename in st.session_state["selected_docs"],
                        "Document Name": filename,
                        "Chunks": info.get("chunks", 0),
                        "Date Indexed": info.get("uploaded_at", "Unknown")[:16]
                    })
                df = pd.DataFrame(df_data)

                dynamic_height = (len(df) * 35) + 43

                edited_df = st.data_editor(
                    df,
                    column_config={
                        "Select": st.column_config.CheckboxColumn("Select", width="small"),
                        "Document Name": st.column_config.TextColumn("Document Name", disabled=True, width="large"),
                        "Chunks": st.column_config.NumberColumn("Chunks", disabled=True, width="small"),
                        "Date Indexed": st.column_config.TextColumn("Date Indexed", disabled=True, width="medium"),
                    },
                    hide_index=True,
                    width="stretch",
                    height=dynamic_height,
                    key=f"doc_table_editor_{st.session_state['table_key']}"
                )

                selected_filenames = edited_df[edited_df["Select"] == True]["Document Name"].tolist()
                st.session_state["selected_docs"] = set(selected_filenames)

                st.markdown("<br>", unsafe_allow_html=True)
                
                btn_left, spacer, btn_right = st.columns([2, 5, 2])
                
                with btn_left:
                    if st.button("Select All", width="stretch"):
                        st.session_state["selected_docs"] = set(manifest.keys())
                        st.session_state["table_key"] += 1  
                        st.rerun()
                        
                with btn_right:
                    if st.button("Deselect All", width="stretch"):
                        st.session_state["selected_docs"] = set()
                        st.session_state["table_key"] += 1 
                        st.rerun()
                
                selected = st.session_state["selected_docs"]
                if selected:
                    st.warning(f"**{len(selected)} Document(s) Selected.**")

                    if "confirm_bulk_delete" not in st.session_state:
                        st.session_state["confirm_bulk_delete"] = False

                    c_insp, c_del = st.columns([1, 1])
                    
                    with c_insp:
                        if len(selected) == 1:
                            if st.button("Inspect LLM Chunks", width="stretch"):
                                st.session_state["inspect_doc"] = list(selected)[0]
                                st.session_state["confirm_bulk_delete"] = False
                                st.rerun()
                        else:
                            st.button("Inspect LLM Chunks", width="stretch", disabled=True, help="Please Select Exactly ONE Document to Inspect")
                            
                    with c_del:
                        if not st.session_state["confirm_bulk_delete"]:
                            if st.button("Delete Selected Documents", type="primary", width="stretch"):
                                st.session_state["confirm_bulk_delete"] = True
                                st.rerun()
                        else:
                            st.error("⚠️ Permanently delete these documents? This action cannot be undone.")
                            col_y, col_n = st.columns([1, 1])
                            if col_y.button("Yes, delete", type="primary", width="stretch"):
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
                                st.session_state["table_key"] += 1
                                if "inspect_doc" in st.session_state:
                                    del st.session_state["inspect_doc"]
                                st.rerun()
                                
                            if col_n.button("Cancel", width="stretch"):
                                st.session_state["confirm_bulk_delete"] = False
                                st.rerun()

                    inspect_target = st.session_state.get("inspect_doc")
                    if inspect_target and inspect_target in selected:
                        st.markdown("<br>", unsafe_allow_html=True)
                        st.markdown(f"""
                            <div style='padding: 12px 16px; background-color: rgba(255, 149, 10, 0.08); border-radius: 8px; border-left: 5px solid #FF950A; margin-bottom: 15px;'>
                                <h4 style='margin: 0; padding: 0; font-size: 1.15rem; font-weight: 700; color: inherit;'>🔎 Inside the LLM's Brain: <span style='font-family: monospace;'>{inspect_target}</span></h4>
                            </div>
                        """, unsafe_allow_html=True)
                        
                        with st.spinner("Fetching Chunks Directly from Vector Database..."):
                            try:
                                vectorstore = get_vectorstore()
                                idx = vectorstore._index
                                
                                stats = idx.describe_index_stats()
                                dim = stats.get('dimension', 384)
                                
                                res = idx.query(
                                    vector=[0.0] * dim,
                                    filter={"source": inspect_target},
                                    top_k=100, 
                                    include_metadata=True
                                )
                                
                                matches = res.get("matches", [])
                                if not matches:
                                    st.warning("No chunks found in Pinecone. This file might be orphaned, or still indexing.")
                                else:
                                    st.success(f"Successfully Retrieved **{len(matches)}** Chunk(s).")
                                    
                                    with st.container(border=True, height=800):
                                        for i, match in enumerate(matches):
                                            chunk_text = match.get("metadata", {}).get("text", "[No text available]")
                                            st.markdown(f"**Chunk {i+1}**")
                                            st.markdown(f"""
                                                <div style='background-color: #FFFFFF; color: #111827; padding: 15px; border-radius: 8px; border: 1px solid rgba(128,128,128,0.2); font-family: monospace; font-size: 0.9rem; margin-bottom: 15px; line-height: 1.5;'>
                                                    {chunk_text}
                                                </div>
                                            """, unsafe_allow_html=True)
                                            
                            except Exception as e:
                                st.error(f"Failed to retrieve chunks: {e}")

    with tab_advanced:
        
        # ── KNOWLEDGE MAP IN ADVANCED TOOLS (FULL WIDTH) ──
        with st.container(border=True):
            st.markdown("""
                <div style='padding: 20px; background-color: rgba(128,128,128,0.03); border-radius: 10px; border: 1px solid rgba(128,128,128,0.1); margin-bottom: 20px;'>
                    <h3 style='margin: 0; padding: 0; font-size: 1.5rem; font-weight: 700; color: inherit;'>🌌 Vector Knowledge Map</h3>
                    <div style='font-size: 0.9rem; opacity: 0.85; margin-top: 10px; line-height: 1.6;'>
                        <strong>How it Works:</strong> The AI converts every document chunk into a 384-dimensional mathematical vector based on its underlying meaning. This tool compresses those dimensions down to 2D or 3D space. Points that group closely together represent text chunks that share similar semantic context.<br><br>
                        <strong>Dimensionality Reduction Algorithms:</strong><br>
                        <span style='color: #FF950A;'>•</span> <strong>PCA (Global View):</strong> Fast and linear. It preserves the overall "big picture" structure of the dataset, making it ideal for visualizing broad relationships across all documents.<br>
                        <span style='color: #FF950A;'>•</span> <strong>t-SNE (Local Clusters):</strong> Advanced and non-linear. It strictly focuses on keeping highly similar points close together, making it perfect for discovering tight, distinct groupings of sub-topics.
                    </div>
                </div>
            """, unsafe_allow_html=True)

            try:
                import numpy as np
                from sklearn.decomposition import PCA
                from sklearn.manifold import TSNE
                import plotly.express as px
                HAS_DEPS = True
            except ImportError:
                HAS_DEPS = False
                st.warning("🚨 Missing Dependencies! To render the Vector Map, open your terminal and run: `pip install scikit-learn numpy plotly`")

            if HAS_DEPS:
                c1, c2 = st.columns(2)
                algo = c1.selectbox("Algorithm", ["PCA", "t-SNE"], help="PCA is faster and shows overall variance. t-SNE is slower but creates tighter local clusters.")
                dims = c2.selectbox("Dimensions", [2, 3], index=1)
                
                if st.button("Generate Knowledge Map", type="primary", width="stretch"):
                    with st.spinner(f"Extracting Vectors from Pinecone and Running {algo}..."):
                        try:
                            vectorstore = get_vectorstore()
                            idx = vectorstore._index
                            
                            query_vec = vectorstore.embeddings.embed_query("University policies handbook computer engineering department")
                            
                            res = idx.query(vector=query_vec, top_k=800, include_values=True, include_metadata=True)
                            matches = res.get('matches', [])
                            
                            if len(matches) < 10:
                                st.error("Not enough document vectors found in the database to generate a cluster map.")
                            else:
                                vectors = [m['values'] for m in matches]
                                texts = [m['metadata'].get('text', '')[:120] + '...' for m in matches]
                                sources = [m['metadata'].get('source', 'Unknown Document') for m in matches]
                                
                                X = np.array(vectors)
                                
                                if algo == "PCA":
                                    reducer = PCA(n_components=dims)
                                else:
                                    perp = min(30, max(2, len(vectors) - 1))
                                    reducer = TSNE(n_components=dims, perplexity=perp, random_state=42)
                                    
                                X_reduced = reducer.fit_transform(X)
                                
                                df_plot = pd.DataFrame(X_reduced, columns=[f"Dim{i+1}" for i in range(dims)])
                                df_plot['Document Source'] = sources
                                df_plot['Chunk Text'] = texts
                                
                                custom_colors = ['#FF950A', '#3B82F6', '#10B981', '#8B5CF6', '#EF4444', '#F43F5E', '#14B8A6', '#F59E0B']

                                if dims == 2:
                                    fig = px.scatter(
                                        df_plot, x="Dim1", y="Dim2", color="Document Source", 
                                        color_discrete_sequence=custom_colors,
                                        hover_data={"Chunk Text": True, "Dim1": False, "Dim2": False}
                                    )
                                    fig.update_traces(marker=dict(size=10, line=dict(width=1.5, color='rgba(255,255,255,0.9)'), opacity=0.85))
                                else:
                                    fig = px.scatter_3d(
                                        df_plot, x="Dim1", y="Dim2", z="Dim3", color="Document Source",
                                        color_discrete_sequence=custom_colors,
                                        hover_data={"Chunk Text": True, "Dim1": False, "Dim2": False, "Dim3": False}
                                    )
                                    fig.update_traces(marker=dict(size=5, line=dict(width=0.5, color='rgba(255,255,255,0.7)'), opacity=0.9))

                                fig.update_layout(
                                    paper_bgcolor='rgba(0,0,0,0)',
                                    plot_bgcolor='rgba(0,0,0,0)',
                                    margin=dict(l=0, r=0, t=0, b=0),
                                    legend=dict(
                                        orientation="v", 
                                        yanchor="top", 
                                        y=1, 
                                        xanchor="left", 
                                        x=1.0,
                                        title_font_family="sans-serif",
                                        font=dict(size=12),
                                        bgcolor='rgba(128,128,128,0.05)', 
                                        bordercolor='rgba(128,128,128,0.2)',
                                        borderwidth=1
                                    ),
                                    hoverlabel=dict(
                                        bgcolor="white",
                                        font_size=13,
                                        font_family="sans-serif",
                                        font_color="black"
                                    ),
                                    height=650
                                )
                                
                                if dims == 2:
                                    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,0.15)', zeroline=False, title_text="")
                                    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,0.15)', zeroline=False, title_text="")
                                else:
                                    fig.update_layout(
                                        scene=dict(
                                            xaxis=dict(showbackground=False, gridcolor='rgba(128,128,128,0.15)', zerolinecolor='rgba(128,128,128,0.15)', title_text=""),
                                            yaxis=dict(showbackground=False, gridcolor='rgba(128,128,128,0.15)', zerolinecolor='rgba(128,128,128,0.15)', title_text=""),
                                            zaxis=dict(showbackground=False, gridcolor='rgba(128,128,128,0.15)', zerolinecolor='rgba(128,128,128,0.15)', title_text=""),
                                        )
                                    )
                                
                                st.plotly_chart(fig, width="stretch")
                                st.success(f"Successfully Mapped {len(matches)} Vector Chunks into {dims}D Space!")

                        except Exception as e:
                            st.error(f"Error generating map: {e}")

        with st.container(border=True):
            st.markdown("""
                <div style='padding: 20px; background-color: rgba(128,128,128,0.03); border-radius: 10px; border: 1px solid rgba(128,128,128,0.1); margin-bottom: 20px;'>
                    <h3 style='margin: 0; padding: 0; font-size: 1.5rem; font-weight: 700; color: inherit;'>⚙️ Pinecone Synchronization</h3>
                    <div style='font-size: 0.9rem; opacity: 0.7; margin-top: 5px;'>Cross-Reference the Supabase Manifest against Actual Vectors in Pinecone to Detect Orphaned Data.</div>
                </div>
            """, unsafe_allow_html=True)

            if st.button("Verify Database Sync", width="stretch"):
                with st.spinner("Checking sync status…"):
                    sync_result = verify_sync()
                    if sync_result:
                        st.write(f"**Healthy (In Both):** `{len(sync_result.get('in_both', []))}` Documents")
                        ghost = sync_result.get("manifest_only", [])
                        orphans = sync_result.get("pinecone_only", [])
                        if ghost:
                            st.warning(f"**Ghost Entries (Manifest Only):** {len(ghost)}")
                        if orphans:
                            st.error(f"**Orphaned Vectors (Pinecone Only):** {len(orphans)} source(s)")
                            for src in orphans:
                                st.write(f"  - `{src}`")
                        if not ghost and not orphans:
                            st.success("Database is Perfectly Synchronized.")
                    else:
                        st.error("Sync check failed. Check logs.")

        with st.expander("🚨 RESET", expanded=False):
            st.error("The actions below are irreversible. They will permanently destroy the LLM's knowledge base.")
            
            if "confirm_purge_all" not in st.session_state:
                st.session_state["confirm_purge_all"] = False

            if not st.session_state["confirm_purge_all"]:
                if st.button("Purge ALL Vectors from Database", type="primary", width="stretch"):
                    st.session_state["confirm_purge_all"] = True
                    st.rerun()
            else:
                st.warning("This will DELETE EVERY vector from the Pinecone index and clear the manifest.")
                confirm_text = st.text_input('Type **DELETE ALL** to confirm:', key="purge_confirm_text")
                c1, c2 = st.columns(2)
                if c1.button("Permanently Execute Purge", type="primary", width="stretch"):
                    if confirm_text == "DELETE ALL":
                        with st.spinner("Purging…"):
                            ok, msg = purge_all_vectors()
                        if ok:
                            st.success(msg)
                        else:
                            st.error(msg)
                        st.session_state["confirm_purge_all"] = False
                        st.session_state["selected_docs"] = set()
                        
                        if "table_key" in st.session_state:
                            st.session_state["table_key"] += 1
                            
                        st.rerun()
                    else:
                        st.error('Please type "DELETE ALL" exactly to confirm.')
                if c2.button("Cancel", width="stretch"):
                    st.session_state["confirm_purge_all"] = False
                    st.rerun()


# ── ADMIN DASHBOARD WITH ANALYTICS ──
def render_admin_view():
    inject_admin_styles()

    st.markdown("""
        <div class="admin-page-header">
            <div class="admin-page-title-row">
                <div class="admin-page-icon">🛠️</div>
                <h1 class="admin-page-title">Admin Dashboard</h1>
            </div>
            <p class="admin-page-subtitle">Monitor System Health, Analyze User Engagement, and Inspect RAG Evaluation Metrics.</p>
        </div>
    """, unsafe_allow_html=True)

    df_eval = fetch_eval_metrics()
    df_runs = fetch_evaluation_runs()

    with st.container(border=True):
        st.markdown("""
            <div style='padding: 20px; background-color: rgba(128,128,128,0.03); border-radius: 10px; border: 1px solid rgba(128,128,128,0.1); margin-bottom: 10px;'>
                <h3 style='margin: 0; padding: 0; font-size: 1.4rem; font-weight: 700; color: inherit;'>📊 Overall Performance Metrics</h3>
            </div>
        """, unsafe_allow_html=True)

        faith_val = "N/A"
        faith_desc = "Score 0–1: Accuracy Across Logs"
        rel_val = "N/A"
        rel_desc = "Score 0–1: Helpfulness Across Logs"
        
        if not df_runs.empty:
            latest = df_runs.iloc[0]
            faith_val = f"{latest.get('faithfulness', 0):.2f}"
            faith_desc = "Score 0–1: Did the LLM Hallucinate?"
            rel_val = f"{latest.get('answer_correctness', 0):.2f}"
            rel_desc = "Score 0–1: Did it Answer Correctly?"
        elif not df_eval.empty:
            avg_faith = df_eval["faithfulness"].mean() if "faithfulness" in df_eval else 0.0
            avg_rel = df_eval["answer_relevancy"].mean() if "answer_relevancy" in df_eval else 0.0
            faith_val = f"{avg_faith:.2f}" if "faithfulness" in df_eval else "N/A"
            rel_val = f"{avg_rel:.2f}" if "answer_relevancy" in df_eval else "N/A"

        likes_val = "0"
        if not df_eval.empty:
            likes_val = str((df_eval["rating"] == "helpful").sum())

        # ── Safely check Database/Pinecone Status ──
        try:
            get_vectorstore()
            sys_status = "Online"
            sys_color = "#10B981"  # Green
            sys_desc = "Database Connected"
        except Exception:
            sys_status = "Offline"
            sys_color = "#EF4444"  # Red
            sys_desc = "Connection Failed"

        # ── HERO CARDS DESIGN FOR OVERALL METRICS ──
        st.markdown(f"""
            <div style="display: flex; gap: 16px; flex-wrap: wrap; padding: 4px 12px 12px 12px;">
                <div class="hero-card" style="border-bottom: 5px solid #8B5CF6;">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;">
                        <span style="font-size: 0.85rem; font-weight: 700; text-transform: uppercase; letter-spacing: 1px; color: #8B5CF6;">Faithfulness</span>
                        <span style="font-size: 1.2rem;">🛡️</span>
                    </div>
                    <div style="font-size: 2.8rem; font-weight: 800; line-height: 1.1; color: #8B5CF6;">{faith_val}</div>
                    <div class="hero-card-desc">{faith_desc}</div>
                </div>
                <div class="hero-card" style="border-bottom: 5px solid #3B82F6;">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;">
                        <span style="font-size: 0.85rem; font-weight: 700; text-transform: uppercase; letter-spacing: 1px; color: #3B82F6;">Correctness</span>
                        <span style="font-size: 1.2rem;">🎯</span>
                    </div>
                    <div style="font-size: 2.8rem; font-weight: 800; line-height: 1.1; color: #3B82F6;">{rel_val}</div>
                    <div class="hero-card-desc">{rel_desc}</div>
                </div>
                <div class="hero-card" style="border-bottom: 5px solid #F59E0B;">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;">
                        <span style="font-size: 0.85rem; font-weight: 700; text-transform: uppercase; letter-spacing: 1px; color: #F59E0B;">Helpful Ratings</span>
                        <span style="font-size: 1.2rem;">⭐</span>
                    </div>
                    <div style="font-size: 2.8rem; font-weight: 800; line-height: 1.1; color: #F59E0B;">{likes_val}</div>
                    <div class="hero-card-desc">Total Positive Feedback</div>
                </div>
                <div class="hero-card" style="border-bottom: 5px solid {sys_color};">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;">
                        <span style="font-size: 0.85rem; font-weight: 700; text-transform: uppercase; letter-spacing: 1px; color: {sys_color};">System Status</span>
                        <span style="font-size: 1.2rem;">⚡</span>
                    </div>
                    <div style="font-size: 2.8rem; font-weight: 800; line-height: 1.1; color: {sys_color};">{sys_status}</div>
                    <div class="hero-card-desc">{sys_desc}</div>
                </div>
            </div>
        """, unsafe_allow_html=True)


    tab_performance, tab_analytics, tab_failures, tab_eval = st.tabs(
        ["System Performance", "Usage Analytics", "Failure Analysis", "Evaluation Logs"]
    )

    # ── SYSTEM PERFORMANCE & LATENCY MONITOR ──
    with tab_performance:
        with st.container(border=True):
            st.markdown("""
                <div style='padding: 20px; background-color: rgba(128,128,128,0.03); border-radius: 10px; border: 1px solid rgba(128,128,128,0.1); margin-bottom: 20px;'>
                    <h3 style='margin: 0; padding: 0; font-size: 1.5rem; font-weight: 700; color: inherit;'>⚡ System Performance Monitor</h3>
                    <div style='font-size: 0.9rem; opacity: 0.7; margin-top: 5px;'>Track AI Response Speeds and Review Live  User Feedback Logs.</div>
                </div>
            """, unsafe_allow_html=True)

            if not df_eval.empty and 'total_latency' in df_eval.columns:
                df_perf = df_eval.dropna(subset=['total_latency']).copy()
                
                if not df_perf.empty:
                    # Calculate strictly the mathematical averages
                    total_mean = df_perf['total_latency'].mean()
                    retrieval_mean = df_perf['retrieval_latency'].mean() if 'retrieval_latency' in df_perf.columns else 0.0
                    gen_mean = df_perf['generation_latency'].mean() if 'generation_latency' in df_perf.columns else 0.0
                    
                    # Output standard metric cards without inner container
                    st.markdown(f"""
                        <div class="metric-card-container" style="margin-top: 0px; margin-bottom: 20px;">
                            <div class="metric-card" style="border-top-color: #10B981; padding: 15px 20px;">
                                <div class="metric-card-header">
                                    <div class="metric-title">Total Latency</div>
                                </div>
                                <div class="metric-value">{total_mean:.2f}s</div>
                                <div class="metric-desc">⚡ End-to-End Speed</div>
                            </div>
                            <div class="metric-card" style="border-top-color: #3B82F6; padding: 15px 20px;">
                                <div class="metric-card-header">
                                    <div class="metric-title">Retrieval Latency</div>
                                </div>
                                <div class="metric-value">{retrieval_mean:.2f}s</div>
                                <div class="metric-desc">🔍 Database Search Time</div>
                            </div>
                            <div class="metric-card" style="border-top-color: #FF950A; padding: 15px 20px;">
                                <div class="metric-card-header">
                                    <div class="metric-title">Generation Latency</div>
                                </div>
                                <div class="metric-value">{gen_mean:.2f}s</div>
                                <div class="metric-desc">🧠 LLM Response Time</div>
                            </div>
                        </div>
                    """, unsafe_allow_html=True)
                else:
                    st.info("There are chat logs in the database, but none of them contain latency metrics yet. Submit a new query in the main app to generate data.")
            else:
                st.info("Latency tracking columns are currently missing or empty in the chat logs database.")

            with st.container(border=True):
                st.markdown("""
                    <div style='padding: 12px 16px; background-color: rgba(255, 149, 10, 0.08); border-radius: 8px; border-left: 5px solid #FF950A; margin-bottom: 15px;'>
                        <h4 style='margin: 0; padding: 0; font-size: 1.15rem; font-weight: 700; color: inherit;'>💬 Live User Feedback Logs</h4>
                    </div>
                """, unsafe_allow_html=True)
                
                if not df_eval.empty:
                    # Removed the raw latency columns to keep the table clean and strictly about Feedback
                    display_cols = [c for c in ["created_at", "query", "rating"] if c in df_eval.columns]
                    display_table = df_eval[display_cols].sort_values(by="created_at", ascending=False)
                    
                    rename_dict = {
                        "created_at": "Time",
                        "query": "Query",
                        "rating": "Rating"
                    }
                    display_table = display_table.rename(columns=rename_dict)
                    
                    # ── INCREASED TO 500 ROWS ──
                    st.dataframe(display_table.head(500), width="stretch", hide_index=True)
                else:
                    st.info("No user feedback logs found.")

    with tab_analytics:
        with st.container(border=True):
            st.markdown("""
                <div style='padding: 20px; background-color: rgba(128,128,128,0.03); border-radius: 10px; border: 1px solid rgba(128,128,128,0.1); margin-bottom: 20px;'>
                    <h3 style='margin: 0; padding: 0; font-size: 1.5rem; font-weight: 700; color: inherit;'>📈 Usage & Engagement Analytics</h3>
                    <div style='font-size: 0.9rem; opacity: 0.7; margin-top: 5px;'>Monitor Daily Active Users, Session Durations, and Popular Discussion Topics.</div>
                </div>
            """, unsafe_allow_html=True)
            
            if df_eval.empty:
                st.info("No chat log data available for analytics yet.")
            else:
                try:
                    import plotly.express as px
                    import plotly.graph_objects as go
                    HAS_PLOTLY = True
                except ImportError:
                    HAS_PLOTLY = False
                    st.warning("💡 Tip: Run `pip install plotly` in your terminal to unlock premium interactive charts.")

                # Setup stop words for reuse in multiple charts
                stop_words = {"what", "how", "the", "for", "and", "can", "you", "tell", "about", "are", "with", "that", "this", "from", "does", "have", "who", "why", "where"}

                # ── NEW: SESSION ENGAGEMENT METRICS (STRICT 30-MIN FILTER) ──
                df_eval['created_at'] = pd.to_datetime(df_eval['created_at'])
                
                if 'session_id' in df_eval.columns and not df_eval['session_id'].isna().all():
                    # Group by session_id to find start time, end time, and total queries per session
                    sessions = df_eval.groupby('session_id').agg(
                        start_time=('created_at', 'min'),
                        end_time=('created_at', 'max'),
                        turns=('query', 'count')
                    )
                    # Calculate duration in seconds
                    sessions['duration_sec'] = (sessions['end_time'] - sessions['start_time']).dt.total_seconds()
                    
                    total_sessions = len(sessions)
                    avg_turns = sessions['turns'].mean()
                    
                    # DATA CLEANING FOR DURATION: 
                    # Disregard 0-second sessions (single questions) and extreme ghost sessions (> 30 mins / 1800s)
                    valid_durations = sessions[(sessions['duration_sec'] > 0) & (sessions['duration_sec'] < 1800)]['duration_sec']
                    
                    if not valid_durations.empty:
                        avg_duration = valid_durations.mean() # True mathematical average, but strictly filtered
                    else:
                        avg_duration = 0
                    
                    # Explicit minutes and seconds formatting
                    if pd.isna(avg_duration) or avg_duration == 0:
                        duration_str = "< 1s"
                    elif avg_duration < 60:
                        duration_str = f"{int(avg_duration)}s"
                    else:
                        minutes = int(avg_duration // 60)
                        seconds = int(avg_duration % 60)
                        if seconds == 0:
                            duration_str = f"{minutes}m"
                        else:
                            duration_str = f"{minutes}m {seconds}s"

                    st.markdown(f"""
                        <div class="metric-card-container" style="margin-top: 0px; margin-bottom: 20px;">
                            <div class="metric-card" style="border-top-color: #8B5CF6; padding: 15px 20px;">
                                <div class="metric-card-header">
                                    <div class="metric-title">Total Sessions</div>
                                </div>
                                <div class="metric-value">{total_sessions}</div>
                                <div class="metric-desc">Unique Conversations Started</div>
                            </div>
                            <div class="metric-card" style="border-top-color: #3B82F6; padding: 15px 20px;">
                                <div class="metric-card-header">
                                    <div class="metric-title">Avg. Session Duration</div>
                                </div>
                                <div class="metric-value">{duration_str}</div>
                                <div class="metric-desc">Time Spent per Conversation</div>
                            </div>
                            <div class="metric-card" style="border-top-color: #FF950A; padding: 15px 20px;">
                                <div class="metric-card-header">
                                    <div class="metric-title">Avg. Queries per Session</div>
                                </div>
                                <div class="metric-value">{avg_turns:.1f}</div>
                                <div class="metric-desc">Questions Asked per Session</div>
                            </div>
                        </div>
                    """, unsafe_allow_html=True)


                # ── CHART 1: DAU ──
                with st.container(border=True):
                    st.markdown("""
                        <div style='padding: 12px 16px; background-color: rgba(255, 149, 10, 0.08); border-radius: 8px; border-left: 5px solid #FF950A; margin-bottom: 15px;'>
                            <h4 style='margin: 0; padding: 0; font-size: 1.15rem; font-weight: 700; color: inherit;'>📅 Daily Active Users</h4>
                        </div>
                    """, unsafe_allow_html=True)
                    
                    df_eval['date'] = df_eval['created_at'].dt.date
                    dau = df_eval.groupby('date')['user_email'].nunique().reset_index()
                    
                    if HAS_PLOTLY:
                        fig_dau = px.area(
                            dau, x='date', y='user_email', 
                            labels={'user_email': 'Number of Users', 'date': ''}
                        )
                        fig_dau.update_layout(
                            paper_bgcolor='rgba(0,0,0,0)',
                            plot_bgcolor='rgba(0,0,0,0)',
                            xaxis=dict(
                                showgrid=False, 
                                tickfont=dict(size=12, weight='bold'),
                            ),
                            yaxis=dict(
                                showgrid=True, 
                                gridcolor='rgba(128, 128, 128, 0.2)', 
                                tickformat='d',      
                                dtick=1,             
                                tickfont=dict(size=12, weight='bold'), 
                                title_font=dict(size=14, weight='bold'), 
                                title="Number of Users"
                            ),
                            margin=dict(l=10, r=10, t=10, b=10),
                            height=250
                        )
                        fig_dau.update_traces(
                            line_color="#F8F5F1", 
                            fillcolor='rgba(255, 149, 10, 0.3)', 
                            mode='lines+markers', 
                            marker=dict(size=8, color='#FD9001')
                        )
                        st.plotly_chart(fig_dau, width="stretch")
                    else:
                        dau_fallback = dau.rename(columns={'user_email': 'Number of Users'}).set_index('date')
                        st.line_chart(dau_fallback, y="Number of Users")

                st.markdown("<br>", unsafe_allow_html=True)
                
                col1, col2 = st.columns(2)

                # ── CHART 2: USER FEEDBACK ──
                with col1:
                    with st.container(border=True, height=420):
                        st.markdown("""
                            <div style='padding: 12px 16px; background-color: rgba(255, 149, 10, 0.08); border-radius: 8px; border-left: 5px solid #FF950A; margin-bottom: 15px;'>
                                <h4 style='margin: 0; padding: 0; font-size: 1.15rem; font-weight: 700; color: inherit;'>⭐ Overall User Feedback</h4>
                            </div>
                        """, unsafe_allow_html=True)
                        
                        ratings = df_eval['rating'].dropna()
                        
                        if not ratings.empty:
                            helpful_count = (ratings == "helpful").sum()
                            not_helpful_count = (ratings == "not_helpful").sum()
                            total_rated = helpful_count + not_helpful_count
                            
                            if total_rated > 0:
                                helpfulness_score = (helpful_count / total_rated) * 100
                                
                                if HAS_PLOTLY:
                                    fig_pie = go.Figure(data=[go.Pie(
                                        labels=['<b>Helpful</b>', '<b>Needs Improvement</b>'],
                                        values=[helpful_count, not_helpful_count],
                                        hole=0.75,
                                        marker_colors=["#FD9001", '#CBD5E1'], 
                                        textinfo='none',
                                        hoverinfo='label+percent+value'
                                    )])
                                    fig_pie.update_layout(
                                        paper_bgcolor='rgba(0,0,0,0)',
                                        plot_bgcolor='rgba(0,0,0,0)',
                                        showlegend=True,
                                        legend=dict(
                                            orientation="h", 
                                            yanchor="bottom", 
                                            y=-0.2, 
                                            xanchor="center", 
                                            x=0.5,
                                            font=dict(size=12) 
                                        ),
                                        margin=dict(l=10, r=10, t=10, b=10),
                                        annotations=[dict(text=f"{helpfulness_score:.0f}%", x=0.5, y=0.5, font_size=36, font_weight="bold", showarrow=False)],
                                        height=280
                                    )
                                    st.plotly_chart(fig_pie, width="stretch")
                                else:
                                    st.metric("Overall Helpfulness", f"{helpfulness_score:.1f}%")
                                    feedback_df = pd.DataFrame({
                                        "Feedback": ["Helpful", "Needs Improvement"],
                                        "Count": [helpful_count, not_helpful_count]
                                    }).set_index("Feedback")
                                    st.bar_chart(feedback_df, color=["#FF950A"])
                            else:
                                st.info("No explicit ratings provided yet.")
                        else:
                            st.info("No feedback data available.")

                # ── CHART 3: TRENDING TOPICS ──
                with col2:
                    with st.container(border=True, height=420):
                        st.markdown("""
                            <div style='padding: 12px 16px; background-color: rgba(255, 149, 10, 0.08); border-radius: 8px; border-left: 5px solid #FF950A; margin-bottom: 15px;'>
                                <h4 style='margin: 0; padding: 0; font-size: 1.15rem; font-weight: 700; color: inherit;'>💬 Trending Topics</h4>
                            </div>
                        """, unsafe_allow_html=True)
                        
                        queries = df_eval['query'].dropna().astype(str).tolist()
                        text = " ".join(queries).lower()
                        
                        if text.strip():
                            words = re.findall(r'\b[a-z]{3,}\b', text)
                            filtered_words = [w for w in words if w not in stop_words]
                            
                            if filtered_words:
                                top_words = Counter(filtered_words).most_common(30)
                                
                                html_pills = "<div style='display: flex; flex-wrap: wrap; align-content: flex-start; gap: 10px; margin-top: 10px; padding-bottom: 10px; max-height: 280px; overflow-y: auto; padding-right: 5px;'>"
                                for word, count in top_words:
                                    html_pills += f"<div style='background-color: #FF950A; color: white; padding: 6px 14px; border-radius: 20px; font-size: 0.9rem; font-weight: 600; box-shadow: 0 2px 4px rgba(255,149,10,0.3); display: flex; align-items: center; gap: 6px;'>{word.title()} <span style='background-color: white; color: #FF950A; border-radius: 50%; padding: 2px 7px; font-size: 0.75rem; font-weight: 800;'>{count}</span></div>"
                                html_pills += "</div>"
                                
                                st.markdown(html_pills, unsafe_allow_html=True)
                            else:
                                st.info("Not enough distinct keywords found.")
                        else:
                            st.info("Not enough query data to extract topics.")

    with tab_failures:
        with st.container(border=True):
            st.markdown("""
                <div style='padding: 20px; background-color: rgba(128, 128, 128, 0.03); border-radius: 10px; border: 1px solid rgba(128, 128, 128, 0.1); margin-bottom: 20px;'>
                    <h3 style='margin: 0; padding: 0; font-size: 1.5rem; font-weight: 700; color: inherit;'>⚠️ Failure Analysis & Knowledge Gaps</h3>
                    <div style='font-size: 0.9rem; opacity: 0.7; margin-top: 5px;'>Analyze Student Queries Receiving Negative Feedback, Identifying Missing or Inaccurate Topics.</div>
                </div>
            """, unsafe_allow_html=True)
            
            if df_eval.empty:
                st.info("No chat log data available for analytics yet.")
            else:
                bad_feedback = df_eval[df_eval['rating'] == 'not_helpful'].dropna(subset=['query'])
                
                if not bad_feedback.empty:
                    with st.container(border=True):
                        st.markdown("""
                            <div style='padding: 12px 16px; background-color: rgba(239, 68, 68, 0.08); border-radius: 8px; border-left: 5px solid #EF4444; margin-bottom: 15px;'>
                                <h4 style='margin: 0; padding: 0; font-size: 1.15rem; font-weight: 700; color: inherit;'>🚨 Problematic Keywords</h4>
                            </div>
                        """, unsafe_allow_html=True)
                        
                        bad_queries = bad_feedback['query'].astype(str).tolist()
                        bad_text = " ".join(bad_queries).lower()
                        bad_words = re.findall(r'\b[a-z]{3,}\b', bad_text)
                        bad_filtered = [w for w in bad_words if w not in stop_words]
                        
                        if bad_filtered:
                            top_bad = Counter(bad_filtered).most_common(15) 
                            bad_df = pd.DataFrame(top_bad, columns=['Keyword', 'Thumbs Down Count'])
                            
                            try:
                                import plotly.express as px
                                fig_bad = px.bar(
                                    bad_df, 
                                    x='Thumbs Down Count', 
                                    y='Keyword', 
                                    orientation='h'
                                )
                                fig_bad.update_layout(
                                    paper_bgcolor='rgba(0,0,0,0)', 
                                    plot_bgcolor='rgba(0,0,0,0)',
                                    yaxis={'categoryorder':'total ascending', 'title': '', 'tickfont': dict(size=12, weight='bold')},
                                    xaxis={'title': 'Thumbs Down Count', 'tickformat': 'd', 'dtick': 1, 'gridcolor': 'rgba(128, 128, 128, 0.2)', 'tickfont': dict(size=12, weight='bold')},
                                    margin=dict(l=10, r=10, t=10, b=10),
                                    height=350
                                )
                                fig_bad.update_traces(marker_color='#EF4444')
                                st.plotly_chart(fig_bad, width="stretch")
                            except ImportError:
                                st.bar_chart(bad_df.set_index('Keyword'))
                        else:
                            st.info("No distinct keywords extracted.")
                    
                    st.markdown("<br>", unsafe_allow_html=True)

                    with st.container(border=True):
                        st.markdown("""
                            <div style='padding: 12px 16px; background-color: rgba(239, 68, 68, 0.08); border-radius: 8px; border-left: 5px solid #EF4444; margin-bottom: 15px;'>
                                <h4 style='margin: 0; padding: 0; font-size: 1.15rem; font-weight: 700; color: inherit;'>📋 Failed Queries</h4>
                            </div>
                        """, unsafe_allow_html=True)
                        
                        failed_table = bad_feedback[['created_at', 'query', 'user_email']].sort_values('created_at', ascending=False)
                        failed_table = failed_table.rename(columns={
                            "created_at": "Time", 
                            "query": "Failed Query", 
                            "user_email": "User"
                        })
                        st.dataframe(failed_table, hide_index=True, width="stretch")
                else:
                    st.success("🎉 Great job! No negative feedback found in the logs yet.")

    with tab_eval:
        with st.container(border=True):
            st.markdown("""
                <div style='padding: 20px; background-color: rgba(128, 128, 128, 0.03); border-radius: 10px; border: 1px solid rgba(128, 128, 128, 0.1); margin-bottom: 20px;'>
                    <h3 style='margin: 0; padding: 0; font-size: 1.5rem; font-weight: 700; color: inherit;'>🧪 RAGAS Evaluation Hub</h3>
                    <div style='font-size: 0.9rem; opacity: 0.7; margin-top: 5px;'>Upload Golden Datasets, Run Pipeline Evaluations, and Monitor Accuracy Trends.</div>
                </div>
            """, unsafe_allow_html=True)
            
            if not df_runs.empty:
                latest = df_runs.iloc[0]
                f_val = f"{latest.get('faithfulness', 0):.2f}"
                cp_val = f"{latest.get('context_precision', 0):.2f}"
                cr_val = f"{latest.get('context_recall', 0):.2f}"
                ac_val = f"{latest.get('answer_correctness', 0):.2f}"

                # Output standard metric cards without inner container
                st.markdown(f"""
                    <div class="metric-card-container" style="margin-top: 0px; margin-bottom: 20px;">
                        <div class="metric-card" style="border-top-color: #8B5CF6; padding: 15px 20px;">
                            <div class="metric-card-header"><div class="metric-title">Faithfulness</div></div>
                            <div class="metric-value">{f_val}</div>
                            <div class="metric-desc">Hallucination Check</div>
                        </div>
                        <div class="metric-card" style="border-top-color: #0EA5E9; padding: 15px 20px;">
                            <div class="metric-card-header"><div class="metric-title">Context Precision</div></div>
                            <div class="metric-value">{cp_val}</div>
                            <div class="metric-desc">Signal-to-Noise Ratio</div>
                        </div>
                        <div class="metric-card" style="border-top-color: #F59E0B; padding: 15px 20px;">
                            <div class="metric-card-header"><div class="metric-title">Context Recall</div></div>
                            <div class="metric-value">{cr_val}</div>
                            <div class="metric-desc">Retrieval Completeness</div>
                        </div>
                        <div class="metric-card" style="border-top-color: #3B82F6; padding: 15px 20px;">
                            <div class="metric-card-header"><div class="metric-title">Answer Correctness</div></div>
                            <div class="metric-value">{ac_val}</div>
                            <div class="metric-desc">Ground Truth Match</div>
                        </div>
                    </div>
                """, unsafe_allow_html=True)
            else:
                st.info("No evaluation runs found. Upload a Golden Dataset below to run your first evaluation!")

            # ── ONE-CLICK EVALUATOR ──
            with st.container(border=True):
                st.markdown("""
                    <div style='padding: 12px 16px; background-color: rgba(255, 149, 10, 0.08); border-radius: 8px; border-left: 5px solid #FF950A; margin-bottom: 15px;'>
                        <h4 style='margin: 0; padding: 0; font-size: 1.15rem; font-weight: 700; color: inherit;'>📤 Upload Dataset & Evaluate</h4>
                    </div>
                """, unsafe_allow_html=True)

                st.caption("Your CSV must Contain Two Columns: `question` and `ground_truth`.")
                eval_file = st.file_uploader("Upload Test Dataset", type=["csv"], label_visibility="collapsed")

                if st.button("Run RAGAS Evaluation", type="primary", width="stretch", disabled=not eval_file):
                    progress_bar = st.progress(5, text="Initializing Evaluation Dataset...")

                    try:
                        # Use the actual filename uploaded by the admin
                        uploaded_filename = eval_file.name
                        with open(uploaded_filename, "wb") as f:
                            f.write(eval_file.getbuffer())

                        progress_bar.progress(25, text="Dataset Saved. Booting up AI Judges & RAG Pipeline...")

                        from src.core.evaluate_rag import evaluate_from_dataset

                        with st.spinner("Generating AI Answers and Grading Metrics... (This will take a few minutes)"):
                            evaluate_from_dataset(uploaded_filename) # Pass the dynamic filename

                        # Clean up the file locally after evaluation finishes
                        if os.path.exists(uploaded_filename):
                            os.remove(uploaded_filename)

                        progress_bar.progress(100, text="✅ Evaluation Complete & Logged!")
                        st.success("Results logged Successfully to Supabase! Refreshing...")
                        time.sleep(2)
                        st.rerun()
                    except Exception as e:
                        progress_bar.progress(100, text="🚨 Evaluation Failed")
                        st.error(f"Error During Evaluation: {e}")

            st.markdown("<br>", unsafe_allow_html=True)

            # ── EVALUATION SCORE TRENDS ──
            if not df_runs.empty:
                with st.container(border=True):
                    st.markdown("""
                        <div style='padding: 12px 16px; background-color: rgba(255, 149, 10, 0.08); border-radius: 8px; border-left: 5px solid #FF950A; margin-bottom: 15px;'>
                            <h4 style='margin: 0; padding: 0; font-size: 1.15rem; font-weight: 700; color: inherit;'>📊 Evaluation Score Trends</h4>
                        </div>
                    """, unsafe_allow_html=True)

                    try:
                        import plotly.graph_objects as go

                        df_plot = df_runs.sort_values(by="run_at", ascending=True)
                        df_plot['run_at'] = pd.to_datetime(df_plot['run_at']).dt.strftime('%b %d, %H:%M')

                        fig_eval = go.Figure()
                        
                        fig_eval.add_trace(go.Scatter(
                            x=df_plot['run_at'], y=df_plot['faithfulness'],
                            mode='lines+markers', name='Faithfulness',
                            line=dict(color='#8B5CF6', width=3), marker=dict(size=8),
                            fill='tozeroy', fillcolor='rgba(139, 92, 246, 0.1)'
                        ))
                        fig_eval.add_trace(go.Scatter(
                            x=df_plot['run_at'], y=df_plot['answer_correctness'],
                            mode='lines+markers', name='Answer Correctness',
                            line=dict(color='#3B82F6', width=3), marker=dict(size=8),
                            fill='tozeroy', fillcolor='rgba(59, 130, 246, 0.1)'
                        ))
                        
                        fig_eval.update_layout(
                            paper_bgcolor='rgba(0,0,0,0)',
                            plot_bgcolor='rgba(0,0,0,0)',
                            yaxis=dict(
                                range=[0, 1.05], 
                                title="Score (0 to 1)", 
                                gridcolor='rgba(128, 128, 128, 0.2)',
                                tickfont=dict(size=12, weight='bold')
                            ),
                            xaxis=dict(
                                title="", 
                                showgrid=False,
                                tickfont=dict(size=12, weight='bold')
                            ),
                            legend=dict(
                                orientation="h", 
                                yanchor="bottom", 
                                y=1.05, 
                                xanchor="center", 
                                x=0.5,
                                font=dict(size=12, weight='bold')
                            ),
                            margin=dict(l=10, r=10, t=50, b=10),
                            height=350,
                            hovermode="x unified"
                        )
                        st.plotly_chart(fig_eval, width="stretch")
                    except ImportError:
                        st.line_chart(df_runs.set_index('run_at')[['faithfulness', 'answer_correctness']])