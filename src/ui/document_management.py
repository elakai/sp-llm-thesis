import streamlit as st
import pandas as pd
import time
import math

from src.config.settings import get_vectorstore
from src.core.memory_ingestion import ingest_uploaded_files
from src.core.ingestion import delete_document, verify_sync, purge_all_vectors
from src.config.constants import VALID_CATEGORIES

# Import shared UI helpers and cache clearers from the admin dashboard
from src.ui.admin_dashboard import (
    inject_admin_styles, 
    get_manifest_cached, 
    get_vector_count_cached, 
    fetch_eval_metrics
)

# ── POPUP DIALOG FOR INSPECTING CHUNKS ──
@st.dialog("🔎 Inspect LLM Chunks", width="large")
def show_inspect_dialog(inspect_target):
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
            
            st.markdown(f"**Document:** `{inspect_target}` &nbsp;&nbsp;|&nbsp;&nbsp; **Chunks:** `{len(matches)}`")
            
            if not matches:
                st.warning("No chunks found in Pinecone. This file might be orphaned, or still indexing.")
            else:
                with st.container(height=480):
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

# ── POPUP DIALOG FOR KNOWLEDGE MAP ──
@st.dialog("🌌 Vector Knowledge Map", width="large")
def show_knowledge_map_dialog(algo, dims):
    with st.spinner(f"Extracting Vectors from Pinecone and Running {algo}..."):
        try:
            import numpy as np
            from sklearn.decomposition import PCA
            from sklearn.manifold import TSNE
            import plotly.express as px
            
            vectorstore = get_vectorstore()
            idx = vectorstore._index
            
            query_vec = vectorstore.embeddings.embed_query("University policies handbook computer engineering department")
            res = idx.query(vector=query_vec, top_k=300, include_values=True, include_metadata=True)
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
                
                custom_colors = [
                    '#FF950A', '#3B82F6', '#10B981', '#8B5CF6', '#EF4444', '#F43F5E', '#14B8A6', '#F59E0B',
                    '#0EA5E9', '#D946EF', '#84CC16', '#F97316', '#06B6D4', '#6366F1', '#EC4899', '#EAB308',
                    '#A855F7', '#22C55E', '#34D399', '#F87171', '#60A5FA', '#C084FC', '#FB923C', '#A3E635',
                    '#2DD4BF', '#F472B6', '#4ADE80', '#818CF8', '#E879F9', '#FACC15', '#FB7185', '#38BDF8',
                    '#9333EA', '#059669', '#EA580C', '#4F46E5', '#BE123C', '#0F766E', '#B45309', '#1D4ED8'
                ]

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
                    paper_bgcolor='rgba(128,128,128,0.05)',
                    plot_bgcolor='rgba(128,128,128,0.05)',
                    margin=dict(l=10, r=10, t=30, b=250), # ── Increased bottom margin ──
                    legend=dict(
                        title=dict(side="top"), # ── Forces the title above the items ──
                        orientation="h",       # ── Horizontal grid orientation ──
                        yanchor="top", 
                        y=-0.15,               # ── Placed below the chart ──
                        xanchor="left",        # ── Anchored to the left so it expands to the right ──
                        x=0, 
                        bgcolor="rgba(255,255,255,0.85)", 
                        font=dict(size=10)
                    ),
                    hoverlabel=dict(bgcolor="white", font_size=13, font_family="sans-serif", font_color="black"),
                    height=650                 # ── Increased height to fit the expanded legend ──
                )
                
                if dims == 2:
                    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,0.15)', zeroline=False, title_text="")
                    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,0.15)', zeroline=False, title_text="")
                else:
                    fig.update_layout(
                        scene=dict(
                            xaxis=dict(showgrid=True, gridwidth=2, showbackground=True, backgroundcolor='rgba(128,128,128,0.05)', gridcolor='rgba(128,128,128,0.25)', zerolinecolor='rgba(128,128,128,0.25)', title_text=""),
                            yaxis=dict(showgrid=True, gridwidth=2, showbackground=True, backgroundcolor='rgba(128,128,128,0.05)', gridcolor='rgba(128,128,128,0.25)', zerolinecolor='rgba(128,128,128,0.25)', title_text=""),
                            zaxis=dict(showgrid=True, gridwidth=2, showbackground=True, backgroundcolor='rgba(128,128,128,0.05)', gridcolor='rgba(128,128,128,0.25)', zerolinecolor='rgba(128,128,128,0.25)', title_text=""),
                        )
                    )
                
                st.plotly_chart(fig, width="stretch", theme=None)

        except Exception as e:
            st.error(f"Error generating map: {e}")

# ── CONSOLIDATED DOCUMENT MANAGEMENT VIEW ──
def render_indexed_documents_view():
    inject_admin_styles()
    
    if 'library_page' not in st.session_state:
        st.session_state.library_page = 0

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
            )

            if uploaded_files:
                for f in uploaded_files:
                    if f.size > 50 * 1024 * 1024:
                        st.warning(f"Note: **{f.name}** is over 50 MB. This may take a while to process.")

            if st.button("Index", type="primary", width="stretch", disabled=not uploaded_files):
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
                
                get_manifest_cached.clear()
                get_vector_count_cached.clear()
                fetch_eval_metrics.clear()

                for fname, success, message in results:
                    if success:
                        st.success(f"Success: **{fname}**: {message}")
                    else:
                        st.error(f"Error: **{fname}**: {message}")

    with tab_library:
        with st.container(border=True):
            count_display = get_vector_count_cached()
                
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
            
            manifest = get_manifest_cached()

            if not manifest:
                st.info("No documents indexed yet. Upload files from the 'Upload & Index' tab to get started.")
            else:
                if "selected_docs" not in st.session_state:
                    st.session_state["selected_docs"] = set()
                if "table_key" not in st.session_state:
                    st.session_state["table_key"] = 0

                # ── GLOBAL SORTING LOGIC ADDED HERE ──
                col_spacer, col_sort = st.columns([7.5, 2.5])
                with col_sort:
                    def reset_page():
                        st.session_state.library_page = 0
                        
                    sort_order = st.selectbox(
                        "Sort by",
                        ["Date Indexed (Newest)", "Date Indexed (Oldest)", "Document Name (A-Z)", "Document Name (Z-A)"],
                        index=0,
                        label_visibility="collapsed",
                        on_change=reset_page
                    )
                
                def safe_date(date_str):
                    try:
                        return pd.to_datetime(date_str, utc=True)
                    except Exception:
                        return pd.to_datetime("1970-01-01", utc=True)
                        
                if sort_order == "Date Indexed (Newest)":
                    sorted_manifest = sorted(manifest.items(), key=lambda x: safe_date(x[1].get("uploaded_at", "")), reverse=True)
                elif sort_order == "Date Indexed (Oldest)":
                    sorted_manifest = sorted(manifest.items(), key=lambda x: safe_date(x[1].get("uploaded_at", "")))
                elif sort_order == "Document Name (A-Z)":
                    sorted_manifest = sorted(manifest.items(), key=lambda x: x[0].lower())
                else:
                    sorted_manifest = sorted(manifest.items(), key=lambda x: x[0].lower(), reverse=True)

                top_controls_placeholder = st.empty()

                df_data = []
                for filename, info in sorted_manifest:
                    raw_date = info.get("uploaded_at", "Unknown")
                    try:
                        dt_obj = pd.to_datetime(raw_date, utc=True)
                        formatted_date = dt_obj.tz_convert('Asia/Manila').strftime('%b %d, %Y • %I:%M %p')
                    except Exception:
                        formatted_date = raw_date
                        
                    df_data.append({
                        "SELECT": filename in st.session_state["selected_docs"],
                        "DOCUMENT NAME": filename,
                        "CHUNKS": info.get("chunks", 0),
                        "DATE INDEXED": formatted_date
                    })
                df = pd.DataFrame(df_data)

                ROWS_PER_PAGE = 15
                total_rows = len(df)
                total_pages = max(1, math.ceil(total_rows / ROWS_PER_PAGE))
                
                if st.session_state.library_page >= total_pages:
                    st.session_state.library_page = max(0, total_pages - 1)
                    
                start_idx = st.session_state.library_page * ROWS_PER_PAGE
                end_idx = start_idx + ROWS_PER_PAGE
                
                page_df = df.iloc[start_idx:end_idx]
                dynamic_height = (len(page_df) * 35) + 43

                edited_page_df = st.data_editor(
                    page_df,
                    column_config={
                        "SELECT": st.column_config.CheckboxColumn("SELECT", width="small"),
                        "DOCUMENT NAME": st.column_config.TextColumn("DOCUMENT NAME", disabled=True, width="large"),
                        "CHUNKS": st.column_config.NumberColumn("CHUNKS", disabled=True, width="small"),
                        "DATE INDEXED": st.column_config.TextColumn("DATE INDEXED", disabled=True, width="medium"),
                    },
                    hide_index=True,
                    width="stretch",
                    height=dynamic_height,
                    key=f"doc_table_editor_{st.session_state['table_key']}_{st.session_state.library_page}"
                )

                current_page_files = set(page_df["DOCUMENT NAME"])
                selected_on_page = set(edited_page_df[edited_page_df["SELECT"] == True]["DOCUMENT NAME"])
                deselected_on_page = current_page_files - selected_on_page

                st.session_state["selected_docs"].update(selected_on_page)
                st.session_state["selected_docs"].difference_update(deselected_on_page)
                
                selected = st.session_state["selected_docs"]

                if "inspect_doc" in st.session_state and st.session_state["inspect_doc"] not in selected:
                    del st.session_state["inspect_doc"]

                with top_controls_placeholder.container():
                    if st.session_state.get("confirm_bulk_delete", False) and selected:
                        st.error("⚠️ Permanently Delete These Documents? This Action Cannot be Undone.")
                    
                    col_sel, col_space, col_btn1, col_btn2 = st.columns([1.5, 5.5, 1.5, 1.5])
                    
                    with col_sel:
                        is_all_selected = len(st.session_state["selected_docs"]) == len(manifest.keys()) and len(manifest) > 0
                        if is_all_selected:
                            if st.button("Deselect All", key="toggle_all", width="stretch"):
                                st.session_state["selected_docs"] = set()
                                st.session_state["table_key"] += 1 
                                st.session_state.pop("inspect_doc", None) 
                                st.rerun()
                        else:
                            if st.button("Select All", key="toggle_all", width="stretch"):
                                st.session_state["selected_docs"] = set(manifest.keys())
                                st.session_state["table_key"] += 1  
                                st.session_state.pop("inspect_doc", None) 
                                st.rerun()
                    
                    if selected:
                        if not st.session_state.get("confirm_bulk_delete", False):
                            with col_btn1:
                                if len(selected) == 1:
                                    if st.button("Inspect", width="stretch"):
                                        st.session_state["inspect_doc"] = list(selected)[0]
                                        st.session_state["confirm_bulk_delete"] = False
                                        show_inspect_dialog(list(selected)[0])
                                else:
                                    st.button("Inspect", width="stretch", disabled=True)
                            with col_btn2:
                                if st.button("Delete", type="primary", width="stretch"):
                                    st.session_state["confirm_bulk_delete"] = True
                                    st.rerun()
                        else:
                            with col_btn1:
                                if st.button("Yes", type="primary", width="stretch"):
                                    progress = st.progress(0, text="Deleting…")
                                    deleted, failed = 0, 0
                                    files_to_delete = list(selected)
                                    for i, fname in enumerate(files_to_delete):
                                        progress.progress((i + 1) / len(files_to_delete), text=f"Deleting {fname}…")
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
                                    
                                    get_manifest_cached.clear()
                                    get_vector_count_cached.clear()
                                    fetch_eval_metrics.clear()

                                    st.session_state["selected_docs"] = set()
                                    st.session_state["confirm_bulk_delete"] = False
                                    st.session_state["table_key"] += 1
                                    st.rerun()
                            with col_btn2:
                                if st.button("Cancel", width="stretch"):
                                    st.session_state["confirm_bulk_delete"] = False
                                    st.rerun()

                col_info, col_prev, col_next = st.columns([7.6, 1.2, 1.2])
                with col_info:
                    st.markdown(f"""
                        <div class="pagination-pill">
                            PAGE <span>{st.session_state.library_page + 1}</span> OF <span>{total_pages}</span>
                        </div>
                    """, unsafe_allow_html=True)
                with col_prev:
                    if st.button("Previous", key="prev_lib", disabled=st.session_state.library_page == 0, width="stretch"):
                        st.session_state.library_page -= 1
                        st.rerun()
                with col_next:
                    if st.button("Next", key="next_lib", type="primary", disabled=st.session_state.library_page >= total_pages - 1, width="stretch"):
                        st.session_state.library_page += 1
                        st.rerun()

    with tab_advanced:
        with st.container(border=True):
            st.markdown("""
                <div style='padding: 20px; background-color: rgba(128,128,128,0.03); border-radius: 10px; border: 1px solid rgba(128,128,128,0.1); margin-bottom: 20px;'>
                    <h3 style='margin: 0; padding: 0; font-size: 1.5rem; font-weight: 700; color: inherit;'>🌌 Vector Knowledge Map</h3>
                    <div style='font-size: 0.9rem; opacity: 0.85; margin-top: 10px; line-height: 1.6;'>
                        Compress High-Dimensional Document Vectors into 2D or 3D Space to Visualize Semantic Relationships and Topic Clusters.
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
                algo = c1.selectbox("Algorithm", ["PCA", "t-SNE"])
                dims = c2.selectbox("Dimensions", [2, 3], index=1)
                
                if st.button("Generate", type="primary", width="stretch"):
                    show_knowledge_map_dialog(algo, dims)

        with st.container(border=True):
            st.markdown("""
                <div style='padding: 20px; background-color: rgba(128,128,128,0.03); border-radius: 10px; border: 1px solid rgba(128,128,128,0.1); margin-bottom: 20px;'>
                    <h3 style='margin: 0; padding: 0; font-size: 1.5rem; font-weight: 700; color: inherit;'>⚙️ Pinecone Synchronization</h3>
                    <div style='font-size: 0.9rem; opacity: 0.7; margin-top: 5px;'>Cross-Reference the Supabase Manifest against Actual Vectors in Pinecone to Detect Orphaned Data.</div>
                </div>
            """, unsafe_allow_html=True)

            if st.button("Verify Database Sync", width="stretch"):
                with st.spinner("Checking Sync Status…"):
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
                            get_manifest_cached.clear()
                            get_vector_count_cached.clear()
                            fetch_eval_metrics.clear()
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