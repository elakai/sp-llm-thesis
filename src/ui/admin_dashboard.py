import streamlit as st
import pandas as pd
import time
import os
import re
import math
from collections import Counter
from src.config.settings import get_vectorstore
from src.core.auth import supabase
from src.core.ingestion import get_uploaded_files

# ── SHARED CACHE HELPERS ──
@st.cache_data(ttl=30, show_spinner=False)
def fetch_eval_metrics():
    try:
        response = supabase.table("chat_logs") \
            .select("session_id, rating, query, user_email, created_at, retrieval_latency, generation_latency, total_latency") \
            .order("created_at", desc=True) \
            .limit(3000) \
            .execute()
        return pd.DataFrame(response.data)
    except Exception as e:
        st.error(f"🚨 Database Error fetching chat logs: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=30, show_spinner=False)
def fetch_evaluation_runs():
    try:
        response = supabase.table("evaluation_runs") \
            .select("*") \
            .order("run_at", desc=True) \
            .limit(50) \
            .execute()
        return pd.DataFrame(response.data) if response.data else pd.DataFrame()
    except Exception:
        return pd.DataFrame()

@st.cache_data(ttl=60, show_spinner=False)
def fetch_user_names():
    """Fetches user full names from Supabase to map against emails."""
    try:
        response = supabase.table("users").select("email, full_name").execute()
        return {row["email"]: row["full_name"] for row in response.data} if response.data else {}
    except Exception:
        return {}

@st.cache_data(ttl=15, show_spinner=False)
def get_manifest_cached():
    return get_uploaded_files()

@st.cache_data(ttl=60, show_spinner=False)  
def get_vector_count_cached() -> str:
    try:
        vectorstore = get_vectorstore()
        stats = vectorstore._index.describe_index_stats()
        return f"{stats.get('total_vector_count', 0):,}"
    except Exception:
        return "Offline"

@st.cache_data(show_spinner=False)
def load_admin_css() -> str:
    css_path = os.path.join(os.path.dirname(__file__), "styles", "admin.css")
    try:
        with open(css_path, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return ""

def inject_admin_styles():
    css = load_admin_css()
    if css:
        st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)
    else:
        st.warning("⚠️ CSS file not found at src/ui/styles/admin.css. Styles may not load correctly.")

def generate_saas_table_html(df, is_scrollable=False):
    if df.empty:
        return "<div style='padding: 20px; text-align: center; opacity: 0.6;'>No data available.</div>"

    container_class = "saas-table-scroll" if is_scrollable else "saas-table-page"
    html = f'<div class="{container_class}"><table class="saas-table">'
    
    html += '<thead><tr>'
    for col in df.columns:
        if col == "User":
            html += f'<th style="width: 200px;">{col}</th>'
        elif col == "Time":
            html += f'<th style="width: 180px;">{col}</th>'
        elif col == "Rating":
            html += f'<th style="width: 130px;">{col}</th>'
        else:
            html += f'<th>{col}</th>'
    html += '</tr></thead><tbody>'

    for _, row in df.iterrows():
        html += '<tr>'
        for col in df.columns:
            val = row[col]
            if pd.isna(val):
                val = "-"
            
            if col == "Rating":
                val_str = str(val).lower()
                if val_str == "helpful":
                    html += f'<td style="vertical-align: middle;"><div class="table-badge badge-helpful">HELPFUL</div></td>'
                elif val_str == "not_helpful" or val_str == "not helpful":
                    html += f'<td style="vertical-align: middle;"><div class="table-badge badge-not-helpful">NOT HELPFUL</div></td>'
                else:
                    html += f'<td style="vertical-align: middle;"><div class="table-badge badge-neutral">{val}</div></td>'
            elif col == "Time":
                html += f'<td style="white-space: nowrap; color: #6B7280; font-size: 0.8rem; font-weight: 500;">{val}</td>'
            elif col == "User":
                html += f'<td style="font-weight: 500;">{val}</td>'
            else:
                html += f'<td>{val}</td>'
        html += '</tr>'

    html += '</tbody></table></div>'
    return html

# ── ADMIN DASHBOARD VIEW ──
def render_admin_view():
    inject_admin_styles()
    
    if 'failed_queries_page' not in st.session_state:
        st.session_state.failed_queries_page = 0

    st.markdown("""
        <div class="admin-page-header">
            <div class="admin-page-title-row">
                <div class="admin-page-icon">🛠️</div>
                <h1 class="admin-page-title">Admin Dashboard</h1>
            </div>
            <p class="admin-page-subtitle">Monitor System Health, Analyze User Engagement, and Inspect RAG Evaluation Metrics.</p>
        </div>
    """, unsafe_allow_html=True)

    # Fetch live data
    df_eval = fetch_eval_metrics()
    df_runs = fetch_evaluation_runs().copy()
    user_mapping = fetch_user_names()

    # Apply mapping logic for Registered Users vs Guests
    if not df_eval.empty and "user_email" in df_eval.columns:
        def get_user_display(email):
            if pd.isna(email) or not email or str(email).strip().lower() == "guest":
                return "Guest"
            return user_mapping.get(email, email)
            
        df_eval["full_name"] = df_eval["user_email"].apply(get_user_display)

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

        count_str = get_vector_count_cached()
        sys_status = "Online" if count_str != "Offline" else "Offline"
        sys_color = "#10B981" if sys_status == "Online" else "#EF4444"
        sys_desc = "Database Connected" if sys_status == "Online" else "Connection Failed"

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

    with tab_performance:
        with st.container(border=True):
            st.markdown("""
                <div style='padding: 20px; background-color: rgba(128,128,128,0.03); border-radius: 10px; border: 1px solid rgba(128,128,128,0.1); margin-bottom: 20px;'>
                    <h3 style='margin: 0; padding: 0; font-size: 1.5rem; font-weight: 700; color: inherit;'>⚡ System Performance Monitor</h3>
                    <div style='font-size: 0.9rem; opacity: 0.7; margin-top: 5px;'>Track AI Response Speeds and Review Live User Feedback Logs.</div>
                </div>
            """, unsafe_allow_html=True)

            if not df_eval.empty and 'total_latency' in df_eval.columns:
                df_perf = df_eval.dropna(subset=['total_latency']).copy()
                
                if not df_perf.empty:
                    total_mean = df_perf['total_latency'].mean()
                    retrieval_mean = df_perf['retrieval_latency'].mean() if 'retrieval_latency' in df_perf.columns else 0.0
                    gen_mean = df_perf['generation_latency'].mean() if 'generation_latency' in df_perf.columns else 0.0
                    
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
                    display_cols = ["full_name", "created_at", "query", "rating"]
                    existing_cols = [c for c in display_cols if c in df_eval.columns]
                    
                    # Because we added .order() in fetch_eval_metrics, it's naturally sorted. But we do it again just to be safe.
                    display_table = df_eval[existing_cols].sort_values(by="created_at", ascending=False)
                    
                    rename_dict = {
                        "full_name": "User",
                        "created_at": "Time",
                        "query": "Query",
                        "rating": "Rating"
                    }
                    display_table = display_table.rename(columns=rename_dict)
                    
                    ordered_final_cols = [rename_dict[c] for c in display_cols if c in df_eval.columns]
                    display_table = display_table[ordered_final_cols]
                    
                    if 'Time' in display_table.columns:
                        display_table['Time'] = pd.to_datetime(display_table['Time'], utc=True).dt.tz_convert('Asia/Manila').dt.strftime('%b %d, %Y • %I:%M %p')
                    
                    html_table = generate_saas_table_html(display_table.head(500), is_scrollable=True)
                    st.markdown(html_table, unsafe_allow_html=True)
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

                stop_words = {"what", "how", "the", "for", "and", "can", "you", "tell", "about", "are", "with", "that", "this", "from", "does", "have", "who", "why", "where"}

                df_eval['created_at'] = pd.to_datetime(df_eval['created_at'])
                
                if 'session_id' in df_eval.columns and not df_eval['session_id'].isna().all():
                    sessions = df_eval.groupby('session_id').agg(
                        start_time=('created_at', 'min'),
                        end_time=('created_at', 'max'),
                        turns=('query', 'count')
                    )
                    sessions['duration_sec'] = (sessions['end_time'] - sessions['start_time']).dt.total_seconds()
                    
                    total_sessions = len(sessions)
                    avg_turns = sessions['turns'].mean()
                    
                    valid_durations = sessions[(sessions['duration_sec'] > 0) & (sessions['duration_sec'] < 1800)]['duration_sec']
                    
                    if not valid_durations.empty:
                        avg_duration = valid_durations.mean()
                    else:
                        avg_duration = 0
                    
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
                                tickfont=dict(size=15, weight='bold'),
                            ),
                            yaxis=dict(
                                showgrid=True, 
                                gridcolor='rgba(255, 255, 255, 0.8)', 
                                tickformat='d',      
                                dtick=1,             
                                tickfont=dict(size=15, weight='bold'), 
                                title_font=dict(size=15, weight='bold'), 
                                title="Number of Users"
                            ),
                            margin=dict(l=10, r=10, t=10, b=10),
                            height=250
                        )
                        fig_dau.update_traces(
                            line_color="#F8F5F1", 
                            fillcolor='rgba(255, 149, 10, 0.7)', 
                            mode='lines+markers', 
                            marker=dict(size=8, color="#A1600A")
                        )
                        st.plotly_chart(fig_dau, width="stretch")
                    else:
                        dau_fallback = dau.rename(columns={'user_email': 'Number of Users'}).set_index('date')
                        st.line_chart(dau_fallback, y="Number of Users")

                st.markdown("<br>", unsafe_allow_html=True)
                col1, col2 = st.columns(2)

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
                                        marker_colors=["#FD9001", "#FFFFFF"], 
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
                        
                        failed_cols = ["full_name", "query", "created_at"]
                        existing_failed_cols = [c for c in failed_cols if c in bad_feedback.columns]
                        
                        failed_table = bad_feedback[existing_failed_cols].sort_values('created_at', ascending=False)
                        
                        failed_rename = {
                            "full_name": "User",
                            "query": "Failed Query", 
                            "created_at": "Time"
                        }
                        failed_table = failed_table.rename(columns=failed_rename)
                        
                        failed_ordered_cols = [failed_rename[c] for c in failed_cols if c in bad_feedback.columns]
                        failed_table = failed_table[failed_ordered_cols]
                        
                        if 'Time' in failed_table.columns:
                            failed_table['Time'] = pd.to_datetime(failed_table['Time'], utc=True).dt.tz_convert('Asia/Manila').dt.strftime('%b %d, %Y • %I:%M %p')
                        
                        ROWS_PER_PAGE = 8
                        total_rows = len(failed_table)
                        total_pages = max(1, math.ceil(total_rows / ROWS_PER_PAGE))
                        
                        if st.session_state.failed_queries_page >= total_pages:
                            st.session_state.failed_queries_page = max(0, total_pages - 1)
                            
                        start_idx = st.session_state.failed_queries_page * ROWS_PER_PAGE
                        end_idx = start_idx + ROWS_PER_PAGE
                        
                        page_df = failed_table.iloc[start_idx:end_idx]

                        html_table = generate_saas_table_html(page_df, is_scrollable=False)
                        st.markdown(html_table, unsafe_allow_html=True)
                        
                        col_info, col_prev, col_next = st.columns([7.6, 1.2, 1.2])
                        with col_info:
                            st.markdown(f"""
                                <div class="pagination-pill">
                                    PAGE <span>{st.session_state.failed_queries_page + 1}</span> OF <span>{total_pages}</span>
                                </div>
                            """, unsafe_allow_html=True)
                        with col_prev:
                            if st.button("Previous", key="prev_fq", disabled=st.session_state.failed_queries_page == 0, width="stretch"):
                                st.session_state.failed_queries_page -= 1
                                st.rerun()
                        with col_next:
                            if st.button("Next", key="next_fq", type="primary", disabled=st.session_state.failed_queries_page >= total_pages - 1, width="stretch"):
                                st.session_state.failed_queries_page += 1
                                st.rerun()
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

            with st.container(border=True):
                st.markdown("""
                    <div style='padding: 12px 16px; background-color: rgba(255, 149, 10, 0.08); border-radius: 8px; border-left: 5px solid #FF950A; margin-bottom: 15px;'>
                        <h4 style='margin: 0; padding: 0; font-size: 1.15rem; font-weight: 700; color: inherit;'>📤 Upload Dataset & Evaluate</h4>
                    </div>
                """, unsafe_allow_html=True)

                st.caption("Your CSV must Contain Two Columns: `Question` and `Ground Truth`.")
                eval_file = st.file_uploader("Upload Test Dataset", type=["csv"], label_visibility="collapsed")

                if st.button("Evaluate", type="primary", width="stretch", disabled=not eval_file):
                    progress_bar = st.progress(5, text="Initializing Evaluation Dataset...")

                    try:
                        uploaded_filename = eval_file.name
                        with open(uploaded_filename, "wb") as f:
                            f.write(eval_file.getbuffer())

                        progress_bar.progress(25, text="Dataset Saved. Booting up AI Judges & RAG Pipeline...")

                        from src.core.evaluate_rag import evaluate_from_dataset

                        with st.spinner("Generating AI Answers and Grading Metrics... (This will take a few minutes)"):
                            evaluate_from_dataset(uploaded_filename) 

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
                        df_plot['run_at'] = pd.to_datetime(df_plot['run_at'], utc=True).dt.tz_convert('Asia/Manila').dt.strftime('%b %d, %H:%M')

                        fig_eval = go.Figure()
                        
                        fig_eval.add_trace(go.Scatter(
                            x=df_plot['run_at'], y=df_plot['faithfulness'],
                            mode='lines+markers', name='Faithfulness',
                            line=dict(color='#8B5CF6', width=5), marker=dict(size=9),
                            fill='tozeroy', fillcolor='rgba(139, 92, 246, 0.1)'
                        ))
                        fig_eval.add_trace(go.Scatter(
                            x=df_plot['run_at'], y=df_plot['answer_correctness'],
                            mode='lines+markers', name='Answer Correctness',
                            line=dict(color='#3B82F6', width=5), marker=dict(size=9),
                            fill='tozeroy', fillcolor='rgba(59, 130, 246, 0.1)'
                        ))
                        
                        fig_eval.update_layout(
                            paper_bgcolor='rgba(0,0,0,0)',
                            plot_bgcolor='rgba(0,0,0,0)',
                            yaxis=dict(
                                range=[0, 1.05], 
                                title="Score (0 to 1)", 
                                gridcolor='rgba(128, 128, 128, 0.7)',
                                tickfont=dict(size=14, weight='bold')
                            ),
                            xaxis=dict(
                                title="", 
                                showgrid=False,
                                tickfont=dict(size=14, weight='bold')
                            ),
                            legend=dict(
                                orientation="h", 
                                yanchor="bottom", 
                                y=1.05, 
                                xanchor="center", 
                                x=0.5,
                                font=dict(size=14, weight='bold')
                            ),
                            margin=dict(l=10, r=10, t=50, b=10),
                            height=350,
                            hovermode="x unified"
                        )
                        st.plotly_chart(fig_eval, width="stretch")
                    except ImportError:
                        st.line_chart(df_runs.set_index('run_at')[['faithfulness', 'answer_correctness']])