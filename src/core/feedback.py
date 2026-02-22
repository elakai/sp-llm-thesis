import streamlit as st
from src.core.auth import supabase
from datetime import datetime
from collections import defaultdict

# ─────────────────────────────────────────────────────────────────────────────
# 🛡️ SMART LOGGING FILTER
# ─────────────────────────────────────────────────────────────────────────────
IGNORED_RESPONSES = [
    "Hello! I am AXIsstant",
    "Hi there! I'm ready",
    "Greetings! Feel free",
    "I am designed to answer questions", 
    "⚠️"
]

def log_conversation(query, response, user_email, session_id, context, metrics=None):
    """
    Saves the chat interaction, context, and performance metrics to Supabase.
    """
    # 1. Guardrail: Don't log greetings or errors
    if any(response.startswith(phrase) for phrase in IGNORED_RESPONSES):
        return

    try:
        # 2. Construct Payload
        # We ensure context is never NULL so the evaluation script can find it
        safe_context = context if context and context.strip() != "" else "No context retrieved"
        
        data = {
            "session_id": session_id,
            "user_email": user_email,
            "query": query,
            "response": response,
            "context": safe_context,
            "created_at": datetime.utcnow().isoformat(),
        }

        # 3. Add Metrics (if they exist)
        if metrics:
            data["retrieval_latency"] = metrics.get("retrieval_latency", 0.0)
            data["generation_latency"] = metrics.get("generation_latency", 0.0)
            data["total_latency"] = metrics.get("total_latency", 0.0)

        # 4. Insert into Supabase
        supabase.table("chat_logs").insert(data).execute()
        
    except Exception as e:
        print(f"❌ Backend Logging Error: {e}")

def save_feedback(query: str, response: str, rating: str, user_id: str = "Anonymous"):
    """
    Updates the most recent log entry for this user with a rating.
    """
    try:
        data = {"rating": rating}
        # Updates the latest match. 
        # Note: In a production app, we would use the specific Log ID, 
        # but this works for the thesis prototype.
        supabase.table("chat_logs") \
            .update(data) \
            .eq("user_email", user_id) \
            .eq("query", query) \
            .execute()
        return True
    except Exception as e:
        print(f"❌ Failed to save feedback: {e}")
        return False

def load_chat_history(user_email: str):
    """
    Reconstructs past conversations grouped by session for the UI.
    """
    try:
        if not user_email: return []
        
        response = supabase.table("chat_logs") \
            .select("session_id, query, response, created_at") \
            .eq("user_email", user_email) \
            .order("created_at", desc=False) \
            .execute()
        
        if not response.data:
            return []

        sessions = defaultdict(list)
        for row in response.data:
            s_id = row.get("session_id")
            if s_id:
                sessions[s_id].append({"role": "user", "content": row["query"]})
                sessions[s_id].append({"role": "assistant", "content": row["response"]})

        return list(sessions.values())

    except Exception as e:
        print(f"❌ Error loading history: {e}")
        return []