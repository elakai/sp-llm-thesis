import streamlit as st
import time
from src.core.auth import supabase
from datetime import datetime
from collections import defaultdict

# ─────────────────────────────────────────────────────────────────────────────
# 🛡️ SMART LOGGING FILTER
# ─────────────────────────────────────────────────────────────────────────────
# Responses starting with these phrases will not be saved to chat_logs.
# This keeps your thesis data clean for evaluation.
IGNORED_RESPONSES = [
    "Hello! I am AXIsstant",
    "Hi there! I'm ready",
    "Greetings! Feel free",
    "I am designed to answer questions", # Off-topic guardrail
    "⚠️" # Connection or API errors
]

def save_feedback(query: str, response: str, rating: str, user_id: str = "Anonymous"):
    """
    Updates an existing log entry with a user rating (e.g., helpful/unhelpful).
    Matches based on the most recent query/response pair for the user.
    """
    try:
        # We update the latest entry for this user rather than inserting a new one
        data = {"rating": rating}
        supabase.table("chat_logs") \
            .update(data) \
            .eq("user_email", user_id) \
            .eq("query", query) \
            .execute()
        return True
    except Exception as e:
        print(f"❌ Failed to save feedback: {e}")
        return False

def log_conversation(query, response, user_email, session_id, context, metrics=None):
    """
    Saves the chat interaction, context, and performance metrics.
    Metrics should be a dict: {'retrieval_latency', 'generation_latency', 'total_latency'}
    """
    if any(response.startswith(phrase) for phrase in IGNORED_RESPONSES):
        return

    try:
        data = {
            "query": query,
            "response": response,
            "user_email": user_email,
            "session_id": session_id,
            "context": context,
            # 🚀 New performance columns
            "retrieval_latency": metrics.get("retrieval_latency") if metrics else None,
            "generation_latency": metrics.get("generation_latency") if metrics else None,
            "total_latency": metrics.get("total_latency") if metrics else None
        }
        
        supabase.table("chat_logs").insert(data).execute()
        
    except Exception as e:
        print(f"❌ Backend Logging Error: {e}")
        
def get_user_history(user_email: str, limit: int = 10):
    """
    Fetches the most recent academic conversations for a specific user.
    """
    try:
        response = supabase.table("chat_logs") \
            .select("query, response, created_at") \
            .eq("user_email", user_email) \
            .order("created_at", desc=True) \
            .limit(limit) \
            .execute()
        
        return response.data
    except Exception as e:
        print(f"❌ Error fetching history: {e}")
        return []
    
def load_chat_history(user_email: str):
    """
    Reconstructs past conversations grouped by session for the UI.
    """
    try:
        response = supabase.table("chat_logs") \
            .select("session_id, query, response, created_at") \
            .eq("user_email", user_email) \
            .order("created_at", desc=False) \
            .execute()
        
        if not response.data:
            return []

        # Group rows by Session ID
        sessions = defaultdict(list)
        
        for row in response.data:
            s_id = row.get("session_id")
            if not s_id: 
                continue 
            
            # Reconstruct the message pair for Streamlit chat
            sessions[s_id].append({"role": "user", "content": row["query"]})
            sessions[s_id].append({"role": "assistant", "content": row["response"]})

        return list(sessions.values())

    except Exception as e:
        print(f"❌ Error loading history: {e}")
        return []
    
    import time

def log_conversation_with_metrics(query, response, user_email, session_id, context, start_time):
    """
    Enhanced logger for Thesis purposes.
    Tracks latency to provide data for Performance Analysis chapters.
    """
    latency = time.time() - start_time # Calculate total time taken
    
    data = {
        "query": query,
        "response": response,
        "user_email": user_email,
        "session_id": session_id,
        "context": context,
        "latency": latency, # New column for performance stats
        "timestamp": datetime.now().isoformat()
    }
    
    try:
        supabase.table("chat_logs").insert(data).execute()
    except Exception as e:
        print(f"❌ Logging failed: {e}")