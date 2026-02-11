import streamlit as st
from src.core.auth import supabase
from datetime import datetime
from collections import defaultdict


# ─────────────────────────────────────────────────────────────────────────────
# 🛡️ SMART LOGGING FILTER
# ─────────────────────────────────────────────────────────────────────────────
# If the bot's response starts with any of these phrases, we won't save it.
# These match the exact strings in your 'router.py' and 'retrieval.py'.
IGNORED_RESPONSES = [
    "Hello! I am AXIsstant",
    "Hi there! I'm ready",
    "Greetings! Feel free",
    "I am designed to answer questions", # The off-topic guardrail
    "⚠️" # Optional: You might want to ignore connection errors too
]

def save_feedback(query: str, response: str, rating: str, user_id: str = "Anonymous"):
    """
    Saves user feedback (1-5 Stars) to Supabase.
    """
    try:
        data = {
            "user_email": user_id,
            "query": query,
            "response": response,
            "rating": rating
        }
        supabase.table("chat_logs").insert(data).execute()
        return True
    except Exception as e:
        print(f"❌ Failed to save feedback: {e}")
        return False

def log_conversation(query, response, user_email, session_id, context):
    """
    Saves the chat interaction and the retrieved context to Supabase.
    This data is required for Ragas thesis evaluation.
    """
    try:
        data = {
            "query": query,
            "response": response,
            "user_email": user_email,
            "session_id": session_id,
            "context": context  # 🚀 This maps to your new Supabase column
        }
        
        # Insert into the chat_logs table
        result = supabase.table("chat_logs").insert(data).execute()
        
        if result:
            print(f"✅ Logged successfully for session: {session_id}")
            
    except Exception as e:
        # We print this to the terminal so the user doesn't see backend errors
        print(f"❌ Auto-log failed: {e}")

def get_user_history(user_email: str, limit: int = 10):
    """
    Fetches the most recent conversations for a specific user.
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
    Fetches all past conversations for the user and groups them by session.
    Returns: List[List[Dict]] (The exact format the UI expects)
    """
    try:
        # 1. Fetch all logs for this user, ordered by time
        response = supabase.table("chat_logs") \
            .select("session_id, query, response, created_at") \
            .eq("user_email", user_email) \
            .order("created_at", desc=False) \
            .execute()
        
        if not response.data:
            return []

        # 2. Group rows by Session ID
        sessions = defaultdict(list)
        
        for row in response.data:
            s_id = row.get("session_id")
            
            # If old data has no session_id, skip or group into "Legacy"
            if not s_id: 
                continue 
            
            # Reconstruct the message pair
            sessions[s_id].append({"role": "user", "content": row["query"]})
            sessions[s_id].append({"role": "assistant", "content": row["response"]})

        # 3. Convert to list (values only)
        # This returns a list of conversation lists: [[msg1, msg2], [msg3, msg4]]
        return list(sessions.values())

    except Exception as e:
        print(f"❌ Error loading history: {e}")
        return []