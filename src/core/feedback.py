import streamlit as st
from datetime import datetime
from collections import defaultdict
from src.core.auth import supabase
from src.config.constants import IGNORED_RESPONSES
from src.config.logging_config import logger

def log_conversation(query, response, user_email, session_id, context, metrics=None, force_log=False):
    """Saves the chat interaction, context, and performance metrics to Supabase."""
    if not force_log and any(response.startswith(phrase) for phrase in IGNORED_RESPONSES):
        return

    try:
        safe_context = context if context and context.strip() != "" else "No context retrieved"
        
        data = {
            "session_id": session_id,
            "user_email": user_email,
            "query": query,
            "response": response,
            "context": safe_context,
            "created_at": datetime.utcnow().isoformat(),
        }

        if metrics:
            data["retrieval_latency"] = metrics.get("retrieval_latency", 0.0)
            data["generation_latency"] = metrics.get("generation_latency", 0.0)
            data["total_latency"] = metrics.get("total_latency", 0.0)

        supabase.table("chat_logs").insert(data).execute()
        
    except Exception as e:
        logger.error(f"Backend Logging Error: {e}")


def save_feedback(query: str, response: str, rating: str, user_email: str, session_id: str):
    """Updates the specific log entry for this session with a rating."""
    try:
        if not session_id or not user_email:
            logger.error("Feedback rejected: Missing session_id or user_email")
            return False

        supabase.table("chat_logs") \
            .update({"rating": rating}) \
            .eq("session_id", session_id) \
            .eq("user_email", user_email) \
            .eq("query", query) \
            .execute()
        return True
    except Exception as e:
        logger.error(f"Failed to save feedback: {e}")
        return False

def delete_conversation(session_id: str, user_email: str):
    try:
        if not session_id or not user_email:
            return False
        
        supabase.table("chat_logs") \
            .delete() \
            .eq("session_id", session_id) \
            .eq("user_email", user_email) \
            .execute()
        return True
    except Exception as e:
        logger.error(f"Error deleting conversation: {e}")
        return False

def load_chat_history(user_email: str):
    try:
        if not user_email: return []
        
        response = supabase.table("chat_logs") \
            .select("session_id, query, response, created_at") \
            .eq("user_email", user_email) \
            .order("created_at", desc=False) \
            .execute()
        
        if not response.data: return []

        sessions = defaultdict(list)
        session_order = [] 
        
        for row in response.data:
            s_id = row.get("session_id")
            if s_id:
                if s_id not in session_order:
                    session_order.append(s_id)
                sessions[s_id].append({"role": "user", "content": row["query"]})
                sessions[s_id].append({"role": "assistant", "content": row["response"]})

        history_list = []
        for s_id in session_order:
            history_list.append({
                "session_id": s_id,
                "messages": sessions[s_id]
            })
            
        return history_list

    except Exception as e:
        logger.error(f"Error loading history: {e}")
        return []