import streamlit as st
from src.core.auth import supabase

def save_feedback(query: str, response: str, rating: str, user_id: str = "Anonymous"):
    """
    Saves user feedback (Thumbs Up/Down) to Supabase.
    """
    try:
        data = {
            "user_email": user_id,
            "query": query,
            "response": response,
            "rating": rating
        }
        
        # Insert into Supabase
        supabase.table("chat_logs").insert(data).execute()
        return True

    except Exception as e:
        print(f"❌ Failed to save feedback: {e}")
        return False

def log_conversation(query: str, response: str, user_id: str):
    """
    Automatically logs the conversation to Supabase (without rating).
    Call this immediately after the bot replies.
    """
    try:
        data = {
            "user_email": user_id,
            "query": query,
            "response": response,
            "rating": None # No rating yet
        }
        supabase.table("chat_logs").insert(data).execute()
        
    except Exception as e:
        print(f"❌ Auto-log failed: {e}")