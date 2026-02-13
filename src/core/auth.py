import os
from dotenv import load_dotenv
from supabase import create_client, Client
from pathlib import Path

# 1. Load Env Vars
env_path = Path(__file__).resolve().parents[2] / ".env"
load_dotenv(env_path)

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

if not SUPABASE_URL or not SUPABASE_KEY:
    raise ValueError("⚠️ Supabase keys not found in .env file!")

# 2. Initialize Supabase Client
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# ─────────────────────────────────────────────────────────────────────────────
# Authentication Functions (Native Supabase Auth)
# ─────────────────────────────────────────────────────────────────────────────

def login_user(email, password):
    """
    Authenticates using Supabase Native Auth.
    Returns: User Object (with Token) if valid, None if invalid.
    """
    try:
        # 🚀 This generates the JWT Token required for RLS
        response = supabase.auth.sign_in_with_password({
            "email": email,
            "password": password
        })
        
        if response.user:
            return {
                "id": response.user.id,
                "email": response.user.email,
                "role": response.user.user_metadata.get("role", "student"),
                "full_name": response.user.user_metadata.get("full_name", "Student")
            }
    except Exception as e:
        # Handle "Email not confirmed" specifically
        if "Email not confirmed" in str(e):
            print("⚠️ Login blocked: Email not verified.")
            # You could raise a specific error here to show in the UI
            return "UNVERIFIED"
        
        print(f"Login Error: {e}")
        return None

def register_user(email, password, full_name="Student"):
    """
    Registers a new student using Supabase Auth.
    Automatically triggers the 'Confirm Email' process.
    """
    try:
        # 🚀 This sends the confirmation email automatically
        response = supabase.auth.sign_up({
            "email": email,
            "password": password,
            "options": {
                "data": {
                    "full_name": full_name,
                    "role": "student"
                }
            }
        })
        
        # Check if registration was successful
        if response.user:
            if response.user.identities and len(response.user.identities) > 0:
                return True, "Registration successful! Please check your email to verify."
            else:
                return False, "User already exists."
                
    except Exception as e:
        return False, str(e)