import os
from dotenv import load_dotenv
from supabase import create_client, Client
from pathlib import Path

# 1. Load Env Vars
env_path = Path(__file__).resolve().parents[2] / ".env.example"
load_dotenv(env_path)

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

if not SUPABASE_URL or not SUPABASE_KEY:
    raise ValueError("⚠️ Supabase keys not found in .env file!")

# 2. Initialize Supabase Client
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# ─────────────────────────────────────────────────────────────────────────────
# Authentication Functions
# ─────────────────────────────────────────────────────────────────────────────

def login_user(email, password):
    """
    Checks if email/password exists in the database.
    Returns: User dictionary if valid, None if invalid.
    """

    clean_email = email.strip() 
    clean_password = password.strip()
    try:
        # Query the 'users' table
        response = supabase.table("users") \
            .select("*") \
            .eq("email", email) \
            .eq("password", password) \
            .execute()

        # Check if we got a match
        if response.data and len(response.data) > 0:
            user = response.data[0]
            return {
                "id": user["id"],
                "email": user["email"],
                "role": user["role"],
                "full_name": user.get("full_name", "Student")
            }
        else:
            return None
            
    except Exception as e:
        print(f"Login Error: {e}")
        return None

def register_user(email, password, full_name="Student"):
    """
    Registers a new student.
    Returns: True if successful, False if email already exists.
    """
    try:
        # 1. Check if email already exists
        check = supabase.table("users").select("email").eq("email", email).execute()
        if check.data:
            return False, "Email already exists."

        # 2. Insert new user
        new_user = {
            "email": email,
            "password": password,
            "full_name": full_name,
            "role": "student"  # Default role
        }
        
        supabase.table("users").insert(new_user).execute()
        return True, "Registration successful!"

    except Exception as e:
        return False, str(e)