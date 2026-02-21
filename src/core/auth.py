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
# Authentication Functions
# ─────────────────────────────────────────────────────────────────────────────

def login_user(email, password):
    """
    Authenticates via Supabase Auth, then IMMEDIATELY fetches the 'role' 
    from the 'public.users' table to ensure real-time accuracy.
    """
    try:
        # 1. Verify Credentials (Login)
        response = supabase.auth.sign_in_with_password({
            "email": email,
            "password": password
        })
        
        if response.user:
            user_id = response.user.id
            
            # 2. 🚀 SOURCE OF TRUTH: Query public.users directly
            # We ignore metadata snapshots and go straight to the live database.
            try:
                profile = supabase.table("users") \
                    .select("role, full_name") \
                    .eq("id", user_id) \
                    .single() \
                    .execute()
                
                # If query succeeds, use that data.
                # If the row is missing (rare edge case), these default to None.
                db_role = profile.data.get("role")
                db_name = profile.data.get("full_name")
                
            except Exception:
                # If the query fails entirely (e.g., row deleted), safer to be a student.
                db_role = "student"
                db_name = "Student"

            return {
                "id": user_id,
                "email": response.user.email,
                "role": db_role if db_role else "student",     # Live Role
                "full_name": db_name if db_name else "Student" # Live Name
            }
            
    except Exception as e:
        if "Email not confirmed" in str(e):
            print("⚠️ Login blocked: Email not verified.")
            return "UNVERIFIED"
        
        print(f"Login Error: {e}")
        return None

def register_user(email, password, full_name="Student"):
    """
    Registers a new user. The Postgres Trigger will automatically 
    copy this user to 'public.users' with the default 'student' role.
    """
    try:
        response = supabase.auth.sign_up({
            "email": email,
            "password": password,
            "options": {
                "data": {
                    "full_name": full_name,
                    # We still send this for the initial Trigger, 
                    # but login_user relies on the table after that.
                    "role": "student" 
                }
            }
        })
        
        if response.user:
            # Check if identity exists (prevents duplicate error masking)
            if response.user.identities and len(response.user.identities) > 0:
                return True, "Registration successful! Please check your email to verify."
            else:
                return False, "User already exists."
                
    except Exception as e:
        return False, str(e)