from supabase import create_client, Client
from src.config.settings import SUPABASE_URL, SUPABASE_KEY
from src.config.logging_config import logger

# Initialize Supabase Client using the centralized keys from settings.py
import os
from supabase import create_client, Client, ClientOptions

def create_supabase_client() -> Client:
    url = os.environ.get("SUPABASE_URL")
    key = os.environ.get("SUPABASE_KEY")
    
    # ── THIS IS THE MAGIC FIX ──
    # Forces Supabase to skip the memory-based PKCE flow and use Implicit flow
    opts = ClientOptions(flow_type="implicit")
    
    return create_client(url, key, options=opts)

def normalize_role(role) -> str:
    return str(role).strip().lower() if role else "student"

supabase: Client = create_supabase_client()

# ─────────────────────────────────────────────────────────────────────────────
# Authentication Functions
# ─────────────────────────────────────────────────────────────────────────────

def _is_valid_domain(email: str) -> bool:
    """Helper to strictly enforce ADNU domain limits."""
    email = email.lower().strip()
    return email.endswith("@gbox.adnu.edu.ph") or email.endswith("@adnu.edu.ph")

def get_google_login_url():
    """Generates the Google OAuth URL restricted to the gbox domain."""
    try:
        # 1. IMPORTANT: We must create a FRESH client here that uses ClientOptions(flow_type="implicit")
        # If we use the global 'supabase' client defined above, it might default to PKCE/Code flow.
        auth_client = create_supabase_client()
        
        res = auth_client.auth.sign_in_with_oauth({
            "provider": "google",
            "options": {
                # 2. UPDATE THIS: For your live app, this must match your Streamlit Cloud URL
                "redirect_to": "https://pamtest.streamlit.app", 
                "query_params": {
                    "hd": "gbox.adnu.edu.ph"
                }
            }
        })
        return res.url
    except Exception as e:
        logger.error(f"Error generating Google login URL: {e}")
        return None
    
def login_user(email, password):
    """
    Authenticates via Supabase Auth, then IMMEDIATELY fetches the 'role' 
    from the 'public.users' table to ensure real-time accuracy.
    """
    try:
        # ── DOMAIN LOCK ──
        if not _is_valid_domain(email):
            return "INVALID_DOMAIN"

        auth_client = create_supabase_client()

        # 1. Verify Credentials (Login)
        response = auth_client.auth.sign_in_with_password({
            "email": email,
            "password": password
        })
        
        if response.user:
            user_id = response.user.id
            
            # 2. 🚀 SOURCE OF TRUTH: Query public.users directly
            try:
                profile = auth_client.table("users") \
                    .select("role, full_name") \
                    .eq("id", user_id) \
                    .single() \
                    .execute()
                
                db_role = profile.data.get("role")
                db_name = profile.data.get("full_name")
                
            except Exception as e:
                logger.error(f"Role lookup failed for user {user_id}: {e}. Defaulting to 'student'.")
                db_role = "student"
                db_name = "Student"

            return {
                "id": user_id,
                "email": response.user.email,
                "role": normalize_role(db_role),                 # Live Role
                "full_name": db_name if db_name else "Student" # Live Name
            }
            
    except Exception as e:
        if "Email not confirmed" in str(e):
            logger.warning(f"Login blocked for {email}: email not verified.")
            return "UNVERIFIED"

        logger.error(f"Login error for {email}: {e}")
        return None

def register_user(email, password, full_name="Student"):
    """
    Registers a new user. The Postgres Trigger will automatically 
    copy this user to 'public.users' with the default 'student' role.
    """
    try:
        # ── DOMAIN LOCK ──
        if not _is_valid_domain(email):
            return False, "Access Restricted: You must use a valid ADNU email (@gbox.adnu.edu.ph or @adnu.edu.ph) to register."

        auth_client = create_supabase_client()
        response = auth_client.auth.sign_up({
            "email": email,
            "password": password,
            "options": {
                "data": {
                    "full_name": full_name,
                    "role": "student" 
                }
            }
        })
        
        if response.user:
            if response.user.identities and len(response.user.identities) > 0:
                return True, "Registration successful! Please check your email to verify."
            else:
                return False, "User already exists."
                
    except Exception as e:
        return False, str(e)