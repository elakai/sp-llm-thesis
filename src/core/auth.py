from supabase import create_client, Client
from src.config.settings import SUPABASE_URL, SUPABASE_KEY
from src.config.logging_config import logger

# Initialize Supabase Client using the centralized keys from settings.py
def create_supabase_client() -> Client:
    return create_client(SUPABASE_URL, SUPABASE_KEY)


def normalize_role(role) -> str:
    return str(role).strip().lower() if role else "student"


supabase: Client = create_supabase_client()

# ─────────────────────────────────────────────────────────────────────────────
# Authentication Functions
# ─────────────────────────────────────────────────────────────────────────────

def login_user(email, password):
    """
    Authenticates via Supabase Auth, then IMMEDIATELY fetches the 'role' 
    from the 'public.users' table to ensure real-time accuracy.
    """
    try:
        auth_client = create_supabase_client()

        # 1. Verify Credentials (Login)
        response = auth_client.auth.sign_in_with_password({
            "email": email,
            "password": password
        })
        
        if response.user:
            user_id = response.user.id
            
            # 2. 🚀 SOURCE OF TRUTH: Query public.users directly
            # We ignore metadata snapshots and go straight to the live database.
            try:
                profile = auth_client.table("users") \
                    .select("role, full_name") \
                    .eq("id", user_id) \
                    .single() \
                    .execute()
                
                # If query succeeds, use that data.
                # If the row is missing (rare edge case), these default to None.
                db_role = profile.data.get("role")
                db_name = profile.data.get("full_name")
                
            except Exception as e:
                # If the query fails entirely (e.g., row deleted or DB unreachable),
                # default to student to fail safe.  Log as error because an admin
                # hitting this path would silently receive a downgraded role.
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
        auth_client = create_supabase_client()
        response = auth_client.auth.sign_up({
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