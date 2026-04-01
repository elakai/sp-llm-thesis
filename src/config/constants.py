from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
# 1. PATHS & DIRECTORIES
# ─────────────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parents[2]
DOCS_FOLDER = BASE_DIR / "documents"
DATA_FOLDER = BASE_DIR / "data"
LOGS_FOLDER = BASE_DIR / "logs"

# Ensure data and logs folders exist
DATA_FOLDER.mkdir(exist_ok=True)
LOGS_FOLDER.mkdir(exist_ok=True)

MANIFEST_FILE = DATA_FOLDER / "pinecone_manifest.json"

# ─────────────────────────────────────────────────────────────────────────────
# 2. ROUTER & KEYWORDS
# ─────────────────────────────────────────────────────────────────────────────
GREETING_KEYWORDS = ["hi", "hello", "hey", "good morning", "good afternoon", "thanks", "thank you", "bye"]
# 🛑 Off-Topic Keywords (Audit: DO NOT include words like "code", "system", "grade")
OFF_TOPIC_KEYWORDS = [
    "recipe", "cook", "bake", "movie", "weather", "sports", 
    "game", "minecraft", "valorant", "programming help", "code a", 
    "write a script", "joke", "poem"
]
# ─────────────────────────────────────────────────────────────────────────────
# 3. PIPELINE TRIGGERS & GUARDRAILS
# ─────────────────────────────────────────────────────────────────────────────
DECOMPOSE_TRIGGERS = [" difference between ", " compare ", " vs "]

IGNORED_RESPONSES = [
    "Hey! I'm AXIsstant, the academic assistant specifically built",
    "That's outside what I can help with",
    "I don't have enough info to answer that confidently",
    "Hmm, I couldn't find anything about that",
    "Something went wrong on my end",
    "Hello! I am AXIsstant",
    "Hi there! I'm ready",
    "Greetings! Feel free",
    "I am designed to answer questions", 
    "⚠️"
]

# ─────────────────────────────────────────────────────────────────────────────
# 4. RAG TUNING PARAMETERS (For Thesis Defense Tuning)
# ─────────────────────────────────────────────────────────────────────────────
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150
RETRIEVAL_K = 20          # was 15 — wider BM25 net
RERANKER_TOP_K = 12       # was 8 — more chunks reach the LLM
CRITIC_CONTEXT_LIMIT = 4000

POSITIONAL_SCORE_WEIGHT = 0.05
SEMANTIC_CACHE_THRESHOLD = 0.88

VALID_CATEGORIES = {
    "curriculum": "Curriculum",
    "thesis": "Thesis Manuscripts",
    "memos": "Memorandums & Circulars",
    "ojt": "OJT Requirements",
    "laboratory": "Lab Manuals"
}

LOW_CONFIDENCE_THRESHOLD = -8.0    
HIGH_CONFIDENCE_THRESHOLD = 2.5    
HIGH_CONFIDENCE_MARGIN = 0.75      

# ⚡ UI Performance
STREAM_DELAY = 0.003  # Seconds per word for the streaming effect

# Guest mode
GUEST_QUERY_LIMIT = 10

# Campus map link used in location/facility responses.
CAMPUS_MAP_URL = "https://vmsgllaubkzlewtytfkn.supabase.co/storage/v1/object/public/campus/campus%20map-1.png"