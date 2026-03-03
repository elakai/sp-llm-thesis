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
    "write a script", "joke", "poem", "story"
]
# ─────────────────────────────────────────────────────────────────────────────
# 3. PIPELINE TRIGGERS & GUARDRAILS
# ─────────────────────────────────────────────────────────────────────────────
DECOMPOSE_TRIGGERS = [" difference between ", " compare ", " vs ", " and the ", " also "]

IGNORED_RESPONSES = [
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
RETRIEVAL_K = 15
RERANKER_TOP_K = 8
CRITIC_CONTEXT_LIMIT = 4000
CONFIDENCE_THRESHOLD = 2.0

# Add this to your existing constants.py
VALID_CATEGORIES = {
    "curriculum": "Curriculum",
    "thesis": "Thesis Manuscripts",
    "memos": "Memorandums & Circulars",
    "ojt": "OJT Requirements",
    "laboratory": "Lab Manuals"
}

#  RAG Confidence Thresholds (Calibrated for ms-marco logits)
# Newer score distributions are often centered around ~[-2, +5].
# Keep critic ON for borderline matches and only bypass when clearly strong.
LOW_CONFIDENCE_THRESHOLD = -13.0   # Below this = truly irrelevant, abort generation
HIGH_CONFIDENCE_THRESHOLD = 2.5    # At/above this can skip critic (with margin check)
HIGH_CONFIDENCE_MARGIN = 0.75      # Top1 must beat Top2 by at least this gap to skip critic

# ⚡ UI Performance
STREAM_DELAY = 0.005  # Seconds per word for the streaming effect