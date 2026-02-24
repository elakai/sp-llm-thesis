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
OFF_TOPIC_KEYWORDS = ["recipe", "cook", "game", "weather", "code", "python", "javascript"]
TABLE_KEYWORDS = ["curriculum", "grade", "grading", "schedule", "fee", "list", "table", "rubric", "courses"]

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
RERANKER_TOP_K = 6
CRITIC_CONTEXT_LIMIT = 2000
CONFIDENCE_THRESHOLD = 2.0

# Add this to your existing constants.py
VALID_CATEGORIES = {
    "curriculum": "Curriculum",
    "thesis": "Thesis Manuscripts",
    "memos": "Memorandums & Circulars",
    "ojt": "OJT Requirements",
    "laboratory": "Lab Manuals",
    "organization": "Organizational Charts"
}