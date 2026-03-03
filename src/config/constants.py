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
# Triggers that signal a query spans multiple distinct topics and should be
# decomposed into sub-queries before retrieval.  Deliberately narrow: common
# words like 'and' or 'also' must NOT appear here or decomposition will fire
# on almost every student question, wasting a Groq API call each time.
DECOMPOSE_TRIGGERS = [" difference between ", " compare ", " vs "]

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
# POSITIONAL_SCORE_WEIGHT: blended into BM25 scores during hybrid rerank.
# The value encodes a mild preference for Pinecone's original ranking order.
POSITIONAL_SCORE_WEIGHT = 0.05

# SEMANTIC_CACHE_THRESHOLD: cosine similarity required for a cache hit.
# 0.95 is too strict; two phrasings of the same question score ~0.82–0.88.
SEMANTIC_CACHE_THRESHOLD = 0.88

VALID_CATEGORIES = {
    "curriculum": "Curriculum",
    "thesis": "Thesis Manuscripts",
    "memos": "Memorandums & Circulars",
    "ojt": "OJT Requirements",
    "laboratory": "Lab Manuals"
}

# RAG Confidence Thresholds (Calibrated for ms-marco-MiniLM-L-6-v2 raw logits)
# Score ranges for this model:
#   clearly relevant  : +3 to +10
#   moderately relevant: -1 to +3
#   marginally relevant: -5 to -1
#   unrelated          : -8 to -5
#   completely wrong   : below -10
# LOW_CONFIDENCE_THRESHOLD: abort generation below this — set to -3.0 so the
# gate actually fires on marginally-relevant or worse retrievals.
LOW_CONFIDENCE_THRESHOLD = -3.0    # Below this = not relevant enough, abort generation
HIGH_CONFIDENCE_THRESHOLD = 2.5    # At/above this can skip critic (with margin check)
HIGH_CONFIDENCE_MARGIN = 0.75      # Top1 must beat Top2 by at least this gap to skip critic

# ⚡ UI Performance
STREAM_DELAY = 0.005  # Seconds per word for the streaming effect