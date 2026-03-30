import re
from src.config.constants import GREETING_KEYWORDS, OFF_TOPIC_KEYWORDS
from src.config.logging_config import logger

def route_query_fast(query: str) -> str:
    q = query.lower().strip()
    
    # ── FIX: Strip punctuation so "Hi, what is CPE?" isn't flagged as a greeting ──
    q_clean = re.sub(r'[^\w\s]', '', q).strip()
    
    if q_clean in GREETING_KEYWORDS:
        return "greeting"
        
    if any(kw in q for kw in OFF_TOPIC_KEYWORDS):
        return "off_topic"
        
    return "search"

def get_dynamic_k(query: str) -> int:
    q = query.lower()

    curriculum_keywords = ['curriculum', 'subject', 'course', 'year', 'semester', 'units', 'prerequisite']
    if any(kw in q for kw in curriculum_keywords): return 20 
    # Add this check to your get_dynamic_k function (around source: 351)
    if any(kw in q for kw in ["history", "background", "origin", "founded", "established"]):
        return 20  # Increased from the default 12 to capture more narrative context
    complex_signals = [" difference ", " compare ", " vs ", " versus ", " list all ", " what are all "]
    if any(signal in q for signal in complex_signals): return 15

    if any(kw in q for kw in ["thesis", "research", "manuscript", "capstone"]): return 15

    if any(kw in q for kw in ["dean", "chairperson", "chair", "faculty", "professor", "department", "who is", "who are", "staff", "organizational", "org structure", "instructor", "engr", "lab technician", "full-time", "part-time", "is he", "is she", "is the"]):
        return 15

    if any(kw in q for kw in ['room', 'building', 'floor', 'where is', 'where', 'location', 'located', 'lab', 'office', 'campus', 'facility']):
        return 20

    if any(kw in q for kw in ['download', 'link', 'pdf', 'get the', 'access', 'where can i']):
        return 20
    
    if any(kw in q for kw in ["org", "orgs", "organization", "organizations", "club", "clubs", "society", "societies", "extracurricular", "co-curricular"]):
        return 20

    return 12

def route_query(query: str) -> tuple:
    intent = route_query_fast(query)
    logger.info(f"Router | Intent: {intent}")
    
    # ── FIX: Stop retrieving chunks for non-search intents ──
    if intent in ["greeting", "off_topic"]:
        return intent, None, "none", None
        
    return intent, None, "all", None