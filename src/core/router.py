import re
from src.config.constants import GREETING_KEYWORDS, OFF_TOPIC_KEYWORDS, TABLE_KEYWORDS
from src.config.logging_config import logger

def extract_metadata_filters(query: str) -> list:
    query_lower = query.lower()
    filters = []
    
    # 1. Check if the user is asking about thesis/manuscripts
    thesis_keywords = ["thesis", "title", "manuscript", "capstone", "project", "research"]
    is_thesis_query = any(kw in query_lower for kw in thesis_keywords)
    
    if is_thesis_query:
        # Route strictly to the Thesis PDFs using regex boundaries
        if re.search(r'\bece\b', query_lower) or "electronics" in query_lower:
            filters.append("Past CSEA Thesis Manuscripts - ECE.pdf")
        if re.search(r'\bce\b', query_lower) or "civil" in query_lower:
            filters.append("Past CSEA Thesis Manuscripts - CE.pdf")
            
        return filters if filters else None

    # 2. Standard Curriculum & Info Routing (Match EXACT filenames)
    if re.search(r'\bece\b', query_lower) or "electronics" in query_lower:
        filters.append("CURRICULUM FOR BACHELOR OF SCIENCE IN ELECTRONICS ENGINEERING (BS ECE).pdf")
        
    if re.search(r'\bce\b', query_lower) or "civil" in query_lower:
        filters.append("CURRICULUM FOR BACHELOR OF SCIENCE IN CIVIL ENGINEERING (BS CE).pdf")
        
    if re.search(r'\barch\b', query_lower) or "architecture" in query_lower:
        filters.append("CURRICULUM FOR BACHELOR OF SCIENCE IN ARCHITECTURE (BS ARCH).pdf")
        
    if re.search(r'\bbio\b', query_lower) or "biology" in query_lower:
        filters.append("CURRICULUM FOR BACHELOR OF SCIENCE IN BIOLOGY (BS BIO).pdf")
        
    if re.search(r'\bcpe\b', query_lower) or "computer" in query_lower:
        filters.append("CURRICULUM FOR BACHELOR OF SCIENCE IN COMPUTER ENGINEERING (BS CpE).pdf")
        
    if re.search(r'\bem\b', query_lower) or "environmental" in query_lower:
        filters.append("CURRICULUM FOR BACHELOR OF SCIENCE IN ENVIRONMENTAL MANAGEMENT (BS EM).pdf")
        
    if re.search(r'\bmath\b', query_lower) or "mathematics" in query_lower:
        filters.append("CURRICULUM FOR BACHELOR OF SCIENCE IN MATHEMATICS (BS MATH).pdf")
        
    return filters if filters else None

def detect_content_type(query: str) -> str:
    if any(kw in query.lower() for kw in TABLE_KEYWORDS): return "table"
    return "all"

def extract_category_filters(query: str) -> str:
    """Detects broad categories to map searches directly to specific subfolders."""
    query_lower = query.lower()
    
    if any(kw in query_lower for kw in ["memo", "memorandum", "circular", "policy"]): 
        return "memos"
    if any(kw in query_lower for kw in ["lab", "manual", "inventory", "equipment"]): 
        return "laboratory"
    if any(kw in query_lower for kw in ["ojt", "intern", "internship", "partner company"]): 
        return "ojt"
    if any(kw in query_lower for kw in ["org chart", "organizational chart", "faculty", "staff"]): 
        return "organization"
        
    return None

def route_query_fast(query: str) -> str:
    """Instant heuristic routing to save LLM calls."""
    q = query.lower().strip()
    if any(q.startswith(g) or q == g for g in GREETING_KEYWORDS): return "greeting"
    if any(kw in q for kw in OFF_TOPIC_KEYWORDS): return "off_topic"
    return "search"

def route_query(query: str) -> tuple:
    filters = extract_metadata_filters(query)
    content_type = detect_content_type(query)
    category_filter = extract_category_filters(query)
    intent = route_query_fast(query)
    
    # DEFENSE DEBUG LOGGING
    logger.info(
        f"🧠 Router Decision | Intent: {intent} | Category: {category_filter} "
        f"| Source Filter: {filters} | Content: {content_type}"
    )
    
    return intent, filters, content_type, category_filter