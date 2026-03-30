import re
from typing import List, Dict
from src.config.settings import get_generator_llm
import json
from src.config.logging_config import logger

def tokenize(text: str) -> list:
    return re.sub(r'[^\w\s]', ' ', text.lower()).split()

def detect_query_intent(query: str) -> str:
    q_lower = (query or "").lower()
    tokens = set(re.findall(r"[a-z0-9']+", q_lower))

    def has_any_words(words: set[str]) -> bool:
        return any(word in tokens for word in words)

    def has_any_phrases(phrases: list[str]) -> bool:
        return any(phrase in q_lower for phrase in phrases)

    # 1. HIGHLY SPECIFIC: Custodian & Technician
    if has_any_words({"custodian", "custodians", "technician"}) or has_any_phrases(["lab technician", "lab custodians", "laboratory custodian"]):
        return "people"
        
    # 2. HIGHLY SPECIFIC: Location (Expanded for Taglish, directions, and lost freshmen)
    if has_any_words({"where", "room", "building", "located", "floor", "lab", "laboratory", "office", "directory", "saan", "nasaan", "location", "hanapin"}) or has_any_phrases(["where is", "how to go", "paano pumunta", "where can i find", "anong floor"]):
        return "location"
        
    # 3. HIGHLY SPECIFIC: People (MOVED UP to prevent being swallowed by Curriculum)
    people_word_hit = has_any_words({
        "who", "faculty", "faculties", "chair", "chairperson", "dean", "professor", "prof", "profs", "staff", 
        "instructor", "teacher", "teachers", "sino", "field", "dr", "engr", "ar", "sir", "maam", "head", "contact", "email"
    })
    people_phrase_hit = has_any_phrases([
        "chairperson", "department chair", "faculty list", "who is the", "email of", "contact number", "sino si", "sino prof"
    ])
    people_stem_hit = any(token.startswith("chair") for token in tokens)
    
    if people_word_hit or people_phrase_hit or people_stem_hit:
        return "people"
        
    # 4. HIGHLY SPECIFIC: Download
    if has_any_words({"download", "link", "pdf", "form", "access", "copy"}) or has_any_phrases(["google form", "download link", "where to get", "can i have a copy", "hingi copy"]):
        return "download"

    # 5. HIGHLY SPECIFIC: History
    if has_any_words({"history", "background", "origin", "origins", "founded", "established", "anniversary", "heritage", "legacy", "story"}):
        return "history"
        
    # 6. BROADER DOMAIN: Policy 
    if has_any_words({"policy", "rule", "rules", "guideline", "guidelines", "procedure", "manual", "uniform", "haircut", "absent", "late", "bawal", "allowed", "pwede", "shift", "shifting", "drop", "failing", "fail", "id"}) or has_any_phrases(["dress code", "what happens if", "is it allowed", "pwede ba", "color ng buhok", "hair color", "civilian", "wash day"]):
        return "policy"
        
    # 7. BROADER DOMAIN: Organizations
    if has_any_words({"org", "orgs", "organization", "organizations", "club", "clubs", "society", "societies", "extracurricular", "co-curricular"}):
        return "organizations"

    # 8. FALLBACK
    return "general"

def is_listing_query(query: str) -> bool:
    """
    HOLISTIC INTENT DETECTOR:
    Identifies if a query requires reading a full document (lists, overviews, plurals)
    rather than finding a single specific factoid.
    """
    q = query.lower()
    
    # Structural triggers that indicate a desire for a comprehensive answer
    list_triggers = [
        "list", "all the", "what are the", "who are the", "show me all", 
        "every", "types of", "kinds of", "examples of", "directory", 
        "organizations", "orgs", "policies", "rules", "guidelines", "facilities", "extracurricular"
    ]
    
    import re
    plural_regex = r"what are \w+s\b"
    
    return any(trigger in q for trigger in list_triggers) or bool(re.search(plural_regex, q))

def is_people_list_query(query: str) -> bool:
    q = (query or "").lower()
    people_markers = [
        "faculty", "faculties", "staff", "professor", "professors", "instructor", 
        "instructors", "teacher", "teachers", "dean", "chair", "chairperson", 
        "chairpersons", "department chair", "department chairs"
    ]
    if detect_query_intent(q) != "people" and not any(marker in q for marker in people_markers):
        return False
    return is_listing_query(q)

def is_curriculum_list_query(query: str) -> bool:
    q = (query or "").lower()
    curriculum_markers = [
        "curriculum", "subject", "subjects", "course", "courses",
        "semester", "year level", "year", "units", "prerequisite"
    ]
    if detect_query_intent(q) != "curriculum" and not any(marker in q for marker in curriculum_markers):
        return False
    return is_listing_query(q) or "all subjects" in q or "all courses" in q

def is_incomplete_query(query: str) -> bool:
    q_lower = (query or "").strip().lower()
    if re.search(r'\b[a-zA-Z]{1,5}\s*\d{3}\b', query): return False
    tokens = re.findall(r"[a-zA-Z0-9']+", q_lower)
    if len(tokens) <= 2: return True
    starters = {"what", "where", "who", "when", "why", "how", "which"}
    verbs = {"is", "are", "can", "do", "does", "did", "show", "list", "find", "tell"}
    return len(tokens) <= 4 and not any(t in starters or t in verbs for t in tokens)

def build_incomplete_query_variants(query: str, chat_history_list: List[Dict[str, str]]) -> List[str]:
    base = (query or "").strip()
    if not base: return []
    intent = detect_query_intent(base)
    variants = [base]

    if intent == "location": variants.append(f"{base} location building room")
    elif intent == "curriculum": variants.append(f"{base} curriculum course prerequisite")
    elif intent == "people": variants.append(f"{base} faculty staff department chair professor")
    elif intent == "download": variants.append(f"{base} official link pdf")
    elif intent == "policy": variants.append(f"{base} policy guideline rule")

    for msg in reversed(chat_history_list):
        if msg.get("role") == "user" and (msg.get("content") or "").strip().lower() != base.lower():
            variants.append(f"{msg.get('content').strip()} {base}")
            break

    deduped, seen = [], set()
    for v in variants:
        norm = v.strip().lower()
        if norm and norm not in seen:
            deduped.append(v.strip())
            seen.add(norm)
    return deduped[:2] 

def is_custodian_lookup_query(query: str) -> bool:
    q = (query or "").lower()
    asks_person = any(term in q for term in ["custodian", "in charge", "responsible", "handler", "assigned"])
    asks_place = any(term in q for term in ["lab", "laboratory", "room", "aelab", "ae lab", "commslab", "comms lab", "cisco lab"])
    return asks_person and asks_place

def is_custodian_list_query(query: str) -> bool:
    q = (query or "").lower()
    asks_custodian = any(term in q for term in ["custodian", "custodians", "lab technician", "technician"])
    asks_labs = any(term in q for term in ["lab", "labs", "laboratory", "laboratories"])
    return asks_custodian and asks_labs and is_listing_query(q)

def normalize_lab_aliases(query: str) -> str:
    if not query: return query
    normalized = query
    alias_patterns = [
        (r"\bae\s*[-_]?\s*lab\b", "Advanced Electronics Laboratory (AE Lab)"),
        (r"\baelab\b", "Advanced Electronics Laboratory (AE Lab)"),
        (r"\bcomms\s*[-_]?\s*lab\b", "Communications Laboratory (Comms Lab)"),
        (r"\bcommslab\b", "Communications Laboratory (Comms Lab)"),
        (r"\bcisco\s*[-_]?\s*lab\b", "CISCO Networking Academy Computer Laboratory (CISCO Lab)"),
        (r"\bciscolab\b", "CISCO Networking Academy Computer Laboratory (CISCO Lab)"),
    ]
    for pattern, replacement in alias_patterns:
        normalized = re.sub(pattern, replacement, normalized, flags=re.IGNORECASE)
    return normalized

def normalize_course_codes(text: str) -> str:
    return re.sub(
        r'\b([A-Za-z]{2,5})[-\s]*(\d{3})\b',
        lambda m: f"{m.group(1).upper()}{m.group(2)}",
        text
    )

def expand_and_normalize_query(user_query: str) -> list[str]:
    """
    Takes a variable user query and normalizes it into highly optimized
    search strings to guarantee semantic matches regardless of phrasing.
    """
    prompt = f"""You are a search query optimization engine for Ateneo de Naga University's College of Science, Engineering, and Architecture (CSEA).
    The user asked: "{user_query}"
    
    Your job is to rewrite this question to maximize retrieval from a vector database. 
    1. Identify the core intent (e.g., finding a room, checking a prerequisite, listing faculty).
    2. Translate slang or conversational phrasing into formal academic terms.
    3. Generate 3 distinct search strings:
       - 'canonical': The formal, perfectly phrased version of the question.
       - 'keyword_rich': A string of related academic nouns and synonyms (no fluff words).
       - 'document_style': How the answer would literally look in a formal handbook or syllabus.

    Respond ONLY with a valid JSON object in this exact format:
    {{"canonical": "...", "keyword_rich": "...", "document_style": "..."}}
    """
    
    try:
        llm = get_generator_llm()
        response = llm.invoke(prompt).content.strip()
        
        # Clean up in case the LLM added markdown formatting (like ```json)
        if response.startswith("```json"):
            response = response[7:-3].strip()
        elif response.startswith("```"):
            response = response[3:-3].strip()
            
        expanded = json.loads(response)
        
        # Return the distinct, optimized queries
        return [
            user_query, # Always keep the original just in case
            expanded.get("canonical", ""),
            expanded.get("keyword_rich", ""),
            expanded.get("document_style", "")
        ]
    except Exception as e:
        logger.error(f"Query Expansion failed: {e}")
        return [user_query] # Safe fallback

def expand_query_semantics(query: str) -> str:
    """
    THESIS FEATURE: Semantic Query Expansion
    Translates local student slang and Taglish into formal search terms.
    """
    slang_triggers = {
        "topnotcher", "topnotchers", "orgs", "ina", 
        "bagsak", "fail", "failed", "pasa", "pass", "passed",
        "lipat", "shift", "shifter", "drop", "prof", "teacher", 
        "sir", "maam", "kailangan", "need to take", "pre-req", "prereq"
    }
    
    q_tokens = set(re.findall(r'\b\w+\b', query.lower()))
    
    if not any(trigger in q_tokens for trigger in slang_triggers) and not any(trigger in query.lower() for trigger in slang_triggers):
        return query 
        
    try:
        llm = get_generator_llm()
        expansion_prompt = f"""
        You are a search optimization module for Ateneo de Naga University (ADNU).
        Rewrite the user's query into formal academic terminology to maximize vector database retrieval.
        KNOWN TRANSLATIONS:
        - "topnotcher" -> "top national passer, highest board exam passer, licensure examination"
        - "orgs" -> "student organizations, extra-curriculars, clubs"
        - "ina" -> "Our Lady of Peñafrancia"
        - "bagsak" / "fail" -> "retake, prerequisite failure, academic probation, failing grade"
        - "lipat" / "shift" -> "shifting program, transfer of degree, change of course"
        - "kailangan" / "need to take" -> "required subjects, prerequisites, curriculum"
        - "prof" / "teacher" -> "faculty member, instructor, department chair"
        
        CRITICAL RULE: DO NOT write a conversational sentence. DO NOT add fluff like "Do you have a list of...". 
        ONLY output a raw, concise string of keywords. 
        
        User Query: {query}
        Expanded Keywords ONLY:"""
        
        response = llm.invoke(expansion_prompt).content.strip()
        logger.info(f"✨ Semantic Expansion Triggered: {response}")
        return response
        
    except Exception as e:
        logger.warning(f"Query expansion failed: {e}. Falling back to original.")
        return query