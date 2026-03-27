import re
from typing import List, Dict

def tokenize(text: str) -> list:
    return re.sub(r'[^\w\s]', ' ', text.lower()).split()

def detect_query_intent(query: str) -> str:
    q_lower = (query or "").lower()
    tokens = set(re.findall(r"[a-z0-9']+", q_lower))

    def has_any_words(words: set[str]) -> bool:
        return any(word in tokens for word in words)

    def has_any_phrases(phrases: list[str]) -> bool:
        return any(phrase in q_lower for phrase in phrases)

    # 1. Custodian & Technician
    if has_any_words({"custodian", "custodians", "technician"}) or has_any_phrases(["lab technician", "lab custodians", "laboratory custodian"]):
        return "people"
        
    # 2. Location (Expanded for Taglish, directions, and lost freshmen)
    if has_any_words({"where", "room", "building", "located", "floor", "lab", "laboratory", "office", "directory", "saan", "nasaan", "location", "hanapin"}) or has_any_phrases(["where is", "how to go", "paano pumunta", "where can i find", "anong floor"]):
        return "location"
        
    # 3. Curriculum (Expanded for student slang, grading, and enrollment panic)
    if has_any_words({"curriculum", "course", "subject", "subjects", "semester", "units", "prerequisite", "prereq", "prospectus", "classes", "load", "flowchart", "bridging", "grades", "grade"}) or has_any_phrases(["what subjects", "how many units", "what course", "ilang units", "kailangan ba"]):
        return "curriculum"
        
    # 4. People (Expanded for titles, student slang, and contact info)
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
        
    # 5. Download (Expanded for requesting copies)
    if has_any_words({"download", "link", "pdf", "form", "access", "copy"}) or has_any_phrases(["google form", "download link", "where to get", "can i have a copy", "hingi copy"]):
        return "download"
        
    # 6. Policy (Massive expansion for freshman FAQs: dress code, attendance, shifting, behavior)
    if has_any_words({"policy", "rule", "rules", "guideline", "guidelines", "procedure", "manual", "uniform", "haircut", "absent", "late", "bawal", "allowed", "pwede", "shift", "shifting", "drop", "failing", "fail", "id"}) or has_any_phrases(["dress code", "what happens if", "is it allowed", "pwede ba", "color ng buhok", "hair color", "civilian", "wash day"]):
        return "policy"

    return "general"

def is_listing_query(query: str) -> bool:
    q = (query or "").lower()
    list_triggers = [
        "list", "enumerate", "show", "show all", "all ",
        "who are", "names", "name", "members", "provide"
    ]
    return any(trigger in q for trigger in list_triggers)

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
