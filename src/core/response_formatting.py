import re
from typing import List
from src.config.constants import LOW_CONFIDENCE_THRESHOLD, HIGH_CONFIDENCE_THRESHOLD, HIGH_CONFIDENCE_MARGIN

_SUGGESTION_STOPWORDS = {
    "what", "when", "where", "which", "who", "whom", "whose", "why", "how",
    "is", "are", "was", "were", "be", "being", "been", "do", "does", "did",
    "to", "for", "of", "in", "on", "at", "by", "from", "with", "about", "and",
    "or", "the", "a", "an", "this", "that", "these", "those", "can", "could",
    "should", "would", "will", "my", "your", "their", "our", "i", "you", "we",
    "they", "it", "me", "us", "them"
}

def _contains_markdown_table(text: str) -> bool:
    return any('|' in line and line.strip().startswith('|') for line in text.strip().split('\n'))

def _contains_speculation(text: str) -> bool:
    return bool(re.search(r"\b(likely|possibly|probably|appears to be|seems to be)\b", text, re.IGNORECASE))

def remove_speculative_sentences(text: str) -> str:
    if not text or not text.strip(): return text
    kept = [s.strip() for s in re.split(r'(?<=[.!?])\s+', text.strip()) if s.strip() and not _contains_speculation(s)]
    return " ".join(kept).strip()

def build_source_certainty_note(top_score: float, score_margin: float, sources: list[str]) -> str:
    unique_sources = list({(s or "Unknown").strip() for s in sources})

    if top_score >= HIGH_CONFIDENCE_THRESHOLD and score_margin >= HIGH_CONFIDENCE_MARGIN: level = "High"
    elif top_score >= LOW_CONFIDENCE_THRESHOLD: level = "Medium"
    else: level = "Low"

    def _clean_source_name(s: str) -> str:
        s = re.sub(r'[-_]', ' ', s)           
        return re.sub(r'\.(pdf|md|txt|docx|csv|xlsx)$', '', s, flags=re.IGNORECASE).strip().title()

    clean_names = [_clean_source_name(s) for s in unique_sources[:2]]
    preview = ", ".join(clean_names) if clean_names else "retrieved documents"
    return f"> **Source certainty:** {level} — based on {len(unique_sources)} document(s): *{preview}*"

def fallback_questions(source_text: str, original_query: str, max_items: int = 3) -> List[str]:
    source_lower = source_text.lower()
    fallbacks = []
    seen = set()

    if "prereq" in source_lower or "pre-requisite" in source_lower or "prerequisite" in source_lower:
        fallbacks.append("What are the exact prerequisites mentioned here?")
    if re.findall(r"\b[A-Z]{2,5}\d{3}\b", source_text.upper()):
        fallbacks.append("Can you list all course codes mentioned in this answer?")
    if "semester" in source_lower or "curriculum" in source_lower or "course" in source_lower:
        fallbacks.append("Can you summarize this by semester or category?")

    fallbacks.extend([
        "Can you summarize that in 3 short bullet points?",
        "Which part of the available documents supports this answer?",
        "What is the key takeaway I should remember from this?",
    ])

    deduped = []
    for q in fallbacks:
        norm = q.lower().strip()
        if norm not in seen and norm != original_query.lower().strip():
            deduped.append(q)
            seen.add(norm)
        if len(deduped) == max_items: break
    return deduped

def is_no_answer_response(text: str) -> bool:
    if not text: return True
    lowered = text.lower()
    patterns = [
        "i couldn't find that",
        "i couldn't find a confident answer",
        "i couldn't find an explicit answer",
        "best to check with your department chair",
        "not explicitly stated"
    ]
    return any(p in lowered for p in patterns)

def build_no_answer_response(query: str = "") -> str:
    return (
        "I couldn't find a confident answer for that in the available documents.\n\n"
        "**Here are a few tips to help me find the right information:**\n"
        "- **Spell out acronyms** — instead of *'OJT in CPE'*, try *'on-the-job training in Computer Engineering'*.\n"
        "- **Be specific about the program** — mention *'BS Computer Engineering'* or *'BS ECE'* rather than just *'engineering'*.\n"
        "- **Include the year or semester** — e.g., *'third year first semester BS CPE'*.\n"
        "- **Use full course titles** — instead of *'OS'*, try *'Operating Systems'*.\n"
        "- **Check for typos** — a small typo in a course code (like *QCP512* instead of *QCPP512*) can hide the result.\n\n"
        "If you've tried rephrasing and still can't find it, your department chair is the best person to ask directly!"
    )

def is_no_answer_response(text: str) -> bool:
    if not text:
        return False
    patterns = [
        r"i\s*couldn['’]?t\s*find\s*that\s*in\s*the\s*available\s*documents",
        r"i\s*do\s*not\s*have\s*enough\s*info\s*to\s*answer\s*that\s*confidently",
        r"i\s*don't\s*have\s*enough\s*info\s*to\s*answer\s*that\s*confidently",
        r"not explicitly stated in the retrieved documents",
        r"best\s*to\s*check\s*with\s*your\s*(respective\s*)?department\s*chair",
        r"your\s*best\s*bet\s*is\s*to\s*check\s*with\s*your\s*(respective\s*)?department\s*chair",
        r"department\s*chair\s*directly",
        r"i couldn't find an explicit answer for that detail",
    ]
    lowered = text.lower()
    return any(re.search(pattern, lowered) for pattern in patterns)

def _strip_decorative_dash_rows(t: str) -> str:
    cleaned = []
    for line in t.split('\n'):
        stripped = line.strip()
        # Ensure it's actually a table row before processing
        if stripped.count('|') >= 2:
            inner = stripped[stripped.find('|')+1 : stripped.rfind('|')]
            cells = [c.strip() for c in inner.split('|')]
            
            # Check if EVERY cell is just hyphens, en-dashes, em-dashes, or spaces
            is_decorative = all(re.sub(r'[-—–\s]', '', cell) == '' for cell in cells)
            
            # If the row is purely decorative dashes, skip it
            if is_decorative:
                continue
                
        cleaned.append(line)
    return '\n'.join(cleaned)

def fix_markdown_tables(text: str) -> str:
    if '|' not in text: return text
    text = _strip_decorative_dash_rows(text)
        
    lines = text.split('\n')
    fixed = []
    in_table, has_separator = False, False

    for i, line in enumerate(lines):
        is_row = line.strip().startswith('|')
        if is_row:
            if not in_table:
                if fixed and fixed[-1].strip(): fixed.append('')
                in_table, has_separator = True, False
            fixed.append(line)
            if '---' in line: has_separator = True
                
            next_is_row = (i + 1 < len(lines) and lines[i+1].strip().startswith('|'))
            if not has_separator and not (next_is_row and '---' in lines[i+1]):
                col_count = line.count('|') - 1
                if col_count > 0:
                    fixed.append('|' + '---|' * col_count)
                    has_separator = True
        else:
            if in_table:
                if line.strip(): fixed.append('')
                in_table = False
            fixed.append(line)
    return '\n'.join(fixed)

def format_raw_links(text: str) -> str:
    # ── FIXED: Removed the single quote exclusion so it catches Borrower's Form ──
    raw_url_pattern = re.compile(r'(?<!\()(https?://[^\s\)\]\"]+)')
    def replace_url(match):
        url = match.group(1)
        
        # Prevent trailing punctuation from sneaking into the URL
        while url and url[-1] in ['.', ',', ';', ':']:
            url = url[:-1]
            
        pos = match.start()
        if re.search(r'\[[^\]]*\]\($', text[max(0, pos-50):pos]): return url 
            
        if 'supabase' in url or 'storage' in url: label = 'Download here'
        elif 'form' in url or 'docs.google' in url: label = 'Access the form here'
        elif 'drive.google' in url: label = 'View document here'
        else: label = 'View link here'
        return f'[{label}]({url})'
        
    return raw_url_pattern.sub(replace_url, text)