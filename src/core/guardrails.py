import re
from langchain_core.prompts import ChatPromptTemplate
from src.config.settings import get_llm
from src.config.constants import CRITIC_CONTEXT_LIMIT
from src.config.logging_config import logger


MAX_QUERY_LENGTH = 500  # characters

def validate_query(query: str) -> tuple[bool, str]:
    """Ensures the query is within length limits and is not empty."""
    if not query or not query.strip():
        return False, "Please enter a question."
    
    if len(query) > MAX_QUERY_LENGTH:
        logger.warning(f"⚠️ Query rejected: Length {len(query)} exceeds limit.")
        return False, f"Your question is too long. Please limit it to {MAX_QUERY_LENGTH} characters."
    
    return True, query.strip()

def redact_pii(text: str) -> str:
    """Masks Philippine student IDs, phone numbers, and emails."""
    # Pattern for AdNU/Philippine Student IDs (e.g., 2021-12345)
    id_pattern = r'\b\d{4}-\d{5}\b'
    # Pattern for Philippine Mobile Numbers (09XX-XXX-XXXX or 09XXXXXXXXX)
    phone_pattern = r'\b(09\d{2}[-\s]?\d{3}[-\s]?\d{4}|09\d{9})\b'
    # Standard Email pattern
    email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    
    redacted = text
    redacted = re.sub(id_pattern, "[STUDENT_ID_REDACTED]", redacted)
    redacted = re.sub(phone_pattern, "[PHONE_REDACTED]", redacted)
    redacted = re.sub(email_pattern, "[EMAIL_REDACTED]", redacted)
    
    if redacted != text:
        logger.info("🛡️ PII Redaction triggered: Sensitive data masked before API call.")
        
    return redacted

def trim_context(context: str, max_chars: int = CRITIC_CONTEXT_LIMIT) -> str:
    """Safely truncates context at the last full newline to avoid cutting mid-sentence."""
    if len(context) <= max_chars:
        return context
    truncated = context[:max_chars]
    last_newline = truncated.rfind('\n')
    return (truncated[:last_newline] if last_newline > 0 else truncated) + "\n\n...[Context truncated for verification]"

def verify_answer(query: str, context: str, draft_answer: str) -> str:
    """Critic Persona: Verifies the draft against the retrieved context."""
    critic_llm = get_llm(temperature=0.0)
    trimmed_context = trim_context(context)

    # Use f-string directly — avoids ChatPromptTemplate variable parsing issues
    full_prompt = f"""You are a Quality Reviewer for AXIsstant, an AI assistant for Ateneo de Naga University's CSEA Department.

Your job is to review a drafted answer and decide if it is faithful to the retrieved context.

CONTEXT TYPES YOU WILL SEE: handbook rules, curriculum tables, thesis abstracts, faculty lists, memos, lab manuals, and OJT documents. All of these are valid sources.

SEMANTIC MATCHING RULE: Treat these as equivalent when verifying:
- "engineering courses" = "Electronics Engineering", "Civil Engineering", "Computer Engineering", etc.
- "thesis" = "manuscript", "capstone", "research project"
- "grading" = "grade computation", "GWA", "final grade"
- "uniform" = "dress code", "attire policy"
A claim is SUPPORTED if the context contains the same information even if worded differently.

STRICT OUTPUT RULES:
- If the draft accurately reflects the context: copy and output the draft exactly as written, with no additions.
- If the draft contains fabricated specific details: rewrite it removing only those fabricated details.
- If the context is completely unrelated to the question: output only this: "I'm sorry, but I cannot find that specific information in the available documents. Please consult the CSEA Department Chair."
- NEVER write words like SUPPORTED, PARTIALLY SUPPORTED, or NOT SUPPORTED in your output.
- NEVER explain your decision. Just output the final answer directly.

CRITICAL RULES:
- Never combine a draft answer AND a sorry message in the same output.
- Thesis abstracts, titles, and author names found in the context ARE valid verifiable facts.
- Paraphrasing of context content is NOT a hallucination. Only flag invented facts.
- If you are unsure whether a claim is supported, pass the draft through verbatim without modification.

Context:
{trimmed_context}

User Query:
{query}

Drafted Answer:
{draft_answer}

Verified Answer:"""

    try:
        response = critic_llm.invoke(full_prompt)
        result = response.content.strip() if response and response.content else None

        if not result:
            logger.warning("Critic returned empty response. Passing draft through.")
            return draft_answer

        return result

    except Exception as e:
        logger.warning(f"Critic Error: {e}. Passing draft through.")
        return draft_answer