import re
from langchain_core.prompts import ChatPromptTemplate
from src.config.settings import get_critic_llm
from src.config.constants import CRITIC_CONTEXT_LIMIT
from src.config.logging_config import logger

MAX_QUERY_LENGTH = 500  # characters

# Pre-compiled PII patterns — avoids regex recompilation on every call
_ID_PATTERN = re.compile(r'\b\d{4}-\d{5}\b')
_PHONE_PATTERN = re.compile(r'\b(09\d{2}[-\s]?\d{3}[-\s]?\d{4}|09\d{9})\b')
_EMAIL_PATTERN = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b')

# Abuse patterns
_ABUSE_PATTERNS = re.compile(
    r'\b(fuck|shit|damn|bitch|asshole|putang|gago|bobo|tanga)\b',
    re.IGNORECASE
)

def validate_query(query: str) -> tuple[bool, str]:
    """Ensures the query is within length limits, is not empty, and has no profanity."""
    if not query or not query.strip():
        return False, "Please enter a question."
    
    if len(query) > MAX_QUERY_LENGTH:
        logger.warning(f"⚠️ Query rejected: Length {len(query)} exceeds limit.")
        return False, f"Your question is too long. Please limit it to {MAX_QUERY_LENGTH} characters."
        
    if _ABUSE_PATTERNS.search(query):
        logger.warning(f"⚠️ Query rejected: Abuse/Profanity detected.")
        return False, "Let's keep it respectful! Try asking something about your curriculum or school policies."
    
    return True, query.strip()

def redact_pii(text: str) -> str:
    """Masks Philippine student IDs, phone numbers, and emails."""
    redacted = text
    redacted = _ID_PATTERN.sub("[STUDENT_ID_REDACTED]", redacted)
    redacted = _PHONE_PATTERN.sub("[PHONE_REDACTED]", redacted)
    redacted = _EMAIL_PATTERN.sub("[EMAIL_REDACTED]", redacted)
    
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

def _count_table_rows(text: str) -> int:
    """Utility to check if the critic maliciously shortened a markdown table."""
    return sum(
        1 for line in text.split('\n')
        if line.strip().startswith('|') and '---' not in line
    )

def verify_answer(query: str, context: str, draft_answer: str) -> str:
    critic_llm = get_critic_llm()
    trimmed_context = trim_context(context)

    system_prompt = """You are a Quality Reviewer for AXIsstant, an AI assistant for Ateneo de Naga University's CSEA Department.

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
- NEVER truncate, shorten, or summarize tables. If the draft has a 15-row table, your output must also have all 15 rows.
- If the draft contains fabricated specific details: rewrite it removing only those fabricated details.
- 🔴 SPECULATION KILL SWITCH: If the drafted answer contains words like "assume", "assuming", "implies", "if we assume", or "we interpret this", it means the AI is guessing missing variables. YOU MUST REJECT IT COMPLETELY. Output ONLY this exact sentence: "I don't have enough specific information (like your exact unit load or class hours) to calculate that accurately. Please check your syllabus or ask your instructor!"
- If the context is completely unrelated to the question: output only this: "I couldn't find that in the available documents — your best bet is to check with your respective department chair directly!"
- NEVER write words like SUPPORTED, PARTIALLY SUPPORTED, or NOT SUPPORTED in your output.
- NEVER explain your decision. Just output the final answer directly.

CRITICAL RULES:
- WHEN IN DOUBT, PASS THE DRAFT THROUGH. Your default action should be to approve, not reject (unless the Kill Switch is triggered).
- Names, titles, positions, and credentials mentioned in the context ARE facts — do NOT reject answers that cite them.
- Never combine a draft answer AND a sorry message in the same output.

Context:
{trimmed_context}"""

    human_prompt = "User Query:\n{query}\n\nDrafted Answer:\n{draft_answer}\n\nVerified Answer:"

    prompt_template = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", human_prompt)
    ])

    try:
        messages = prompt_template.format_messages(
            trimmed_context=trimmed_context,
            query=query,
            draft_answer=draft_answer
        )
        response = critic_llm.invoke(messages)
        result = response.content.strip() if response and response.content else None

        if not result:
            logger.warning("Critic returned empty response. Passing draft through.")
            return draft_answer
            
        draft_rows = _count_table_rows(draft_answer)
        result_rows = _count_table_rows(result)
        if draft_rows > 0 and result_rows < draft_rows * 0.8:
            logger.warning(f"Critic truncated table ({draft_rows} → {result_rows} rows). Passing draft through.")
            return draft_answer

        return result

    except Exception as e:
        logger.warning(f"Critic Error: {e}. Passing draft through.")
        return draft_answer