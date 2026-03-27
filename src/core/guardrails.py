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

# ─────────────────────────────────────────────────────────────────────────────
# ADNU "CONSTITUTIONAL" GUARDRAILS
# ─────────────────────────────────────────────────────────────────────────────

# Basic profanity/disrespect filter
PROFANITY_TRIGGERS = re.compile(
    r'\b(fuck|shit|damn|bitch|asshole|cunt|motherfucker|putang|gago|bobo|tanga|putangina|diputa|yawa|piste|hayop|pakyu)\b',
    re.IGNORECASE
)

# Catch attempts to cheat or bypass academic effort (Handbook Appendix A: Intellectual Honesty)
DISHONESTY_TRIGGERS = re.compile(
    r'\b(write my essay|do my homework|code this for me|write the code|give me the answers|'
    r'do my assignment|solve this exam|hack|bypass|do my project)\b', 
    re.IGNORECASE
)

# Catch attempts to violate school policies (Handbook Chapters 3, 7, 8)
POLICY_VIOLATION_TRIGGERS = re.compile(
    r'\b(sneak in|smuggle|fake id|fake pass|get away with|cheat the system|'
    r'drink on campus|bring alcohol|vape inside|smoke inside|fight|bully)\b', 
    re.IGNORECASE
)

def validate_query(query: str) -> tuple[bool, str]:
    """Ensures the query is within length limits, is not empty, and aligns with ADNU values."""
    if not query or not query.strip():
        return False, "Please enter a question."
    
    if len(query) > MAX_QUERY_LENGTH:
        logger.warning(f"⚠️ Query rejected: Length {len(query)} exceeds limit.")
        return False, f"Your question is too long. Please limit it to {MAX_QUERY_LENGTH} characters."
        
    q_lower = query.lower()

    # 1. Check for Profanity / Disrespect
    if PROFANITY_TRIGGERS.search(q_lower):
        logger.warning("Guardrail Triggered: Profanity.")
        return False, "As an Atenean AI assistant, I am designed to maintain respectful and professional communication. Please rephrase your question respectfully."

    # 2. Check for Academic Dishonesty (Appendix A)
    if DISHONESTY_TRIGGERS.search(q_lower):
        logger.warning("Guardrail Triggered: Academic Dishonesty.")
        return False, "Under ADNU's **Policy on Intellectual Honesty (Appendix A)**, I cannot write your code, do your assignments, or provide direct answers for graded work. However, as an instrument for your *Competence*, I would be very happy to explain the concepts or guide you on how to solve it yourself!"

    # 3. Check for Policy Violations (Chapter 7/8/9)
    if POLICY_VIOLATION_TRIGGERS.search(q_lower):
        logger.warning("Guardrail Triggered: Policy Violation.")
        return False, "As part of the ADNU community, I uphold the university's Code of Discipline and Safe Campus policies. I cannot help you bypass rules or engage in prohibited activities. Let's focus on how to succeed within the university guidelines!"

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

Your job is to review a drafted answer and decide if it is faithful to the retrieved context and aligns with ADNU's Jesuit values (Competence, Conscience, Compassion, Christ-centeredness).

CONTEXT TYPES YOU WILL SEE: handbook rules, curriculum tables, thesis abstracts, faculty lists, memos, lab manuals, and OJT documents. All of these are valid sources.

SEMANTIC MATCHING RULE: Treat these as equivalent when verifying:
- "engineering courses" = "Electronics Engineering", "Civil Engineering", "Computer Engineering", etc.
- "thesis" = "manuscript", "capstone", "research project"
- "grading" = "grade computation", "GWA", "final grade"
- "uniform" = "dress code", "attire policy"
A claim is SUPPORTED if the context contains the same information even if worded differently.

STRICT OUTPUT RULES:
- If the draft accurately reflects the context and maintains a respectful, Atenean tone: copy and output the draft exactly as written, with no additions.
- NEVER truncate, shorten, or summarize tables. If the draft has a 15-row table, your output must also have all 15 rows.
- If the draft contains fabricated specific details or violates ADNU values: rewrite it removing the fabrication/violation.
- 🔴 SPECULATION KILL SWITCH: If the drafted answer contains words like "assume", "assuming", "implies", "if we assume", or "we interpret this", it means the AI is guessing missing variables. YOU MUST REJECT IT COMPLETELY. Output ONLY this exact sentence: "I don't have enough specific information to answer that accurately. Please check your syllabus or ask your instructor!"
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