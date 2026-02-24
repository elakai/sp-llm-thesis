from langchain_core.prompts import ChatPromptTemplate
from src.config.settings import get_llm
from src.config.constants import CRITIC_CONTEXT_LIMIT
from src.config.logging_config import logger

def trim_context(context: str, max_chars: int = CRITIC_CONTEXT_LIMIT) -> str:
    """Safely truncates context at the last full newline to avoid cutting mid-sentence."""
    if len(context) <= max_chars:
        return context
    truncated = context[:max_chars]
    last_newline = truncated.rfind('\n')
    return (truncated[:last_newline] if last_newline > 0 else truncated) + "\n\n...[Context truncated for verification]"

def verify_answer(query: str, context: str, draft_answer: str) -> str:
    """The Critic Persona. Reviews the drafted answer against retrieved context."""
    critic_llm = get_llm(temperature=0.0) # Ensure critic is strictly factual
    trimmed_context = trim_context(context)
    
    system_prompt = """You are a strict Fact-Checker for the Ateneo de Naga CSEA Academic Assistant.
    Your ONLY job: Check if the drafted answer contains claims, numbers, or course codes NOT explicitly present in the context.

    Rules:
    - If every claim in the draft is supported by the context, output the draft EXACTLY as-is.
    - If any claim is NOT in the context (hallucination), rewrite to remove ONLY the unsupported claim.
    - If the context has NO relevant info, reply: "I'm sorry, but I cannot find the specific information in the current handbook documents. Please consult the department chair."

    DO NOT change formatting, tone, or style unless fixing a hallucination.

    Context:
    {context}
    
    User Query: {query}
    
    Drafted Answer:
    {draft_answer}
    
    Verified Answer:"""
    
    prompt = ChatPromptTemplate.from_template(system_prompt)
    
    try:
        response = critic_llm.invoke(prompt.format(
            context=trimmed_context, 
            query=query, 
            draft_answer=draft_answer
        ))
        return response.content.strip()
    except Exception as e:
        logger.warning(f"Critic Error triggered safe fallback: {e}")
        return "I found some relevant information, but I could not verify the answer with enough confidence. Please consult the department directly."