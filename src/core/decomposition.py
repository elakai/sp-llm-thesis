from typing import List
from langchain_core.prompts import PromptTemplate
from src.config.settings import get_generator_llm
from src.config.logging_config import logger

_DECOMPOSE_PROMPT = PromptTemplate.from_template(
    """You are an expert Query Decomposer.
    Your task is to break down a complex student question into simple, single-topic search queries.
    
    Rules:
    1. If the query asks for a comparison (e.g. "Difference between X and Y"), split it into "What is X?" and "What is Y?".
    2. If the query is already simple (e.g. "What is the grading system?"), return it exactly as is.
    3. Return the queries separated by a newline character.
    4. Do not add numbering or bullet points.
    
    User Query: {query}
    
    Sub-queries:"""
)

def decompose_query(query: str) -> List[str]:
    # ── FIX: Strict Keyword Guard to throttle unnecessary LLM calls ──
    complex_triggers = {"and", "vs", "versus", "compare", "difference", "both", "also", "between"}
    q_tokens = set(query.lower().split())
    
    if not any(trigger in q_tokens for trigger in complex_triggers):
        return [query]
        
    if len(query.split()) < 6:
        return [query]
        
    llm = get_generator_llm()
    
    try:
        response = (_DECOMPOSE_PROMPT | llm).invoke({"query": query}).content
        
        sub_queries = [
            line.strip() for line in response.split('\n')
            if line.strip() and not line.strip().endswith(':') and len(line.strip()) > 10
        ]
        
        if not sub_queries: return [query]
        return sub_queries[:3]
        
    except Exception as e:
        logger.warning(f"Decomposition error, falling back to original query: {e}")
        return [query]