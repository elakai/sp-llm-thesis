from typing import List
from langchain_core.prompts import PromptTemplate
from src.config.settings import get_generator_llm
from src.config.logging_config import logger

# Pre-built at import time — avoids reconstructing the template on every call
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
    """
    Breaks a complex query into simple sub-queries.
    If the query is simple, it returns a list containing just the original query.
    """
    
    # Short-query guard: don't waste LLM calls on inherently simple strings
    if len(query.split()) < 8:
        return [query]
        
    llm = get_generator_llm()
    
    try:
        # invoke the chain
        response = (_DECOMPOSE_PROMPT | llm).invoke({"query": query}).content
        
        # Split by newlines, clean up, and filter out LLM preambles (like "Here are the queries:")
        sub_queries = [
            line.strip() for line in response.split('\n')
            if line.strip() and not line.strip().endswith(':') and len(line.strip()) > 10
        ]
        
        # Fallback: If LLM fails or returns nothing, just use original
        if not sub_queries:
            return [query]
            
        # Hard cap to prevent rate limiting / cascading DB calls
        return sub_queries[:3]
        
    except Exception as e:
        logger.warning(f"Decomposition error, falling back to original query: {e}")
        return [query]