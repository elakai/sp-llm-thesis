from typing import List
from langchain_core.prompts import PromptTemplate
from src.config.settings import get_llm

def decompose_query(query: str) -> List[str]:
    """
    Breaks a complex query into simple sub-queries.
    If the query is simple, it returns a list containing just the original query.
    """
    llm = get_llm()
    
    prompt = PromptTemplate.from_template(
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
    
    try:
        # invoke the chain
        response = (prompt | llm).invoke({"query": query}).content
        
        # Split by newlines and clean up
        sub_queries = [line.strip() for line in response.split('\n') if line.strip()]
        
        # Fallback: If LLM fails or returns nothing, just use original
        if not sub_queries:
            return [query]
            
        return sub_queries
        
    except Exception as e:
        print(f"Decomposition Error: {e}")
        return [query]