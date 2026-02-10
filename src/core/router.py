from src.config.settings import get_llm
from langchain_core.prompts import PromptTemplate

def route_query(query: str) -> str:
    """
    Determines the 'intent' of the user's query.
    Returns one of: 'greeting', 'search', 'off_topic'
    """
    
    # 1. Fast Keyword Check (Optimization for speed)
    greetings = ["hi", "hello", "hey", "good morning", "good afternoon", "thanks", "thank you"]
    if query.lower().strip() in greetings:
        return "greeting"

    # 2. LLM Classification (The "Brain")
    llm = get_llm()
    
    # We ask the LLM to act as a classifier.
    # It must strictly return ONE word.
    prompt = PromptTemplate.from_template(
        """
        You are a Router for a University Student Handbook Chatbot. 
        Classify the following user query into exactly one of these categories:
        
        1. 'search': The user is asking about university rules, curriculum, grading, uniforms, teachers, or events.
        2. 'greeting': The user is saying hello, thanks, or goodbye.
        3. 'off_topic': The user is asking about cooking, video games, general world knowledge, or coding help unrelated to the handbook.

        Query: {query}
        
        Classification (return ONLY the word):
        """
    )
    
    try:
        chain = prompt | llm
        category = chain.invoke({"query": query}).content.strip().lower()
        
        # Fallback if LLM is chatty
        if "search" in category: return "search"
        if "greeting" in category: return "greeting"
        if "off" in category: return "off_topic"
        
        return "search" # Default to search if unsure
        
    except Exception as e:
        print(f"Router Error: {e}")
        return "search" # Fail safe: just search