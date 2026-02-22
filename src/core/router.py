from src.config.settings import get_llm
from langchain_core.prompts import PromptTemplate

def extract_metadata_filters(query: str) -> list:
    """
    Detects specific programs mentioned in the query.
    Returns a list of exact filenames to filter the Pinecone vector database.
    """
    query_lower = query.lower()
    filters = []
    
    # Check for Electronics Engineering
    if "ece" in query_lower or "electronics" in query_lower:
        filters.append("BS ECE SY2024-2025 Curriculum.pdf")
        
    # Check for Civil Engineering (pad with spaces to avoid matching 'piece' or 'space')
    if " ce " in f" {query_lower} " or "civil" in query_lower:
        filters.append("BS CE SY2024-2025 Curriculum.pdf")
        
    # Check for Architecture
    if "arch" in query_lower or "architecture" in query_lower:
        filters.append("BS Architecture SY2022-2023 Curriculum.pdf") 
        
    return filters if filters else None

def route_query(query: str) -> tuple:
    """
    Returns (intent, filters).
    Intent: 'greeting', 'search', 'off_topic'
    Filters: List of filenames or None
    """
    # 1. Fast Keyword Check
    greetings = ["hi", "hello", "hey", "good morning", "good afternoon", "thanks", "thank you"]
    if query.lower().strip() in greetings:
        return "greeting", None

    # 2. Extract specific program filters
    filters = extract_metadata_filters(query)

    # 3. LLM Classification (The "Brain")
    llm = get_llm()
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
        
        if "search" in category: return "search", filters
        if "greeting" in category: return "greeting", filters
        if "off" in category: return "off_topic", filters
        
        return "search", filters 
        
    except Exception as e:
        print(f"Router Error: {e}")
        return "search", filters