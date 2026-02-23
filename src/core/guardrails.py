from langchain_core.prompts import ChatPromptTemplate
from src.config.settings import get_llm

def verify_answer(query: str, context: str, draft_answer: str) -> str:
    """
    The Critic Persona. Reviews the drafted answer against the retrieved context.
    Prevents hallucinations before they reach the user.
    """
    critic_llm = get_llm()
    
    system_prompt = """You are the strict Quality Assurance Reviewer for the Ateneo de Naga CSEA Academic Assistant.
    Your job is to read a user's query, the retrieved handbook context, and a drafted AI response.
    
    Critique the drafted response based on these STRICT rules:
    1. FAITHFULNESS: Does the draft contain ANY information, numbers, or course codes NOT explicitly present in the context? (Hallucination check)
    2. RELEVANCE: Does it actually answer the user's question?
    3. FORMATTING: Are tables formatted correctly in Markdown? Are lists clean?
    
    ACTIONS:
    - If the draft is 100% accurate and strictly based on the context, output the draft EXACTLY as is. Do not add any introductory text.
    - If the draft includes hallucinations, guesses, or information not found in the context, REWRITE it to be strictly factual based ONLY on the context.
    - If the context does not contain the answer at all, REWRITE the response to simply say: "I'm sorry, but I cannot find the specific information in the current handbook documents. Please consult the department chair."
    
    Context:
    {context}
    
    User Query: {query}
    
    Drafted Answer:
    {draft_answer}
    
    Final Approved Answer:"""
    
    prompt = ChatPromptTemplate.from_template(system_prompt)
    
    try:
        # The critic analyzes and returns the final cleaned string
        response = critic_llm.invoke(prompt.format(
            context=context, 
            query=query, 
            draft_answer=draft_answer
        ))
        return response.content.strip()
    except Exception as e:
        print(f"⚠️ Critic Error: {e}")
        return draft_answer # Fallback to draft if the critic crashes