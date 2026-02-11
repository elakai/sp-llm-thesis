from src.config.settings import get_llm
from langchain_core.prompts import PromptTemplate

def verify_answer(question: str, context: str, answer: str) -> str:
    """
    Checks if the generated answer is actually supported by the context.
    Returns: The original answer OR a correction message.
    """
    llm = get_llm()
    
    prompt = PromptTemplate.from_template(
        """
        You are a Fact-Checking Judge. 
        Your job is to verify if the 'Answer' is fully supported by the 'Context'.
        
        Question: {question}
        Context: {context}
        Generated Answer: {answer}
        
        Task:
        1. If the answer is supported by the context, output ONLY the word 'pass'.
        2. If the answer contradicts the context or contains information NOT in the context, output 'fail'.
        
        Verdict:
        """
    )
    
    try:
        chain = prompt | llm
        verdict = chain.invoke({
            "question": question,
            "context": context, 
            "answer": answer
        }).content.strip().lower()
        
        if "pass" in verdict:
            return answer
        else:
            # If it failed, we return a safe fallback message
            return "I found some documents, but they didn't contain a specific answer to your question. I don't want to guess and give you wrong information."
            
    except Exception as e:
        print(f"Guardrail Check Failed: {e}")
        return answer # Fallback to original if check fails