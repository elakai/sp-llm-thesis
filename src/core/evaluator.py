from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision
from datasets import Dataset
import pandas as pd

def evaluate_response(query: str, response: str, contexts: list):
    """
    Evaluates a single response using Ragas metrics.
    contexts: List of strings (the page_content of retrieved docs)
    """
    # 1. Prepare data in the format Ragas expects
    data = {
        "question": [query],
        "answer": [response],
        "contexts": [contexts],
    }
    
    dataset = Dataset.from_dict(data)
    
    # 2. Run Evaluation
    # Note: Ragas uses your LLM (OpenAI/Groq) to "judge" the answer
    result = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_precision]
    )
    
    return result.to_pandas()