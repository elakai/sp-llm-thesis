# src/core/evaluator.py
from ragas import evaluate, RunConfig
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall
)
from datasets import Dataset
from src.config.settings import get_llm
from langchain_huggingface import HuggingFaceEmbeddings # <--- REQUIRED
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper

# 1. Prepare Judge & Embeddings
# We need to wrap them so Ragas understands them
judge_llm = get_llm()
judge_wrapper = LangchainLLMWrapper(judge_llm)

# 2. Define Embeddings (CRITICAL FIX)
# Without this, Ragas looks for OpenAI and crashes.
hf_embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
embeddings_wrapper = LangchainEmbeddingsWrapper(hf_embeddings)

def evaluate_rag_response(
    query: str,
    generated_answer: str,
    retrieved_contexts: list[str],
    reference_answer: str = None
):
    """
    Evaluates a single RAG response using Ragas with Groq-safe settings.
    """
    
    # 3. Dynamic Metrics List
    # Only run Context metrics if we actually have a ground truth
    active_metrics = [faithfulness, answer_relevancy]
    
    data = {
        "question": [query],
        "answer": [generated_answer],
        "contexts": [retrieved_contexts], # List of lists
    }

    if reference_answer:
        data["ground_truth"] = [reference_answer]
        # Only add these if we have ground truth, otherwise they crash
        active_metrics.extend([context_precision, context_recall])

    dataset = Dataset.from_dict(data)

    # 4. Inject Dependencies into Metrics
    # This ensures Ragas uses GROQ, not OpenAI, for grading
    for m in active_metrics:
        m.llm = judge_wrapper
        # AnswerRelevancy needs embeddings to calculate cosine similarity
        if hasattr(m, 'embeddings'):
            m.embeddings = embeddings_wrapper

    try:
        # 5. Run with Safety Limits (RunConfig)
        result = evaluate(
            dataset=dataset,
            metrics=active_metrics,
            llm=judge_wrapper,       
            embeddings=embeddings_wrapper, # <--- Pass embeddings here too
            # ⚠️ CRITICAL FOR GROQ FREE TIER:
            # Force 1 worker to prevent 429 Rate Limits
            run_config=RunConfig(max_workers=1, timeout=120) 
        )
        return result.to_pandas()

    except Exception as e:
        print(f"Ragas evaluation failed: {e}")
        return None