import os
import pandas as pd
from datasets import Dataset
import nest_asyncio

# Apply Async Fix
nest_asyncio.apply()

from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_correctness,
    context_recall,
    context_precision
)
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper

from src.config.settings import get_llm, get_embeddings, get_vectorstore

def run_evaluation():
    print("🚀 Starting Ragas Evaluation Pipeline...")
    
    # 1. Load the synthetic dataset we just generated
    csv_path = "csea_evaluation_dataset.csv"
    if not os.path.exists(csv_path):
        print(f"❌ Could not find {csv_path}")
        return
        
    test_df = pd.read_csv(csv_path)
    questions = test_df["question"].tolist()
    ground_truths = test_df["ground_truth"].tolist()
    
    # 2. Setup your system components (The Brain & The Reader)
    llm = get_llm()
    embeddings = get_embeddings()
    vectorstore = get_vectorstore()
    
    # We set k=5 to see if the right answer is in the top 5 chunks
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5}) 
    
    # Wrap models for Ragas v0.2+
    eval_llm = LangchainLLMWrapper(llm)
    eval_embeddings = LangchainEmbeddingsWrapper(embeddings)
    
    # 3. Simulate the User Chatting with the Bot
    print(f"🤖 Asking your bot {len(questions)} test questions...")
    answers = []
    contexts = []
    
    for q in questions:
        # A. Fetch the documents from Pinecone
        docs = retriever.invoke(q)
        retrieved_contexts = [doc.page_content for doc in docs]
        contexts.append(retrieved_contexts)
        
        # B. Generate the bot's answer
        prompt = f"Use this context to answer the question:\n{retrieved_contexts}\n\nQuestion: {q}\nAnswer:"
        response = llm.invoke(prompt)
        answers.append(response.content)

    # 4. Format everything for the Ragas Judge
    data = {
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": ground_truths
    }
    eval_dataset = Dataset.from_dict(data)
    
    # 5. The Final Judgment
    print("⚖️ Grading the responses... (Traces are being sent to LangSmith)")
    result = evaluate(
        dataset=eval_dataset,
        metrics=[
            context_precision, # Did it retrieve the right chunks?
            context_recall,    # Did it retrieve ALL the needed info?
            faithfulness,      # Did the LLM hallucinate?
            answer_correctness,  # Did the LLM actually answer the question?
        ],
        llm=eval_llm,
        embeddings=eval_embeddings,
    )
    
    print("\n✅ Evaluation Complete!")
    print("========================")
    print(result)
    
    # Save the detailed grading report
    result_df = result.to_pandas()
    result_df.to_csv("final_evaluation_report.csv", index=False)
    print("📁 Detailed report saved to final_evaluation_report.csv")

if __name__ == "__main__":
    run_evaluation()