import os
import sys
import pandas as pd
import time
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

# SETUP PATHS
project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
load_dotenv(project_root / ".env")

from datasets import Dataset
from ragas import evaluate, RunConfig
from ragas.metrics.collections import Faithfulness, AnswerRelevancy 
from ragas.llms import llm_factory
from ragas.embeddings import embedding_factory
from groq import Groq
from src.core.auth import supabase
from src.config.logging_config import logger

def run_ragas_evaluation():
    print("🔄 Fetching latest 10 rated interactions from Supabase...")
    # Fetch only "helpful" or recently logged queries for a focused evaluation
    response = supabase.table("chat_logs").select("query, response, context")\
        .order("created_at", desc=True).limit(10).execute()
    
    # Filter for valid data
    logs = [r for r in response.data if r.get("context") and "[[Source:" in str(r["context"])]
    if not logs: 
        print("❌ No valid logs found for evaluation."); return

    # INITIALIZE MODERN CLIENTS
    groq_api_key = os.getenv("GROQ_API_KEY")
    groq_client = Groq(api_key=groq_api_key)

    evaluator_llm = llm_factory(
        model="llama-3.3-70b-versatile", 
        provider="groq", 
        client=groq_client
    )
    
    evaluator_embeddings = embedding_factory(
        provider="huggingface", 
        model="sentence-transformers/all-mpnet-base-v2"
    )

    metrics = [
        Faithfulness(llm=evaluator_llm),
        AnswerRelevancy(llm=evaluator_llm, embeddings=evaluator_embeddings)
    ]

    # PREPARE DATASET
    # Logic: Split the giant context string into a list of chunks based on source markers
    questions = [x["query"] for x in logs]
    answers = [x["response"] for x in logs]
    contexts = [x["context"].split("[[Source:") for x in logs] # Split into real list

    print(f"⚖️ Grading {len(logs)} interactions with Llama 3.3...")
    
    try:
        results = evaluate(
            Dataset.from_dict({
                "question": questions,
                "answer": answers,
                "contexts": contexts, 
            }),
            metrics=metrics,
            run_config=RunConfig(max_workers=1, timeout=180) 
        )
        
        # SAVE TIMESTAMPED REPORT
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"thesis_eval_{timestamp}.csv"
        results.to_pandas().to_csv(filename, index=False)
        
        # LOG SUMMARY TO SUPABASE (For Admin Dashboard)
        summary = results.scores_as_dict()
        try:
            supabase.table("evaluation_runs").insert({
                "run_at": datetime.now().isoformat(),
                "faithfulness": summary.get("faithfulness"),
                "answer_relevancy": summary.get("answer_relevancy"),
                "sample_size": len(logs)
            }).execute()
            print("📊 Summary metrics logged to Supabase.")
        except Exception as se:
            print(f"⚠️ Could not log summary to Supabase: {se}")

        print(f"\n✅ Success! Detailed results saved to {filename}")
        print(results)
        
    except Exception as e:
        print(f"\n❌ Evaluation Failed: {e}")

if __name__ == "__main__":
    run_ragas_evaluation()