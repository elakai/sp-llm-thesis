import os
import sys
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv

# 1. SETUP PATHS
project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
load_dotenv(project_root / ".env")

from datasets import Dataset
from ragas import evaluate, RunConfig
from ragas.metrics.collections import Faithfulness, AnswerRelevancy 
from ragas.llms import llm_factory
from ragas.embeddings import embedding_factory
from groq import Groq # Native Groq client
from src.core.auth import supabase

def run_ragas_evaluation():
    print("🔄 Fetching logs from Supabase...")
    response = supabase.table("chat_logs").select("query, response, context")\
        .order("created_at", desc=True).limit(10).execute()
    
    logs = [r for r in response.data if r.get("context") and "[[Source:" in str(r["context"])]
    if not logs: 
        print("❌ No valid logs found."); return

    # 🚀 STEP 2: INITIALIZE MODERN CLIENTS
    # Groq Client
    groq_api_key = os.getenv("GROQ_API_KEY")
    groq_client = Groq(api_key=groq_api_key)

    # 🚀 STEP 3: USE MODERN FACTORIES (THE FIX)
    # This creates the 'InstructorLLM' that Ragas collections actually require.
    evaluator_llm = llm_factory(
        model="llama-3.3-70b-versatile", 
        provider="groq", 
        client=groq_client
    )
    
    # Standardizing Embeddings with the modern factory
    evaluator_embeddings = embedding_factory(
        provider="huggingface", 
        model="sentence-transformers/all-mpnet-base-v2"
    )

    # 🚀 STEP 4: INSTANTIATE METRICS
    # Faithfulness and AnswerRelevancy now accept the factory-made LLM
    metrics = [
        Faithfulness(llm=evaluator_llm),
        AnswerRelevancy(llm=evaluator_llm, embeddings=evaluator_embeddings)
    ]

    # 🚀 STEP 5: EXECUTION
    print(f"⚖️  Grading {len(logs)} interactions with Llama 3.3 (Instructor-ready)...")
    try:
        results = evaluate(
            Dataset.from_dict({
                "question": [x["query"] for x in logs],
                "answer": [x["response"] for x in logs],
                "contexts": [[x["context"]] for x in logs], 
            }),
            metrics=metrics,
            # max_workers=1 is mandatory for Groq's Free Tier RPM
            run_config=RunConfig(max_workers=1, timeout=180) 
        )
        
        results.to_pandas().to_csv("thesis_evaluation_results.csv", index=False)
        print("\n✅ Success! Results saved to 'thesis_evaluation_results.csv'")
        print(results)
        
    except Exception as e:
        print(f"\n❌ Evaluation Failed: {e}")

if __name__ == "__main__":
    run_ragas_evaluation()