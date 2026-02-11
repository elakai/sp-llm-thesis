import pandas as pd
from datasets import Dataset
from ragas import evaluate
# Updated imports to avoid DeprecationWarnings
from ragas.metrics import faithfulness, answer_relevancy, context_precision
from src.core.auth import supabase

def run_ragas_evaluation():
    # 1. Fetch logs that HAVE context (using the new column)
    response = supabase.table("chat_logs") \
        .select("query, response, context") \
        .not_.is_("context", "null") \
        .limit(10) \
        .execute()
    
    logs = response.data

    if not logs:
        print("⚠️ No logs with context found. Make sure your bot is saving the 'context' to Supabase!")
        return

    # 2. Reformat for Ragas
    data = {
        "question": [item["query"] for item in logs],
        "answer": [item["response"] for item in logs],
        "contexts": [[item["context"]] for item in logs], 
    }

    dataset = Dataset.from_dict(data)

    # 3. Execute Evaluation
    result = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_precision]
    )

    # 4. Save and Print Results
    df = result.to_pandas()
    df.to_csv("evaluation_results.csv", index=False)
    print("✅ Evaluation Complete!")
    print(result)
    return result

if __name__ == "__main__":
    run_ragas_evaluation()