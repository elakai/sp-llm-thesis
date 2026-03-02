"""
RAGAS Evaluation Pipeline for the CSEA Assistant

Mirrors the actual RAG pipeline: Dynamic K -> Hybrid Rerank -> Cross-Encoder -> LLM
Supports two evaluation modes:
  - Dataset mode: CSV with ground truth (5 metrics)
  - Live mode: Recent Supabase chat logs (2 metrics, no ground truth)

Usage:
  python -m src.core.evaluate_rag                          # dataset mode (default)
  python -m src.core.evaluate_rag --mode live --limit 15   # live mode
  python -m src.core.evaluate_rag --mode both              # run both
"""

import os
import sys
import time
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import List, Tuple

# ── Path & env setup ──
project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from dotenv import load_dotenv
load_dotenv(project_root / ".env")

import nest_asyncio
nest_asyncio.apply()

from datasets import Dataset
from ragas import evaluate, RunConfig
from ragas.metrics import (
    faithfulness,
    answer_correctness,
    context_recall,
    context_precision,
)
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper

from langchain_core.documents import Document
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore

from src.config.constants import RETRIEVAL_K, RERANKER_TOP_K
from src.config.logging_config import logger
from src.core.router import get_dynamic_k
from src.core.feedback import supabase

# Try to import answer_relevancy (available in ragas >=0.2)
try:
    from ragas.metrics import answer_relevancy
    HAS_ANSWER_RELEVANCY = True
except ImportError:
    HAS_ANSWER_RELEVANCY = False


# ─────────────────────────────────────────────────────────────────────────────
# STANDALONE COMPONENTS (no Streamlit dependency)
# ─────────────────────────────────────────────────────────────────────────────
def _init_components():
    """Initialize LLM, embeddings, vectorstore, and reranker without Streamlit."""
    llm = ChatGroq(
        model_name="llama-3.3-70b-versatile",
        temperature=0.1,
        api_key=os.getenv("GROQ_API_KEY"),
    )
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    vectorstore = PineconeVectorStore(
        index=pc.Index(os.getenv("PINECONE_INDEX_NAME")),
        embedding=embeddings,
        text_key="text",
    )
    reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    return llm, embeddings, vectorstore, reranker


# ─────────────────────────────────────────────────────────────────────────────
# PIPELINE HELPERS (mirrors retrieval.py without st.cache_resource)
# ─────────────────────────────────────────────────────────────────────────────
def _hybrid_rerank(query: str, docs: List[Document]) -> List[Document]:
    """BM25 + position scoring (same logic as retrieval.py)."""
    if not docs:
        return []
    tokenized_docs = [doc.page_content.split() for doc in docs]
    bm25 = BM25Okapi(tokenized_docs)
    bm25_scores = bm25.get_scores(query.lower().split())

    ranked = []
    for i, doc in enumerate(docs):
        position_score = (len(docs) - i) * 0.05
        ranked.append((bm25_scores[i] + position_score, doc))
    ranked.sort(reverse=True, key=lambda x: x[0])
    return [doc for _, doc in ranked[:RETRIEVAL_K]]


def _prefer_latest(docs: List[Document]) -> List[Document]:
    """Keep only chunks from the latest upload per source."""
    if not docs:
        return []
    grouped = {}
    for doc in docs:
        src = doc.metadata.get("source", "unknown")
        grouped.setdefault(src, []).append(doc)

    result = []
    for group in grouped.values():
        latest = max((d.metadata.get("uploaded_at", 0) for d in group), default=0)
        result.extend(d for d in group if d.metadata.get("uploaded_at", 0) == latest)
    return result


# ─────────────────────────────────────────────────────────────────────────────
# RAG PIPELINE (mirrors generate_response without streaming/guardrails)
# ─────────────────────────────────────────────────────────────────────────────
def run_rag_pipeline(
    query: str, llm, vectorstore, reranker
) -> Tuple[str, List[str], float]:
    """
    Execute the full RAG pipeline for a single query.
    Returns: (answer, retrieved_contexts_list, confidence_score)
    """
    # Step 1: Dynamic K retrieval
    k = get_dynamic_k(query)
    retriever = vectorstore.as_retriever(search_kwargs={"k": k})
    raw_docs = retriever.invoke(query)

    if not raw_docs:
        return "No relevant information found.", [], 0.0

    # Step 2: Deduplicate + keep latest version per source
    unique = {hash(d.page_content): d for d in raw_docs}
    latest = _prefer_latest(list(unique.values()))

    # Step 3: Hybrid rerank (BM25 + position)
    hybrid = _hybrid_rerank(query, latest)

    # Step 4: Cross-encoder rerank
    top_score = 0.0
    if hybrid:
        pairs = [(query, doc.page_content) for doc in hybrid]
        scores = reranker.predict(pairs)
        sorted_idx = sorted(
            range(len(scores)), key=lambda i: scores[i], reverse=True
        )
        top_score = float(scores[sorted_idx[0]])
        top_docs = [hybrid[i] for i in sorted_idx[:RERANKER_TOP_K]]
    else:
        top_docs = []

    # Step 5: Build context & generate answer
    contexts = [doc.page_content for doc in top_docs]
    context_str = "\n\n".join(
        f"[[Source: {doc.metadata.get('source', 'Unknown')}]]\n{doc.page_content}"
        for doc in top_docs
    )

    prompt = f"""You are AXIsstant, the official Academic AI of Ateneo de Naga University.
Answer the question using ONLY the context provided.
If the context contains tables or structured data, present them as Markdown tables.
If the context does not contain the answer, say: \
'The retrieved documents do not contain this information.'

**Context:**
{context_str}

**Question:** {query}

**Answer:**"""

    response = llm.invoke(prompt)
    return response.content, contexts, top_score


# ─────────────────────────────────────────────────────────────────────────────
# MODE 1: EVALUATE FROM CSV DATASET (with ground truth -> 5 metrics)
# ─────────────────────────────────────────────────────────────────────────────
def evaluate_from_dataset(csv_path: str = "csea_evaluation_dataset.csv"):
    """
    Run each question through the full RAG pipeline, then grade with RAGAS.
    Metrics: context_precision, context_recall, faithfulness,
             answer_relevancy, answer_correctness
    """
    if not os.path.exists(csv_path):
        print(f"❌ Dataset not found: {csv_path}")
        return None

    test_df = pd.read_csv(csv_path)
    questions = test_df["question"].tolist()
    ground_truths = test_df["ground_truth"].tolist()

    print("⚙️  Initializing pipeline components...")
    llm, embeddings, vectorstore, reranker = _init_components()
    eval_llm = LangchainLLMWrapper(llm)
    eval_embeddings = LangchainEmbeddingsWrapper(embeddings)

    print(f"🤖 Running {len(questions)} queries through the full RAG pipeline...")
    answers, all_contexts, scores = [], [], []

    for i, q in enumerate(questions):
        print(f"  [{i+1}/{len(questions)}] {q[:80]}...")
        try:
            answer, contexts, score = run_rag_pipeline(
                q, llm, vectorstore, reranker
            )
            answers.append(answer)
            all_contexts.append(contexts)
            scores.append(score)
            time.sleep(2)  # Groq rate-limit buffer
        except Exception as e:
            print(f"  ⚠️ Error: {e}")
            answers.append("Error generating response.")
            all_contexts.append([])
            scores.append(0.0)

    # Build RAGAS evaluation dataset
    eval_dataset = Dataset.from_dict({
        "question": questions,
        "answer": answers,
        "contexts": all_contexts,
        "ground_truth": ground_truths,
    })

    # Select metrics
    metrics = [
        context_precision, context_recall, faithfulness, answer_correctness
    ]
    if HAS_ANSWER_RELEVANCY:
        metrics.append(answer_relevancy)

    print(f"⚖️  Grading with RAGAS ({len(metrics)} metrics)...")
    result = evaluate(
        dataset=eval_dataset,
        metrics=metrics,
        llm=eval_llm,
        embeddings=eval_embeddings,
        run_config=RunConfig(max_workers=1, timeout=180),
    )

    # Save reports
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = f"data/eval_report_{timestamp}.csv"
    result_df = result.to_pandas()
    result_df["confidence_score"] = scores
    result_df.to_csv(report_path, index=False)
    result_df.to_csv("final_evaluation_report.csv", index=False)

    _log_to_supabase(result, len(questions), mode="dataset")

    print(f"\n✅ Evaluation Complete!")
    print("=" * 50)
    print(result)
    print(f"📁 Report saved to {report_path}")
    return result


# ─────────────────────────────────────────────────────────────────────────────
# MODE 2: EVALUATE FROM LIVE SUPABASE LOGS (no ground truth -> 2 metrics)
# ─────────────────────────────────────────────────────────────────────────────
def evaluate_from_logs(limit: int = 10):
    """
    Evaluate recent real user interactions stored in Supabase.
    Metrics: faithfulness, answer_relevancy (no ground truth available).
    """
    print(f"🔄 Fetching latest {limit} interactions from Supabase...")
    response = supabase.table("chat_logs").select("query, response, context") \
        .order("created_at", desc=True).limit(limit).execute()

    logs = [r for r in response.data if r.get("context") and r.get("response")]
    if not logs:
        print("❌ No valid logs found.")
        return None

    print(f"📋 Found {len(logs)} valid interactions.")
    llm, embeddings, _, _ = _init_components()
    eval_llm = LangchainLLMWrapper(llm)
    eval_embeddings = LangchainEmbeddingsWrapper(embeddings)

    # Parse contexts: split "[[Source: ...]] content" into a list of chunks
    questions = [x["query"] for x in logs]
    answers = [x["response"] for x in logs]
    contexts = []
    for log in logs:
        ctx = log["context"]
        chunks = [c.strip() for c in ctx.split("[[Source:") if c.strip()]
        clean_chunks = []
        for chunk in chunks:
            lines = chunk.split("\n", 1)
            clean_chunks.append(
                lines[1].strip() if len(lines) > 1 else chunk
            )
        contexts.append(clean_chunks if clean_chunks else [ctx])

    eval_dataset = Dataset.from_dict({
        "question": questions,
        "answer": answers,
        "contexts": contexts,
    })

    metrics = [faithfulness]
    if HAS_ANSWER_RELEVANCY:
        metrics.append(answer_relevancy)

    print(f"⚖️  Grading with RAGAS ({len(metrics)} metrics)...")
    result = evaluate(
        dataset=eval_dataset,
        metrics=metrics,
        llm=eval_llm,
        embeddings=eval_embeddings,
        run_config=RunConfig(max_workers=1, timeout=180),
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = f"data/live_eval_{timestamp}.csv"
    result.to_pandas().to_csv(report_path, index=False)

    _log_to_supabase(result, len(logs), mode="live")

    print(f"\n✅ Live Evaluation Complete!")
    print("=" * 50)
    print(result)
    print(f"📁 Report saved to {report_path}")
    return result


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def _log_to_supabase(result, sample_size: int, mode: str):
    """Persist aggregate metrics to Supabase for admin dashboard tracking."""
    try:
        averages = result.to_pandas().mean(numeric_only=True).to_dict()
        record = {
            "run_at": datetime.now().isoformat(),
            "mode": mode,
            "sample_size": sample_size,
            "faithfulness": averages.get("faithfulness"),
        }
        if HAS_ANSWER_RELEVANCY:
            record["answer_relevancy"] = averages.get("answer_relevancy")
        if mode == "dataset":
            record["context_precision"] = averages.get("context_precision")
            record["context_recall"] = averages.get("context_recall")
            record["answer_correctness"] = averages.get("answer_correctness")

        supabase.table("evaluation_runs").insert(record).execute()
        print("📊 Metrics logged to Supabase.")
    except Exception as e:
        print(f"⚠️ Could not log to Supabase: {e}")


def export_ground_truth_from_logs():
    """Converts 'Helpful' rated logs into ground-truth evaluation pairs."""
    response = supabase.table("chat_logs") \
        .select("*").eq("rating", "helpful").execute()
    logs = response.data

    if not logs:
        return "No helpful logs found to export."

    df = pd.DataFrame(logs)
    df_eval = df[['query', 'context', 'response']].copy()
    df_eval.columns = ['question', 'contexts', 'ground_truth']

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = f"data/real_ground_truth_{timestamp}.csv"
    df_eval.to_csv(path, index=False)
    return f"Exported {len(df_eval)} ground truth pairs to {path}"


# ─────────────────────────────────────────────────────────────────────────────
# CLI ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="CSEA RAG Evaluation Pipeline")
    parser.add_argument(
        "--mode", choices=["dataset", "live", "both"], default="dataset",
        help="dataset = CSV with ground truth, live = Supabase logs, both = run both",
    )
    parser.add_argument(
        "--csv", default="csea_evaluation_dataset.csv",
        help="Path to CSV dataset",
    )
    parser.add_argument(
        "--limit", type=int, default=10,
        help="Number of logs for live mode",
    )
    args = parser.parse_args()

    if args.mode in ("dataset", "both"):
        evaluate_from_dataset(args.csv)
    if args.mode in ("live", "both"):
        evaluate_from_logs(args.limit)
