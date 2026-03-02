"""
Test Dataset Generator for CSEA Assistant Evaluation

Queries Pinecone for indexed content and uses LLM to generate
diverse Q&A pairs covering all document categories.

Usage:
  python -m src.core.generate_testset            # generate 20 Q&A pairs (default)
  python -m src.core.generate_testset --num 30   # generate 30 Q&A pairs
"""

import os
import sys
import json
import time
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv

project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
load_dotenv(project_root / ".env")

from pinecone import Pinecone
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_groq import ChatGroq
from src.config.logging_config import logger

# Diverse seed queries to discover content across all categories
SEED_QUERIES = [
    # Curriculum
    "BS Mathematics curriculum subjects units prerequisites",
    "BS Civil Engineering curriculum course requirements",
    "BS Computer Engineering curriculum plan of study",
    "BS Electronics Engineering curriculum subjects",
    "BS Architecture curriculum design courses",
    # Handbook — Academic Policies
    "grading system grade scale university",
    "delayed examination policy requirements procedure",
    "academic probation retention policy scholastic standing",
    "enrollment registration requirements procedures",
    "academic honors dean's list cum laude requirements",
    "student code of conduct discipline violations",
    # Handbook — General
    "tuition fees payment schedule installment",
    "attendance policy absences maximum allowed",
    "withdrawal dropping of subjects procedure",
    # Thesis / Research
    "thesis capstone research study electronics engineering",
    "research manuscript format proposal guidelines",
    # OJT
    "on the job training OJT requirements hours industry",
    # Faculty / Organization
    "dean chairperson faculty department head",
    "laboratory rules safety guidelines equipment",
    "organizational structure college departments",
]


def generate_testset(num_questions: int = 20):
    """Generate a diverse evaluation dataset from Pinecone content."""
    print("⚙️  Initializing components...")

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    vectorstore = PineconeVectorStore(
        index=pc.Index(os.getenv("PINECONE_INDEX_NAME")),
        embedding=embeddings,
        text_key="text",
    )
    llm = ChatGroq(
        model_name="llama-3.3-70b-versatile",
        temperature=0.3,
        api_key=os.getenv("GROQ_API_KEY"),
    )

    # ── Step 1: Retrieve diverse chunks from Pinecone ──
    print(f"🔍 Querying Pinecone with {len(SEED_QUERIES)} seed queries...")
    seen_chunks = set()
    unique_chunks = []

    for query in SEED_QUERIES:
        docs = vectorstore.similarity_search(query, k=3)
        for doc in docs:
            # Deduplicate by first 200 chars
            chunk_key = doc.page_content[:200]
            if chunk_key not in seen_chunks:
                seen_chunks.add(chunk_key)
                unique_chunks.append(doc)

    print(f"📦 Found {len(unique_chunks)} unique chunks across all categories.")

    # ── Step 2: Generate Q&A pairs via LLM ──
    prompt_template = """You are an expert academic Q&A generator. Read the following text chunk from a university document.

Generate ONE specific, factual question that a student would realistically ask about this content, and provide the precise correct answer based ONLY on the text.

Rules:
- The question should be natural and specific (e.g., "What are the prerequisites for CE 301?" not "What does this say?")
- The answer must be concise but complete
- If the text contains a table or list, ask about specific items in it
- If the text is about a policy, ask about requirements or procedures
- Do NOT make up information that is not in the text

Source document: {source}

Text Chunk:
{content}

Output ONLY a raw JSON object with no markdown backticks:
{{"question": "your question here", "ground_truth": "the correct answer here"}}"""

    dataset = {"question": [], "contexts": [], "ground_truth": []}
    chunks_to_process = unique_chunks[:num_questions]

    print(f"🧠 Generating {len(chunks_to_process)} Q&A pairs...")

    for i, chunk in enumerate(chunks_to_process):
        source = chunk.metadata.get("source", "unknown")
        print(f"  [{i+1}/{len(chunks_to_process)}] {source}")

        try:
            prompt = prompt_template.format(
                source=source,
                content=chunk.page_content[:2000],  # Cap to avoid token overflow
            )
            response = llm.invoke(prompt)
            clean_json = response.content.strip()

            # Remove markdown code fences if LLM adds them
            if clean_json.startswith("```"):
                clean_json = clean_json.split("\n", 1)[1].rsplit("```", 1)[0].strip()

            data = json.loads(clean_json)

            dataset["question"].append(data["question"])
            dataset["contexts"].append(str([chunk.page_content]))
            dataset["ground_truth"].append(data["ground_truth"])

            time.sleep(1.5)  # Groq rate-limit buffer

        except Exception as e:
            print(f"    ⚠️ Failed: {e}")

    # ── Step 3: Save ──
    df = pd.DataFrame(dataset)
    output_path = "csea_evaluation_dataset.csv"
    df.to_csv(output_path, index=False)

    print(f"\n✅ Generated {len(df)} Q&A pairs → {output_path}")
    print(f"   Sources covered: {len(set(c.metadata.get('source', '') for c in chunks_to_process))}")
    return df


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate CSEA evaluation dataset")
    parser.add_argument(
        "--num", type=int, default=20,
        help="Number of Q&A pairs to generate",
    )
    args = parser.parse_args()
    generate_testset(args.num)
