"""
Test Dataset Generator for CSEA Assistant Evaluation

Reads manifest_rows.csv to guarantee every uploaded document is tested.
Uses Pinecone metadata filtering to extract chunks from each specific file.

Usage:
  python -m src.core.generate_testset            # generate 150 Q&A pairs (default)
  python -m src.core.generate_testset --num 200  # generate 200 Q&A pairs
"""

import os
import sys
import json
import time
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv
from pinecone import Pinecone
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_groq import ChatGroq

load_dotenv()

def generate_testset(num_questions: int = 150):
    print("⚙️  Initializing components...")
    
  # 1. Load the manifest
    manifest_path = Path("manifest_rows.csv")
        
    if not manifest_path.exists():
        print("❌ Error: manifest_rows.csv not found! Please ensure it is in your project directory.")
        return None

    manifest_df = pd.read_csv(manifest_path)
    # Only pull from documents marked as Active
    active_files = manifest_df[manifest_df['status'] == 'Active']['filename'].tolist()
    print(f"📄 Found {len(active_files)} active documents in the manifest.")

    if not active_files:
        print("❌ No active files found in manifest.")
        return None

    # 2. Setup Pinecone and LLM
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

    # 3. Calculate how many chunks to pull per file to reach the target number
    questions_per_file = max(1, num_questions // len(active_files))
    
    print(f"🔍 Fetching ~{questions_per_file} chunks per document from Pinecone...")
    seen_chunks = set()
    unique_chunks = []

    for filename in active_files:
        try:
            # We use a generic query but force Pinecone to ONLY look inside this specific filename
            docs = vectorstore.similarity_search(
                "academic policies curriculum information guidelines rules", 
                k=questions_per_file + 2, # Fetch slightly more to account for duplicates
                filter={"source": filename}
            )
            
            for doc in docs:
                chunk_key = doc.page_content[:200]
                if chunk_key not in seen_chunks:
                    seen_chunks.add(chunk_key)
                    unique_chunks.append(doc)
        except Exception as e:
            print(f"  ⚠️ Could not fetch chunks for {filename}: {e}")

    # Ensure we don't exceed the requested number of questions
    chunks_to_process = unique_chunks[:num_questions]
    print(f"📦 Successfully loaded {len(chunks_to_process)} unique chunks across {len(active_files)} categories.")

    # 4. Generate the Ground Truth (Updated Prompt to fix Verbosity Penalty)
    prompt_template = """You are an expert academic Q&A generator. Read the following text chunk from a university document.

Generate ONE specific, factual question that a student would realistically ask about this content, and provide the precise correct answer based ONLY on the text.

Rules:
- The question should be natural and specific (e.g., "What are the prerequisites for CE 301?" not "What does this say?")
- The answer MUST be a detailed, full-sentence explanation. 
- If the data belongs in a table or list, format the ground truth as a markdown table or bulleted list so it matches the expected helpful output of a chatbot.
- Do NOT make up information that is not in the text.

Source document: {source}

Text Chunk:
{content}

Output ONLY a raw JSON object with no markdown backticks:
{{"question": "your question here", "ground_truth": "the detailed correct answer here"}}"""

    dataset = {"question": [], "ground_truth": []}
    
    print(f"🧠 Generating {len(chunks_to_process)} Q&A pairs...")
    for i, chunk in enumerate(chunks_to_process):
        source = chunk.metadata.get("source", "unknown")
        print(f"  [{i+1}/{len(chunks_to_process)}] {source}")
        
        try:
            prompt = prompt_template.format(
                source=source,
                content=chunk.page_content[:2000],
            )
            response = llm.invoke(prompt)
            clean_json = response.content.strip()
            
            if clean_json.startswith("```"):
                clean_json = clean_json.split("\n", 1)[1].rsplit("```", 1)[0].strip()
                
            data = json.loads(clean_json)
            dataset["question"].append(data["question"])
            dataset["ground_truth"].append(data["ground_truth"])
            
        except Exception as e:
            print(f"    ⚠️ Failed generating Q&A for {source}: {e}")

    # 5. Save the Dataset
    df = pd.DataFrame(dataset)
    output_path = "csea_evaluation_dataset.csv"
    df.to_csv(output_path, index=False)
    
    print(f"\n✅ Generated {len(df)} Q&A pairs → {output_path}")
    print(f"   Total Documents Covered: {len(set(c.metadata.get('source', '') for c in chunks_to_process))} out of {len(active_files)}")
    return df

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate CSEA evaluation dataset from Manifest")
    parser.add_argument(
        "--num", type=int, default=150,
        help="Number of total Q&A pairs to generate",
    )
    args = parser.parse_args()
    generate_testset(args.num)