import os
import json
import pandas as pd
from langchain_core.prompts import ChatPromptTemplate
from src.config.settings import get_generator_llm, DOCS_FOLDER
from src.core.ingestion import load_pdf

def generate_synthetic_data():
    print("🚀 Initializing Custom Q&A Generator...")
    llm = get_generator_llm()

    print(f"📂 Loading documents from {DOCS_FOLDER}...")
    all_docs = []
    files = [f for f in os.listdir(DOCS_FOLDER) if f.lower().endswith(".pdf")]
    
    if not files:
        print("❌ No PDFs found.")
        return

    test_file = files[0] 
    file_path = os.path.join(DOCS_FOLDER, test_file)
    print(f"   Reading: {test_file}")
    
    # Load the document using your visual layout extractor
    all_docs.extend(load_pdf(file_path, test_file))
    
    if not all_docs:
        print("❌ No text extracted.")
        return

    # We will use the first 3 chunks to test the pipeline
    chunks = all_docs[:3] 
    
    # This prompt forces the LLM to output clean, parsable JSON
    prompt = ChatPromptTemplate.from_template("""
    You are an expert academic data extractor. Read the following curriculum text chunk.
    Create ONE realistic question a student would ask about this specific text.
    Then, provide the exact, correct answer based ONLY on the text.
    
    Output ONLY a raw JSON object with no markdown backticks, using this exact format:
    {{"question": "your question", "ground_truth": "the answer"}}
    
    Text Chunk:
    {context}
    """)

    dataset = {"question": [], "contexts": [], "ground_truth": []}
    
    print(f"🧠 Generating {len(chunks)} test questions. Please wait...")
    
    for i, chunk in enumerate(chunks):
        print(f"   Processing chunk {i+1}/{len(chunks)}...")
        try:
            response = llm.invoke(prompt.format(context=chunk.page_content))
            
            # Clean up the LLM response in case it adds markdown formatting
            clean_json = response.content.strip().replace("```json", "").replace("```", "")
            data = json.loads(clean_json)
            
            dataset["question"].append(data["question"])
            # Ragas expects contexts to be a list of strings
            dataset["contexts"].append([chunk.page_content]) 
            dataset["ground_truth"].append(data["ground_truth"])
            
        except Exception as e:
            print(f"   ⚠️ Failed to parse chunk {i+1}: {e}")

    # Export exactly how Ragas expects it
    df = pd.DataFrame(dataset)
    output_file = "csea_evaluation_dataset.csv"
    df.to_csv(output_file, index=False)
    print(f"✅ Success! Test dataset saved to {output_file}")

if __name__ == "__main__":
    generate_synthetic_data()