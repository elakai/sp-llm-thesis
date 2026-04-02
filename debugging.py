import os
from dotenv import load_dotenv
from langchain_pinecone import PineconeVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from pinecone import Pinecone

# Load environment variables
load_dotenv()

print("⚙️  Connecting to Pinecone...")
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
vectorstore = PineconeVectorStore(
    index=pc.Index(os.getenv("PINECONE_INDEX_NAME")),
    embedding=embeddings,
    text_key="text",
)

query = "who is karla sobrevilla"
print(f"\n🔍 Searching for: '{query}'")

# Fetch the top 5 chunks
docs = vectorstore.similarity_search(query, k=5)

print("\n" + "="*50)
print("📦 WHAT THE LLM IS ACTUALLY SEEING:")
print("="*50)

found_karla = False
for i, doc in enumerate(docs):
    source = doc.metadata.get('source', 'Unknown')
    content = doc.page_content
    print(f"\n--- CHUNK {i+1} (Source: {source}) ---")
    print(content)
    
    if "Karla" in content and "Sobrevilla" in content:
        found_karla = True

print("\n" + "="*50)
if found_karla:
    print("✅ CONCLUSION: Pinecone IS finding her! The LLM is ignoring her.")
    print("   Fix: We need to make the LLM prompt even more forceful.")
else:
    print("❌ CONCLUSION: Pinecone is NOT finding her! The database is failing.")
    print("   Fix: We need to check your chunk size or clear old conflicting PDF vectors.")