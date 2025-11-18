import streamlit as st
import os
import tempfile
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader, CSVLoader, UnstructuredExcelLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# ==================== 1. STREAMLIT SETUP ====================
st.set_page_config(page_title="CSEA Info Assistant", page_icon="🦅")

# ==================== 2. CONFIGURATION ====================
DB_PATH = "./chroma_db"
COLLECTION_NAME = "csea_docs"
LOCAL_DOCS_FOLDER = "./documents"

# === GROQ API KEY ===
GROQ_API_KEY = "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"

# ==================== 3. MODEL INITIALIZATION ====================
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0.2,
    groq_api_key=GROQ_API_KEY
)

@st.cache_resource(show_spinner="🧠 Loading embedding model...")
def load_embeddings():
    return HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'}
    )

embeddings = load_embeddings()

# Initialize Vector DB
vectordb = Chroma(
    persist_directory=DB_PATH, 
    embedding_function=embeddings, 
    collection_name=COLLECTION_NAME
)

# ==================== 4. HELPER FUNCTION: INGEST DOCUMENTS ====================
def ingest_documents(folder_path):
    """Scans a folder and adds files to the vector DB."""
    if not os.path.exists(folder_path):
        return False, f"Folder not found: {folder_path}"

    docs = []
    files_found = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
    
    if not files_found:
        return False, "No files found in the folder."

    status_text = st.empty()
    status_text.info(f"🔄 Auto-processing {len(files_found)} files from {folder_path}...")
    
    for filename in files_found:
        file_path = os.path.join(folder_path, filename)
        try:
            loader = None
            if filename.endswith(".pdf"):
                loader = PyPDFLoader(file_path)
            elif filename.endswith(".docx"):
                loader = Docx2txtLoader(file_path)
            elif filename.endswith(".txt"):
                loader = TextLoader(file_path, encoding="utf-8")
            elif filename.endswith(".csv"):
                loader = CSVLoader(file_path)
            elif filename.endswith(".xlsx"):
                loader = UnstructuredExcelLoader(file_path)
            
            if loader:
                docs.extend(loader.load())
        except Exception as e:
            st.warning(f"Could not process {filename}: {e}")

    if docs:
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = splitter.split_documents(docs)
        vectordb.add_documents(chunks)
        status_text.success(f"✅ Successfully processed {len(files_found)} files ({len(chunks)} chunks)!")
        return True, "Success"
    else:
        status_text.warning("No compatible documents found to add.")
        return False, "No docs"

# ==================== 5. AUTOMATIC STARTUP CHECK ====================
# Check if DB is empty. If so, ingest automatically.
try:
    # Check count of documents in the collection
    if len(vectordb.get()['ids']) == 0:
        ingest_documents(LOCAL_DOCS_FOLDER)
except Exception as e:
    st.error(f"Error checking database state: {e}")

# ==================== 6. RAG CHAIN SETUP ====================
retriever = vectordb.as_retriever(search_kwargs={"k": 6})

template = """You are the official CSEA Information Assistant for the College of Science, Engineering, and Architecture at Ateneo de Naga University.
Answer ONLY using the provided official documents. Be polite, accurate, and concise.
If the information is not in the documents, reply exactly:
"I don't have that information in the official documents yet. Please contact the CSEA Dean's Office or your department."

Context:
{context}

Question: {question}
Answer:"""

prompt = ChatPromptTemplate.from_template(template)

chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# ==================== 7. STREAMLIT UI ====================
st.title("🦅 CSEA Information Assistant")
st.markdown("**College of Science, Engineering, and Architecture – Ateneo de Naga University**")
st.caption("Powered by Llama 3.3 70B + RAG • Thesis Prototype 2025")

# Sidebar – Admin
with st.sidebar:
    st.header("🔑 Admin Settings")
    password = st.text_input("Admin Password", type="password")
    
    if password == "csea2025":
        st.success("Logged in")
        
        if st.button("🔄 Force Re-scan './documents'"):
            with st.spinner("Rescanning folder..."):
                ingest_documents(LOCAL_DOCS_FOLDER)
        
        st.divider()
        
        uploaded_files = st.file_uploader(
            "Upload Additional Files",
            accept_multiple_files=True,
            type=["pdf", "docx", "txt", "csv", "xlsx"]
        )
        
        if st.button("🚀 Add Uploaded Files") and uploaded_files:
            with st.spinner("Processing uploads..."):
                # Reuse logic for uploads
                docs = []
                for file in uploaded_files:
                    suffix = "." + file.name.split('.')[-1]
                    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tfile:
                        tfile.write(file.read())
                        path = tfile.name
                    
                    try:
                        if file.name.endswith(".pdf"): docs.extend(PyPDFLoader(path).load())
                        elif file.name.endswith(".docx"): docs.extend(Docx2txtLoader(path).load())
                        elif file.name.endswith(".txt"): docs.extend(TextLoader(path, encoding="utf-8").load())
                        elif file.name.endswith(".csv"): docs.extend(CSVLoader(path).load())
                        elif file.name.endswith(".xlsx"): docs.extend(UnstructuredExcelLoader(path).load())
                    except Exception as e:
                        st.error(f"Error: {e}")
                    finally:
                        if os.path.exists(path): os.unlink(path)
                
                if docs:
                    chunks = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200).split_documents(docs)
                    vectordb.add_documents(chunks)
                    st.success(f"Added {len(chunks)} new chunks!")

# Chat Interface
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello! I'm the CSEA Information Assistant. Ask me anything about schedules, requirements, faculty, offices, etc. 🦅"}
    ]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if query := st.chat_input("Ask a question about CSEA..."):
    st.session_state.messages.append({"role": "user", "content": query})
    st.chat_message("user").write(query)
    
    with st.chat_message("assistant"):
        with st.spinner("Searching official documents..."):
            try:
                response = chain.invoke(query)
                st.write(response)
                st.session_state.messages.append({"role": "assistant", "content": response})
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")