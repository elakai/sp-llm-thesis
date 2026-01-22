# CSEA Information Assistant

**Thesis Prototype**  
College of Science, Engineering and Architecture  
Ateneo de Naga University

A **Retrieval-Augmented Generation (RAG)** chatbot that answers student inquiries using official CSEA documents (Student Handbook, dress code rules, typhoon guidelines, pregnancy exemptions, etc.).

### Key technologies
- LLM: Llama 3.3 70B via Groq  
- Embeddings: all-MiniLM-L6-v2 (Sentence Transformers)  
- Vector database: Chroma (persistent)  
- PDF handling: PyMuPDF + Tesseract OCR (for scanned/image-based pages)  
- Frontend: Streamlit

---

## Setup Instructions

### 1. Python Version
- Recommended: **Python 3.11** or **3.12**  
  (3.12.7 tested; avoid Python 3.13 due to compatibility issues with some dependencies)

### 2. Install Tesseract OCR (Required for scanned PDFs)
1. Download the Windows installer:  
   https://github.com/UB-Mannheim/tesseract/wiki  
2. Install to default location:  
   `C:\Program Files\Tesseract-OCR\`  
3. Verify that `tesseract.exe` exists in that folder  
   → Essential for processing scanned/handwritten pages in the handbook PDFs.

### 3. Clone the Repository
```bash
git clone https://gitlab.com/pmagnifico/csea-sp-llm.git
cd csea-sp-llm
```

### 4. Create & Activate Virtual Environment

**Windows – Command Prompt**
```bash
python -m venv venv
venv\Scripts\activate
```

**Windows – PowerShell**
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

### 5. Install Dependencies
```bash
pip install -r requirements.txt
```

**Current `requirements.txt` content:**
```txt
streamlit
langchain-community
langchain-core
langchain-huggingface
langchain-groq
langchain-chroma
chromadb
PyMuPDF
pytesseract
pillow
sentence-transformers
```

### 6. Add Documents
1. Create folder `documents/` in the project root (if not already present)  
2. Download the PDFs from Google Drive:  
   https://drive.google.com/drive/folders/1HVGwo_GAvfIZna3_6jTanJB2A9ixEYbH?usp=sharing  
3. Place all PDF files directly inside the `documents/` folder

---

## Running the Application

With the virtual environment activated, from the project root:

```bash
python -m streamlit run src/ui/main.py
```

(or shorter: `streamlit run src/ui/main.py`)


## Admin Features
**Password:** `csea2025`

- View current number of knowledge chunks  
- **FULL RESET** → delete entire vector database  
- **TRAIN ALL PDFs** → re-index all documents in the `documents/` folder (with OCR fallback)

## Project Structure (Modular)

```
csea-sp-llm/
├── src/
│   ├── __init__.py
│   ├── config/
│   │   └── settings.py           # constants, LLM, embeddings, DB factory
│   ├── core/
│   │   ├── __init__.py
│   │   ├── ingestion.py          # PDF loading, OCR, chunking, indexing
│   │   ├── retrieval.py          # retrieval + prompt + generation
│   │   └── feedback.py           # thumbs up/down saving
│   └── ui/
│       ├── __init__.py
│       ├── components.py         # header, admin panel UI
│       └── main.py               # ← entry point: streamlit run this file
├── documents/                    # your PDFs (not in git)
├── chroma_db/                    # vector store (auto-created, ignore)
├── feedback.json                 # user ratings (local only)
├── requirements.txt
├── .gitignore
└── README.md
```

## .gitignore – Important Entries

```
chroma_db/
documents/
feedback.json
venv/
.venv/
__pycache__/
*.pyc
*.py[cod]
```


