
# CSEA Information Assistant
### Thesis Prototype – College of Science, Engineering and Architecture  
Ateneo de Naga University

A Retrieval-Augmented Generation (RAG) chatbot that answers student inquiries using official CSEA documents (Student Handbook, dress code, etc.).  
Uses **Llama 3.3 70B** via Groq and **ChromaDB** as vector store.

---

## Setup Instructions (Tested: November 19, 2025)

### 1. Python Version
- Use **Python 3.12.7** (do not use Python 3.13)

### 2. Install Tesseract OCR (Manual – Required for scanned PDFs)
- Download installer: https://github.com/UB-Mannheim/tesseract/wiki
- Run the `.exe` and install to default location  
  → `C:\Program Files\Tesseract-OCR\tesseract.exe`
- This is needed because the handbook and dress code.pdf are scanned/image-only.

### 3. Clone the Repository
```bash
git clone https://github.com/elakai/sp-llm-thesis.git
cd sp-llm-thesis
```

### 4. Create Virtual Environment (Recommended)
**Windows:**
```bash
python -m venv venv
.\venv\Scripts\activate
```

**Mac/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 5. Install Dependencies
```bash
pip install -r requirements.txt
```

`requirements.txt` (current working set):
```txt
streamlit
langchain-community
langchain-core
langchain-huggingface
langchain-groq
langchain-chroma
chromadb
pymupdf
pytesseract
pillow
sentence-transformers
```

### 6. Add Documents (Not in Git – too large)
1. Create a folder named `documents` in the project root.
2. Download these files from Google Drive:  
   https://drive.google.com/drive/folders/1HVGwo_GAvfIZna3_6jTanJB2A9ixEYbH?usp=sharing
3. Place inside `documents` folder

---

## Running the App

```bash
streamlit run app.py
```

### First-Time Setup (Do this once)
1. Open the app in browser
2. Sidebar → Password: `csea2025`
3. Click **FULL RESET**
4. Click **TRAIN BOT**  
   → Takes ~60–90 seconds for both PDFs
5. Wait for balloons → training complete


## Admin Access
- **Password:** `csea2025`
- Allows:
  - Full reset of knowledge base
  - Retrain from `documents` folder
  - View current chunk count


## Project Structure
```
sp-llm-thesis/
├── app.py              Main application
├── requirements.txt
├── documents/          ← Put PDFs here (not tracked by Git)
├── chroma_db/          ← Auto-generated (do not commit)
└── README.md
```

### .gitignore should contain:
```
chroma_db/
documents/
venv/
__pycache__/
*.pyc
```


## Tested Questions (Should answer correctly)
- What is the uniform rule during typhoon signal no. 2 in Naga City?
- Can pregnant students be exempted from wearing uniform?
- What is the vision of ADNU

## Troubleshooting
- If training hangs → make sure Tesseract is installed correctly
- If model error → code already uses `llama-3.3-70b-versatile` (current as of Nov 2025)
- Never commit `chroma_db` or `documents` folder

