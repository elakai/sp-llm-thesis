# CSEA Information Assistant (AXIsstant)

**Thesis Prototype** 

College of Science, Engineering and Architecture

Ateneo de Naga University

A **Retrieval-Augmented Generation (RAG)** chatbot designed to provide accurate, conversational answers to student inquiries using official CSEA documents (Thesis manuscripts, Curriculums, Memos, and Lab Manuals).

## Key Technologies

* **LLM**: Llama 3.3 70B via Groq
* **Embeddings**: `all-MiniLM-L6-v2` (Sentence Transformers)
* **Vector Database**: Pinecone (Serverless)
* **PDF Handling**: PyMuPDF + pdfplumber + Tesseract OCR
* **Document Loading**: Support for `.pdf`, `.docx`, `.csv`, `.xlsx`, and images (with text)
* **Frontend**: Streamlit

---

## 📂 Subfolder Category Routing

The system uses a hierarchical folder structure to automatically tag documents with metadata categories. Organize your `documents/` folder as follows:

* `documents/curriculum/` — Academic program matrices
* `documents/thesis/` — Past CSEA Thesis Manuscripts
* `documents/memos/` — Official memorandums and circulars
* `documents/ojt/` — Internship requirements and partner lists
* `documents/laboratory/` — Lab manuals and equipment inventories

---

## ⚡ Setup Instructions

### 1. Python Version

* **Recommended**: Python 3.11 or 3.12 (3.12.7 tested).

### 2. Tesseract OCR Installation

1. Download installer: [UB-Mannheim/tesseract](https://github.com/UB-Mannheim/tesseract/wiki).
2. Install to: `C:\Program Files\Tesseract-OCR\`.
3. Add the path to your `.env` file as `TESSERACT_CMD`.

### 3. Virtual Environment & Dependencies

```bash
python -m venv venv
# Windows
venv\Scripts\activate
pip install -r requirements.txt

```

### 4. Storage Optimization: Delete After Ingest

**Note:** To maintain a lean server environment, the system is configured to **automatically delete** physical files from the `documents/` subfolders once they are successfully indexed in Pinecone. The `pinecone_manifest.json` serves as the permanent record of active documents.

---

## 🛠 Running the Application

```bash
streamlit run src/ui/main.py

```

---

### Admin Features

* **Document Ledger**: View and manage "invisible" cloud-stored documents via the manifest.
* **Chunk Inspector**: Debug the top-K retrieved context chunks.
* **Auto-Wipe Cache**: The semantic cache is automatically invalidated after every ingestion run to ensure data freshness.


---

## Project Structure (Modular Architecture)

The project is organized to separate core RAG logic from the user interface and configuration settings.

```text
csea-sp-llm/
├── src/
│   ├── config/
│   │   └── settings.py           # API keys, LLM/Embedding factory, and DB config
│   ├── core/                     # Backend Logic (The "Brain")
│   │   ├── auth.py               # User authentication and session management
│   │   ├── decomposition.py      # Query decomposition for complex multi-part questions
│   │   ├── evaluate_rag.py       # Ragas metrics and evaluation logic
│   │   ├── feedback.py           # Supabase logging and user rating (thumbs up/down)
│   │   ├── generate_testset.py   # Synthetic test set generation for RAG evaluation
│   │   ├── guardrails.py         # Topic validation and safety filtering
│   │   ├── ingestion.py          # PDF/Word/OCR loading, chunking, and Pinecone indexing
│   │   ├── retrieval.py          # Contextualization, Semantic Cache, and Reranking
│   │   └── router.py             # Intent detection and Metadata/Category routing
│   ├── ui/                       # Frontend (Streamlit)
│   │   ├── admin_dashboard.py    # Document ledger and performance analytics
│   │   ├── components.py         # Reusable UI widgets (headers, footers)
│   │   ├── main.py               # Entry point: run `streamlit run src/ui/main.py`
│   │   ├── views.py              # Chat and History view controllers
│   │   └── styles/               # Custom CSS for login and main interface
│   │       ├── login.css
│   │       └── main.css
│   └── run_eval.py               # Script to execute the RAG evaluation pipeline
├── assets/                       # Images and logos
│   └── kraken_logo.png
├── pinecone_manifest.json        # Primary ledger for cloud-stored document chunks
├── csea_evaluation_dataset.csv   # Ground truth data for model testing
├── final_evaluation_report.csv   # Generated metrics (Faithfulness, Relevancy, etc.)
├── requirements.txt              # Project dependencies
└── README.md                     # Documentation

```

---


