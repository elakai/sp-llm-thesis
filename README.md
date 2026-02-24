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
C:.
│   .gitignore
│   csea_evaluation_dataset.csv   # Ground truth dataset for RAG testing
│   final_evaluation_report.csv   # Metrics output from the evaluation pipeline
│   get-pip.py
│   pam's instructions.txt
│   pinecone_manifest.json        # Primary ledger for cloud-stored document chunks
│   README.md
│   requirements.txt              # Project dependencies
│   test.py                       # General utility testing script
│
├───assets                        # Visual branding assets
│       kraken_logo.png
│
└───src
    │   run_eval.py               # Main script to execute the RAGAS evaluation suite
    │   __init__.py
    │
    ├───config                    # Global Configuration & Environment Settings
    │       constants.py          # Valid categories, paths, and chunking parameters
    │       logging_config.py     # Windows-compatible Unicode logging setup
    │       settings.py           # LLM, Embeddings, and Vector Database factory
    │
    ├───core                      # Backend Processing (The "Brain")
    │       auth.py               # User authentication and session management
    │       decomposition.py      # Multi-query logic for complex student inquiries
    │       evaluate_rag.py       # RAGAS metrics implementation (Faithfulness, Relevancy)
    │       feedback.py           # Conversation logging and user rating storage
    │       generate_testset.py   # Synthetic data generation for evaluation prep
    │       guardrails.py         # Topic validation and safety filtering
    │       ingestion.py          # Multimodal loading and "Delete After Ingest" logic
    │       retrieval.py          # Semantic Cache, Contextualization, and Reranking
    │       router.py             # Metadata-based category and program routing
    │       __init__.py
    │
    └───ui                        # Frontend Interface (Streamlit)
        │   admin_dashboard.py    # Analytics, Document Ledger, and Index Manager
        │   components.py         # Reusable UI widgets and thinking animations
        │   main.py               # Entry point: `streamlit run src/ui/main.py`
        │   views.py              # Chat and History view controllers
        │   __init__.py
        │
        └───styles                # Custom CSS Assets
                login.css         # Portal styling for the authentication screen
                main.css          # Core visual theme and chat bubble styling

```

---


