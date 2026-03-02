# AXIsstant — CSEA Academic AI Assistant

**Thesis Project** — College of Science, Engineering and Architecture, Ateneo de Naga University

A **Retrieval-Augmented Generation (RAG)** chatbot that answers student inquiries using official CSEA documents — curriculum guides, thesis manuscripts, memorandums, lab manuals, and OJT requirements.

---

## Architecture

```
User Query
  → Input Validation & PII Redaction
  → Conversational Memory (pronoun-aware contextualization)
  → Semantic Cache Check
  → Intent Detection (greeting / off-topic / search)
  → Query Decomposition (for complex multi-part questions)
  → Pinecone Vector Search (dynamic K = 10 or 15)
  → Deduplication & Version Filtering
  → Hybrid Reranking (BM25 + CrossEncoder ms-marco-MiniLM-L-6-v2)
  → Three-Tier Confidence Gate
      ├─ Low  (< -13.0): abort, no hallucination risk
      ├─ Mid  (-13 to -5): generate → Critic verifies against context
      └─ High (> -5.0):  generate → trust draft, skip Critic
  → Streaming Response
```

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| LLM | Llama 3.3 70B via Groq API |
| Embeddings | `all-MiniLM-L6-v2` (HuggingFace) |
| Reranker | `ms-marco-MiniLM-L-6-v2` (CrossEncoder) |
| Vector DB | Pinecone (Serverless) |
| Auth & Storage | Supabase (authentication, chat logs, document manifest) |
| PDF Processing | PyMuPDF + pdfplumber + Tesseract OCR |
| Frontend | Streamlit |
| Evaluation | RAGAS (Faithfulness, Answer Relevancy) |

---

## Project Structure

```
src/
├── config/
│   ├── constants.py          # Tuning parameters, keywords, thresholds
│   ├── logging_config.py     # Unicode-safe logging
│   └── settings.py           # Lazy-loaded LLM, embeddings, vectorstore factories
│
├── core/
│   ├── auth.py               # Supabase user authentication
│   ├── decomposition.py      # Multi-query decomposition for complex questions
│   ├── evaluate_rag.py       # RAGAS evaluation pipeline
│   ├── feedback.py           # Chat logging and user ratings
│   ├── generate_testset.py   # Synthetic test data generation
│   ├── guardrails.py         # Input validation, PII redaction, Critic persona
│   ├── ingestion.py          # Batch document ingestion from filesystem
│   ├── memory_ingestion.py   # In-browser upload → process in memory → Pinecone
│   ├── retrieval.py          # Main RAG pipeline (cache, retrieve, rerank, generate)
│   └── router.py             # Intent detection and dynamic retrieval depth
│
└── ui/
    ├── admin_dashboard.py    # Document management, health metrics, eval logs
    ├── components.py         # Login screen, sidebar, reusable widgets
    ├── main.py               # App entry point
    ├── views.py              # Chat and history views
    └── styles/               # CSS for login and main themes
```

---

## Setup

### Prerequisites

- Python 3.11 or 3.12
- [Tesseract OCR](https://github.com/UB-Mannheim/tesseract/wiki) (required for scanned PDFs)

### Installation

```bash
python -m venv venv
venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

### Environment Variables

Create a `.env` file in the project root:

```env
PINECONE_API_KEY=your_key
PINECONE_INDEX_NAME=csea-assistant
GROQ_API_KEY=your_key
SUPABASE_URL=your_url
SUPABASE_KEY=your_key
TESSERACT_CMD=C:\Program Files\Tesseract-OCR\tesseract.exe
```

On Streamlit Cloud, set these in **Settings → Secrets** instead.

### Run

```bash
streamlit run src/ui/main.py
```

---

## Document Management

Documents are organized in subfolders that map to metadata categories:

```
documents/
├── curriculum/    # Academic program guides
├── thesis/        # Past CSEA thesis manuscripts
├── memos/         # Official memorandums and circulars
├── ojt/           # Internship requirements
└── laboratory/    # Lab manuals and equipment inventories
```

**Admin dashboard** allows uploading documents directly through the browser. Files are processed entirely in memory — never stored permanently — and only the vector embeddings are saved to Pinecone. The Supabase manifest table tracks what has been indexed.

---

## Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| No metadata filters in retrieval | Keyword-based filters caused false positives (e.g., "faculty" → organization → 0 results). Hybrid reranking handles relevance accurately with ~945 vectors. |
| Split generator / critic LLMs | Generator uses temp=0.1 for natural responses; Critic uses temp=0.0 for deterministic fact-checking. |
| Lazy ML imports | `sentence-transformers` (~400 MB) and `langchain-huggingface` (~90 MB) load only when first called, cutting cold-start time. |
| Semantic cache | Embedding-based similarity cache (threshold 0.95) avoids redundant LLM calls for repeated questions. |
| Three-tier confidence | CrossEncoder logit scores gate generation: abort on junk retrieval, verify moderate matches, trust strong matches. Prevents hallucination without over-relying on the Critic. |

---

## Evaluation

```bash
python src/run_eval.py
```

Uses RAGAS metrics (Faithfulness, Answer Relevancy) against `csea_evaluation_dataset.csv`. Results are written to `final_evaluation_report.csv` and logged to Supabase.


