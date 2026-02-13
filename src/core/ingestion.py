import os
import re
import shutil
import fitz  # PyMuPDF
from PIL import Image
import io
import pytesseract
from datetime import datetime
from typing import List
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from src.config.settings import DOCS_FOLDER, get_vectorstore

# ─────────────────────────────────────────────────────────────────────────────
# Intelligent Tesseract Path Configuration
# ─────────────────────────────────────────────────────────────────────────────
def configure_tesseract():
    """
    Automatically detects Tesseract binary location.
    Prioritizes system PATH, then falls back to known Windows paths.
    """
    tesseract_path = shutil.which("tesseract")

    if not tesseract_path and os.name == 'nt':
        possible_paths = [
            r"C:\Program Files\Tesseract-OCR\tesseract.exe",
            r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
            os.path.expandvars(r"%LOCALAPPDATA%\Tesseract-OCR\tesseract.exe")
        ]
        for path in possible_paths:
            if os.path.exists(path):
                tesseract_path = path
                break

    if tesseract_path:
        pytesseract.pytesseract.tesseract_cmd = tesseract_path
    else:
        print("⚠️ Tesseract not found! OCR will fail if needed.")

# Run configuration on import
configure_tesseract()

# ─────────────────────────────────────────────────────────────────────────────
# Text Sanitization Logic (The "Cleaner")
# ─────────────────────────────────────────────────────────────────────────────
def clean_handbook_text(text: str) -> str:
    """
    Sanitizes raw PDF text to improve retrieval and LLM readability.
    Crucial for Thesis Data Quality.
    """
    if not text: return ""

    # 1. Fix broken hyphenation (e.g., "exam- ination" -> "examination")
    text = re.sub(r'(\w+)-\s+(\w+)', r'\1\2', text)
    
    # 2. Remove multiple spaces (e.g., "The   student" -> "The student")
    text = re.sub(r'\s+', ' ', text)
    
    # 3. Fix "orphan" line breaks in the middle of sentences
    # This keeps paragraphs together so the LLM understands the full context
    text = re.sub(r'(?<!\.)\n(?!\n)', ' ', text)
    
    # 4. Remove weird artifacts often found in headers/footers
    text = text.replace("..", ".").replace(" .", ".")
    
    # 5. Standardize bullet points
    text = text.replace("•", "\n* ").replace("–", "-").replace("—", "-")
    
    return text.strip()

# ─────────────────────────────────────────────────────────────────────────────
# PDF Loading Logic (Updated for Page Metadata)
# ─────────────────────────────────────────────────────────────────────────────
def load_pdf_pages(path: str, filename: str) -> List[Document]:
    """
    Extracts text page-by-page, CLEANS it, and returns a list of Document objects.
    Each document contains the text of one page + metadata (page number).
    """
    doc = fitz.open(path)
    page_docs = []
    
    timestamp = int(datetime.utcnow().timestamp())
    doc_type = "handbook" if "handbook" in filename.lower() else "general"

    for page_num, page in enumerate(doc):
        # A. Extract Raw Text
        page_text = page.get_text()
        char_count = len(page_text.strip())

        # B. Fallback to OCR if page looks scanned (< 100 chars)
        if char_count < 100:
            try:
                pix = page.get_pixmap(dpi=300)
                img = Image.open(io.BytesIO(pix.tobytes()))
                # Only runs if Tesseract is configured
                page_text = pytesseract.image_to_string(img, lang='eng', config='--oem 3 --psm 6')
            except Exception:
                page_text = "" # Graceful failure

        # 🚀 C. APPLY THE CLEANER (The Critical Update)
        cleaned_text = clean_handbook_text(page_text)

        # D. Create Document only if valuable text remains
        if len(cleaned_text) > 50:
            page_docs.append(Document(
                page_content=cleaned_text,
                metadata={
                    "source": filename,
                    "page": page_num + 1,  # Human-readable page number (starts at 1)
                    "upload_timestamp": timestamp,
                    "document_type": doc_type
                }
            ))

    doc.close()
    return page_docs

# ─────────────────────────────────────────────────────────────────────────────
# Training / Ingestion Logic (Function Name MUST be 'train_all_pdfs')
# ─────────────────────────────────────────────────────────────────────────────
def train_all_pdfs():
    """Rebuild Pinecone vector store from all PDFs in DOCS_FOLDER."""
    if not os.path.exists(DOCS_FOLDER):
        raise FileNotFoundError(f"Folder not found: {DOCS_FOLDER}. Create it and add PDFs.")

    all_raw_docs = []
    print(f"Scanning {DOCS_FOLDER}...")

    for filename in os.listdir(DOCS_FOLDER):
        if filename.lower().endswith(".pdf"):
            full_path = os.path.join(DOCS_FOLDER, filename)
            try:
                # Load pages individually to preserve page numbers
                file_pages = load_pdf_pages(full_path, filename)
                all_raw_docs.extend(file_pages)
                print(f"✅ Loaded & Cleaned {len(file_pages)} pages from {filename}")
            except Exception as e:
                print(f"❌ Failed to load {filename}: {e}")
                continue

    if not all_raw_docs:
        print("⚠️ No valid text found in PDFs. Check if they are image-only and Tesseract is installed.")
        return False

    print(f"Total pages processed: {len(all_raw_docs)}")

    # 1. Split Text (Respecting Page Boundaries)
    # The splitter will now split large pages but KEEP the 'page' metadata
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=250,
        separators=["\n\n", "\n", ".", " ", ""] # Prioritize paragraph breaks
    )
    chunks = text_splitter.split_documents(all_raw_docs)
    print(f"Created {len(chunks)} chunks")

    # 2. Save to Pinecone
    try:
        vectorstore = get_vectorstore()
        # Adding in batches is often safer, but basic add is fine for now
        vectorstore.add_documents(chunks)
        print(f"🚀 Successfully indexed {len(chunks)} chunks into Pinecone")
        return True
    except Exception as e:
        print(f"❌ Vector Database Error: {e}")
        return False