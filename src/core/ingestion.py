# src/core/ingestion.py
import os
from datetime import datetime
import fitz  # PyMuPDF
from PIL import Image
import io
import pytesseract
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from src.config.settings import DOCS_FOLDER, get_vectorstore

# Tesseract path – set explicitly to avoid PATH issues
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

def load_pdf(path: str) -> str:
    """Extract text from PDF, fallback to OCR if page is scanned."""
    doc = fitz.open(path)
    text = ""
    for page_num, page in enumerate(doc):
        page_text = page.get_text()
        char_count = len(page_text.strip())
        print(f"Page {page_num+1}: {char_count} chars from normal extraction")

        if char_count < 100:
            print(f" → Page {page_num+1} likely scanned, running OCR")
            try:
                pix = page.get_pixmap(dpi=300)  # increased DPI for better OCR
                img = Image.open(io.BytesIO(pix.tobytes()))
                page_text = pytesseract.image_to_string(img, lang='eng', config='--oem 3 --psm 6')
                print(f" → OCR extracted {len(page_text.strip())} chars")
            except pytesseract.TesseractNotFoundError:
                print(f" ✗ Tesseract not found - skipping OCR for page {page_num+1}")
                print("   Install Tesseract: https://github.com/UB-Mannheim/tesseract/wiki")
            except Exception as e:
                print(f" ✗ OCR error on page {page_num+1}: {str(e)}")

        text += page_text + "\n\n"

    doc.close()
    return text

def train_all_pdfs():
    """Rebuild Pinecone vector store from all PDFs in DOCS_FOLDER."""
    if not os.path.exists(DOCS_FOLDER):
        raise FileNotFoundError(f"Folder not found: {DOCS_FOLDER}. Create it and add PDFs.")

    docs = []
    for filename in os.listdir(DOCS_FOLDER):
        if filename.lower().endswith(".pdf"):
            full_path = os.path.join(DOCS_FOLDER, filename)
            text = load_pdf(full_path)

            if not text.strip():
                print(f"Warning: No usable text extracted from {filename} (skipping)")
                continue

            docs.append(Document(
                page_content=text,
                metadata={
                    "source": filename,
                    "upload_timestamp": int(datetime.utcnow().timestamp()),  # number for filtering
                    "document_type": "handbook" if "handbook" in filename.lower() else "dress_code" if "dress" in filename.lower() else "other"
                }
            ))

    if not docs:
        raise ValueError("No valid PDFs with extractable text found in documents folder.")

    print(f"Processed {len(docs)} valid documents")

    chunks = RecursiveCharacterTextSplitter(
        chunk_size=800,          # slightly smaller for better precision
        chunk_overlap=150         # moderate overlap
    ).split_documents(docs)

    print(f"Created {len(chunks)} chunks")

    # Save to Pinecone
    vectorstore = get_vectorstore()
    vectorstore.add_documents(chunks)
    print(f"Successfully added {len(chunks)} chunks to Pinecone")