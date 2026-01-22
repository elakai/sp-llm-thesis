# src/core/ingestion.py
import os
import pymupdf as fitz
from PIL import Image
import io
import pytesseract
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from src.config.settings import DB_PATH, DOCS_FOLDER, get_embeddings, get_db

# TESSERACT
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

def load_pdf(path: str) -> str:
    """Extract text from PDF, fallback to OCR if page is scanned."""
    doc = fitz.open(path)
    text = ""
    for page in doc:
        page_text = page.get_text()
        if len(page_text.strip()) < 100:
            pix = page.get_pixmap(dpi=250)
            img = Image.open(io.BytesIO(pix.tobytes()))
            page_text = pytesseract.image_to_string(img, lang='eng', config='--oem 3 --psm 6')
        text += page_text + "\n\n"
    doc.close()
    return text

def train_all_pdfs():
    """Rebuild vector database from all PDFs in DOCS_FOLDER."""
    if not os.path.exists(DOCS_FOLDER):
        raise FileNotFoundError(f"Folder not found: {DOCS_FOLDER}. Create it and add PDFs.")

    docs = []
    for filename in os.listdir(DOCS_FOLDER):
        if filename.lower().endswith(".pdf"):
            full_path = os.path.join(DOCS_FOLDER, filename)
            text = load_pdf(full_path)
            docs.append(Document(page_content=text, metadata={"source": filename}))

    if not docs:
        raise ValueError("No PDF files found in documents folder.")

    chunks = RecursiveCharacterTextSplitter(
        chunk_size=1200,
        chunk_overlap=200
    ).split_documents(docs)

    db = get_db()
    db.add_documents(chunks)