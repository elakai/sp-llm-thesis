import os
import io
import json
import re
import fitz  # PyMuPDF
import pdfplumber
import pandas as pd
from PIL import Image
import pytesseract
from datetime import datetime
from typing import List, Dict, Tuple # <-- Make sure Tuple is capitalized here
from pinecone import Pinecone
from src.config.settings import PINECONE_INDEX_NAME
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from src.config.settings import DOCS_FOLDER, get_vectorstore

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
# ─────────────────────────────────────────────────────────────────────────────
# 1. THE VISUAL RECONSTRUCTION ALGORITHM (No AI, Pure Math)
# ─────────────────────────────────────────────────────────────────────────────
def extract_text_by_visual_layout(pdf_path: str) -> str:
    """
    Extracts text by clustering words that are visually on the same line.
    Fixes 'floating column' issues in curriculum PDFs without using AI.
    """
    full_text = ""
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                # 1. Extract all words with their coordinates (x, y, width, height)
                words = page.extract_words(
                    x_tolerance=3, 
                    y_tolerance=3, 
                    keep_blank_chars=False
                )
                
                # 2. Group words into lines based on "top" Y-coordinate
                # We use a tolerance of 5 pixels to handle slight misalignments
                lines: Dict[int, List[dict]] = {}
                Y_TOLERANCE = 5 
                
                for word in words:
                    found_line = False
                    # Check if this word belongs to an existing line cluster
                    for y_coord in lines.keys():
                        if abs(word['top'] - y_coord) < Y_TOLERANCE:
                            lines[y_coord].append(word)
                            found_line = True
                            break
                    
                    if not found_line:
                        lines[word['top']] = [word]

                # 3. Sort lines from top to bottom
                sorted_y_coords = sorted(lines.keys())
                
                # 4. Construct the text page-by-page
                full_text += f"\n--- Page {page.page_number} ---\n"
                for y in sorted_y_coords:
                    # Sort words in this line from left to right (x0)
                    line_words = sorted(lines[y], key=lambda w: w['x0'])
                    
                    # Reconstruct the string with intelligent spacing
                    line_str = ""
                    last_x1 = 0
                    for w in line_words:
                        # If gap between words is large (>20px), insert a " | " separator
                        # This simulates a table column break
                        if last_x1 > 0 and (w['x0'] - last_x1) > 20:
                            line_str += " | "
                        elif last_x1 > 0:
                            line_str += " "
                        
                        line_str += w['text']
                        last_x1 = w['x1']
                    
                    full_text += line_str + "\n"
                    
    except Exception as e:
        print(f"   ⚠️ Visual Layout Extract Error: {e}")
        return ""

    return full_text

# ─────────────────────────────────────────────────────────────────────────────
# 2. STANDARD LOADERS
# ─────────────────────────────────────────────────────────────────────────────
def clean_text(text: str) -> str:
    """Basic cleanup."""
    if not text: return ""
    text = re.sub(r'\n\s*\n', '\n\n', text) # Max 2 newlines
    return text.strip()

def load_pdf(path: str, filename: str) -> List[Document]:
    """
    Uses the Visual Layout extractor. 
    Falls back to OCR only if the PDF is scanned (image-only).
    """
    print(f"   📖 Reading {filename} (Visual Layout Mode)...")
    
    text_content = extract_text_by_visual_layout(path)
    
    # NEW: Remove our injected page headers before checking length
    real_text = re.sub(r'--- Page \d+ ---', '', text_content).strip()
    
    # If there are fewer than 100 actual characters, it's a scanned image.
    if len(real_text) < 100:
        print(f"   ⚠️ Scanned PDF detected. Switching to Tesseract OCR... This will take a while.")
        try:
            doc = fitz.open(path)
            ocr_text = ""
            for page in doc:
                pix = page.get_pixmap(dpi=150)
                img = Image.open(io.BytesIO(pix.tobytes()))
                ocr_text += f"\n--- Page {page.number + 1} ---\n"
                ocr_text += pytesseract.image_to_string(img) + "\n"
            text_content = ocr_text
        except Exception as e:
            print(f"   ❌ OCR Failed: {e}")

    cleaned = clean_text(text_content)
    if len(cleaned) > 20:
        return [Document(
            page_content=cleaned, 
            metadata={
                "source": filename, 
                "type": "pdf", 
                "uploaded_at": int(datetime.utcnow().timestamp())
            }
        )]
    return []

def load_spreadsheet(path: str, filename: str, is_csv: bool = False) -> List[Document]:
    docs = []
    try:
        df = pd.read_csv(path) if is_csv else pd.read_excel(path)
        df.fillna("N/A", inplace=True)
        text_rows = []
        for _, row in df.iterrows():
            text_rows.append(", ".join([f"{col}: {val}" for col, val in row.items()]))
        docs.append(Document(
            page_content="\n".join(text_rows),
            metadata={"source": filename, "type": "table"}
        ))
    except Exception as e:
        print(f"   ❌ Spreadsheet Error: {e}")
    return docs

# ─────────────────────────────────────────────────────────────────────────────
# 3. MAIN INGESTION LOOP
# ─────────────────────────────────────────────────────────────────────────────
def ingest_all_files():
    if not os.path.exists(DOCS_FOLDER):
        os.makedirs(DOCS_FOLDER)
        return False

    all_docs = []
    print(f"📂 Scanning {DOCS_FOLDER}...")
    files = [f for f in os.listdir(DOCS_FOLDER)]

    if not files:
        print("⚠️ No files found.")
        return False

    for filename in files:
        file_path = os.path.join(DOCS_FOLDER, filename)
        ext = filename.lower().split('.')[-1]
        
        if ext == 'pdf':
            all_docs.extend(load_pdf(file_path, filename))
        elif ext in ['xlsx', 'xls']:
            all_docs.extend(load_spreadsheet(file_path, filename, False))
        elif ext == 'csv':
            all_docs.extend(load_spreadsheet(file_path, filename, True))

    if not all_docs:
        print("❌ No text extracted.")
        return False

    # CHUNKING
    print(f"🧩 Chunking {len(all_docs)} documents...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
    chunks = text_splitter.split_documents(all_docs)

    # --- NEW: COUNT CHUNKS PER FILE ---
    chunk_counts = {}
    for chunk in chunks:
        src = chunk.metadata.get("source", "unknown")
        chunk_counts[src] = chunk_counts.get(src, 0) + 1

    # UPLOAD
    try:
        vectorstore = get_vectorstore()
        print(f"🚀 Uploading {len(chunks)} chunks to Pinecone...")
        vectorstore.add_documents(chunks)
        print("✅ Ingestion Complete!")
        
        # --- NEW: LOG TO MANIFEST WITH CHUNK COUNTS ---
        for filename, count in chunk_counts.items():
            update_manifest(filename, count)
            
        return True
    except Exception as e:
        print(f"❌ Upload Failed: {e}")
        return False

def train_all_pdfs():
    return ingest_all_files()

# ─────────────────────────────────────────────────────────────────────────────
# 4. FILE MANAGEMENT & MANIFEST LEDGER
# ─────────────────────────────────────────────────────────────────────────────
MANIFEST_FILE = "pinecone_manifest.json"

def get_uploaded_files() -> dict:
    """Reads the tracking manifest to see what is currently inside Pinecone."""
    if not os.path.exists(MANIFEST_FILE):
        return {} # <-- Returns an empty DICTIONARY now, preventing the crash!
    try:
        with open(MANIFEST_FILE, "r") as f:
            return json.load(f)
    except:
        return {}

def update_manifest(filename: str, chunk_count: int):
    """Adds or updates a file in the Pinecone tracking manifest."""
    manifest = get_uploaded_files()
    manifest[filename] = {
        "chunks": chunk_count,
        "status": "Active",
        "uploaded_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    with open(MANIFEST_FILE, "w") as f:
        json.dump(manifest, f, indent=4)

def remove_from_manifest(filename: str):
    """Removes a file from the Pinecone tracking manifest."""
    manifest = get_uploaded_files()
    if filename in manifest:
        del manifest[filename]
        with open(MANIFEST_FILE, "w") as f:
            json.dump(manifest, f, indent=4)

def delete_document(filename: str) -> Tuple[bool, str]:
    """Deletes chunks from Pinecone and removes the file from the manifest."""
    try:
        pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))
        index = pc.Index(PINECONE_INDEX_NAME)
        
        index.delete(filter={"source": filename})
        print(f"🗑️ Deleted Pinecone vectors for: {filename}")
        
        remove_from_manifest(filename)
            
        return True, f"Successfully purged {filename} chunks from the database."
        
    except Exception as e:
        print(f"❌ Deletion failed: {e}")
        return False, f"Failed to delete {filename} chunks: {str(e)}"