import os
import io
import json
import re
import cv2
import numpy as np
import fitz 
import pdfplumber
import pandas as pd
import docx
from PIL import Image
import pytesseract
from datetime import datetime
from typing import List, Dict, Tuple
from pinecone import Pinecone
from collections import defaultdict
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from src.config.settings import PINECONE_INDEX_NAME, get_vectorstore, TESSERACT_CMD
from src.config.constants import DOCS_FOLDER, MANIFEST_FILE, CHUNK_SIZE, CHUNK_OVERLAP
from src.config.logging_config import logger
from src.core.retrieval import invalidate_cache
from src.config.constants import VALID_CATEGORIES
from src.core.feedback import supabase


pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD

def normalize_source_key(filename: str) -> str:
    return filename.strip().replace("\\", "/")

def is_already_ingested(filename: str) -> bool:
    """Checks Supabase manifest instead of local JSON."""
    norm_name = normalize_source_key(filename)
    try:
        response = supabase.table("manifest") \
            .select("status") \
            .eq("filename", norm_name) \
            .execute()
        return bool(response.data) and response.data[0]["status"] == "Active"
    except Exception as e:
        logger.error(f"Manifest check failed: {e}")
        return False

def clean_text(text: str) -> str:
    if not text: return ""
    text = re.sub(r'\n\s*\n', '\n\n', text) 
    return text.strip()

# --- VISION & TABLE PROCESSING ---
def preprocess_image_for_ocr(pil_image: Image) -> Image:
    img = np.array(pil_image.convert("RGB"))
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, blockSize=31, C=10)
    img = cv2.medianBlur(img, 3)
    return Image.fromarray(img)

def convert_table_to_markdown(table_data: list) -> str:
    if not table_data: return ""
    cleaned = [[str(cell).strip() if cell else "" for cell in row] for row in table_data]
    header = cleaned[0]
    md = "| " + " | ".join(header) + " |\n"
    md += "| " + " | ".join(["---"] * len(header)) + " |\n"
    for row in cleaned[1:]:
        while len(row) < len(header): row.append("")
        md += "| " + " | ".join(row) + " |\n"
    return md

def is_inside_any_bbox(word: dict, bboxes: list) -> bool:
    wx = (word['x0'] + word['x1']) / 2
    wy = (word['top'] + word['bottom']) / 2
    return any(x0 <= wx <= x1 and top <= wy <= bottom for (x0, top, x1, bottom) in bboxes)

def reconstruct_body_text(words: list) -> str:
    lines: Dict[int, List[dict]] = {}
    for word in words:
        found = False
        for y_coord in lines.keys():
            if abs(word['top'] - y_coord) < 5:
                lines[y_coord].append(word)
                found = True
                break
        if not found: lines[word['top']] = [word]
    sorted_y = sorted(lines.keys())
    text = ""
    for y in sorted_y:
        line_words = sorted(lines[y], key=lambda w: w['x0'])
        line_str = ""
        last_x1 = 0
        for w in line_words:
            if last_x1 > 0 and (w['x0'] - last_x1) > 10: line_str += " "
            line_str += w['text']
            last_x1 = w['x1']
        text += line_str + "\n"
    return text

# --- LOADERS ---
def load_pdf(path: str, filename: str) -> List[Document]:
    logger.info(f"Reading PDF: {filename}...")
    norm_filename = normalize_source_key(filename)
    docs = []
    try:
        with fitz.open(path) as fitz_doc:
            with pdfplumber.open(path) as pdf:
                for page_num, page in enumerate(pdf.pages, start=1):
                    tables = page.find_tables()
                    table_bboxes = [t.bbox for t in tables]
                    for table in tables:
                        data = table.extract()
                        if data:
                            docs.append(Document(
                                page_content=convert_table_to_markdown(data),
                                metadata={"source": norm_filename, "page": page_num, "type": "table"}
                            ))
                    
                    words = page.extract_words(x_tolerance=3, y_tolerance=3)
                    non_table_words = [w for w in words if not is_inside_any_bbox(w, table_bboxes)]
                    body_text = clean_text(reconstruct_body_text(non_table_words))
                    
                    if len(body_text) < 100:
                        logger.warning(f"Page {page_num} appears scanned. Initiating OCR...")
                        fitz_page = fitz_doc[page_num - 1]
                        pix = fitz_page.get_pixmap(dpi=300)
                        img = Image.open(io.BytesIO(pix.tobytes()))
                        clean_img = preprocess_image_for_ocr(img)
                        ocr_text = pytesseract.image_to_string(clean_img, config="--psm 6 --oem 3")
                        body_text = clean_text(ocr_text)

                    if len(body_text) > 20:
                        docs.append(Document(
                            page_content=body_text,
                            metadata={"source": norm_filename, "page": page_num, "type": "text"}
                        ))
    except Exception as e:
        logger.error(f"PDF Processing Error: {e}")
    return docs

def load_spreadsheet(path: str, filename: str, is_csv: bool = False) -> List[Document]:
    docs = []
    norm_filename = normalize_source_key(filename)
    try:
        df = pd.read_csv(path) if is_csv else pd.read_excel(path)
        df.fillna("N/A", inplace=True)
        try:
            md_table = df.to_markdown(index=False)
        except ImportError:
            md_table = convert_table_to_markdown([df.columns.tolist()] + df.values.tolist())
        docs.append(Document(
            page_content=md_table,
            metadata={"source": norm_filename, "page": 1, "type": "table"}
        ))
    except Exception as e:
        logger.error(f"Spreadsheet Error: {e}")
    return docs

def load_txt(path: str, filename: str) -> List[Document]:
    docs = []
    try:
        with open(path, "r", encoding="utf-8") as f:
            docs.append(Document(page_content=clean_text(f.read()), metadata={"source": normalize_source_key(filename), "page": 1, "type": "text"}))
    except Exception as e:
        logger.error(f"TXT Error: {e}")
    return docs

def load_docx(path: str, filename: str) -> List[Document]:
    docs = []
    try:
        doc = docx.Document(path)
        # Extract from paragraphs
        parts = [para.text for para in doc.paragraphs if para.text.strip()]
        # Also extract text from table cells (Word often uses tables for layout)
        for table in doc.tables:
            for row in table.rows:
                seen = set()
                for cell in row.cells:
                    cell_text = cell.text.strip()
                    if cell_text and cell_text not in seen:
                        seen.add(cell_text)
                        parts.append(cell_text)
        docs.append(Document(page_content=clean_text("\n".join(parts)), metadata={"source": normalize_source_key(filename), "page": 1, "type": "text"}))
    except Exception as e:
        logger.error(f"DOCX Error: {e}")
    return docs

def load_image(path: str, filename: str) -> List[Document]:
    docs = []
    try:
        img = Image.open(path)
        clean_img = preprocess_image_for_ocr(img)
        ocr_text = pytesseract.image_to_string(clean_img, config="--psm 6 --oem 3")
        if len(ocr_text.strip()) > 10:
            docs.append(Document(page_content=clean_text(ocr_text), metadata={"source": normalize_source_key(filename), "page": 1, "type": "text"}))
    except Exception as e:
        logger.error(f"Image OCR Error: {e}")
    return docs

# --- INGESTION PIPELINE ---
def upload_in_batches(vectorstore, chunks, batch_size=50):
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i+batch_size]
        try:
            vectorstore.add_documents(batch)
            logger.info(f"Uploaded batch {i//batch_size + 1}")
        except Exception as e:
            logger.error(f"Batch {i//batch_size + 1} failed: {e}")
            raise 

def ingest_all_files():
    if not os.path.exists(DOCS_FOLDER):
        os.makedirs(DOCS_FOLDER)
        return False

    all_docs = []
    logger.info(f"📂 Scanning {DOCS_FOLDER} and all subfolders...")

    # Phase 1: Recursive Scanning & Loading
    for root, dirs, files in os.walk(DOCS_FOLDER):
        rel_path = os.path.relpath(root, DOCS_FOLDER)
        raw_category = "general" if rel_path == "." else rel_path.split(os.sep)[0].lower()
    
        # Validate Category against constants
        if raw_category != "general" and raw_category not in VALID_CATEGORIES:
            logger.warning(f"⚠️ Unknown folder '{raw_category}' detected. Tagging as 'general'.")
            category = "general"
        else:
            category = raw_category

        # FIXED: This loop is now outside the 'else' so 'general' files are actually processed
        for filename in files:
            if is_already_ingested(filename):
                logger.info(f"Skipping {filename} — active in Pinecone.")
                continue
            
            file_path = os.path.join(root, filename)
            ext = filename.lower().split('.')[-1]
            
            file_docs = []
            if ext == 'pdf': file_docs = load_pdf(file_path, filename)
            elif ext in ['xlsx', 'xls', 'csv']: file_docs = load_spreadsheet(file_path, filename, ext == 'csv')
            elif ext == 'txt': file_docs = load_txt(file_path, filename)
            elif ext in ['doc', 'docx']: file_docs = load_docx(file_path, filename)
            elif ext in ['jpg', 'jpeg', 'png']: file_docs = load_image(file_path, filename)

            # Inject the category and timestamp
            for d in file_docs:
                d.metadata["category"] = category
                d.metadata["uploaded_at"] = int(datetime.utcnow().timestamp())
                
            all_docs.extend(file_docs)

    if not all_docs: return True

    # Phase 2: Atomic Chunking
    table_docs = [d for d in all_docs if d.metadata.get("type") == "table"]
    text_docs = [d for d in all_docs if d.metadata.get("type") == "text"]

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    split_text_docs = text_splitter.split_documents(text_docs)
    final_chunks = table_docs + split_text_docs

    chunk_counts = {}
    source_counters = defaultdict(int)
    for chunk in final_chunks:
        src = chunk.metadata.get("source", "unknown")
        chunk.metadata["chunk_index"] = source_counters[src]
        source_counters[src] += 1
        chunk_counts[src] = chunk_counts.get(src, 0) + 1

    # Phase 3: Upload & Cleanup
    try:
        vectorstore = get_vectorstore()
        logger.info(f"🚀 Uploading {len(final_chunks)} chunks to Pinecone...")
        
        # 1. Batched Upload
        upload_in_batches(vectorstore, final_chunks)
        logger.info("✅ Ingestion Complete!")
        
        # 2. Update Manifest Ledger
        for filename, count in chunk_counts.items():
            update_manifest(filename, count)

        # 3. Automatic Cache Wipe (Ensures immediate AI updates)
        invalidate_cache()
            
        # 4. FIXED DELETE LOGIC: Matches normalized keys to save space
        for root, dirs, files in os.walk(DOCS_FOLDER):
            for filename in files:
                # Build the same relative path key used in the manifest
                rel_file_path = os.path.relpath(os.path.join(root, filename), DOCS_FOLDER)
                norm_name = normalize_source_key(rel_file_path)
                
                if norm_name in chunk_counts:
                    file_path = os.path.join(root, filename)
                    try:
                        os.remove(file_path)
                        logger.info(f"🗑️ Storage Optimized: Removed {filename}")
                    except Exception as e:
                        logger.error(f"⚠️ Could not delete {filename}: {e}")
                        
        return True
        
    except Exception as e:
        logger.error(f"❌ Upload Failed. Local files preserved for retry. Error: {e}")
        return False


# --- LEDGER ---
def get_uploaded_files() -> dict:
    """Reads manifest from Supabase instead of local JSON."""
    try:
        response = supabase.table("manifest").select("*").execute()
        return {
            row["filename"]: {
                "chunks": row["chunks"],
                "status": row["status"],
                "uploaded_at": str(row["uploaded_at"])
            }
            for row in response.data
        }
    except Exception as e:
        logger.error(f"Failed to read manifest from Supabase: {e}")
        return {}

def update_manifest(filename: str, chunk_count: int):
    """Upserts a file record into Supabase manifest."""
    norm_filename = normalize_source_key(filename)
    try:
        supabase.table("manifest").upsert({
            "filename": norm_filename,
            "chunks": chunk_count,
            "status": "Active",
            "uploaded_at": datetime.utcnow().isoformat()
        }).execute()
    except Exception as e:
        logger.error(f"Failed to update manifest: {e}")

def remove_from_manifest(filename: str):
    """Deletes a file record from Supabase manifest."""
    norm_filename = normalize_source_key(filename)
    try:
        supabase.table("manifest").delete().eq("filename", norm_filename).execute()
    except Exception as e:
        logger.error(f"Failed to remove from manifest: {e}")

def delete_document(filename: str) -> Tuple[bool, str]:
    norm_filename = normalize_source_key(filename)
    try:
        pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))
        index = pc.Index(PINECONE_INDEX_NAME)
        index.delete(filter={"source": norm_filename})
        remove_from_manifest(norm_filename)
        return True, f"Successfully purged {norm_filename} chunks."
    except Exception as e:
        return False, f"Failed to delete {norm_filename}: {str(e)}"
    
def verify_sync() -> dict:
    """Compares the local manifest ledger against actual Pinecone metadata."""
    manifest = get_uploaded_files()
    manifest_sources = set(manifest.keys())
    
    try:
        pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))
        index = pc.Index(PINECONE_INDEX_NAME)
        
        # We fetch unique 'source' values from the index
        # Note: Depending on Pinecone version, you may need to use list_ids or a dummy query
        results = index.query(vector=[0]*1536, top_k=10000, include_metadata=True)
        pinecone_sources = {d['metadata']['source'] for d in results['matches']}
        
        return {
            "in_both": list(manifest_sources & pinecone_sources),
            "manifest_only": list(manifest_sources - pinecone_sources), # Ghost entries
            "pinecone_only": list(pinecone_sources - manifest_sources)  # Untracked vectors
        }
    except Exception as e:
        logger.error(f"Sync check failed: {e}")
        return {}
    
def check_pinecone_health() -> bool:
    """Checks Pinecone connectivity and index availability via describe_index_stats."""
    try:
        from pinecone import Pinecone
        pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))
        index = pc.Index(PINECONE_INDEX_NAME)
        stats = index.describe_index_stats()
        logger.info(f"✅ Pinecone healthy: {stats.total_vector_count} vectors indexed")
        return True
    except Exception as e:
        logger.error(f"🚨 Pinecone Health Check Failed: {e}")
        return False