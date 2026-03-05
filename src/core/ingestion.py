import os
import io
import json
import re
import cv2
import numpy as np
import fitz
import pymupdf4llm
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
from src.core.table_ocr import extract_page_tables
from src.config.constants import VALID_CATEGORIES
from src.core.feedback import supabase

pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD
logger.info(f"Tesseract CMD set to: {TESSERACT_CMD}")

def normalize_source_key(filename: str) -> str:
    return filename.strip().replace("\\", "/")

def is_already_ingested(filename: str) -> bool:
    norm_name = normalize_source_key(filename)
    try:
        response = supabase.table("manifest").select("status").eq("filename", norm_name).execute()
        return bool(response.data) and response.data[0]["status"] == "Active"
    except Exception as e:
        logger.error(f"Manifest check failed: {e}")
        return False

def clean_text(text: str) -> str:
    if not text: return ""
    text = re.sub(r'\n\s*\n', '\n\n', text) 
    return text.strip()

def preprocess_image_for_ocr(pil_image: Image) -> Image:
    img = np.array(pil_image.convert("RGB"))
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return Image.fromarray(binary)

def post_process_ocr_text(text: str) -> str:
    if not text: return text
    text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)
    text = re.sub(r'([a-zA-Z0-9])\(', r'\1 (', text)
    text = re.sub(r'\)([a-zA-Z])', r') \1', text)
    text = re.sub(r'([.,;:])([A-Z])', r'\1 \2', text)
    text = re.sub(r' {2,}', ' ', text)
    return text

_SECTION_HDR_RE = re.compile(
    r'(?:(?:FIRST|SECOND|THIRD|FOURTH|FIFTH)\s*YEAR|Year\s*\d+)'
    r'\s*[-\u2013\u2014,.:;\s]+\s*'
    r'(?:(?:1st|2nd|First|Second)\s*Semester|Semester\s*\d+|Summer|Intersession)',
    re.IGNORECASE,
)

def extract_program_info(filename: str) -> dict:
    match = re.search(r'CURRICULUM\s+FOR\s+(.+?)\s*\(([^)]+)\)', filename, re.IGNORECASE)
    if match: return {"program_full": match.group(1).strip(), "program_code": match.group(2).strip()}
    return {}

def find_section_headers_for_tables(all_words: list, table_bboxes: list) -> dict:
    if not all_words or not table_bboxes: return {}
    lines: Dict[int, List[dict]] = {}
    for w in all_words:
        y = round(w['top'])
        merged = False
        for ey in list(lines.keys()):
            if abs(y - ey) < 5:
                lines[ey].append(w)
                merged = True
                break
        if not merged: lines[y] = [w]
    line_list = []
    for y in sorted(lines.keys()):
        words_in_line = sorted(lines[y], key=lambda w: w['x0'])
        text = ' '.join(w['text'] for w in words_in_line)
        line_list.append((y, text))
    result = {}
    for idx, bbox in enumerate(table_bboxes):
        table_top = bbox[1]
        best = None
        for y, text in line_list:
            if y >= table_top - 2: break
            if _SECTION_HDR_RE.search(text): best = text.strip()
        if best: result[idx] = best
    return result

def convert_table_to_markdown(table_data: list) -> str:
    if not table_data: return ""
    filled = []
    for row in table_data:
        filled_row = []
        last_val = ""
        for cell in row:
            if cell is None or str(cell).strip() == "":
                filled_row.append(last_val)
            else:
                last_val = str(cell).replace('\n', ' ').strip()
                filled_row.append(last_val)
        filled.append(filled_row)

    if not filled: return ""
    max_cols = max(len(row) for row in filled)
    for row in filled:
        while len(row) < max_cols:
            row.append("")

    def _is_header_row(row: list) -> bool:
        for cell in row:
            try:
                float(str(cell).replace('-', '').replace(' ', ''))
                return False 
            except ValueError:
                continue
        return True

    header_rows = []
    data_start = 0
    for i, row in enumerate(filled):
        if _is_header_row(row):
            header_rows.append(row)
            data_start = i + 1
        else:
            break

    if len(header_rows) > 1:
        merged_header = []
        for col_idx in range(max_cols):
            col_values = []
            for hrow in header_rows:
                val = hrow[col_idx] if col_idx < len(hrow) else ""
                if val and val not in col_values: col_values.append(val)
            merged_header.append(" — ".join(col_values))
        header = merged_header
    else:
        header = filled[0]
        data_start = 1

    md = "| " + " | ".join(header) + " |\n"
    md += "| " + " | ".join(["---"] * len(header)) + " |\n"
    for row in filled[data_start:]:
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
            if last_x1 > 0 and (w['x0'] - last_x1) > 5: line_str += " "
            line_str += w['text']
            last_x1 = w['x1']
        text += line_str + "\n"
    return text

def extract_docx_text(file_bytes: bytes) -> str:
    import zipfile
    from lxml import etree
    W_NS = '{http://schemas.openxmlformats.org/wordprocessingml/2006/main}'
    A_NS = '{http://schemas.openxmlformats.org/drawingml/2006/main}'
    parts = []
    seen = set()

    with zipfile.ZipFile(io.BytesIO(file_bytes)) as zf:
        for entry in zf.namelist():
            if not entry.endswith('.xml'): continue
            if not any(entry.startswith(p) for p in ['word/document', 'word/header', 'word/footer', 'word/diagrams/data']): continue
            try:
                tree = etree.fromstring(zf.read(entry))
            except Exception:
                continue

            for p in tree.iter(f'{W_NS}p'):
                runs = [t.text for t in p.iter(f'{W_NS}t') if t.text]
                line = ''.join(runs).strip()
                if line and line not in seen:
                    seen.add(line)
                    parts.append(line)

            for t in tree.iter(f'{A_NS}t'):
                if t.text and t.text.strip() and t.text.strip() not in seen:
                    seen.add(t.text.strip())
                    parts.append(t.text.strip())
    return '\n'.join(parts)

# --- LOADERS ---
def load_pdf(path: str, filename: str) -> List[Document]:
    logger.info(f"Reading PDF: {filename}...")
    norm_filename = normalize_source_key(filename)
    program_info = extract_program_info(filename)
    docs = []
    
    try:
        md_pages = pymupdf4llm.to_markdown(path, page_chunks=True)
        fitz_doc = fitz.open(path)

        for page_data in md_pages:
            page_text = page_data.get("text", "").strip()
            page_num = page_data.get("metadata", {}).get("page", 1)

            # ── OCR FALLBACK ROUTING ───────────────────────────────────────────
            if not page_text:
                fitz_page = fitz_doc[page_num - 1]
                raw_fitz_text = fitz_page.get_text("text").strip()
                
                if raw_fitz_text and len(raw_fitz_text) > 50:
                    # 🛡️ MIDDLEMAN FALLBACK: Text layer exists!
                    logger.info(f"Page {page_num}: pymupdf4llm empty but text layer exists. Using PyMuPDF fallback.")
                    page_text = clean_text(raw_fitz_text)
                else:
                    # 🚨 NUCLEAR FALLBACK: Genuinely scanned/image-only page.
                    try:
                        logger.warning(f"Page {page_num} appears scanned. Running CV2 table extraction + OCR...")
                        import cv2 as _cv2
                        import numpy as _np

                        # Step 1: OpenCV table detection
                        md_tables, img_bboxes = extract_page_tables(fitz_page, dpi=300)

                        for md_table in md_tables:
                            if md_table.strip():
                                prefix_parts = []
                                if program_info:
                                    prefix_parts.append(f"Program: {program_info['program_full']} ({program_info['program_code']})")
                                prefix = '\n'.join(prefix_parts) + '\n\n' if prefix_parts else ''
                                meta_t = {"source": norm_filename, "page": page_num, "type": "table"}
                                if program_info: meta_t["program_code"] = program_info.get("program_code", "")
                                docs.append(Document(page_content=prefix + md_table, metadata=meta_t))

                        # Step 2: Full-page OCR for body text, masking table regions
                        pix = fitz_page.get_pixmap(dpi=300)
                        img_np = _np.frombuffer(pix.samples, dtype=_np.uint8).copy()
                        img_np = img_np.reshape(pix.height, pix.width, pix.n)
                        if pix.n == 4:
                            img_np = _cv2.cvtColor(img_np, _cv2.COLOR_RGBA2RGB)

                        for (x0, y0, x1, y1) in img_bboxes:
                            img_np[y0:y1, x0:x1] = 255

                        from PIL import Image as _PIL_Image
                        pil_img = _PIL_Image.fromarray(img_np)
                        clean_img = preprocess_image_for_ocr(pil_img)
                        ocr_text = pytesseract.image_to_string(clean_img, config="--psm 3 --oem 3")
                        page_text = clean_text(post_process_ocr_text(ocr_text))

                    except Exception as ocr_err:
                        logger.warning(f"OCR fallback failed on page {page_num}: {ocr_err}")

            if not page_text:
                continue

            prefix_parts = []
            if program_info:
                prefix_parts.append(f"Program: {program_info['program_full']} ({program_info['program_code']})")
            prefix = '\n'.join(prefix_parts) + '\n\n' if prefix_parts else ''

            meta = {"source": norm_filename, "page": page_num, "type": "mixed_markdown"}
            if program_info:
                meta["program_code"] = program_info.get("program_code", "")

            docs.append(Document(page_content=prefix + page_text, metadata=meta))

        fitz_doc.close()

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
        docs.append(Document(page_content=md_table, metadata={"source": norm_filename, "page": 1, "type": "table"}))
    except Exception as e:
        logger.error(f"Spreadsheet Error: {e}")
    return docs

def load_txt(path: str, filename: str) -> List[Document]:
    docs = []
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            docs.append(Document(page_content=clean_text(f.read()), metadata={"source": normalize_source_key(filename), "page": 1, "type": "text"}))
    except Exception as e:
        logger.error(f"TXT Error: {e}")
    return docs

def load_docx(path: str, filename: str) -> List[Document]:
    docs = []
    try:
        with open(path, 'rb') as f:
            file_bytes = f.read()
        full_text = clean_text(extract_docx_text(file_bytes))
        if full_text:
            docs.append(Document(page_content=full_text, metadata={"source": normalize_source_key(filename), "page": 1, "type": "text"}))
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

def split_table_by_rows(doc: Document, max_rows: int = 20) -> List[Document]:
    lines = [l for l in doc.page_content.split('\n') if l.strip()]
    header_lines = []
    data_lines = []
    found_separator = False

    for line in lines:
        if not found_separator:
            header_lines.append(line)
            if line.strip().startswith('|') and '---' in line:
                found_separator = True
        else:
            data_lines.append(line)

    if len(data_lines) <= max_rows or not found_separator:
        return [doc]

    header = '\n'.join(header_lines)
    chunks = []
    for i in range(0, len(data_lines), max_rows):
        batch = data_lines[i:i + max_rows]
        chunk_content = header + '\n' + '\n'.join(batch)
        chunks.append(Document(
            page_content=chunk_content,
            metadata=dict(doc.metadata) 
        ))
    return chunks

def ingest_all_files():
    if not os.path.exists(DOCS_FOLDER):
        os.makedirs(DOCS_FOLDER)
        return False

    all_docs = []
    logger.info(f"📂 Scanning {DOCS_FOLDER} and all subfolders...")

    for root, dirs, files in os.walk(DOCS_FOLDER):
        rel_path = os.path.relpath(root, DOCS_FOLDER)
        raw_category = "general" if rel_path == "." else rel_path.split(os.sep)[0].lower()
        
        if raw_category != "general" and raw_category not in VALID_CATEGORIES:
            logger.warning(f"⚠️ Unknown folder '{raw_category}' detected. Tagging as 'general'.")
            category = "general"
        else:
            category = raw_category

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

            for d in file_docs:
                d.metadata["category"] = category
                d.metadata["uploaded_at"] = int(datetime.utcnow().timestamp())

            all_docs.extend(file_docs)

    if not all_docs: return True

    table_docs = [d for d in all_docs if d.metadata.get("type") == "table"]
    text_docs  = [d for d in all_docs if d.metadata.get("type") in ("text", "mixed_markdown")]

    split_table_docs = []
    for doc in table_docs:
        split_table_docs.extend(split_table_by_rows(doc, max_rows=20))

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    split_text_docs = text_splitter.split_documents(text_docs)
    final_chunks = split_table_docs + split_text_docs

    for chunk in final_chunks:
        if not chunk.page_content.startswith("Program:"):
            src = chunk.metadata.get("source", "")
            category_label = chunk.metadata.get("category", "general")
            src_label = src.rsplit(".", 1)[0].replace("_", " ").replace("-", " ")
            chunk.page_content = f"[{category_label.upper()}] {src_label}\n\n{chunk.page_content}"

    chunk_counts = {}
    source_counters = defaultdict(int)
    for chunk in final_chunks:
        src = chunk.metadata.get("source", "unknown")
        chunk.metadata["chunk_index"] = source_counters[src]
        source_counters[src] += 1
        chunk_counts[src] = chunk_counts.get(src, 0) + 1

    try:
        vectorstore = get_vectorstore()
        logger.info(f"🚀 Uploading {len(final_chunks)} chunks to Pinecone...")
        upload_in_batches(vectorstore, final_chunks)
        logger.info("✅ Ingestion Complete!")
        
        for filename, count in chunk_counts.items():
            update_manifest(filename, count)

        invalidate_cache()
            
        for root, dirs, files in os.walk(DOCS_FOLDER):
            for filename in files:
                if normalize_source_key(filename) in chunk_counts:
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

def purge_all_vectors() -> Tuple[bool, str]:
    try:
        pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))
        index = pc.Index(PINECONE_INDEX_NAME)
        index.delete(delete_all=True)
        try:
            supabase.table("manifest").delete().neq("filename", "").execute()
        except Exception:
            pass 
        logger.info("🗑️ Purged ALL vectors from Pinecone and cleared manifest.")
        return True, "All vectors deleted from Pinecone. Index is now empty."
    except Exception as e:
        return False, f"Purge failed: {str(e)}"
    
def verify_sync() -> dict:
    manifest = get_uploaded_files()
    manifest_sources = set(manifest.keys())
    try:
        pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))
        index = pc.Index(PINECONE_INDEX_NAME)
        results = index.query(vector=[0]*384, top_k=10000, include_metadata=True)
        pinecone_sources = {d['metadata']['source'] for d in results['matches']}
        
        return {
            "in_both": list(manifest_sources & pinecone_sources),
            "manifest_only": list(manifest_sources - pinecone_sources),
            "pinecone_only": list(pinecone_sources - manifest_sources)
        }
    except Exception as e:
        logger.error(f"Sync check failed: {e}")
        return {}
    
def check_pinecone_health() -> bool:
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