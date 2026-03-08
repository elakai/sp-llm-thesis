import os
import io
import re
import cv2
import numpy as np
import fitz
import pdfplumber
import pandas as pd
from PIL import Image
import pytesseract
from datetime import datetime
from typing import List, Dict, Tuple
from pinecone import Pinecone
from collections import defaultdict
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from src.config.settings import PINECONE_INDEX_NAME, get_vectorstore, TESSERACT_CMD
from src.config.constants import DOCS_FOLDER, CHUNK_SIZE, CHUNK_OVERLAP
from src.config.logging_config import logger
from src.core.retrieval import invalidate_cache
from src.config.constants import VALID_CATEGORIES
from src.core.feedback import supabase
from src.core.curriculum_splitter import split_curriculum_by_section

pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD
logger.info(f"Tesseract CMD set to: {TESSERACT_CMD}")


# ─────────────────────────────────────────────────────────────────────────────
# UTILITIES
# ─────────────────────────────────────────────────────────────────────────────

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
    if not text:
        return ""
    text = re.sub(r'\n\s*\n', '\n\n', text)
    return text.strip()

def is_curriculum_file(filename: str) -> bool:
    return bool(re.search(r'curriculum', filename, re.IGNORECASE))


# ─────────────────────────────────────────────────────────────────────────────
# IMAGE / OCR HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def preprocess_image_for_ocr(pil_image: Image.Image) -> Image.Image:
    img = np.array(pil_image.convert("RGB"))
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return Image.fromarray(binary)

def post_process_ocr_text(text: str) -> str:
    if not text:
        return text
    text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)
    text = re.sub(r'([a-zA-Z0-9])\(', r'\1 (', text)
    text = re.sub(r'\)([a-zA-Z])', r') \1', text)
    text = re.sub(r'([.,;:])([A-Z])', r'\1 \2', text)
    text = re.sub(r' {2,}', ' ', text)
    return text


# ─────────────────────────────────────────────────────────────────────────────
# TABLE HELPERS
# ─────────────────────────────────────────────────────────────────────────────

_SECTION_HDR_RE = re.compile(
    r'(?:(?:FIRST|SECOND|THIRD|FOURTH|FIFTH)\s*YEAR|Year\s*\d+)'
    r'\s*[-\u2013\u2014,.:;\s]+\s*'
    r'(?:(?:1st|2nd|First|Second)\s*Semester|Semester\s*\d+|Summer|Intersession)',
    re.IGNORECASE,
)

def extract_program_info(filename: str) -> dict:
    match = re.search(r'CURRICULUM\s+FOR\s+(.+?)\s*\(([^)]+)\)', filename, re.IGNORECASE)
    if match:
        return {"program_full": match.group(1).strip(), "program_code": match.group(2).strip()}
    return {}

def find_section_headers_for_tables(all_words: list, table_bboxes: list) -> dict:
    if not all_words or not table_bboxes:
        return {}
    lines: Dict[int, List[dict]] = {}
    for w in all_words:
        y = round(w['top'])
        merged = False
        for ey in list(lines.keys()):
            if abs(y - ey) < 5:
                lines[ey].append(w)
                merged = True
                break
        if not merged:
            lines[y] = [w]
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
            if y >= table_top - 2:
                break
            if _SECTION_HDR_RE.search(text):
                best = text.strip()
        if best:
            result[idx] = best
    return result

def convert_table_to_markdown(table_data: list) -> str:
    if not table_data:
        return ""
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
    if not filled:
        return ""
    max_cols = max(len(row) for row in filled)
    for row in filled:
        while len(row) < max_cols:
            row.append("")
    for col_idx in range(max_cols):
        last_val = ""
        for row in filled:
            if row[col_idx] == "":
                row[col_idx] = last_val
            else:
                last_val = row[col_idx]

    def _is_header_row(row: list) -> bool:
        for cell in row:
            stripped = str(cell).replace('-', '').replace(' ', '').replace('–', '')
            try:
                float(stripped)
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
                if val and val not in col_values:
                    col_values.append(val)
            merged_header.append(" — ".join(col_values))
        header = merged_header
    elif header_rows:
        header = header_rows[0]
        data_start = 1
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
        if not found:
            lines[word['top']] = [word]
    text = ""
    for y in sorted(lines.keys()):
        line_words = sorted(lines[y], key=lambda w: w['x0'])
        line_str = ""
        last_x1 = 0
        for w in line_words:
            if last_x1 > 0 and (w['x0'] - last_x1) > 5:
                line_str += " "
            line_str += w['text']
            last_x1 = w['x1']
        text += line_str + "\n"
    return text


# ─────────────────────────────────────────────────────────────────────────────
# DOCX HELPER
# ─────────────────────────────────────────────────────────────────────────────

def extract_docx_text(file_bytes: bytes) -> str:
    import zipfile
    from lxml import etree
    W_NS = '{http://schemas.openxmlformats.org/wordprocessingml/2006/main}'
    A_NS = '{http://schemas.openxmlformats.org/drawingml/2006/main}'
    parts = []
    seen = set()
    with zipfile.ZipFile(io.BytesIO(file_bytes)) as zf:
        for entry in zf.namelist():
            if not entry.endswith('.xml'):
                continue
            if not any(entry.startswith(p) for p in [
                'word/document', 'word/header', 'word/footer', 'word/diagrams/data'
            ]):
                continue
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


# ─────────────────────────────────────────────────────────────────────────────
# LOADERS & PRE-PROCESSORS
# ─────────────────────────────────────────────────────────────────────────────

def load_pdf(path: str, filename: str, header_margin_pct=0.08, footer_margin_pct=0.08) -> List[Document]:
    """
    Reads a PDF, applies SPATIAL FILTERING to crop out headers and footers,
    extracts tables, extracts body text, and applies regex cleanup to remove page numbers.
    """
    logger.info(f"Reading PDF: {filename}...")
    norm_filename = normalize_source_key(filename)
    program_info = extract_program_info(filename)
    docs = []
    
    try:
        fitz_doc = fitz.open(path)
        with pdfplumber.open(path) as pdf:
            for page_num, page in enumerate(pdf.pages, 1):
                fitz_page = fitz_doc[page_num - 1]
                
                # ── 1. SPATIAL FILTERING (Cropping the Page) ──
                width = page.width
                height = page.height
                
                # Bounding box for pdfplumber (Left, Top, Right, Bottom)
                plumber_bbox = (
                    0, 
                    height * header_margin_pct, 
                    width, 
                    height * (1 - footer_margin_pct)
                )
                cropped_page = page.crop(plumber_bbox)
                
                # Clipping Rectangle for PyMuPDF (fitz)
                f_rect = fitz_page.rect
                fitz_clip = fitz.Rect(
                    f_rect.x0, 
                    f_rect.y0 + (f_rect.height * header_margin_pct), 
                    f_rect.x1, 
                    f_rect.y1 - (f_rect.height * footer_margin_pct)
                )

                # ── 2. EXTRACT TABLES (From cropped page) ──
                tables = cropped_page.find_tables()
                table_bboxes = [t.bbox for t in tables]
                all_words = cropped_page.extract_words(x_tolerance=3, y_tolerance=3)

                for table in tables:
                    data = table.extract()
                    if not data:
                        continue
                    md = convert_table_to_markdown(data)
                    if not md.strip():
                        continue
                    prefix = (f"Program: {program_info['program_full']} ({program_info['program_code']})\n\n"
                              if program_info else "")
                    meta_t = {"source": norm_filename, "page": page_num, "type": "table"}
                    if program_info:
                        meta_t["program_code"] = program_info.get("program_code", "")
                    docs.append(Document(page_content=prefix + md, metadata=meta_t))

                # ── 3. EXTRACT BODY TEXT ──
                if table_bboxes:
                    non_table_words = [w for w in all_words if not is_inside_any_bbox(w, table_bboxes)]
                    body_text = clean_text(reconstruct_body_text(non_table_words))
                else:
                    # Extract from fitz using the clipped region to avoid headers/footers
                    body_text = clean_text(fitz_page.get_text("text", clip=fitz_clip))
                    if len(body_text) < 100:
                        plumber_text = clean_text(reconstruct_body_text(all_words))
                        if len(plumber_text) > len(body_text):
                            body_text = plumber_text

                # ── 4. OCR FALLBACK (On Clipped Region Only) ──
                if len(body_text) < 50 and not table_bboxes:
                    raw_fitz = fitz_page.get_text("text", clip=fitz_clip).strip()
                    if not raw_fitz or len(raw_fitz) < 50:
                        try:
                            logger.warning(f"Page {page_num} appears scanned. Running Tesseract OCR...")
                            # OCR ONLY the body, skipping header/footer
                            pix = fitz_page.get_pixmap(dpi=300, clip=fitz_clip)
                            pil_img = Image.open(io.BytesIO(pix.tobytes("png")))
                            clean_img = preprocess_image_for_ocr(pil_img)
                            ocr_text = pytesseract.image_to_string(clean_img, config="--psm 3 --oem 3")
                            body_text = clean_text(post_process_ocr_text(ocr_text))
                        except Exception as ocr_err:
                            logger.warning(f"OCR fallback failed on page {page_num}: {ocr_err}")

                # ── 5. REGEX CLEANUP (Remove leftover artifacts) ──
                # Remove standalone numbers (like page numbers that barely survived the crop)
                body_text = re.sub(r'(?m)^\s*-?\s*\d+\s*-?\s*$', '', body_text)
                # Remove repetitive handbook headers just in case
                body_text = re.sub(r'(?i)Ateneo de Naga University', '', body_text)
                body_text = re.sub(r'(?i)College of Science, Engineering, and Architecture', '', body_text)
                # Fix spacing
                body_text = re.sub(r'\n{3,}', '\n\n', body_text).strip()

                # Add Document if valid
                if len(body_text) > 20:
                    prefix = (f"Program: {program_info['program_full']} ({program_info['program_code']})\n\n"
                              if program_info else "")
                    meta = {"source": norm_filename, "page": page_num, "type": "text"}
                    if program_info:
                        meta["program_code"] = program_info.get("program_code", "")
                    docs.append(Document(page_content=prefix + body_text, metadata=meta))

        fitz_doc.close()
    except Exception as e:
        logger.error(f"PDF Processing Error: {e}")
    return docs


def load_txt_or_md(path: str, filename: str) -> List[Document]:
    """Loads .txt and .md files."""
    docs = []
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            raw = f.read()
        text = clean_text(raw)
        if text:
            docs.append(Document(
                page_content=text,
                metadata={"source": normalize_source_key(filename), "page": 1, "type": "text"}
            ))
    except Exception as e:
        logger.error(f"TXT/MD Error ({filename}): {e}")
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
        logger.error(f"Spreadsheet Error ({filename}): {e}")
    return docs


def load_docx(path: str, filename: str) -> List[Document]:
    docs = []
    try:
        with open(path, 'rb') as f:
            file_bytes = f.read()
        full_text = clean_text(extract_docx_text(file_bytes))
        if full_text:
            docs.append(Document(
                page_content=full_text,
                metadata={"source": normalize_source_key(filename), "page": 1, "type": "text"}
            ))
    except Exception as e:
        logger.error(f"DOCX Error ({filename}): {e}")
    return docs


def load_image(path: str, filename: str) -> List[Document]:
    docs = []
    try:
        img = Image.open(path)
        clean_img = preprocess_image_for_ocr(img)
        ocr_text = pytesseract.image_to_string(clean_img, config="--psm 6 --oem 3")
        if len(ocr_text.strip()) > 10:
            docs.append(Document(
                page_content=clean_text(ocr_text),
                metadata={"source": normalize_source_key(filename), "page": 1, "type": "text"}
            ))
    except Exception as e:
        logger.error(f"Image OCR Error ({filename}): {e}")
    return docs


# ─────────────────────────────────────────────────────────────────────────────
# CHUNKING HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def upload_in_batches(vectorstore, chunks, batch_size=50):
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        try:
            vectorstore.add_documents(batch)
            logger.info(f"Uploaded batch {i // batch_size + 1}")
        except Exception as e:
            logger.error(f"Batch {i // batch_size + 1} failed: {e}")
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
        chunks.append(Document(
            page_content=header + '\n' + '\n'.join(batch),
            metadata=dict(doc.metadata)
        ))
    return chunks


# ─────────────────────────────────────────────────────────────────────────────
# MAIN INGESTION PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

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
            logger.warning(f"⚠️ Unknown folder '{raw_category}'. Tagging as 'general'.")
            category = "general"
        else:
            category = raw_category

        for filename in files:
            if is_already_ingested(filename):
                logger.info(f"Skipping {filename} — already indexed.")
                continue

            file_path = os.path.join(root, filename)
            ext = filename.lower().rsplit('.', 1)[-1]

            file_docs = []
            if ext == 'pdf':
                file_docs = load_pdf(file_path, filename)
            elif ext in ('xlsx', 'xls', 'csv'):
                file_docs = load_spreadsheet(file_path, filename, ext == 'csv')
            elif ext in ('txt', 'md'):
                file_docs = load_txt_or_md(file_path, filename)
            elif ext in ('doc', 'docx'):
                file_docs = load_docx(file_path, filename)
            elif ext in ('jpg', 'jpeg', 'png'):
                file_docs = load_image(file_path, filename)

            for d in file_docs:
                d.metadata["category"] = category
                d.metadata["uploaded_at"] = int(datetime.utcnow().timestamp())

            all_docs.extend(file_docs)

    if not all_docs:
        return True

    table_docs = [d for d in all_docs if d.metadata.get("type") == "table"]
    text_docs  = [d for d in all_docs if d.metadata.get("type") == "text"]

    split_table_docs = []
    for doc in table_docs:
        split_table_docs.extend(split_table_by_rows(doc, max_rows=20))

    curriculum_chunks = []
    regular_text_docs = []
    for doc in text_docs:
        src = doc.metadata.get("source", "")
        if is_curriculum_file(src):
            sections = split_curriculum_by_section(doc)
            logger.info(f"📚 Curriculum split: {src} → {len(sections)} sections")
            curriculum_chunks.extend(sections)
        else:
            regular_text_docs.append(doc)

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    split_regular_docs = text_splitter.split_documents(regular_text_docs)

    final_chunks = split_table_docs + curriculum_chunks + split_regular_docs

    for chunk in final_chunks:
        content = chunk.page_content
        if content.startswith("Program:") or content.startswith("**"):
            continue
        src = chunk.metadata.get("source", "")
        category_label = chunk.metadata.get("category", "general")
        src_label = src.rsplit(".", 1)[0].replace("_", " ").replace("-", " ")
        chunk.page_content = f"[{category_label.upper()}] {src_label}\n\n{content}"

    source_counters = defaultdict(int)
    chunk_counts = {}
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
                        logger.info(f"🗑️ Removed {filename}")
                    except Exception as e:
                        logger.error(f"Could not delete {filename}: {e}")
        return True

    except Exception as e:
        logger.error(f"❌ Upload Failed. Files preserved for retry. Error: {e}")
        return False


# ─────────────────────────────────────────────────────────────────────────────
# MANIFEST / LEDGER
# ─────────────────────────────────────────────────────────────────────────────

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
        logger.error(f"Failed to read manifest: {e}")
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
        return True, f"Successfully purged {norm_filename}."
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
        logger.info("🗑️ Purged ALL vectors and cleared manifest.")
        return True, "All vectors deleted from Pinecone."
    except Exception as e:
        return False, f"Purge failed: {str(e)}"

def verify_sync() -> dict:
    manifest = get_uploaded_files()
    manifest_sources = set(manifest.keys())
    try:
        pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))
        index = pc.Index(PINECONE_INDEX_NAME)
        results = index.query(vector=[0] * 384, top_k=10000, include_metadata=True)
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
        pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))
        index = pc.Index(PINECONE_INDEX_NAME)
        stats = index.describe_index_stats()
        logger.info(f"✅ Pinecone healthy: {stats.total_vector_count} vectors indexed")
        return True
    except Exception as e:
        logger.error(f"🚨 Pinecone Health Check Failed: {e}")
        return False