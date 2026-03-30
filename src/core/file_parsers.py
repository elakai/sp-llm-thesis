import io
import re
import cv2
import numpy as np
import fitz
import pdfplumber
import pandas as pd
from PIL import Image
import pytesseract
from typing import List, Dict
from langchain_core.documents import Document
from src.config.settings import TESSERACT_CMD
from src.config.logging_config import logger

pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD

# ── MISSING UTILITY FUNCTIONS ADDED ──
def normalize_source_key(filename: str) -> str:
    return filename.strip().replace("\\", "/")

def is_curriculum_file(filename: str) -> bool:
    return bool(re.search(r'curriculum', filename, re.IGNORECASE))
# ──────────────────────────────────────

def clean_text(text: str) -> str:
    if not text: return ""
    return re.sub(r'\n\s*\n', '\n\n', text).strip()

def preprocess_image_for_ocr(pil_image: Image.Image) -> Image.Image:
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
    return re.sub(r' {2,}', ' ', text)

def extract_docx_text(file_bytes: bytes) -> str:
    import zipfile
    from lxml import etree
    W_NS = '{http://schemas.openxmlformats.org/wordprocessingml/2006/main}'
    A_NS = '{http://schemas.openxmlformats.org/drawingml/2006/main}'
    parts, seen = [], set()
    with zipfile.ZipFile(io.BytesIO(file_bytes)) as zf:
        for entry in zf.namelist():
            if not entry.endswith('.xml'): continue
            if not any(entry.startswith(p) for p in ['word/document', 'word/header', 'word/footer', 'word/diagrams/data']): continue
            try: tree = etree.fromstring(zf.read(entry))
            except Exception: continue
            for p in tree.iter(f'{W_NS}p'):
                line = ''.join([t.text for t in p.iter(f'{W_NS}t') if t.text]).strip()
                if line and line not in seen:
                    seen.add(line); parts.append(line)
            for t in tree.iter(f'{A_NS}t'):
                if t.text and t.text.strip() and t.text.strip() not in seen:
                    seen.add(t.text.strip()); parts.append(t.text.strip())
    return '\n'.join(parts)

def convert_table_to_markdown(table_data: list) -> str:
    if not table_data: return ""
    filled = []
    for row in table_data:
        filled_row = []
        last_val = ""
        for col_idx, cell in enumerate(row):
            cell_str = str(cell).replace('\n', ' ').strip() if cell is not None else ""
            
            if cell_str == "":
                # Only forward-fill Columns 0 and 1 (Code and Title)
                if col_idx < 2:
                    filled_row.append(last_val)
                else:
                    filled_row.append("None")
            else:
                last_val = cell_str
                filled_row.append(cell_str)
        filled.append(filled_row)
        
    if not filled: return ""
    max_cols = max(len(row) for row in filled)
    for row in filled:
        while len(row) < max_cols: row.append("")

    def _is_header_row(row: list) -> bool:
        for cell in row:
            try: float(str(cell).replace('-', '').replace(' ', '').replace('–', '')); return False
            except ValueError: continue
        return True

    header_rows, data_start = [], 0
    for i, row in enumerate(filled):
        if _is_header_row(row):
            header_rows.append(row); data_start = i + 1
        else: break

    if len(header_rows) > 1:
        merged_header = []
        for col_idx in range(max_cols):
            col_values = []
            for hrow in header_rows:
                val = hrow[col_idx] if col_idx < len(hrow) else ""
                if val and val not in col_values: col_values.append(val)
            merged_header.append(" — ".join(col_values))
        header = merged_header
    elif header_rows: header, data_start = header_rows[0], 1
    else: header, data_start = filled[0], 1

    md = "| " + " | ".join(header) + " |\n" + "| " + " | ".join(["---"] * len(header)) + " |\n"
    for row in filled[data_start:]: md += "| " + " | ".join(row) + " |\n"
    return md

def split_table_by_rows(doc: Document, max_rows: int = 20) -> List[Document]:
    lines = [l for l in doc.page_content.split('\n') if l.strip()]
    header_lines, data_lines, found_separator = [], [], False
    for line in lines:
        if not found_separator:
            header_lines.append(line)
            if line.strip().startswith('|') and '---' in line: found_separator = True
        else: data_lines.append(line)
            
    if len(data_lines) <= max_rows or not found_separator: return [doc]
        
    header, chunks = '\n'.join(header_lines), []
    for i in range(0, len(data_lines), max_rows):
        batch = data_lines[i:i + max_rows]
        chunks.append(Document(page_content=header + '\n' + '\n'.join(batch), metadata=dict(doc.metadata)))
    return chunks

def extract_program_info(filename: str) -> dict:
    match = re.search(r'CURRICULUM\s+FOR\s+(.+?)\s*\(([^)]+)\)', filename, re.IGNORECASE)
    return {"program_full": match.group(1).strip(), "program_code": match.group(2).strip()} if match else {}

def is_inside_any_bbox(word: dict, bboxes: list) -> bool:
    wx, wy = (word['x0'] + word['x1']) / 2, (word['top'] + word['bottom']) / 2
    return any(x0 <= wx <= x1 and top <= wy <= bottom for (x0, top, x1, bottom) in bboxes)

def reconstruct_body_text(words: list) -> str:
    lines: Dict[int, List[dict]] = {}
    for word in words:
        found = False
        for y_coord in lines.keys():
            if abs(word['top'] - y_coord) < 5:
                lines[y_coord].append(word); found = True; break
        if not found: lines[word['top']] = [word]
    text = ""
    for y in sorted(lines.keys()):
        line_words = sorted(lines[y], key=lambda w: w['x0'])
        line_str, last_x1 = "", 0
        for w in line_words:
            if last_x1 > 0 and (w['x0'] - last_x1) > 5: line_str += " "
            line_str += w['text']; last_x1 = w['x1']
        text += line_str + "\n"
    return text

def load_pdf(path: str, filename: str, norm_filename: str, header_margin_pct=0.08, footer_margin_pct=0.08) -> List[Document]:
    logger.info(f"Reading PDF: {filename}...")
    program_info = extract_program_info(filename)
    docs = []
    try:
        # ── THESIS FIX: CONTEXT MANAGER FOR MEMORY SAFETY ──
        with fitz.open(path) as fitz_doc:
            with pdfplumber.open(path) as pdf:
                for page_num, page in enumerate(pdf.pages, 1):
                    fitz_page = fitz_doc[page_num - 1]
                    width, height = page.width, page.height
                    
                    plumber_bbox = (0, height * header_margin_pct, width, height * (1 - footer_margin_pct))
                    cropped_page = page.crop(plumber_bbox)
                    
                    f_rect = fitz_page.rect
                    fitz_clip = fitz.Rect(f_rect.x0, f_rect.y0 + (f_rect.height * header_margin_pct), f_rect.x1, f_rect.y1 - (f_rect.height * footer_margin_pct))

                    tables = cropped_page.find_tables()
                    table_bboxes = [t.bbox for t in tables]
                    all_words = cropped_page.extract_words(x_tolerance=3, y_tolerance=3)

                    for table in tables:
                        data = table.extract()
                        if not data: continue
                        md = convert_table_to_markdown(data)
                        if not md.strip(): continue
                        prefix = f"Program: {program_info['program_full']} ({program_info['program_code']})\n\n" if program_info else ""
                        meta_t = {"source": norm_filename, "page": page_num, "type": "table"}
                        if program_info: meta_t["program_code"] = program_info.get("program_code", "")
                        docs.append(Document(page_content=prefix + md, metadata=meta_t))

                    if table_bboxes:
                        non_table_words = [w for w in all_words if not is_inside_any_bbox(w, table_bboxes)]
                        body_text = clean_text(reconstruct_body_text(non_table_words))
                    else:
                        body_text = clean_text(fitz_page.get_text("text", clip=fitz_clip))
                        if len(body_text) < 100:
                            plumber_text = clean_text(reconstruct_body_text(all_words))
                            if len(plumber_text) > len(body_text): body_text = plumber_text

                    if len(body_text) < 50 and not table_bboxes:
                        raw_fitz = fitz_page.get_text("text", clip=fitz_clip).strip()
                        if not raw_fitz or len(raw_fitz) < 50:
                            try:
                                logger.warning(f"Page {page_num} appears scanned. Running OCR...")
                                pix = fitz_page.get_pixmap(dpi=300, clip=fitz_clip)
                                pil_img = Image.open(io.BytesIO(pix.tobytes("png")))
                                ocr_text = pytesseract.image_to_string(preprocess_image_for_ocr(pil_img), config="--psm 3 --oem 3")
                                body_text = clean_text(post_process_ocr_text(ocr_text))
                            except Exception as e: logger.warning(f"OCR fallback failed: {e}")

                    body_text = re.sub(r'(?m)^\s*-?\s*\d+\s*-?\s*$', '', body_text)
                    body_text = re.sub(r'(?i)Ateneo de Naga University', '', body_text)
                    body_text = re.sub(r'(?i)College of Science, Engineering, and Architecture', '', body_text)
                    body_text = re.sub(r'\n{3,}', '\n\n', body_text).strip()

                    if len(body_text) > 20:
                        prefix = f"Program: {program_info['program_full']} ({program_info['program_code']})\n\n" if program_info else ""
                        meta = {"source": norm_filename, "page": page_num, "type": "text"}
                        if program_info: meta["program_code"] = program_info.get("program_code", "")
                        docs.append(Document(page_content=prefix + body_text, metadata=meta))
        # ───────────────────────────────────────────────────
        # Note: fitz_doc.close() is safely removed because the 'with' block handles it automatically!

    except Exception as e:
        logger.error(f"PDF Processing Error: {e}")
    return docs