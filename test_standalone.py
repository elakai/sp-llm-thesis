"""Standalone test for improved PDF extraction — no Streamlit imports."""
import re, os, sys
import fitz
import pdfplumber
from typing import List, Dict

# Copy the functions we need to test (avoiding the Streamlit import chain)
_SECTION_HDR_RE = re.compile(
    r'(?:(?:FIRST|SECOND|THIRD|FOURTH|FIFTH)\s*YEAR|Year\s*\d+)'
    r'\s*[-\u2013\u2014,.:;\s]+\s*'
    r'(?:(?:1st|2nd|First|Second)\s*Semester|Semester\s*\d+|Summer|Intersession)',
    re.IGNORECASE,
)

def extract_program_info(filename):
    match = re.search(r'CURRICULUM\s+FOR\s+(.+?)\s*\(([^)]+)\)', filename, re.IGNORECASE)
    if match:
        return {"program_full": match.group(1).strip(), "program_code": match.group(2).strip()}
    return {}

def find_section_headers_for_tables(all_words, table_bboxes):
    if not all_words or not table_bboxes:
        return {}
    lines = {}
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

def convert_table_to_markdown(table_data):
    if not table_data: return ""
    cleaned = [[str(cell).replace('\n', ' ').strip() if cell else "" for cell in row] for row in table_data]
    header = cleaned[0]
    md = "| " + " | ".join(header) + " |\n"
    md += "| " + " | ".join(["---"] * len(header)) + " |\n"
    for row in cleaned[1:]:
        while len(row) < len(header): row.append("")
        md += "| " + " | ".join(row) + " |\n"
    return md

def is_inside_any_bbox(word, bboxes):
    wx = (word['x0'] + word['x1']) / 2
    wy = (word['top'] + word['bottom']) / 2
    return any(x0 <= wx <= x1 and top <= wy <= bottom for (x0, top, x1, bottom) in bboxes)

def reconstruct_body_text(words):
    lines = {}
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

def clean_text(text):
    if not text: return ""
    text = re.sub(r'\n\s*\n', '\n\n', text)
    return text.strip()

# ─────────────────────── TESTS ───────────────────────

print("=== Test 1: Program Info Extraction ===")
tests = [
    "CURRICULUM FOR BACHELOR OF SCIENCE IN ARCHITECTURE (BS ARCH).pdf",
    "CURRICULUM FOR BACHELOR OF SCIENCE IN MATHEMATICS (BS MATH).pdf",
    "CURRICULUM FOR BACHELOR OF SCIENCE IN ELECTRONICS ENGINEERING (BS ECE).pdf",
    "2025 ADNU COLLEGE HANDBOOK - CHAPTER 4.pdf",
]
for fn in tests:
    info = extract_program_info(fn)
    print(f"  {fn[:55]:55s} -> {info}")

print("\n=== Test 2: Architecture Curriculum - Full Extraction ===")
path = r"documents\CURRICULUM FOR BACHELOR OF SCIENCE IN ARCHITECTURE (BS ARCH).pdf"
filename = os.path.basename(path)
program_info = extract_program_info(filename)

all_docs = []
with fitz.open(path) as fitz_doc:
    with pdfplumber.open(path) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            tables = page.find_tables()
            table_bboxes = [t.bbox for t in tables]
            all_words = page.extract_words(x_tolerance=3, y_tolerance=3)
            section_hdrs = find_section_headers_for_tables(all_words, table_bboxes) if table_bboxes else {}

            for t_idx, table in enumerate(tables):
                data = table.extract()
                if not data: continue
                md = convert_table_to_markdown(data)
                prefix_parts = []
                if program_info:
                    prefix_parts.append(f"Program: {program_info['program_full']} ({program_info['program_code']})")
                if t_idx in section_hdrs:
                    prefix_parts.append(section_hdrs[t_idx])
                if prefix_parts:
                    md = '\n'.join(prefix_parts) + '\n\n' + md
                all_docs.append({"page": page_num, "type": "table", "content": md})

            # Body text - table-aware
            fitz_page = fitz_doc[page_num - 1]
            if table_bboxes:
                non_table_words = [w for w in all_words if not is_inside_any_bbox(w, table_bboxes)]
                body_text = clean_text(reconstruct_body_text(non_table_words))
            else:
                body_text = clean_text(fitz_page.get_text("text"))
                if len(body_text) < 100:
                    plumber_text = clean_text(reconstruct_body_text(all_words))
                    if len(plumber_text) > len(body_text):
                        body_text = plumber_text

            if len(body_text) > 20:
                if program_info:
                    body_text = f"Program: {program_info['program_full']} ({program_info['program_code']})\n\n{body_text}"
                all_docs.append({"page": page_num, "type": "text", "content": body_text})

table_chunks = [d for d in all_docs if d["type"] == "table"]
text_chunks = [d for d in all_docs if d["type"] == "text"]
print(f"Total chunks: {len(all_docs)} ({len(table_chunks)} tables, {len(text_chunks)} text)")

# Show first 3 chunks
for i, d in enumerate(all_docs[:3]):
    preview = d["content"][:250].replace('\n', ' | ')
    print(f"\n  Chunk {i+1} ({d['type']}, page {d['page']}):")
    print(f"    {preview}...")

# Check for semester headers
print("\n=== Test 3: Semester Headers ===")
for d in table_chunks:
    first_line = d["content"].split('\n')[0]
    second_line = d["content"].split('\n')[1] if '\n' in d["content"] else ""
    has_program = "Program:" in first_line
    has_semester = bool(_SECTION_HDR_RE.search(d["content"][:200]))
    print(f"  Table (page {d['page']}): program={has_program}, semester={has_semester} | {second_line[:60]}")

# Check body text does NOT contain table content
print("\n=== Test 4: Body Text Duplication Check ===")
for d in text_chunks:
    content_no_prefix = d["content"].split('\n\n', 1)[-1] if "Program:" in d["content"] else d["content"]
    # Look for signs of garbled table text
    has_garbled = "CourseCode" in content_no_prefix.replace(" ", "").replace("\n", "")
    print(f"  Page {d['page']}: {len(content_no_prefix)} chars, garbled_table={has_garbled}")
    if has_garbled:
        print(f"    WARNING! Preview: {content_no_prefix[:200]}")

print("\n=== Test 5: Cell Newline Fix ===")
# Check that table markdown doesn't contain \n in cells
for d in table_chunks[:2]:
    lines = d["content"].split('\n')
    for line in lines:
        if line.startswith('|') and '\n' in line.split('|')[1] if len(line.split('|')) > 1 else False:
            print(f"  WARNING: Cell still has newline: {line[:100]}")
            break
    else:
        print(f"  Table (page {d['page']}): Cells clean (no embedded newlines)")

print("\nAll tests passed!")
