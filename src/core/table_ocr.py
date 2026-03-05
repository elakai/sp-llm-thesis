"""
table_ocr.py — OpenCV-based table cell segmentation and OCR reconstruction.

Pipeline for image-embedded PDFs (e.g. scanned handbooks) where pymupdf4llm
and pdfplumber return no text because there is no text layer.
"""

import io
import cv2
import numpy as np
import pytesseract
from PIL import Image
from typing import List, Tuple

from src.config.settings import TESSERACT_CMD
from src.config.logging_config import logger

pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD

def page_to_image(fitz_page, dpi: int = 300) -> np.ndarray:
    pix = fitz_page.get_pixmap(dpi=dpi)
    img_bytes = pix.tobytes("png")
    pil_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    return np.array(pil_img)

def binarize(img: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return thresh

def detect_grid_lines(thresh: np.ndarray, h_min_frac: float = 0.04, v_min_frac: float = 0.02) -> Tuple[np.ndarray, np.ndarray]:
    img_h, img_w = thresh.shape
    h_len = max(40, int(img_w * h_min_frac))
    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (h_len, 1))
    h_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, h_kernel, iterations=2)
    h_lines = cv2.dilate(h_lines, cv2.getStructuringElement(cv2.MORPH_RECT, (1, 3)))

    v_len = max(30, int(img_h * v_min_frac))
    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, v_len))
    v_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, v_kernel, iterations=2)
    v_lines = cv2.dilate(v_lines, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 1)))
    return h_lines, v_lines

def find_cells(h_lines: np.ndarray, v_lines: np.ndarray, img_h: int, img_w: int, min_cell_area: int = 400, min_cell_w: int = 18, min_cell_h: int = 10) -> List[Tuple[int, int, int, int]]:
    grid = cv2.add(h_lines, v_lines)
    grid = cv2.dilate(grid, np.ones((3, 3), np.uint8), iterations=2)
    cell_mask = cv2.bitwise_not(grid)

    for corner in [(0, 0), (img_w - 1, 0), (0, img_h - 1), (img_w - 1, img_h - 1)]:
        cv2.floodFill(cell_mask, None, corner, 0)

    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(cell_mask, connectivity=8)
    cells = []
    for i in range(1, num_labels):
        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]
        area = stats[i, cv2.CC_STAT_AREA]

        if area < min_cell_area or w < min_cell_w or h < min_cell_h:
            continue
        if w > img_w * 0.90 or h > img_h * 0.90:
            continue
        cells.append((x, y, w, h))
    return cells

def group_cells_into_tables(cells: List[Tuple[int, int, int, int]], gap_threshold_frac: float = 0.04, img_h: int = 3300) -> List[List[Tuple[int, int, int, int]]]:
    if not cells: return []
    gap_threshold = max(20, int(img_h * gap_threshold_frac))
    sorted_cells = sorted(cells, key=lambda c: c[1])

    groups: List[List[Tuple[int, int, int, int]]] = [[sorted_cells[0]]]
    for cell in sorted_cells[1:]:
        prev_cell = groups[-1][-1]
        prev_bottom = prev_cell[1] + prev_cell[3]
        current_top = cell[1]
        if current_top - prev_bottom > gap_threshold:
            groups.append([cell])
        else:
            groups[-1].append(cell)
    return [g for g in groups if len(g) >= 2]

def fill_spanning_cells(all_cells: List[Tuple[int, int, int, int]], rows: List[List[Tuple[int, int, int, int]]], span_ratio: float = 1.6) -> List[List[Tuple[int, int, int, int]]]:
    if not rows or not all_cells: return rows
    row_heights = [max(c[3] for c in row) for row in rows if row]
    if not row_heights: return rows

    row_heights.sort()
    median_row_h = row_heights[len(row_heights) // 2]
    if median_row_h == 0: return rows

    threshold = span_ratio * median_row_h
    spanning = [c for c in all_cells if c[3] > threshold]
    if not spanning: return rows

    def row_y_range(row):
        return min([c[1] for c in row]), max([c[1] + c[3] for c in row])

    for span_cell in spanning:
        sx, sy, sw, sh = span_cell
        span_top, span_bot = sy, sy + sh

        overlapping_row_indices = []
        for idx, row in enumerate(rows):
            ry0, ry1 = row_y_range(row)
            if ry0 < span_bot and ry1 > span_top:
                overlapping_row_indices.append(idx)

        for idx in overlapping_row_indices:
            row = rows[idx]
            if span_cell in row: continue
            if any(abs(c[0] - sx) < sw * 0.5 for c in row): continue
            row.append(span_cell)

    for i in range(len(rows)):
        rows[i] = sorted(rows[i], key=lambda c: c[0])
    return rows

def cluster_into_rows(cells: List[Tuple[int, int, int, int]], y_tolerance: int = 15) -> List[List[Tuple[int, int, int, int]]]:
    if not cells: return []
    def y_center(c): return c[1] + c[3] / 2
    sorted_by_y = sorted(cells, key=y_center)

    rows: List[List[Tuple[int, int, int, int]]] = []
    current_row: List[Tuple[int, int, int, int]] = []
    row_y_avg: float = 0.0

    for cell in sorted_by_y:
        cy = y_center(cell)
        if not current_row:
            current_row.append(cell)
            row_y_avg = cy
        elif abs(cy - row_y_avg) <= y_tolerance:
            current_row.append(cell)
            row_y_avg = (row_y_avg * (len(current_row) - 1) + cy) / len(current_row)
        else:
            rows.append(sorted(current_row, key=lambda c: c[0]))
            current_row = [cell]
            row_y_avg = cy

    if current_row: rows.append(sorted(current_row, key=lambda c: c[0]))
    return rows

def ocr_cell(img_gray: np.ndarray, x: int, y: int, w: int, h: int, padding: int = 6, min_upscale_dim: int = 48) -> str:
    img_h, img_w = img_gray.shape
    x1, y1 = max(0, x), max(0, y)
    x2, y2 = min(img_w, x + w), min(img_h, y + h)

    cell_img = img_gray[y1:y2, x1:x2]
    if cell_img.size == 0: return ""

    if padding > 0:
        cell_img = cv2.copyMakeBorder(cell_img, padding, padding, padding, padding, cv2.BORDER_CONSTANT, value=255)

    ch, cw = cell_img.shape
    if ch < min_upscale_dim or cw < min_upscale_dim:
        scale = max(min_upscale_dim / ch, min_upscale_dim / cw, 2.0)
        cell_img = cv2.resize(cell_img, None, fx=scale, fy=scale, interpolation=cv2.INTER_LANCZOS4)

    psm = 7 if h < 35 else 6
    config = f"--psm {psm} --oem 3"
    text = pytesseract.image_to_string(cell_img, config=config)
    return text.strip().replace("\n", " ").replace("|", "/")

def rows_to_markdown(img_gray: np.ndarray, rows: List[List[Tuple[int, int, int, int]]]) -> str:
    if not rows: return ""
    first_col_x = min(row[0][0] for row in rows if row)
    col_tolerance = 20

    ocr_rows: List[List[str]] = []
    for row in rows:
        if not row or abs(row[0][0] - first_col_x) > col_tolerance: continue
        ocr_row = [ocr_cell(img_gray, x, y, w, h) for x, y, w, h in row]
        if any(cell.strip() for cell in ocr_row): ocr_rows.append(ocr_row)

    if len(ocr_rows) < 2: return ""
    max_cols = max(len(r) for r in ocr_rows)
    padded = [r + [""] * (max_cols - len(r)) for r in ocr_rows]

    header = padded[0]
    separator = ["---"] * max_cols
    body = padded[1:]

    lines = ["| " + " | ".join(header) + " |", "| " + " | ".join(separator) + " |"]
    for row in body: lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)

def table_bbox(cells: List[Tuple[int, int, int, int]]) -> Tuple[int, int, int, int]:
    x0 = min(c[0] for c in cells)
    y0 = min(c[1] for c in cells)
    x1 = max(c[0] + c[2] for c in cells)
    y1 = max(c[1] + c[3] for c in cells)
    return x0, y0, x1, y1

def extract_page_tables(fitz_page, dpi: int = 300) -> Tuple[List[str], List[Tuple[int, int, int, int]]]:
    md_tables: List[str] = []
    img_bboxes: List[Tuple[int, int, int, int]] = []
    try:
        img = page_to_image(fitz_page, dpi=dpi)
        img_h, img_w = img.shape[:2]

        thresh = binarize(img)
        h_lines, v_lines = detect_grid_lines(thresh)

        cells = find_cells(h_lines, v_lines, img_h, img_w)
        if not cells: return md_tables, img_bboxes

        table_groups = group_cells_into_tables(cells, img_h=img_h)
        if not table_groups: return md_tables, img_bboxes

        img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        for group in table_groups:
            rows = cluster_into_rows(group)
            rows = fill_spanning_cells(group, rows)
            md = rows_to_markdown(img_gray, rows)
            if md:
                bbox = table_bbox(group)
                md_tables.append(md)
                img_bboxes.append(bbox)
    except Exception as e:
        logger.error(f"table_ocr.extract_page_tables error: {e}", exc_info=True)

    return md_tables, img_bboxes