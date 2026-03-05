"""
table_ocr.py — OpenCV-based table cell segmentation and OCR reconstruction.

Pipeline for image-embedded PDFs (e.g. scanned handbooks) where pymupdf4llm
and pdfplumber return no text because there is no text layer:

  1. Render page at high DPI with PyMuPDF → RGB numpy array
  2. Otsu binarization → binary image
  3. Morphological open with wide/tall kernels → isolate horizontal/vertical lines
  4. Flood-fill corners to remove page border → connectedComponentsWithStats
     to find enclosed cell bounding boxes
  5. Group cells by large vertical gap → separate tables on the same page
  6. Cluster cells into rows by y-center proximity, sort each row left-to-right
  7. OCR each cell individually at high DPI with Tesseract PSM 7 (single-line)
  8. Assemble rows into a GFM pipe-table string

Entry point:
    extract_page_tables(fitz_page, dpi=300)
        → (List[str], List[Tuple[int,int,int,int]])
        Returns a list of Markdown table strings and a parallel list of
        (x0, y0, x1, y1) bounding boxes in image-pixel coordinates so the
        caller can mask those regions before running full-page OCR for body text.
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


# ─────────────────────────────────────────────────────────────────────────────
# 1.  PAGE → IMAGE
# ─────────────────────────────────────────────────────────────────────────────

def page_to_image(fitz_page, dpi: int = 300) -> np.ndarray:
    """Render a PyMuPDF page to an RGB numpy array at the given DPI."""
    pix = fitz_page.get_pixmap(dpi=dpi)
    img_bytes = pix.tobytes("png")
    pil_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    return np.array(pil_img)


# ─────────────────────────────────────────────────────────────────────────────
# 2.  BINARIZATION
# ─────────────────────────────────────────────────────────────────────────────

def binarize(img: np.ndarray) -> np.ndarray:
    """
    Convert RGB → grayscale → binary (255 = foreground / dark, 0 = background).
    Uses Otsu thresholding after a light Gaussian blur to reduce scan noise.
    The result is inverted so lines and text are white on a black background,
    which is what the morphological operations expect.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return thresh


# ─────────────────────────────────────────────────────────────────────────────
# 3.  GRID LINE DETECTION
# ─────────────────────────────────────────────────────────────────────────────

def detect_grid_lines(
    thresh: np.ndarray,
    h_min_frac: float = 0.04,
    v_min_frac: float = 0.02,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Morphologically extract horizontal and vertical table lines.

    h_min_frac: minimum horizontal line length as fraction of image width.
    v_min_frac: minimum vertical line length as fraction of image height.

    A kernel that is wider/taller than most text strokes but shorter than a
    table border selectively preserves only line-like structures.
    """
    img_h, img_w = thresh.shape

    # Horizontal lines: kernel is wide (min line length) × 1 pixel tall
    h_len = max(40, int(img_w * h_min_frac))
    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (h_len, 1))
    h_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, h_kernel, iterations=2)
    # Dilate vertically slightly to bridge 1-2 px gaps from scan artifacts
    h_lines = cv2.dilate(h_lines, cv2.getStructuringElement(cv2.MORPH_RECT, (1, 3)))

    # Vertical lines: kernel is 1 pixel wide × tall (min line length)
    v_len = max(30, int(img_h * v_min_frac))
    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, v_len))
    v_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, v_kernel, iterations=2)
    # Dilate horizontally slightly
    v_lines = cv2.dilate(v_lines, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 1)))

    return h_lines, v_lines


# ─────────────────────────────────────────────────────────────────────────────
# 4.  CELL DETECTION
# ─────────────────────────────────────────────────────────────────────────────

def find_cells(
    h_lines: np.ndarray,
    v_lines: np.ndarray,
    img_h: int,
    img_w: int,
    min_cell_area: int = 400,
    min_cell_w: int = 18,
    min_cell_h: int = 10,
) -> List[Tuple[int, int, int, int]]:
    """
    Return a list of (x, y, w, h) cell bounding boxes detected from the table grid.

    Strategy:
      1. Combine h_lines + v_lines → table grid mask.
      2. Dilate to close gaps in broken scan lines.
      3. Invert: the enclosed white regions are the cells.
      4. Flood-fill page corners to remove the large outer background region.
      5. connectedComponentsWithStats to label each enclosed region (cell).
      6. Filter by area, width and height minimums.
    """
    grid = cv2.add(h_lines, v_lines)
    # Close small gaps (scan noise, broken borders)
    grid = cv2.dilate(grid, np.ones((3, 3), np.uint8), iterations=2)

    # Invert: cells become white, lines and background become black
    cell_mask = cv2.bitwise_not(grid)

    # Flood-fill the four corners to erase the page background
    # (so we only see enclosed cell regions, not the infinite outer background)
    for corner in [(0, 0), (img_w - 1, 0), (0, img_h - 1), (img_w - 1, img_h - 1)]:
        cv2.floodFill(cell_mask, None, corner, 0)

    # Label connected components
    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(cell_mask, connectivity=8)

    cells = []
    for i in range(1, num_labels):  # 0 = background
        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]
        area = stats[i, cv2.CC_STAT_AREA]

        if area < min_cell_area:
            continue
        if w < min_cell_w or h < min_cell_h:
            continue
        # Reject full-page-wide components (page background fragments)
        if w > img_w * 0.90 or h > img_h * 0.90:
            continue

        cells.append((x, y, w, h))

    return cells


# ─────────────────────────────────────────────────────────────────────────────
# 5.  GROUP CELLS INTO SEPARATE TABLES (by large vertical gap)
# ─────────────────────────────────────────────────────────────────────────────

def group_cells_into_tables(
    cells: List[Tuple[int, int, int, int]],
    gap_threshold_frac: float = 0.04,
    img_h: int = 3300,
) -> List[List[Tuple[int, int, int, int]]]:
    """
    Split a flat list of cells into groups that represent separate tables.
    Cells sorted by y; a new group starts when the vertical gap between
    consecutive cells exceeds gap_threshold_frac × image height.

    Returns a list of cell groups. Each group is a list of (x,y,w,h) tuples.
    """
    if not cells:
        return []

    gap_threshold = max(20, int(img_h * gap_threshold_frac))
    sorted_cells = sorted(cells, key=lambda c: c[1])  # sort by y

    groups: List[List[Tuple[int, int, int, int]]] = [[sorted_cells[0]]]
    for cell in sorted_cells[1:]:
        prev_cell = groups[-1][-1]
        prev_bottom = prev_cell[1] + prev_cell[3]
        current_top = cell[1]
        if current_top - prev_bottom > gap_threshold:
            groups.append([cell])
        else:
            groups[-1].append(cell)

    # Discard singleton groups (likely noise, not tables)
    return [g for g in groups if len(g) >= 2]


# ─────────────────────────────────────────────────────────────────────────────
# 6.  FILL VERTICALLY MERGED (SPANNING) CELLS INTO ROWS
# ─────────────────────────────────────────────────────────────────────────────

def fill_spanning_cells(
    all_cells: List[Tuple[int, int, int, int]],
    rows: List[List[Tuple[int, int, int, int]]],
    span_ratio: float = 1.6,
) -> List[List[Tuple[int, int, int, int]]]:
    """
    Merged table cells span multiple rows vertically.  cluster_into_rows assigns
    such a cell to only one row (whichever row-cluster its y-center is closest to),
    leaving the other rows with a gap in that column.

    This function does a post-clustering pass:
      1. Estimate typical single-row cell height (median across all rows).
      2. Identify spanning cells: height > span_ratio × median row height.
      3. For each spanning cell, compute the y-range it covers.
      4. For every row whose y-range overlaps the spanning cell's y-range:
         a. Skip the row that already has the cell (it got it from clustering).
         b. Insert a copy of the spanning cell into every other overlapping row,
            preserving the original (x, y, w, h) so ocr_cell crops the full
            merged region and returns the same text for each row.
      5. Re-sort each row left-to-right after insertions.

    Returns the updated rows list.
    """
    if not rows or not all_cells:
        return rows

    # Estimate median single-row cell height from the row structure
    row_heights = []
    for row in rows:
        if row:
            row_h = max(c[3] for c in row)
            row_heights.append(row_h)
    if not row_heights:
        return rows

    row_heights.sort()
    median_row_h = row_heights[len(row_heights) // 2]
    if median_row_h == 0:
        return rows

    threshold = span_ratio * median_row_h

    # Identify spanning cells
    spanning = [c for c in all_cells if c[3] > threshold]
    if not spanning:
        return rows

    # Build row y-ranges for overlap testing
    def row_y_range(row):
        ys = [c[1] for c in row]
        ybs = [c[1] + c[3] for c in row]
        return min(ys), max(ybs)

    for span_cell in spanning:
        sx, sy, sw, sh = span_cell
        span_top = sy
        span_bot = sy + sh

        overlapping_row_indices = []
        for idx, row in enumerate(rows):
            ry0, ry1 = row_y_range(row)
            # Overlap if ranges intersect (not just touch)
            if ry0 < span_bot and ry1 > span_top:
                overlapping_row_indices.append(idx)

        for idx in overlapping_row_indices:
            row = rows[idx]
            # Skip if this row already contains the spanning cell
            if span_cell in row:
                continue
            # Skip if any cell in the row already occupies the same x region
            col_occupied = any(
                abs(c[0] - sx) < sw * 0.5
                for c in row
            )
            if col_occupied:
                continue
            row.append(span_cell)

    # Re-sort each row left-to-right by x
    for i in range(len(rows)):
        rows[i] = sorted(rows[i], key=lambda c: c[0])

    return rows


# ─────────────────────────────────────────────────────────────────────────────
# 7.  CLUSTER CELLS INTO ROWS
# ─────────────────────────────────────────────────────────────────────────────

def cluster_into_rows(
    cells: List[Tuple[int, int, int, int]],
    y_tolerance: int = 15,
) -> List[List[Tuple[int, int, int, int]]]:
    """
    Group cells into rows by y-center proximity, then sort each row left-to-right.

    Two cells are in the same row when their y-centers are within y_tolerance
    pixels of the running row average.  Returns a list of rows, each row being
    a list of (x, y, w, h) sorted by x (left → right).
    """
    if not cells:
        return []

    # Compute y-center for each cell, sort by it
    def y_center(c):
        return c[1] + c[3] / 2

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
            # Running average keeps the row anchor from drifting
            row_y_avg = (row_y_avg * (len(current_row) - 1) + cy) / len(current_row)
        else:
            rows.append(sorted(current_row, key=lambda c: c[0]))
            current_row = [cell]
            row_y_avg = cy

    if current_row:
        rows.append(sorted(current_row, key=lambda c: c[0]))

    return rows


# ─────────────────────────────────────────────────────────────────────────────
# 8.  PER-CELL OCR
# ─────────────────────────────────────────────────────────────────────────────

def ocr_cell(
    img_gray: np.ndarray,
    x: int,
    y: int,
    w: int,
    h: int,
    padding: int = 6,
    min_upscale_dim: int = 48,
) -> str:
    """
    OCR a single table cell.

    - Adds padding so characters near the border are not clipped.
    - Upscales very small cells so Tesseract has enough pixels to work with.
    - Uses PSM 7 (single line) for short cells, PSM 6 (uniform block) for taller ones.
    - Returns stripped text with internal newlines replaced by spaces.
    """
    img_h, img_w = img_gray.shape

    # Crop the exact cell boundary (no negative-offset slice — that captures
    # thick table-border lines from neighbouring cells which confuse Tesseract)
    x1 = max(0, x)
    y1 = max(0, y)
    x2 = min(img_w, x + w)
    y2 = min(img_h, y + h)

    cell_img = img_gray[y1:y2, x1:x2]
    if cell_img.size == 0:
        return ""

    # Add synthetic white padding so characters near the edge are not clipped
    if padding > 0:
        cell_img = cv2.copyMakeBorder(
            cell_img, padding, padding, padding, padding,
            cv2.BORDER_CONSTANT, value=255,
        )

    # Upscale very small cells for better OCR accuracy
    ch, cw = cell_img.shape
    if ch < min_upscale_dim or cw < min_upscale_dim:
        scale = max(min_upscale_dim / ch, min_upscale_dim / cw, 2.0)
        cell_img = cv2.resize(
            cell_img, None, fx=scale, fy=scale,
            interpolation=cv2.INTER_LANCZOS4
        )

    # PSM 7 = treat image as a single text line (good for short cells)
    # PSM 6 = uniform block of text (good for multi-line cells)
    psm = 7 if h < 35 else 6
    config = f"--psm {psm} --oem 3"
    text = pytesseract.image_to_string(cell_img, config=config)
    return text.strip().replace("\n", " ").replace("|", "/")  # sanitize pipe char


# ─────────────────────────────────────────────────────────────────────────────
# 9.  ROWS → GFM MARKDOWN TABLE
# ─────────────────────────────────────────────────────────────────────────────

def rows_to_markdown(img_gray: np.ndarray, rows: List[List[Tuple[int, int, int, int]]]) -> str:
    """
    Given a row-major cell grid, OCR every cell and assemble a GFM pipe table.
    The first row is treated as the header.

    Rows with zero cells or whose OCR output is entirely empty are skipped.
    Returns an empty string if fewer than 2 rows survive.
    """
    if not rows:
        return ""

    # Determine the x-coordinate of the first (leftmost) column so we can
    # filter out orphaned rows whose only cells are spanning-cell injections
    # from fill_spanning_cells (those rows start at a larger x and produce
    # misleading left-shifted content in the final markdown).
    first_col_x = min(row[0][0] for row in rows if row)
    col_tolerance = 20  # pixels — allow grid sub-pixel variance

    ocr_rows: List[List[str]] = []
    for row in rows:
        # Skip rows whose first cell is not aligned to the first column
        if not row or abs(row[0][0] - first_col_x) > col_tolerance:
            continue
        ocr_row = [ocr_cell(img_gray, x, y, w, h) for x, y, w, h in row]
        # Skip completely empty rows
        if any(cell.strip() for cell in ocr_row):
            ocr_rows.append(ocr_row)

    if len(ocr_rows) < 2:
        return ""

    # Normalise column count across all rows (pad with "" where cols are missing)
    max_cols = max(len(r) for r in ocr_rows)
    padded = [r + [""] * (max_cols - len(r)) for r in ocr_rows]

    header = padded[0]
    separator = ["---"] * max_cols
    body = padded[1:]

    lines = []
    lines.append("| " + " | ".join(header) + " |")
    lines.append("| " + " | ".join(separator) + " |")
    for row in body:
        lines.append("| " + " | ".join(row) + " |")

    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# 10.  TABLE BOUNDING BOX (image coords)
# ─────────────────────────────────────────────────────────────────────────────

def table_bbox(cells: List[Tuple[int, int, int, int]]) -> Tuple[int, int, int, int]:
    """Return the tight bounding box (x0, y0, x1, y1) for a group of cells."""
    x0 = min(c[0] for c in cells)
    y0 = min(c[1] for c in cells)
    x1 = max(c[0] + c[2] for c in cells)
    y1 = max(c[1] + c[3] for c in cells)
    return x0, y0, x1, y1


# ─────────────────────────────────────────────────────────────────────────────
# 11.  MAIN ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def extract_page_tables(
    fitz_page,
    dpi: int = 300,
) -> Tuple[List[str], List[Tuple[int, int, int, int]]]:
    """
    Main entry point.  Given a PyMuPDF page object, detect all tables, OCR
    each cell, and return structured Markdown.

    Returns:
        md_tables  : list of GFM Markdown table strings, one per detected table.
        img_bboxes : parallel list of (x0, y0, x1, y1) in image-pixel coordinates,
                     one per table.  Use these to mask table regions before running
                     full-page Tesseract for body text.

    Both lists are empty if no tables are detected on the page.
    """
    md_tables: List[str] = []
    img_bboxes: List[Tuple[int, int, int, int]] = []

    try:
        img = page_to_image(fitz_page, dpi=dpi)
        img_h, img_w = img.shape[:2]

        thresh = binarize(img)
        h_lines, v_lines = detect_grid_lines(thresh)

        cells = find_cells(h_lines, v_lines, img_h, img_w)
        if not cells:
            logger.debug("table_ocr: no cells detected on this page")
            return md_tables, img_bboxes

        table_groups = group_cells_into_tables(cells, img_h=img_h)
        if not table_groups:
            logger.debug("table_ocr: no table groups after filtering singletons")
            return md_tables, img_bboxes

        # Use grayscale image for cell OCR (faster, more consistent)
        img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        for group in table_groups:
            rows = cluster_into_rows(group)
            # Fill spanning (merged) cells into every row they cover
            rows = fill_spanning_cells(group, rows)
            md = rows_to_markdown(img_gray, rows)
            if md:
                bbox = table_bbox(group)
                md_tables.append(md)
                img_bboxes.append(bbox)
                logger.info(
                    f"table_ocr: extracted table with {len(rows)} rows, "
                    f"bbox={bbox}"
                )

    except Exception as e:
        logger.error(f"table_ocr.extract_page_tables error: {e}", exc_info=True)

    return md_tables, img_bboxes
