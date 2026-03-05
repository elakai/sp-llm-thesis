import io
import os
import tempfile
from datetime import datetime
from collections import defaultdict
from typing import List

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.core.ingestion import (
    normalize_source_key,
    is_already_ingested,
    convert_table_to_markdown,
    clean_text,
    extract_docx_text,
    upload_in_batches,
    update_manifest,
    split_table_by_rows,
)
from src.core.retrieval import invalidate_cache
from src.config.constants import CHUNK_SIZE, CHUNK_OVERLAP
from src.config.logging_config import logger


def process_uploaded_file(uploaded_file, category: str) -> List[Document]:
    """
    Processes a Streamlit UploadedFile object entirely in memory.
    Writes to a temp file only for libraries that require a file path (pymupdf4llm, fitz).
    Deletes the temp file immediately after processing.
    Returns list of Documents with metadata.
    """
    filename = uploaded_file.name
    norm_filename = normalize_source_key(filename)
    file_bytes = uploaded_file.read()
    ext = filename.lower().rsplit('.', 1)[-1]
    docs = []

    # Use tempfile for PDF processing since pymupdf4llm/fitz need a file path.
    # The temp file is deleted immediately after processing.
    if ext == 'pdf':
        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tmp:
            tmp.write(file_bytes)
            tmp_path = tmp.name
        try:
            from src.core.ingestion import load_pdf
            docs = load_pdf(tmp_path, filename)
        finally:
            os.unlink(tmp_path)  # Always delete temp file

    elif ext in ('xlsx', 'xls', 'csv'):
        # pandas can read from BytesIO directly — no temp file needed
        import pandas as pd
        try:
            buffer = io.BytesIO(file_bytes)
            df = pd.read_csv(buffer) if ext == 'csv' else pd.read_excel(buffer)
            df.fillna("N/A", inplace=True)
            try:
                md_table = df.to_markdown(index=False)
            except ImportError:
                md_table = convert_table_to_markdown(
                    [df.columns.tolist()] + df.values.tolist()
                )
            docs.append(Document(
                page_content=md_table,
                metadata={"source": norm_filename, "page": 1, "type": "table"}
            ))
        except Exception as e:
            logger.error(f"Spreadsheet processing error: {e}")

    elif ext in ('doc', 'docx'):
        # XML-level extraction catches text boxes, shapes, SmartArt
        # that python-docx's doc.paragraphs misses
        try:
            full_text = clean_text(extract_docx_text(file_bytes))
            if full_text:
                docs.append(Document(
                    page_content=full_text,
                    metadata={"source": norm_filename, "page": 1, "type": "text"}
                ))
            else:
                logger.warning(f"No text extracted from {filename}")
        except Exception as e:
            logger.error(f"DOCX processing error: {e}")

    elif ext == 'txt':
        try:
            text = clean_text(file_bytes.decode('utf-8', errors='ignore'))
            if text:
                docs.append(Document(
                    page_content=text,
                    metadata={"source": norm_filename, "page": 1, "type": "text"}
                ))
        except Exception as e:
            logger.error(f"TXT processing error: {e}")

    elif ext in ('png', 'jpg', 'jpeg', 'tiff', 'bmp'):
        # OCR: extract text from images using Tesseract
        try:
            from PIL import Image
            import pytesseract
            from src.config.settings import TESSERACT_CMD
            pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD

            image = Image.open(io.BytesIO(file_bytes))
            extracted_text = clean_text(pytesseract.image_to_string(image))
            if extracted_text:
                docs.append(Document(
                    page_content=extracted_text,
                    metadata={"source": norm_filename, "page": 1, "type": "text"}
                ))
            else:
                logger.warning(f"OCR extracted no text from {filename}")
        except Exception as e:
            logger.error(f"Image OCR processing error: {e}")

    # Tag all docs with category and timestamp
    now = int(datetime.utcnow().timestamp())
    for d in docs:
        d.metadata["category"] = category
        d.metadata["uploaded_at"] = now

    return docs


def ingest_uploaded_files(uploaded_files: list, category: str) -> tuple:
    """
    Main ingestion function called from the admin dashboard.
    Takes a list of Streamlit UploadedFile objects and a category string.
    Returns (success: bool, summary_message: str)
    """
    if not uploaded_files:
        return False, "No files provided."

    all_docs = []
    skipped = []
    processed = []

    for uploaded_file in uploaded_files:
        filename = uploaded_file.name

        if is_already_ingested(filename):
            skipped.append(filename)
            logger.info(f"Skipping {filename} — already in Pinecone")
            continue

        logger.info(f"Processing: {filename} (category: {category})")
        docs = process_uploaded_file(uploaded_file, category)

        if docs:
            all_docs.extend(docs)
            processed.append(filename)
        else:
            logger.warning(f"No content extracted from {filename}")

    if not all_docs:
        if skipped:
            return True, f"All files already indexed: {', '.join(skipped)}"
        return False, "No content could be extracted from the uploaded files."

    # Two-phase atomic chunking
    table_docs = [d for d in all_docs if d.metadata.get("type") == "table"]
    text_docs = [d for d in all_docs if d.metadata.get("type") == "text"]

    # Split large tables row-by-row to stay within the 256 word-piece embedding limit.
    split_table_docs = []
    for doc in table_docs:
        split_table_docs.extend(split_table_by_rows(doc, max_rows=20))

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )
    split_text_docs = text_splitter.split_documents(text_docs)
    final_chunks = split_table_docs + split_text_docs

    # ── Context Header Injection ───────────────────────────────────────────
    # CRITICAL: all-MiniLM-L6-v2 only embeds page_content — metadata is
    # invisible to it.  Without a document-origin header, a raw faculty
    # table chunk won't semantically match "who are the ECE CpE faculty".
    # Applied AFTER splitting so every split chunk carries the header,
    # not just the first.  Curriculum chunks are exempt (load_pdf already
    # prepends "Program: BACHELOR OF SCIENCE IN ...").
    for chunk in final_chunks:
        if not chunk.page_content.startswith("Program:"):
            src = chunk.metadata.get("source", "")
            category_label = chunk.metadata.get("category", "general")
            src_label = src.rsplit(".", 1)[0].replace("_", " ").replace("-", " ")
            chunk.page_content = (
                f"[{category_label.upper()}] {src_label}\n\n{chunk.page_content}"
            )

    # Add chunk indices per source
    source_counters = defaultdict(int)
    chunk_counts = {}
    for chunk in final_chunks:
        src = chunk.metadata.get("source", "unknown")
        chunk.metadata["chunk_index"] = source_counters[src]
        source_counters[src] += 1
        chunk_counts[src] = chunk_counts.get(src, 0) + 1

    # Upload to Pinecone
    try:
        from src.config.settings import get_vectorstore
        vectorstore = get_vectorstore()
        logger.info(f"Uploading {len(final_chunks)} chunks to Pinecone...")
        upload_in_batches(vectorstore, final_chunks)

        # Update manifest
        for filename, count in chunk_counts.items():
            update_manifest(filename, count)

        # Wipe semantic cache so new content is immediately searchable
        invalidate_cache()

        summary = f"Successfully indexed {len(final_chunks)} chunks from {len(processed)} file(s)."
        if skipped:
            summary += f" Skipped {len(skipped)} already-indexed file(s)."

        logger.info(summary)
        return True, summary

    except Exception as e:
        logger.error(f"Pinecone upload failed: {e}")
        return False, f"Upload failed: {str(e)}"
