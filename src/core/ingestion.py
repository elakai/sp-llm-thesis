import os
import io
import tempfile
import pandas as pd
from datetime import datetime
from typing import Tuple
from collections import defaultdict
from pinecone import Pinecone
from PIL import Image
import pytesseract


from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter, MarkdownHeaderTextSplitter

from src.config.settings import PINECONE_INDEX_NAME, get_vectorstore
from src.config.constants import DOCS_FOLDER, CHUNK_SIZE, CHUNK_OVERLAP, VALID_CATEGORIES
from src.config.logging_config import logger
from src.core.retrieval import invalidate_cache
from src.core.feedback import supabase
from src.core.curriculum_splitter import split_curriculum_by_section

# ── NEW IMPORTS FROM REFACTOR ──
from src.core.file_parsers import (
    clean_text, extract_docx_text, preprocess_image_for_ocr, 
    post_process_ocr_text, convert_table_to_markdown, 
    split_table_by_rows, load_pdf, normalize_source_key,
    is_curriculum_file
)
from src.core.document_classifier import classify_document, DocumentType
from src.core.chunking_strategies import chunk_document

def is_already_ingested(filename: str) -> bool:
    norm_name = normalize_source_key(filename)
    try:
        response = supabase.table("manifest").select("status").eq("filename", norm_name).execute()
        return bool(response.data) and response.data[0]["status"] == "Active"
    except Exception as e:
        logger.error(f"Manifest check failed: {e}")
        return False

def process_uploaded_file(uploaded_file, category: str) -> list[Document]:
    filename = uploaded_file.name
    norm_filename = normalize_source_key(filename)
    file_bytes = uploaded_file.read()
    ext = filename.lower().rsplit('.', 1)[-1]
    docs = []

    if ext == 'pdf':
        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tmp:
            tmp.write(file_bytes)
            tmp_path = tmp.name
        try: docs = load_pdf(tmp_path, filename, norm_filename)
        except Exception as e: logger.error(f"PDF error: {e}")
        finally:
            if os.path.exists(tmp_path): os.unlink(tmp_path)

    elif ext in ('xlsx', 'xls', 'csv'):
        try:
            buffer = io.BytesIO(file_bytes)
            df = pd.read_csv(buffer) if ext == 'csv' else pd.read_excel(buffer)
            df.fillna("N/A", inplace=True)
            try: md_table = df.to_markdown(index=False)
            except ImportError: md_table = convert_table_to_markdown([df.columns.tolist()] + df.values.tolist())
            docs.append(Document(page_content=md_table, metadata={"source": norm_filename, "page": 1, "type": "table"}))
        except Exception as e: logger.error(f"Spreadsheet error: {e}")

    elif ext in ('doc', 'docx'):
        try:
            full_text = clean_text(extract_docx_text(file_bytes))
            if full_text: docs.append(Document(page_content=full_text, metadata={"source": norm_filename, "page": 1, "type": "text"}))
        except Exception as e: logger.error(f"DOCX error: {e}")

    elif ext in ('txt', 'md'):
        try:
            raw_text = file_bytes.decode('utf-8', errors='ignore')
            text = raw_text.strip() if ext == 'md' else clean_text(raw_text)
            if text: docs.append(Document(page_content=text, metadata={"source": norm_filename, "page": 1, "type": "markdown" if ext == "md" else "text"}))
        except Exception as e: logger.error(f"Text error: {e}")

    elif ext in ('png', 'jpg', 'jpeg', 'tiff', 'bmp'):
        try:
            image = Image.open(io.BytesIO(file_bytes))
            extracted_text = clean_text(pytesseract.image_to_string(preprocess_image_for_ocr(image)))
            if extracted_text: docs.append(Document(page_content=extracted_text, metadata={"source": norm_filename, "page": 1, "type": "text"}))
        except Exception as e: logger.error(f"OCR error: {e}")

    now = int(datetime.utcnow().timestamp())
    for d in docs:
        d.metadata["category"], d.metadata["uploaded_at"] = category, now

    return docs

def ingest_uploaded_files(uploaded_files: list, category: str) -> tuple:
    if not uploaded_files: return False, "No files provided."

    all_docs, skipped, processed = [], [], []
    for uploaded_file in uploaded_files:
        if is_already_ingested(uploaded_file.name):
            skipped.append(uploaded_file.name)
            continue
        docs = process_uploaded_file(uploaded_file, category)
        if docs:
            all_docs.extend(docs)
            processed.append(uploaded_file.name)

    if not all_docs:
        if skipped: return True, f"All files already indexed: {', '.join(skipped)}"
        return False, "No content extracted."

    table_docs = [d for d in all_docs if d.metadata.get("type") == "table"]
    text_docs  = [d for d in all_docs if d.metadata.get("type") in ("text", "markdown")]

    split_table_docs = []
    for doc in table_docs:
        split_table_docs.extend(split_table_by_rows(doc, max_rows=20))

    split_text_docs = []
    for doc in text_docs:
        doc_type = classify_document(
            source=doc.metadata.get("source", ""),
            content=doc.page_content
        )
        doc.metadata["doc_type"] = doc_type.value
        split_text_docs.extend(chunk_document(doc, doc_type))

    final_chunks = split_table_docs + split_text_docs


    for chunk in final_chunks:
        if not chunk.page_content.startswith("Program:") and not chunk.page_content.startswith("**"):
            src, cat = chunk.metadata.get("source", ""), chunk.metadata.get("category", "general")
            lbl = src.rsplit(".", 1)[0].replace("_", " ").replace("-", " ")
            chunk.page_content = f"[{cat.upper()}] {lbl}\n\n{chunk.page_content}"

    source_counters, chunk_counts = defaultdict(int), {}
    for chunk in final_chunks:
        src = chunk.metadata.get("source", "unknown")
        chunk.metadata["chunk_index"] = source_counters[src]
        source_counters[src] += 1
        chunk_counts[src] = chunk_counts.get(src, 0) + 1

    try:
        vectorstore = get_vectorstore()
        upload_in_batches(vectorstore, final_chunks)
        for filename, count in chunk_counts.items(): update_manifest(filename, count)
        invalidate_cache()
        return True, f"Indexed {len(final_chunks)} chunks from {len(processed)} file(s)."
    except Exception as e:
        logger.error(f"Pinecone upload failed: {e}")
        return False, str(e)

class LocalFileWrapper:
    def __init__(self, path: str, name: str):
        self.path, self.name = path, name
    def read(self):
        with open(self.path, 'rb') as f: return f.read()

def ingest_all_files():
    if not os.path.exists(DOCS_FOLDER): return False
    all_files_to_process = []
    for root, _, files in os.walk(DOCS_FOLDER):
        rel_path = os.path.relpath(root, DOCS_FOLDER)
        cat = "general" if rel_path == "." else rel_path.split(os.sep)[0].lower()
        if cat != "general" and cat not in VALID_CATEGORIES: cat = "general"

        for filename in files:
            if is_already_ingested(filename): continue
            all_files_to_process.append((LocalFileWrapper(os.path.join(root, filename), filename), cat))

    if not all_files_to_process: return True

    cat_groups = defaultdict(list)
    for fw, cat in all_files_to_process: cat_groups[cat].append(fw)
    for cat, files in cat_groups.items(): ingest_uploaded_files(files, cat)

    for fw, _ in all_files_to_process:
        try: os.remove(fw.path)
        except Exception: pass
    return True

def upload_in_batches(vectorstore, chunks, batch_size=50):
    for i in range(0, len(chunks), batch_size): vectorstore.add_documents(chunks[i:i + batch_size])

def get_uploaded_files() -> dict:
    try:
        response = supabase.table("manifest").select("*").execute()
        return {
            row["filename"]: {"chunks": row["chunks"], "status": row["status"], "uploaded_at": str(row["uploaded_at"])} 
            for row in response.data
        }
    except Exception: return {}

def update_manifest(filename: str, chunk_count: int):
    try:
        supabase.table("manifest").upsert({
            "filename": normalize_source_key(filename), "chunks": chunk_count,
            "status": "Active", "uploaded_at": datetime.utcnow().isoformat()
        }).execute()
    except Exception as e: logger.error(f"Manifest update failed: {e}")

def delete_document(filename: str) -> Tuple[bool, str]:
    norm = normalize_source_key(filename)
    try:
        pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))
        pc.Index(PINECONE_INDEX_NAME).delete(filter={"source": norm})
        supabase.table("manifest").delete().eq("filename", norm).execute()
        return True, f"Purged {norm}."
    except Exception as e: return False, str(e)

def purge_all_vectors() -> Tuple[bool, str]:
    try:
        pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))
        pc.Index(PINECONE_INDEX_NAME).delete(delete_all=True)
        try: supabase.table("manifest").delete().neq("filename", "").execute()
        except Exception: pass
        return True, "All vectors deleted."
    except Exception as e: return False, str(e)

def verify_sync() -> dict:
    manifest = get_uploaded_files()
    manifest_sources = set(manifest.keys())
    try:
        pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))
        results = pc.Index(PINECONE_INDEX_NAME).query(vector=[0]*384, top_k=10000, include_metadata=True)
        pinecone_sources = {d['metadata']['source'] for d in results['matches']}
        return {
            "in_both": list(manifest_sources & pinecone_sources),
            "manifest_only": list(manifest_sources - pinecone_sources),
            "pinecone_only": list(pinecone_sources - manifest_sources)
        }
    except Exception: return {}