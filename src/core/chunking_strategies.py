from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from typing import List
from src.core.document_classifier import DocumentType
from src.config.constants import CHUNK_SIZE, CHUNK_OVERLAP

def chunk_document(doc: Document, doc_type: DocumentType) -> List[Document]:
    if doc_type == DocumentType.CURRICULUM:
        return _chunk_curriculum(doc)
    elif doc_type == DocumentType.DIRECTORY:
        return _chunk_directory(doc)
    elif doc_type == DocumentType.CALENDAR: # ── NEW ──
        return _chunk_calendar(doc)
    else:
        return _chunk_narrative(doc)


def _chunk_directory(doc: Document) -> List[Document]:
    """
    Directory documents (orgs, faculty, rooms) must preserve section boundaries.
    Split ONLY on ## (section level) — never on ### (individual entry level).
    Each ## section becomes one chunk so the LLM always sees a complete list.
    """
    section_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=[("#", "h1"), ("##", "h2")],
        strip_headers=False
    )
    chunks = []
    for hchunk in section_splitter.split_text(doc.page_content):
        meta = {**doc.metadata, **hchunk.metadata}
        # Inject the section label into content so embeddings capture it
        section = hchunk.metadata.get("h2", "") or hchunk.metadata.get("h1", "")
        content = hchunk.page_content
        if section and section not in content[:100]:
            content = f"Section: {section}\n\n{content}"
        chunks.append(Document(page_content=content, metadata=meta))
    return chunks


def _chunk_curriculum(doc: Document) -> List[Document]:
    """
    Curriculum documents are handled by the existing curriculum splitter.
    This is a passthrough to maintain the existing behavior.
    """
    from src.core.curriculum_splitter import split_curriculum_by_section
    return split_curriculum_by_section(doc)


def _chunk_narrative(doc: Document) -> List[Document]:
    """
    Narrative documents (policies, handbooks, thesis) use standard
    header-aware splitting with character fallback.
    """
    md_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=[("#", "h1"), ("##", "h2"), ("###", "h3")],
        strip_headers=False
    )
    char_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
    )
    chunks = []
    for hchunk in md_splitter.split_text(doc.page_content):
        meta = {**doc.metadata, **hchunk.metadata}
        content = hchunk.page_content
        section = (hchunk.metadata.get("h2", "") or 
                   hchunk.metadata.get("h1", "") or "")
        if section and section not in content[:150]:
            content = f"Section: {section}\n\n{content}"
        if len(content) <= CHUNK_SIZE:
            chunks.append(Document(page_content=content, metadata=meta))
        else:
            for subchunk in char_splitter.split_text(content):
                chunks.append(Document(page_content=subchunk, metadata=meta))
    return chunks

def _chunk_calendar(doc: Document) -> List[Document]:
    """
    Calendars (Excel or PDF) must stay intact so chronological context isn't lost.
    We use massive 4000-character chunks with a large overlap.
    """
    char_splitter = RecursiveCharacterTextSplitter(
        chunk_size=4000, 
        chunk_overlap=500
    )
    return char_splitter.split_documents([doc])