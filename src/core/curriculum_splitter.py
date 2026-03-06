import re
from typing import List
from langchain_core.documents import Document


def split_curriculum_by_section(doc: Document) -> List[Document]:
    """
    Splits a curriculum markdown Document at every ## YEAR and ### Semester heading.
    Each chunk gets the year + semester injected at the top so retrieval
    always returns correctly labelled sections.

    Returns the original doc unchanged if no year headings are detected.
    """
    text = doc.page_content
    source = doc.metadata.get("source", "")

    # Detect if this is a curriculum file
    has_year_headings = bool(re.search(
        r'^##\s+(FIRST|SECOND|THIRD|FOURTH|FIFTH)\s+YEAR',
        text, re.MULTILINE | re.IGNORECASE
    ))
    if not has_year_headings:
        return [doc]

    # Split on ## YEAR headings (keep delimiter)
    year_blocks = re.split(r'(?=^##\s+(?:FIRST|SECOND|THIRD|FOURTH|FIFTH)\s+YEAR)',
                           text, flags=re.MULTILINE | re.IGNORECASE)

    chunks = []
    for year_block in year_blocks:
        if not year_block.strip():
            continue

        # Extract the year label from the first line
        year_match = re.match(
            r'^##\s+((?:FIRST|SECOND|THIRD|FOURTH|FIFTH)\s+YEAR[^\n]*)',
            year_block.strip(), re.IGNORECASE
        )
        year_label = year_match.group(1).strip() if year_match else "Unknown Year"

        # Split each year block further on ### Semester/Intersession/Summer headings
        sem_blocks = re.split(
            r'(?=^###\s+(?:First|Second|1st|2nd|Third|Summer|Intersession))',
            year_block, flags=re.MULTILINE | re.IGNORECASE
        )

        for sem_block in sem_blocks:
            if not sem_block.strip():
                continue

            # Extract semester label
            sem_match = re.match(
                r'^###\s+([^\n]+)',
                sem_block.strip(), re.IGNORECASE
            )
            sem_label = sem_match.group(1).strip() if sem_match else ""

            # Build context header
            if sem_label:
                header = f"**{year_label} — {sem_label}**\n\n"
            else:
                header = f"**{year_label}**\n\n"

            chunk_content = header + sem_block.strip()

            meta = dict(doc.metadata)
            meta["year"] = year_label
            meta["semester"] = sem_label

            chunks.append(Document(page_content=chunk_content, metadata=meta))

    return chunks if chunks else [doc]