import re
from typing import List
from langchain_core.documents import Document

# Matches ALL year heading formats found across curriculum files:
#   ## FIRST YEAR / ## SECOND YEAR (ARCH, BIO, EM)
#   ## 1st Year / ## 2nd Year / ## 3rd Year / ## 4th Year (CE, ECE)
#   ## Year 1 / ## Year 2 / ## Year 3 / ## Year 4 (MATH)
_YEAR_HEADING_RE = re.compile(
    r'^##\s+('
    r'(?:FIRST|SECOND|THIRD|FOURTH|FIFTH)\s+YEAR'   # ARCH/BIO/EM style
    r'|(?:1st|2nd|3rd|4th|5th)\s+[Yy]ear'           # CE/ECE style
    r'|[Yy]ear\s+\d+'                                # MATH style
    r')',
    re.MULTILINE
)

# Matches ALL semester/intersession heading formats:
#   ### First Semester / ### Second Semester (ARCH)
#   ### 1st Semester / ### 2nd Semester (BIO, EM, CE, ECE)
#   ### Semester 1 / ### Semester 2 (MATH)
#   ### Summer / ### Intersession
_SEM_HEADING_RE = re.compile(
    r'^###\s+('
    r'(?:First|Second|Third)\s+Semester'             # ARCH style
    r'|(?:1st|2nd|3rd)\s+(?:Semester|SEMESTER)'      # BIO/EM/CE/ECE style
    r'|Semester\s+\d+'                               # MATH style
    r'|Summer|Intersession|SUMMER'                   # Summer/Intersession
    r')',
    re.MULTILINE | re.IGNORECASE
)


def split_curriculum_by_section(doc: Document) -> List[Document]:
    """
    Splits a curriculum markdown Document at every year and semester heading.
    Each chunk gets a bold context header prepended so retrieval always returns
    correctly labelled sections regardless of where the chunk boundary falls.

    Supports all heading formats used across ADNU curriculum files:
      - FIRST/SECOND/... YEAR  (ARCH, BIO, EM)
      - 1st/2nd/... Year       (CE, ECE)
      - Year 1/2/3/4           (MATH)

    Returns the original doc unchanged if no year headings are detected.
    """
    text = doc.page_content

    # Only apply to curriculum files with year headings
    if not _YEAR_HEADING_RE.search(text):
        return [doc]

    # Split on year headings (keep delimiter with lookahead)
    year_split_pattern = re.compile(
        r'(?=^##\s+(?:'
        r'(?:FIRST|SECOND|THIRD|FOURTH|FIFTH)\s+YEAR'
        r'|(?:1st|2nd|3rd|4th|5th)\s+[Yy]ear'
        r'|[Yy]ear\s+\d+'
        r'))',
        re.MULTILINE
    )
    year_blocks = year_split_pattern.split(text)

    chunks = []
    for year_block in year_blocks:
        if not year_block.strip():
            continue

        # Extract year label from first line of block
        year_match = re.match(r'^##\s+([^\n]+)', year_block.strip(), re.IGNORECASE)
        year_label = year_match.group(1).strip() if year_match else None
        if not year_label:
            continue

        # Split year block on semester/intersession headings
        sem_split_pattern = re.compile(
            r'(?=^###\s+(?:'
            r'(?:First|Second|Third)\s+Semester'
            r'|(?:1st|2nd|3rd)\s+(?:Semester|SEMESTER)'
            r'|Semester\s+\d+'
            r'|Summer|Intersession|SUMMER'
            r'))',
            re.MULTILINE | re.IGNORECASE
        )
        sem_blocks = sem_split_pattern.split(year_block)

        for sem_block in sem_blocks:
            if not sem_block.strip():
                continue

            sem_match = re.match(r'^###\s+([^\n]+)', sem_block.strip(), re.IGNORECASE)
            sem_label = sem_match.group(1).strip() if sem_match else ""

            # Extract program name from filename e.g. "Curriculum_BS_ECE.md" → "BS ECE"
            src = doc.metadata.get("source", "")
            program_match = re.search(r'Curriculum[_\s]+(BS[_\s]+\w+)', src, re.IGNORECASE)
            program_label = program_match.group(1).replace("_", " ").upper() if program_match else ""

            if program_label and sem_label:
                header = f"**{program_label} — {year_label} — {sem_label}**\n\n"
            elif program_label:
                header = f"**{program_label} — {year_label}**\n\n"
            elif sem_label:
                header = f"**{year_label} — {sem_label}**\n\n"
            else:
                header = f"**{year_label}**\n\n"

            chunk_content = header + sem_block.strip()

            meta = dict(doc.metadata)
            meta["year"] = year_label
            meta["semester"] = sem_label

            chunks.append(Document(page_content=chunk_content, metadata=meta))

    return chunks if chunks else [doc]