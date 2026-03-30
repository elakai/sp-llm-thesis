import re
from enum import Enum

class DocumentType(Enum):
    CURRICULUM    = "curriculum"    # Tables of subjects, units, prerequisites
    DIRECTORY     = "directory"     # Lists of people, orgs, rooms — must keep sections intact
    NARRATIVE     = "narrative"     # Handbook prose, policies, thesis abstracts
    SPREADSHEET   = "spreadsheet"   # Already handled as table type

DIRECTORY_SIGNALS = [
    "organization", "org", "faculty", "staff", "personnel",
    "directory", "roster", "room", "laboratory", "contact"
]

CURRICULUM_SIGNALS = [
    "curriculum", "prospectus", "subjects", "units", "prerequisite"
]

def classify_document(source: str, content: str) -> DocumentType:
    src = source.lower()
    content_lower = content.lower()[:500]  # Check only the top of the document

    if any(s in src for s in CURRICULUM_SIGNALS):
        return DocumentType.CURRICULUM

    if any(s in src for s in DIRECTORY_SIGNALS):
        return DocumentType.DIRECTORY

    # Content-based fallback: if the document has many ### headers close together,
    # it's a directory (each entry has its own header)
    header_count = len(re.findall(r'^###\s', content, re.MULTILINE))
    content_length = len(content)
    
    if header_count >= 5 and content_length / max(header_count, 1) < 800:
        # Average section is short — this is a list document
        return DocumentType.DIRECTORY

    return DocumentType.NARRATIVE