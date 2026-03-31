import re
from enum import Enum

class DocumentType(Enum):
    CURRICULUM    = "curriculum"    # Tables of subjects, units, prerequisites
    DIRECTORY     = "directory"     # Lists of people, orgs, rooms — must keep sections intact
    NARRATIVE     = "narrative"     # Handbook prose, policies, thesis abstracts
    CALENDAR      = "calendar"
    SPREADSHEET   = "spreadsheet"   # Already handled as table type

DIRECTORY_SIGNALS = [
    "organization", "org", "faculty", "staff", "personnel",
    "directory", "roster", "room", "laboratory", "contact"
]

CURRICULUM_SIGNALS = [
    "curriculum", "prospectus", "subjects", "units", "prerequisite"
]

CALENDAR_SIGNALS = [
    "calendar", "schedule", "semester", "academic year", "dates", "deadlines"
]

def classify_document(source: str, content: str) -> DocumentType:
    src = source.lower()
    content_lower = content.lower()[:500] 

    if any(s in src for s in CALENDAR_SIGNALS) or any(s in content_lower for s in CALENDAR_SIGNALS):
        return DocumentType.CALENDAR

    if any(s in src for s in CURRICULUM_SIGNALS):
        return DocumentType.CURRICULUM

    if any(s in src for s in DIRECTORY_SIGNALS):
        return DocumentType.DIRECTORY

    # Content-based fallback for directories
    header_count = len(re.findall(r'^###\s', content, re.MULTILINE))
    content_length = len(content)
    if header_count >= 5 and content_length / max(header_count, 1) < 800:
        return DocumentType.DIRECTORY

    return DocumentType.NARRATIVE