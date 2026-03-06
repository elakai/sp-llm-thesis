from src.config.constants import GREETING_KEYWORDS, OFF_TOPIC_KEYWORDS
from src.config.logging_config import logger


def route_query_fast(query: str) -> str:
    """Instant keyword-based intent classification."""
    q = query.lower().strip()
    if any(q.startswith(g) or q == g for g in GREETING_KEYWORDS):
        return "greeting"
    if any(kw in q for kw in OFF_TOPIC_KEYWORDS):
        return "off_topic"
    return "search"


def get_dynamic_k(query: str) -> int:
    """Returns retrieval depth based on query complexity."""
    q = query.lower()

    # Curriculum/subject queries need more chunks to cover all year levels
    curriculum_keywords = [
        'curriculum', 'subject', 'course', 'year', 'semester', 'units', 'prerequisite'
    ]
    if any(kw in q for kw in curriculum_keywords):
        return 20  # Need more chunks to cover all year levels

    # Comparison or multi-topic queries need more context
    complex_signals = [
        " difference ", " compare ", " vs ", " versus ",
        " list all ", " what are all ",
    ]
    if any(signal in q for signal in complex_signals):
        return 15

    # Thesis searches benefit from more results
    if any(kw in q for kw in ["thesis", "research", "manuscript", "capstone"]):
        return 15

    # Organizational / people queries need broader retrieval
    if any(kw in q for kw in [
        "dean", "chairperson", "chair", "faculty", "professor",
        "department", "who is", "who are", "staff", "organizational",
        "org structure", "instructor", "engr", "lab technician",
    ]):
        return 15

    # Default for simple factual questions
    return 12


def route_query(query: str) -> tuple:
    """
    Main router.  Returns (intent, filters, content_type, category_filter).
    Metadata filtering is disabled — the flat document structure makes
    keyword-based filters unreliable (false positives cause 0-result queries).
    Retrieval depth is now handled by get_dynamic_k() instead.
    """
    intent = route_query_fast(query)
    logger.info(f"Router | Intent: {intent}")
    return intent, None, "all", None