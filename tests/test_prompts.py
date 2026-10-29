"""Answer post-processing and context formatting (src/prompts.py)."""

from src.prompts import clean_answer
from src.prompts import format_context
from src.state import RetrievedDoc


def test_clean_answer_strips_quotes_and_short_trailing_period():
    assert clean_answer(' "Kramer Guitars." ') == ("Kramer Guitars", False)
    assert clean_answer("1736") == ("1736", False)


def test_abstention_detected_case_insensitively():
    assert clean_answer("Insufficient evidence.") == ("Insufficient evidence", True)


def test_context_lists_doc_ids_for_citation():
    docs = [RetrievedDoc(doc_id="Ed_Wood", title="Ed Wood", text="American filmmaker.", score=1.0, rank=1)]
    assert format_context(docs) == "[Ed_Wood] Ed Wood\nAmerican filmmaker."
