"""Answer post-processing and prompt building (src/prompts.py)."""

from src.prompts import clean_answer
from src.prompts import verifier_user_prompt
from src.state import Claim
from src.state import RetrievedDoc


def test_clean_answer_strips_quotes_and_short_trailing_period():
    assert clean_answer(' "Kramer Guitars." ') == ("Kramer Guitars", False)
    assert clean_answer("1736") == ("1736", False)


def test_abstention_detected_case_insensitively():
    assert clean_answer("Insufficient evidence.") == ("Insufficient evidence", True)


def test_verifier_prompt_lists_passages_claims_and_claim_zero():
    docs = [RetrievedDoc(doc_id="Ed_Wood", title="Ed Wood", text="American filmmaker.", score=1.0, rank=1)]
    claims = [Claim(claim_id=1, text="Ed Wood was American.", doc_id="Ed_Wood", quote="American filmmaker")]
    prompt = verifier_user_prompt("What nationality was Ed Wood?", "American", claims, docs)
    assert "[Ed_Wood] Ed Wood: American filmmaker." in prompt
    assert '"claim_id": 1' in prompt and '"cites": "Ed_Wood"' in prompt
    assert 'Claim 0 (reasoning check): The answer to the question "What nationality was Ed Wood?" is: American' \
        in prompt
