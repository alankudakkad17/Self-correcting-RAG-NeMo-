"""Wikipedia fact check (src/wikipedia.py, src/functions/fact_checker.py, verifier.apply_fact_checks). No network."""

from src.functions.fact_checker import check_claim
from src.functions.verifier import apply_fact_checks
from src.state import Claim
from src.state import FactCheck
from src.state import Verdict
from src.wikipedia import quote_match_score

ARTICLE = ("Field marshal is the highest rank in the British Army. It has been the highest rank since 1736, "
           "when it was introduced.")


def test_quote_match_exact_fuzzy_and_absent():
    assert quote_match_score("has been the highest rank since 1736", ARTICLE) == 100.0
    assert quote_match_score("has been the highest rank since 1736.", ARTICLE) == 100.0  # punctuation ignored
    assert quote_match_score("It had been the top rank since 1736", ARTICLE) >= 80  # light rewording
    assert quote_match_score("The rank was abolished in 1802", ARTICLE) < 80
    assert quote_match_score("", ARTICLE) == 0.0


class FakeWiki:

    def __init__(self, pages=None, fail=False):
        self.pages, self.fail = pages or {}, fail

    async def article_text(self, title):
        if self.fail:
            raise ConnectionError("offline")
        return self.pages.get(title)


def claim(quote, i=1):
    return Claim(claim_id=i, text="t", doc_id="Field_marshal", quote=quote)


async def test_check_claim_statuses():
    wiki = FakeWiki({"Field marshal": ARTICLE})
    assert (await check_claim(wiki, claim("highest rank since 1736"), "Field marshal", 80)).status == "confirmed"
    assert (await check_claim(wiki, claim("abolished in 1802"), "Field marshal", 80)).status == "not_found"
    assert (await check_claim(wiki, claim("x"), "Field marshal (archived copy)", 80)).status == "page_missing"
    assert (await check_claim(FakeWiki(fail=True), claim("x"), "Field marshal", 80)).status == "error"


def test_apply_fact_checks_downgrades_only_failed_supported_claims():
    verdicts = [Verdict(claim_id=0, label="supported"), Verdict(claim_id=1, label="supported"),
                Verdict(claim_id=2, label="supported"), Verdict(claim_id=3, label="contradicted"),
                Verdict(claim_id=4, label="supported")]
    checks = [FactCheck(claim_id=1, title="A", status="confirmed", score=99),
              FactCheck(claim_id=2, title="B", status="not_found", score=40, detail="quote not found"),
              FactCheck(claim_id=3, title="C", status="page_missing"),
              FactCheck(claim_id=4, title="D", status="error", detail="offline")]
    out = {v.claim_id: v for v in apply_fact_checks(verdicts, checks)}
    assert out[0].label == "supported"  # claim 0 (answer check) has no quote to check
    assert out[1].label == "supported"
    assert out[2].label == "unsupported" and "Wikipedia check failed for 'B'" in out[2].reason
    assert out[3].label == "contradicted"  # never upgraded or changed
    assert out[4].label == "supported"  # Wikipedia unreachable: keep the model's verdict


def test_user_agent_requires_contact(monkeypatch):
    import pytest

    from src.wikipedia import user_agent
    monkeypatch.delenv("WIKIPEDIA_CONTACT", raising=False)
    with pytest.raises(RuntimeError, match="WIKIPEDIA_CONTACT"):
        user_agent()
    monkeypatch.setenv("WIKIPEDIA_CONTACT", "https://example.org/me")
    assert user_agent() == "SelfCorrectingRAG/0.1 (https://example.org/me; research prototype)"
