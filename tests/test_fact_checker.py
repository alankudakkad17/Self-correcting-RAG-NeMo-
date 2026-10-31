"""Web fact check (src/websearch.py, src/functions/fact_checker.py, verifier.apply_fact_checks). No network."""

import pytest

from src.functions.fact_checker import check_claim
from src.functions.verifier import apply_fact_checks
from src.state import Claim
from src.state import FactCheck
from src.state import Verdict
from src.websearch import TavilyClient
from src.websearch import TavilyError
from src.websearch import best_match
from src.websearch import match_score

RESULTS = [
    {"title": "Unrelated page", "url": "https://a.example", "content": "Bridges and tunnels of Europe.", "score": 0.9},
    {"title": "Field marshal (United Kingdom)", "url": "https://b.example",
     "content": "Field marshal has been the highest rank in the British Army since 1736.", "score": 0.8},
]


def test_match_score_exact_fuzzy_and_absent():
    text = RESULTS[1]["content"]
    assert match_score("highest rank in the British Army since 1736", text) == 100.0
    assert match_score("highest rank in the British Army since 1736.", text) == 100.0  # punctuation ignored
    assert match_score("had been the top rank in the British Army since 1736", text) >= 80  # light rewording
    assert match_score("The rank was abolished in 1802", text) < 80
    assert match_score("", text) == 0.0


def test_best_match_picks_the_matching_result():
    score, best = best_match("Field marshal has been the top rank since 1736.", "highest rank in the British Army",
                             RESULTS)
    assert score >= 80 and best["url"] == "https://b.example"
    assert best_match("anything", "", []) == (0.0, None)


class FakeSearch:

    def __init__(self, results=None, fail=False):
        self.results, self.fail, self.queries = results or [], fail, []

    async def search(self, query):
        self.queries.append(query)
        if self.fail:
            raise TavilyError("Tavily rate or plan limit reached (432)")
        return self.results


def claim(text, quote="", i=1):
    return Claim(claim_id=i, text=text, doc_id="Field_marshal", quote=quote)


async def test_check_claim_statuses():
    ok = await check_claim(FakeSearch(RESULTS), claim("Field marshal is the top rank since 1736",
                                                      "highest rank in the British Army since 1736"), 80)
    assert ok.status == "confirmed" and ok.url == "https://b.example" and ok.source_title.startswith("Field")
    miss = await check_claim(FakeSearch(RESULTS), claim("The rank was abolished in 1802"), 80)
    assert miss.status == "not_found" and miss.url == ""
    assert (await check_claim(FakeSearch([]), claim("x"), 80)).status == "no_results"
    err = await check_claim(FakeSearch(fail=True), claim("x"), 80)
    assert err.status == "error" and "432" in err.detail


async def test_search_query_is_the_claim_text():
    search = FakeSearch(RESULTS)
    await check_claim(search, claim("Tom Mandrake drew Martian Manhunter."), 80)
    assert search.queries == ["Tom Mandrake drew Martian Manhunter."]


async def test_missing_api_key_is_a_clear_error(monkeypatch, tmp_path):
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    with pytest.raises(TavilyError, match="TAVILY_API_KEY"):
        await TavilyClient(cache=False).search("anything")


def test_apply_fact_checks_downgrades_only_failed_supported_claims():
    verdicts = [Verdict(claim_id=0, label="supported"), Verdict(claim_id=1, label="supported"),
                Verdict(claim_id=2, label="supported"), Verdict(claim_id=3, label="contradicted"),
                Verdict(claim_id=4, label="supported"), Verdict(claim_id=5, label="supported")]
    checks = [FactCheck(claim_id=1, query="a", status="confirmed", score=99, url="https://ok"),
              FactCheck(claim_id=2, query="b", status="not_found", score=40, detail="no web result matches"),
              FactCheck(claim_id=3, query="c", status="no_results"),
              FactCheck(claim_id=4, query="d", status="error", detail="offline"),
              FactCheck(claim_id=5, query="e", status="no_results", detail="the web search found nothing")]
    out = {v.claim_id: v for v in apply_fact_checks(verdicts, checks)}
    assert out[0].label == "supported"  # claim 0 (answer check) is not web-checked
    assert out[1].label == "supported"
    assert out[2].label == "unsupported" and "Web check failed" in out[2].reason
    assert out[3].label == "contradicted"  # never upgraded or changed
    assert out[4].label == "supported"  # search unavailable: keep the model's verdict
    assert out[5].label == "unsupported"
