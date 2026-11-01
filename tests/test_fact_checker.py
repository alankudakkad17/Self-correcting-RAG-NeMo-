"""Web fact check: search + LLM judge (src/functions/fact_checker.py), the Verifier's downgrade policy, and the
Tavily client's key handling. No network, no LLM."""

import pytest

from src.functions.fact_checker import to_fact_checks
from src.functions.verifier import apply_fact_checks
from src.parsing import parse_web_checks
from src.prompts import web_check_user_prompt
from src.state import Claim
from src.state import FactCheck
from src.state import RetrievedDoc
from src.state import Verdict
from src.websearch import TavilyClient
from src.websearch import TavilyError
from src.websearch import web_doc


def claim(i, text="t"):
    return Claim(claim_id=i, text=text, doc_id="d", quote="q")


def doc(i, url):
    return RetrievedDoc(doc_id=f"web_{i}", title=f"Page {i}", text=f"snippet {i}", score=1, rank=i, url=url)


def test_judge_labels_become_fact_checks_with_the_cited_source():
    searched = [(claim(1), [doc(1, "https://a"), doc(2, "https://b")]),  # judged supported, source 2
                (claim(2), [doc(1, "https://c")]),  # judged contradicted
                (claim(3), [doc(1, "https://d")]),  # judged not mentioned
                (claim(4), []),  # search found nothing: no judge needed
                (claim(5), "TavilyError: limit reached (432)"),  # search failed
                (claim(6), [doc(1, "https://e")])]  # judge forgot this claim
    judged = {1: ("supported", 2, "page says so"), 2: ("contradicted", 1, "says 1802"),
              3: ("not_mentioned", 0, "not covered")}
    out = {c.claim_id: c for c in to_fact_checks(searched, judged)}
    assert (out[1].status, out[1].url, out[1].source_title) == ("confirmed", "https://b", "Page 2")
    assert out[2].status == "contradicted" and out[2].detail == "says 1802"
    assert out[3].status == "not_found" and out[3].url == ""  # source 0 = none
    assert out[4].status == "no_results"
    assert out[5].status == "error" and "432" in out[5].detail
    assert out[6].status == "error"


def test_judge_failure_marks_judged_claims_as_errors():
    out = to_fact_checks([(claim(1), [doc(1, "https://a")]), (claim(2), [])], judged=None)
    assert [c.status for c in out] == ["error", "no_results"]


def test_parse_web_checks_keeps_only_valid_entries():
    text = ('```json {"checks": [{"claim_id": 1, "label": "Supported", "source": 2, "reason": "ok"},'
            '{"claim_id": 2, "label": "maybe", "source": 1}, {"claim_id": 9, "label": "supported"},'
            '{"claim_id": "x", "label": "supported"}, {"claim_id": 3, "label": "not_mentioned", "source": null}]} ```')
    assert parse_web_checks(text, [1, 2, 3]) == {1: ("supported", 2, "ok"), 3: ("not_mentioned", 0, "")}
    assert parse_web_checks("no json", [1]) == {}
    # seen live: the judge answered under "claims" instead of "checks"
    assert parse_web_checks('{"claims": [{"claim_id": 2, "label": "not_mentioned", "source": 0}]}', [2]) == \
        {2: ("not_mentioned", 0, "")}


def test_web_check_prompt_numbers_each_claims_snippets():
    prompt = web_check_user_prompt([(claim(1, "Heart endorsed Kramer."), [doc(1, "https://a"), doc(2, "https://b")])])
    assert "Claim 1: Heart endorsed Kramer." in prompt
    assert "[1] Page 1 (https://a): snippet 1" in prompt and "[2] Page 2 (https://b): snippet 2" in prompt


def test_web_doc_ids_are_stable():
    a = web_doc({"title": "T", "url": "https://a", "content": "text"}, 1)
    assert a.doc_id == web_doc({"title": "T", "url": "https://a", "content": "text"}, 2).doc_id
    assert a.doc_id.startswith("web_")


async def test_missing_api_key_is_a_clear_error(monkeypatch):
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    with pytest.raises(TavilyError, match="TAVILY_API_KEY"):
        await TavilyClient(cache=False).search("anything")


VERDICTS = [Verdict(claim_id=i, label="supported") for i in range(6)]
CHECKS = [FactCheck(claim_id=1, query="a", status="confirmed", url="https://ok"),
          FactCheck(claim_id=2, query="b", status="not_found", detail="not covered"),
          FactCheck(claim_id=3, query="c", status="no_results"),
          FactCheck(claim_id=4, query="d", status="error", detail="offline"),
          FactCheck(claim_id=5, query="e", status="contradicted", detail="says 1802")]


def test_downgrade_policy_requiring_web_confirmation():
    out = {v.claim_id: v.label for v in apply_fact_checks(VERDICTS, CHECKS, require_confirmation=True)}
    assert out == {0: "supported", 1: "supported", 2: "unsupported", 3: "unsupported", 4: "supported",
                   5: "unsupported"}  # claim 0 is never web-checked; errors never change a verdict


def test_downgrade_policy_only_contradictions():
    out = {v.claim_id: v.label for v in apply_fact_checks(VERDICTS, CHECKS, require_confirmation=False)}
    assert out == {0: "supported", 1: "supported", 2: "supported", 3: "supported", 4: "supported",
                   5: "unsupported"}


def test_downgrade_reason_names_the_web_status():
    out = apply_fact_checks([Verdict(claim_id=5, label="supported")], CHECKS)
    assert out[0].reason == "Web check (contradicted): says 1802"


def test_other_labels_are_never_changed():
    out = apply_fact_checks([Verdict(claim_id=1, label="contradicted")], CHECKS)
    assert out[0].label == "contradicted"
