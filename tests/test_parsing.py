"""Claim / plan / verdict parsing (src/parsing.py)."""

from src.parsing import extract_json
from src.parsing import parse_draft
from src.parsing import parse_plan
from src.parsing import parse_verdicts
from src.state import ABSTAIN_TEXT
from src.state import Claim


def test_extract_json_handles_fences_prose_and_braces_in_strings():
    text = 'Sure! ```json\n{"answer": "a {b}", "claims": []}\n``` hope this helps {not json}'
    assert extract_json(text) == {"answer": "a {b}", "claims": []}


def test_extract_json_skips_invalid_object_and_finds_next():
    assert extract_json('{bad} then {"ok": 1}') == {"ok": 1}
    assert extract_json("no json here") is None


def test_parse_draft_numbers_claims_and_fixes_bracketed_ids():
    text = ('{"answer": "Spain", "claims": ['
            '{"text": "Ana was born in Toledo.", "doc_id": "[Ana_Ruiz]", "quote": "born in Toledo"},'
            '{"text": "Toledo is in Spain.", "doc_id": "toledo", "quote": "Toledo is a city in Spain"}]}')
    draft = parse_draft(text, {"Ana_Ruiz", "Toledo"})
    assert draft.answer == "Spain"
    assert [(c.claim_id, c.doc_id) for c in draft.claims] == [(1, "Ana_Ruiz"), (2, "Toledo")]


def test_parse_draft_keeps_unknown_citation_for_verifier_to_reject():
    draft = parse_draft('{"answer": "x", "claims": [{"text": "t", "doc_id": "Ghost", "quote": "q"}]}', {"Real"})
    assert draft.claims[0].doc_id == "Ghost"


def test_parse_draft_drops_malformed_claims():
    text = '{"answer": "x", "claims": ["just a string", {"doc_id": "D"}, {"text": "ok", "doc_id": "D"}]}'
    draft = parse_draft(text, {"D"})
    assert [c.text for c in draft.claims] == ["ok"] and draft.claims[0].claim_id == 1


def test_parse_draft_falls_back_to_abstention():
    assert parse_draft("I think it is Paris.", set()).answer == ABSTAIN_TEXT
    assert parse_draft('{"claims": []}', set()).answer == ABSTAIN_TEXT


def test_parse_plan_truncates_and_falls_back():
    assert parse_plan('{"sub_questions": ["a", "b", "c", "d", "e"]}', "Q").sub_questions == ["a", "b", "c", "d"]
    assert parse_plan("garbage", "Q").sub_questions == ["Q"]
    assert parse_plan('{"sub_questions": ["", "  "]}', "Q").sub_questions == ["Q"]


def test_parse_verdicts_defaults_missing_and_invalid_to_unsupported():
    claims = [Claim(claim_id=i, text="t", doc_id="d", quote="q") for i in (1, 2, 3)]
    text = ('{"verdicts": [{"claim_id": 1, "label": "Supported", "reason": "ok"},'
            '{"claim_id": 2, "label": "probably"}]}')
    verdicts = parse_verdicts(text, claims)
    assert [x.label for x in verdicts] == ["supported", "unsupported", "unsupported"]
    assert [x.claim_id for x in verdicts] == [1, 2, 3]


def test_parse_verdicts_garbage_means_nothing_is_supported():
    claims = [Claim(claim_id=1, text="t", doc_id="d", quote="q")]
    assert parse_verdicts("all good!", claims)[0].label == "unsupported"
