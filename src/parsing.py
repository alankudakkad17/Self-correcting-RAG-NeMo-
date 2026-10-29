"""Turn model output into Pydantic objects. Models wrap JSON in prose or ``` fences; be lenient in, strict out.

Every parser returns a safe fallback instead of raising, so one malformed reply never crashes a run:
- plan:     the question itself as the only sub-question
- draft:    an abstaining draft with no claims (the Controller then retrieves more or abstains)
- verdicts: every claim "unsupported" (never silently accept an unverified claim)
"""

import json
import re

from pydantic import ValidationError

from src.state import ABSTAIN_TEXT
from src.state import Claim
from src.state import Draft
from src.state import Plan
from src.state import Verdict


def extract_json(text: str) -> dict | None:
    """First JSON object in the text (handles ```json fences and leading/trailing prose)."""
    text = re.sub(r"```(?:json)?", "", text)
    decoder = json.JSONDecoder()
    start = text.find("{")
    while start != -1:
        try:
            obj, _ = decoder.raw_decode(text, start)
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            pass
        start = text.find("{", start + 1)
    return None


def parse_plan(text: str, question: str) -> Plan:
    obj = extract_json(text) or {}
    subs = [s.strip() for s in obj.get("sub_questions", []) if isinstance(s, str) and s.strip()]
    return Plan(sub_questions=subs[:4] or [question])


def parse_draft(text: str, valid_doc_ids: set[str]) -> Draft:
    """Parse {"answer": str, "claims": [{"text", "doc_id", "quote"}]}. Claims are numbered from 1.

    A claim citing a doc_id that is not in the evidence is kept (the Verifier marks it unsupported), so a
    hallucinated citation is visible in the trace rather than silently dropped. `valid_doc_ids` only fixes
    the common formatting slip of wrapping the id in brackets.
    """
    obj = extract_json(text)
    if not obj or not isinstance(obj.get("answer"), str):
        return Draft(answer=ABSTAIN_TEXT)
    claims = []
    for raw in obj.get("claims") or []:
        if not isinstance(raw, dict) or not str(raw.get("text", "")).strip():
            continue  # skipped before numbering, so claim ids stay 1..n
        doc_id = str(raw.get("doc_id", "")).strip().strip("[]")
        if doc_id not in valid_doc_ids:
            doc_id = next((d for d in valid_doc_ids if d.lower() == doc_id.lower()), doc_id)
        try:
            claims.append(Claim(claim_id=len(claims) + 1, text=str(raw.get("text", "")).strip(), doc_id=doc_id,
                                quote=str(raw.get("quote", "")).strip()))
        except ValidationError:
            continue
    return Draft(answer=obj["answer"].strip() or ABSTAIN_TEXT, claims=claims)


def parse_verdicts(text: str, claims: list[Claim]) -> list[Verdict]:
    """One verdict per claim, in claim order. Missing or invalid entries default to "unsupported"."""
    obj = extract_json(text) or {}
    found = {}
    for raw in obj.get("verdicts") or []:
        if not isinstance(raw, dict):
            continue
        label = str(raw.get("label", "")).strip().lower()
        try:
            found[int(raw.get("claim_id"))] = Verdict(claim_id=int(raw["claim_id"]), label=label,
                                                       reason=str(raw.get("reason", ""))[:500])
        except (KeyError, TypeError, ValueError, ValidationError):
            continue
    return [found.get(c.claim_id) or Verdict(claim_id=c.claim_id, label="unsupported", reason="no valid verdict")
            for c in claims]
