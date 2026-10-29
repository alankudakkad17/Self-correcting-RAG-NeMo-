"""NeMo Guardrails custom actions (loaded automatically from this config folder) plus their pure-Python logic.

The logic lives in plain functions so it can be unit-tested without Guardrails; each @action is a thin wrapper.
All checks here are deterministic (regex / token rules), including the unsafe-request check.
(A hosted NemoGuard content-safety model could replace that rule later via Guardrails' built-in
`content safety check input` flow.)

Injection patterns are written from general prompt-injection phrasing (role overrides, "ignore previous
instructions", fake system tags, answer forcing), not copied from the eval set's injection templates.
"""

import json
import re

from nemoguardrails.actions import action

# ---- prompt injection (questions and retrieved passages) ----

INJECTION_PATTERNS = [
    r"\bignore\b[^.]{0,40}\b(instructions?|rules|prompts?|documents?|context|sources?|question)\b",
    r"\bdisregard\b[^.]{0,40}\b(instructions?|rules|prompts?|documents?|context|sources?|question)\b",
    r"\bforget\b[^.]{0,30}\b(instructions?|rules|everything)\b",
    r"\b(system|developer|admin)\s*(override|prompt|message|mode|instruction)s?\b",
    r"\[\s*(system|assistant|inst)\s*\]",
    r"<\s*/?\s*(system|instructions?)\s*>",
    r"\bdebug mode\b",
    r"\byou are now\b",
    r"\bnew (policy|instructions?|rules?)\b",
    r"\b(reveal|print|show|output|repeat)\b[^.]{0,30}\b(system prompt|instructions|hidden prompt)\b",
    r"\b(instructions?|note|notes|policy|message|reminder)\s+(to|for)\s+(the\s+)?(ai|assistants?|language models?"
    r"|llms?|chatbots?|models?)\b",
    r"\b(reply|respond|answer|output)\s+(only|exactly)\s+(with\b|')",
    r"\b(every|any|all)\s+answers?\b[^.]{0,40}\bmust\b",
    r"\bdo not (answer|use the (documents|passages|context|sources))\b",
    r"\bthe (correct|verified|real|true) answer is\b",
]
_INJECTION_RE = [re.compile(p, re.IGNORECASE) for p in INJECTION_PATTERNS]
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+|\n+")


def find_injections(text: str) -> list[str]:
    """Sentences of `text` that look like instructions aimed at the model."""
    return [s for s in _SENTENCE_SPLIT.split(text) if s.strip() and any(p.search(s) for p in _INJECTION_RE)]


def strip_injections(text: str) -> tuple[str, list[str]]:
    """Remove injection sentences. Returns (clean text, removed sentences)."""
    removed = find_injections(text)
    if not removed:
        return text, []
    kept = [s for s in _SENTENCE_SPLIT.split(text) if s.strip() and s not in removed]
    return " ".join(kept), removed


# ---- off-topic requests (the system only answers factual questions about the corpus domain) ----

OFF_TOPIC_PATTERNS = [
    r"\b(write|compose|generate|draft)\b[^.?]{0,30}\b(poem|essay|story|song|lyrics|email|letter|code|program|script"
    r"|function|joke)\b",
    r"\btell me a joke\b",
    r"\b(help|fix|debug)\b[^.?]{0,20}\b(my )?(code|homework|program|bug)\b",
    r"\b(translate|summari[sz]e|paraphrase|rewrite)\b[^.?]{0,30}\b(this|the following|my)\b",
    r"^\s*(hi|hello|hey|how are you|good (morning|evening))\b",
    r"\b(what do you think|your opinion|should i|give me advice|recommend me)\b",
]
_OFF_TOPIC_RE = [re.compile(p, re.IGNORECASE) for p in OFF_TOPIC_PATTERNS]


def is_off_topic(text: str) -> bool:
    return any(p.search(text) for p in _OFF_TOPIC_RE)


# ---- unsafe requests (asking for help to cause harm) ----

UNSAFE_PATTERNS = [
    r"\bhow (do|can|could|would|should) (i|you|we|someone)\b[^?]{0,40}\b(make|build|create|cook|synthesi[sz]e|buy)"
    r"\b[^?]{0,30}\b(bomb|explosive|weapon|poison|meth|nerve agent|bioweapon|malware|ransomware)\b",
    r"\bhow (do|can|could|would|should) (i|you|we|someone)\b[^?]{0,30}\b(kill|murder|poison|hurt|stalk|kidnap)\b",
    r"\b(step[- ]by[- ]step|instructions|guide)\b[^?]{0,30}\b(to|for)\b[^?]{0,20}\b(hack|steal|kill|poison|"
    r"make a bomb|launder)\b",
    r"\bhow (do|can) i (hack|break) into\b",
    r"\b(kill|hurt) myself\b|\bsuicide method",
]
_UNSAFE_RE = [re.compile(p, re.IGNORECASE) for p in UNSAFE_PATTERNS]


def is_unsafe(text: str) -> bool:
    return any(p.search(text) for p in _UNSAFE_RE)


# ---- PII ----

PII_PATTERNS = {
    "EMAIL": r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b",
    "SSN": r"\b\d{3}-\d{2}-\d{4}\b",
    "CARD": r"\b(?:\d[ -]?){13,16}\b",
    "PHONE": r"(?<!\w)(?:\+\d{1,3}[ .-]?)?(?:\(\d{2,4}\)[ .-]?|\d{2,4}[ .-])\d{3,4}[ .-]\d{3,4}\b",
}
_PII_RE = {k: re.compile(p) for k, p in PII_PATTERNS.items()}


def _luhn_ok(candidate: str) -> bool:
    digits = [int(d) for d in re.sub(r"\D", "", candidate)][::-1]
    total = sum(d if i % 2 == 0 else (d * 2 - 9 if d * 2 > 9 else d * 2) for i, d in enumerate(digits))
    return total % 10 == 0


def redact_pii(text: str) -> tuple[str, list[str]]:
    """Replace emails, SSNs, card and phone numbers with [EMAIL] etc. Returns (text, kinds found).

    Card numbers must pass the Luhn checksum, so catalogue / ISBN-like numbers are left alone.
    """
    found = []
    for kind, pattern in _PII_RE.items():  # order matters: SSN/card before the looser phone pattern
        hits = [m for m in pattern.finditer(text) if kind != "CARD" or _luhn_ok(m.group(0))]
        if hits:
            found.append(kind)
            for m in reversed(hits):
                text = text[:m.start()] + f"[{kind}]" + text[m.end():]
    return text, found


# ---- output: groundedness and citation format ----

YES_NO = {"yes", "no"}


def _tokens(text: str) -> set[str]:
    return {t for t in re.findall(r"\w+", text.lower()) if len(t) > 2 or t.isdigit()}


def groundedness_problems(answer: str, report: dict) -> list[str]:
    """Why the final answer is not grounded (empty list = grounded).

    `report` = {"accepted": bool, "claims": [{"text", "doc_id", "quote", "label"}], "answer_label": str|None,
                "evidence_ids": [...]}
    """
    problems = []
    claims = report.get("claims", [])
    if not report.get("accepted"):
        problems.append("the Controller did not accept the answer")
    if not claims:
        problems.append("no claims back the answer")
    if any(c.get("label") != "supported" for c in claims):
        problems.append("some claims are not supported")
    if report.get("answer_label") not in (None, "supported"):
        problems.append("the claims do not establish the answer")
    answer_tokens = _tokens(answer)
    if answer.strip().lower() not in YES_NO and answer_tokens:
        claim_tokens = set().union(*(_tokens(c.get("text", "") + " " + c.get("quote", "")) for c in claims)) \
            if claims else set()
        if len(answer_tokens & claim_tokens) / len(answer_tokens) < 0.5:
            problems.append("the answer does not appear in its claims")
    if find_injections(answer):
        problems.append("the answer contains instruction-like text")
    return problems


def citation_problems(report: dict) -> list[str]:
    evidence = set(report.get("evidence_ids", []))
    problems = []
    for c in report.get("claims", []):
        if c.get("doc_id") not in evidence:
            problems.append(f"claim cites unknown passage {c.get('doc_id')!r}")
        if not c.get("quote", "").strip():
            problems.append(f"claim citing {c.get('doc_id')!r} has no quote")
        elif len(c["quote"]) > 400:
            problems.append(f"quote from {c.get('doc_id')!r} is longer than 400 characters")
    return problems


# ---- @action wrappers (names are used in rails.co) ----

@action(name="check_question_injection")
async def check_question_injection(text: str) -> bool:
    return not find_injections(text)


@action(name="check_question_topic")
async def check_question_topic(text: str) -> bool:
    return not is_off_topic(text)


@action(name="check_question_safety")
async def check_question_safety(text: str) -> bool:
    return not is_unsafe(text)


@action(name="sanitize_passage_text")
async def sanitize_passage_text(text: str) -> str:
    return strip_injections(text)[0]


@action(name="check_answer_grounded")
async def check_answer_grounded(text: str, report: str) -> bool:
    return not groundedness_problems(text, json.loads(report))


@action(name="check_citation_format")
async def check_citation_format(report: str) -> bool:
    return not citation_problems(json.loads(report))


@action(name="redact_answer_pii")
async def redact_answer_pii(text: str) -> str:
    return redact_pii(text)[0]
