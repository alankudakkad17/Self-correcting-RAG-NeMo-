"""Pure helpers that turn API data into safe Streamlit markdown. Retrieved text is never rendered as HTML:
Streamlit escapes HTML by default, and these helpers also escape markdown so passage text cannot format the page.
"""

import re

from rapidfuzz import fuzz

VERDICT_COLORS = {"supported": "green", "unsupported": "orange", "contradicted": "red", None: "gray"}
WEB_LABELS = {"confirmed": "Web: confirmed", "not_found": "Web: not confirmed", "no_results": "Web: no results",
              "error": "Web: check failed", None: "Web: not checked"}

_MD_SPECIAL = re.compile(r"([\\`*_{}\[\]()#+\-.!|>~:$<])")


def escape_md(text: str) -> str:
    """Escape every markdown / Streamlit-directive character so text shows literally."""
    return _MD_SPECIAL.sub(r"\\\1", text)


def find_quote(passage: str, quote: str, min_score: float = 80.0) -> tuple[int, int] | None:
    """(start, end) of the quote in the passage: exact (case-insensitive) first, else best fuzzy alignment."""
    if not quote.strip():
        return None
    start = passage.lower().find(quote.lower())
    if start != -1:
        return start, start + len(quote)
    align = fuzz.partial_ratio_alignment(quote.lower(), passage.lower())
    if align is None or align.score < min_score:
        return None
    return align.dest_start, align.dest_end


def highlight_quote(passage: str, quote: str) -> str:
    """Escaped passage markdown with the cited span on an orange background."""
    span = find_quote(passage, quote)
    if span is None:
        return escape_md(passage)
    start, end = span
    return (escape_md(passage[:start]) + ":orange-background[" + escape_md(passage[start:end]) + "]"
            + escape_md(passage[end:]))


def verdict_badge(label: str | None) -> str:
    return f":{VERDICT_COLORS.get(label, 'gray')}-badge[{label or 'not verified'}]"


def answer_with_citations(answer: str, claims: list[dict]) -> str:
    """Answer followed by numbered citation markers, e.g. 'Paris [1][2]' (markers map to the claims table)."""
    markers = "".join(f"[{i}]" for i in range(1, len(claims) + 1))
    return f"{escape_md(answer)} {escape_md(markers)}".strip()


def claim_rows(claims: list[dict]) -> list[dict]:
    """Rows for the claims table."""
    rows = []
    for i, c in enumerate(claims, start=1):
        check = c.get("web_check") or {}
        rows.append({"#": i, "claim": c["text"], "verdict": c.get("label") or "not verified", "source": c["doc_id"],
                     "web check": WEB_LABELS.get(check.get("status")), "web source": check.get("url", ""),
                     "reason": c.get("reason", "")})
    return rows
