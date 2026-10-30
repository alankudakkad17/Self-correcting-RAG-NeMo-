"""Wikipedia fact checker: is each claim's quote present on the live Wikipedia page of the passage it cites?

Corpus passages are Wikipedia paragraphs, so the cited passage's title is a Wikipedia title.
- confirmed:    quote found (fuzzy score >= threshold)
- not_found:    page exists but the quote is not on it
- page_missing: no such Wikipedia page (e.g. a fabricated or poisoned "archived copy" passage)
- error:        Wikipedia unreachable; the Verifier keeps the model's verdict and records the error
Deterministic apart from the live page content, which is cached on first fetch.
"""

import asyncio

from pydantic import Field

from nat.plugin_api import Builder
from nat.plugin_api import FunctionBaseConfig
from nat.plugin_api import FunctionInfo
from nat.plugin_api import register_function

from src.state import Claim
from src.state import FactCheck
from src.state import FactCheckOutput
from src.state import FactCheckRequest
from src.wikipedia import WikipediaClient
from src.wikipedia import quote_match_score


async def check_claim(wiki: WikipediaClient, claim: Claim, title: str, threshold: float) -> FactCheck:
    try:
        article = await wiki.article_text(title)
    except Exception as e:  # network / HTTP failure: report it, do not guess
        return FactCheck(claim_id=claim.claim_id, title=title, status="error", detail=f"{type(e).__name__}: {e}"[:200])
    if article is None:
        return FactCheck(claim_id=claim.claim_id, title=title, status="page_missing",
                         detail=f"no Wikipedia page titled {title!r}")
    score = quote_match_score(claim.quote or claim.text, article)
    status = "confirmed" if score >= threshold else "not_found"
    return FactCheck(claim_id=claim.claim_id, title=title, status=status, score=score,
                     detail="" if status == "confirmed" else f"quote not found on the live page (best match {score:.0f})")


class FactCheckerConfig(FunctionBaseConfig, name="wikipedia_fact_checker"):
    """Check claim quotes against live Wikipedia."""
    threshold: float = Field(default=80.0, ge=0, le=100, description="Minimum fuzzy match score (0-100).")
    timeout_s: float = Field(default=20.0, gt=0)


@register_function(config_type=FactCheckerConfig)
async def wikipedia_fact_checker(config: FactCheckerConfig, builder: Builder):
    wiki = WikipediaClient(timeout=config.timeout_s)

    async def _check(request: FactCheckRequest) -> FactCheckOutput:
        """Check every claim's quote on the Wikipedia page of its cited passage."""
        titles = {d.doc_id: d.title for d in request.evidence}
        checks = await asyncio.gather(*[
            check_claim(wiki, c, titles.get(c.doc_id, c.doc_id.replace("_", " ")), config.threshold)
            for c in request.claims
        ])
        return FactCheckOutput(checks=list(checks))

    yield FunctionInfo.from_fn(_check, description=_check.__doc__)
