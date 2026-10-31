"""Web fact checker (Tavily): do live web search results back each claim?

For every claim the Verifier marked supported, search the web with the claim's text and fuzzy-match the claim
(and its quote) against the result titles and snippets.
- confirmed:  a result matches (score >= threshold); its URL is recorded as the web source
- not_found:  results came back but none matches the claim
- no_results: the search returned nothing
- error:      Tavily unreachable, key missing or credits exhausted; the Verifier keeps the model's verdict
Results are cached (see src/websearch.py), so repeated questions cost no credits.
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
from src.websearch import TavilyClient
from src.websearch import best_match


async def check_claim(search: TavilyClient, claim: Claim, threshold: float) -> FactCheck:
    query = claim.text
    try:
        results = await search.search(query)
    except Exception as e:  # network, key or credit failure: report it, do not guess
        return FactCheck(claim_id=claim.claim_id, query=query, status="error", detail=f"{type(e).__name__}: {e}"[:200])
    if not results:
        return FactCheck(claim_id=claim.claim_id, query=query, status="no_results", detail="the web search found nothing")
    score, best = best_match(claim.text, claim.quote, results)
    if score >= threshold:
        return FactCheck(claim_id=claim.claim_id, query=query, status="confirmed", score=score, url=best["url"] or "",
                         source_title=best["title"] or "")
    return FactCheck(claim_id=claim.claim_id, query=query, status="not_found", score=score,
                     detail=f"no web result matches the claim (best match {score:.0f})")


class FactCheckerConfig(FunctionBaseConfig, name="tavily_fact_checker"):
    """Check claims against live web search results (Tavily)."""
    threshold: float = Field(default=80.0, ge=0, le=100, description="Minimum fuzzy match score (0-100).")
    max_results: int = Field(default=5, ge=1, le=20)
    search_depth: str = Field(default="basic", description="basic (1 credit) or advanced (2 credits)")
    cache: bool = Field(default=True, description="false = fresh search results every time (spends credits)")
    timeout_s: float = Field(default=30.0, gt=0)


@register_function(config_type=FactCheckerConfig)
async def tavily_fact_checker(config: FactCheckerConfig, builder: Builder):
    search = TavilyClient(search_depth=config.search_depth, max_results=config.max_results, timeout=config.timeout_s,
                          cache=config.cache)

    async def _check(request: FactCheckRequest) -> FactCheckOutput:
        """Check every claim against live web search results."""
        checks = await asyncio.gather(*[check_claim(search, c, config.threshold) for c in request.claims])
        return FactCheckOutput(checks=list(checks))

    yield FunctionInfo.from_fn(_check, description=_check.__doc__)
