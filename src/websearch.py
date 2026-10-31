"""Live web search for fact checking, via the Tavily Search API (https://docs.tavily.com).

POST https://api.tavily.com/search with a Bearer key (TAVILY_API_KEY). A "basic" search costs 1 credit.
Results are cached on disk (data/cache/tavily_cache.sqlite) by (query, params) so re-runs are reproducible and
spend no credits; set `cache: false` on the fact checker for fresh results every time.
Web text is untrusted: it is only fuzzy-matched against, never shown to an LLM.
"""

import os
import re
import time

import httpx
from rapidfuzz import fuzz

from src import paths
from src.llm import LLMCache
from src.llm import cache_key

API_URL = "https://api.tavily.com/search"


class TavilyError(Exception):
    pass


class TavilyClient:

    def __init__(self, search_depth: str = "basic", max_results: int = 5, timeout: float = 30.0, cache: bool = True):
        self.params = {"search_depth": search_depth, "max_results": max_results, "include_answer": False,
                       "include_raw_content": False}
        self.timeout = timeout
        self.cache = LLMCache(paths.CACHE_DIR / "tavily_cache.sqlite") if cache else None

    async def search(self, query: str) -> list[dict]:
        """Ranked results: [{"title", "url", "content", "score"}]."""
        key = cache_key("tavily-search", [{"role": "query", "content": query}], self.params)
        hit = self.cache.get(key) if self.cache else None
        if hit is not None:
            return hit["results"]
        api_key = os.environ.get("TAVILY_API_KEY", "").strip()
        if not api_key:
            raise TavilyError("TAVILY_API_KEY is not set (get one at https://app.tavily.com, put it in .env)")
        async with httpx.AsyncClient(timeout=self.timeout, transport=httpx.AsyncHTTPTransport(retries=2)) as client:
            resp = await client.post(API_URL, headers={"Authorization": f"Bearer {api_key}"},
                                     json={"query": query[:400], **self.params})
        if resp.status_code == 401:
            raise TavilyError("Tavily rejected the API key (401)")
        if resp.status_code in (429, 432, 433):
            raise TavilyError(f"Tavily rate or plan limit reached ({resp.status_code})")
        resp.raise_for_status()
        results = [{k: r.get(k) for k in ("title", "url", "content", "score")} for r in resp.json().get("results", [])]
        if self.cache:
            self.cache.put(key, {"results": results, "fetched_unix": int(time.time())})
        return results


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", (text or "").lower())).strip()


def match_score(needle: str, haystack: str) -> float:
    """0-100: how well `needle` appears somewhere in `haystack` (best partial alignment)."""
    n, h = _normalize(needle), _normalize(haystack)
    if not n or not h:
        return 0.0
    if n in h:
        return 100.0
    return float(fuzz.partial_ratio(n, h))


def best_match(claim_text: str, quote: str, results: list[dict]) -> tuple[float, dict | None]:
    """Best (score, result): the claim's quote or its text, matched against each result's title + snippet."""
    best_score, best = 0.0, None
    for r in results:
        content = f"{r.get('title') or ''}. {r.get('content') or ''}"
        score = max(match_score(quote, content) if quote else 0.0, match_score(claim_text, content))
        if score > best_score:
            best_score, best = score, r
    return best_score, best
