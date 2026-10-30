"""Live Wikipedia lookups for fact checking (MediaWiki Action API, plain-text extracts).

Responses are cached on disk (data/cache/wikipedia_cache.sqlite), including "page missing", so re-runs are
reproducible and polite to Wikipedia. Wikipedia text is only fuzzy-matched against, never shown to an LLM.
"""

import os
import re
import time

import httpx
from rapidfuzz import fuzz

from src import paths
from src.llm import LLMCache
from src.llm import cache_key

API_URL = "https://en.wikipedia.org/w/api.php"


def user_agent() -> str:
    """Wikimedia rejects (HTTP 403) clients without contact info: https://meta.wikimedia.org/wiki/User-Agent_policy"""
    contact = os.environ.get("WIKIPEDIA_CONTACT", "").strip()
    if not contact:
        raise RuntimeError("WIKIPEDIA_CONTACT is not set (an email or URL where you can be reached; put it in .env)")
    return f"SelfCorrectingRAG/0.1 ({contact}; research prototype)"


class WikipediaClient:

    def __init__(self, timeout: float = 20.0):
        self.cache = LLMCache(paths.CACHE_DIR / "wikipedia_cache.sqlite")
        self.timeout = timeout

    async def article_text(self, title: str) -> str | None:
        """Plain text of the article (redirects followed), or None if no such page exists."""
        key = cache_key("wikipedia-extract", [{"role": "title", "content": title}], {})
        hit = self.cache.get(key)
        if hit is not None:
            return hit["text"]
        params = {"action": "query", "prop": "extracts", "explaintext": 1, "redirects": 1, "titles": title,
                  "format": "json", "formatversion": 2}
        async with httpx.AsyncClient(timeout=self.timeout, headers={"User-Agent": user_agent()},
                                     transport=httpx.AsyncHTTPTransport(retries=2)) as client:
            resp = await client.get(API_URL, params=params)
        resp.raise_for_status()
        pages = resp.json().get("query", {}).get("pages", [])
        page = pages[0] if pages else {}
        text = None if page.get("missing") or page.get("invalid") else page.get("extract") or None
        self.cache.put(key, {"text": text, "fetched_unix": int(time.time())})
        return text


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", text.lower())).strip()


def quote_match_score(quote: str, article: str) -> float:
    """0-100: how well the quote appears somewhere in the article (best partial alignment)."""
    q, a = _normalize(quote), _normalize(article)
    if not q or not a:
        return 0.0
    if q in a:
        return 100.0
    return float(fuzz.partial_ratio(q, a))
