"""Live web search via the Tavily Search API (https://docs.tavily.com). Used by the agent's web tool and the
web fact check.

POST https://api.tavily.com/search with a Bearer key (TAVILY_API_KEY). A "basic" search costs 1 credit.
Results are cached on disk (data/cache/tavily_cache.sqlite) by (query, params) so re-runs are reproducible and
spend no credits; pass cache=False for fresh results every time.
Web text is untrusted: callers send it through the retrieval rail before any model sees it.
"""

import hashlib
import os
import time

import httpx

from src import paths
from src.llm import LLMCache
from src.llm import cache_key
from src.state import RetrievedDoc

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


def web_doc(result: dict, rank: int) -> RetrievedDoc:
    """A Tavily result as a passage with a stable id: web_<hash of url + snippet>."""
    url, content = result.get("url") or "", result.get("content") or ""
    doc_id = "web_" + hashlib.sha256(f"{url}\n{content}".encode()).hexdigest()[:10]
    return RetrievedDoc(doc_id=doc_id, title=result.get("title") or url, text=content,
                        score=float(result.get("score") or 0.0), rank=rank, url=url)
