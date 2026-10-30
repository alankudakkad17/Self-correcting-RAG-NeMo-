"""Reranker: NVIDIA-hosted llama-nemotron-rerank-vl-1b-v2 (text queries work). Cached like LLM calls.

Takes (query, [(doc_id, text)]) and returns [(doc_id, score)] best first; ties keep the input order.
"""

import os

import httpx

from src import paths
from src.llm import LLMCache
from src.llm import cache_key


def _order(doc_ids: list[str], scores: list[float]) -> list[tuple[str, float]]:
    ranked = sorted(range(len(doc_ids)), key=lambda i: (-scores[i], i))
    return [(doc_ids[i], float(scores[i])) for i in ranked]


class HostedReranker:

    def __init__(self, model: str = "nvidia/llama-nemotron-rerank-vl-1b-v2"):
        self.model = model
        self.url = f"https://ai.api.nvidia.com/v1/retrieval/{model.replace('.', '_')}/reranking"
        self.cache = LLMCache(paths.CACHE_DIR / "rerank_cache.sqlite")
        self._headers = {"Authorization": f"Bearer {os.environ.get('NVIDIA_API_KEY')}", "Accept": "application/json"}

    async def rerank(self, query: str, docs: list[tuple[str, str]]) -> list[tuple[str, float]]:
        if not docs:
            return []
        texts = [t for _, t in docs]
        key = cache_key(self.model, [{"role": "query", "content": query}], {"passages": texts})
        scores = self.cache.get(key)
        if scores is None:
            body = {"model": self.model, "query": {"text": query}, "passages": [{"text": t} for t in texts],
                    "truncate": "END"}
            async with httpx.AsyncClient(timeout=120,
                                         transport=httpx.AsyncHTTPTransport(retries=3)) as client:
                resp = await client.post(self.url, headers=self._headers, json=body)
            resp.raise_for_status()
            scores = [0.0] * len(texts)
            for r in resp.json()["rankings"]:
                scores[r["index"]] = r["logit"]
            self.cache.put(key, scores)
        return _order([d for d, _ in docs], scores)
