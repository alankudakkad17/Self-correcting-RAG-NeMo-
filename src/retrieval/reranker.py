"""Rerankers. Both take (query, [(doc_id, text)]) and return [(doc_id, score)] best first; ties keep input order.

- LocalReranker:  ms-marco-MiniLM-L6-v2 cross-encoder on CPU. Free and deterministic.
- HostedReranker: NVIDIA-hosted llama-nemotron-rerank-vl-1b-v2 (text queries work). Cached like LLM calls.
"""

import os

import httpx

from src import paths
from src.llm import LLMCache
from src.llm import cache_key


def _order(doc_ids: list[str], scores: list[float]) -> list[tuple[str, float]]:
    ranked = sorted(range(len(doc_ids)), key=lambda i: (-scores[i], i))
    return [(doc_ids[i], float(scores[i])) for i in ranked]


class LocalReranker:

    def __init__(self):
        from sentence_transformers import CrossEncoder  # heavy import (torch): only when used
        if not paths.RERANKER_DIR.exists():
            raise FileNotFoundError(f"{paths.RERANKER_DIR} missing: run `python scripts/download_data.py` first.")
        self._model = CrossEncoder(str(paths.RERANKER_DIR), device="cpu")

    async def rerank(self, query: str, docs: list[tuple[str, str]]) -> list[tuple[str, float]]:
        if not docs:
            return []
        scores = self._model.predict([(query, text) for _, text in docs], show_progress_bar=False)
        return _order([d for d, _ in docs], [float(s) for s in scores])


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
