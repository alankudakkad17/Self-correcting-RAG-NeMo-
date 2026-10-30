"""Hybrid retrieval: BM25 + dense (FAISS) candidates, fused with reciprocal rank fusion, then reranked.

Dev-set check that chose this setup (50 answerable questions, recall@5): bm25 0.71, dense 0.92, plain RRF 0.77,
RRF + local MiniLM rerank 0.80, RRF top-30 + hosted Nemotron rerank 0.95.
"""

import json

from src import paths
from src.retrieval.bm25 import BM25Index
from src.retrieval.fusion import reciprocal_rank_fusion
from src.retrieval.reranker import HostedReranker
from src.retrieval.vector import Embedder
from src.retrieval.vector import VectorIndex
from src.state import Passage
from src.state import RetrievedDoc


def load_corpus() -> dict[str, Passage]:
    with open(paths.CORPUS_PATH, encoding="utf-8") as f:
        passages = [Passage(**json.loads(line)) for line in f]
    return {p.doc_id: p for p in passages}


class RetrievalPipeline:

    def __init__(self, embedder: Embedder):
        self.corpus = load_corpus()
        self.bm25 = BM25Index(list(self.corpus.values()))
        self.vectors = VectorIndex.load()
        if self.vectors.model != embedder.model:
            raise ValueError(f"index was built with {self.vectors.model}, config asks for {embedder.model}")
        self.embedder = embedder
        self.reranker = HostedReranker()

    async def search(self, query: str, k: int = 5, candidates: int = 50, rerank_pool: int = 30) -> list[RetrievedDoc]:
        sparse = [d for d, _ in self.bm25.search(query, candidates)]
        dense = [d for d, _ in await self._dense(query, candidates)]
        fused = reciprocal_rank_fusion([sparse, dense])[:rerank_pool]
        ranked = await self.reranker.rerank(query, [(d, self._doc_text(d)) for d, _ in fused])
        return [self._to_doc(d, s, r) for r, (d, s) in enumerate(ranked[:k], start=1)]

    async def _dense(self, query: str, k: int) -> list[tuple[str, float]]:
        vec = await self.embedder.embed([query], input_type="query")
        return self.vectors.search(vec[0], k)

    def _doc_text(self, doc_id: str) -> str:
        p = self.corpus[doc_id]
        return f"{p.title}. {p.text}"

    def _to_doc(self, doc_id: str, score: float, rank: int) -> RetrievedDoc:
        p = self.corpus[doc_id]
        return RetrievedDoc(doc_id=doc_id, title=p.title, text=p.text, score=score, rank=rank)
