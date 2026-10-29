"""Retriever as a registered toolkit function: query -> top-k passages from the frozen corpus."""

from pydantic import BaseModel
from pydantic import Field

from nat.plugin_api import Builder
from nat.plugin_api import EmbedderRef
from nat.plugin_api import FunctionBaseConfig
from nat.plugin_api import FunctionInfo
from nat.plugin_api import register_function

from src.retrieval.pipeline import Mode
from src.retrieval.pipeline import RerankerKind
from src.retrieval.pipeline import RetrievalPipeline
from src.retrieval.vector import Embedder
from src.state import RetrievedDoc


class RetrievalRequest(BaseModel):
    query: str
    top_k: int | None = None  # None = the configured top_k
    exclude_ids: list[str] = Field(default_factory=list)  # e.g. passages already in the evidence pool


class RetrievalOutput(BaseModel):
    docs: list[RetrievedDoc]


class RetrieverConfig(FunctionBaseConfig, name="corpus_retriever"):
    """Search the frozen corpus (BM25, dense, or hybrid with RRF + reranking)."""
    embedder_name: EmbedderRef = Field(description="Entry under `embedders:` the index was built with.")
    mode: Mode = "hybrid"
    reranker: RerankerKind = "hosted"
    top_k: int = Field(default=5, ge=1)
    candidates: int = Field(default=50, ge=1, description="Per-retriever candidates fed into RRF.")
    rerank_pool: int = Field(default=30, ge=1, description="Fused candidates passed to the reranker.")


@register_function(config_type=RetrieverConfig)
async def corpus_retriever(config: RetrieverConfig, builder: Builder):
    emb_cfg = builder.get_embedder_config(config.embedder_name)
    # Loaded once per workflow: BM25 is built in memory, FAISS index read from data/index/.
    pipeline = RetrievalPipeline(Embedder(emb_cfg.model_name, base_url=emb_cfg.base_url),
                                 reranker=config.reranker if config.mode == "hybrid" else "none")

    async def _retrieve(request: RetrievalRequest) -> RetrievalOutput:
        """Return the top-k corpus passages for the query, skipping excluded ids."""
        k = request.top_k or config.top_k
        exclude = set(request.exclude_ids)
        docs = await pipeline.search(request.query, config.mode, k=k + len(exclude), candidates=config.candidates,
                                     rerank_pool=max(config.rerank_pool, k + len(exclude)))
        docs = [d for d in docs if d.doc_id not in exclude][:k]
        for rank, d in enumerate(docs, start=1):
            d.rank = rank
        return RetrievalOutput(docs=docs)

    yield FunctionInfo.from_fn(_retrieve, description=_retrieve.__doc__)
