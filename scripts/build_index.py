"""Embed the frozen corpus and build the FAISS index in data/index/.

Usage: python scripts/build_index.py
Embeddings are cached per passage, so an interrupted build resumes without re-spending quota.
The embedding model comes from `embedders.corpus_embedder` in configs/llm.yml.
"""

import asyncio
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from nat.utils.io.yaml_tools import yaml_load  # noqa: E402

from src.retrieval.pipeline import load_corpus  # noqa: E402
from src.retrieval.vector import Embedder  # noqa: E402
from src.retrieval.vector import VectorIndex  # noqa: E402


async def main():
    cfg = yaml_load(ROOT / "configs" / "llm.yml")["embedders"]["corpus_embedder"]
    corpus = load_corpus()
    doc_ids = list(corpus)
    # Same text the reranker sees: title first, it carries the entity name.
    texts = [f"{corpus[d].title}. {corpus[d].text}" for d in doc_ids]

    start = time.perf_counter()
    vectors = await Embedder(cfg["model_name"], base_url=cfg.get("base_url")).embed(texts, input_type="passage")
    VectorIndex.build(doc_ids, vectors, cfg["model_name"]).save()
    print(f"indexed {len(doc_ids)} passages, dim {vectors.shape[1]}, in {time.perf_counter() - start:.0f}s")


if __name__ == "__main__":
    asyncio.run(main())
