"""Dense retrieval: NVIDIA-hosted embeddings + an exact FAISS inner-product index.

Why FAISS (not Chroma): ~3.6k passages fit an exact flat index, search is deterministic,
and there is no server or database to manage. The index is two files in data/index/.
"""

import json
import os

import faiss
import numpy as np
from openai import AsyncOpenAI

from src import paths
from src.llm import DEFAULT_BASE_URL
from src.llm import LLMCache
from src.llm import cache_key

INDEX_FILE = paths.INDEX_DIR / "faiss.index"
META_FILE = paths.INDEX_DIR / "faiss_meta.json"


class Embedder:
    """Cached embedding calls. NVIDIA retrieval embedders need input_type: 'query' or 'passage'."""

    def __init__(self, model: str, base_url: str | None = None, batch_size: int = 32, client=None):
        self.model = model
        self.batch_size = batch_size
        self.cache = LLMCache(paths.CACHE_DIR / "embed_cache.sqlite")
        self._client = client or AsyncOpenAI(base_url=base_url or os.environ.get("NVIDIA_BASE_URL", DEFAULT_BASE_URL),
                                             api_key=os.environ.get("NVIDIA_API_KEY"),
                                             max_retries=6,
                                             timeout=120)

    async def embed(self, texts: list[str], input_type: str) -> np.ndarray:
        keys = [cache_key(self.model, [{"role": input_type, "content": t}], {}) for t in texts]
        vectors: list = [self.cache.get(k) for k in keys]
        missing = [i for i, v in enumerate(vectors) if v is None]
        for start in range(0, len(missing), self.batch_size):
            batch = missing[start:start + self.batch_size]
            resp = await self._client.embeddings.create(model=self.model,
                                                        input=[texts[i] for i in batch],
                                                        extra_body={"input_type": input_type, "truncate": "END"})
            for i, item in zip(batch, resp.data):
                vectors[i] = item.embedding
                self.cache.put(keys[i], item.embedding)
        arr = np.asarray(vectors, dtype="float32")
        faiss.normalize_L2(arr)  # inner product on unit vectors = cosine similarity
        return arr


class VectorIndex:

    def __init__(self, index: faiss.Index, doc_ids: list[str], model: str):
        self.index = index
        self.doc_ids = doc_ids
        self.model = model

    @classmethod
    def build(cls, doc_ids: list[str], vectors: np.ndarray, model: str) -> "VectorIndex":
        index = faiss.IndexFlatIP(vectors.shape[1])
        index.add(vectors)
        return cls(index, doc_ids, model)

    def save(self):
        paths.INDEX_DIR.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(INDEX_FILE))
        META_FILE.write_text(json.dumps({"model": self.model, "doc_ids": self.doc_ids}))

    @classmethod
    def load(cls) -> "VectorIndex":
        if not INDEX_FILE.exists():
            raise FileNotFoundError(f"{INDEX_FILE} missing: run `python scripts/build_index.py` first.")
        meta = json.loads(META_FILE.read_text())
        return cls(faiss.read_index(str(INDEX_FILE)), meta["doc_ids"], meta["model"])

    def search(self, query_vector: np.ndarray, k: int) -> list[tuple[str, float]]:
        scores, idx = self.index.search(query_vector.reshape(1, -1), k)
        return [(self.doc_ids[i], float(s)) for i, s in zip(idx[0], scores[0]) if i >= 0]
