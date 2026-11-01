"""Vector store for user-uploaded documents: ChromaDB, embedded and persistent (data/uploads/chroma).

Why a separate store: the benchmark corpus is frozen (static FAISS index); uploads need adding, deleting,
metadata and persistence. Chroma runs in-process on Windows (the toolkit's Milvus retriever needs Milvus
Lite, which is Linux/macOS only, or Docker).
Embeddings come from our cached Nemotron embedder; Chroma's own default embedding model is never used.
"""

import hashlib
import time
from functools import lru_cache
from pathlib import Path

import chromadb
from chromadb.config import Settings

from src import paths
from src.retrieval.vector import Embedder
from src.state import RetrievedDoc
from src.uploads.extract import chunk_text
from src.uploads.extract import extract_text

STORE_DIR = paths.ROOT / "data" / "uploads" / "chroma"
COLLECTION = "uploads"
MAX_CHUNKS = 2000  # per file


class UploadStore:

    def __init__(self, embedder: Embedder, path: Path = STORE_DIR):
        path.mkdir(parents=True, exist_ok=True)
        client = chromadb.PersistentClient(path=str(path), settings=Settings(anonymized_telemetry=False))
        self.collection = client.get_or_create_collection(name=COLLECTION, embedding_function=None,
                                                          configuration={"hnsw": {"space": "cosine"}})
        self.embedder = embedder

    async def ingest(self, filename: str, data: bytes) -> dict:
        """Extract, chunk, embed and store a file. Re-uploading identical content is a no-op."""
        text = extract_text(filename, data)
        doc_id = "up_" + hashlib.sha256(text.encode("utf-8")).hexdigest()[:10]
        existing = self.collection.get(where={"doc_id": doc_id}, limit=1)
        if existing["ids"]:
            return {**self._summary(doc_id), "status": "already uploaded"}
        chunks = chunk_text(text)[:MAX_CHUNKS]
        vectors = await self.embedder.embed(chunks, input_type="passage")
        now = int(time.time())
        self.collection.add(
            ids=[f"{doc_id}_{i:04d}" for i in range(len(chunks))],
            embeddings=vectors.tolist(),
            documents=chunks,
            metadatas=[{"doc_id": doc_id, "filename": filename, "chunk": i, "chunks": len(chunks), "uploaded_unix": now}
                       for i in range(len(chunks))])
        return {"doc_id": doc_id, "filename": filename, "chunks": len(chunks), "chars": len(text), "status": "added"}

    async def search(self, query: str, k: int = 4) -> list[RetrievedDoc]:
        if self.collection.count() == 0:
            return []
        vector = await self.embedder.embed([query], input_type="query")
        res = self.collection.query(query_embeddings=vector.tolist(), n_results=k,
                                    include=["documents", "metadatas", "distances"])
        docs = []
        for rank, (chunk_id, text, meta, dist) in enumerate(
                zip(res["ids"][0], res["documents"][0], res["metadatas"][0], res["distances"][0]), start=1):
            docs.append(RetrievedDoc(doc_id=chunk_id, title=f"{meta['filename']} (part {meta['chunk'] + 1})",
                                     text=text, score=round(1.0 - float(dist), 4), rank=rank))
        return docs

    def list_documents(self) -> list[dict]:
        metas = self.collection.get(include=["metadatas"])["metadatas"]
        docs = {}
        for m in metas:
            docs.setdefault(m["doc_id"], {"doc_id": m["doc_id"], "filename": m["filename"], "chunks": m["chunks"],
                                          "uploaded_unix": m["uploaded_unix"]})
        return sorted(docs.values(), key=lambda d: d["uploaded_unix"], reverse=True)

    def delete(self, doc_id: str) -> bool:
        if not self.collection.get(where={"doc_id": doc_id}, limit=1)["ids"]:
            return False
        self.collection.delete(where={"doc_id": doc_id})
        return True

    def _summary(self, doc_id: str) -> dict:
        return next(d for d in self.list_documents() if d["doc_id"] == doc_id)


@lru_cache(maxsize=None)
def get_store(embedding_model: str, base_url: str | None = None) -> UploadStore:
    """One store per process, shared by the agent's docs_search tool and the API's upload endpoints."""
    return UploadStore(Embedder(embedding_model, base_url=base_url))
