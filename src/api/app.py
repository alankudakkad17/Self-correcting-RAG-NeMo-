"""FastAPI service. The NVIDIA key lives only here (read from the environment / .env), never in the UI.

Run:  uvicorn src.api.app:app --port 8000
  GET  /health    liveness
  GET  /settings  default settings and the valid rail names
  POST /ask       {"question": "...", "settings": {"retry_cap": 1, "rails": {"pii": false}, "fact_check": true}}
  POST /documents         upload .txt / .md / .pdf files (multipart field "files"); searchable by the agent
  GET  /documents         uploaded documents
  DELETE /documents/{id}  remove one
Upstream rate limits come back as HTTP 429 with a readable message.
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi import File
from fastapi import HTTPException
from fastapi import UploadFile

from nat.utils.io.yaml_tools import yaml_load

from src import paths
from src.api.runner import UnknownRailError
from src.api.runner import WorkflowPool
from src.api.runner import build_config
from src.state import ApiRequest
from src.state import ApiResponse
from src.state import ApiSettings
from src.uploads.extract import UnsupportedFile
from src.uploads.store import get_store

logger = logging.getLogger(__name__)


def is_rate_limit(error: BaseException) -> bool:
    """True if the error, or anything in its cause/context chain, is an HTTP 429 from the model API."""
    seen = set()
    while error is not None and id(error) not in seen:
        seen.add(id(error))
        if ("RateLimit" in type(error).__name__ or getattr(error, "status_code", None) == 429
                or "Error code: 429" in str(error)):
            return True
        error = error.__cause__ or error.__context__
    return False


def upload_store():
    """The same store the agent's docs_search tool uses (same embedder as the corpus index)."""
    emb = yaml_load(paths.ROOT / "configs" / "llm.yml")["embedders"]["corpus_embedder"]
    return get_store(emb["model_name"], emb.get("base_url"))


@asynccontextmanager
async def lifespan(app: FastAPI):
    async with WorkflowPool() as pool:
        app.state.pool = pool
        app.state.store = upload_store()
        yield


app = FastAPI(title="Self-Correcting RAG", lifespan=lifespan)


@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}


@app.get("/settings")
async def settings() -> dict:
    _, effective = build_config(ApiSettings())
    return effective


@app.post("/ask", response_model=ApiResponse)
async def ask(request: ApiRequest) -> ApiResponse:
    try:
        return await app.state.pool.ask(request.question, request.settings)
    except UnknownRailError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e
    except Exception as e:
        if is_rate_limit(e):
            raise HTTPException(status_code=429,
                                detail="The NVIDIA API rate limit was reached. Wait a minute and try again.") from e
        logger.exception("question failed")
        raise HTTPException(status_code=502, detail=f"The pipeline failed: {type(e).__name__}: {e}"[:500]) from e


@app.post("/documents")
async def upload_documents(files: list[UploadFile] = File(...)) -> list[dict]:
    """Add files to the uploaded-documents vector store. Each file is reported as added, already uploaded or
    rejected (with the reason); one bad file does not stop the others."""
    results = []
    for f in files:
        try:
            results.append(await app.state.store.ingest(f.filename or "upload", await f.read()))
        except UnsupportedFile as e:
            results.append({"filename": f.filename, "status": "rejected", "reason": str(e)})
        except Exception as e:
            if is_rate_limit(e):
                raise HTTPException(status_code=429, detail="The NVIDIA API rate limit was reached while "
                                    "embedding the file. Wait a minute and try again.") from e
            logger.exception("upload failed")
            results.append({"filename": f.filename, "status": "rejected",
                            "reason": f"{type(e).__name__}: {e}"[:300]})
    return results


@app.get("/documents")
async def list_documents() -> list[dict]:
    return app.state.store.list_documents()


@app.delete("/documents/{doc_id}")
async def delete_document(doc_id: str) -> dict:
    if not app.state.store.delete(doc_id):
        raise HTTPException(status_code=404, detail=f"No uploaded document {doc_id!r}")
    return {"deleted": doc_id}
