"""FastAPI service. The NVIDIA key lives only here (read from the environment / .env), never in the UI.

Run:  uvicorn src.api.app:app --port 8000
  GET  /health    liveness
  GET  /settings  default settings and the valid rail names
  GET  /examples  cached example questions with stored responses (demo works without live calls)
  POST /ask       {"question": "...", "settings": {"retry_cap": 1, "rails": {"pii": false}, "fact_check": true}}
Upstream rate limits come back as HTTP 429 with a readable message.
"""

import json
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi import HTTPException

from src import paths
from src.api.runner import UnknownRailError
from src.api.runner import WorkflowPool
from src.api.runner import build_config
from src.state import ApiRequest
from src.state import ApiResponse
from src.state import ApiSettings

logger = logging.getLogger(__name__)
EXAMPLES_PATH = paths.ROOT / "data" / "examples.json"


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


def load_examples() -> list[dict]:
    if not EXAMPLES_PATH.exists():
        return []
    return json.loads(EXAMPLES_PATH.read_text(encoding="utf-8"))


@asynccontextmanager
async def lifespan(app: FastAPI):
    async with WorkflowPool() as pool:
        app.state.pool = pool
        yield


app = FastAPI(title="Self-Correcting RAG", lifespan=lifespan)


@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}


@app.get("/settings")
async def settings() -> dict:
    _, effective = build_config(ApiSettings())
    return effective


@app.get("/examples")
async def examples() -> list[dict]:
    return load_examples()


@app.post("/ask", response_model=ApiResponse)
async def ask(request: ApiRequest) -> ApiResponse:
    try:
        return await app.state.pool.ask(request.question, request.settings)
    except UnknownRailError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e
    except Exception as e:
        if is_rate_limit(e):
            raise HTTPException(status_code=429,
                                detail="The NVIDIA API rate limit was reached. Wait a minute and try again, "
                                "or pick a cached example.") from e
        logger.exception("question failed")
        raise HTTPException(status_code=502, detail=f"The pipeline failed: {type(e).__name__}: {e}"[:500]) from e
