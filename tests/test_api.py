"""API schema, override handling and endpoints (src/api). The workflow pool is faked: no models are called."""

import httpx
import openai
import pytest
from fastapi.testclient import TestClient

from src.api import app as app_module
from src.api.app import is_rate_limit
from src.api.runner import UnknownRailError
from src.api.runner import build_config
from src.state import ApiMetrics
from src.state import ApiRequest
from src.state import ApiResponse
from src.state import ApiSettings

BASE = {"workflow": {"retry_cap": 2},
        "functions": {"verifier": {"llm_name": "verifier", "fact_checker": "fact_checker"},
                      "guardrails": {"injection": True, "pii": True},
                      "research_agent": {"_type": "react_agent", "tool_names": ["corpus_search", "web_search"]}}}


def test_build_config_applies_overrides_without_touching_the_base():
    config, effective = build_config(ApiSettings(retry_cap=0, rails={"pii": False}, fact_check=False), base=BASE)
    assert config["workflow"]["retry_cap"] == 0
    assert config["functions"]["guardrails"]["pii"] is False
    assert "fact_checker" not in config["functions"]["verifier"]
    assert effective["retry_cap"] == 0 and effective["fact_check"] is False and effective["rails"]["pii"] is False
    assert BASE["workflow"]["retry_cap"] == 2 and "fact_checker" in BASE["functions"]["verifier"]  # deep-copied


def test_build_config_defaults_and_unknown_rail():
    _, effective = build_config(ApiSettings(), base=BASE)
    assert effective == {"retry_cap": 2, "fact_check": True, "web_search": True, "uploads": False,
                         "rails": {r: True for r in effective["rails"]}}
    assert set(effective["rails"]) == {"injection", "off_topic", "unsafe", "retrieval", "groundedness", "citations",
                                       "pii", "abstention"}
    with pytest.raises(UnknownRailError):
        build_config(ApiSettings(rails={"telepathy": True}), base=BASE)


def test_web_search_switch_edits_the_agent_tool_list():
    config, effective = build_config(ApiSettings(web_search=False), base=BASE)
    assert config["functions"]["research_agent"]["tool_names"] == ["corpus_search"]
    assert effective["web_search"] is False
    assert BASE["functions"]["research_agent"]["tool_names"] == ["corpus_search", "web_search"]  # base untouched
    off = {**BASE, "functions": {**BASE["functions"],
                                 "research_agent": {"_type": "react_agent", "tool_names": ["corpus_search"]}}}
    config, effective = build_config(ApiSettings(web_search=True), base=off)
    assert config["functions"]["research_agent"]["tool_names"] == ["corpus_search", "web_search"]
    assert effective["web_search"] is True


def test_web_search_is_off_without_the_react_agent():
    plain = {**BASE, "functions": {k: v for k, v in BASE["functions"].items() if k != "research_agent"}}
    _, effective = build_config(ApiSettings(web_search=True), base=plain)
    assert effective["web_search"] is False


def test_request_validation():
    with pytest.raises(ValueError):
        ApiRequest(question="")
    with pytest.raises(ValueError):
        ApiRequest(question="Who?", settings={"retry_cap": 9})


def fake_response(question: str, **overrides) -> ApiResponse:
    base = dict(question=question, answer="Paris", abstained=False, blocked=False, claims=[], answer_verdict=None,
                plan=None, attempts=[], retrieved=[], guardrail_events=[], settings={},
                metrics=ApiMetrics(latency_s=1.0, llm_calls=3, cached_llm_calls=0, prompt_tokens=10,
                                   completion_tokens=2, retries=0))
    return ApiResponse(**{**base, **overrides})


class FakePool:

    def __init__(self, error: Exception | None = None):
        self.error, self.calls = error, []

    async def ask(self, question, settings):
        self.calls.append((question, settings))
        if self.error:
            raise self.error
        return fake_response(question)


@pytest.fixture
def client():
    with TestClient(app_module.app) as c:
        yield c


def test_ask_returns_the_response_schema(client):
    pool = FakePool()
    client.app.state.pool = pool
    resp = client.post("/ask", json={"question": "Capital of France?", "settings": {"retry_cap": 1}})
    assert resp.status_code == 200
    body = resp.json()
    assert body["answer"] == "Paris" and body["metrics"]["llm_calls"] == 3
    assert pool.calls[0][1].retry_cap == 1


def wrapped_rate_limit() -> Exception:
    """A RateLimitError from the SDK, wrapped the way the toolkit re-raises it."""
    rate_limit = openai.RateLimitError("Error code: 429 - too many requests", body=None,
                                       response=httpx.Response(429, request=httpx.Request("POST", "https://x")))
    try:
        raise RuntimeError("workflow failed") from rate_limit
    except RuntimeError as outer:
        return outer


def test_rate_limit_is_found_through_the_cause_chain():
    assert is_rate_limit(wrapped_rate_limit())
    assert not is_rate_limit(ValueError("Who won in 1429?"))  # a bare number is not a rate limit


def test_rate_limit_becomes_429(client):
    client.app.state.pool = FakePool(wrapped_rate_limit())
    resp = client.post("/ask", json={"question": "Capital of France?"})
    assert resp.status_code == 429 and "rate limit" in resp.json()["detail"].lower()


def test_unknown_rail_is_422_and_other_failures_502(client):
    client.app.state.pool = FakePool(UnknownRailError("unknown rails: ['x']"))
    assert client.post("/ask", json={"question": "Capital of France?"}).status_code == 422
    client.app.state.pool = FakePool(ValueError("boom"))
    resp = client.post("/ask", json={"question": "Capital of France?"})
    assert resp.status_code == 502 and "boom" in resp.json()["detail"]


def test_health_and_settings(client):
    assert client.get("/health").json() == {"status": "ok"}
    assert client.get("/settings").json()["retry_cap"] == 2  # from configs/system.yml


def test_uploads_switch_edits_the_agent_tool_list():
    config, effective = build_config(ApiSettings(uploads=True), base=BASE)
    assert config["functions"]["research_agent"]["tool_names"] == ["corpus_search", "web_search", "docs_search"]
    assert effective["uploads"] is True
    config, effective = build_config(ApiSettings(uploads=False, web_search=False), base=config)
    assert config["functions"]["research_agent"]["tool_names"] == ["corpus_search"]
    assert effective["uploads"] is False and effective["web_search"] is False


class FakeStore:

    def __init__(self):
        self.docs = {}

    async def ingest(self, filename, data):
        if filename.endswith(".exe"):
            from src.uploads.extract import UnsupportedFile
            raise UnsupportedFile(f"{filename}: only .md, .pdf, .txt files are supported")
        self.docs["up_1"] = {"doc_id": "up_1", "filename": filename, "chunks": 1, "uploaded_unix": 0}
        return {**self.docs["up_1"], "status": "added"}

    def list_documents(self):
        return list(self.docs.values())

    def delete(self, doc_id):
        return self.docs.pop(doc_id, None) is not None


def test_document_endpoints(client):
    client.app.state.store = FakeStore()
    resp = client.post("/documents", files=[("files", ("notes.txt", b"hello")), ("files", ("virus.exe", b"x"))])
    assert resp.status_code == 200
    assert [r["status"] for r in resp.json()] == ["added", "rejected"]  # one bad file does not stop the others
    assert "only" in resp.json()[1]["reason"]
    assert [d["filename"] for d in client.get("/documents").json()] == ["notes.txt"]
    assert client.delete("/documents/up_1").json() == {"deleted": "up_1"}
    assert client.delete("/documents/up_1").status_code == 404


def test_uploads_switch_also_controls_the_controller_retrieval():
    with_uploads = {"workflow": {"retry_cap": 2, "uploads": "uploads_retriever"},
                    "functions": {**BASE["functions"], "uploads_retriever": {"_type": "uploads_retriever"}}}
    config, effective = build_config(ApiSettings(uploads=False), base=with_uploads)
    assert config["workflow"]["uploads"] is None and effective["uploads"] is False
    config, effective = build_config(ApiSettings(uploads=True), base=config)
    assert config["workflow"]["uploads"] == "uploads_retriever" and effective["uploads"] is True
