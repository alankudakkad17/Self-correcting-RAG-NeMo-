"""UI client (src/ui/client.py) against a mocked HTTP transport, and the safe-rendering helpers."""

import json

import httpx
import pytest

from src.ui.client import ApiClient
from src.ui.client import ApiError
from src.ui.client import RateLimited
from src.ui.components.render import answer_with_citations
from src.ui.components.render import outcome
from src.ui.components.render import web_badge
from src.ui.components.render import escape_md
from src.ui.components.render import find_quote
from src.ui.components.render import highlight_quote


def client_with(handler) -> ApiClient:
    return ApiClient("http://api.test", transport=httpx.MockTransport(handler))


def test_ask_posts_question_and_settings():
    seen = {}

    def handler(request):
        seen["path"], seen["body"] = request.url.path, request.read()
        return httpx.Response(200, json={"answer": "Paris"})

    assert client_with(handler).ask("Capital?", {"retry_cap": 1}) == {"answer": "Paris"}
    assert seen["path"] == "/ask"
    assert json.loads(seen["body"]) == {"question": "Capital?", "settings": {"retry_cap": 1}}


def test_rate_limit_and_errors_become_readable_exceptions():
    with pytest.raises(RateLimited, match="Wait a minute"):
        client_with(lambda r: httpx.Response(429, json={"detail": "Wait a minute"})).ask("q?")
    with pytest.raises(ApiError, match="API error 502: The pipeline failed"):
        client_with(lambda r: httpx.Response(502, json={"detail": "The pipeline failed"})).ask("q?")
    with pytest.raises(ApiError, match="API error 500"):
        client_with(lambda r: httpx.Response(500, text="not json")).ask("q?")


def test_unreachable_api():
    def handler(request):
        raise httpx.ConnectError("refused")

    client = client_with(handler)
    with pytest.raises(ApiError, match="Cannot reach the API"):
        client.ask("q?")
    assert client.health() is False


def test_escape_md_neutralises_markup_and_directives():
    hostile = "**bold** <script>x</script> :red[alert] [link](http://evil) # heading"
    escaped = escape_md(hostile)
    for raw in ("**", ":red[", "](", "<script>"):
        assert raw not in escaped


def test_quote_is_found_exactly_or_fuzzily_and_highlighted():
    passage = "Field marshal is the highest rank. It has been the highest rank since 1736."
    assert find_quote(passage, "HIGHEST RANK SINCE 1736") == (passage.index("highest rank since"), len(passage) - 1)
    assert find_quote(passage, "it had been the highest rank since 1736") is not None  # light paraphrase
    assert find_quote(passage, "the rank was abolished in 1802") is None
    marked = highlight_quote(passage, "since 1736")
    assert ":orange-background[since 1736]" in marked
    assert highlight_quote(passage, "nowhere in the text at all") == escape_md(passage)


def test_citation_markers_and_web_badges():
    claims = [{"text": "A", "doc_id": "D1", "label": "supported", "web_check": {"status": "confirmed"}},
              {"text": "B", "doc_id": "D2", "label": None, "web_check": None}]
    assert answer_with_citations("Paris", claims) == r"Paris \[1\]\[2\]"
    assert web_badge(claims[0]["web_check"]) == ":green-badge[Web: confirmed]"
    assert web_badge(None) == ":gray-badge[Web: not checked]"
    assert web_badge({"status": "contradicted"}) == ":red-badge[Web: contradicted]"


def response(**overrides):
    base = {"blocked": False, "abstained": False, "answer": "Kramer Guitars", "answer_verdict": {"label": "supported"},
            "claims": [{"label": "supported"}, {"label": "supported"}]}
    return {**base, **overrides}


def test_outcome_stamp_for_each_result():
    verified = outcome(response())
    assert (verified["label"], verified["color"]) == ("Verified", "green")
    assert verified["detail"] == "2 of 2 claims supported by their sources, and the answer follows from them."
    assert outcome(response(answer_verdict=None))["detail"].endswith("sources.")
    abstained = outcome(response(abstained=True, answer="insufficient evidence"))
    assert (abstained["label"], abstained["color"]) == ("Not enough evidence", "orange")
    refused = outcome(response(blocked=True, abstained=True, answer="I can't help with that request."))
    assert (refused["label"], refused["detail"]) == ("Refused", "I can't help with that request.")


def test_document_upload_list_and_delete():
    seen = []

    def handler(request):
        seen.append((request.method, request.url.path))
        if request.method == "POST":
            body = request.read()
            assert b'filename="notes.txt"' in body and b"hello" in body  # multipart upload
            return httpx.Response(200, json=[{"filename": "notes.txt", "status": "added", "chunks": 1}])
        if request.method == "GET":
            return httpx.Response(200, json=[{"doc_id": "up_1", "filename": "notes.txt", "chunks": 1}])
        return httpx.Response(200, json={"deleted": "up_1"})

    client = client_with(handler)
    assert client.upload_documents([("notes.txt", b"hello")])[0]["status"] == "added"
    assert client.list_documents()[0]["doc_id"] == "up_1"
    assert client.delete_document("up_1") == {"deleted": "up_1"}
    assert seen == [("POST", "/documents"), ("GET", "/documents"), ("DELETE", "/documents/up_1")]
