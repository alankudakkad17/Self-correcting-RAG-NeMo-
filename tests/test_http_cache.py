"""HTTP-level cache used for RAGAS judge calls (src/llm.py: CachingTransport)."""

import json

import httpx

from src.llm import CachingTransport
from src.llm import LLMCache


class CountingTransport(httpx.AsyncBaseTransport):

    def __init__(self, status=200):
        self.calls = 0
        self.status = status

    async def handle_async_request(self, request):
        self.calls += 1
        return httpx.Response(self.status, json={"n": self.calls}, request=request)


def post(body: dict):
    return httpx.Request("POST", "https://integrate.api.nvidia.com/v1/chat/completions", content=json.dumps(body))


async def test_identical_body_is_served_from_cache(tmp_path):
    inner = CountingTransport()
    t = CachingTransport(LLMCache(tmp_path / "h.sqlite"), inner)
    r1 = await t.handle_async_request(post({"model": "m", "temperature": 0, "messages": []}))
    r2 = await t.handle_async_request(post({"messages": [], "temperature": 0, "model": "m"}))  # key order differs
    assert json.loads(r1.content) == json.loads(r2.content) == {"n": 1}
    assert inner.calls == 1 and (t.hits, t.misses) == (1, 1)


async def test_different_params_miss(tmp_path):
    inner = CountingTransport()
    t = CachingTransport(LLMCache(tmp_path / "h.sqlite"), inner)
    await t.handle_async_request(post({"model": "m", "temperature": 0}))
    await t.handle_async_request(post({"model": "m", "temperature": 0.5}))
    assert inner.calls == 2


async def test_errors_are_not_cached(tmp_path):
    inner = CountingTransport(status=429)
    t = CachingTransport(LLMCache(tmp_path / "h.sqlite"), inner)
    for _ in range(2):
        r = await t.handle_async_request(post({"model": "m"}))
        assert r.status_code == 429
    assert inner.calls == 2
