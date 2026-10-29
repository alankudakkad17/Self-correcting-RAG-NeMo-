"""Offline tests for src/llm.py: no network, the OpenAI client is faked."""

from pathlib import Path
from types import SimpleNamespace

import pytest

from src import llm
from src.llm import ChatClient
from src.llm import LLMCache
from src.llm import cache_key

ROOT = Path(__file__).resolve().parents[1]


class FakeCompletions:
    """Records every request and answers with a fixed text."""

    def __init__(self):
        self.calls = []

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=" Paris "))],
                               usage=SimpleNamespace(prompt_tokens=12, completion_tokens=3))


class FakeOpenAI:

    def __init__(self, *args, **kwargs):
        self.chat = SimpleNamespace(completions=FakeCompletions())


MSG = [{"role": "user", "content": "capital of France?"}]


def make_client(tmp_path, **kwargs):
    return ChatClient("m1", {"temperature": 0.0}, cache=LLMCache(tmp_path / "c.sqlite"), use_cache=True,
                      openai_client=FakeOpenAI(), **kwargs)


def test_cache_key_is_stable_and_sensitive():
    k = cache_key("m", MSG, {"temperature": 0.0, "max_tokens": 5})
    assert k == cache_key("m", MSG, {"max_tokens": 5, "temperature": 0.0})  # dict order does not matter
    assert k != cache_key("m2", MSG, {"temperature": 0.0, "max_tokens": 5})
    assert k != cache_key("m", MSG, {"temperature": 0.5, "max_tokens": 5})
    assert k != cache_key("m", [{"role": "user", "content": "other"}], {"temperature": 0.0, "max_tokens": 5})


def test_cache_roundtrip_persists(tmp_path):
    LLMCache(tmp_path / "c.sqlite").put("k", {"text": "hi"})
    assert LLMCache(tmp_path / "c.sqlite").get("k") == {"text": "hi"}  # new connection, same file
    assert LLMCache(tmp_path / "c.sqlite").get("missing") is None


async def test_second_identical_call_is_served_from_cache(tmp_path):
    client = make_client(tmp_path)
    first = await client.chat(MSG)
    second = await client.chat(MSG)
    assert first.text == second.text == "Paris"
    assert (first.cached, second.cached) == (False, True)
    assert second.prompt_tokens == 12  # original token counts are kept, so reruns report stable usage
    assert len(client._client.chat.completions.calls) == 1


async def test_param_override_misses_cache(tmp_path):
    client = make_client(tmp_path)
    await client.chat(MSG)
    await client.chat(MSG, temperature=0.7)
    assert len(client._client.chat.completions.calls) == 2


async def test_cache_disabled_always_calls_api(tmp_path):
    client = ChatClient("m1", openai_client=FakeOpenAI(), use_cache=False)
    await client.chat(MSG)
    await client.chat(MSG)
    assert client.cache is None
    assert len(client._client.chat.completions.calls) == 2


async def test_from_toolkit_config_sends_thinking_switch_in_extra_body(tmp_path):
    from nat.llm.nim_llm import NIMModelConfig
    cfg = NIMModelConfig(model_name="nvidia/nemotron-3-super-120b-a12b", temperature=0.0, max_tokens=64,
                         chat_template_kwargs={"enable_thinking": False})
    client = ChatClient.from_toolkit_config(cfg, cache=LLMCache(tmp_path / "c.sqlite"), openai_client=FakeOpenAI())
    await client.chat(MSG)
    call = client._client.chat.completions.calls[0]
    assert call["model"] == "nvidia/nemotron-3-super-120b-a12b"
    assert call["extra_body"] == {"chat_template_kwargs": {"enable_thinking": False}}
    assert call["temperature"] == 0.0 and call["max_tokens"] == 64
    assert "chat_template_kwargs" not in call  # must not be sent as a top-level OpenAI argument


def test_missing_api_key_gives_clear_error(monkeypatch):
    monkeypatch.delenv("NVIDIA_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="NVIDIA_API_KEY"):
        ChatClient("m1", use_cache=False)


async def test_smoke_workflow_runs_in_toolkit_and_reports_llm_calls(tmp_path, monkeypatch):
    """End-to-end through the real toolkit runtime, with the endpoint faked."""
    from nat.builder.context import Context
    from nat.data_models.intermediate_step import IntermediateStepType
    from nat.runtime.loader import load_workflow

    monkeypatch.setenv("NVIDIA_API_KEY", "test-not-a-real-key")
    monkeypatch.setattr(llm, "AsyncOpenAI", FakeOpenAI)
    monkeypatch.setattr(llm, "DEFAULT_CACHE_PATH", tmp_path / "c.sqlite")
    monkeypatch.setattr(LLMCache.__init__, "__defaults__", (tmp_path / "c.sqlite", ))

    steps = []
    async with load_workflow(ROOT / "configs" / "smoke.yml") as session_manager:
        async with session_manager.run("capital of France?") as runner:
            Context.get().intermediate_step_manager.subscribe(on_next=steps.append)
            answer = await runner.result(to_type=str)

    assert answer == "Paris"
    ends = [s for s in steps if s.payload.event_type == IntermediateStepType.LLM_END]
    assert len(ends) == 1
    assert ends[0].payload.name == "nvidia/nemotron-3.5-super-vl-120b-a12b"  # `generator` in configs/llm.yml
    assert ends[0].payload.usage_info.token_usage.total_tokens == 15
