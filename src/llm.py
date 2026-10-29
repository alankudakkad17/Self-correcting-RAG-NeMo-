"""Cached chat client for NVIDIA-hosted models (OpenAI-compatible API).

Why our own client: the toolkit's builder.get_llm() only returns framework wrappers
(LangChain, LlamaIndex, ...). We read the model settings from the toolkit config
(builder.get_llm_config) and call the endpoint with the `openai` SDK directly.

Every call is cached on disk by sha256 of (model, messages, params), so re-runs cost
no API quota. Each call is also reported to the toolkit profiler as LLM_START/LLM_END
intermediate steps when it runs inside a toolkit workflow.
"""

import hashlib
import json
import logging
import os
import sqlite3
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import httpx
from dotenv import load_dotenv
from openai import AsyncOpenAI

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CACHE_PATH = PROJECT_ROOT / "data" / "cache" / "llm_cache.sqlite"
DEFAULT_BASE_URL = "https://integrate.api.nvidia.com/v1"

# .env is git-ignored; real environment variables win over it.
load_dotenv(PROJECT_ROOT / ".env", override=False)


@dataclass
class LLMResult:
    text: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    cached: bool
    latency_s: float  # wall time of this call (near 0 on a cache hit)


def cache_key(model: str, messages: list[dict], params: dict) -> str:
    """Stable hash of everything that can change the model's output."""
    blob = json.dumps({"model": model, "messages": messages, "params": params}, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


class LLMCache:
    """Tiny sqlite key-value store. Safe to share across threads in one process."""

    def __init__(self, path: Path = DEFAULT_CACHE_PATH):
        path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(path, check_same_thread=False)
        self._lock = threading.Lock()
        with self._lock:
            self._conn.execute("CREATE TABLE IF NOT EXISTS cache (key TEXT PRIMARY KEY, value TEXT, created REAL)")
            self._conn.commit()

    def get(self, key: str) -> dict | None:
        with self._lock:
            row = self._conn.execute("SELECT value FROM cache WHERE key = ?", (key, )).fetchone()
        return json.loads(row[0]) if row else None

    def put(self, key: str, value: dict) -> None:
        with self._lock:
            self._conn.execute("INSERT OR REPLACE INTO cache VALUES (?, ?, ?)", (key, json.dumps(value), time.time()))
            self._conn.commit()


def _step_manager():
    """The toolkit's intermediate-step manager for the current run, or None outside a toolkit workflow.

    Fetch it once per LLM call: Context.get() returns a new Context (and a new manager) each time,
    and a manager only matches an END event to a START that it pushed itself.
    """
    try:
        from nat.builder.context import Context
        return Context.get().intermediate_step_manager
    except Exception:
        return None


def _emit_profiler_step(manager, event: str, uuid: str | None, model: str, messages: list[dict],
                        result: LLMResult | None):
    """Report an LLM call to the toolkit profiler. Returns the step UUID so END can match START."""
    if manager is None:
        return None
    try:
        from nat.data_models.intermediate_step import IntermediateStepPayload
        from nat.data_models.intermediate_step import IntermediateStepType
        from nat.data_models.intermediate_step import StreamEventData
        from nat.data_models.intermediate_step import UsageInfo
        from nat.data_models.token_usage import TokenUsageBaseModel

        if event == "start":
            payload = IntermediateStepPayload(event_type=IntermediateStepType.LLM_START,
                                              name=model,
                                              data=StreamEventData(input=messages[-1]["content"]))
        else:
            usage = TokenUsageBaseModel(prompt_tokens=result.prompt_tokens,
                                        completion_tokens=result.completion_tokens,
                                        total_tokens=result.prompt_tokens + result.completion_tokens)
            payload = IntermediateStepPayload(event_type=IntermediateStepType.LLM_END,
                                              UUID=uuid,
                                              name=model,
                                              metadata={"cached": result.cached},
                                              data=StreamEventData(output=result.text),
                                              usage_info=UsageInfo(token_usage=usage, num_llm_calls=1))
        manager.push_intermediate_step(payload)
        return payload.UUID
    except Exception as e:  # e.g. no active run in this context (scripts, unit tests)
        logger.debug("profiler step skipped: %s", e)
        return None


class ChatClient:
    """One client per model config. `params` are fixed per model (temperature, max_tokens, ...)."""

    def __init__(self,
                 model: str,
                 params: dict | None = None,
                 base_url: str | None = None,
                 api_key: str | None = None,
                 cache: LLMCache | None = None,
                 use_cache: bool | None = None,
                 openai_client: AsyncOpenAI | None = None):
        self.model = model
        self.params = params or {}
        if use_cache is None:
            use_cache = os.environ.get("LLM_CACHE_ENABLED", "1") != "0"
        self.cache = (cache or LLMCache()) if use_cache else None
        if openai_client is None:
            key = api_key or os.environ.get("NVIDIA_API_KEY")
            if not key:
                raise RuntimeError("NVIDIA_API_KEY is not set (put it in .env or the environment).")
            # The SDK retries 429/5xx with exponential backoff and honours Retry-After.
            openai_client = AsyncOpenAI(base_url=base_url or os.environ.get("NVIDIA_BASE_URL", DEFAULT_BASE_URL),
                                        api_key=key,
                                        max_retries=6,
                                        timeout=120)
        self._client = openai_client

    @classmethod
    def from_toolkit_config(cls, llm_config, **kwargs) -> "ChatClient":
        """Build from a toolkit `nim` LLM config (builder.get_llm_config(name))."""
        params = {"temperature": llm_config.temperature, "max_tokens": llm_config.max_tokens}
        if llm_config.top_p is not None:
            params["top_p"] = llm_config.top_p
        # Extra YAML keys are allowed on the nim config; we use chat_template_kwargs for thinking on/off
        # because the toolkit's `thinking` field does not recognise nvidia/nemotron-3* model names.
        extra = getattr(llm_config, "chat_template_kwargs", None)
        if extra:
            params["chat_template_kwargs"] = dict(extra)
        params = {k: v for k, v in params.items() if v is not None}
        api_key = llm_config.api_key.get_secret_value() if llm_config.api_key else None
        return cls(llm_config.model_name, params, base_url=llm_config.base_url, api_key=api_key, **kwargs)

    async def chat(self, messages: list[dict], **overrides) -> LLMResult:
        params = {**self.params, **overrides}
        key = cache_key(self.model, messages, params)
        manager = _step_manager()
        step_id = _emit_profiler_step(manager, "start", None, self.model, messages, None)

        hit = self.cache.get(key) if self.cache else None
        if hit is not None:
            result = LLMResult(**{**hit, "cached": True, "latency_s": 0.0})
        else:
            try:
                result = await self._call_api(messages, params)
            except Exception:
                if step_id:  # close the open profiler span before re-raising
                    _emit_profiler_step(manager, "end", step_id, self.model, messages,
                                        LLMResult("", self.model, 0, 0, False, 0.0))
                raise
            if self.cache:
                self.cache.put(key, {k: v for k, v in result.__dict__.items() if k not in ("cached", "latency_s")})

        if step_id:
            _emit_profiler_step(manager, "end", step_id, self.model, messages, result)
        return result

    async def _call_api(self, messages: list[dict], params: dict) -> LLMResult:
        api_params = dict(params)
        extra_body = {}
        if "chat_template_kwargs" in api_params:
            extra_body["chat_template_kwargs"] = api_params.pop("chat_template_kwargs")
        start = time.perf_counter()
        resp = await self._client.chat.completions.create(model=self.model,
                                                          messages=messages,
                                                          extra_body=extra_body or None,
                                                          **api_params)
        usage = resp.usage
        return LLMResult(text=(resp.choices[0].message.content or "").strip(),
                         model=self.model,
                         prompt_tokens=usage.prompt_tokens if usage else 0,
                         completion_tokens=usage.completion_tokens if usage else 0,
                         cached=False,
                         latency_s=time.perf_counter() - start)


class CachingTransport(httpx.AsyncBaseTransport):
    """HTTP-level cache for libraries that drive the `openai` SDK themselves (RAGAS / Instructor judges).

    The key is the sha256 of the request path + canonical JSON body, i.e. (model, prompt, params).
    Only successful (200) responses are stored; errors and retries pass through untouched.
    """

    def __init__(self, cache: LLMCache, inner: httpx.AsyncBaseTransport | None = None):
        self.cache = cache
        self.inner = inner or httpx.AsyncHTTPTransport()
        self.hits = 0
        self.misses = 0

    @staticmethod
    def request_key(request: httpx.Request) -> str:
        try:
            body = json.dumps(json.loads(request.content or b"{}"), sort_keys=True, ensure_ascii=False)
        except ValueError:
            body = request.content.decode("utf-8", "replace")
        return hashlib.sha256(f"{request.url.host}{request.url.path}\n{body}".encode("utf-8")).hexdigest()

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        if request.method != "POST":
            return await self.inner.handle_async_request(request)
        key = self.request_key(request)
        hit = self.cache.get(key)
        if hit is not None:
            self.hits += 1
            return httpx.Response(200, content=hit["body"].encode("utf-8"),
                                  headers={"content-type": "application/json"}, request=request)
        self.misses += 1
        response = await self.inner.handle_async_request(request)
        if response.status_code == 200:
            body = await response.aread()
            self.cache.put(key, {"body": body.decode("utf-8")})
            return httpx.Response(200, content=body, headers={"content-type": "application/json"}, request=request)
        return response

    async def aclose(self) -> None:
        await self.inner.aclose()


HTTP_CACHE_PATH = DEFAULT_CACHE_PATH.parent / "http_cache.sqlite"


def cached_async_openai(cache_path: Path = HTTP_CACHE_PATH,
                        base_url: str | None = None) -> tuple[AsyncOpenAI, CachingTransport]:
    """AsyncOpenAI client for NVIDIA's endpoint whose every successful call is cached on disk."""
    transport = CachingTransport(LLMCache(cache_path))
    client = AsyncOpenAI(base_url=base_url or os.environ.get("NVIDIA_BASE_URL", DEFAULT_BASE_URL),
                         api_key=os.environ.get("NVIDIA_API_KEY"),
                         max_retries=6,
                         timeout=180,
                         http_client=httpx.AsyncClient(transport=transport, timeout=180))
    return client, transport
