"""Runs the toolkit workflow for the API, with per-request overrides.

Overrides (retry cap, guardrails on/off, Wikipedia check on/off) change the workflow config, so each distinct
set of overrides gets its own loaded workflow: the merged YAML is written to data/cache/api_configs/<hash>.yml
and loaded with the toolkit's load_workflow. The few most recent variants stay loaded.
"""

import contextlib
import copy
import hashlib
import json
import time
from collections import OrderedDict

import yaml

from nat.builder.context import Context
from nat.data_models.intermediate_step import IntermediateStepType
from nat.runtime.loader import load_workflow
from nat.utils.io.yaml_tools import yaml_load

from src import paths
from src.guardrails.rails import GuardrailSettings
from src.state import ApiMetrics
from src.state import ApiResponse
from src.state import ApiSettings
from src.state import SystemResult

SYSTEM_CONFIG = paths.ROOT / "configs" / "system.yml"
VARIANT_DIR = paths.CACHE_DIR / "api_configs"
MAX_LOADED = 3  # each loaded variant holds its own index, BM25 and Guardrails instances


class UnknownRailError(ValueError):
    pass


def build_config(settings: ApiSettings, base: dict | None = None) -> tuple[dict, dict]:
    """(merged workflow config with overrides applied, the effective settings). Pure: no I/O when base is given."""
    config = copy.deepcopy(base if base is not None else yaml_load(SYSTEM_CONFIG))
    unknown = set(settings.rails) - set(GuardrailSettings.model_fields)
    if unknown:
        raise UnknownRailError(f"unknown rails: {sorted(unknown)}; valid: {sorted(GuardrailSettings.model_fields)}")
    if settings.retry_cap is not None:
        config["workflow"]["retry_cap"] = settings.retry_cap
    for rail, enabled in settings.rails.items():
        config["functions"]["guardrails"][rail] = enabled
    if settings.fact_check is False:
        config["functions"]["verifier"].pop("fact_checker", None)
    elif settings.fact_check is True and "fact_checker" not in config["functions"]["verifier"]:
        config["functions"]["verifier"]["fact_checker"] = "fact_checker"

    rails = config["functions"]["guardrails"]
    effective = {"retry_cap": config["workflow"]["retry_cap"],
                 "rails": {r: bool(rails.get(r, True)) for r in GuardrailSettings.model_fields},
                 "fact_check": "fact_checker" in config["functions"]["verifier"]}
    return config, effective


def metrics_from_steps(steps: list, latency_s: float, retries: int) -> ApiMetrics:
    ends = [s.payload for s in steps if s.payload.event_type == IntermediateStepType.LLM_END]
    usage = [p.usage_info.token_usage for p in ends if p.usage_info]
    return ApiMetrics(latency_s=round(latency_s, 3), llm_calls=len(ends),
                      cached_llm_calls=sum(bool((p.metadata or {}).get("cached")) for p in ends
                                           if isinstance(p.metadata, dict)),
                      prompt_tokens=sum(u.prompt_tokens for u in usage),
                      completion_tokens=sum(u.completion_tokens for u in usage), retries=retries)


def to_response(question: str, result: SystemResult, metrics: ApiMetrics, effective: dict) -> ApiResponse:
    return ApiResponse(question=question, answer=result.answer, abstained=result.abstained, blocked=result.blocked,
                       claims=result.claims, answer_verdict=result.answer_verdict, plan=result.plan,
                       attempts=result.attempts, retrieved=result.retrieved, guardrail_events=result.guardrail_events,
                       metrics=metrics, settings=effective)


class WorkflowPool:
    """Loaded workflows keyed by their config. Use as an async context manager (closes them all)."""

    def __init__(self):
        self._loaded: OrderedDict[str, tuple[contextlib.AsyncExitStack, object]] = OrderedDict()

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        for stack, _ in self._loaded.values():
            await stack.aclose()
        self._loaded.clear()

    async def _session_manager(self, config: dict):
        key = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()[:16]
        if key in self._loaded:
            self._loaded.move_to_end(key)
            return self._loaded[key][1]
        VARIANT_DIR.mkdir(parents=True, exist_ok=True)
        path = VARIANT_DIR / f"{key}.yml"
        path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
        stack = contextlib.AsyncExitStack()
        session_manager = await stack.enter_async_context(load_workflow(path))
        self._loaded[key] = (stack, session_manager)
        if len(self._loaded) > MAX_LOADED:
            _, (old_stack, _) = self._loaded.popitem(last=False)
            await old_stack.aclose()
        return session_manager

    async def ask(self, question: str, settings: ApiSettings) -> ApiResponse:
        config, effective = build_config(settings)
        session_manager = await self._session_manager(config)
        steps = []
        start = time.perf_counter()
        async with session_manager.run(question) as runner:
            Context.get().intermediate_step_manager.subscribe(on_next=steps.append)
            result = await runner.result(to_type=SystemResult)
        return to_response(question, result, metrics_from_steps(steps, time.perf_counter() - start, result.retries),
                           effective)
