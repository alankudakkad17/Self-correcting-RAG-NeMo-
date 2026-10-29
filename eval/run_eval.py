"""Run one system over a frozen eval split and score it.

Usage:
  python eval/run_eval.py --config configs/A.yml --split pilot
  python eval/run_eval.py --config configs/B.yml --split test --limit 50

Writes eval/results/runs/<timestamp>_<system>_<split>/:
  run_config.json   config files, model names, data manifest hash, args
  predictions.jsonl one line per item: system output, gold, latency, LLM calls, tokens, errors
  metrics.json      deterministic metrics (eval/metrics.py)
Questions run one at a time to stay within the free-tier rate limit.
"""

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from nat.builder.context import Context  # noqa: E402
from nat.data_models.intermediate_step import IntermediateStepType  # noqa: E402
from nat.runtime.loader import load_workflow  # noqa: E402
from nat.utils.io.yaml_tools import yaml_load  # noqa: E402

from eval.metrics import summarize  # noqa: E402
from src import paths  # noqa: E402
from src.state import EvalItem  # noqa: E402
from src.state import SystemResult  # noqa: E402


def load_items(split: str, limit: int | None) -> list[EvalItem]:
    with open(paths.EVAL_DIR / f"{split}.jsonl", encoding="utf-8") as f:
        items = [EvalItem(**json.loads(line)) for line in f]
    return items[:limit] if limit else items


def run_config(config_path: Path, args) -> dict:
    resolved = yaml_load(config_path)  # with `base: llm.yml` merged in
    manifest = json.loads((paths.EVAL_DIR / "manifest.json").read_text())
    return {
        "args": vars(args),
        "config_file": str(config_path.relative_to(ROOT)),
        "config_text": config_path.read_text(),
        "llm_yml_text": (ROOT / "configs" / "llm.yml").read_text(),
        "models": {name: cfg.get("model_name") for name, cfg in resolved.get("llms", {}).items()},
        "embedders": {name: cfg.get("model_name") for name, cfg in resolved.get("embedders", {}).items()},
        "workflow": resolved.get("workflow"),
        "functions": resolved.get("functions"),
        "data_manifest_sha256": manifest["sha256"],
        "data_version": manifest["version"],
        "llm_cache_enabled": os.environ.get("LLM_CACHE_ENABLED", "1") != "0",
        "started_unix": int(time.time()),
    }


async def run_item(session_manager, item: EvalItem) -> dict:
    steps = []
    row = {"qid": item.qid, "item_type": item.item_type, "question": item.question, "answer": item.answer,
           "gold_doc_ids": item.gold_doc_ids, "canary": item.canary, "injected_doc_id": item.injected_doc_id,
           "result": None, "error": None}
    start = time.perf_counter()
    try:
        async with session_manager.run(item.question) as runner:
            Context.get().intermediate_step_manager.subscribe(on_next=steps.append)
            result = await runner.result(to_type=SystemResult)
        row["result"] = result.model_dump()
    except Exception as e:  # record and continue: one failure must not kill a long run
        row["error"] = f"{type(e).__name__}: {e}"[:500]
    row["latency_s"] = time.perf_counter() - start

    ends = [s.payload for s in steps if s.payload.event_type == IntermediateStepType.LLM_END]
    row["llm_calls"] = len(ends)
    row["cached_llm_calls"] = sum(bool((p.metadata or {}).get("cached")) for p in ends if isinstance(p.metadata, dict))
    row["prompt_tokens"] = sum(p.usage_info.token_usage.prompt_tokens for p in ends if p.usage_info)
    row["completion_tokens"] = sum(p.usage_info.token_usage.completion_tokens for p in ends if p.usage_info)
    return row


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="system YAML, e.g. configs/A.yml")
    parser.add_argument("--split", default="pilot", choices=["pilot", "dev", "test"])
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--tag", default="", help="optional suffix for the run directory")
    args = parser.parse_args()

    config_path = (ROOT / args.config).resolve()
    items = load_items(args.split, args.limit)
    cfg = run_config(config_path, args)
    system = (cfg["workflow"] or {}).get("system_id", config_path.stem)

    run_dir = paths.RESULTS_DIR / "runs" / f"{time.strftime('%Y%m%d-%H%M%S')}_{system}_{args.split}{args.tag}"
    run_dir.mkdir(parents=True)
    (run_dir / "run_config.json").write_text(json.dumps(cfg, indent=2))

    rows = []
    async with load_workflow(config_path) as session_manager:
        with open(run_dir / "predictions.jsonl", "w", encoding="utf-8") as out:
            for n, item in enumerate(items, start=1):
                row = await run_item(session_manager, item)
                rows.append(row)
                out.write(json.dumps(row) + "\n")
                out.flush()
                answer = row["error"] or (row["result"] or {}).get("answer")
                print(f"[{n}/{len(items)}] {item.item_type:<21} gold={item.answer!r:<28.28} got={answer!r:.60}")

    metrics = summarize(rows)
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics["summary"], indent=2))
    print(f"\nrun written to {run_dir.relative_to(ROOT)}")


if __name__ == "__main__":
    asyncio.run(main())
