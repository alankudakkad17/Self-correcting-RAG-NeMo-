"""Ping every model in configs/llm.yml once (cached) and report which ones answer.

Usage: python scripts/smoke_models.py [--only generator,verifier]
Writes eval/results/smoke_models.json with model names, latency and token counts.
"""

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from nat.llm.nim_llm import NIMModelConfig  # noqa: E402
from nat.utils.io.yaml_tools import yaml_load  # noqa: E402

from src.llm import ChatClient  # noqa: E402

PROMPT = [{"role": "user", "content": "Reply with exactly one word: the capital of France."}]


async def ping(name: str, cfg: dict) -> dict:
    llm_cfg = NIMModelConfig(**{k: v for k, v in cfg.items() if k != "_type"})
    try:
        result = await ChatClient.from_toolkit_config(llm_cfg).chat(PROMPT)
        return {"role": name, "model": llm_cfg.model_name, "ok": True, "reply": result.text[:80],
                "cached": result.cached, "latency_s": round(result.latency_s, 2),
                "prompt_tokens": result.prompt_tokens, "completion_tokens": result.completion_tokens}
    except Exception as e:
        return {"role": name, "model": llm_cfg.model_name, "ok": False, "error": f"{type(e).__name__}: {e}"[:300]}


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", default="", help="comma-separated llm names from configs/llm.yml")
    args = parser.parse_args()

    llms = yaml_load(ROOT / "configs" / "llm.yml")["llms"]
    names = [n for n in llms if not args.only or n in args.only.split(",")]
    results = [await ping(n, llms[n]) for n in names]  # sequential: kind to the free-tier rate limit

    for r in results:
        status = "OK  " if r["ok"] else "FAIL"
        detail = f"{r['reply']!r} ({r['latency_s']}s, cached={r['cached']})" if r["ok"] else r["error"]
        print(f"{status} {r['role']:<18} {r['model']:<42} {detail}")

    out = ROOT / "eval" / "results" / "smoke_models.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"timestamp": time.time(), "results": results}, indent=2))
    sys.exit(0 if all(r["ok"] for r in results) else 1)


if __name__ == "__main__":
    asyncio.run(main())
