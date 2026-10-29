"""RAGAS scoring of a finished run (standalone; see setup.md for why not the toolkit's RAGAS evaluator).

Usage:
  python eval/ragas_eval.py --run eval/results/runs/<run_dir> [--judge judge] [--limit N]

Writes <run_dir>/ragas_<judge>.json with per-item scores and means.

- Judge: an entry under `llms:` in configs/llm.yml (default `judge`), temperature 0, thinking off.
  It must not be the generator; the script refuses if it is.
- Every judge and embedding call goes through an HTTP-level disk cache (src/llm.py: CachingTransport).
- Short-span answers are wrapped as "The answer to the question "<q>" is <a>." for the answer metrics,
  because claim-based metrics are unreliable on bare one-word answers.
- Which items get which metric:
    items with a reference answer (answerable + adversarial); unanswerable items are skipped
    faithfulness, response_relevancy: skipped for abstentions (nothing to judge)
    context metrics: only for systems that retrieve
"""

import argparse
import asyncio
import inspect
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
from nat.utils.io.yaml_tools import yaml_load  # noqa: E402
from ragas.embeddings.base import BaseRagasEmbedding  # noqa: E402
from ragas.llms import llm_factory  # noqa: E402
from ragas.metrics import collections as rm  # noqa: E402

from src.llm import cached_async_openai  # noqa: E402
from src.retrieval.vector import Embedder  # noqa: E402

CONCURRENCY = 4  # judge calls in flight; keeps us near the free-tier rate limit


class CachedNvidiaEmbeddings(BaseRagasEmbedding):
    """RAGAS embeddings backed by our cached NVIDIA embedder (input_type=query: we compare questions)."""

    def __init__(self, model: str, base_url: str | None):
        super().__init__()
        self._embedder = Embedder(model, base_url=base_url)

    async def aembed_text(self, text: str, **kwargs) -> list[float]:
        return (await self._embedder.embed([text], input_type="query"))[0].tolist()

    def embed_text(self, text: str, **kwargs) -> list[float]:
        return asyncio.run(self.aembed_text(text))


def as_sentence(question: str, answer: str) -> str:
    return f'The answer to the question "{question}" is {answer}.'


def build_metrics(llm, embeddings) -> dict:
    return {
        "faithfulness": rm.Faithfulness(llm=llm),
        "answer_correctness": rm.AnswerCorrectness(llm=llm, embeddings=embeddings),
        "factual_correctness": rm.FactualCorrectness(llm=llm),
        "context_precision": rm.ContextPrecisionWithReference(llm=llm),
        "context_recall": rm.ContextRecall(llm=llm),
        "response_relevancy": rm.AnswerRelevancy(llm=llm, embeddings=embeddings),
        "noise_sensitivity": rm.NoiseSensitivity(llm=llm),  # lower is better
    }


def metric_inputs(row: dict) -> dict | None:
    """All fields any metric might need, or None if the item is not scored."""
    result = row.get("result")
    if row.get("error") or not result or row["answer"] is None:
        return None
    contexts = [d["text"] for d in result.get("retrieved", [])]
    return {
        "user_input": row["question"],
        "response": as_sentence(row["question"], result["answer"]),
        "reference": as_sentence(row["question"], row["answer"]),
        "retrieved_contexts": contexts,
        "_abstained": result["abstained"],
    }


def applicable(name: str, inputs: dict) -> bool:
    if name in ("faithfulness", "response_relevancy", "noise_sensitivity") and inputs["_abstained"]:
        return False
    if name in ("faithfulness", "context_precision", "context_recall", "noise_sensitivity"):
        return bool(inputs["retrieved_contexts"])
    return True


async def score_one(sem, metric, name, qid, inputs):
    wanted = inspect.signature(metric.ascore).parameters
    kwargs = {k: v for k, v in inputs.items() if k in wanted}
    async with sem:
        try:
            result = await metric.ascore(**kwargs)
            return qid, name, float(result.value), None
        except Exception as e:  # keep going; failures are counted in the output
            return qid, name, None, f"{type(e).__name__}: {e}"[:300]


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True)
    parser.add_argument("--judge", default="judge", help="llms: entry in configs/llm.yml")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--metrics", default="", help="comma-separated subset")
    args = parser.parse_args()

    run_dir = (ROOT / args.run).resolve()
    run_cfg = json.loads((run_dir / "run_config.json").read_text())
    llms = yaml_load(ROOT / "configs" / "llm.yml")
    judge_cfg = llms["llms"][args.judge]
    if judge_cfg["model_name"] in run_cfg["models"].get("generator", ""):
        sys.exit("judge model must differ from the generator")
    emb_cfg = llms["embedders"]["corpus_embedder"]

    client, transport = cached_async_openai(base_url=judge_cfg.get("base_url"))
    extra = {"extra_body": {"chat_template_kwargs": dict(judge_cfg["chat_template_kwargs"])}} \
        if judge_cfg.get("chat_template_kwargs") else {}
    llm = llm_factory(judge_cfg["model_name"], client=client, temperature=0.0, max_tokens=2048, **extra)
    metrics = build_metrics(llm, CachedNvidiaEmbeddings(emb_cfg["model_name"], emb_cfg.get("base_url")))
    if args.metrics:
        metrics = {k: v for k, v in metrics.items() if k in args.metrics.split(",")}

    rows = [json.loads(line) for line in open(run_dir / "predictions.jsonl", encoding="utf-8")]
    rows = rows[:args.limit] if args.limit else rows
    sem = asyncio.Semaphore(CONCURRENCY)
    tasks = []
    for row in rows:
        inputs = metric_inputs(row)
        if inputs is None:
            continue
        for name, metric in metrics.items():
            if applicable(name, inputs):
                tasks.append(score_one(sem, metric, name, row["qid"], inputs))

    start = time.perf_counter()
    per_item: dict[str, dict] = {}
    errors = []
    for n, done in enumerate(asyncio.as_completed(tasks), start=1):
        qid, name, value, err = await done
        per_item.setdefault(qid, {})[name] = value
        if err:
            errors.append({"qid": qid, "metric": name, "error": err})
        if n % 20 == 0 or n == len(tasks):
            print(f"[{n}/{len(tasks)}] {time.perf_counter() - start:.0f}s  cache hits={transport.hits} "
                  f"misses={transport.misses} errors={len(errors)}")

    means = {}
    for name in metrics:
        values = [s[name] for s in per_item.values() if s.get(name) is not None]
        means[name] = {"mean": float(np.mean(values)) if values else None, "n": len(values)}
    out = {"judge": args.judge, "judge_model": judge_cfg["model_name"], "system": run_cfg["workflow"].get("system_id"),
           "means": means, "items": per_item, "errors": errors,
           "cache": {"hits": transport.hits, "misses": transport.misses}}
    out_path = run_dir / f"ragas_{args.judge}.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(json.dumps(means, indent=2))
    print(f"written to {out_path.relative_to(ROOT)}")


if __name__ == "__main__":
    asyncio.run(main())
