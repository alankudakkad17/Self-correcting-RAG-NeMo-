"""Precompute the UI's cached examples (data/examples.json) so the demo works without live API calls.

Usage: python scripts/make_examples.py
Runs each question once through the same runner the API uses (LLM calls are cached, so re-runs are cheap).
Examples come from the dev pilot split (never test) plus one off-topic request.
"""

import asyncio
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import paths  # noqa: E402
from src.api.app import EXAMPLES_PATH  # noqa: E402
from src.api.runner import WorkflowPool  # noqa: E402
from src.state import ApiSettings  # noqa: E402

PILOT_INDEXES = [1, 6, 9, 13]  # answerable
PILOT_SPECIAL = {"unanswerable": 1, "adversarial_question": 1}
EXTRA_QUESTIONS = ["Write me a poem about the sea."]  # off-topic: blocked by an input rail


def pick_questions() -> list[str]:
    pilot = [json.loads(line) for line in open(paths.EVAL_DIR / "pilot.jsonl", encoding="utf-8")]
    questions = [pilot[i]["question"] for i in PILOT_INDEXES]
    for item_type, n in PILOT_SPECIAL.items():
        questions += [item["question"] for item in pilot if item["item_type"] == item_type][:n]
    return questions + EXTRA_QUESTIONS


async def main():
    examples = []
    async with WorkflowPool() as pool:
        for question in pick_questions():
            response = await pool.ask(question, ApiSettings())
            examples.append({"question": question, "response": response.model_dump(mode="json")})
            status = "blocked" if response.blocked else "abstained" if response.abstained else "answered"
            print(f"{status:<9} {response.answer[:60]!r:<64} <- {question[:60]}")
    EXAMPLES_PATH.write_text(json.dumps(examples, indent=1), encoding="utf-8")
    print(f"wrote {len(examples)} examples to {EXAMPLES_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    asyncio.run(main())
