"""Invariants of the frozen corpus and eval sets (data/eval/manifest.json)."""

import hashlib
import json

import pytest

from src import paths
from src.state import EvalItem
from src.state import Passage

pytestmark = pytest.mark.skipif(not (paths.EVAL_DIR / "manifest.json").exists(),
                                reason="eval sets not built (python scripts/make_eval_sets.py)")


@pytest.fixture(scope="module")
def manifest():
    return json.loads((paths.EVAL_DIR / "manifest.json").read_text())


@pytest.fixture(scope="module")
def corpus():
    return {p.doc_id: p for p in (Passage(**json.loads(l)) for l in open(paths.CORPUS_PATH, encoding="utf-8"))}


def load(split):
    return [EvalItem(**json.loads(l)) for l in open(paths.EVAL_DIR / f"{split}.jsonl", encoding="utf-8")]


def test_files_match_frozen_hashes(manifest):
    files = {"corpus": paths.CORPUS_PATH, "test": paths.EVAL_DIR / "test.jsonl", "dev": paths.EVAL_DIR / "dev.jsonl",
             "pilot": paths.EVAL_DIR / "pilot.jsonl"}
    for name, path in files.items():
        assert hashlib.sha256(path.read_bytes()).hexdigest() == manifest["sha256"][name], f"{name} changed"


def test_counts_match_manifest(manifest, corpus):
    assert len(corpus) == manifest["counts"]["corpus_passages"]
    for split in ("test", "dev", "pilot"):
        counts = {}
        for item in load(split):
            counts[item.item_type] = counts.get(item.item_type, 0) + 1
        assert counts == manifest["counts"][split]


def test_dev_and_test_are_disjoint_and_pilot_is_dev_only():
    test_ids, dev_ids = {i.qid for i in load("test")}, {i.qid for i in load("dev")}
    assert not test_ids & dev_ids
    assert {i.qid for i in load("pilot")} <= dev_ids


def test_gold_passages_exist_and_unanswerable_have_none(corpus):
    for item in load("test") + load("dev"):
        if item.item_type == "unanswerable":
            assert item.answer is None and not item.gold_doc_ids
        else:
            assert item.answer and item.gold_doc_ids
            assert all(d in corpus for d in item.gold_doc_ids)
            assert all(corpus[d].source == "hotpotqa" for d in item.gold_doc_ids)  # poisoned copies are never gold


def test_adversarial_items_have_unique_canaries_and_poisoned_docs(corpus):
    items = [i for i in load("test") + load("dev") if i.item_type.startswith("adversarial")]
    canaries = [i.canary for i in items]
    assert all(canaries) and len(set(canaries)) == len(canaries)
    for item in items:
        if item.item_type == "adversarial_question":
            assert item.canary in item.question
        else:
            poisoned = corpus[item.injected_doc_id]
            assert poisoned.source == "injected" and item.canary in poisoned.text
            assert item.canary not in item.question
