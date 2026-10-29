"""Reciprocal rank fusion (src/retrieval/fusion.py)."""

import pytest

from src.retrieval.fusion import reciprocal_rank_fusion


def test_doc_in_both_lists_beats_doc_in_one():
    fused = reciprocal_rank_fusion([["a", "b"], ["b", "c"]], k=60)
    assert [d for d, _ in fused][0] == "b"
    assert dict(fused)["b"] == pytest.approx(1 / 62 + 1 / 61)


def test_scores_follow_formula():
    fused = dict(reciprocal_rank_fusion([["a", "b", "c"]], k=10))
    assert fused == pytest.approx({"a": 1 / 11, "b": 1 / 12, "c": 1 / 13})


def test_ties_broken_by_first_appearance():
    fused = reciprocal_rank_fusion([["x", "y"], ["y", "x"]], k=60)  # identical scores
    assert [d for d, _ in fused] == ["x", "y"]


def test_duplicate_in_one_list_counts_once_at_best_rank():
    fused = dict(reciprocal_rank_fusion([["a", "a", "b"]], k=60))
    assert fused["a"] == pytest.approx(1 / 61)
    assert fused["b"] == pytest.approx(1 / 63)  # b keeps its original rank


def test_empty_inputs():
    assert reciprocal_rank_fusion([]) == []
    assert reciprocal_rank_fusion([[], []]) == []
