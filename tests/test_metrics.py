import pytest

from rag_eval_lab.metrics import (
    hit_at_k,
    mean,
    percentile,
    recall_at_k,
    reciprocal_rank,
    unique_in_order,
)


def test_unique_in_order():
    assert unique_in_order(["a", "b", "a", "c", "b"]) == ["a", "b", "c"]


def test_recall():
    assert recall_at_k(["a", "b"], ["a", "c"]) == 0.5
    assert recall_at_k(["a", "c"], ["a", "c"]) == 1.0
    with pytest.raises(ValueError):
        recall_at_k(["a"], [])


def test_hit():
    assert hit_at_k(["x", "a"], ["a"]) == 1.0
    assert hit_at_k(["x"], ["a"]) == 0.0


def test_reciprocal_rank_uses_unique_doc_ranking():
    assert reciprocal_rank(["a", "b"], ["a"]) == 1.0
    assert reciprocal_rank(["x", "x", "a"], ["a"]) == 0.5  # duplicate x counts once
    assert reciprocal_rank(["x", "y"], ["a"]) == 0.0


def test_mean_ignores_none():
    assert mean([1.0, None, 3.0]) == 2.0
    assert mean([None]) is None


def test_percentile():
    data = list(range(1, 101))
    assert percentile(data, 95) == 95
    assert percentile(data, 50) == 50
    assert percentile([], 95) is None
    assert percentile([7.0], 99) == 7.0
