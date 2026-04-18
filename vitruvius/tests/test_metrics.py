"""Hand-checked metric values.

These cases are easy enough to compute on paper. They are the safety net for
the from-scratch retrieval_metrics implementation.
"""
from __future__ import annotations

import math

from vitruvius.evaluation.retrieval_metrics import (
    dcg_at_k,
    mean_metric,
    mrr_at_k,
    ndcg_at_k,
    recall_at_k,
)


def test_dcg_simple():
    # ranks = [3, 2, 3, 0, 1, 2]; DCG@6 hand value
    rels = [3, 2, 3, 0, 1, 2]
    dcg = dcg_at_k(rels, 6)
    expected = (
        (2**3 - 1) / math.log2(2)
        + (2**2 - 1) / math.log2(3)
        + (2**3 - 1) / math.log2(4)
        + 0
        + (2**1 - 1) / math.log2(6)
        + (2**2 - 1) / math.log2(7)
    )
    assert math.isclose(dcg, expected, rel_tol=1e-9)


def test_ndcg_perfect_ranking_is_one():
    qrels = {"q": {"d1": 1, "d2": 1, "d3": 1}}
    run = {"q": [("d1", 0.9), ("d2", 0.8), ("d3", 0.7), ("d4", 0.6)]}
    out = ndcg_at_k(qrels, run, 3)
    assert math.isclose(out["q"], 1.0, rel_tol=1e-9)


def test_ndcg_no_relevant_excluded():
    qrels = {"q": {"d1": 1}, "r": {}}
    run = {"q": [("d1", 1.0)], "r": [("d99", 1.0)]}
    out = ndcg_at_k(qrels, run, 5)
    assert "q" in out and "r" not in out


def test_recall_at_k():
    qrels = {"q": {"a": 1, "b": 1, "c": 1, "d": 1}}
    run = {"q": [("a", 1), ("x", 0.9), ("c", 0.8), ("y", 0.7)]}
    out = recall_at_k(qrels, run, 4)
    assert math.isclose(out["q"], 0.5, rel_tol=1e-9)


def test_mrr_at_k_first_rank():
    qrels = {"q": {"hit": 1}}
    run = {"q": [("a", 1), ("b", 0.9), ("hit", 0.8), ("c", 0.7)]}
    out = mrr_at_k(qrels, run, 5)
    assert math.isclose(out["q"], 1 / 3, rel_tol=1e-9)


def test_mrr_at_k_misses_in_topk():
    qrels = {"q": {"hit": 1}}
    run = {"q": [("a", 1), ("b", 0.9), ("c", 0.8)]}
    out = mrr_at_k(qrels, run, 3)
    assert out["q"] == 0.0


def test_mean_metric_empty():
    assert mean_metric({}) == 0.0
