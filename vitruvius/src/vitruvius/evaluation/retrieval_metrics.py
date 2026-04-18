"""Retrieval metrics derived from scratch.

This module deliberately does NOT delegate to ``pytrec_eval`` or ``ranx``. The
formulas are kept in docstrings so the implementation doubles as study notes
for interview gate IR-2 ("derive nDCG@10 on paper").

All functions accept the same shape:

* ``qrels``: ``dict[qid, dict[docid, relevance_int]]`` (relevance >= 1 means
  relevant; 0 or missing means non-relevant).
* ``run``:   ``dict[qid, list[(docid, score)]]``, ranked best-first per qid.
* ``k``:     truncation depth.

They return a ``dict[qid, float]``. Use :func:`mean_metric` to reduce to a
single number across queries.
"""
from __future__ import annotations

import math
from collections.abc import Iterable

Qrels = dict[str, dict[str, int]]
Run = dict[str, list[tuple[str, float]]]


def _topk(ranking: list[tuple[str, float]], k: int) -> list[tuple[str, float]]:
    return ranking[:k]


def recall_at_k(qrels: Qrels, run: Run, k: int) -> dict[str, float]:
    """Recall@k = (# relevant docs retrieved in top-k) / (# relevant docs total).

    Skips queries with zero relevant docs (they cannot contribute a
    well-defined recall). Use a constant for those if needed downstream.
    """
    out: dict[str, float] = {}
    for qid, ranking in run.items():
        rel = {d for d, r in qrels.get(qid, {}).items() if r >= 1}
        if not rel:
            continue
        retrieved = {d for d, _ in _topk(ranking, k)}
        out[qid] = len(rel & retrieved) / len(rel)
    return out


def mrr_at_k(qrels: Qrels, run: Run, k: int) -> dict[str, float]:
    """MRR@k = mean over queries of 1/rank(first relevant), 0 if none in top-k.

    rank is 1-indexed: the top-1 doc has rank 1, contributing 1/1 = 1.0.
    """
    out: dict[str, float] = {}
    for qid, ranking in run.items():
        rel = {d for d, r in qrels.get(qid, {}).items() if r >= 1}
        if not rel:
            continue
        score = 0.0
        for rank, (docid, _) in enumerate(_topk(ranking, k), start=1):
            if docid in rel:
                score = 1.0 / rank
                break
        out[qid] = score
    return out


def dcg_at_k(rels: Iterable[int], k: int) -> float:
    """DCG@k using the standard "gain = 2^rel - 1, discount = log2(rank+1)" form.

        DCG@k = sum_{i=1..k} (2^{rel_i} - 1) / log2(i + 1)
    """
    total = 0.0
    for i, rel in enumerate(list(rels)[:k], start=1):
        if rel <= 0:
            continue
        gain = (2 ** rel) - 1
        discount = math.log2(i + 1)
        total += gain / discount
    return total


def ndcg_at_k(qrels: Qrels, run: Run, k: int) -> dict[str, float]:
    """nDCG@k = DCG@k / iDCG@k where iDCG is the DCG of the ideal ranking."""
    out: dict[str, float] = {}
    for qid, ranking in run.items():
        q_rel = qrels.get(qid, {})
        if not any(r >= 1 for r in q_rel.values()):
            continue
        retrieved_rels = [q_rel.get(d, 0) for d, _ in _topk(ranking, k)]
        ideal_rels = sorted(q_rel.values(), reverse=True)
        idcg = dcg_at_k(ideal_rels, k)
        if idcg == 0.0:
            continue
        out[qid] = dcg_at_k(retrieved_rels, k) / idcg
    return out


def mean_metric(scores: dict[str, float]) -> float:
    """Mean over query scores. 0.0 if scores is empty."""
    if not scores:
        return 0.0
    return sum(scores.values()) / len(scores)


def evaluate(qrels: Qrels, run: Run, ks: tuple[int, ...] = (1, 5, 10, 100)) -> dict[str, float]:
    """Compute nDCG@k, Recall@k, MRR@k for several k and return a flat dict."""
    flat: dict[str, float] = {}
    for k in ks:
        flat[f"nDCG@{k}"] = mean_metric(ndcg_at_k(qrels, run, k))
        flat[f"Recall@{k}"] = mean_metric(recall_at_k(qrels, run, k))
        flat[f"MRR@{k}"] = mean_metric(mrr_at_k(qrels, run, k))
    return flat
