"""Cross-check wrapper around pytrec_eval.

Used only to validate ``retrieval_metrics`` (our from-scratch derivations)
against the reference trec_eval implementation. Does NOT replace the
from-scratch metrics — gate IR-2 requires those to stay hand-rolled.
"""
from __future__ import annotations

from collections.abc import Iterable

Qrels = dict[str, dict[str, int]]
Run = dict[str, list[tuple[str, float]]]


def evaluate_pytrec(
    qrels: Qrels,
    run: Run,
    ks: Iterable[int] = (1, 5, 10, 100),
) -> dict[str, float]:
    """Aggregate nDCG@k and Recall@k via pytrec_eval, plus uncut recip_rank.

    pytrec_eval's ``ndcg_cut_k`` uses the trec_eval definition with
    gain = ``2^rel - 1`` — matches our from-scratch ``dcg_at_k`` so a bit-exact
    match on nDCG is expected.
    """
    import pytrec_eval

    run_pt = {q: {d: float(s) for d, s in items} for q, items in run.items()}
    measures: set[str] = {"recip_rank"}
    for k in ks:
        measures.add(f"ndcg_cut_{k}")
        measures.add(f"recall_{k}")

    evaluator = pytrec_eval.RelevanceEvaluator(qrels, measures)
    per_q = evaluator.evaluate(run_pt)

    def _mean(key: str) -> float:
        vals = [m[key] for m in per_q.values() if key in m]
        return sum(vals) / len(vals) if vals else 0.0

    out: dict[str, float] = {}
    for k in ks:
        out[f"nDCG@{k}"] = _mean(f"ndcg_cut_{k}")
        out[f"Recall@{k}"] = _mean(f"recall_{k}")
    out["MRR@full"] = _mean("recip_rank")
    return out
