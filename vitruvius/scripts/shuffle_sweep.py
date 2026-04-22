"""Phase 8 shuffle sweep.

For each (encoder × dataset × mode), encode + retrieve + score, writing:
    experiments/phase8/<encoder>__<dataset>__<mode>.json

Modes: docs-shuffled, queries-shuffled, both-shuffled. Baseline is already
in Phases 3 / 5 and not re-run here (the analysis step computes
`position_sensitivity = (baseline - shuffled) / baseline` by pulling
baseline from Phase 3/5 bench JSONs).

54 total runs = 6 encoders × 3 datasets × 3 modes. Each ShuffledEncoder
is instantiated once per (encoder, mode), reused across datasets.

Shuffle seed: 1729 (per handoff §8.2, same seed across all encoders for
fair cross-encoder comparison).
"""
from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import torch

from vitruvius import __version__
from vitruvius.data.beir_loader import load_beir
from vitruvius.encoders.shuffled import ShuffledEncoder
from vitruvius.evaluation.faiss_index import IndexWrapper
from vitruvius.evaluation.pytrec_bridge import evaluate_pytrec
from vitruvius.evaluation.retrieval_metrics import (
    evaluate,
    ndcg_at_k,
    recall_at_k,
)
from vitruvius.utils.logging import get_logger
from vitruvius.utils.seed import set_seed

_log = get_logger("shuffle_sweep")


MODES = {
    "docs-shuffled":    (False, True),
    "queries-shuffled": (True, False),
    "both-shuffled":    (True, True),
}

CHECKPOINT_MAP = {
    "lstm-retriever":     "models/lstm-retriever/best.pt",
    "conv-retriever":     "models/conv-retriever/best.pt",
    "mamba-retriever-fs": "models/mamba-retriever-fs/best.pt",
}


def _hardware():
    return {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_device": (
            torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
        ),
    }


def _git():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
    except Exception:
        return "unknown"


def _bench(encoder, corpus, queries_subset, qrels_subset, docids, qids, batch_size, top_k):
    doc_texts = [
        (corpus[d].get("title", "") + " " + corpus[d].get("text", "")).strip()
        for d in docids
    ]
    q_texts = [queries_subset[q] for q in qids]

    doc_emb = encoder.encode_documents(doc_texts, batch_size=batch_size)
    q_emb = encoder.encode_queries(q_texts, batch_size=batch_size)

    index = IndexWrapper(dim=encoder.embedding_dim)
    index.add(doc_emb, docids)
    scores, retrieved = index.search(q_emb, top_k=top_k)

    run = {
        qids[i]: [(retrieved[i][j], float(scores[i][j])) for j in range(len(retrieved[i]))]
        for i in range(len(qids))
    }

    ks = (1, 5, 10, 100)
    metrics_ours = evaluate(qrels_subset, run, ks=ks)
    metrics_pytrec = evaluate_pytrec(qrels_subset, run, ks=ks)
    delta_abs = {
        k: abs(metrics_ours[k] - metrics_pytrec[k])
        for k in metrics_ours if k in metrics_pytrec
    }

    # Per-query results (§5.7 schema for Phase 6 / cross-phase analysis)
    per_q_ndcg10 = ndcg_at_k(qrels_subset, run, 10)
    per_q_recall10 = recall_at_k(qrels_subset, run, 10)
    per_query_results: dict[str, dict] = {}
    for qid in qids:
        ranked = [d for d, _ in run[qid]]
        rel_map = qrels_subset.get(qid, {})
        hit_10 = any(d in rel_map and rel_map[d] >= 1 for d in ranked[:10])
        per_query_results[qid] = {
            "query_text": queries_subset[qid],
            "ranked_doc_ids": ranked,
            "relevance_judgments": {d: int(r) for d, r in rel_map.items()},
            "nDCG@10": round(per_q_ndcg10.get(qid, 0.0), 6),
            "Recall@10": round(per_q_recall10.get(qid, 0.0), 6),
            "hit@10": bool(hit_10),
        }

    return metrics_ours, metrics_pytrec, delta_abs, per_query_results, run


def run_cell(
    encoder_name: str,
    dataset: str,
    mode: str,
    split: str,
    batch_size: int,
    top_k: int,
    device: str,
    seed: int,
    output_dir: Path,
    preloaded_split=None,
) -> dict:
    shuffle_q, shuffle_d = MODES[mode]
    checkpoint = CHECKPOINT_MAP.get(encoder_name)

    t_total = time.perf_counter()
    bsplit = preloaded_split if preloaded_split is not None else load_beir(dataset, split=split)
    qids_with_rels = [q for q in bsplit.queries if q in bsplit.qrels and bsplit.qrels[q]]
    queries_subset = {q: bsplit.queries[q] for q in qids_with_rels}
    qrels_subset = {q: bsplit.qrels[q] for q in qids_with_rels}
    docids = list(bsplit.corpus.keys())
    qids = list(queries_subset.keys())

    _log.info("cell.start encoder=%s dataset=%s mode=%s n_docs=%d n_queries=%d",
              encoder_name, dataset, mode, len(docids), len(qids))

    enc = ShuffledEncoder(
        base_encoder_name=encoder_name,
        shuffle_queries=shuffle_q,
        shuffle_docs=shuffle_d,
        seed=seed,
        device=device,
        checkpoint_path=checkpoint,
    )

    metrics_ours, metrics_pytrec, delta_abs, per_query, run = _bench(
        enc, bsplit.corpus, queries_subset, qrels_subset, docids, qids,
        batch_size, top_k,
    )

    artifact = {
        "vitruvius_version": __version__,
        "git_commit": _git(),
        "run_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "encoder": encoder_name,
        "dataset": dataset,
        "split": split,
        "mode": mode,
        "shuffle_queries": shuffle_q,
        "shuffle_docs": shuffle_d,
        "seed": seed,
        "config": {
            "batch_size": batch_size,
            "top_k": top_k,
            "device": device,
            "checkpoint": checkpoint,
        },
        "hardware": _hardware(),
        "metrics": {
            "ours_from_scratch": {k: round(v, 6) for k, v in metrics_ours.items()},
            "pytrec_eval": {k: round(v, 6) for k, v in metrics_pytrec.items()},
            "delta_abs": {k: round(v, 6) for k, v in delta_abs.items()},
        },
        "per_query_results": per_query,
        "runtime_seconds": round(time.perf_counter() - t_total, 3),
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    out = output_dir / f"{encoder_name}__{dataset}__{mode}.json"
    with open(out, "w") as f:
        json.dump(artifact, f, indent=2, sort_keys=True)

    _log.info("cell.done encoder=%s dataset=%s mode=%s nDCG@10=%.4f wall_s=%.1f out=%s",
              encoder_name, dataset, mode, metrics_ours["nDCG@10"],
              artifact["runtime_seconds"], out)

    del enc
    torch.cuda.empty_cache()
    return artifact


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--encoders", nargs="+", required=True)
    p.add_argument("--datasets", nargs="+", required=True)
    p.add_argument("--modes", nargs="+", default=list(MODES.keys()),
                   choices=list(MODES.keys()))
    p.add_argument("--split", default="test")
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--top-k", type=int, default=100)
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=1729)
    p.add_argument("--output-dir", type=Path, default=Path("experiments/phase8"))
    args = p.parse_args(argv)

    set_seed(args.seed)
    results = []
    for ds in args.datasets:
        # Preload split once per dataset to amortize I/O across all (encoder × mode) cells
        bsplit = load_beir(ds, split=args.split)
        for enc in args.encoders:
            for mode in args.modes:
                results.append(run_cell(
                    encoder_name=enc, dataset=ds, mode=mode,
                    split=args.split, batch_size=args.batch_size, top_k=args.top_k,
                    device=args.device, seed=args.seed, output_dir=args.output_dir,
                    preloaded_split=bsplit,
                ))

    print(json.dumps([
        {"encoder": r["encoder"], "dataset": r["dataset"], "mode": r["mode"],
         "nDCG@10": r["metrics"]["ours_from_scratch"]["nDCG@10"],
         "wall_s": r["runtime_seconds"]}
        for r in results
    ], indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
