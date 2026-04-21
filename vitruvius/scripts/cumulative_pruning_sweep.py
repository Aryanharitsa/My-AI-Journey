"""Phase 7 cumulative pruning sweep.

Reads experiments/phase7/head_importance/<encoder>__<dataset>.json (the
per-head ranking produced by head_importance_sweep.py), then for each cell:

  For N in {0, 4, 8, 16, 24, 32, 48, 64, 96, min(128, total_heads-1)}:
    zero the N least-important heads, measure nDCG@10 with the pruned encoder.

Writes experiments/phase7/cumulative_pruning/<encoder>__<dataset>.json with
the curve + "heads prunable at 5%/10% drop" thresholds.

Budget: 10 points x 9 cells = 90 bench runs at ~0.5s/run = ~45s compute +
tokenization overhead. Expected total ~15-30 min.
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

import numpy as np
import torch

from vitruvius import __version__
from vitruvius.data.beir_loader import load_beir
from vitruvius.encoders.pruned_transformer import get_base_encoder_info
from vitruvius.evaluation.faiss_index import IndexWrapper
from vitruvius.evaluation.retrieval_metrics import mean_metric, ndcg_at_k
from vitruvius.utils.logging import get_logger

_log = get_logger("cumulative_pruning")


POOLING = {"minilm-l6-v2": "mean", "bert-base": "mean", "gte-small": "mean"}
MAX_SEQ_LEN = {"minilm-l6-v2": 256, "bert-base": 350, "gte-small": 512}

CUM_POINTS = [0, 4, 8, 16, 24, 32, 48, 64, 96]


def _mean_pool(last, mask):
    m = mask.unsqueeze(-1).float()
    return (last * m).sum(dim=1) / m.sum(dim=1).clamp(min=1.0)


def _cls_pool(last, mask):
    return last[:, 0, :]


def _encode_batched(model, tokenized, head_mask, pool_fn, normalize, batch_size, device):
    n = tokenized["input_ids"].shape[0]
    out = []
    with torch.no_grad():
        for s in range(0, n, batch_size):
            e = min(s + batch_size, n)
            ids = tokenized["input_ids"][s:e].to(device, non_blocking=True)
            am = tokenized["attention_mask"][s:e].to(device, non_blocking=True)
            with torch.cuda.amp.autocast(dtype=torch.float16):
                r = model(input_ids=ids, attention_mask=am, head_mask=head_mask)
                pooled = pool_fn(r.last_hidden_state, am)
                if normalize:
                    pooled = torch.nn.functional.normalize(pooled, p=2, dim=-1)
            out.append(pooled.detach().to(torch.float32).cpu().numpy())
    return np.concatenate(out, axis=0)


def _score(model, doc_toks, q_toks, mask, pool_fn, normalize, bs, device,
           docids, qids, qrels_subset, dim, top_k) -> float:
    d_emb = _encode_batched(model, doc_toks, mask, pool_fn, normalize, bs, device)
    q_emb = _encode_batched(model, q_toks, mask, pool_fn, normalize, bs, device)
    idx = IndexWrapper(dim=dim)
    idx.add(d_emb, docids)
    scores, retr = idx.search(q_emb, top_k=top_k)
    run = {
        qids[i]: [(retr[i][j], float(scores[i][j])) for j in range(len(retr[i]))]
        for i in range(len(qids))
    }
    return float(mean_metric(ndcg_at_k(qrels_subset, run, 10)))


def _hardware():
    return {"torch": torch.__version__,
            "cuda_device": (torch.cuda.get_device_name(0)
                            if torch.cuda.is_available() else "cpu"),
            "python": platform.python_version(),
            "platform": platform.platform()}


def _git():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
    except Exception:
        return "unknown"


def run_cell(encoder_name, dataset, split, batch_size, top_k, device_str,
             importance_dir: Path, output_dir: Path) -> dict:
    from transformers import AutoModel, AutoTokenizer

    imp_path = importance_dir / f"{encoder_name}__{dataset}.json"
    if not imp_path.exists():
        raise FileNotFoundError(f"missing importance ranking: {imp_path}")
    importance = json.loads(imp_path.read_text())
    num_layers = importance["num_layers"]
    num_heads = importance["num_heads_per_layer"]
    total_heads = importance["total_heads"]
    baseline_ndcg = importance["baseline_nDCG@10"]

    # Heads sorted from LEAST to MOST important for retrieval.
    # ranked_by_importance is MOST -> LEAST (rank 0 = most important);
    # we want the reverse = least important first for cumulative pruning.
    ranked_desc = importance["ranked_by_importance"]
    least_first = list(reversed(ranked_desc))  # rank N-1 -> 0 (least -> most)

    info = get_base_encoder_info(encoder_name)
    pool_fn = _mean_pool if POOLING[encoder_name] == "mean" else _cls_pool
    normalize = (info["similarity"] == "cosine")
    max_len = MAX_SEQ_LEN[encoder_name]

    device = torch.device(device_str)
    _log.info("cum.start %s x %s baseline=%.4f total_heads=%d",
              encoder_name, dataset, baseline_ndcg, total_heads)

    bsplit = load_beir(dataset, split=split)
    qids_with_rels = [q for q in bsplit.queries if q in bsplit.qrels and bsplit.qrels[q]]
    queries_subset = {q: bsplit.queries[q] for q in qids_with_rels}
    qrels_subset = {q: bsplit.qrels[q] for q in qids_with_rels}
    docids = list(bsplit.corpus.keys())
    doc_texts = [
        (bsplit.corpus[d].get("title", "") + " " + bsplit.corpus[d].get("text", "")).strip()
        for d in docids
    ]
    qids = list(queries_subset.keys())
    q_texts = [queries_subset[q] for q in qids]

    tok = AutoTokenizer.from_pretrained(info["hf_id"])
    model = AutoModel.from_pretrained(info["hf_id"], attn_implementation="eager").to(device).eval()
    dim = model.config.hidden_size
    doc_toks = tok(doc_texts, padding=True, truncation=True, max_length=max_len,
                   return_tensors="pt")
    q_toks = tok(q_texts, padding=True, truncation=True, max_length=max_len,
                 return_tensors="pt")

    points_N = [N for N in CUM_POINTS + [min(128, total_heads - 1)]
                if 0 <= N <= total_heads - 1]
    points_N = sorted(set(points_N))

    curve = []
    t0 = time.perf_counter()
    for N in points_N:
        mask = torch.ones(num_layers, num_heads, device=device)
        for entry in least_first[:N]:
            mask[entry["layer"], entry["head"]] = 0.0
        ndcg = _score(model, doc_toks, q_toks, mask, pool_fn, normalize,
                      batch_size, device, docids, qids, qrels_subset, dim, top_k)
        rel_drop = ((baseline_ndcg - ndcg) / baseline_ndcg * 100
                    if baseline_ndcg > 0 else 0.0)
        curve.append({"heads_pruned": N, "nDCG@10": round(ndcg, 6),
                      "rel_drop_pct": round(rel_drop, 3)})
        _log.info("cum.point %s x %s N=%d nDCG@10=%.4f rel_drop=%.2f%%",
                  encoder_name, dataset, N, ndcg, rel_drop)

    # Thresholds: largest N such that drop <= 5% / 10%.
    def _threshold(pct: float) -> int | None:
        best = None
        for pt in curve:
            if pt["rel_drop_pct"] <= pct:
                best = pt["heads_pruned"]
        return best

    artifact = {
        "vitruvius_version": __version__,
        "git_commit": _git(),
        "run_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "encoder": encoder_name,
        "dataset": dataset,
        "split": split,
        "baseline_nDCG@10": baseline_ndcg,
        "ordering": "single-head-importance-ascending (least important pruned first)",
        "ordering_caveat": (
            "Single-head importance != cumulative-optimal ordering (heads can "
            "compensate for each other). A Taylor-saliency or iterative-greedy "
            "baseline would be stronger; flagged as future work per handoff §7.4."
        ),
        "num_layers": num_layers,
        "num_heads_per_layer": num_heads,
        "total_heads": total_heads,
        "curve": curve,
        "thresholds": {
            "heads_prunable_at_5pct_drop": _threshold(5.0),
            "heads_prunable_at_10pct_drop": _threshold(10.0),
        },
        "seed": 1729,
        "config": {"batch_size": batch_size, "top_k": top_k, "device": device_str,
                   "pooling": POOLING[encoder_name],
                   "similarity": info["similarity"], "max_seq_len": max_len},
        "hardware": _hardware(),
        "runtime_seconds": round(time.perf_counter() - t0, 2),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    out = output_dir / f"{encoder_name}__{dataset}.json"
    out.write_text(json.dumps(artifact, indent=2, sort_keys=True))
    _log.info("cum.done %s x %s 5%%_at=%s 10%%_at=%s wall_s=%.1f",
              encoder_name, dataset,
              artifact["thresholds"]["heads_prunable_at_5pct_drop"],
              artifact["thresholds"]["heads_prunable_at_10pct_drop"],
              artifact["runtime_seconds"])
    del model, doc_toks, q_toks
    torch.cuda.empty_cache()
    return artifact


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--encoders", nargs="+", required=True,
                   choices=["minilm-l6-v2", "bert-base", "gte-small"])
    p.add_argument("--datasets", nargs="+", required=True,
                   choices=["nfcorpus", "scifact", "fiqa"])
    p.add_argument("--split", default="test")
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--top-k", type=int, default=100)
    p.add_argument("--device", default="cuda")
    p.add_argument("--importance-dir", type=Path,
                   default=Path("experiments/phase7/head_importance"))
    p.add_argument("--output-dir", type=Path,
                   default=Path("experiments/phase7/cumulative_pruning"))
    args = p.parse_args(argv)

    results = []
    for e in args.encoders:
        for d in args.datasets:
            results.append(run_cell(e, d, args.split, args.batch_size, args.top_k,
                                    args.device, args.importance_dir, args.output_dir))
    print(json.dumps([{"encoder": r["encoder"], "dataset": r["dataset"],
                       "5%": r["thresholds"]["heads_prunable_at_5pct_drop"],
                       "10%": r["thresholds"]["heads_prunable_at_10pct_drop"],
                       "wall_s": r["runtime_seconds"]}
                      for r in results], indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
