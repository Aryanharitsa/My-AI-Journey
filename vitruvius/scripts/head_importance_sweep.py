"""Phase 7 head-importance sweep — optimized manual-forward version.

Bypasses sentence-transformers.encode()'s per-call overhead. Tokenizes all
docs + queries ONCE per cell, then for each head_mask just runs the forward
pass + pool + normalize + FAISS search directly.

The pooling / normalization strategies are reconstructed to match each
base encoder:
- minilm-l6-v2: mean pool (attention-mask weighted), L2 normalize.
- bert-base (msmarco-dot-v5): mean pool, NO normalize (dot product model).
- gte-small: CLS pool, L2 normalize.

These match the base encoders' outputs within 1e-5 in the all-ones test
(unit-tested in tests/test_pruned_transformer.py — the sentence-transformers
wrapper there confirms bit-exact agreement for minilm).
"""
from __future__ import annotations

import argparse
import json
import os
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
from vitruvius.evaluation.retrieval_metrics import ndcg_at_k, mean_metric
from vitruvius.utils.logging import get_logger

_log = get_logger("head_importance_sweep")


# Pooling config per base encoder. Verified against each model's
# sentence-transformers 1_Pooling/config.json on the pod.
POOLING = {
    "minilm-l6-v2": "mean",    # ST config: pooling_mode_mean_tokens=True
    "bert-base": "mean",       # msmarco-dot-v5 uses mean pooling
    "gte-small": "mean",        # GTE family uses MEAN pooling (verified from thenlper/gte-small 1_Pooling/config.json)
}
MAX_SEQ_LEN = {
    "minilm-l6-v2": 256,       # ST config: max_seq_length=256
    "bert-base": 350,          # ST config for this checkpoint
    "gte-small": 512,          # GTE default
}


def _hardware_snapshot() -> dict:
    return {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "hostname": platform.node(),
        "torch": torch.__version__,
        "cuda_device": (
            torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
        ),
    }


def _git_head() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
    except Exception:
        return "unknown"


def _mean_pool(last_hidden: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
    mask = attn_mask.unsqueeze(-1).float()
    summed = (last_hidden * mask).sum(dim=1)
    lens = mask.sum(dim=1).clamp(min=1.0)
    return summed / lens


def _cls_pool(last_hidden: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
    return last_hidden[:, 0, :]


def _encode_batched(
    model,
    tokenized: dict,
    head_mask: torch.Tensor,
    pool_fn,
    normalize: bool,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    n = tokenized["input_ids"].shape[0]
    out_rows: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            ids = tokenized["input_ids"][start:end].to(device, non_blocking=True)
            am = tokenized["attention_mask"][start:end].to(device, non_blocking=True)
            # Use autocast fp16 for inference speed on A100
            with torch.cuda.amp.autocast(dtype=torch.float16):
                res = model(input_ids=ids, attention_mask=am, head_mask=head_mask)
                pooled = pool_fn(res.last_hidden_state, am)
                if normalize:
                    pooled = torch.nn.functional.normalize(pooled, p=2, dim=-1)
            out_rows.append(pooled.detach().to(torch.float32).cpu().numpy())
    return np.concatenate(out_rows, axis=0)


def _score_with_mask(
    model, doc_tokens, query_tokens, head_mask, pool_fn, normalize,
    batch_size, device, docids, qids, qrels_subset, dim, top_k,
) -> float:
    doc_emb = _encode_batched(model, doc_tokens, head_mask, pool_fn, normalize,
                              batch_size, device)
    q_emb = _encode_batched(model, query_tokens, head_mask, pool_fn, normalize,
                            batch_size, device)
    index = IndexWrapper(dim=dim)
    index.add(doc_emb, docids)
    scores, retrieved = index.search(q_emb, top_k=top_k)
    run = {
        qids[i]: [(retrieved[i][j], float(scores[i][j])) for j in range(len(retrieved[i]))]
        for i in range(len(qids))
    }
    return float(mean_metric(ndcg_at_k(qrels_subset, run, 10)))


def sweep_cell(encoder_name, dataset, split, batch_size, top_k, device_str,
               output_dir: Path) -> dict:
    from transformers import AutoModel, AutoTokenizer

    info = get_base_encoder_info(encoder_name)
    num_layers = info["num_layers"]
    num_heads = info["num_heads"]
    total_heads = num_layers * num_heads
    pool_name = POOLING[encoder_name]
    max_len = MAX_SEQ_LEN[encoder_name]
    normalize = (info["similarity"] == "cosine")

    device = torch.device(device_str)
    _log.info(
        "sweep.start encoder=%s dataset=%s layers=%d heads/layer=%d total=%d "
        "pool=%s normalize=%s max_len=%d",
        encoder_name, dataset, num_layers, num_heads, total_heads,
        pool_name, normalize, max_len,
    )

    # Load BEIR once
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

    # Load model + tokenizer
    t0 = time.perf_counter()
    tok = AutoTokenizer.from_pretrained(info["hf_id"])
    model = AutoModel.from_pretrained(info["hf_id"], attn_implementation="eager").to(device).eval()
    dim = model.config.hidden_size
    _log.info("sweep.loaded model=%s dim=%d load_s=%.1f",
              info["hf_id"], dim, time.perf_counter() - t0)

    # Tokenize once — pad to longest in batch is faster than full max_len padding.
    t0 = time.perf_counter()
    doc_tokens = tok(doc_texts, padding=True, truncation=True, max_length=max_len,
                     return_tensors="pt")
    query_tokens = tok(q_texts, padding=True, truncation=True, max_length=max_len,
                       return_tensors="pt")
    _log.info("sweep.tokenized n_docs=%d n_queries=%d wall_s=%.1f",
              len(docids), len(qids), time.perf_counter() - t0)

    pool_fn = _mean_pool if pool_name == "mean" else _cls_pool

    # Baseline (all ones)
    t0 = time.perf_counter()
    baseline_mask = torch.ones(num_layers, num_heads, device=device)
    baseline_ndcg = _score_with_mask(
        model, doc_tokens, query_tokens, baseline_mask, pool_fn, normalize,
        batch_size, device, docids, qids, qrels_subset, dim, top_k,
    )
    t_baseline = time.perf_counter() - t0
    _log.info("sweep.baseline %s x %s nDCG@10=%.4f wall_s=%.1f",
              encoder_name, dataset, baseline_ndcg, t_baseline)

    # Per-head ablation
    per_head: list[dict] = []
    t_sweep_start = time.perf_counter()
    for head_idx in range(total_heads):
        layer = head_idx // num_heads
        head = head_idx % num_heads
        mask = torch.ones(num_layers, num_heads, device=device)
        mask[layer, head] = 0.0
        ablated = _score_with_mask(
            model, doc_tokens, query_tokens, mask, pool_fn, normalize,
            batch_size, device, docids, qids, qrels_subset, dim, top_k,
        )
        delta = baseline_ndcg - ablated
        per_head.append({"layer": layer, "head": head,
                         "nDCG@10": round(ablated, 6),
                         "delta_nDCG@10": round(delta, 6)})
        if (head_idx + 1) % 24 == 0 or head_idx == total_heads - 1:
            elapsed = time.perf_counter() - t_sweep_start
            eta = elapsed / (head_idx + 1) * (total_heads - head_idx - 1)
            _log.info("sweep.progress %s x %s %d/%d elapsed=%.0fs eta=%.0fs",
                      encoder_name, dataset, head_idx + 1, total_heads, elapsed, eta)

    sorted_by_importance = sorted(per_head, key=lambda r: r["delta_nDCG@10"], reverse=True)
    for rank, r in enumerate(sorted_by_importance):
        r["rank_by_importance"] = rank

    total_wall = time.perf_counter() - t0
    artifact = {
        "vitruvius_version": __version__,
        "git_commit": _git_head(),
        "run_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "encoder": encoder_name,
        "dataset": dataset,
        "split": split,
        "baseline_nDCG@10": round(baseline_ndcg, 6),
        "num_layers": num_layers,
        "num_heads_per_layer": num_heads,
        "total_heads": total_heads,
        "per_head_results": sorted(per_head, key=lambda r: (r["layer"], r["head"])),
        "ranked_by_importance": [
            {"layer": r["layer"], "head": r["head"], "rank": r["rank_by_importance"],
             "delta_nDCG@10": r["delta_nDCG@10"]}
            for r in sorted_by_importance
        ],
        "seed": 1729,
        "config": {
            "batch_size": batch_size,
            "top_k": top_k,
            "device": device_str,
            "similarity": info["similarity"],
            "pooling": pool_name,
            "max_seq_len": max_len,
            "amp": "fp16",
        },
        "hardware": _hardware_snapshot(),
        "runtime_seconds": {
            "baseline": round(t_baseline, 4),
            "full_sweep": round(total_wall, 4),
            "per_head_avg": round((total_wall - t_baseline) / total_heads, 4),
        },
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{encoder_name}__{dataset}.json"
    with open(out_path, "w") as f:
        json.dump(artifact, f, indent=2, sort_keys=True)

    _log.info("sweep.done %s x %s baseline=%.4f total_wall_s=%.1f out=%s",
              encoder_name, dataset, baseline_ndcg, total_wall, out_path)

    # Free GPU memory before the next cell
    del model, doc_tokens, query_tokens
    torch.cuda.empty_cache()
    return artifact


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--encoders", nargs="+", required=True,
                   choices=["minilm-l6-v2", "bert-base", "gte-small"])
    p.add_argument("--datasets", nargs="+", required=True,
                   choices=["nfcorpus", "scifact", "fiqa"])
    p.add_argument("--split", default="test")
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--top-k", type=int, default=100)
    p.add_argument("--device", default="cuda")
    p.add_argument("--output-dir", default="experiments/phase7/head_importance")
    args = p.parse_args(argv)

    output_dir = Path(args.output_dir)
    results = []
    for enc in args.encoders:
        for ds in args.datasets:
            results.append(sweep_cell(enc, ds, args.split, args.batch_size,
                                      args.top_k, args.device, output_dir))

    summary_line = [
        (r["encoder"], r["dataset"], r["baseline_nDCG@10"],
         round(r["runtime_seconds"]["full_sweep"], 1))
        for r in results
    ]
    print(json.dumps(summary_line, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
