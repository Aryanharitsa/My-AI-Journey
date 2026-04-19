"""Command-line entry for Project Vitruvius.

Subcommands implemented so far:
    smoke         — synthetic-data end-to-end CPU sanity run (Phase 1)
    bench         — BEIR retrieval bench, one (encoder, dataset) (Phase 2)
    bench-sweep   — Cartesian product of encoders × datasets + SUMMARY.md (Phase 3)

Subcommands stubbed (return non-zero) until later phases:
    profile   — Phase 3.5 (latency profiler)
    shuffle   — Phase 8  (position sensitivity)
    prune     — Phase 7  (attention head pruning)
"""
from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence

from vitruvius import __version__
from vitruvius.utils.logging import get_logger
from vitruvius.utils.seed import set_seed

_log = get_logger(__name__)


# Phase 3 reference table (handoff §3.4 — approximate BEIR leaderboard numbers).
# Each value is the published nDCG@10 for (encoder_registry_name, dataset).
REFERENCES_PHASE3: dict[tuple[str, str], float] = {
    ("minilm-l6-v2", "nfcorpus"): 0.30,
    ("minilm-l6-v2", "scifact"): 0.64,
    ("minilm-l6-v2", "fiqa"): 0.36,
    ("bert-base", "nfcorpus"): 0.31,
    ("bert-base", "scifact"): 0.68,
    ("bert-base", "fiqa"): 0.30,
    ("gte-small", "nfcorpus"): 0.34,
    ("gte-small", "scifact"): 0.73,
    ("gte-small", "fiqa"): 0.42,
}
TOLERANCE_PHASE3 = 0.03


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="vitruvius",
        description=f"Dense retrieval encoder architecture study (v{__version__})",
    )
    p.add_argument("--version", action="version", version=f"vitruvius {__version__}")
    sub = p.add_subparsers(dest="command", required=True)

    p_smoke = sub.add_parser("smoke", help="run the synthetic-data smoke test")
    p_smoke.add_argument("--cpu", action="store_true", help="force CPU (smoke default)")
    p_smoke.add_argument("--seed", type=int, default=1729)
    p_smoke.add_argument("--no-encoder", action="store_true",
                         help="skip MiniLM load; use a hash-based stand-in embedding")

    p_bench = sub.add_parser("bench", help="benchmark an encoder on a BEIR dataset")
    p_bench.add_argument("--encoder", required=True)
    p_bench.add_argument("--dataset", required=True)
    p_bench.add_argument("--split", default="test")
    p_bench.add_argument("--batch-size", type=int, default=32)
    p_bench.add_argument("--top-k", type=int, default=100)
    p_bench.add_argument("--limit", type=int, default=None)
    p_bench.add_argument("--output", type=str, default=None)
    p_bench.add_argument("--device", default="auto")

    p_sweep = sub.add_parser(
        "bench-sweep",
        help="run bench across the Cartesian product of encoders × datasets and emit SUMMARY.md",
    )
    p_sweep.add_argument("--encoders", nargs="+", required=True)
    p_sweep.add_argument("--datasets", nargs="+", required=True)
    p_sweep.add_argument("--split", default="test")
    p_sweep.add_argument("--batch-size", type=int, default=128)
    p_sweep.add_argument("--top-k", type=int, default=100)
    p_sweep.add_argument("--device", default="auto")
    p_sweep.add_argument("--output-dir", required=True)
    p_sweep.add_argument("--tolerance", type=float, default=TOLERANCE_PHASE3)
    p_sweep.add_argument(
        "--stop-on-out-of-band",
        action="store_true",
        help="abort the sweep on the first out-of-band cell (default: continue and flag)",
    )

    p_prof = sub.add_parser(
        "profile",
        help="latency profile across encoders x datasets x batch sizes with real BEIR inputs",
    )
    p_prof.add_argument("--encoders", nargs="+", required=True)
    p_prof.add_argument("--datasets", nargs="+", required=True)
    p_prof.add_argument("--batch-sizes", nargs="+", type=int, default=[1, 8, 32])
    p_prof.add_argument("--split", default="test")
    p_prof.add_argument("--sample-size", type=int, default=200)
    p_prof.add_argument("--n-warmup", type=int, default=10)
    p_prof.add_argument("--n-measure", type=int, default=100)
    p_prof.add_argument("--device", default="auto")
    p_prof.add_argument("--output-dir", required=True)
    p_prof.add_argument("--seed", type=int, default=1729)

    p_shuf = sub.add_parser("shuffle", help="position-sensitivity probe")
    p_shuf.add_argument("--encoder", required=True)
    p_shuf.add_argument("--dataset", required=True)

    p_prune = sub.add_parser("prune", help="attention head pruning probe")
    p_prune.add_argument("--encoder", required=True)
    p_prune.add_argument("--dataset", required=True)

    return p


def _hash_embed(texts: list[str], dim: int = 256):
    """Cheap deterministic embedding via word-hash bag-of-features.

    Used in the smoke test when no real model is loaded so the synthetic
    corpus still produces a non-degenerate ranking signal.
    """
    import hashlib

    import numpy as np

    out = np.zeros((len(texts), dim), dtype=np.float32)
    for i, t in enumerate(texts):
        for tok in t.lower().split():
            h = int(hashlib.md5(tok.encode()).hexdigest(), 16)
            out[i, h % dim] += 1.0
    norms = np.linalg.norm(out, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return out / norms


def _cmd_smoke(args: argparse.Namespace) -> int:

    from vitruvius.data.synthetic import make_corpus
    from vitruvius.evaluation.faiss_index import IndexWrapper
    from vitruvius.evaluation.retrieval_metrics import evaluate

    set_seed(args.seed)

    _log.info("smoke.start mode=%s", "no-encoder" if args.no_encoder else "auto")
    corpus = make_corpus(seed=args.seed)
    _log.info("smoke.corpus n_queries=%d n_docs=%d",
              len(corpus.queries), len(corpus.corpus))

    if args.no_encoder:
        dim = 256
        encoder_name = "hash-bag-of-features"
        doc_texts = [f"{d['title']} {d['text']}" for d in corpus.corpus.values()]
        doc_emb = _hash_embed(doc_texts, dim=dim)
        q_emb = _hash_embed(list(corpus.queries.values()), dim=dim)
    else:
        try:
            from vitruvius.encoders import get_encoder

            enc = get_encoder("minilm-l6-v2", device="cpu")
            dim = enc.embedding_dim
            encoder_name = enc.name
            doc_texts = [f"{d['title']} {d['text']}" for d in corpus.corpus.values()]
            doc_emb = enc.encode_documents(doc_texts, batch_size=32)
            q_emb = enc.encode_queries(list(corpus.queries.values()), batch_size=32)
        except Exception as e:
            _log.warning("smoke.encoder_failed err=%r — falling back to hash bag", e)
            dim = 256
            encoder_name = "hash-bag-of-features"
            doc_texts = [f"{d['title']} {d['text']}" for d in corpus.corpus.values()]
            doc_emb = _hash_embed(doc_texts, dim=dim)
            q_emb = _hash_embed(list(corpus.queries.values()), dim=dim)

    index = IndexWrapper(dim=dim)
    docids = list(corpus.corpus.keys())
    index.add(doc_emb, docids)

    qids = list(corpus.queries.keys())
    scores, retrieved = index.search(q_emb, top_k=10)
    run = {
        qids[i]: [(retrieved[i][j], float(scores[i][j])) for j in range(len(retrieved[i]))]
        for i in range(len(qids))
    }

    metrics = evaluate(corpus.qrels, run, ks=(1, 5, 10))

    summary = {
        "encoder": encoder_name,
        "embedding_dim": int(dim),
        "n_queries": len(qids),
        "n_docs": len(docids),
        "metrics": {k: round(v, 4) for k, v in metrics.items()},
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    _log.info("smoke.done status=ok")
    return 0


def _not_yet(phase: str) -> int:
    _log.error("not implemented yet — see %s", phase)
    return 2


def _hardware_snapshot() -> dict:
    import platform
    info = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "hostname": platform.node(),
        "torch": "unknown",
        "cuda_device": "unknown",
        "faiss": "unknown",
    }
    try:
        import torch
        info["torch"] = torch.__version__
        info["cuda_device"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    except Exception:
        pass
    try:
        import faiss
        info["faiss"] = faiss.__version__
    except Exception:
        pass
    return info


def _git_head() -> str:
    import subprocess
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
    except Exception:
        return "unknown"


def _run_bench_core(
    *,
    encoder,
    encoder_name: str,
    dataset: str,
    split_name: str,
    batch_size: int,
    top_k: int,
    device_str: str,
    limit: int | None,
    output_path: str | None,
    reference: float | None,
    tolerance: float,
    source_tag: str = "BEIR leaderboard (approximate; see handoff §3.4)",
) -> dict:
    """Run one (encoder, dataset) bench with a pre-loaded encoder.

    Separated out so bench-sweep can share one model load across datasets.
    The returned dict is the artifact (also written to output_path if given).
    """
    import os
    import time
    from datetime import datetime, timezone

    import numpy as np

    from vitruvius.data.beir_loader import load_beir
    from vitruvius.evaluation.faiss_index import IndexWrapper
    from vitruvius.evaluation.pytrec_bridge import evaluate_pytrec
    from vitruvius.evaluation.retrieval_metrics import evaluate

    set_seed(1729)
    total_t0 = time.perf_counter()

    _log.info("bench.start encoder=%s dataset=%s split=%s device=%s top_k=%d batch_size=%d",
              encoder_name, dataset, split_name, device_str, top_k, batch_size)

    bsplit = load_beir(dataset, split=split_name)
    n_corpus = len(bsplit.corpus)
    n_queries_all = len(bsplit.queries)

    q_ids_with_qrels = [q for q in bsplit.queries if q in bsplit.qrels and bsplit.qrels[q]]
    if limit:
        q_ids_with_qrels = q_ids_with_qrels[: limit]
    queries_subset = {q: bsplit.queries[q] for q in q_ids_with_qrels}
    qrels_subset = {q: bsplit.qrels[q] for q in q_ids_with_qrels}
    max_grade = max(
        (r for per_q in qrels_subset.values() for r in per_q.values()), default=0
    )
    avg_rels = (
        sum(len([r for r in per_q.values() if r >= 1]) for per_q in qrels_subset.values())
        / max(len(qrels_subset), 1)
    )
    _log.info("bench.data n_corpus=%d n_queries_eval=%d avg_rel_per_q=%.2f max_grade=%d",
              n_corpus, len(queries_subset), avg_rels, max_grade)

    t0 = time.perf_counter()
    docids = list(bsplit.corpus.keys())
    doc_texts = [
        (bsplit.corpus[d].get("title", "") + " " + bsplit.corpus[d].get("text", "")).strip()
        for d in docids
    ]
    doc_emb = encoder.encode_documents(doc_texts, batch_size=batch_size)
    t_encode_docs = time.perf_counter() - t0

    t0 = time.perf_counter()
    qids = list(queries_subset.keys())
    q_texts = [queries_subset[q] for q in qids]
    q_emb = encoder.encode_queries(q_texts, batch_size=batch_size)
    t_encode_queries = time.perf_counter() - t0

    doc_norm_max = float(np.max(np.linalg.norm(doc_emb, axis=1))) if len(doc_emb) else 0.0
    q_norm_max = float(np.max(np.linalg.norm(q_emb, axis=1))) if len(q_emb) else 0.0
    _log.info("bench.encoded doc_norm_max=%.4f q_norm_max=%.4f dim=%d",
              doc_norm_max, q_norm_max, encoder.embedding_dim)

    t0 = time.perf_counter()
    index = IndexWrapper(dim=encoder.embedding_dim)
    index.add(doc_emb, docids)
    t_index = time.perf_counter() - t0

    t0 = time.perf_counter()
    scores, retrieved = index.search(q_emb, top_k=top_k)
    t_search = time.perf_counter() - t0

    run = {
        qids[i]: [(retrieved[i][j], float(scores[i][j])) for j in range(len(retrieved[i]))]
        for i in range(len(qids))
    }

    ks = (1, 5, 10, 100)
    t0 = time.perf_counter()
    metrics_ours = evaluate(qrels_subset, run, ks=ks)
    t_eval_ours = time.perf_counter() - t0

    t0 = time.perf_counter()
    metrics_pytrec = evaluate_pytrec(qrels_subset, run, ks=ks)
    t_eval_pytrec = time.perf_counter() - t0

    delta_abs = {
        k: abs(metrics_ours[k] - metrics_pytrec[k])
        for k in metrics_ours
        if k in metrics_pytrec
    }

    # Per-query results (session-03 §5.7) — Phase 6 failure analysis reads these.
    # Aggregate metrics are identical; this just preserves the (qid -> ranking) fan-out.
    from vitruvius.evaluation.retrieval_metrics import ndcg_at_k as _ndcg_at_k
    from vitruvius.evaluation.retrieval_metrics import recall_at_k as _recall_at_k
    per_q_ndcg10 = _ndcg_at_k(qrels_subset, run, 10)
    per_q_recall10 = _recall_at_k(qrels_subset, run, 10)
    per_query_results: dict[str, dict] = {}
    for _qid in qids:
        _ranked = [d for d, _ in run[_qid]]
        _rel_map = qrels_subset.get(_qid, {})
        _top10 = _ranked[:10]
        _hit_10 = any(_d in _rel_map and _rel_map[_d] >= 1 for _d in _top10)
        per_query_results[_qid] = {
            "query_text": queries_subset[_qid],
            "ranked_doc_ids": _ranked,
            "relevance_judgments": {d: int(r) for d, r in _rel_map.items()},
            "nDCG@10": round(per_q_ndcg10.get(_qid, 0.0), 6),
            "Recall@10": round(per_q_recall10.get(_qid, 0.0), 6),
            "hit@10": bool(_hit_10),
        }

    observed = metrics_ours["nDCG@10"]
    if reference is not None:
        reference_block = {
            "metric": "nDCG@10",
            "reference": reference,
            "tolerance": tolerance,
            "source": source_tag,
        }
        in_band = abs(observed - reference) <= tolerance
        delta_from_ref = observed - reference
    else:
        reference_block = None
        in_band = None
        delta_from_ref = None

    hw = _hardware_snapshot()
    total_seconds = time.perf_counter() - total_t0

    artifact = {
        "vitruvius_version": __version__,
        "git_commit": _git_head(),
        "run_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "config": {
            "encoder": encoder_name,
            "dataset": dataset,
            "split": split_name,
            "batch_size": batch_size,
            "top_k": top_k,
            "limit": limit,
            "device": device_str,
            "seed": 1729,
            "doc_format": "title + ' ' + text (stripped)",
            "similarity": encoder.similarity,
            "normalize_embeddings": (encoder.similarity == "cosine"),
        },
        "dataset_stats": {
            "n_corpus": n_corpus,
            "n_queries_total": n_queries_all,
            "n_queries_eval": len(queries_subset),
            "avg_relevant_per_query": round(avg_rels, 4),
            "max_relevance_grade": max_grade,
        },
        "hardware": hw,
        "runtime_seconds": {
            "encode_docs": round(t_encode_docs, 4),
            "encode_queries": round(t_encode_queries, 4),
            "index_build": round(t_index, 4),
            "search": round(t_search, 4),
            "eval_ours": round(t_eval_ours, 4),
            "eval_pytrec": round(t_eval_pytrec, 4),
            "total": round(total_seconds, 4),
        },
        "embedding_norms": {
            "doc_norm_max": round(doc_norm_max, 6),
            "query_norm_max": round(q_norm_max, 6),
        },
        "metrics": {
            "ours_from_scratch": {k: round(v, 6) for k, v in metrics_ours.items()},
            "pytrec_eval": {k: round(v, 6) for k, v in metrics_pytrec.items()},
            "delta_abs": {k: round(v, 6) for k, v in delta_abs.items()},
            "notes": (
                "ours uses DCG with gain=2^rel-1. pytrec_eval ndcg_cut_k uses the "
                "same trec_eval form. Small nDCG deltas are tie-breaks, not formula drift."
            ),
        },
        "target_band": reference_block,
        "per_query_results": per_query_results,
        "result": {
            "primary_metric": "nDCG@10 (ours_from_scratch)",
            "value": round(observed, 6),
            "reference": reference,
            "tolerance": tolerance if reference is not None else None,
            "delta_from_reference": round(delta_from_ref, 6) if delta_from_ref is not None else None,
            "in_band": bool(in_band) if in_band is not None else None,
        },
    }

    if output_path:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(artifact, f, indent=2, sort_keys=True)

    _log.info(
        "bench.done encoder=%s dataset=%s nDCG@10_ours=%.4f nDCG@10_pytrec=%.4f delta=%.2e "
        "in_band=%s total_s=%.1f artifact=%s",
        encoder_name, dataset,
        metrics_ours["nDCG@10"], metrics_pytrec["nDCG@10"],
        delta_abs.get("nDCG@10", float("nan")),
        in_band, total_seconds, output_path or "<none>",
    )
    return artifact


def _cmd_bench(args: argparse.Namespace) -> int:
    """BEIR retrieval benchmark with from-scratch + pytrec_eval cross-check."""
    from vitruvius.encoders import get_encoder
    from vitruvius.utils.device import pick_device

    device_str = str(pick_device(None if args.device == "auto" else args.device))
    enc = get_encoder(args.encoder, device=device_str)

    reference = REFERENCES_PHASE3.get((args.encoder, args.dataset))
    tolerance = TOLERANCE_PHASE3 if reference is not None else 0.0

    out_path = args.output or f"experiments/{args.dataset}_{args.encoder}_{args.split}.json"
    artifact = _run_bench_core(
        encoder=enc,
        encoder_name=args.encoder,
        dataset=args.dataset,
        split_name=args.split,
        batch_size=args.batch_size,
        top_k=args.top_k,
        device_str=device_str,
        limit=args.limit,
        output_path=out_path,
        reference=reference,
        tolerance=tolerance,
    )
    print(json.dumps(artifact, indent=2, sort_keys=True))
    return 0


def _write_sweep_summary(summary_path: str, sweep_results: list[dict], encoders: list[str], datasets: list[str]) -> None:
    """Emit a 3×3 grid SUMMARY.md with deltas and in-band flags."""
    import os
    lookup = {(r["config"]["encoder"], r["config"]["dataset"]): r for r in sweep_results}

    lines = ["# Phase 3 — 3×3 encoder × BEIR sweep", "",
             "Primary metric: nDCG@10 (from-scratch `retrieval_metrics.evaluate`).",
             "References from handoff §3.4 (approximate BEIR leaderboard).",
             f"Tolerance: ±{TOLERANCE_PHASE3:.2f} per cell.",
             "", "## Measured nDCG@10 (delta vs reference in parentheses)", ""]

    header = "| Encoder \\ Dataset | " + " | ".join(datasets) + " |"
    sep = "|" + "|".join(["---"] * (1 + len(datasets))) + "|"
    lines.append(header)
    lines.append(sep)

    for enc in encoders:
        cells = [f"`{enc}`"]
        for ds in datasets:
            r = lookup.get((enc, ds))
            if r is None:
                cells.append("—")
                continue
            val = r["result"]["value"]
            delta = r["result"]["delta_from_reference"]
            in_band = r["result"]["in_band"]
            ref = r["result"]["reference"]
            flag = "✅" if in_band else "❌"
            cells.append(f"{val:.4f} (Δ{delta:+.4f} vs {ref:.2f}) {flag}")
        lines.append("| " + " | ".join(cells) + " |")

    lines += ["", "## Per-cell breakdown", ""]
    for enc in encoders:
        for ds in datasets:
            r = lookup.get((enc, ds))
            if r is None:
                lines.append(f"- `{enc}` × `{ds}`: NO RESULT (skipped or failed)")
                continue
            val = r["result"]["value"]
            ref = r["result"]["reference"]
            delta = r["result"]["delta_from_reference"]
            band = "in-band" if r["result"]["in_band"] else "**OUT OF BAND**"
            py = r["metrics"]["pytrec_eval"]["nDCG@10"]
            dly = r["metrics"]["delta_abs"]["nDCG@10"]
            lines.append(
                f"- `{enc}` × `{ds}`: measured {val:.4f}, ref {ref:.2f}, "
                f"delta {delta:+.4f}, {band}. "
                f"pytrec_eval nDCG@10={py:.4f} (|Δ|={dly:.2e})."
            )

    lines += ["", "## Cross-check status (from-scratch vs pytrec_eval)", ""]
    lines.append("| Cell | max|Δ| across nDCG@{1,5,10,100} | Recall@k bit-exact? |")
    lines.append("|---|---:|:---:|")
    for enc in encoders:
        for ds in datasets:
            r = lookup.get((enc, ds))
            if r is None:
                continue
            deltas = r["metrics"]["delta_abs"]
            ndcg_deltas = [v for k, v in deltas.items() if k.startswith("nDCG@")]
            recall_deltas = [v for k, v in deltas.items() if k.startswith("Recall@")]
            max_ndcg = max(ndcg_deltas) if ndcg_deltas else 0
            recall_ok = "✅" if all(v == 0 for v in recall_deltas) else "❌"
            lines.append(f"| `{enc}` × `{ds}` | {max_ndcg:.2e} | {recall_ok} |")

    lines += ["", "## Runtime (seconds, total per cell)", ""]
    lines.append("| Encoder \\ Dataset | " + " | ".join(datasets) + " |")
    lines.append(sep)
    for enc in encoders:
        cells = [f"`{enc}`"]
        for ds in datasets:
            r = lookup.get((enc, ds))
            cells.append(f"{r['runtime_seconds']['total']:.2f}" if r else "—")
        lines.append("| " + " | ".join(cells) + " |")

    os.makedirs(os.path.dirname(summary_path) or ".", exist_ok=True)
    with open(summary_path, "w") as f:
        f.write("\n".join(lines) + "\n")


def _cmd_bench_sweep(args: argparse.Namespace) -> int:
    """Cartesian sweep: encoders × datasets. One model load per encoder.

    Writes one JSON per cell + a SUMMARY.md with a 3×3 grid and deltas.
    Out-of-band cells are flagged but do not abort the sweep by default
    (override with --stop-on-out-of-band). Truthful logging; no silent
    re-runs with different seeds.
    """
    import os

    from vitruvius.encoders import get_encoder
    from vitruvius.utils.device import pick_device

    device_str = str(pick_device(None if args.device == "auto" else args.device))
    os.makedirs(args.output_dir, exist_ok=True)

    results: list[dict] = []
    out_of_band: list[tuple[str, str]] = []
    failures: list[tuple[str, str, str]] = []

    total_cells = len(args.encoders) * len(args.datasets)
    cell_idx = 0

    for encoder_name in args.encoders:
        _log.info("sweep.encoder_load encoder=%s device=%s", encoder_name, device_str)
        enc = get_encoder(encoder_name, device=device_str)
        for dataset in args.datasets:
            cell_idx += 1
            _log.info("sweep.cell %d/%d encoder=%s dataset=%s",
                      cell_idx, total_cells, encoder_name, dataset)
            reference = REFERENCES_PHASE3.get((encoder_name, dataset))
            out_path = os.path.join(args.output_dir, f"{encoder_name}__{dataset}.json")
            try:
                artifact = _run_bench_core(
                    encoder=enc,
                    encoder_name=encoder_name,
                    dataset=dataset,
                    split_name=args.split,
                    batch_size=args.batch_size,
                    top_k=args.top_k,
                    device_str=device_str,
                    limit=None,
                    output_path=out_path,
                    reference=reference,
                    tolerance=args.tolerance,
                )
            except Exception as e:
                _log.error("sweep.cell_failed encoder=%s dataset=%s err=%r",
                           encoder_name, dataset, e)
                failures.append((encoder_name, dataset, repr(e)))
                continue
            results.append(artifact)
            if artifact["result"].get("in_band") is False:
                out_of_band.append((encoder_name, dataset))
                _log.warning(
                    "sweep.out_of_band encoder=%s dataset=%s measured=%.4f ref=%.2f delta=%+.4f tol=%.2f",
                    encoder_name, dataset,
                    artifact["result"]["value"],
                    artifact["result"]["reference"],
                    artifact["result"]["delta_from_reference"],
                    args.tolerance,
                )
                if args.stop_on_out_of_band:
                    _log.error("sweep.abort reason=out-of-band (--stop-on-out-of-band)")
                    break
        else:
            continue
        break

    summary_path = os.path.join(args.output_dir, "SUMMARY.md")
    _write_sweep_summary(summary_path, results, args.encoders, args.datasets)
    _log.info("sweep.summary_written path=%s", summary_path)

    print(json.dumps({
        "cells_run": len(results),
        "cells_total": total_cells,
        "out_of_band": out_of_band,
        "failures": failures,
        "summary_md": summary_path,
    }, indent=2))

    return 0 if (not out_of_band and not failures) else 1


def _write_profile_summary(
    path: str,
    rows: list[dict],
    encoders: list[str],
    datasets: list[str],
    batch_sizes: list[int],
) -> None:
    import os as _os

    lookup = {(r["encoder"], r["dataset"]): r for r in rows}

    lines = [
        "# Phase 3.5 — Latency profile (30% milestone)",
        "",
        "Query encoding latency is the production-critical number (one batch per"
        " retrieval request); document encoding throughput is the offline cost.",
        "",
        "## Query encoding latency at batch size 1 — median ms",
        "",
        "| Encoder \\ Dataset | " + " | ".join(datasets) + " |",
        "|" + "|".join(["---"] * (1 + len(datasets))) + "|",
    ]
    for enc in encoders:
        cells = [f"`{enc}`"]
        for ds in datasets:
            r = lookup.get((enc, ds))
            cells.append(f"{r['lat_ms'][1]['median']:.3f}" if r else "—")
        lines.append("| " + " | ".join(cells) + " |")

    lines += [
        "",
        "## Query latency at batch 1 — percentiles (ms)",
        "",
        "| Cell | median | p50 | p90 | p99 |",
        "|---|---:|---:|---:|---:|",
    ]
    for enc in encoders:
        for ds in datasets:
            r = lookup.get((enc, ds))
            if r is None:
                continue
            x = r["lat_ms"][1]
            lines.append(
                f"| `{enc}` x `{ds}` | {x['median']:.3f} | {x['p50']:.3f} "
                f"| {x['p90']:.3f} | {x['p99']:.3f} |"
            )

    lines += [
        "",
        "## All batch sizes — median ms",
        "",
        "| Cell | " + " | ".join(f"bs={bs}" for bs in batch_sizes) + " |",
        "|" + "|".join(["---"] * (1 + len(batch_sizes))) + "|",
    ]
    for enc in encoders:
        for ds in datasets:
            r = lookup.get((enc, ds))
            if r is None:
                continue
            cells = [f"`{enc}` x `{ds}`"]
            for bs in batch_sizes:
                cells.append(f"{r['lat_ms'][bs]['median']:.3f}")
            lines.append("| " + " | ".join(cells) + " |")

    lines += [
        "",
        "## Document encoding throughput at batch 32 — docs / second",
        "",
        "| Encoder \\ Dataset | " + " | ".join(datasets) + " |",
        "|" + "|".join(["---"] * (1 + len(datasets))) + "|",
    ]
    for enc in encoders:
        cells = [f"`{enc}`"]
        for ds in datasets:
            r = lookup.get((enc, ds))
            cells.append(f"{r['throughput']:.1f}" if r else "—")
        lines.append("| " + " | ".join(cells) + " |")

    lines += [
        "",
        "## Methodology",
        "",
        "- Latency measured via `torch.cuda.Event` on CUDA (see "
        "`vitruvius.evaluation.latency_profiler`). 10 warmup + 100 measured"
        " passes per batch size. Percentiles are over the 100 measured times.",
        "- Query samples: 200 queries sampled (`random.Random(seed=1729).sample`)"
        " from each dataset's test split qrels-having set.",
        "- Document samples: 200 documents sampled (same seed, separate call)"
        " from each dataset's full corpus.",
        "- Document encoding throughput: 3 warmup rounds then one wall-clock"
        " timed encode of all 200 sampled docs at batch size 32."
        " Reported as `200 / wall_time`.",
        "- Token length distributions (computed once, encoder-agnostic) are"
        " in `dataset_length_stats.json`. Canonical tokenizer:"
        " `sentence-transformers/all-MiniLM-L6-v2`.",
        "- Seed 1729. Device: `cuda`. Hardware: see per-cell JSONs.",
        "- These numbers are for this pod run only. Production latency is"
        " hardware-sensitive; treat these as within-study comparison numbers,"
        " not absolute benchmarks.",
        "",
    ]

    _os.makedirs(_os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def _cmd_profile(args: argparse.Namespace) -> int:
    """Latency profile across encoders x datasets x batch sizes.

    Token-length distributions affect transformer latency nonlinearly
    (attention is O(n^2)), so this profiles on real BEIR queries/documents
    rather than synthetic fixed-length strings (session-02 handoff 3.5.3).
    """
    import os
    import random
    import statistics
    import time
    from datetime import datetime, timezone

    import torch

    from vitruvius.data.beir_loader import load_beir
    from vitruvius.encoders import get_encoder
    from vitruvius.evaluation.latency_profiler import profile as run_profile
    from vitruvius.utils.device import pick_device

    set_seed(args.seed)
    device_str = str(pick_device(None if args.device == "auto" else args.device))
    os.makedirs(args.output_dir, exist_ok=True)

    _log.info(
        "profile.start encoders=%s datasets=%s batch_sizes=%s device=%s sample_size=%d",
        args.encoders, args.datasets, args.batch_sizes, device_str, args.sample_size,
    )

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

    def _percs(xs: list[int]) -> dict:
        if not xs:
            return {"min": 0, "median": 0, "max": 0, "p95": 0, "count": 0}
        xs_sorted = sorted(xs)
        return {
            "min": xs_sorted[0],
            "median": int(statistics.median(xs_sorted)),
            "max": xs_sorted[-1],
            "p95": xs_sorted[int(0.95 * (len(xs_sorted) - 1))],
            "count": len(xs_sorted),
        }

    length_stats: dict[str, dict] = {}
    sample_q_by_ds: dict[str, list[str]] = {}
    sample_d_by_ds: dict[str, list[str]] = {}

    for ds in args.datasets:
        _log.info("profile.length_stats dataset=%s", ds)
        split = load_beir(ds, split=args.split)
        qids_ok = [q for q in split.queries if q in split.qrels and split.qrels[q]]
        q_texts_all = [split.queries[q] for q in qids_ok]
        doc_texts_all = [
            (split.corpus[d].get("title", "") + " " + split.corpus[d].get("text", "")).strip()
            for d in split.corpus.keys()
        ]

        q_lens = [len(tok.encode(q, add_special_tokens=True, truncation=False)) for q in q_texts_all]
        doc_subsample = random.Random(args.seed).sample(
            doc_texts_all, min(1000, len(doc_texts_all))
        )
        d_lens = [
            len(tok.encode(d, add_special_tokens=True, truncation=False))
            for d in doc_subsample
        ]

        length_stats[ds] = {
            "corpus_size": len(doc_texts_all),
            "n_queries_eval": len(q_texts_all),
            "query_token_lengths": _percs(q_lens),
            "doc_token_lengths_from_sample": _percs(d_lens),
            "doc_token_sample_size": len(doc_subsample),
        }

        rng_q = random.Random(args.seed)
        rng_d = random.Random(args.seed + 1)
        sample_q_by_ds[ds] = rng_q.sample(
            q_texts_all, min(args.sample_size, len(q_texts_all))
        )
        sample_d_by_ds[ds] = rng_d.sample(
            doc_texts_all, min(args.sample_size, len(doc_texts_all))
        )

    length_stats_path = os.path.join(args.output_dir, "dataset_length_stats.json")
    with open(length_stats_path, "w") as f:
        json.dump(
            {
                "tokenizer": "sentence-transformers/all-MiniLM-L6-v2",
                "datasets": length_stats,
                "note": "token lengths are untruncated, including [CLS]/[SEP].",
            },
            f, indent=2, sort_keys=True,
        )
    _log.info("profile.length_stats_written path=%s", length_stats_path)

    def make_q_fn(enc, samples):
        def fn(batch_size):
            enc.encode_queries(samples[:batch_size], batch_size=batch_size)
        return fn

    summary_rows: list[dict] = []
    BATCH_THROUGHPUT = 32

    for enc_name in args.encoders:
        _log.info("profile.encoder_load encoder=%s device=%s", enc_name, device_str)
        enc = get_encoder(enc_name, device=device_str)
        for ds in args.datasets:
            _log.info("profile.cell encoder=%s dataset=%s", enc_name, ds)
            q_samples = sample_q_by_ds[ds]
            d_samples = sample_d_by_ds[ds]

            t0 = time.perf_counter()
            lat = run_profile(
                make_q_fn(enc, q_samples),
                n_warmup=args.n_warmup,
                n_measure=args.n_measure,
                batch_sizes=tuple(args.batch_sizes),
                device=device_str,
            )
            t_lat = time.perf_counter() - t0

            for _ in range(3):
                enc.encode_documents(d_samples[:BATCH_THROUGHPUT], batch_size=BATCH_THROUGHPUT)
            if device_str == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            enc.encode_documents(d_samples, batch_size=BATCH_THROUGHPUT)
            if device_str == "cuda":
                torch.cuda.synchronize()
            t_throughput = time.perf_counter() - t0
            throughput = len(d_samples) / t_throughput

            artifact = {
                "vitruvius_version": __version__,
                "git_commit": _git_head(),
                "run_timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "config": {
                    "encoder": enc_name,
                    "similarity": enc.similarity,
                    "dataset": ds,
                    "split": args.split,
                    "sample_size": args.sample_size,
                    "n_warmup": args.n_warmup,
                    "n_measure": args.n_measure,
                    "batch_sizes": list(args.batch_sizes),
                    "device": device_str,
                    "seed": args.seed,
                    "throughput_batch_size": BATCH_THROUGHPUT,
                },
                "hardware": _hardware_snapshot(),
                "query_latency_ms": {str(k): v for k, v in lat.items()},
                "doc_throughput": {
                    "batch_size": BATCH_THROUGHPUT,
                    "n_docs": len(d_samples),
                    "total_seconds": round(t_throughput, 4),
                    "docs_per_second": round(throughput, 2),
                },
                "runtime_seconds": {
                    "query_latency_measurement": round(t_lat, 4),
                    "doc_throughput_measurement": round(t_throughput, 4),
                },
            }
            out = os.path.join(args.output_dir, f"{enc_name}__{ds}.json")
            with open(out, "w") as f:
                json.dump(artifact, f, indent=2, sort_keys=True)

            lat1 = lat[1]["median"]
            _log.info(
                "profile.done encoder=%s dataset=%s lat@1_median_ms=%.3f docs/s=%.1f out=%s",
                enc_name, ds, lat1, throughput, out,
            )
            summary_rows.append(
                {"encoder": enc_name, "dataset": ds, "lat_ms": lat, "throughput": throughput}
            )

    summary_path = os.path.join(args.output_dir, "SUMMARY.md")
    _write_profile_summary(summary_path, summary_rows, args.encoders, args.datasets, list(args.batch_sizes))
    _log.info("profile.summary_written path=%s", summary_path)

    print(json.dumps(
        {
            "cells_run": len(summary_rows),
            "cells_total": len(args.encoders) * len(args.datasets),
            "output_dir": args.output_dir,
            "length_stats": length_stats_path,
            "summary_md": summary_path,
        },
        indent=2,
    ))
    return 0


def _cmd_shuffle(args: argparse.Namespace) -> int:
    return _not_yet("Phase 8 (position sensitivity)")


def _cmd_prune(args: argparse.Namespace) -> int:
    return _not_yet("Phase 7 (attention head pruning)")


_DISPATCH = {
    "smoke": _cmd_smoke,
    "bench": _cmd_bench,
    "bench-sweep": _cmd_bench_sweep,
    "profile": _cmd_profile,
    "shuffle": _cmd_shuffle,
    "prune": _cmd_prune,
}


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    return _DISPATCH[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
