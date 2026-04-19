"""Phase 5 SUMMARY.md generator — reads from phase5/{training_jsons, bench/, profile/}."""
from __future__ import annotations

import json
from pathlib import Path

DIR = Path("experiments/phase5")
BENCH_DIR = DIR / "bench"
PROFILE_DIR = DIR / "profile"
ENCODERS = ["lstm-retriever", "conv-retriever", "mamba-retriever-fs"]
DATASETS = ["nfcorpus", "scifact", "fiqa"]


def _load(p: Path) -> dict | None:
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except Exception:
        return None


def main() -> None:
    trainings = {enc: _load(DIR / f"{enc}__training.json") for enc in ENCODERS}
    bench = {(enc, ds): _load(BENCH_DIR / f"{enc}__{ds}.json")
             for enc in ENCODERS for ds in DATASETS}
    latency = {(enc, ds): _load(PROFILE_DIR / f"{enc}__{ds}.json")
               for enc in ENCODERS for ds in DATASETS}

    lines = [
        "# Phase 5 — From-scratch encoder training + evaluation (55% milestone)",
        "",
        "Three non-transformer bi-encoders trained from random init on 500K MS MARCO",
        "triplets with identical hyperparameters (InfoNCE τ=0.05, AdamW lr=1e-4, 3 epochs,",
        "batch 64, linear-warmup + cosine decay, AMP fp16, max seq len 128,",
        "BERT WordPiece tokenizer, seed 1729). Evaluated through the same bench-sweep",
        "and latency profiler used on the pre-trained transformers in Phases 3 / 3.5.",
        "",
        "Absorbs the deferred Phase 4 (session-02 §4.7 kill-switch on SPScanner): instead",
        "of integrating a pre-trained Mamba bi-encoder that doesn't exist publicly, Phase 5",
        "trains a Mamba2 bi-encoder from scratch alongside LSTM and CNN, putting all three",
        "alternative architectures on equal footing vs. the pre-trained transformers.",
        "",
        "## Training",
        "",
        "| Encoder | Params | Steps | Best val loss | Final val loss | Wall (s) | Peak GPU (MB) | AMP | num_workers |",
        "|---|---:|---:|---:|---:|---:|---:|:---:|:---:|",
    ]
    for enc in ENCODERS:
        t = trainings[enc]
        if t is None:
            lines.append(f"| `{enc}` | — | — | — | — | — | — | — | — |")
            continue
        best = t.get("best_val_loss")
        final = t.get("final_val_loss")
        cfg = t.get("config", {})
        lines.append(
            f"| `{enc}` | "
            f"{t['param_count']/1e6:.2f}M | "
            f"{t['steps_completed']} | "
            f"{best:.4f} | {final:.4f} | "
            f"{t['wall_seconds']:.1f} | "
            f"{t.get('peak_gpu_memory_mb', '—')} | "
            f"{'✓' if t.get('amp_enabled') else '—'} | "
            f"{cfg.get('num_workers', 2)} |"
        )

    lines += [
        "",
        "## nDCG@10 (ours_from_scratch) on BEIR test subsets",
        "",
        "| Encoder \\ Dataset | " + " | ".join(DATASETS) + " |",
        "|---|" + "|".join([":---:"] * len(DATASETS)) + "|",
    ]
    for enc in ENCODERS:
        cells = [f"`{enc}`"]
        for ds in DATASETS:
            r = bench.get((enc, ds))
            cells.append(f"{r['metrics']['ours_from_scratch']['nDCG@10']:.4f}" if r else "—")
        lines.append("| " + " | ".join(cells) + " |")

    lines += [
        "",
        "### Cross-check status (from-scratch vs pytrec_eval)",
        "",
        "| Cell | nDCG@10 ours | nDCG@10 pytrec | \\|Δ\\| | Recall@10 bit-exact? |",
        "|---|---:|---:|---:|:---:|",
    ]
    for enc in ENCODERS:
        for ds in DATASETS:
            r = bench.get((enc, ds))
            if r is None:
                continue
            ours = r["metrics"]["ours_from_scratch"]["nDCG@10"]
            py = r["metrics"]["pytrec_eval"]["nDCG@10"]
            delta = abs(ours - py)
            recall_deltas = [v for k, v in r["metrics"]["delta_abs"].items()
                             if k.startswith("Recall@")]
            recall_ok = "✅" if all(v == 0 for v in recall_deltas) else "❌"
            lines.append(
                f"| `{enc}` × `{ds}` | {ours:.4f} | {py:.4f} | {delta:.2e} | {recall_ok} |"
            )

    for title, key in [
        ("Query encoding latency @ batch size 1 — median ms", "1"),
        ("Query encoding latency @ batch size 32 — median ms", "32"),
    ]:
        lines += [
            "",
            f"## {title}",
            "",
            "| Encoder \\ Dataset | " + " | ".join(DATASETS) + " |",
            "|---|" + "|".join([":---:"] * len(DATASETS)) + "|",
        ]
        for enc in ENCODERS:
            cells = [f"`{enc}`"]
            for ds in DATASETS:
                r = latency.get((enc, ds))
                if r and key in r.get("query_latency_ms", {}):
                    cells.append(f"{r['query_latency_ms'][key]['median']:.3f}")
                else:
                    cells.append("—")
            lines.append("| " + " | ".join(cells) + " |")

    lines += [
        "",
        "## Document throughput @ batch 32 — docs/s",
        "",
        "| Encoder \\ Dataset | " + " | ".join(DATASETS) + " |",
        "|---|" + "|".join([":---:"] * len(DATASETS)) + "|",
    ]
    for enc in ENCODERS:
        cells = [f"`{enc}`"]
        for ds in DATASETS:
            r = latency.get((enc, ds))
            if r and "doc_throughput" in r:
                cells.append(f"{r['doc_throughput']['docs_per_second']:.1f}")
            else:
                cells.append("—")
        lines.append("| " + " | ".join(cells) + " |")

    lines += [
        "",
        "## Methodology",
        "",
        "- InfoNCE contrastive loss with in-batch negatives, temperature 0.05. "
        "Each batch of 64 triplets provides 64 queries and 128 candidate passages "
        "(positives + explicit hard negatives).",
        "- AdamW lr=1e-4, wd=0.01, linear warmup (first 10% of steps) then cosine "
        "decay. Gradient clipping at 1.0. AMP fp16 throughout.",
        "- 3 epochs × 500K triplets / batch 64 = 23,437 planned steps; actual is "
        "slightly lower due to `drop_last=True` and a small number of skipped "
        "malformed MS MARCO rows (control chars in scraped web text).",
        "- Validation every 500 steps on held-out 5K triplets. Best-val checkpoint "
        "saved to `models/<encoder>/best.pt`; final checkpoint to `final.pt`.",
        "- BERT-base-uncased WordPiece tokenizer; max_seq_len=128.",
        "- `lstm-retriever` / `conv-retriever` trained with DataLoader `num_workers=2`. "
        "`mamba-retriever-fs` trained with `num_workers=0` — standard multiprocessing "
        "fork inherits inconsistent Triton-JIT state from mamba_ssm's selective-scan "
        "kernels and workers segfault. Same model, same hyperparameters, same data; "
        "only the dataloader parallelism differs. Full diagnosis in "
        "`notes/mamba_install_attempt_02.md`.",
        "- Bench: FAISS `IndexFlatIP`, top-k=100, L2-normalized embeddings, "
        "graded-gain nDCG@k. pytrec_eval runs as a cross-check (gate IR-2).",
        "- Latency: `torch.cuda.Event`, 10 warmup + 100 measured passes per batch "
        "size, 200 queries / 200 docs sampled per dataset (seed 1729).",
        "- Hardware: A100-SXM4-80GB, torch 2.4.1+cu124, FAISS 1.13.2 (CPU), "
        "Python 3.11.10.",
        "",
    ]

    (DIR / "SUMMARY.md").write_text("\n".join(lines) + "\n")
    print(f"wrote {DIR / 'SUMMARY.md'}")


if __name__ == "__main__":
    main()
