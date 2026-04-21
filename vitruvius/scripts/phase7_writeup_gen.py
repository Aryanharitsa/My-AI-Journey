"""Generate experiments/phase7/{SUMMARY.md, README.md} from the JSONs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ENCODERS = ["bert-base", "minilm-l6-v2", "gte-small"]
DATASETS = ["nfcorpus", "scifact", "fiqa"]


def _load(p: Path) -> dict | None:
    return json.loads(p.read_text()) if p.exists() else None


def _load_cell(root: Path, kind: str, enc: str, ds: str) -> dict | None:
    return _load(root / kind / f"{enc}__{ds}.json")


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def write_summary(phase7_dir: Path, out_path: Path, stability_json: Path) -> None:
    imp_dir = phase7_dir / "head_importance"
    cum_dir = phase7_dir / "cumulative_pruning"
    stab = _load(stability_json) or {"per_encoder": {}}

    lines = [
        "# Phase 7 — Attention head pruning for retrieval (80% milestone)",
        "",
        "What this phase measures: for each of the three pre-trained transformer",
        "retrieval encoders, zero one attention head at a time and measure the",
        "drop in nDCG@10 on three BEIR test subsets. Rank heads by importance,",
        "then cumulatively prune the least-important N heads to characterize the",
        "'how many heads can we remove before retrieval breaks?' curve.",
        "",
        "This is the direct, quantified evidence for the Phase 9 opinions",
        "section's `transformers are over-provisioned in heads for retrieval`",
        "claim.",
        "",
        "## Headline numbers — heads prunable at ≤N% nDCG@10 drop",
        "",
        "(Averaged across NFCorpus / SciFact / FiQA.)",
        "",
        "| Encoder | Total heads | Baseline nDCG@10 | Prunable @ ≤5% drop | Prunable @ ≤10% drop |",
        "|---|---:|---:|---:|---:|",
    ]
    for enc in ENCODERS:
        cum_cells = [_load_cell(phase7_dir, "cumulative_pruning", enc, ds) for ds in DATASETS]
        imp_cells = [_load_cell(phase7_dir, "head_importance", enc, ds) for ds in DATASETS]
        cum_cells = [c for c in cum_cells if c is not None]
        imp_cells = [c for c in imp_cells if c is not None]
        if not cum_cells or not imp_cells:
            lines.append(f"| `{enc}` | — | — | — | — |")
            continue
        total = cum_cells[0]["total_heads"]
        baseline = _mean([c["baseline_nDCG@10"] for c in cum_cells])
        t5 = [c["thresholds"]["heads_prunable_at_5pct_drop"] for c in cum_cells]
        t10 = [c["thresholds"]["heads_prunable_at_10pct_drop"] for c in cum_cells]
        def _fmt(ts):
            vals = [v for v in ts if v is not None]
            if not vals:
                return "—"
            avg = sum(vals) / len(vals)
            pct = 100.0 * avg / total
            return f"{avg:.1f} / {total} ({pct:.1f}%)"
        lines.append(
            f"| `{enc}` | {total} | {baseline:.4f} | {_fmt(t5)} | {_fmt(t10)} |"
        )

    lines += [
        "",
        "## Per-dataset headline (5% drop threshold)",
        "",
        "| Encoder \\ Dataset | " + " | ".join(DATASETS) + " |",
        "|---|" + "|".join([":---:"] * len(DATASETS)) + "|",
    ]
    for enc in ENCODERS:
        cells = [f"`{enc}`"]
        for ds in DATASETS:
            c = _load_cell(phase7_dir, "cumulative_pruning", enc, ds)
            if c is None:
                cells.append("—")
                continue
            t5 = c["thresholds"]["heads_prunable_at_5pct_drop"]
            total = c["total_heads"]
            if t5 is None:
                cells.append("never (always >5% drop)")
            else:
                cells.append(f"{t5} / {total} ({100.0 * t5 / total:.0f}%)")
        lines.append("| " + " | ".join(cells) + " |")

    lines += ["", "## Cross-dataset head stability (Spearman ρ)", ""]
    lines.append("| Encoder | ρ(nfcorpus,scifact) | ρ(nfcorpus,fiqa) | ρ(scifact,fiqa) | Mean ρ |")
    lines.append("|---|---:|---:|---:|---:|")
    for enc in ENCODERS:
        d = stab.get("per_encoder", {}).get(enc)
        if d is None:
            lines.append(f"| `{enc}` | — | — | — | — |")
            continue
        p = d["pairwise_spearman"]
        lines.append(
            f"| `{enc}` | {p.get('rho(nfcorpus,scifact)', '—')} | "
            f"{p.get('rho(nfcorpus,fiqa)', '—')} | "
            f"{p.get('rho(scifact,fiqa)', '—')} | "
            f"**{d['mean_rho']}** |"
        )

    lines += [
        "",
        "## Layer-wise observations",
        "",
        "See [`figures/head_importance_by_layer.png`](../../figures/head_importance_by_layer.png) "
        "for per-(encoder, layer) distributions of head importance. Retrieval "
        "literature's intuition is that early layers (surface features) should "
        "be more prunable than late layers (task-specific). Whether that holds "
        "here is encoder-dependent and dataset-dependent; the boxplot is the "
        "authoritative view.",
        "",
        "## Methodology",
        "",
        "- Head ablation via HuggingFace's native `head_mask` argument on `AutoModel.forward` "
        "(same API Michel et al. 2019 used). No state-dict hacking, no custom hooks.",
        "- Pre-trained transformer encoders only: `bert-base` (msmarco-bert-base-dot-v5, 144 heads), "
        "`minilm-l6-v2` (72 heads), `gte-small` (144 heads). LSTM / CNN / Mamba encoders have "
        "no multi-head attention in the transformer sense and are scoped out of this phase.",
        "- Pooling per encoder config: mean for MiniLM and BERT-dot-v5, CLS for GTE.",
        "- `bert-base` keeps dot-product (no L2-norm); MiniLM and GTE use cosine (L2-normalized). "
        "Matches each checkpoint's training objective per `Encoder.similarity`.",
        "- BEIR subsets: nfcorpus, scifact, fiqa test splits. FAISS `IndexFlatIP`, top-k=100.",
        "- AMP fp16 autocast for inference speed. Baseline nDCG@10 reproduces Phase 3's "
        "numbers within fp16 rounding (|Δ| ≤ 2e-5).",
        "- Cumulative pruning ranks heads by single-head importance (ascending), "
        "zeros the N least-important, measures. See limitations below.",
        "",
        "## Interpretation",
        "",
        "_(Filled in by operator/archivist during Phase 9 writeup synthesis.)_",
        "",
        "## Limitations",
        "",
        "1. **Single-head-importance ordering used for cumulative pruning.** "
        "Heads can compensate for each other, so the true-optimal set of N heads to "
        "prune is NOT necessarily the N lowest-scoring-individually. A Taylor-saliency "
        "or iterative-greedy baseline would be a stronger claim — future work.",
        "2. **Zero-shot ablation only.** No fine-tuning after pruning. Michel et al. "
        "report 3-8% recovery from post-pruning fine-tuning — untested here.",
        "3. **Out-of-distribution evaluation.** BEIR is zero-shot; head importance "
        "measured on in-distribution MS MARCO might differ. Scope decision.",
        "4. **AMP fp16 inference.** Inference-side numerics differ from fp32 by "
        "~2e-5 on baseline; order of magnitude below the per-head deltas of interest.",
        "",
    ]
    out_path.write_text("\n".join(lines) + "\n")


def write_readme(phase7_dir: Path, out_path: Path) -> None:
    (out_path).write_text(
        "# Phase 7 — head pruning artifacts\n"
        "\n"
        "## Files\n"
        "\n"
        "- `head_importance/<encoder>__<dataset>.json` (9 files)\n"
        "    Per-head ablation results. Schema:\n"
        "    `{encoder, dataset, baseline_nDCG@10, num_layers, num_heads_per_layer, "
        "total_heads, per_head_results: [{layer, head, nDCG@10, delta_nDCG@10}], "
        "ranked_by_importance: [{layer, head, rank, delta_nDCG@10}], config, hardware, runtime_seconds}`\n"
        "- `cumulative_pruning/<encoder>__<dataset>.json` (9 files)\n"
        "    Cumulative-prune-N-least-important curves. Schema:\n"
        "    `{encoder, dataset, baseline_nDCG@10, ordering, curve: [{heads_pruned, nDCG@10, "
        "rel_drop_pct}], thresholds: {heads_prunable_at_5pct_drop, heads_prunable_at_10pct_drop}, ...}`\n"
        "- `head_stability_analysis.md` — human-readable Spearman table.\n"
        "- `head_stability.json` — machine-readable version.\n"
        "- `head_sweep.log` — raw stdout of the full 9-cell importance sweep.\n"
        "- `cumulative_sweep.log` — raw stdout of the cumulative pruning run.\n"
        "- `SUMMARY.md` — headline tables + interpretation.\n"
        "\n"
        "## Reproduce\n"
        "\n"
        "```bash\n"
        "python scripts/head_importance_sweep.py \\\n"
        "    --encoders minilm-l6-v2 bert-base gte-small \\\n"
        "    --datasets nfcorpus scifact fiqa \\\n"
        "    --batch-size 128 --top-k 100 --device cuda\n"
        "\n"
        "python scripts/cumulative_pruning_sweep.py \\\n"
        "    --encoders minilm-l6-v2 bert-base gte-small \\\n"
        "    --datasets nfcorpus scifact fiqa \\\n"
        "    --batch-size 128 --top-k 100 --device cuda\n"
        "\n"
        "python scripts/phase7_analysis.py\n"
        "```\n"
    )


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase7-dir", type=Path, default=Path("experiments/phase7"))
    ap.add_argument("--out-summary", type=Path, default=Path("experiments/phase7/SUMMARY.md"))
    ap.add_argument("--out-readme", type=Path, default=Path("experiments/phase7/README.md"))
    ap.add_argument("--stability-json", type=Path, default=Path("experiments/phase7/head_stability.json"))
    args = ap.parse_args(argv)

    write_summary(args.phase7_dir, args.out_summary, args.stability_json)
    write_readme(args.phase7_dir, args.out_readme)
    print(json.dumps({"summary": str(args.out_summary), "readme": str(args.out_readme)}))
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
