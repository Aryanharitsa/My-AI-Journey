"""Phase 8 analysis: position sensitivity scores + figures.

Reads:
- experiments/phase3/bench/<enc>__<ds>.json  (transformer baselines)
- experiments/phase5/bench/<enc>__<ds>.json  (from-scratch baselines)
- experiments/phase3/<enc>__<ds>.json        (fallback for transformer baselines)
- experiments/phase5/bench/<enc>__<ds>.json  (fallback for from-scratch)
- experiments/phase8/<enc>__<ds>__<mode>.json × 54

Writes:
- experiments/phase8/SUMMARY.md
- experiments/phase8/README.md
- experiments/phase8/position_sensitivity.json  (machine-readable)
- experiments/phase8/query_level_sensitivity.parquet  (per-query shuffle deltas)
- figures/position_sensitivity.png  (grouped bar + caption)
- figures/sensitivity_by_family.png (per-family average + caption)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

ENCODERS = [
    # (name, family, color)
    ("minilm-l6-v2",     "transformer",   "#1f77b4"),
    ("bert-base",        "transformer",   "#2a6aa8"),
    ("gte-small",        "transformer",   "#5fa6d6"),
    ("lstm-retriever",   "recurrent",     "#ff7f0e"),
    ("conv-retriever",   "convolutional", "#2ca02c"),
    ("mamba-retriever-fs", "ssm",         "#d62728"),
]
DATASETS = ["nfcorpus", "scifact", "fiqa"]
MODES = ["docs-shuffled", "queries-shuffled", "both-shuffled"]
FAMILIES = ["transformer", "recurrent", "convolutional", "ssm"]


def _load(p: Path):
    return json.loads(p.read_text()) if p.exists() else None


def _baseline_ndcg(enc: str, ds: str, phase3_dir: Path, phase5_dir: Path):
    """Find baseline nDCG@10 from Phase 3 (transformers) or Phase 5 (from-scratch)."""
    # Phase 5 has bench/ subdir; Phase 3 has direct JSONs (from backfill commit).
    for p in [
        phase5_dir / "bench" / f"{enc}__{ds}.json",
        phase3_dir / f"{enc}__{ds}.json",
        phase3_dir / "bench" / f"{enc}__{ds}.json",
    ]:
        d = _load(p)
        if d is not None:
            return float(d["metrics"]["ours_from_scratch"]["nDCG@10"]), p
    return None, None


def build_sensitivity_table(phase8_dir: Path, phase3_dir: Path, phase5_dir: Path):
    """Return dict {(enc, ds, mode): {shuffled_ndcg, baseline_ndcg, sensitivity}}."""
    out = {}
    for enc, _, _ in ENCODERS:
        baseline_per_ds = {}
        for ds in DATASETS:
            b, _ = _baseline_ndcg(enc, ds, phase3_dir, phase5_dir)
            baseline_per_ds[ds] = b
        for ds in DATASETS:
            baseline = baseline_per_ds[ds]
            for mode in MODES:
                p = phase8_dir / f"{enc}__{ds}__{mode}.json"
                d = _load(p)
                if d is None or baseline is None:
                    out[(enc, ds, mode)] = None
                    continue
                shuffled = float(d["metrics"]["ours_from_scratch"]["nDCG@10"])
                sens = (baseline - shuffled) / baseline if baseline > 0 else None
                out[(enc, ds, mode)] = {
                    "baseline_nDCG@10": round(baseline, 6),
                    "shuffled_nDCG@10": round(shuffled, 6),
                    "position_sensitivity": round(sens, 6) if sens is not None else None,
                }
    return out


def write_stability_json(out_path: Path, table: dict):
    rows = []
    for (enc, ds, mode), v in table.items():
        rows.append({"encoder": enc, "dataset": ds, "mode": mode, **(v or {})})
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({"rows": rows}, indent=2, sort_keys=True))


def build_query_parquet(phase8_dir: Path, phase3_dir: Path, phase5_dir: Path, out_path: Path):
    """Per-query baseline vs shuffled nDCG@10 deltas.
    Returns (rows_list, wrote_parquet: bool). Falls back to JSON if pandas/pyarrow missing.
    """
    rows = []
    for enc, fam, _ in ENCODERS:
        for ds in DATASETS:
            # Find baseline per-query results
            baseline_d = None
            for p in [
                phase5_dir / "bench" / f"{enc}__{ds}.json",
                phase3_dir / f"{enc}__{ds}.json",
                phase3_dir / "bench" / f"{enc}__{ds}.json",
            ]:
                baseline_d = _load(p)
                if baseline_d is not None:
                    break
            if baseline_d is None:
                continue
            per_q_baseline = baseline_d.get("per_query_results", {})
            for mode in MODES:
                p = phase8_dir / f"{enc}__{ds}__{mode}.json"
                d = _load(p)
                if d is None:
                    continue
                per_q_shuffled = d.get("per_query_results", {})
                for qid, bq in per_q_baseline.items():
                    sq = per_q_shuffled.get(qid)
                    if sq is None:
                        continue
                    rows.append({
                        "encoder": enc, "family": fam, "dataset": ds, "mode": mode,
                        "query_id": qid,
                        "baseline_nDCG@10": bq["nDCG@10"],
                        "shuffled_nDCG@10": sq["nDCG@10"],
                        "delta_nDCG@10": bq["nDCG@10"] - sq["nDCG@10"],
                        "baseline_hit@10": bq["hit@10"],
                        "shuffled_hit@10": sq["hit@10"],
                    })
    try:
        import pandas as pd
        df = pd.DataFrame(rows)
        df.to_parquet(out_path, index=False)
        return rows, True
    except Exception:
        json_path = out_path.with_suffix(".json")
        json_path.write_text(json.dumps(rows, indent=None))
        return rows, False


def figure_position_sensitivity(table: dict, out_path: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(DATASETS), figsize=(6 * len(DATASETS), 4.5),
                             constrained_layout=True, sharey=True)
    x = np.arange(len(ENCODERS))
    width = 0.28

    for ax, ds in zip(axes, DATASETS, strict=True):
        for mi, mode in enumerate(MODES):
            vals = []
            for enc, _, _ in ENCODERS:
                cell = table.get((enc, ds, mode))
                vals.append(
                    cell["position_sensitivity"] if cell and cell["position_sensitivity"] is not None else 0.0
                )
            ax.bar(x + (mi - 1) * width, vals, width, label=mode)
        ax.axhline(0.0, color="gray", linestyle="--", alpha=0.5, linewidth=0.8)
        ax.axhline(1.0, color="gray", linestyle=":", alpha=0.5, linewidth=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels([e[0] for e in ENCODERS], rotation=30, ha="right", fontsize=9)
        ax.set_title(ds)
        ax.set_ylabel("position sensitivity  (baseline − shuffled) / baseline")
        if ds == DATASETS[0]:
            ax.legend(loc="upper left", fontsize=9)

    fig.suptitle("Position sensitivity per encoder × dataset × shuffle mode", fontsize=11)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def figure_sensitivity_by_family(table: dict, out_path: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Average across encoders in family, across datasets, for docs-shuffled mode
    by_family_per_mode = {}
    for mode in MODES:
        by_family_per_mode[mode] = {}
        for fam in FAMILIES:
            vals = []
            for enc, f, _ in ENCODERS:
                if f != fam:
                    continue
                for ds in DATASETS:
                    c = table.get((enc, ds, mode))
                    if c and c["position_sensitivity"] is not None:
                        vals.append(c["position_sensitivity"])
            by_family_per_mode[mode][fam] = (
                float(np.mean(vals)) if vals else 0.0,
                float(np.std(vals)) if vals else 0.0,
                len(vals),
            )

    fig, ax = plt.subplots(figsize=(8, 4.5), constrained_layout=True)
    x = np.arange(len(FAMILIES))
    width = 0.28
    for mi, mode in enumerate(MODES):
        vals = [by_family_per_mode[mode][f][0] for f in FAMILIES]
        errs = [by_family_per_mode[mode][f][1] for f in FAMILIES]
        ax.bar(x + (mi - 1) * width, vals, width, yerr=errs, capsize=3, label=mode)
    ax.axhline(0.0, color="gray", linestyle="--", alpha=0.5, linewidth=0.8)
    ax.axhline(1.0, color="gray", linestyle=":", alpha=0.5, linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(FAMILIES)
    ax.set_ylabel("mean position sensitivity")
    ax.set_title("Position sensitivity by architecture family "
                 "(mean over encoders × datasets, error bars = 1 std)")
    ax.legend(fontsize=9)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase8-dir", type=Path, default=Path("experiments/phase8"))
    ap.add_argument("--phase3-dir", type=Path, default=Path("experiments/phase3"))
    ap.add_argument("--phase5-dir", type=Path, default=Path("experiments/phase5"))
    ap.add_argument("--figures-dir", type=Path, default=Path("figures"))
    args = ap.parse_args(argv)

    args.figures_dir.mkdir(parents=True, exist_ok=True)
    args.phase8_dir.mkdir(parents=True, exist_ok=True)

    table = build_sensitivity_table(args.phase8_dir, args.phase3_dir, args.phase5_dir)
    write_stability_json(args.phase8_dir / "position_sensitivity.json", table)

    parquet_path = args.phase8_dir / "query_level_sensitivity.parquet"
    rows, wrote_parquet = build_query_parquet(
        args.phase8_dir, args.phase3_dir, args.phase5_dir, parquet_path,
    )

    figure_position_sensitivity(table, args.figures_dir / "position_sensitivity.png")
    figure_sensitivity_by_family(table, args.figures_dir / "sensitivity_by_family.png")

    # Captions
    (args.figures_dir / "position_sensitivity_caption.md").write_text(
        "# Position sensitivity per encoder × dataset × shuffle mode\n\n"
        "Position sensitivity = `(baseline_nDCG@10 - shuffled_nDCG@10) / baseline_nDCG@10`.\n\n"
        "- **0.0** = shuffle-invariant (shuffling tokens doesn't change retrieval quality).\n"
        "- **1.0** = complete collapse (shuffle reduces nDCG@10 to 0).\n"
        "- **< 0** would indicate shuffle *helped*, treated as a bug signal.\n\n"
        "Three panels, one per BEIR test subset. Three bars per encoder, one per "
        "shuffle mode (docs-only, queries-only, both). Shuffle seed = 1729, identical "
        "across all encoders for cross-encoder fairness. Content tokens are permuted "
        "within each sequence after tokenization but before encoding; special tokens "
        "(`[CLS]`, `[SEP]`, `[PAD]`) stay at their original positions.\n\n"
        "Source: `experiments/phase8/<encoder>__<dataset>__<mode>.json`.\n"
    )
    (args.figures_dir / "sensitivity_by_family_caption.md").write_text(
        "# Position sensitivity by architecture family (aggregated)\n\n"
        "Mean position sensitivity averaged over (encoder, dataset) pairs within each "
        "architecture family, one bar per shuffle mode. Error bars are 1 standard "
        "deviation across the (encoder, dataset) pairs contributing to each family "
        "(transformer family has 3 encoders × 3 datasets = 9 points; each non-transformer "
        "family has 1 encoder × 3 datasets = 3 points).\n\n"
        "This is the headline synthesis for the Phase 9 opinions section: does retrieval "
        "quality depend on sequential position, and if so, how much does it differ across "
        "the four architectural families?\n\n"
        "Source: `experiments/phase8/<encoder>__<dataset>__<mode>.json`.\n"
    )

    print(json.dumps({
        "sensitivity_json": str(args.phase8_dir / "position_sensitivity.json"),
        "query_parquet": str(parquet_path) if wrote_parquet else str(parquet_path.with_suffix(".json")),
        "n_query_rows": len(rows),
        "figures": [
            str(args.figures_dir / "position_sensitivity.png"),
            str(args.figures_dir / "sensitivity_by_family.png"),
        ],
    }, indent=2))
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
