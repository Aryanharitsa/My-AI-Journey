"""Generate the Phase 5 centerpiece figure: latency-accuracy Pareto frontier.

Reads:
    experiments/phase3/<encoder>__<dataset>.json  (transformer nDCG@10)
    experiments/phase3_5/<encoder>__<dataset>.json  (transformer batch-1 latency)
    experiments/phase5/<encoder>__<dataset>.json  (from-scratch encoder bench + latency)

Writes:
    figures/pareto_v2.png
    figures/pareto_v2.pdf
    figures/pareto_v2_caption.md

X-axis: query encoding latency at batch 1 (median ms), averaged across the
three BEIR test subsets.
Y-axis: nDCG@10 (from-scratch retrieval_metrics.evaluate), averaged across
the three subsets.

Colors by architecture family:
    transformer (minilm, bert, gte)     blue family
    recurrent   (lstm-retriever)        orange
    convolutional (conv-retriever)      green
    SSM/Mamba   (mamba-retriever-fs)    red
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ENCODERS = [
    # (registry_name, family, color, marker)
    ("minilm-l6-v2",      "transformer", "#1f77b4", "o"),
    ("bert-base",         "transformer", "#2a6aa8", "o"),
    ("gte-small",         "transformer", "#5fa6d6", "o"),
    ("lstm-retriever",    "recurrent",   "#ff7f0e", "s"),
    ("conv-retriever",    "convolutional", "#2ca02c", "^"),
    ("mamba-retriever-fs", "ssm",        "#d62728", "D"),
]
DATASETS = ["nfcorpus", "scifact", "fiqa"]


def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else float("nan")


def load_bench(encoder: str, bench_dirs: list[Path]) -> tuple[float, list[float]]:
    """Return mean nDCG@10 + per-dataset list across DATASETS. NaN if any missing."""
    per_ds: list[float] = []
    for ds in DATASETS:
        found = None
        for root in bench_dirs:
            p = root / f"{encoder}__{ds}.json"
            if p.exists():
                d = json.loads(p.read_text())
                found = float(d["metrics"]["ours_from_scratch"]["nDCG@10"])
                break
        if found is None:
            per_ds.append(float("nan"))
        else:
            per_ds.append(found)
    return _mean([x for x in per_ds if x == x]), per_ds


def load_latency(encoder: str, latency_dirs: list[Path]) -> tuple[float, list[float]]:
    """Return mean batch-1 median latency (ms) + per-dataset list."""
    per_ds: list[float] = []
    for ds in DATASETS:
        found = None
        for root in latency_dirs:
            p = root / f"{encoder}__{ds}.json"
            if p.exists():
                d = json.loads(p.read_text())
                # query_latency_ms is keyed by batch size as str
                lat = d.get("query_latency_ms", {}).get("1")
                if lat and "median" in lat:
                    found = float(lat["median"])
                    break
        if found is None:
            per_ds.append(float("nan"))
        else:
            per_ds.append(found)
    return _mean([x for x in per_ds if x == x]), per_ds


def pareto_frontier(points: list[tuple[str, float, float]]) -> set[str]:
    """Return names that are Pareto-optimal (low latency, high nDCG).

    A point is Pareto-optimal if no other point dominates it: i.e., no other
    point has both latency <= and nDCG >= (with at least one strict).
    """
    frontier: set[str] = set()
    for name, lat, ndcg in points:
        dominated = False
        for name2, lat2, ndcg2 in points:
            if name2 == name:
                continue
            if lat2 <= lat and ndcg2 >= ndcg and (lat2 < lat or ndcg2 > ndcg):
                dominated = True
                break
        if not dominated:
            frontier.add(name)
    return frontier


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase3-bench-dir", type=Path, default=Path("experiments/phase3"))
    ap.add_argument("--phase3-latency-dir", type=Path, default=Path("experiments/phase3_5"))
    ap.add_argument("--phase5-dir", type=Path, default=Path("experiments/phase5"))
    ap.add_argument("--out-dir", type=Path, default=Path("figures"))
    args = ap.parse_args(argv)

    args.out_dir.mkdir(parents=True, exist_ok=True)

    bench_dirs = [args.phase5_dir, args.phase3_bench_dir]
    latency_dirs = [args.phase5_dir, args.phase3_latency_dir]

    points: list[tuple[str, str, str, str, float, float, list[float], list[float]]] = []
    for enc, fam, color, marker in ENCODERS:
        ndcg_mean, ndcg_per = load_bench(enc, bench_dirs)
        lat_mean, lat_per = load_latency(enc, latency_dirs)
        if ndcg_mean != ndcg_mean or lat_mean != lat_mean:
            print(f"skip {enc}: missing data (ndcg={ndcg_mean} lat={lat_mean})",
                  file=sys.stderr)
            continue
        points.append((enc, fam, color, marker, lat_mean, ndcg_mean, ndcg_per, lat_per))

    if not points:
        print("ERROR: no points to plot", file=sys.stderr)
        return 1

    # Decide log scale
    lats = [p[4] for p in points]
    dynamic_range = max(lats) / min(lats)
    use_log = dynamic_range > 10.0

    # Pareto
    simple_points = [(p[0], p[4], p[5]) for p in points]
    frontier = pareto_frontier(simple_points)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 6))
    for enc, fam, color, marker, lat, ndcg, _, _ in points:
        is_frontier = enc in frontier
        ax.scatter(
            lat, ndcg, s=200 if is_frontier else 120,
            c=color, marker=marker,
            edgecolor="black" if is_frontier else "none",
            linewidth=2 if is_frontier else 0,
            zorder=3,
            label=f"{enc} ({fam})",
        )
        # Annotation with small offset
        dx, dy = 1.03, 1.008
        ax.annotate(enc, (lat * dx if use_log else lat + 0.3, ndcg * dy),
                    fontsize=9, color="black")

    # Draw Pareto frontier line (sorted by latency)
    frontier_points = sorted(
        [(lat, ndcg) for enc, _, _, _, lat, ndcg, _, _ in points if enc in frontier],
        key=lambda t: t[0],
    )
    if len(frontier_points) >= 2:
        xs = [p[0] for p in frontier_points]
        ys = [p[1] for p in frontier_points]
        ax.plot(xs, ys, "--", color="gray", alpha=0.5, zorder=2, label="Pareto frontier")

    if use_log:
        ax.set_xscale("log")
    ax.set_xlabel("Query encoding latency @ batch 1 (median ms, avg over 3 BEIR subsets)")
    ax.set_ylabel("nDCG@10 (avg over NFCorpus / SciFact / FiQA)")
    ax.set_title("Phase 5 — Encoder latency-accuracy Pareto frontier")
    ax.grid(True, alpha=0.3, zorder=1)
    ax.legend(loc="best", fontsize=8, framealpha=0.9)

    png = args.out_dir / "pareto_v2.png"
    pdf = args.out_dir / "pareto_v2.pdf"
    plt.tight_layout()
    plt.savefig(png, dpi=180)
    plt.savefig(pdf)
    plt.close(fig)

    # Caption
    caption = args.out_dir / "pareto_v2_caption.md"
    pareto_members = sorted(frontier)
    lines = [
        "# Pareto v2 — caption",
        "",
        "**X-axis:** query encoding latency at batch size 1, median ms measured "
        "via `torch.cuda.Event` (10 warmup + 100 measured passes per dataset), "
        "averaged across NFCorpus / SciFact / FiQA test splits (200 sampled "
        "queries each). "
        + ("Log scale (dynamic range > 10×)." if use_log else "Linear scale.") + "",
        "",
        "**Y-axis:** nDCG@10 using the from-scratch `retrieval_metrics.evaluate` "
        "(graded-gain, `2^rel - 1` / `log2(i+1)`, iDCG over full qrels), "
        "averaged across the same three test splits. pytrec_eval cross-check "
        "was run but not plotted.",
        "",
        "**Points** ({n}):".format(n=len(points)),
    ]
    for enc, fam, _, _, lat, ndcg, ndcg_per, lat_per in points:
        star = " ⋆ Pareto-optimal" if enc in frontier else ""
        lines.append(
            f"- `{enc}` ({fam}): latency {lat:.2f} ms, nDCG@10 {ndcg:.4f}{star}. "
            f"Per-dataset nDCG {ndcg_per}. Per-dataset latency {lat_per}."
        )
    lines += [
        "",
        f"**Pareto-optimal subset**: {', '.join(f'`{n}`' for n in pareto_members)}.",
        "",
        "**What this plot shows:** The latency-accuracy frontier across "
        "transformer (blue family), recurrent BiLSTM (orange), 1D-CNN (green), "
        "and SSM/Mamba2 (red) architectures, all evaluated via the same "
        "FAISS `IndexFlatIP` retrieval harness. Transformer encoders are "
        "pre-trained checkpoints (MiniLM, MSMARCO-BERT-dot, GTE-small); the "
        "non-transformer encoders are trained from scratch on the same 500K "
        "MS MARCO triplets with identical hyperparameters (InfoNCE τ=0.05, "
        "AdamW lr=1e-4, 3 epochs, batch 64).",
        "",
        "**What this plot does NOT show:** (1) throughput at batch 32 — see "
        "`experiments/phase5/SUMMARY.md` and `experiments/phase3_5/SUMMARY.md`. "
        "(2) per-query failure modes — that's Phase 6. (3) comparison at equal "
        "FLOPs or equal parameter counts — the three from-scratch encoders "
        "have ~5-25M parameters vs. ~22-110M for the pre-trained transformers, "
        "so they're not parameter-matched and any frontier claim is about "
        "*this specific training budget*, not about architecture classes in "
        "general. (4) out-of-distribution robustness — all cells use BEIR "
        "test splits that overlap MS MARCO's domain.",
        "",
        "Source JSONs: `experiments/phase3/`, `experiments/phase3_5/`, "
        "`experiments/phase5/`.",
    ]
    caption.write_text("\n".join(lines) + "\n")

    print(json.dumps({
        "png": str(png),
        "pdf": str(pdf),
        "caption": str(caption),
        "n_points": len(points),
        "pareto_optimal": sorted(frontier),
        "log_x": use_log,
    }, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
