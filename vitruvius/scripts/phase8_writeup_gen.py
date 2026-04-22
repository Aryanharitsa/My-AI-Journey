"""Generate experiments/phase8/{SUMMARY.md, README.md} from cell JSONs + sensitivity table."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

ENCODERS = [
    ("minilm-l6-v2",     "transformer"),
    ("bert-base",        "transformer"),
    ("gte-small",        "transformer"),
    ("lstm-retriever",   "recurrent"),
    ("conv-retriever",   "convolutional"),
    ("mamba-retriever-fs", "ssm"),
]
DATASETS = ["nfcorpus", "scifact", "fiqa"]
MODES = ["docs-shuffled", "queries-shuffled", "both-shuffled"]
FAMILIES = ["transformer", "recurrent", "convolutional", "ssm"]


def _fmt(x):
    if x is None:
        return "—"
    return f"{x:.3f}"


def write_summary(phase8_dir: Path, out_path: Path):
    sens = json.loads((phase8_dir / "position_sensitivity.json").read_text())
    table = {}
    for row in sens["rows"]:
        key = (row["encoder"], row["dataset"], row["mode"])
        table[key] = row.get("position_sensitivity")

    # Family averages per mode (over encoder × dataset cells)
    fam_avgs = {}
    for mode in MODES:
        fam_avgs[mode] = {}
        for fam in FAMILIES:
            vals = []
            for enc, f in ENCODERS:
                if f != fam:
                    continue
                for ds in DATASETS:
                    v = table.get((enc, ds, mode))
                    if v is not None:
                        vals.append(v)
            fam_avgs[mode][fam] = float(np.mean(vals)) if vals else None

    lines = [
        "# Phase 8 — Token-position shuffle & position sensitivity (90% milestone)",
        "",
        "What this phase measures: for each of the 6 Pareto encoders × 3 BEIR test subsets × 3 shuffle modes,",
        "re-encode docs and/or queries after permuting content-token positions (special tokens stay put).",
        "Position sensitivity score = `(baseline_nDCG@10 − shuffled_nDCG@10) / baseline_nDCG@10`.",
        "Higher = more sequentially-position-dependent; 0.0 = bag-of-concepts; 1.0 = full collapse.",
        "",
        "Baselines are pulled from Phase 3 (transformers) and Phase 5 (from-scratch) — not re-run.",
        "Shuffle seed 1729 is identical across all encoders for cross-encoder fairness.",
        "",
        "## Headline — mean position sensitivity by architecture family (over 3 datasets × encoders in family)",
        "",
        "| Family | docs-shuffled | queries-shuffled | both-shuffled |",
        "|---|---:|---:|---:|",
    ]
    for fam in FAMILIES:
        d = _fmt(fam_avgs["docs-shuffled"][fam])
        q = _fmt(fam_avgs["queries-shuffled"][fam])
        b = _fmt(fam_avgs["both-shuffled"][fam])
        lines.append(f"| {fam} | {d} | {q} | {b} |")

    lines += [
        "",
        "## Full table — position sensitivity per (encoder × dataset × mode)",
        "",
        "Each cell is `(baseline − shuffled) / baseline` at nDCG@10.",
        "",
        "### docs-shuffled",
        "",
        "| Encoder \\ Dataset | " + " | ".join(DATASETS) + " |",
        "|---|" + "|".join([":---:"] * len(DATASETS)) + "|",
    ]
    for enc, _ in ENCODERS:
        row = [f"`{enc}`"]
        for ds in DATASETS:
            row.append(_fmt(table.get((enc, ds, "docs-shuffled"))))
        lines.append("| " + " | ".join(row) + " |")

    lines += [
        "",
        "### queries-shuffled",
        "",
        "| Encoder \\ Dataset | " + " | ".join(DATASETS) + " |",
        "|---|" + "|".join([":---:"] * len(DATASETS)) + "|",
    ]
    for enc, _ in ENCODERS:
        row = [f"`{enc}`"]
        for ds in DATASETS:
            row.append(_fmt(table.get((enc, ds, "queries-shuffled"))))
        lines.append("| " + " | ".join(row) + " |")

    lines += [
        "",
        "### both-shuffled",
        "",
        "| Encoder \\ Dataset | " + " | ".join(DATASETS) + " |",
        "|---|" + "|".join([":---:"] * len(DATASETS)) + "|",
    ]
    for enc, _ in ENCODERS:
        row = [f"`{enc}`"]
        for ds in DATASETS:
            row.append(_fmt(table.get((enc, ds, "both-shuffled"))))
        lines.append("| " + " | ".join(row) + " |")

    lines += [
        "",
        "## Per-cell baseline vs shuffled (nDCG@10)",
        "",
        "Sanity audit: every row shows the raw nDCG@10 that produced the sensitivity number.",
        "",
        "| Encoder × Dataset | mode | baseline | shuffled | sensitivity |",
        "|---|---|---:|---:|---:|",
    ]
    for row in sens["rows"]:
        if row.get("position_sensitivity") is None:
            continue
        lines.append(
            f"| `{row['encoder']}` × `{row['dataset']}` | {row['mode']} | "
            f"{row['baseline_nDCG@10']:.4f} | {row['shuffled_nDCG@10']:.4f} | "
            f"{row['position_sensitivity']:.3f} |"
        )

    lines += [
        "",
        "## Cross-phase synthesis — query-level shuffle damage",
        "",
        "Per-query baseline vs shuffled nDCG@10 and hit@10 are preserved in",
        "`experiments/phase8/query_level_sensitivity.parquet` (or `.json` fallback).",
        "Phase 6 (failure taxonomy) cross-referencing is out of scope for this",
        "pod session and can be layered in offline using the preserved per-query",
        "records — `LEN-LONG` queries should be more position-sensitive than",
        "`LEN-SHORT`; `PARAPHRASE` failures should be position-insensitive under",
        "the bag-of-concepts hypothesis.",
        "",
        "## Methodology",
        "",
        "- Shuffle operates on `input_ids` AFTER tokenization and BEFORE encoding.",
        "- Special tokens (`[CLS]`, `[SEP]`, `[PAD]`, any tokenizer-special ID) are",
        "  pinned at their original positions. Padding is never touched.",
        "- For a given (dataset, mode, sample), the shuffle permutation is the",
        "  same regardless of which encoder is evaluating — this ensures the",
        "  comparison across encoders is apples-to-apples and not confounded by",
        "  shuffle-noise.",
        "- Pre-trained transformer encoders use the default `sdpa` attention",
        "  (faster than `eager` by ~1.7×). `head_mask` is not used in Phase 8",
        "  so the Phase 7 `attn_implementation=\"eager\"` requirement does not apply.",
        "- From-scratch encoders load checkpoints from `models/<encoder>/best.pt`",
        "  (trained in Session 03 / Phase 5).",
        "- Graded-gain nDCG@10 from the from-scratch `retrieval_metrics.evaluate`",
        "  is the primary metric; `pytrec_eval` runs as a cross-check.",
        "",
        "## Limitations",
        "",
        "- **Uniform random shuffle**. A local-shuffle variant (swap adjacent",
        "  tokens only) would separate local-order sensitivity from global-order",
        "  sensitivity; flagged as future work.",
        "- **Fixed seed (1729)**. Averaging over multiple shuffle seeds would",
        "  produce error bars but triples compute. Out of scope for Session 06.",
        "- **Short queries are degenerate**. Queries with ≤2 content tokens",
        "  cannot be meaningfully permuted. The `nfcorpus` median query length",
        "  is documented in `experiments/phase3_5/dataset_length_stats.json`",
        "  (computed in Phase 3.5).",
        "- **No fine-tuning on shuffled data**. This measures how much already-",
        "  trained encoders rely on position. \"Can you train position-invariant",
        "  encoders?\" is a separate research question.",
        "",
    ]
    out_path.write_text("\n".join(lines) + "\n")


def write_readme(out_path: Path):
    out_path.write_text(
        "# Phase 8 — token-position shuffle artifacts\n\n"
        "## Files\n\n"
        "- `<encoder>__<dataset>__<mode>.json` (54 files)\n"
        "    One per (encoder, dataset, shuffle mode). Full metric block,\n"
        "    per-query results, hardware, config, and the shuffle flags.\n"
        "- `position_sensitivity.json`\n"
        "    Machine-readable aggregated `(baseline − shuffled) / baseline`\n"
        "    table. 54 rows.\n"
        "- `query_level_sensitivity.parquet` (or `.json` if parquet unavailable)\n"
        "    Per-query baseline vs shuffled nDCG@10 and hit@10. Used for\n"
        "    Phase 6 cross-referencing offline.\n"
        "- `SUMMARY.md` — headline family table + full 18-row pivot +\n"
        "    methodology + limitations.\n"
        "- `README.md` — this file.\n\n"
        "## Reproduce\n\n"
        "```bash\n"
        "python scripts/shuffle_sweep.py \\\n"
        "    --encoders minilm-l6-v2 bert-base gte-small lstm-retriever conv-retriever mamba-retriever-fs \\\n"
        "    --datasets nfcorpus scifact fiqa \\\n"
        "    --modes docs-shuffled queries-shuffled both-shuffled \\\n"
        "    --split test --batch-size 128 --top-k 100 --device cuda --seed 1729\n"
        "\n"
        "python scripts/phase8_analysis.py\n"
        "python scripts/phase8_writeup_gen.py\n"
        "```\n"
    )


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase8-dir", type=Path, default=Path("experiments/phase8"))
    ap.add_argument("--out-summary", type=Path, default=Path("experiments/phase8/SUMMARY.md"))
    ap.add_argument("--out-readme", type=Path, default=Path("experiments/phase8/README.md"))
    args = ap.parse_args(argv)
    write_summary(args.phase8_dir, args.out_summary)
    write_readme(args.out_readme)
    print(json.dumps({"summary": str(args.out_summary), "readme": str(args.out_readme)}))
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
