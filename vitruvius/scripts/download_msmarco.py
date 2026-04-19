"""Download MS MARCO passage-retrieval triplets and subsample 500K for Phase 5.

Uses ``sentence-transformers/msmarco`` on HF Hub, which ships the canonical
corpus / queries / triplets split. Triplets contain IDs only, so we build
in-memory lookup tables from the corpus + queries configs and emit
(query_text, positive_text, negative_text) JSONL records.

Subsamples 500K for training + 5K for validation, both seeded (1729).

Usage:
    python scripts/download_msmarco.py [--n-train 500000] [--n-val 5000] [--seed 1729]
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

from vitruvius.utils.logging import get_logger

_log = get_logger("download_msmarco")

DEFAULT_OUT = Path("data/msmarco")
HF_DATASET = "sentence-transformers/msmarco"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n-train", type=int, default=500_000)
    p.add_argument("--n-val", type=int, default=5_000)
    p.add_argument("--seed", type=int, default=1729)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = p.parse_args(argv)

    try:
        from datasets import load_dataset
    except ImportError:
        _log.error("`datasets` package required. pip install datasets")
        return 1

    args.out.mkdir(parents=True, exist_ok=True)

    # ---- Step 1: triplets (IDs only) -----------------------------------
    _log.info("msmarco.load_triplets")
    t0 = time.perf_counter()
    triplets = load_dataset(HF_DATASET, "triplets", split="train")
    _log.info(
        "msmarco.triplets n=%d columns=%s wall_s=%.1f",
        len(triplets), triplets.column_names, time.perf_counter() - t0,
    )

    total_needed = args.n_train + args.n_val
    if total_needed > len(triplets):
        _log.error(
            "msmarco.insufficient_triplets available=%d requested=%d",
            len(triplets), total_needed,
        )
        return 2

    # ---- Step 2: subsample BEFORE resolving text (saves memory) --------
    rng = random.Random(args.seed)
    all_idx = list(range(len(triplets)))
    rng.shuffle(all_idx)
    train_idx = sorted(all_idx[: args.n_train])
    val_idx = sorted(all_idx[args.n_train : args.n_train + args.n_val])

    _log.info("msmarco.gather_ids")
    needed_qids: set = set()
    needed_pids: set = set()
    selected = []  # list of (split, query_id, positive_id, negative_id)
    for i in train_idx:
        r = triplets[i]
        needed_qids.add(r["query_id"])
        needed_pids.add(r["positive_id"])
        needed_pids.add(r["negative_id"])
        selected.append(("train", r["query_id"], r["positive_id"], r["negative_id"]))
    for i in val_idx:
        r = triplets[i]
        needed_qids.add(r["query_id"])
        needed_pids.add(r["positive_id"])
        needed_pids.add(r["negative_id"])
        selected.append(("val", r["query_id"], r["positive_id"], r["negative_id"]))
    _log.info(
        "msmarco.ids n_triplets=%d unique_qids=%d unique_pids=%d",
        len(selected), len(needed_qids), len(needed_pids),
    )

    # ---- Step 3: filter corpus + queries to only the IDs we need -------
    _log.info("msmarco.load_queries_full")
    t0 = time.perf_counter()
    queries_ds = load_dataset(HF_DATASET, "queries", split="train")
    _log.info(
        "msmarco.queries n=%d columns=%s wall_s=%.1f",
        len(queries_ds), queries_ds.column_names, time.perf_counter() - t0,
    )
    t0 = time.perf_counter()
    q_text: dict[int, str] = {}
    for row in queries_ds:
        if row["query_id"] in needed_qids:
            q_text[row["query_id"]] = row["query"]
    _log.info("msmarco.queries_filter resolved=%d wall_s=%.1f",
              len(q_text), time.perf_counter() - t0)

    _log.info("msmarco.load_corpus_full")
    t0 = time.perf_counter()
    corpus_ds = load_dataset(HF_DATASET, "corpus", split="train")
    _log.info(
        "msmarco.corpus n=%d columns=%s wall_s=%.1f",
        len(corpus_ds), corpus_ds.column_names, time.perf_counter() - t0,
    )
    t0 = time.perf_counter()
    p_text: dict[int, str] = {}
    for row in corpus_ds:
        pid = row["passage_id"]
        if pid in needed_pids:
            p_text[pid] = row["passage"]
    _log.info("msmarco.corpus_filter resolved=%d wall_s=%.1f",
              len(p_text), time.perf_counter() - t0)

    missing_q = needed_qids - set(q_text)
    missing_p = needed_pids - set(p_text)
    if missing_q or missing_p:
        _log.error("msmarco.unresolved_ids missing_qids=%d missing_pids=%d",
                   len(missing_q), len(missing_p))
        return 3

    # ---- Step 4: write JSONL -----------------------------------------
    train_path = args.out / f"train_{args.n_train // 1000}k.jsonl"
    val_path = args.out / f"val_{args.n_val // 1000}k.jsonl"

    q_lens: list[int] = []
    p_lens: list[int] = []

    with open(train_path, "w") as f_train, open(val_path, "w") as f_val:
        for split, qid, pid, nid in selected:
            rec = {
                "query": q_text[qid],
                "positive": p_text[pid],
                "negative": p_text[nid],
            }
            line = json.dumps(rec) + "\n"
            if split == "train":
                f_train.write(line)
                q_lens.append(len(rec["query"].split()))
                p_lens.append(len(rec["positive"].split()))
            else:
                f_val.write(line)

    def _percs(xs: list[int]) -> dict[str, int]:
        xs = sorted(xs)
        return {
            "min": xs[0],
            "p50": xs[len(xs) // 2],
            "p90": xs[int(0.9 * (len(xs) - 1))],
            "p99": xs[int(0.99 * (len(xs) - 1))],
            "max": xs[-1],
        }

    summary = {
        "dataset": HF_DATASET,
        "configs_used": ["triplets", "queries", "corpus"],
        "seed": args.seed,
        "n_train": args.n_train,
        "n_val": args.n_val,
        "train_path": str(train_path),
        "val_path": str(val_path),
        "query_whitespace_token_lengths": _percs(q_lens),
        "positive_whitespace_token_lengths": _percs(p_lens),
    }
    (args.out / "download_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
    print(json.dumps(summary, indent=2, sort_keys=True))
    _log.info("msmarco.done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
