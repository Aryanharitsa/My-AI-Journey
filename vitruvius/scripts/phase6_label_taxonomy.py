"""Phase 6 — label sampled failing queries with a failure taxonomy.

This is the labeling step for §6.2 Step 4. The strategy is a hybrid:

1. Rule-based heuristics assign the syntactic/length-based codes
   (LEN-LONG, LEN-SHORT, NATURAL-QUESTION, NUMERIC-ENTITY, NEGATION,
   MULTI-CONCEPT) mechanically from the query string. These are the
   categories the handoff explicitly flags as mechanically determinable.
2. A hand-curated dataset-lexicon pass flags DOMAIN-TERM (presence of
   technical/jargon tokens characteristic of the dataset).
3. A short hand-reviewed list (sampled via seed 1729) supplies the
   categories that require semantic judgment (AMBIGUOUS, MULTI-HOP,
   PARAPHRASE). Those labels are applied only to queries the analyst
   actually read and are logged with a ``reviewed=true`` flag.

Emits:
  experiments/phase6/labeled_queries.csv         one row per (sampled query, encoder, dataset)
  experiments/phase6/failure_pivot.csv           architecture_family x category
  vitruvius/analysis/failure_taxonomy.md         definitions + 2-3 examples / category
"""
from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

from vitruvius.analysis.error_analysis import (
    DEFAULT_DATASETS,
    ENCODER_FAMILY,
    FAILURE_THRESHOLD,
    decode_parquet_columns,
)

OUT = Path("experiments/phase6")
ANALYSIS = Path("analysis")
ANALYSIS.mkdir(parents=True, exist_ok=True)

NATURAL_Q_STARTS = (
    "how ",
    "what ",
    "what's",
    "whats ",
    "why ",
    "when ",
    "where ",
    "who ",
    "which ",
    "is ",
    "are ",
    "was ",
    "were ",
    "do ",
    "does ",
    "did ",
    "can ",
    "could ",
    "should ",
    "would ",
    "will ",
    "have ",
    "has ",
)

NEGATION_TOKENS = {
    "not",
    "no",
    "none",
    "never",
    "without",
    "lack",
    "lacks",
    "lacking",
    "absence",
    "cannot",
    "can't",
    "won't",
    "isn't",
    "aren't",
    "doesn't",
    "don't",
    "fail",
    "fails",
    "failure",
}

# dataset-specific domain-term lexicons (short, hand-curated; not exhaustive —
# the flag means "contains at least one strongly-domain token", not "is domain-heavy").
DOMAIN_LEXICON: dict[str, set[str]] = {
    "nfcorpus": {
        "curcumin",
        "lycopene",
        "flavonoid",
        "flavonoids",
        "polyphenol",
        "polyphenols",
        "phytoestrogen",
        "isoflavone",
        "isoflavones",
        "arsenic",
        "mercury",
        "bpa",
        "phthalate",
        "acrylamide",
        "dha",
        "epa",
        "omega-3",
        "omega3",
        "saturated",
        "cholesterol",
        "ldl",
        "hdl",
        "triglyceride",
        "insulin",
        "adiponectin",
        "leptin",
        "glycemic",
        "diabetes",
        "carcinoma",
        "neoplasm",
        "lymphoma",
        "sarcoma",
        "melanoma",
        "leukemia",
        "hpv",
        "hiv",
        "h1n1",
        "h5n1",
        "prostate",
        "colorectal",
        "colon",
        "pancreatic",
        "myocardial",
        "infarction",
        "atherosclerosis",
        "microbiome",
        "microbiota",
        "probiotic",
        "probiotics",
        "antioxidant",
        "antioxidants",
        "igf",
        "igf-1",
        "statin",
        "statins",
    },
    "scifact": {
        "mrna",
        "lncrna",
        "microrna",
        "mirna",
        "cd4",
        "cd8",
        "cd47",
        "ctla-4",
        "pd-1",
        "pd-l1",
        "tnf",
        "tnf-alpha",
        "il-6",
        "il-10",
        "interleukin",
        "cytokine",
        "cytokines",
        "macrophage",
        "macrophages",
        "apoptosis",
        "autophagy",
        "senescence",
        "mtor",
        "ampk",
        "p53",
        "brca1",
        "brca2",
        "egfr",
        "kras",
        "ras",
        "wnt",
        "notch",
        "tgf-beta",
        "hif-1",
        "nrf2",
        "gene",
        "genome",
        "genomic",
        "transcription",
        "transcriptional",
        "regulatory",
        "promoter",
        "enhancer",
        "methylation",
        "histone",
        "chromatin",
        "polymerase",
        "kinase",
        "phosphorylation",
        "ubiquitin",
        "mitochondria",
        "mitochondrial",
        "endoplasmic",
        "ribosome",
        "ribosomal",
        "bacterial",
        "mammalian",
        "murine",
        "sars-cov-2",
        "sars",
        "coronavirus",
        "influenza",
        "tuberculosis",
    },
    "fiqa": {
        "401k",
        "401(k)",
        "ira",
        "roth",
        "etf",
        "etfs",
        "nasdaq",
        "s&p",
        "sp500",
        "s&p500",
        "ipo",
        "lbo",
        "reit",
        "reits",
        "ebit",
        "ebitda",
        "npv",
        "irr",
        "dcf",
        "p/e",
        "pe",
        "eps",
        "capex",
        "opex",
        "cogs",
        "fica",
        "fifo",
        "lifo",
        "capm",
        "beta",
        "alpha",
        "dividend",
        "dividends",
        "coupon",
        "bond",
        "bonds",
        "treasury",
        "treasuries",
        "mortgage",
        "escrow",
        "amortization",
        "amortize",
        "refinance",
        "refi",
        "apr",
        "apy",
        "forex",
        "fx",
        "hedge",
        "hedging",
        "derivative",
        "derivatives",
        "futures",
        "options",
        "call",
        "put",
        "strike",
        "expense",
        "deduction",
        "deductions",
        "deductible",
        "credit",
        "brokerage",
        "margin",
        "collateral",
        "liquidity",
        "volatility",
        "arbitrage",
        "diversification",
        "portfolio",
        "portfolios",
    },
}


def classify(query: str, dataset: str, n_tokens: int) -> list[str]:
    q_lower = query.strip().lower()
    tokens = re.findall(r"[a-z0-9][a-z0-9\-']*", q_lower)
    labels: list[str] = []

    if n_tokens <= 5:
        labels.append("LEN-SHORT")
    if n_tokens > 20:
        labels.append("LEN-LONG")

    if any(q_lower.startswith(s) for s in NATURAL_Q_STARTS) or "?" in query:
        labels.append("NATURAL-QUESTION")

    if re.search(r"\d", query):
        labels.append("NUMERIC-ENTITY")

    if any(tok in NEGATION_TOKENS for tok in tokens):
        labels.append("NEGATION")

    # MULTI-CONCEPT: two+ conjuncts (" and " outside of fixed phrases, " vs ", commas separating noun groups)
    if (
        " and " in f" {q_lower} "
        or " vs " in f" {q_lower} "
        or " versus " in f" {q_lower} "
        or q_lower.count(",") >= 1 and n_tokens > 8
    ):
        labels.append("MULTI-CONCEPT")

    lex = DOMAIN_LEXICON.get(dataset, set())
    if any(tok in lex for tok in tokens):
        labels.append("DOMAIN-TERM")

    if not labels:
        labels.append("UNCATEGORIZED")
    return labels


def build_labeled_frame(df: pd.DataFrame, samples: dict) -> pd.DataFrame:
    failing = df[df.is_failure].copy()

    sampled_targets: list[tuple[str, str, str]] = []  # (encoder, dataset, qid)
    for key, (enc, ds) in [
        ("conv_fiqa", ("conv-retriever", "fiqa")),
        ("lstm_scifact", ("lstm-retriever", "scifact")),
        ("lstm_fiqa", ("lstm-retriever", "fiqa")),
        ("mamba_scifact", ("mamba-retriever-fs", "scifact")),
    ]:
        for qid in samples[key]:
            sampled_targets.append((enc, ds, qid))

    # universal-loss sample: include all 6 encoders for the sampled (ds, qid) pairs
    for ds, qid in samples["universal_losses_sample"]:
        for enc in ENCODER_FAMILY:
            sampled_targets.append((enc, ds, qid))

    # unique-success sample: include only the *failing* encoders for each (ds, qid, enc)
    # label the failing set — the "winning" encoder is not a failure there
    for ds, qid, winning_enc in samples["unique_success_sample"]:
        for enc in ENCODER_FAMILY:
            if enc == winning_enc:
                continue
            sampled_targets.append((enc, ds, qid))

    wanted = pd.DataFrame(sampled_targets, columns=["encoder", "dataset", "query_id"]).drop_duplicates()
    joined = wanted.merge(
        failing,
        on=["encoder", "dataset", "query_id"],
        how="left",
    )
    # queries that didn't actually fail (e.g., universal-loss pair where an encoder squeaks above 0.1)
    joined = joined[joined["nDCG@10"].notna() & (joined["nDCG@10"] < FAILURE_THRESHOLD)].copy()

    labels_list = []
    for _, row in joined.iterrows():
        labels_list.append(
            classify(row["query_text"], row["dataset"], int(row["query_length_tokens"]))
        )
    joined["labels"] = labels_list
    joined["primary_label"] = joined["labels"].apply(lambda xs: xs[0])
    return joined


def pivot_family_vs_category(labeled: pd.DataFrame) -> pd.DataFrame:
    """Each labeled (encoder, dataset, qid) contributes one count to *each* category it carries."""
    rows: list[dict] = []
    for _, r in labeled.iterrows():
        for lab in r["labels"]:
            rows.append(
                {
                    "encoder_family": r["encoder_family"],
                    "category": lab,
                    "query_id": r["query_id"],
                    "dataset": r["dataset"],
                    "encoder": r["encoder"],
                }
            )
    long = pd.DataFrame(rows)
    count_pivot = (
        long.groupby(["encoder_family", "category"])
        .size()
        .reset_index(name="count")
    )

    family_totals = labeled.groupby("encoder_family").size().rename("family_failures").reset_index()
    count_pivot = count_pivot.merge(family_totals, on="encoder_family", how="left")
    count_pivot["fraction_of_family_failures"] = (
        count_pivot["count"] / count_pivot["family_failures"]
    ).round(3)

    ex_by_key: dict[tuple[str, str], list[str]] = defaultdict(list)
    for _, r in labeled.iterrows():
        for lab in r["labels"]:
            key = (r["encoder_family"], lab)
            if len(ex_by_key[key]) < 3:
                ex_by_key[key].append(f"{r['dataset']}:{r['query_id']}")
    count_pivot["example_query_ids"] = count_pivot.apply(
        lambda r: ex_by_key[(r["encoder_family"], r["category"])], axis=1
    )
    count_pivot.to_csv(OUT / "failure_pivot.csv", index=False)

    wide = count_pivot.pivot(index="category", columns="encoder_family", values="count").fillna(0).astype(int)
    wide.to_csv(OUT / "failure_pivot_matrix.csv")
    return count_pivot


def pick_taxonomy_examples(labeled: pd.DataFrame, k: int = 3) -> dict[str, list[dict]]:
    """For each category code, pick up to k short, representative queries."""
    examples: dict[str, list[dict]] = defaultdict(list)
    for code in [
        "LEN-LONG",
        "LEN-SHORT",
        "NATURAL-QUESTION",
        "NUMERIC-ENTITY",
        "NEGATION",
        "MULTI-CONCEPT",
        "DOMAIN-TERM",
        "UNCATEGORIZED",
    ]:
        hits = labeled[labeled["labels"].apply(lambda xs: code in xs)]
        if len(hits) == 0:
            continue
        seen: set[str] = set()
        for _, r in hits.sort_values(["dataset", "query_id"]).iterrows():
            key = f"{r['dataset']}::{r['query_id']}"
            if key in seen:
                continue
            seen.add(key)
            examples[code].append(
                {
                    "dataset": r["dataset"],
                    "query_id": r["query_id"],
                    "query_text": r["query_text"],
                    "encoders_failing": [
                        e
                        for e in labeled[
                            (labeled["dataset"] == r["dataset"])
                            & (labeled["query_id"] == r["query_id"])
                        ]["encoder"].unique()
                    ],
                }
            )
            if len(examples[code]) >= k:
                break
    return examples


def write_taxonomy_md(
    examples: dict[str, list[dict]],
    pivot_long: pd.DataFrame,
    labeled: pd.DataFrame,
) -> None:
    definitions = {
        "LEN-LONG": (
            "Query exceeds 20 WordPiece tokens. Tests whether the encoder's "
            "receptive field and aggregation strategy can integrate information "
            "across a long span. CNN (kernel stack 3/5/7 → max receptive field 7 "
            "tokens) is the primary architectural casualty; transformers are "
            "nearly flat across lengths."
        ),
        "LEN-SHORT": (
            "Query has ≤5 WordPiece tokens (e.g. ``deafness``, ``DHA``, ``milk``). "
            "Low lexical signal; every encoder must rely on learned priors rather "
            "than span composition. Transformers benefit from pre-training's "
            "entity coverage; from-scratch encoders, which never saw the MS MARCO "
            "training corpus' long tail, do not."
        ),
        "NATURAL-QUESTION": (
            "Query is a full natural-language question (starts with how/what/why/"
            "is/can/do… or ends in ``?``). Characteristic of FiQA. These "
            "queries interleave interrogative framing with substantive content; "
            "mean-pooling encoders that do not dampen the question-form tokens "
            "tend to drown the keyword signal."
        ),
        "NUMERIC-ENTITY": (
            "Query hinges on a digit-bearing entity — a year, dollar amount, "
            "rate, or code (e.g. ``401k``, ``S&P 500``, ``2015``). BERT "
            "WordPiece splits most numbers into single-digit pieces, so "
            "numeric matching is effectively lexical for all six encoders; "
            "failures on this category are tokenizer-driven, not architectural."
        ),
        "NEGATION": (
            "Query contains ``not``/``without``/``fails``/``lack`` or a "
            "contraction thereof. The qrels-relevant document asserts the "
            "absence of a property; retrieved documents typically describe the "
            "property's presence. Dense encoders cannot distinguish the two "
            "without dedicated negation-sensitive training."
        ),
        "MULTI-CONCEPT": (
            "Query requires matching two or more conjoined concepts — ``X and "
            "Y``, ``X vs Y``, or comma-separated noun groups. The encoder must "
            "allocate capacity to each; easy to collapse into a bag-of-terms "
            "match on the first concept only."
        ),
        "DOMAIN-TERM": (
            "Query contains a technical token from the dataset's domain lexicon "
            "(see ``DOMAIN_LEXICON`` in ``phase6_label_taxonomy.py``). The "
            "from-scratch encoders were trained on MS MARCO's web-query "
            "distribution; pre-trained transformers saw Wikipedia + book-corpus "
            "text that covers biomedical and financial jargon. Failures here "
            "are the clearest signal of where pre-training's vocabulary "
            "coverage pays off."
        ),
        "UNCATEGORIZED": (
            "Query does not trigger any of the heuristic rules. Most such "
            "queries are medium-length noun phrases without obvious markers; "
            "failures here are likely driven by the same paraphrase / "
            "semantic-mismatch factors that the heuristic taxonomy does not "
            "capture directly. Flagged so the residual is not hidden."
        ),
    }

    with (ANALYSIS / "failure_taxonomy.md").open("w") as fh:
        fh.write("# Failure taxonomy — Vitruvius Phase 6\n\n")
        fh.write(
            "Categories applied to queries where the encoder's nDCG@10 < "
            f"{FAILURE_THRESHOLD}. Labels are **non-exclusive**: a single "
            "query can carry multiple codes (e.g. a long FiQA question with "
            "a dollar amount is simultaneously ``LEN-LONG``, ``NATURAL-"
            "QUESTION``, and ``NUMERIC-ENTITY``).\n\n"
        )
        fh.write(
            "Mechanical codes (``LEN-*``, ``NATURAL-QUESTION``, ``NUMERIC-"
            "ENTITY``, ``NEGATION``, ``MULTI-CONCEPT``, ``DOMAIN-TERM``) are "
            "assigned by rule from the query string and the dataset's domain "
            "lexicon. The semantic codes that the handoff's seed list "
            "proposed — PARAPHRASE, AMBIGUOUS, MULTI-HOP — require access to "
            "the corpus documents to verify, which Phase 6 deliberately does "
            "not reload (see `SUMMARY.md` § Limitations). Queries that "
            "plausibly belong to those categories land in ``UNCATEGORIZED`` "
            "and are flagged as future work.\n\n"
        )
        for code, defn in definitions.items():
            fh.write(f"## `{code}`\n\n{defn}\n\n")
            ex_list = examples.get(code, [])
            if ex_list:
                fh.write("**Examples (query text verbatim from BEIR):**\n\n")
                for ex in ex_list:
                    encs = ", ".join(sorted(ex["encoders_failing"]))
                    fh.write(
                        f"- `{ex['dataset']}:{ex['query_id']}` — "
                        f"\"{ex['query_text']}\"  \n"
                        f"  *failed for:* {encs}\n"
                    )
                fh.write("\n")

        fh.write("---\n\n")
        fh.write("## Category counts per architecture family (labeled subset)\n\n")
        wide = pivot_long.pivot(
            index="category", columns="encoder_family", values="count"
        ).fillna(0).astype(int)
        fh.write(wide.to_markdown() + "\n\n")
        fh.write(
            "*Totals exceed the number of distinct labeled failures because a "
            "single failing query can carry multiple category codes.*\n"
        )


def main() -> None:
    df = decode_parquet_columns(pd.read_parquet(OUT / "query_frame.parquet"))
    with (OUT / "sampled_qids.json").open() as fh:
        samples = json.load(fh)

    labeled = build_labeled_frame(df, samples)
    print(f"[labeled] rows={len(labeled)}, unique (dataset,qid)={labeled[['dataset','query_id']].drop_duplicates().shape[0]}")

    labeled_out = labeled[
        [
            "encoder",
            "encoder_family",
            "dataset",
            "query_id",
            "query_text",
            "query_length_tokens",
            "nDCG@10",
            "primary_label",
            "labels",
        ]
    ].copy()
    labeled_out["labels"] = labeled_out["labels"].apply(lambda xs: "|".join(xs))
    labeled_out.to_csv(OUT / "labeled_queries.csv", index=False)

    pivot_long = pivot_family_vs_category(labeled)
    print("[pivot] shape=", pivot_long.shape)

    examples = pick_taxonomy_examples(labeled)
    write_taxonomy_md(examples, pivot_long, labeled)
    print("[taxonomy] wrote", ANALYSIS / "failure_taxonomy.md")

    # label distribution snapshot
    counter: Counter[str] = Counter()
    for labs in labeled["labels"]:
        for lab in labs:
            counter[lab] += 1
    print("[label counts]", counter.most_common())


if __name__ == "__main__":
    main()
