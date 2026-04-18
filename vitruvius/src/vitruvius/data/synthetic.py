"""Tiny synthetic corpus used by the CPU smoke test.

Deterministic given a fixed seed. The structure mirrors a BEIR triplet:
``(corpus, queries, qrels)`` where ``qrels[qid][docid] = relevance``.
The 'relevance signal' is purely lexical (shared topic keywords) so that any
encoder that respects token similarity should retrieve relevant docs above
unrelated noise.
"""
from __future__ import annotations

import random
from dataclasses import dataclass

from vitruvius.utils.seed import set_seed

TOPICS = [
    ("astronomy", ["telescope", "galaxy", "nebula", "supernova", "quasar"]),
    ("baking", ["sourdough", "yeast", "oven", "flour", "fermentation"]),
    ("cycling", ["pedal", "drivetrain", "cassette", "derailleur", "tire"]),
    ("databases", ["index", "btree", "transaction", "rollback", "mvcc"]),
    ("ecology", ["wetland", "biome", "predator", "succession", "estuary"]),
    ("finance", ["bond", "yield", "duration", "coupon", "treasury"]),
    ("gardening", ["compost", "perennial", "trowel", "mulch", "loam"]),
    ("hiking", ["trail", "elevation", "switchback", "scree", "pass"]),
    ("immunology", ["antibody", "tcell", "antigen", "cytokine", "vaccine"]),
    ("jazz", ["bebop", "syncopation", "saxophone", "improvisation", "chord"]),
]
NOISE = [
    "weekend", "morning", "schedule", "calendar", "color", "ladder",
    "neighbor", "envelope", "metric", "ribbon", "surface", "border",
    "signal", "register", "bucket", "channel", "message",
]


@dataclass
class SyntheticCorpus:
    corpus: dict[str, dict[str, str]]   # docid -> {"title": ..., "text": ...}
    queries: dict[str, str]              # qid -> query string
    qrels: dict[str, dict[str, int]]     # qid -> docid -> relevance


def make_corpus(
    n_queries: int = 10,
    docs_per_topic: int = 5,
    seed: int = 1729,
) -> SyntheticCorpus:
    """Return a tiny corpus: 10 queries, 50 docs, 2 relevant docs per query."""
    if n_queries > len(TOPICS):
        raise ValueError(f"max queries supported is {len(TOPICS)}")
    set_seed(seed)
    rng = random.Random(seed)

    corpus: dict[str, dict[str, str]] = {}
    queries: dict[str, str] = {}
    qrels: dict[str, dict[str, int]] = {}

    docid = 0
    relevant_by_topic: dict[str, list[str]] = {}
    for topic, keywords in TOPICS[:n_queries]:
        relevant_by_topic[topic] = []
        for _ in range(docs_per_topic):
            picked_kws = rng.sample(keywords, k=min(3, len(keywords)))
            picked_noise = rng.sample(NOISE, k=4)
            text_words = picked_kws + picked_noise
            rng.shuffle(text_words)
            doc = {
                "title": f"Notes on {topic}",
                "text": " ".join(text_words),
            }
            did = f"d{docid}"
            corpus[did] = doc
            relevant_by_topic[topic].append(did)
            docid += 1

    for i, (topic, keywords) in enumerate(TOPICS[:n_queries]):
        qid = f"q{i}"
        q_kws = rng.sample(keywords, k=2)
        queries[qid] = f"What can you tell me about {topic}? Mention {q_kws[0]} and {q_kws[1]}."
        relevant = rng.sample(relevant_by_topic[topic], k=2)
        qrels[qid] = {did: 1 for did in relevant}

    return SyntheticCorpus(corpus=corpus, queries=queries, qrels=qrels)
