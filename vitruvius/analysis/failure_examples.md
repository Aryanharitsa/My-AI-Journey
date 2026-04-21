# Failure examples — Vitruvius Phase 6

Six characteristic failures, chosen to illustrate the phase's distinct findings. Each query text is verbatim from BEIR; doc IDs are from the Phase 3 / Phase 5 bench JSONs. Document titles and relevant-passage paraphrases are intentionally omitted — Phase 6 did not reload the BEIR corpora (see `experiments/phase6/SUMMARY.md`, Limitations). The analyst interpretations below are based solely on the query text, the qrels ID set, and the retrieved ID set.

## Example 1. CNN catastrophic failure on a long FiQA question

**Finding.** CNN's max-kernel receptive field (7 tokens) cannot span a 20+ token natural-language question; a pre-trained transformer of comparable size retrieves correctly. Directly supports the query-length-vs-nDCG plot.

**Query** (`fiqa:5380`, 23 WordPiece tokens): "Can somebody explain “leveraged debt investment positions” and “exposures” in this context for me, please?"

**Qrels.** 1 relevance-judged document(s). Relevant IDs (trim to 8): ['549254']


| Encoder | nDCG@10 | Top-3 retrieved (✓ = qrels-relevant) |
|---|---:|---|

| minilm-l6-v2 | 1.000 | 549254 ✓ / 286141 · / 314410 · |

| bert-base | 1.000 | 549254 ✓ / 177563 · / 314410 · |

| gte-small | 1.000 | 549254 ✓ / 286141 · / 51715 · |

| lstm-retriever | 0.000 | 66399 · / 584808 · / 349594 · |

| conv-retriever | 0.000 | 459726 · / 156116 · / 384613 · |

| mamba-retriever-fs | 0.000 | 330683 · / 8652 · / 586795 · |


**Rank of first relevant doc** (or `>100` if none in top-100):


| Encoder | first-relevant rank |
|---|---:|

| minilm-l6-v2 | 1 |

| bert-base | 1 |

| gte-small | 1 |

| lstm-retriever | >100 |

| conv-retriever | >100 |

| mamba-retriever-fs | >100 |


## Example 2. Short ambiguous query that defeats every from-scratch encoder

**Finding.** With three tokens or fewer, the query carries no span structure for any architecture to exploit. Transformer pre-training supplies the entity prior that bridges the gap; from-scratch encoders lack it.

**Query** (`nfcorpus:PLAIN-2301`, 3 WordPiece tokens): "uterine health"

**Qrels.** 13 relevance-judged document(s). Relevant IDs (trim to 8): ['MED-3503', 'MED-3504', 'MED-3794', 'MED-4753', 'MED-4755', 'MED-4756', 'MED-4757', 'MED-4938'] …


| Encoder | nDCG@10 | Top-3 retrieved (✓ = qrels-relevant) |
|---|---:|---|

| minilm-l6-v2 | 0.344 | MED-4160 · / MED-557 ✓ / MED-4941 ✓ |

| bert-base | 0.000 | MED-4160 · / MED-1210 · / MED-3645 · |

| gte-small | 0.110 | MED-5191 · / MED-5186 · / MED-557 ✓ |

| lstm-retriever | 0.000 | MED-1598 · / MED-1886 · / MED-1479 · |

| conv-retriever | 0.000 | MED-4540 · / MED-4842 · / MED-1352 · |

| mamba-retriever-fs | 0.000 | MED-1213 · / MED-4289 · / MED-4951 · |


**Rank of first relevant doc** (or `>100` if none in top-100):


| Encoder | first-relevant rank |
|---|---:|

| minilm-l6-v2 | 2 |

| bert-base | 37 |

| gte-small | 3 |

| lstm-retriever | >100 |

| conv-retriever | >100 |

| mamba-retriever-fs | 13 |


## Example 3. Scientific-claim failure on a domain-jargon SciFact query

**Finding.** Where pre-training's value concentrates: a biomedical claim containing dataset-specific jargon is correctly grounded by BERT (seen in PubMed-adjacent text during pre-training) but not by any encoder trained only on MS MARCO.

**Query** (`scifact:718`, 20 WordPiece tokens): "Low nucleosome occupancy correlates with low methylation levels across species."

**Qrels.** 1 relevance-judged document(s). Relevant IDs (trim to 8): ['17587795']


| Encoder | nDCG@10 | Top-3 retrieved (✓ = qrels-relevant) |
|---|---:|---|

| minilm-l6-v2 | 1.000 | 17587795 ✓ / 175735 · / 25254425 · |

| bert-base | 1.000 | 17587795 ✓ / 2754534 · / 33554389 · |

| gte-small | 1.000 | 17587795 ✓ / 25254425 · / 34034749 · |

| lstm-retriever | 0.000 | 16734530 · / 175735 · / 5579368 · |

| conv-retriever | 0.000 | 7488455 · / 34034749 · / 9889151 · |

| mamba-retriever-fs | 0.000 | 1259280 · / 20761364 · / 18895793 · |


**Rank of first relevant doc** (or `>100` if none in top-100):


| Encoder | first-relevant rank |
|---|---:|

| minilm-l6-v2 | 1 |

| bert-base | 1 |

| gte-small | 1 |

| lstm-retriever | >100 |

| conv-retriever | >100 |

| mamba-retriever-fs | 93 |


## Example 4. Multi-concept FiQA question where all three transformers disagree on ranking

**Finding.** Even within the transformer family, retrieval is not uniform. GTE's contrastive pre-training on web text outperforms generic BERT/MiniLM on multi-concept financial questions — a reminder that 'transformer' is not one architecture but three in this Pareto.

**Query** (`fiqa:7096`, 14 WordPiece tokens): "What's the formula for profits and losses when I delta hedge?"

**Qrels.** 1 relevance-judged document(s). Relevant IDs (trim to 8): ['482238']


| Encoder | nDCG@10 | Top-3 retrieved (✓ = qrels-relevant) |
|---|---:|---|

| minilm-l6-v2 | 0.000 | 346641 · / 477597 · / 234935 · |

| bert-base | 0.000 | 338344 · / 209492 · / 197863 · |

| gte-small | 0.631 | 202432 · / 482238 ✓ / 168006 · |

| lstm-retriever | 0.000 | 270345 · / 158614 · / 435855 · |

| conv-retriever | 0.000 | 12779 · / 555273 · / 246706 · |

| mamba-retriever-fs | 0.000 | 41160 · / 231646 · / 75437 · |


**Rank of first relevant doc** (or `>100` if none in top-100):


| Encoder | first-relevant rank |
|---|---:|

| minilm-l6-v2 | 14 |

| bert-base | 24 |

| gte-small | 2 |

| lstm-retriever | >100 |

| conv-retriever | >100 |

| mamba-retriever-fs | 82 |


## Example 5. Numeric-entity FiQA query — a tokenizer story, not an architecture story

**Finding.** WordPiece splits most digit sequences into single-digit pieces; retrieval over numeric entities is therefore lexical for every encoder here. From-scratch encoders fail harder because they also lack the surrounding-vocabulary grounding that transformers bring, not because the numbers themselves are processed differently.

**Query** (`fiqa:10827`, 17 WordPiece tokens): "How much should I be contributing to my 401k given my employer's contribution?"

**Qrels.** 5 relevance-judged document(s). Relevant IDs (trim to 8): ['107554', '160786', '42301', '7748', '95282']


| Encoder | nDCG@10 | Top-3 retrieved (✓ = qrels-relevant) |
|---|---:|---|

| minilm-l6-v2 | 0.000 | 41330 · / 576391 · / 290105 · |

| bert-base | 0.000 | 41330 · / 290105 · / 101490 · |

| gte-small | 0.000 | 38532 · / 290105 · / 39071 · |

| lstm-retriever | 0.000 | 399543 · / 168890 · / 13275 · |

| conv-retriever | 0.000 | 101490 · / 451501 · / 399543 · |

| mamba-retriever-fs | 0.000 | 497561 · / 448358 · / 480036 · |


**Rank of first relevant doc** (or `>100` if none in top-100):


| Encoder | first-relevant rank |
|---|---:|

| minilm-l6-v2 | 27 |

| bert-base | 49 |

| gte-small | 15 |

| lstm-retriever | >100 |

| conv-retriever | >100 |

| mamba-retriever-fs | 55 |


## Example 6. Universal-loss query — intrinsic hardness, not an architecture gap

**Finding.** All six encoders fail here. The qrels-relevant document is plausibly absent from the top-100 for every encoder — a dataset-hardness signal, not an architecture signal. No amount of swapping encoders rescues this query; it motivates reranking or reasoning, not better dense retrieval.

**Query** (`fiqa:10547`, 10 WordPiece tokens): "How much do brokerages pay exchanges per trade?"

**Qrels.** 1 relevance-judged document(s). Relevant IDs (trim to 8): ['571306']


| Encoder | nDCG@10 | Top-3 retrieved (✓ = qrels-relevant) |
|---|---:|---|

| minilm-l6-v2 | 0.000 | 4883 · / 234983 · / 149153 · |

| bert-base | 0.000 | 503981 · / 582908 · / 110608 · |

| gte-small | 0.000 | 503981 · / 234983 · / 4883 · |

| lstm-retriever | 0.000 | 149153 · / 530155 · / 246986 · |

| conv-retriever | 0.000 | 502223 · / 443925 · / 585706 · |

| mamba-retriever-fs | 0.000 | 469830 · / 30556 · / 270221 · |


**Rank of first relevant doc** (or `>100` if none in top-100):


| Encoder | first-relevant rank |
|---|---:|

| minilm-l6-v2 | 40 |

| bert-base | >100 |

| gte-small | >100 |

| lstm-retriever | >100 |

| conv-retriever | >100 |

| mamba-retriever-fs | >100 |


