# Failure taxonomy — Vitruvius Phase 6

Categories applied to queries where the encoder's nDCG@10 < 0.1. Labels are **non-exclusive**: a single query can carry multiple codes (e.g. a long FiQA question with a dollar amount is simultaneously ``LEN-LONG``, ``NATURAL-QUESTION``, and ``NUMERIC-ENTITY``).

Mechanical codes (``LEN-*``, ``NATURAL-QUESTION``, ``NUMERIC-ENTITY``, ``NEGATION``, ``MULTI-CONCEPT``, ``DOMAIN-TERM``) are assigned by rule from the query string and the dataset's domain lexicon. The semantic codes that the handoff's seed list proposed — PARAPHRASE, AMBIGUOUS, MULTI-HOP — require access to the corpus documents to verify, which Phase 6 deliberately does not reload (see `SUMMARY.md` § Limitations). Queries that plausibly belong to those categories land in ``UNCATEGORIZED`` and are flagged as future work.

## `LEN-LONG`

Query exceeds 20 WordPiece tokens. Tests whether the encoder's receptive field and aggregation strategy can integrate information across a long span. CNN (kernel stack 3/5/7 → max receptive field 7 tokens) is the primary architectural casualty; transformers are nearly flat across lengths.

**Examples (query text verbatim from BEIR):**

- `fiqa:10136` — "How to minimise the risk of a reduction in purchase power in case of Brexit for money held in a bank account?"  
  *failed for:* conv-retriever
- `fiqa:10447` — "Is there an advantage to a traditional but non-deductable IRA over a taxable account? [duplicate]"  
  *failed for:* conv-retriever
- `fiqa:10975` — "How to contribute to Roth IRA when income is at the maximum limit & you have employer-sponsored 401k plans?"  
  *failed for:* bert-base, conv-retriever, gte-small, lstm-retriever, mamba-retriever-fs, minilm-l6-v2

## `LEN-SHORT`

Query has ≤5 WordPiece tokens (e.g. ``deafness``, ``DHA``, ``milk``). Low lexical signal; every encoder must rely on learned priors rather than span composition. Transformers benefit from pre-training's entity coverage; from-scratch encoders, which never saw the MS MARCO training corpus' long tail, do not.

**Examples (query text verbatim from BEIR):**

- `fiqa:1783` — "Freelancing Tax implication"  
  *failed for:* lstm-retriever
- `fiqa:2713` — "Physical Checks - Mailing"  
  *failed for:* conv-retriever
- `fiqa:3759` — "Simplifying money management"  
  *failed for:* conv-retriever

## `NATURAL-QUESTION`

Query is a full natural-language question (starts with how/what/why/is/can/do… or ends in ``?``). Characteristic of FiQA. These queries interleave interrogative framing with substantive content; mean-pooling encoders that do not dampen the question-form tokens tend to drown the keyword signal.

**Examples (query text verbatim from BEIR):**

- `fiqa:10034` — "Tax implications of holding EWU (or other such UK ETFs) as a US citizen?"  
  *failed for:* conv-retriever, gte-small, lstm-retriever, mamba-retriever-fs
- `fiqa:10039` — "Do individual investors use Google to obtain stock quotes?"  
  *failed for:* lstm-retriever
- `fiqa:10136` — "How to minimise the risk of a reduction in purchase power in case of Brexit for money held in a bank account?"  
  *failed for:* conv-retriever

## `NUMERIC-ENTITY`

Query hinges on a digit-bearing entity — a year, dollar amount, rate, or code (e.g. ``401k``, ``S&P 500``, ``2015``). BERT WordPiece splits most numbers into single-digit pieces, so numeric matching is effectively lexical for all six encoders; failures on this category are tokenizer-driven, not architectural.

**Examples (query text verbatim from BEIR):**

- `fiqa:10827` — "How much should I be contributing to my 401k given my employer's contribution?"  
  *failed for:* conv-retriever
- `fiqa:10845` — "Rationale behind using 12, 26 and 9 to calculate MACD"  
  *failed for:* conv-retriever
- `fiqa:10975` — "How to contribute to Roth IRA when income is at the maximum limit & you have employer-sponsored 401k plans?"  
  *failed for:* bert-base, conv-retriever, gte-small, lstm-retriever, mamba-retriever-fs, minilm-l6-v2

## `NEGATION`

Query contains ``not``/``without``/``fails``/``lack`` or a contraction thereof. The qrels-relevant document asserts the absence of a property; retrieved documents typically describe the property's presence. Dense encoders cannot distinguish the two without dedicated negation-sensitive training.

**Examples (query text verbatim from BEIR):**

- `fiqa:10994` — "Net loss not distributed by mutual funds to their shareholders?"  
  *failed for:* lstm-retriever
- `fiqa:1871` — "Is there any US bank that does not charge for incoming wire transfers?"  
  *failed for:* conv-retriever
- `fiqa:2010` — "Paypal website donations without being a charity"  
  *failed for:* lstm-retriever

## `MULTI-CONCEPT`

Query requires matching two or more conjoined concepts — ``X and Y``, ``X vs Y``, or comma-separated noun groups. The encoder must allocate capacity to each; easy to collapse into a bag-of-terms match on the first concept only.

**Examples (query text verbatim from BEIR):**

- `fiqa:10213` — "Looking for good investment vehicle for seasonal work and savings"  
  *failed for:* bert-base, conv-retriever, gte-small, lstm-retriever, mamba-retriever-fs, minilm-l6-v2
- `fiqa:10710` — "Probablity of touching In the money vs expiring in the money for an american option"  
  *failed for:* conv-retriever, lstm-retriever
- `fiqa:10845` — "Rationale behind using 12, 26 and 9 to calculate MACD"  
  *failed for:* conv-retriever

## `DOMAIN-TERM`

Query contains a technical token from the dataset's domain lexicon (see ``DOMAIN_LEXICON`` in ``phase6_label_taxonomy.py``). The from-scratch encoders were trained on MS MARCO's web-query distribution; pre-trained transformers saw Wikipedia + book-corpus text that covers biomedical and financial jargon. Failures here are the clearest signal of where pre-training's vocabulary coverage pays off.

**Examples (query text verbatim from BEIR):**

- `fiqa:10034` — "Tax implications of holding EWU (or other such UK ETFs) as a US citizen?"  
  *failed for:* conv-retriever, gte-small, lstm-retriever, mamba-retriever-fs
- `fiqa:10152` — "What does a high operating margin but a small but positive ROE imply about a company?"  
  *failed for:* conv-retriever
- `fiqa:10447` — "Is there an advantage to a traditional but non-deductable IRA over a taxable account? [duplicate]"  
  *failed for:* conv-retriever

## `UNCATEGORIZED`

Query does not trigger any of the heuristic rules. Most such queries are medium-length noun phrases without obvious markers; failures here are likely driven by the same paraphrase / semantic-mismatch factors that the heuristic taxonomy does not capture directly. Flagged so the residual is not hidden.

**Examples (query text verbatim from BEIR):**

- `fiqa:10601` — "Bitcoin Cost Basis Purchases"  
  *failed for:* lstm-retriever
- `fiqa:1284` — "Tax consequences when foreign currency changes in value"  
  *failed for:* lstm-retriever
- `fiqa:2580` — "Stock market vs. baseball card trading analogy"  
  *failed for:* conv-retriever

---

## Category counts per architecture family (labeled subset)

| category         |   convolutional |   recurrent |   ssm |   transformer |
|:-----------------|----------------:|------------:|------:|--------------:|
| DOMAIN-TERM      |              64 |          65 |    48 |            73 |
| LEN-LONG         |              27 |          58 |    48 |            23 |
| LEN-SHORT        |              40 |          34 |    28 |            78 |
| MULTI-CONCEPT    |              45 |          58 |    33 |            40 |
| NATURAL-QUESTION |             166 |         140 |    70 |           152 |
| NEGATION         |               8 |          20 |    16 |             5 |
| NUMERIC-ENTITY   |              29 |          48 |    43 |            22 |
| UNCATEGORIZED    |              25 |          47 |    44 |            38 |

*Totals exceed the number of distinct labeled failures because a single failing query can carry multiple category codes.*
