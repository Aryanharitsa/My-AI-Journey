# failure_by_architecture — caption

Per-query failures (nDCG@10 < 0.1) from the Phase 6 labeled sample (470
distinct queries, 1,116 (encoder, dataset, query) triples) assigned to the
eight-category failure taxonomy defined in
`vitruvius/analysis/failure_taxonomy.md`. Left panel (a) shows raw counts;
right panel (b) column-normalizes each architecture family so the reader
can compare *distribution of failure types* rather than absolute counts.
Labels are non-exclusive, so column sums are not the number of queries.

**Reading aid.** The cells worth staring at:

- `NATURAL-QUESTION` dominates the convolutional and recurrent columns —
  FiQA's full-sentence question style is the modal failure for both.
- `LEN-SHORT` is the single largest transformer failure, consistent with
  the high zero-nDCG rate on nfcorpus where queries are 2–5 tokens.
- `DOMAIN-TERM` loads roughly evenly across all four families — a reminder
  that jargon coverage is a shared weakness of both web-pre-trained and
  MS MARCO-trained encoders on biomedical / scientific corpora.
- `LEN-LONG` spares transformers but hits recurrent/SSM models hardest.

Sampling procedure and threshold choice documented in
`vitruvius/experiments/phase6/README.md`.
