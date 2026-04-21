# query_length_vs_ndcg — caption

Mean nDCG@10 as a function of query-length quartile (BERT WordPiece tokens)
for the six encoders across the three BEIR subsets. Quartile bounds differ
per dataset (nfcorpus: q1=2, q3=7; scifact: q1=15, q3=25; fiqa: q1=10,
q3=17) — see `experiments/phase6/length_quartile_bounds.csv`. Pre-trained
transformers (blue family) are near-flat in length, confirming that
attention-based aggregation handles query-length variation; convolutional
and recurrent models degrade visibly toward the longest quartile on
scifact and fiqa, supporting the receptive-field hypothesis from §4c of
the Phase 6 handoff.

FiQA's right-most quartile is the cleanest demonstration of the CNN
receptive-field ceiling: the `conv-retriever` line drops below all other
encoders as query length crosses ~17 tokens — the point at which no single
kernel-7 window can span the query.
