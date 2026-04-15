# Encoder Archaeology — Methodology

## Phase 1: Benchmarking Infrastructure (Week 1, ~15 hours)

### 1.1 Dataset Setup
- Download and preprocess BEIR subsets: NFCorpus, SciFact, FiQA
- Standardize data loading: queries, documents, relevance judgments (qrels)
- Verify: correct number of queries, documents, relevance labels per dataset

### 1.2 Encoder Integration
- Wrap each encoder in a common interface:
  - `encode_queries(queries: List[str]) -> np.ndarray`
  - `encode_documents(docs: List[str]) -> np.ndarray`
- Pre-trained models (BERT-base, MiniLM, GTE-small): load from HuggingFace sentence-transformers
- Mamba Retriever: load from published checkpoint or replicate training
- LSTM / CNN: train from scratch on MS MARCO passage subset (~100K pairs, not full dataset)

### 1.3 Evaluation Pipeline
- Index documents using FAISS (flat index for accuracy, no approximation)
- For each query: retrieve top-100, compute nDCG@10, Recall@100, MRR@10
- Latency profiling: torch.cuda.Event timing, warm-up runs, report median over 100 runs
- Batch size sweep: 1, 8, 32

## Phase 2: Pareto Analysis (Week 1-2, ~10 hours)

### 2.1 Results Table
- Full table: Model x Dataset x Metric
- Pareto frontier plot: nDCG@10 vs median query latency
- Throughput comparison: queries/second at batch=32

### 2.2 Sanity Checks
- Verify BERT-base results match published BEIR numbers (within 1-2 points)
- Verify MiniLM is faster than BERT (obvious but must confirm setup is correct)
- If any result is surprising, investigate before proceeding

## Phase 3: Architectural Analysis (Week 2-3, ~20 hours)

### 3.1 Per-Query Error Analysis
- For each query where BERT-base succeeds but MiniLM/Mamba fails:
  - Characterize query: length, specificity, domain terminology
  - Characterize the missed relevant document: length, position of relevant passage
  - Hypothesis: failures concentrate on queries requiring cross-sentence reasoning
- Produce: failure taxonomy with examples

### 3.2 Attention Head Pruning (Transformer Models Only)
- For BERT-base: iteratively prune attention heads (zero out one head at a time)
- Measure: nDCG@10 after each head is removed
- Question: How many heads can be removed before retrieval accuracy drops significantly?
- Expected finding: Many heads are redundant for retrieval (following Michel et al. 2019)
- Produce: head importance ranking for retrieval specifically

### 3.3 Position Sensitivity Test
- For each architecture: randomly shuffle token positions in documents before encoding
- Measure: nDCG@10 with shuffled vs original
- Hypothesis: Retrieval is more position-insensitive than LM tasks — if true, positional encoding is partially wasted compute
- Control: Also shuffle query tokens (expected to hurt more)

### 3.4 The Opinions Section
Based on experimental evidence, argue 2-3 concrete positions. Draft examples (to be refined based on actual results):
- "Full bidirectional attention is overkill for document encoding in retrieval because [evidence from pruning experiment]"
- "Mamba's linear scaling provides genuine advantage over transformer for documents above N tokens because [latency data]"
- "The retrieval accuracy gap between deep and shallow encoders concentrates on [specific query type], suggesting targeted architectural solutions rather than uniform depth"

## Phase 4: Writeup (Week 3, ~5 hours)

### Structure
1. Abstract (3-4 sentences)
2. Introduction + motivation (citing MSR framing explicitly)
3. Experimental setup (encoders, datasets, metrics)
4. Results (tables, Pareto plot)
5. Analysis (error analysis, pruning, position sensitivity)
6. Opinions (the argument section — 2-3 concrete positions)
7. Limitations and future work
8. References

### Target: 5-7 pages, honest, specific, opinionated.
