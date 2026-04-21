# Cumulative pruning curves

Three panels, one per BEIR test subset. X-axis: fraction of heads pruned. Y-axis: nDCG@10 relative to baseline. Each line = one encoder. The 0.95 / 0.90 horizontal reference lines correspond to 5% / 10% relative drop; each encoder's X marker annotates the 5%-drop crossing.

Ordering: single-head-importance ascending (least-important first). This is NOT the true cumulative-optimal set (heads can compensate); a Taylor-saliency or iterative-greedy baseline would likely push these curves higher. Flagged per handoff §7.4 as a limitation.

Source: `experiments/phase7/cumulative_pruning/`.
