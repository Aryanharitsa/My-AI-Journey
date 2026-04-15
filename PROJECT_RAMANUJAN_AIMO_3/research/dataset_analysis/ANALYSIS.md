# Dataset Analysis

## Reference10
The official 10-problem calibration set with known answers and verified difficulty ratings.

| Solve Rate | Count | Example Problems |
|---|---|---|
| >80% (consistent) | 5 | Alice/Bob ages, function summation, tournament ranking |
| ~50% (borderline) | 2 | Triangle geometry (0e644e), rectangle tiling (a295e9) |
| 0% (knowledge wall) | 3 | Fibonacci incircle (641659), n-Norwegian (86e8e5), shifty functions (dd7f5e) |

## Failure Analysis on the 3 Walls
- **641659** (Fibonacci + incircle): Multi-layer geometric construction. 0/16 samples produced any answer across all experiments.
- **86e8e5** (n-Norwegian at scale 3^{2025!}): Requires deep number-theoretic structure. 0/16.
- **dd7f5e** (shifty functions): Abstract functional algebra with shift operators. 0/16 in most runs, but 1/4 vanilla sample cracked it in the diversity experiment — pure sampling lottery.

## Public-50 Distribution Estimate
Given v63 scores 36/50: ~15 easy (95% solve), ~20 medium (75%), ~10 hard (50%), ~5 wall (20%).
Expected: 14.25 + 15 + 5 + 1 = 35.25. Matches 36.
