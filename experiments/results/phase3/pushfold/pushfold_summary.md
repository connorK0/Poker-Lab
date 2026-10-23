# P3.1: heads-up push/fold hold'em, exact Nash

Matchup table: 50,000 boards per class pair (seed 20261023), max standard error 0.0022. Equity vs a random hand agrees with the Phase 1 table to 0.0067 at worst (J4s), 0.0018 on average.

Exact LP on 381 stacks from 1 to 20 bb; worst exploitability of a solution 1.2e-15 bb/hand.

| Stack (bb) | SB value (bb/hand) | SB push share | BB call share |
|---|---|---|---|
| 1.5 | +0.0003 | 98.2% | 100.0% |
| 2 | +0.0095 | 89.4% | 100.0% |
| 3 | +0.0516 | 77.9% | 92.1% |
| 5 | +0.0561 | 71.5% | 62.0% |
| 8 | -0.0042 | 61.8% | 45.0% |
| 10 | -0.0454 | 58.3% | 37.4% |
| 15 | -0.1272 | 44.9% | 28.5% |
| 20 | -0.1832 | 40.0% | 21.7% |

Per-hand charts vs HoldemResources (https://www.holdemresources.net/hune, retrieved 2026-10-23):

| Chart | Exact on the grid | Within 0.1 bb | Within 0.25 bb | Mean | Worst |
|---|---|---|---|---|---|
| pusher | 104/169 | 151/169 | 158/169 | 0.069 bb | 1.30 bb (43s, 84s, 94s) |
| caller | 93/169 | 169/169 | 169/169 | 0.025 bb | 0.10 bb (A4o, K5s, K2s) |

CFR+ (average strategy) vs LP:

| Stack | Iterations | Exploitability | Push decisions = LP | Call decisions = LP |
|---|---|---|---|---|
| 5 | 3,000 | 3.9e-07 | 169/169 | 169/169 |
| 10 | 3,000 | 2.3e-07 | 169/169 | 169/169 |
| 20 | 3,000 | 3.3e-07 | 169/169 | 169/169 |

Exploitability of heuristic range rules (bb/hand; Nash = 0):

| Rule | 2 bb | 3 bb | 4 bb | 5 bb | 6 bb | 8 bb | 10 bb | 12 bb | 15 bb | 20 bb |
|---|---|---|---|---|---|---|---|---|---|---|
| ranked: top hands by equity vs a random hand, Nash's own push and call shares | 0.000 | 0.001 | 0.001 | 0.003 | 0.003 | 0.006 | 0.007 | 0.008 | 0.012 | 0.013 |
| tight: push top 20%, call top 10% at every stack | 0.557 | 0.518 | 0.497 | 0.478 | 0.460 | 0.425 | 0.391 | 0.357 | 0.306 | 0.222 |
| medium: push top 45%, call top 25% at every stack | 0.369 | 0.302 | 0.260 | 0.228 | 0.200 | 0.146 | 0.094 | 0.046 | 0.015 | 0.026 |
| loose: push top 75%, call top 45% at every stack | 0.197 | 0.130 | 0.083 | 0.051 | 0.026 | 0.019 | 0.048 | 0.089 | 0.160 | 0.289 |
