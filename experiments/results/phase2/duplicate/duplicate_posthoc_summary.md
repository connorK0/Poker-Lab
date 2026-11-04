# P2.11b, post hoc: reading the duplicate step's results

Seed 2026, commit c7f0029; from duplicate.json at 0fb6e53. Nothing here is pre-registered.

## The share of perfect cancellation, -(n - 1) rho

| Lineup kind | n | Median rho | Bound | Share: median [IQR] | 1 / (1 - s) | Measured ratio |
|---|---|---|---|---|---|---|
| Six copies of one bot | 6 | -0.069 | -0.20 | 34.5% [29.7%, 39.1%] | 1.526 | 1.525 |
| Six bots of one archetype | 6 | -0.061 | -0.20 | 30.7% [27.0%, 34.3%] | 1.444 | 1.443 |
| Six bots, random mix | 6 | -0.046 | -0.20 | 23.0% [17.3%, 28.4%] | 1.299 | 1.298 |
| Three bots, random mix | 3 | -0.100 | -0.50 | 20.0% [13.6%, 26.1%] | 1.251 | 1.250 |
| Two bots, random mix | 2 | -0.224 | -1.00 | 22.4% [14.6%, 30.4%] | 1.289 | 1.288 |

## Coverage of a nominal 95% percentile bootstrap

Normal theory, mean of G values: G = 5: 84.6%, G = 8: 89.1%, G = 10: 90.4%, G = 20: 92.9%, G = 50: 94.2%, G = 100: 94.6%.

| Lineup kind | G = 8 | G = 10 | G = 20 | G = 50 |
|---|---|---|---|---|
| Six copies of one bot | 88.2% | 89.4% | 93.0% | 94.2% |
| Six bots of one archetype | 89.2% | 90.5% | 93.1% | 93.6% |
| Six bots, random mix | 89.6% | 90.5% | 93.0% | 94.0% |
| Three bots, random mix | 90.0% | 91.8% | 94.0% | 93.8% |
| Two bots, random mix | 91.5% | 93.5% | 94.4% | 94.0% |

## Resampling lineups against resampling players

| Lineup kind | Lineups | Players | Width, lineups / players |
|---|---|---|---|
| Six copies of one bot | [1.491, 1.571] | [1.502, 1.558] | 1.44 |
| Six bots of one archetype | [1.426, 1.469] | [1.432, 1.463] | 1.37 |
| Six bots, random mix | [1.278, 1.319] | [1.284, 1.311] | 1.45 |
| Three bots, random mix | [1.216, 1.279] | [1.225, 1.269] | 1.43 |
| Two bots, random mix | [1.240, 1.324] | [1.245, 1.313] | 1.24 |

## Other readings

* Heads-up: same archetype 1.46x (33 lineups), different archetypes 1.26x (167); largest 6.08x (tag against nit, rho -0.836), next 2.33x.
* Rank correlation of ratio and per-hand SD over players: Six copies of one bot 0.28, Six bots of one archetype 0.40, Six bots, random mix 0.52, Three bots, random mix 0.24, Two bots, random mix 0.15.
* Six copies of one bot: median relative SD of the ratios within a lineup 6.6%; lineup medians from 1.15x (nit) to 2.47x (calling_station).
* Distinct bots in the 60 seats of a replicate's mixed_6 lineups: 35 to 43.
* The identity from rho: largest relative gap 1.1e-15. Every reported CI contains its median: True.

Run time 73 s.
