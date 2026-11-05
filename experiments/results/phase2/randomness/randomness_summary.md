# P2.11c: how six copies' shortfall divides between the bots' randomness

Seed 2026, commit 2b02bed.

50 lineups of six copies of one bot, 2,000 deals each, played in every condition with the same bots and the same cards. Remaining share: n s_u^2 / sigma^2 = 1 / ratio, the part of a hand's variance the rotation does not cancel. Shared: the six tables of a deal use the same draws of that source, so each hand's distribution is the default's. Off: the source is a constant, so the bots play differently.

## By condition

| Condition | Median ratio [IQR] | 95% CI | Remaining share (mean) | Median rho | Median SD per hand (bb) | SD change from default (median) | VPIP | PFR | Fold to bet |
|---|---|---|---|---|---|---|---|---|---|
| Default: all three live | 1.53x [1.42x, 1.64x] | [1.49x, 1.57x] | 66.2% | -0.069 | 19.30 | +0.0% | 0.316 | 0.160 | 0.397 |
| Mixed actions shared | 6.26x [5.20x, 8.79x] | [5.89x, 7.40x] | 15.9% | -0.168 | 18.89 | +0.6% | 0.316 | 0.160 | 0.396 |
| Bet-size noise shared | 1.54x [1.42x, 1.65x] | [1.49x, 1.56x] | 65.8% | -0.070 | 19.40 | +0.4% | 0.316 | 0.160 | 0.398 |
| Equity shared | 1.53x [1.45x, 1.66x] | [1.50x, 1.58x] | 64.8% | -0.070 | 19.40 | +0.1% | 0.316 | 0.160 | 0.397 |
| Only mixed actions live | 1.54x [1.46x, 1.67x] | [1.52x, 1.59x] | 65.1% | -0.071 | 19.53 | +0.4% | 0.316 | 0.160 | 0.398 |
| Only bet-size noise live | 31x [22x, 47x] | [27x, 36x] | 3.6% | -0.194 | 18.77 | +1.0% | 0.316 | 0.160 | 0.396 |
| Only equity live | 7.72x [5.99x, 12x] | [6.76x, 9.11x] | 13.7% | -0.174 | 18.92 | +0.7% | 0.316 | 0.160 | 0.396 |
| All three shared (check) | no variance [no variance, no variance] | [no variance, no variance] | 0.0% | -0.200 | 18.66 | +0.7% | 0.316 | 0.160 | 0.396 |
| Mixed actions shared, history key | 1.78x [1.65x, 1.97x] | [1.71x, 1.86x] | 55.9% | -0.088 | 18.91 | +0.2% | 0.316 | 0.160 | 0.398 |
| Bet-size noise shared, history key | 1.52x [1.43x, 1.63x] | [1.48x, 1.56x] | 66.2% | -0.069 | 19.11 | -0.1% | 0.316 | 0.160 | 0.399 |
| Equity shared, history key | 1.54x [1.44x, 1.65x] | [1.51x, 1.57x] | 65.7% | -0.070 | 19.26 | +0.4% | 0.316 | 0.160 | 0.397 |
| Only mixed actions live, history key | 1.54x [1.45x, 1.66x] | [1.50x, 1.58x] | 65.7% | -0.070 | 19.38 | +0.6% | 0.316 | 0.160 | 0.399 |
| Only bet-size noise live, history key | 1.77x [1.64x, 1.97x] | [1.72x, 1.84x] | 55.9% | -0.087 | 18.80 | +0.3% | 0.316 | 0.159 | 0.398 |
| Only equity live, history key | 6.61x [5.50x, 9.21x] | [6.17x, 7.29x] | 15.7% | -0.170 | 19.43 | -0.2% | 0.316 | 0.160 | 0.400 |
| Mixed actions off (u = 0.5) | 7.81x [5.49x, 51x] | [6.64x, 9.56x] | 12.3% | -0.174 | 20.25 | +38.1% | 0.309 | 0.121 | 0.329 |
| Bet-size noise off (factor 1) | 1.53x [1.43x, 1.64x] | [1.50x, 1.57x] | 65.9% | -0.070 | 19.09 | +0.4% | 0.316 | 0.160 | 0.399 |
| All three off (check) | no variance [no variance, no variance] | [no variance, no variance] | 0.0% | -0.200 | 20.66 | +39.8% | 0.309 | 0.121 | 0.331 |

## The split of the shortfall (shared conditions only)

Cancelled by the rotation: 33.8% [31.9%, 35.8%] of a hand's variance. Remaining: 66.2% [64.2%, 68.1%], which divides as follows.

| Term | Share of a hand's variance | Share of the shortfall |
|---|---|---|
| Mixed actions alone | 50.3% [48.2%, 52.3%] | 76.0% [73.6%, 78.2%] |
| Bet-size noise alone | 0.3% [-0.5%, 1.2%] | 0.5% [-0.8%, 1.9%] |
| Monte Carlo equity alone | 1.4% [0.4%, 2.6%] | 2.1% [0.6%, 3.8%] |
| Mixed x noise | 1.9% [0.5%, 3.2%] | 2.9% [0.8%, 4.9%] |
| Mixed x equity | 10.9% [9.5%, 12.4%] | 16.5% [14.1%, 18.8%] |
| Noise x equity | -0.6% [-2.0%, 0.5%] | -1.0% [-3.0%, 0.8%] |
| All three together | 2.0% [0.4%, 3.8%] | 3.0% [0.7%, 5.8%] |
| Mixed actions, interactions split equally | 57.3% [55.6%, 59.2%] | 86.7% [85.0%, 88.1%] |
| Bet-size noise, interactions split equally | 1.6% [1.2%, 2.0%] | 2.4% [1.8%, 3.0%] |
| Monte Carlo equity, interactions split equally | 7.2% [6.2%, 8.4%] | 10.9% [9.4%, 12.6%] |

## The same split with the tables paired by the exact public history

The two ends do not depend on the pairing; the terms between them do.

| Term | Share of a hand's variance | Share of the shortfall |
|---|---|---|
| Mixed actions alone | 10.3% [8.7%, 12.1%] | 15.6% [13.3%, 18.1%] |
| Bet-size noise alone | -0.1% [-0.9%, 0.8%] | -0.1% [-1.4%, 1.3%] |
| Monte Carlo equity alone | 0.4% [-0.5%, 1.3%] | 0.7% [-0.7%, 2.0%] |
| Mixed x noise | 40.2% [37.6%, 42.7%] | 60.8% [56.7%, 64.6%] |
| Mixed x equity | -0.5% [-2.0%, 1.0%] | -0.7% [-3.1%, 1.5%] |
| Noise x equity | 0.1% [-1.3%, 1.5%] | 0.2% [-2.0%, 2.3%] |
| All three together | 15.7% [13.4%, 18.2%] | 23.7% [20.4%, 27.2%] |
| Mixed actions, interactions split equally | 35.4% [34.1%, 36.8%] | 53.5% [52.4%, 54.5%] |
| Bet-size noise, interactions split equally | 25.3% [24.3%, 26.2%] | 38.2% [37.1%, 39.3%] |
| Monte Carlo equity, interactions split equally | 5.5% [4.8%, 6.2%] | 8.3% [7.3%, 9.3%] |

## By archetype

| Archetype | Median ratio | Cancelled | Mixed (split) | Noise (split) | Equity (split) | Mixed, noise, equity (split, history key) | Median ratio, mixed off | SD, default -> mixed off (bb) | VPIP, default -> mixed off |
|---|---|---|---|---|---|---|---|---|---|
| Nit | 1.34x | 25.5% [21.4%, 29.6%] | 81.8% [76.7%, 86.3%] | 1.7% [-0.4%, 3.6%] | 16.5% [12.9%, 21.0%] | 54.3%, 36.5%, 9.2% | 4.60x | 3.46 -> 6.38 | 0.112 -> 0.107 |
| TAG | 1.65x | 37.8% [34.8%, 40.5%] | 84.2% [83.2%, 85.2%] | 2.9% [1.8%, 4.3%] | 12.9% [11.2%, 14.3%] | 50.8%, 38.5%, 10.7% | 5.30x | 13.62 -> 20.00 | 0.229 -> 0.217 |
| LAG | 1.53x | 35.0% [33.2%, 37.1%] | 86.7% [85.5%, 87.9%] | 2.2% [1.5%, 2.8%] | 11.1% [9.8%, 12.2%] | 50.7%, 40.6%, 8.7% | 7.35x | 20.56 -> 28.53 | 0.310 -> 0.295 |
| Calling station | 1.57x | 36.8% [31.5%, 43.0%] | 88.1% [86.6%, 89.6%] | 3.5% [2.6%, 4.4%] | 8.3% [7.4%, 9.3%] | 58.3%, 33.2%, 8.5% | no variance | 24.11 -> 1.27 | 0.436 -> 0.410 |
| Maniac | 1.51x | 34.1% [32.2%, 36.1%] | 93.1% [91.8%, 94.2%] | 2.0% [1.3%, 2.8%] | 4.9% [4.3%, 5.5%] | 53.2%, 42.4%, 4.3% | 14x | 49.76 -> 73.91 | 0.496 -> 0.518 |

## Checks

Largest |per-deal sum| in chips with all three shared or off: {'shared_all': 0, 'off_all': 0}. Direct within-deal variance over n s_u^2: at most 0.8% from 1. Shared conditions against the default: SD at most 1.0% (median over lineups), stats at most 0.0054.
Default against P2.11b's committed clones_6: 300 players, largest ratio difference 0.

## Predictions

| | Prediction | Result | Holds |
|---|---|---|---|
| A1 | The baseline repeats P2.11b's committed clones_6: largest ratio difference 0 | 0 | yes |
| A2 | All three shared, all three off: every per-deal sum is exactly 0 chips | 0, 0 | yes |
| A3 | Direct within-deal variance over n s_u^2: largest |log| at most 0.08 | 0.00786 | yes |
| B1 | Sharing a source: median log SD change from the default within 0.05 | 0.00619, 0.00406, 0.000547, 0.00443, 0.0102, 0.00689, 0.00658, 0.00176, -0.000546, 0.00369, 0.00565, 0.00306, -0.00226 | yes |
| B2 | Sharing a source: each stat's mean within 0.03 of the default (largest) | 0.00541 | yes |
| B3 | Sharing a source, by archetype: median SD factor within [0.8, 1.4] | 0.851, 1.08 | yes |
| B4 | Sharing a source, by archetype: each stat's mean within 0.15 (largest) | 0.0283 | yes |
| C1 | Only mixed live: 90-110% of the default's remaining share, both keys | 0.984, 0.993 | yes |
| C2 | Only equity live: 10-20% of a hand's variance, both keys | 0.137, 0.157 | yes |
| C3 | Noise shared alone, equity shared alone: median ratio in [1.35, 1.9], both keys | 1.54, 1.53, 1.52, 1.54 | yes |
| D1 | Decision key, mixed shared alone: median ratio in [4.5, 12] | 6.26 | yes |
| D2 | Decision key, split: mixed at least 75% of the shortfall | 0.867 | yes |
| D3 | Decision key, split: equity 5-20% of the shortfall | 0.109 | yes |
| D4 | Decision key, split: noise at most 8% of the shortfall | 0.0244 | yes |
| D5a | Decision key: mixed x equity at least 4% of a hand's variance | 0.109 | yes |
| D5b | Decision key: mixed x equity is above equity alone | 0.109, 0.014 | yes |
| D6 | Decision key: maniacs have the lowest equity split share of the five | 0.0493, 0.165, 0.129, 0.111, 0.0833 | yes |
| H1a | Only noise live, decision key: at most 10% of a hand's variance | 0.0358 | yes |
| H1b | Only noise live, history key: at least 40% of a hand's variance | 0.559 | yes |
| H2 | History key, mixed shared alone: median ratio at most 2.2 | 1.78 | yes |
| H3 | History key, split: mixed 40-65% of the shortfall | 0.535 | yes |
| H4 | History key, split: noise 30-50% of the shortfall | 0.382 | yes |
| H5 | History key, split: equity at most 15% of the shortfall | 0.0828 | yes |
| H6 | History key: mixed x noise at least 25% of a hand's variance | 0.402 | yes |
| H7a | Decision key: mixed's split share above noise's | 0.867, 0.0244 | yes |
| H7b | History key: mixed's split share above noise's | 0.535, 0.382 | yes |
| E1a | Mixed actions off: the calling station's median SD below 0.1x the default's | 0.0525 | yes |
| E1b | Mixed actions off: nits' median SD above 1.5x the default's | 1.84 | yes |
| E2a | Mixed actions off: mean c-bet rate up at least 0.05 (lineups with opportunities) | 0.219 | yes |
| E2b | Mixed actions off: mean fold to a bet down at least 0.04 (lineups with them) | -0.113 | yes |
| E3a | Noise off: median log SD change within 0.03 | 0.00434 | yes |
| E3b | Noise off: each stat's mean within 0.03 of the default | -3.31e-05, 2.64e-07, 0.000314, -0.00159, -8.87e-05, 0.00188, 0.000376 | yes |
| E3c | Noise off: median ratio in [1.35, 1.9] | 1.53 | yes |

Run time 4311 s.
