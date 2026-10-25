# P3.7: priors fit to real players, and do the synthetic-trained models transfer?

13 players (pluribus), 50,000 player-hands, 9,000 test decisions; synthetic: 5 Phase 1 replicates. Brackets: 95% bootstrap intervals over players unless stated.

## Pre-registered predictions

- B1 human prior beats uniform (n <= 200, every stat): **does not hold**; fails: vpip at 200, pfr at 200, fold_to_cbet at 200
- B2 human prior beats synthetic: **does not hold**; fails: fold_to_cbet at 25, wsd at 25, fold_to_cbet at 50, wsd at 50, cbet at 100, fold_to_cbet at 100, wsd at 100, vpip at 200, pfr at 200, cbet at 200, fold_to_cbet at 200, wsd at 200
- B3 human prior worse than uniform on bots for >= 5 of 10 stats at 100 and 500 hands: 100 hands 8; 500 hands 8; **holds**
- C1a support check flags at most 7% of held-out bot windows: max 5.8%; **holds**
- C1b flags at least 80% of human windows at 500 / 1,000 / 2,000 hands: 100%, 100%, 100%; **holds**
- C3 consistency verdict *fails* (predicted: fails); **holds**
- D1 synthetic situation model worse than the human one at every checkpoint: **holds**
- D2 the profile reduces log-loss by less than 3%: 25 hands 1.0% / -0.0%, 100 hands 1.0% / -0.2%, 500 hands 1.7% / -0.6%, 1500 hands 3.8% / -2.3% (synthetic / human); **does not hold**
- D3 synthetic situation model worse calibrated: **does not hold**

Control (the same seating replayed by Phase 1 bots):

- C1 at most 25% flagged: max 55.8%; **does not hold**
- C3 not *fails*: verdict *inconclusive*; **holds**
- D1 synthetic model not worse: **holds**
- D2 profile reduction at least 3%: 25 hands 13.3%, 100 hands 13.1%, 500 hands 12.6%, 1500 hands 16.3%; **holds**

B4 decision: `refit_priors` default `min_players` = **8** (humans: 2, bots: 8; was 30).

## A. Priors

| Stat | Humans: mean | Humans: concentration | Synthetic: mean | Synthetic: concentration | Pluribus |
|---|---|---|---|---|---|
| VPIP | 26.5% [24.8%, 28.2%] | 227 [129, 1,616] | 31.9% | 6.4 | 26.9% |
| PFR | 17.9% [16.9%, 19.0%] | 585 [269, 66,137] | 17.8% | 8.8 | 18.0% |
| 3-bet | 8.2% [7.3%, 9.0%] | 335 [220, 1,348] | 10.0% | 8.7 | 7.3% |
| Fold to 3-bet | 48.0% [42.6%, 53.4%] | 36 [18, 678] | 63.6% | 24.4 | 49.7% |
| C-bet | 51.6% [47.3%, 56.3%] | 46 [23, 176] | 65.3% | 5.1 | 53.7% |
| Fold to c-bet | 45.6% [41.7%, 50.1%] | 97 [26, 39,465] | 38.9% | 9.2 | 45.0% |
| Aggression share | 69.6% [66.8%, 72.7%] | 96 [42, 11,198] | 62.5% | 5.8 | 70.0% |
| Fold to a bet | 51.3% [48.8%, 54.1%] | 186 [82, 124,788] | 38.8% | 9.5 | 47.9% |
| WTSD | 30.3% [29.0%, 31.5%] | 1,551 [754, 162,755] | 50.7% | 19.5 | 30.6% |
| W$SD | 51.9% [48.8%, 54.9%] | 225 [115, 162,755] | 50.1% | 56.1 | 48.8% |

Concentration = alpha + beta (pseudo-opportunities). Synthetic: mean over replicates.

## B. Estimating a new player's stats (leave one player out)

RMSE against the true rate (noise-corrected, percentage points, mean over the ten stats); stats where the human prior is better (95% interval above 0):

| Hands | Players | Uniform | Synthetic | Human | Beats uniform | Beats synthetic |
|---|---|---|---|---|---|---|
| 25 | 13 | 13.83 | 10.24 | 4.82 | 10/10 | 8/10 |
| 50 | 13 | 12.47 | 9.38 | 4.73 | 10/10 | 8/10 |
| 100 | 13 | 10.78 | 8.12 | 4.62 | 10/10 | 7/10 |
| 200 | 13 | 8.49 | 6.86 | 4.41 | 7/10 | 5/10 |
| 500 | 11 | 5.94 | 5.27 | 3.34 | 8/10 | 7/10 |
| 1,000 | 8 | 4.93 | 4.53 | 3.07 | 5/10 | 5/10 |

B3: the human prior on the synthetic held-out bots (t-intervals over replicates):

| Hands | Uniform | Synthetic | Human | Stats where human is worse than uniform |
|---|---|---|---|---|
| 25 | 14.62 | 10.64 | 16.50 | 6/10 |
| 50 | 13.02 | 9.64 | 15.85 | 7/10 |
| 100 | 11.11 | 8.48 | 14.78 | 8/10 |
| 200 | 9.12 | 7.28 | 13.22 | 8/10 |
| 500 | 6.66 | 5.74 | 10.58 | 8/10 |
| 1,000 | 5.14 | 4.66 | 8.47 | 8/10 |

B4: refit on k players vs uniform (MSE difference x 10^4, positive = refit better): the smallest over n of the mean over the ten stats (the decision rule), and the worst stat:

| k | Humans: mean | Humans: worst stat | Bots: mean | Bots: worst stat |
|---|---|---|---|---|
| 2 | +5.34 | Fold to 3-bet -35.63 | -120.89 | C-bet -349.50 |
| 3 | +13.12 | Aggression share -6.99 | -43.30 | C-bet -184.46 |
| 5 | +16.03 | VPIP -1.64 | -5.78 | C-bet -92.69 |
| 8 | +20.24 | VPIP -0.60 | +2.98 | C-bet -59.26 |
| 12 | +21.86 | VPIP -0.10 | +4.41 | C-bet -45.55 |
| 30 | - | - | +7.23 | C-bet -31.12 |
| 100 | - | - | +7.65 | C-bet -26.23 |
| 300 | - | - | +7.78 | C-bet -24.70 |

## C. The synthetic-trained classifier on these players

| Hands | Bots flagged | Players flagged | Control flagged | Pluribus flagged | Bots: accuracy | Players: top probability | Players' labels (nit/TAG/LAG/station/maniac) |
|---|---|---|---|---|---|---|---|
| 25 | 5.7% | 52% | 4% | 77% | 73.9% | 0.63 | 31% / 31% / 33% / 3% / 1% |
| 50 | 4.5% | 76% | 2% | 100% | 81.4% | 0.65 | 33% / 38% / 28% / 0% / 0% |
| 100 | 4.4% | 95% | 3% | 90% | 86.4% | 0.65 | 25% / 44% / 31% / 0% / 0% |
| 200 | 3.8% | 100% | 6% | 100% | 90.1% | 0.65 | 19% / 49% / 32% / 0% / 0% |
| 500 | 5.8% | 100% | 24% | 100% | 93.5% | 0.67 | 10% / 58% / 33% / 0% / 0% |
| 1,000 | 5.6% | 100% | 28% | 100% | 94.8% | 0.70 | 6% / 58% / 35% / 0% / 0% |
| 2,000 | 2.8% | 100% | 56% | 100% | 95.7% | 0.71 | 2% / 64% / 33% / 0% / 0% |

C3 consistency (reference: 1,000 hands; gap = mean top probability - agreement with the reference label):

| Hands | Bots: gap | Players: gap | Players: expected gap | Difference | Control: difference |
|---|---|---|---|---|---|
| 25 | +0.032 | +0.217 | +0.078 | +0.139 [+0.080, +0.210] | -0.008 [-0.059, +0.051] |
| 50 | +0.008 | +0.182 | +0.049 | +0.133 [+0.101, +0.164] | -0.002 [-0.069, +0.075] |
| 100 | -0.012 | +0.111 | -0.012 | +0.122 [+0.041, +0.211] | +0.027 [-0.038, +0.094] |
| 200 | -0.029 | +0.027 | -0.053 | +0.080 [-0.024, +0.216] | -0.002 [-0.076, +0.077] |
| 500 | -0.040 | +0.000 | -0.070 | +0.071 [-0.069, +0.249] | -0.021 [-0.075, +0.025] |
| 1,000 | -0.050 | +0.006 | -0.095 | +0.101 [-0.009, +0.245] | +0.072 [-0.019, +0.176] |

## D. Next-action log-loss on these players' decisions

| Hands observed | Decisions | Trained on bots: situation | Trained on bots: + profile | Trained on the other players: situation | Trained on the other players: + profile | Synthetic - human (situation) | ECE synthetic / human |
|---|---|---|---|---|---|---|---|
| 0 | 1,968 | 0.726 | 0.721 | 0.620 | 0.621 | +0.106 [+0.085, +0.125] | 0.046 / 0.013 |
| 25 | 1,993 | 0.733 | 0.725 | 0.635 | 0.636 | +0.097 [+0.082, +0.110] | 0.044 / 0.015 |
| 100 | 1,989 | 0.727 | 0.720 | 0.648 | 0.649 | +0.079 [+0.062, +0.095] | 0.039 / 0.027 |
| 500 | 1,788 | 0.723 | 0.711 | 0.630 | 0.634 | +0.093 [+0.072, +0.112] | 0.038 / 0.023 |
| 1,500 | 1,262 | 0.707 | 0.680 | 0.591 | 0.605 | +0.116 [+0.103, +0.129] | 0.047 / 0.026 |

Control:

| Hands observed | Trained on bots: situation | Trained on bots: + profile | Trained on the other players: situation | Trained on the other players: + profile |
|---|---|---|---|---|
| 0 | 0.747 | 0.746 | 0.784 | 0.787 |
| 25 | 0.753 | 0.653 | 0.790 | 0.679 |
| 100 | 0.741 | 0.644 | 0.767 | 0.663 |
| 500 | 0.730 | 0.638 | 0.742 | 0.664 |
| 1,500 | 0.657 | 0.550 | 0.650 | 0.574 |

Run time 529 s.
