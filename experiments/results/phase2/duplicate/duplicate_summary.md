# P2.11b: duplicate dealing for n players, no-limit hold'em

Seed 2026, commit 0fb6e53.

10 replicates of 12 bots per archetype; 2,000 deals per lineup, each played at n tables with the seats rotated; blinds 1/2, 100 bb stacks, 150 equity samples. Ratio = sigma^2 / (n s_u^2): how many times fewer hands duplicate dealing needs than independent dealing for the same interval (above 1: it pays). rho: mean correlation of a player's results at two tables of one deal.

## By lineup kind (players pooled over lineups and replicates)

| Lineup kind | Lineups played | Players | Ratio: median [IQR] | 95% CI of the median | Below 1 | rho: median | rho by offset k = 1.. | SD per hand (bb) | Median hands for +/-5 bb/100: naive | duplicate |
|---|---|---|---|---|---|---|---|---|---|---|
| Six copies of one bot | 50 | 300 | 1.53x [1.42x, 1.64x] | [1.49x, 1.57x] | 0% | -0.069 | -0.067, -0.071, -0.067 | 19.27 | 570,490 | 388,778 |
| Six bots of one archetype | 50 | 300 | 1.44x [1.37x, 1.52x] | [1.43x, 1.47x] | 0% | -0.061 | -0.059, -0.064, -0.062 | 19.92 | 610,030 | 421,662 |
| Six bots, random mix | 100 | 600 | 1.30x [1.21x, 1.40x] | [1.28x, 1.32x] | 0% | -0.046 | -0.040, -0.040, -0.041 | 24.74 | 940,565 | 713,764 |
| Three bots, random mix | 100 | 300 | 1.25x [1.16x, 1.35x] | [1.22x, 1.28x] | 1% | -0.100 | -0.100 | 14.42 | 319,506 | 276,703 |
| Two bots, random mix | 200 | 400 | 1.29x [1.17x, 1.44x] | [1.24x, 1.32x] | 1% | -0.224 | -0.224 | 8.29 | 105,680 | 86,536 |

## By lineup kind and the player's archetype

| Lineup kind | Archetype | Players | Ratio: median [IQR] | 95% CI of the median | rho: median | rho by offset | SD per hand (bb) |
|---|---|---|---|---|---|---|---|
| Six copies of one bot | Nit | 60 | 1.34x [1.20x, 1.49x] | [1.27x, 1.42x] | -0.051 | -0.044, -0.040, -0.033 | 3.50 |
| Six copies of one bot | TAG | 60 | 1.65x [1.52x, 1.77x] | [1.55x, 1.72x] | -0.079 | -0.067, -0.088, -0.079 | 13.64 |
| Six copies of one bot | LAG | 60 | 1.53x [1.48x, 1.61x] | [1.50x, 1.59x] | -0.069 | -0.072, -0.071, -0.071 | 20.49 |
| Six copies of one bot | Calling station | 60 | 1.57x [1.44x, 1.68x] | [1.47x, 1.67x] | -0.073 | -0.067, -0.073, -0.077 | 24.47 |
| Six copies of one bot | Maniac | 60 | 1.51x [1.46x, 1.57x] | [1.48x, 1.56x] | -0.068 | -0.069, -0.068, -0.065 | 49.83 |
| Six bots of one archetype | Nit | 60 | 1.31x [1.20x, 1.45x] | [1.26x, 1.40x] | -0.047 | -0.031, -0.048, -0.046 | 4.78 |
| Six bots of one archetype | TAG | 60 | 1.43x [1.35x, 1.51x] | [1.37x, 1.49x] | -0.060 | -0.058, -0.063, -0.070 | 10.87 |
| Six bots of one archetype | LAG | 60 | 1.45x [1.39x, 1.52x] | [1.42x, 1.49x] | -0.062 | -0.062, -0.064, -0.059 | 20.25 |
| Six bots of one archetype | Calling station | 60 | 1.52x [1.44x, 1.61x] | [1.49x, 1.58x] | -0.068 | -0.068, -0.074, -0.066 | 27.67 |
| Six bots of one archetype | Maniac | 60 | 1.46x [1.42x, 1.50x] | [1.43x, 1.49x] | -0.063 | -0.060, -0.066, -0.066 | 47.58 |
| Six bots, random mix | Nit | 130 | 1.18x [1.12x, 1.25x] | [1.16x, 1.20x] | -0.031 | -0.027, -0.027, -0.028 | 13.54 |
| Six bots, random mix | TAG | 125 | 1.27x [1.21x, 1.33x] | [1.24x, 1.29x] | -0.043 | -0.039, -0.043, -0.040 | 17.91 |
| Six bots, random mix | LAG | 105 | 1.34x [1.29x, 1.41x] | [1.32x, 1.36x] | -0.050 | -0.045, -0.047, -0.067 | 25.13 |
| Six bots, random mix | Calling station | 116 | 1.43x [1.35x, 1.51x] | [1.40x, 1.46x] | -0.060 | -0.059, -0.058, -0.050 | 38.11 |
| Six bots, random mix | Maniac | 124 | 1.30x [1.22x, 1.40x] | [1.28x, 1.34x] | -0.046 | -0.040, -0.041, -0.042 | 41.19 |
| Three bots, random mix | Nit | 62 | 1.17x [1.09x, 1.23x] | [1.14x, 1.20x] | -0.072 | -0.072 | 8.13 |
| Three bots, random mix | TAG | 54 | 1.23x [1.16x, 1.32x] | [1.19x, 1.29x] | -0.094 | -0.094 | 11.41 |
| Three bots, random mix | LAG | 66 | 1.27x [1.19x, 1.36x] | [1.24x, 1.31x] | -0.107 | -0.107 | 18.29 |
| Three bots, random mix | Calling station | 70 | 1.32x [1.26x, 1.46x] | [1.29x, 1.39x] | -0.122 | -0.122 | 17.22 |
| Three bots, random mix | Maniac | 48 | 1.22x [1.09x, 1.32x] | [1.14x, 1.29x] | -0.090 | -0.090 | 27.00 |
| Two bots, random mix | Nit | 86 | 1.21x [1.10x, 1.31x] | [1.15x, 1.23x] | -0.171 | -0.171 | 4.63 |
| Two bots, random mix | TAG | 89 | 1.28x [1.18x, 1.42x] | [1.22x, 1.34x] | -0.217 | -0.217 | 7.83 |
| Two bots, random mix | LAG | 68 | 1.30x [1.21x, 1.42x] | [1.24x, 1.34x] | -0.232 | -0.232 | 11.06 |
| Two bots, random mix | Calling station | 75 | 1.42x [1.31x, 1.53x] | [1.39x, 1.47x] | -0.298 | -0.298 | 6.27 |
| Two bots, random mix | Maniac | 82 | 1.24x [1.13x, 1.43x] | [1.19x, 1.34x] | -0.195 | -0.195 | 17.61 |

## Mixed lineups by the number of players

| Players | Median ratio | 95% CI | Median rho | Median SD per hand (bb) |
|---|---|---|---|---|
| 2 | 1.29x | [1.24x, 1.32x] | -0.224 | 8.29 |
| 3 | 1.25x | [1.22x, 1.28x] | -0.100 | 14.42 |
| 6 | 1.30x | [1.28x, 1.32x] | -0.046 | 24.74 |

Calling check: six always-call agents, 200 deals: largest |per-deal mean| 0 chips (per-hand SD 2.19 bb).
Heads-up push/fold reference (Step 28, P3.3's bridge): median over the matchups against the bots of naive / duplicate hands needed: 2.07x at 5 bb, 2.30x at 10 bb, 2.01x at 20 bb.

## Predictions

| | Prediction | Result | Holds |
|---|---|---|---|
| A1 | Calling check: every per-deal mean is exactly 0 chips | 0 | yes |
| B1 | Six bots, random mix: median ratio in [1.15, 1.5] | 1.3 | yes |
| B2 | Six copies of one bot: median ratio in [1.35, 1.9] | 1.53 | yes |
| B3 | The median ratio falls: copies, then one archetype, then the random mix | 1.53, 1.44, 1.3 | yes |
| B4 | At most 5% of the players have a ratio below 1, in each six-player kind | 0, 0.00333, 0.00167 | yes |
| C1 | Random mix: nits have the lowest median ratio of the five archetypes | 1.18, 1.27, 1.34, 1.43, 1.3 | yes |
| C2 | Copies of one bot: nits have the lowest median ratio of the five archetypes | 1.34, 1.65, 1.53, 1.57, 1.51 | yes |
| D1 | Copies of one bot: correlations at offsets 1-3 within 0.03 | -0.0665, -0.0712, -0.0674 | yes |
| D2 | One archetype: correlations at offsets 1-3 within 0.03 | -0.0589, -0.0639, -0.0621 | yes |
| D3 | Random mix: correlations at offsets 1-3 within 0.03 | -0.0398, -0.0405, -0.0413 | yes |
| E1 | Random mixes: median ratio in [1.1, 1.6] at n = 2, 3 and 6 | 1.29, 1.25, 1.3 | yes |
| E2 | Random mixes: median rho at most -0.02 at n = 2, 3 and 6 | -0.224, -0.1, -0.046 | yes |

Run time 1455 s.
