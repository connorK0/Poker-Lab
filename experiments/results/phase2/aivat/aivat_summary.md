# P2.11: AIVAT in Leduc and push/fold, exact variances

Seed 2026, commit 695cb26. Leduc: 50 bots (10 per archetype), hero = the equilibrium or a restricted Nash response (p = 0.9) to the bot. Ratio = variance of the plain result / variance of the estimator, per hand: how many times fewer hands the estimator needs for the same interval.

## Leduc self-play (Burch et al. 2018, Table 1)

| Estimator | SD per hand (chips) | Paper's SD | Bias |
|---|---|---|---|
| Chips (the plain result) | 3.4973 | 3.513 | +0.0e+00 |
| Duplicate dealing | 2.0695 |  | +0.0e+00 |
| Chance terms only (MIVAT) | 1.4632 | 2.327 | +5.2e-18 |
| Chance terms + imaginary observations | 1.9328 | 1.928 | -4.5e-17 |
| AIVAT (every hand's cards shown) | 0.0000 | 0.00643 | +1.3e-17 |
| AIVAT + duplicate dealing | 0.0000 |  | +1.3e-17 |
| AIVAT, opponent's strategy known too | 0.0000 | 0.00377 | +4.9e-17 |
| Hero's view (folded cards stay hidden) | 3.3999 |  | -6.9e-18 |
| Hero's view + public-card terms (biased) | 2.4489 |  | +5.7e-03 |

## Leduc bots, equilibrium hero, equilibrium value function

| Estimator | Ratio: median [IQR] | min | max | worse than chips | max abs bias | median hands for +/-0.1 |
|---|---|---|---|---|---|---|
| Chips (the plain result) | 1.00x [1.00x, 1.00x] | 1.00x | 1.00x | 0% | 3.3e-16 | 6,332 |
| Duplicate dealing | 2.05x [1.82x, 2.32x] | 1.61x | 3.09x | 0% | 3.3e-16 | 2,646 |
| Chance terms only (MIVAT) | 3.62x [3.28x, 3.92x] | 2.91x | 4.51x | 0% | 2.2e-16 | 1,605 |
| Chance terms + imaginary observations | 2.26x [1.90x, 2.88x] | 1.75x | 3.57x | 0% | 3.3e-16 | 2,628 |
| AIVAT (every hand's cards shown) | 27x [19x, 32x] | 10x | 39x | 0% | 2.2e-16 | 282 |
| AIVAT + duplicate dealing | 28x [19x, 33x] | 10x | 42x | 0% | 3.3e-16 | 268 |
| AIVAT, opponent's strategy known too | 52x [36x, 61x] | 12x | 76x | 0% | 2.8e-16 | 146 |
| Hero's view (folded cards stay hidden) | 1.05x [1.05x, 1.05x] | 1.04x | 1.06x | 0% | 2.2e-16 | 6,036 |
| Hero's view + public-card terms (biased) | 1.85x [1.74x, 1.94x] | 1.60x | 2.14x | 0% | 1.9e-02 | 3,711 |

## Leduc bots, exploiter hero, equilibrium value function

| Estimator | Ratio: median [IQR] | min | max | worse than chips | max abs bias | median hands for +/-0.1 |
|---|---|---|---|---|---|---|
| Chips (the plain result) | 1.00x [1.00x, 1.00x] | 1.00x | 1.00x | 0% | 4.4e-16 | 8,074 |
| Duplicate dealing | 2.95x [2.11x, 3.50x] | 1.14x | 4.87x | 0% | 6.7e-16 | 3,622 |
| Chance terms only (MIVAT) | 3.40x [2.79x, 3.78x] | 1.38x | 4.35x | 0% | 6.7e-16 | 3,288 |
| Chance terms + imaginary observations | 1.86x [1.76x, 2.22x] | 1.59x | 3.07x | 0% | 6.7e-16 | 4,018 |
| AIVAT (every hand's cards shown) | 4.30x [4.00x, 4.65x] | 3.76x | 5.94x | 0% | 6.7e-16 | 1,907 |
| AIVAT + duplicate dealing | 5.33x [4.62x, 5.80x] | 3.79x | 8.09x | 0% | 8.9e-16 | 1,569 |
| AIVAT, opponent's strategy known too | 5.28x [4.43x, 7.33x] | 3.40x | 17x | 0% | 6.7e-16 | 1,624 |
| Hero's view (folded cards stay hidden) | 1.04x [1.04x, 1.05x] | 1.00x | 1.05x | 0% | 4.4e-16 | 7,774 |
| Hero's view + public-card terms (biased) | 1.54x [1.35x, 1.74x] | 1.18x | 2.34x | 0% | 5.8e-02 | 5,386 |

## Leduc bots, equilibrium hero, oracle value function

| Estimator | Ratio: median [IQR] | min | max | worse than chips | max abs bias | median hands for +/-0.1 |
|---|---|---|---|---|---|---|
| Chips (the plain result) | 1.00x [1.00x, 1.00x] | 1.00x | 1.00x | 0% | 3.3e-16 | 6,332 |
| Duplicate dealing | 2.05x [1.82x, 2.32x] | 1.61x | 3.09x | 0% | 3.3e-16 | 2,646 |
| Chance terms only (MIVAT) | 4.58x [4.22x, 5.07x] | 3.70x | 7.34x | 0% | 2.2e-16 | 1,252 |
| Chance terms + imaginary observations | 2.28x [1.92x, 3.00x] | 1.76x | 3.94x | 0% | 4.4e-16 | 2,598 |
| AIVAT (every hand's cards shown) | 45x [35x, 57x] | 28x | 85x | 0% | 2.2e-16 | 161 |
| AIVAT + duplicate dealing | 45x [35x, 57x] | 28x | 85x | 0% | 3.3e-16 | 161 |
| AIVAT, opponent's strategy known too | no variance [no variance, no variance] | no variance | no variance | 0% | 2.2e-16 | 0 |
| Hero's view (folded cards stay hidden) | 1.09x [1.08x, 1.10x] | 1.06x | 1.14x | 0% | 3.3e-16 | 5,782 |
| Hero's view + public-card terms (biased) | 2.31x [2.20x, 2.40x] | 1.95x | 2.54x | 0% | 4.0e-02 | 2,730 |

## Leduc bots, exploiter hero, oracle value function

| Estimator | Ratio: median [IQR] | min | max | worse than chips | max abs bias | median hands for +/-0.1 |
|---|---|---|---|---|---|---|
| Chips (the plain result) | 1.00x [1.00x, 1.00x] | 1.00x | 1.00x | 0% | 4.4e-16 | 8,074 |
| Duplicate dealing | 2.95x [2.11x, 3.50x] | 1.14x | 4.87x | 0% | 6.7e-16 | 3,622 |
| Chance terms only (MIVAT) | 7.04x [5.11x, 10x] | 1.98x | 14x | 0% | 6.7e-16 | 1,505 |
| Chance terms + imaginary observations | 2.29x [1.95x, 2.89x] | 1.73x | 3.59x | 0% | 6.7e-16 | 3,670 |
| AIVAT (every hand's cards shown) | 9.34x [7.06x, 12x] | 4.62x | 24x | 0% | 6.7e-16 | 1,136 |
| AIVAT + duplicate dealing | 9.34x [7.06x, 12x] | 4.62x | 24x | 0% | 1.1e-15 | 1,136 |
| AIVAT, opponent's strategy known too | no variance [no variance, no variance] | no variance | no variance | 0% | 6.7e-16 | 0 |
| Hero's view (folded cards stay hidden) | 1.05x [1.04x, 1.06x] | 1.02x | 1.07x | 0% | 6.7e-16 | 7,763 |
| Hero's view + public-card terms (biased) | 1.77x [1.58x, 2.03x] | 1.44x | 2.65x | 0% | 2.1e-01 | 4,732 |

## Value function quality, equilibrium hero (median ratio over bots)

| Value function | Exploitability | AIVAT (every hand's cards shown) | Hero's view (folded cards stay hidden) | Hero's view + public-card terms (biased) |
|---|---|---|---|---|
| uniform | 2.3736 | 9.12x | 1.02x | 1.52x |
| cfr+_1 | 2.3736 | 9.12x | 1.02x | 1.52x |
| cfr+_3 | 1.7935 | 8.52x | 1.00x | 1.66x |
| cfr+_10 | 0.6104 | 13x | 1.05x | 2.06x |
| cfr+_30 | 0.0818 | 29x | 1.06x | 1.92x |
| cfr+_100 | 0.0134 | 30x | 1.05x | 1.83x |
| cfr+_300 | 0.0023 | 30x | 1.05x | 1.84x |
| cfr+_1000 | 0.0003 | 30x | 1.05x | 1.83x |
| equilibrium | 0.0000 | 27x | 1.05x | 1.85x |

## Value function quality, exploiter hero (median ratio over bots)

| Value function | Exploitability | AIVAT (every hand's cards shown) | Hero's view (folded cards stay hidden) | Hero's view + public-card terms (biased) |
|---|---|---|---|---|
| uniform | 2.3736 | 3.95x | 1.04x | 1.50x |
| cfr+_1 | 2.3736 | 3.95x | 1.04x | 1.50x |
| cfr+_3 | 1.7935 | 4.11x | 1.03x | 1.54x |
| cfr+_10 | 0.6104 | 6.18x | 1.04x | 1.67x |
| cfr+_30 | 0.0818 | 5.09x | 1.04x | 1.56x |
| cfr+_100 | 0.0134 | 4.65x | 1.04x | 1.54x |
| cfr+_300 | 0.0023 | 4.52x | 1.04x | 1.54x |
| cfr+_1000 | 0.0003 | 4.37x | 1.04x | 1.54x |
| equilibrium | 0.0000 | 4.30x | 1.04x | 1.54x |

## Sampled check

| Estimator | z-scores | mean z | max abs z | share abs z > 1.96 | t-test p | SD sampled / exact (median) |
|---|---|---|---|---|---|---|
| Chips (the plain result) | 20 | -0.04 | 2.15 | 5% | 0.85 | 0.999 |
| Duplicate dealing | 20 | +0.11 | 2.71 | 5% | 0.67 | 1.000 |
| Chance terms only (MIVAT) | 20 | +0.15 | 3.04 | 5% | 0.56 | 1.001 |
| Chance terms + imaginary observations | 20 | -0.04 | 2.20 | 10% | 0.88 | 0.999 |
| AIVAT (every hand's cards shown) | 20 | +0.17 | 3.05 | 5% | 0.51 | 1.006 |
| AIVAT + duplicate dealing | 20 | +0.31 | 3.42 | 15% | 0.31 | 1.003 |
| AIVAT, opponent's strategy known too | 20 | -0.18 | 2.57 | 20% | 0.54 | 1.006 |
| Hero's view (folded cards stay hidden) | 20 | -0.08 | 2.24 | 5% | 0.74 | 0.999 |
| Hero's view + public-card terms (biased) | 20 | +0.02 | 2.35 | 10% | 0.95 | 1.000 |

## Push/fold (median over the five archetype bots)

| Stack | Hero | Chips (the plain result) | Duplicate dealing | AIVAT (every hand's cards shown) | AIVAT + duplicate dealing | Hero's view (folded cards stay hidden) |
|---|---|---|---|---|---|---|
| 5 | nash | 1,737 (1.00x) | 578 (3.01x) | 60 (29x) | 53 (33x) | 1,508 (1.14x) |
| 5 | best_response | 1,722 (1.00x) | 690 (2.72x) | 181 (13x) | 173 (13x) | 1,797 (1.15x) |
| 10 | nash | 4,329 (1.00x) | 1,209 (3.26x) | 45 (70x) | 47 (67x) | 3,401 (1.17x) |
| 10 | best_response | 4,380 (1.00x) | 1,845 (2.44x) | 444 (12x) | 349 (10x) | 3,405 (1.20x) |
| 20 | nash | 10,796 (1.00x) | 4,260 (2.25x) | 206 (22x) | 215 (25x) | 8,654 (1.16x) |
| 20 | best_response | 11,042 (1.00x) | 5,737 (2.55x) | 2,338 (6.95x) | 1,901 (11x) | 9,624 (1.17x) |

P3.3 cross-check: sampled / exact SD over 66 estimates: 0.957 to 1.022.

## Predictions

| | Prediction | Result | Holds |
|---|---|---|---|
| A1 | Every estimator but the approximate hero's view is unbiased (|bias| < 1e-9) | largest |bias| 1.4e-13 | yes |
| A2 | Sampled hands agree with the exact moments: no |z| > 4, mean-z p >= 0.01, median sampled/exact SD in [0.97, 1.03] | max |z| 3.42, min p 0.31, SD ratios 0.999-1.006 | yes |
| B1 | AIVAT, equilibrium hero: median ratio >= 15x and every bot >= 5x | median 27x, min 10x | yes |
| B2 | AIVAT, exploiting hero: median ratio between 3x and 8x | 4.30x | yes |
| B3 | Oracle values beat equilibrium values for the equilibrium hero by >= 1.4x | 1.66x | yes |
| C1 | Hero's view (folded cards hidden): median ratio < 1.15x for both heroes | 1.05x / 1.04x | yes |
| C2 | Approximate hero's view: median ratio in [1.4x, 2.5x] for both heroes, and its bias reaches >= 0.02 chips/hand against some bot (exploiting hero) | 1.85x / 1.54x; max |bias| 0.058 | yes |
| D1 | Uniform-play values still give AIVAT a median ratio >= 4x (equilibrium hero) | 9.12x | yes |
| D2 | From 30 CFR+ iterations on, AIVAT's median ratio is within 20% of the equilibrium values' (equilibrium hero) | 29x, 30x, 30x, 30x vs 27x | yes |
| E1 | Duplicate dealing adds < 25% on top of AIVAT (equilibrium hero) | 1.02x | yes |
| F1 | Push/fold: exact SDs of all-in EV and duplicate + all-in EV within 3% of P3.3's sampled SDs in every matchup | 66 estimates, 0.957-1.022 | NO |
| F2 | Push/fold, Nash hero: AIVAT needs >= 4x fewer hands than all-in EV at every stack (median over bots) | 5 bb 29x, 10 bb 70x, 20 bb 22x | yes |
| F3 | Push/fold: hero's view gains < 1.3x over all-in EV at every stack, both heroes | 1.14x-1.20x | yes |

Run time 19 s.
