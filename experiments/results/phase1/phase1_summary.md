Mean over 5 seeds; brackets are 95% t-intervals across seeds (clipped to the metric's natural range for display). Primary model (fixed before the run): logistic, shrunk stats. Items marked *post hoc* were added after the first full run.

### Hands-to-identify: accuracy on held-out players

| Hands | Logistic, shrunk stats | Boosted trees, shrunk stats | Logistic, raw stats | Boosted trees, raw stats | Logistic, shrunk, no bet sizing (post hoc) |
|---|---|---|---|---|---|
| 25 | 73.9% [72.5, 75.2] | 73.5% [71.2, 75.7] | 73.7% [71.8, 75.6] | 72.5% [70.9, 74.1] | 70.5% [69.5, 71.6] |
| 50 | 81.4% [80.0, 82.7] | 79.7% [76.5, 82.8] | 80.4% [79.1, 81.7] | 78.9% [76.7, 81.0] | 77.6% [76.0, 79.2] |
| 100 | 86.4% [84.1, 88.7] | 84.1% [81.4, 86.8] | 85.3% [83.1, 87.5] | 84.0% [81.1, 87.0] | 82.9% [81.4, 84.4] |
| 200 | 90.1% [89.0, 91.2] | 87.9% [85.9, 90.0] | 88.1% [85.5, 90.7] | 87.6% [85.0, 90.3] | 87.8% [86.7, 89.0] |
| 500 | 93.5% [92.2, 94.8] | 91.0% [89.0, 93.1] | 89.6% [86.1, 93.1] | 91.4% [89.0, 93.8] | 91.8% [90.4, 93.1] |
| 1000 | 94.8% [93.5, 96.1] | 92.5% [89.5, 95.6] | 89.7% [85.7, 93.8] | 93.8% [91.6, 96.0] | 94.6% [93.7, 95.5] |
| 2000 | 95.7% [95.0, 96.5] | 94.4% [91.4, 97.4] | 90.4% [86.9, 93.9] | 94.1% [91.2, 97.1] | 95.5% [93.7, 97.3] |

Log-loss (nats; chance = 1.609; one global temperature):

| Hands | Logistic, shrunk stats | Boosted trees, shrunk stats | Logistic, raw stats | Boosted trees, raw stats | Logistic, shrunk, no bet sizing (post hoc) |
|---|---|---|---|---|---|
| 25 | 0.580 [0.548, 0.612] | 0.612 [0.560, 0.663] | 0.646 [0.602, 0.691] | 0.631 [0.583, 0.679] | 0.679 [0.645, 0.712] |
| 50 | 0.429 [0.400, 0.457] | 0.472 [0.420, 0.525] | 0.447 [0.414, 0.481] | 0.482 [0.429, 0.535] | 0.513 [0.482, 0.543] |
| 100 | 0.322 [0.288, 0.356] | 0.375 [0.332, 0.419] | 0.359 [0.327, 0.391] | 0.375 [0.325, 0.426] | 0.390 [0.361, 0.418] |
| 200 | 0.246 [0.213, 0.280] | 0.297 [0.251, 0.342] | 0.317 [0.281, 0.352] | 0.296 [0.249, 0.342] | 0.300 [0.273, 0.327] |
| 500 | 0.189 [0.156, 0.221] | 0.229 [0.179, 0.279] | 0.300 [0.262, 0.338] | 0.235 [0.181, 0.289] | 0.230 [0.206, 0.254] |
| 1000 | 0.164 [0.134, 0.194] | 0.199 [0.152, 0.246] | 0.296 [0.257, 0.335] | 0.200 [0.151, 0.248] | 0.197 [0.178, 0.216] |
| 2000 | 0.152 [0.122, 0.182] | 0.178 [0.121, 0.234] | 0.295 [0.254, 0.336] | 0.184 [0.130, 0.239] | 0.178 [0.160, 0.196] |

Oracle ceiling (classifier on the true parameters, i.e. infinite hands): accuracy 99.2% [98.1, 100.0] (logistic), 97.3% [96.5, 98.2] (boosted trees). Permutation control (labels shuffled across training players): accuracy 19.5% [13.2, 25.8] vs 20% chance.

### Hands needed to reach an accuracy

| Model | Target | Seeds reaching it | Geometric mean | Range |
|---|---|---|---|---|
| Logistic, shrunk stats | 80% | 5/5 | 44 | 40–49 |
| Logistic, shrunk stats | 90% | 5/5 | 194 | 150–264 |
| Logistic, shrunk stats | 95% | 4/5 | 934 | 630–1320 |
| Boosted trees, shrunk stats | 80% | 5/5 | 54 | 39–91 |
| Boosted trees, shrunk stats | 90% | 5/5 | 433 | 231–1587 |
| Boosted trees, shrunk stats | 95% | 3/5 | 1357 | 1189–1486 |
| Logistic, shrunk, no bet sizing (post hoc) | 80% | 5/5 | 68 | 53–80 |
| Logistic, shrunk, no bet sizing (post hoc) | 90% | 5/5 | 334 | 236–431 |
| Logistic, shrunk, no bet sizing (post hoc) | 95% | 3/5 | 969 | 891–1091 |

### Calibration: expected calibration error (10 bins)

| Hands | Logistic: uncal. | global T | T per size (post hoc) | Trees: uncal. | global T | T per size (post hoc) |
|---|---|---|---|---|---|---|
| 25 | 0.043 [0.031, 0.054] | 0.043 [0.025, 0.062] | 0.030 [0.023, 0.037] | 0.079 [0.058, 0.100] | 0.030 [0.019, 0.040] | 0.030 [0.016, 0.044] |
| 50 | 0.018 [0.014, 0.022] | 0.018 [0.012, 0.024] | 0.018 [0.015, 0.021] | 0.070 [0.047, 0.092] | 0.026 [0.011, 0.041] | 0.024 [0.011, 0.038] |
| 100 | 0.021 [0.012, 0.030] | 0.020 [0.011, 0.028] | 0.019 [0.014, 0.025] | 0.061 [0.042, 0.080] | 0.027 [0.016, 0.037] | 0.025 [0.020, 0.030] |
| 200 | 0.031 [0.027, 0.034] | 0.031 [0.024, 0.037] | 0.015 [0.011, 0.018] | 0.042 [0.029, 0.056] | 0.015 [0.006, 0.023] | 0.016 [0.007, 0.025] |
| 500 | 0.048 [0.039, 0.057] | 0.046 [0.031, 0.062] | 0.016 [0.005, 0.027] | 0.027 [0.011, 0.043] | 0.023 [0.019, 0.027] | 0.018 [0.010, 0.026] |
| 1000 | 0.057 [0.042, 0.071] | 0.057 [0.039, 0.075] | 0.024 [0.007, 0.040] | 0.030 [0.017, 0.043] | 0.032 [0.020, 0.044] | 0.027 [0.010, 0.044] |
| 2000 | 0.056 [0.046, 0.066] | 0.056 [0.044, 0.067] | 0.025 [0.012, 0.038] | 0.036 [0.024, 0.049] | 0.034 [0.017, 0.050] | 0.032 [0.019, 0.044] |

Global temperatures (held-out training players): logistic 0.99 [0.92, 1.06], boosted trees 1.50 [1.34, 1.66] (T > 1: the raw model was overconfident; T < 1: underconfident). Logistic per-size temperatures: 25: 1.24, 50: 1.10, 100: 0.97, 200: 0.78, 500: 0.67, 1000: 0.61, 2000: 0.55.

### Shrinkage ablation (shrunk − raw; negative Δlog-loss = shrinkage helps)

| Hands | Logistic Δacc | Logistic Δlog-loss | Trees Δacc | Trees Δlog-loss |
|---|---|---|---|---|
| 25 | 0.2% [-0.4, 0.8] | -0.066 [-0.084, -0.049] | 1.0% [-0.1, 2.1] | -0.019 [-0.032, -0.007] |
| 50 | 1.0% [-0.4, 2.3] | -0.019 [-0.038, 0.001] | 0.8% [-0.4, 2.0] | -0.010 [-0.020, 0.000] |
| 100 | 1.1% [-0.3, 2.6] | -0.037 [-0.051, -0.024] | 0.1% [-0.8, 1.0] | -0.000 [-0.019, 0.018] |
| 200 | 2.1% [0.2, 3.9] | -0.070 [-0.089, -0.051] | 0.3% [-1.3, 1.9] | 0.001 [-0.019, 0.021] |
| 500 | 3.9% [0.6, 7.3] | -0.112 [-0.133, -0.090] | -0.3% [-1.9, 1.2] | -0.006 [-0.029, 0.018] |
| 1000 | 5.1% [0.2, 10.0] | -0.132 [-0.157, -0.107] | -1.3% [-3.3, 0.7] | -0.000 [-0.014, 0.013] |
| 2000 | 5.3% [1.8, 8.9] | -0.143 [-0.170, -0.116] | 0.3% [-2.4, 2.9] | -0.007 [-0.042, 0.029] |

### Parameter regression, boosted trees: out-of-sample R²

First number: R² vs predicting the population mean. Second (*post hoc*): within-archetype R², i.e. vs predicting the true archetype's mean, which measures what is recovered *beyond* the type (negative = worse than knowing the type).

| Parameter | 25 hands | 100 hands | 500 hands | 2000 hands |
|---|---|---|---|---|
| range_width | 0.81 / 0.02 | 0.93 / 0.62 | 0.97 / 0.84 | 0.98 / 0.90 |
| raise_share | 0.87 / -0.90 | 0.92 / -0.12 | 0.94 / 0.14 | 0.94 / 0.21 |
| three_bet_freq | 0.69 / -0.45 | 0.78 / -0.04 | 0.84 / 0.24 | 0.86 / 0.33 |
| continue_vs_raise | 0.71 / -0.47 | 0.79 / -0.07 | 0.83 / 0.13 | 0.86 / 0.29 |
| cbet_freq | 0.77 / -0.70 | 0.84 / -0.22 | 0.88 / 0.11 | 0.90 / 0.29 |
| fold_to_aggression | 0.72 / -2.78 | 0.85 / -0.98 | 0.91 / -0.14 | 0.94 / 0.14 |
| bluff_freq | 0.78 / -2.10 | 0.86 / -0.90 | 0.90 / -0.35 | 0.92 / -0.15 |
| value_aggression | 0.78 / -0.99 | 0.85 / -0.34 | 0.89 / -0.01 | 0.91 / 0.15 |
| bet_size_pot | 0.87 / 0.64 | 0.96 / 0.90 | 0.98 / 0.96 | 0.99 / 0.97 |

Mean R² over parameters (vs population / within type), boosted trees: 25: 0.78 / -0.86, 100: 0.86 / -0.13, 500: 0.91 / 0.21, 2000: 0.92 / 0.35

Mean R² over parameters (vs population / within type), ridge: 25: 0.70 / -1.58, 100: 0.83 / -0.47, 500: 0.87 / -0.10, 2000: 0.88 / -0.01

Mean R² over parameters (vs population / within type), classify-then-archetype-mean: 25: 0.76 / -0.85, 100: 0.82 / -0.25, 500: 0.84 / -0.07, 2000: 0.84 / -0.03


### Next-action prediction (held-out players; log-loss in nats; profile from the first c hands predicts decisions in hands [c, c+100))

| Hands observed | Population (situation only) | + profile | Oracle: archetype | Oracle: parameters | Reduction vs population | Share of oracle gap closed |
|---|---|---|---|---|---|---|
| 0 | 0.773 [0.769, 0.777] | 0.771 [0.767, 0.774] | 0.679 [0.675, 0.683] | 0.666 [0.663, 0.669] | 0.3% [0.2, 0.4] | 2.3% [1.7, 2.8] |
| 25 | 0.774 [0.771, 0.777] | 0.688 [0.683, 0.692] | 0.681 [0.677, 0.685] | 0.667 [0.662, 0.673] | 11.2% [10.8, 11.6] | 81.1% [77.6, 84.7] |
| 100 | 0.771 [0.764, 0.779] | 0.678 [0.668, 0.688] | 0.682 [0.674, 0.690] | 0.668 [0.659, 0.677] | 12.0% [11.3, 12.8] | 90.3% [87.8, 92.8] |
| 500 | 0.771 [0.760, 0.781] | 0.672 [0.662, 0.683] | 0.679 [0.668, 0.689] | 0.667 [0.657, 0.677] | 12.8% [12.2, 13.3] | 94.5% [93.4, 95.7] |
| 1500 | 0.771 [0.760, 0.783] | 0.674 [0.659, 0.688] | 0.680 [0.668, 0.692] | 0.668 [0.655, 0.681] | 12.6% [12.0, 13.3] | 94.2% [92.6, 95.9] |
