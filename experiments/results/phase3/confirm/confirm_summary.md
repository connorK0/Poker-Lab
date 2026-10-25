# P3.5b: confirming the stability-only fallback on fresh seeds

5 fresh seeds x 30 bots per population, the P3.5 design otherwise unchanged. Gains in bb/hand over equilibrium against each bot's true current strategy; 95% t-intervals across seeds.

## Pre-registered predictions

1. Stability only loses to equilibrium against at most 10% of gear-change bots: 1,000 hands 0.7%, 3,000 hands 0.0%; holds.
2. Stability only earns more than any check against wrong hand orders (difference, 95% interval above 0): pairs first 300 hands +0.028 [+0.023, +0.032], 1,000 hands +0.053 [+0.042, +0.065], 3,000 hands +0.074 [+0.065, +0.083]; connectors first 300 hands +0.080 [+0.070, +0.089], 1,000 hands +0.069 [+0.054, +0.084], 3,000 hands +0.052 [+0.044, +0.061]; holds.
3. Stability only loses to equilibrium against at most 5% of wrong-order bots from 100 hands: max pairs first 2.0%, connectors first 2.0%; holds.

Decision: default fallback_on becomes ("stability",).

## The three rules

| Hands | Population | Oracle | Lower-bound pick (P3.4) | Checked pick | Stability-only fallback (post hoc) | Lower-bound pick (P3.4) loses | Checked pick loses | Stability-only fallback (post hoc) loses |
|---|---|---|---|---|---|---|---|---|
| 10 | Standard (model right) | +0.098 | +0.015 | +0.014 | +0.014 | 2.0% | 1.3% | 1.3% |
| 10 | Pairs first | +0.094 | +0.013 | +0.013 | +0.013 | 3.3% | 3.3% | 3.3% |
| 10 | Connectors first | +0.111 | +0.011 | +0.008 | +0.009 | 8.7% | 5.3% | 8.0% |
| 10 | Gear change | +0.082 | +0.006 | +0.005 | +0.005 | 12.0% | 12.0% | 12.0% |
| 30 | Standard (model right) | +0.098 | +0.053 | +0.051 | +0.052 | 1.3% | 1.3% | 1.3% |
| 30 | Pairs first | +0.094 | +0.044 | +0.040 | +0.043 | 3.3% | 3.3% | 3.3% |
| 30 | Connectors first | +0.111 | +0.032 | +0.021 | +0.027 | 8.7% | 4.0% | 5.3% |
| 30 | Gear change | +0.082 | +0.019 | +0.015 | +0.017 | 17.3% | 14.7% | 14.7% |
| 100 | Standard (model right) | +0.098 | +0.090 | +0.088 | +0.089 | 0.0% | 0.0% | 0.0% |
| 100 | Pairs first | +0.094 | +0.075 | +0.056 | +0.074 | 2.0% | 2.0% | 2.0% |
| 100 | Connectors first | +0.111 | +0.068 | +0.016 | +0.063 | 2.7% | 0.7% | 2.0% |
| 100 | Gear change | +0.082 | +0.024 | +0.016 | +0.016 | 26.7% | 22.0% | 23.3% |
| 300 | Standard (model right) | +0.098 | +0.096 | +0.091 | +0.093 | 0.0% | 0.0% | 0.0% |
| 300 | Pairs first | +0.094 | +0.088 | +0.060 | +0.088 | 0.7% | 0.7% | 0.7% |
| 300 | Connectors first | +0.111 | +0.093 | +0.006 | +0.085 | 0.0% | 0.0% | 0.0% |
| 300 | Gear change | +0.082 | +0.029 | +0.014 | +0.014 | 28.0% | 9.3% | 9.3% |
| 1,000 | Standard (model right) | +0.098 | +0.097 | +0.091 | +0.094 | 0.0% | 0.0% | 0.0% |
| 1,000 | Pairs first | +0.094 | +0.092 | +0.037 | +0.090 | 1.3% | 0.7% | 1.3% |
| 1,000 | Connectors first | +0.111 | +0.102 | +0.000 | +0.069 | 0.0% | 0.0% | 0.0% |
| 1,000 | Gear change | +0.082 | +0.035 | +0.005 | +0.005 | 26.7% | 0.7% | 0.7% |
| 3,000 | Standard (model right) | +0.098 | +0.097 | +0.091 | +0.094 | 0.0% | 0.0% | 0.0% |
| 3,000 | Pairs first | +0.094 | +0.093 | +0.018 | +0.092 | 0.0% | 0.0% | 0.0% |
| 3,000 | Connectors first | +0.111 | +0.105 | +0.000 | +0.052 | 0.0% | 0.0% | 0.0% |
| 3,000 | Gear change | +0.082 | +0.028 | +0.001 | +0.001 | 34.0% | 0.0% | 0.0% |

## Flag rates (either check), a replication of P3.5

| Hands | Standard (model right) | Pairs first | Connectors first | Gear change |
|---|---|---|---|---|
| 10 | 6.0% | 5.3% | 23.3% | 4.0% |
| 30 | 4.0% | 8.0% | 38.0% | 18.0% |
| 100 | 4.7% | 18.7% | 76.7% | 31.3% |
| 300 | 5.3% | 22.0% | 94.0% | 62.7% |
| 1,000 | 8.0% | 43.3% | 100.0% | 91.3% |
| 3,000 | 6.0% | 69.3% | 100.0% | 97.3% |
