# P2.11, post hoc: F1's sampled SDs against their sampling error

Seed 2026, commit 43e466b. Not pre-registered. Sample SDs of 100,000 units (P3.3) have a relative SE of about sqrt((kurtosis - 1) / (4n)); z = (sampled / exact - 1) / that SE.

1 of 66 estimates outside F1's band [0.97, 1.03]. Over the 65 with variance: max |z| 2.92, share |z| > 1.96 5%, mean z -0.28, SD of z 1.03; relative SE median 0.0043, max 0.0580; 2 estimates have a 95% sampling interval wider than the band.

## Largest |z|

| Estimate | Sampled / exact | Kurtosis | Relative SE | z |
|---|---|---|---|---|
| 20 bb, Nash hero vs TAG, duplicate + all-in EV | 0.972 | 39.0 | 0.0098 | -2.92 |
| 5 bb, Nash hero vs TAG, duplicate + all-in EV | 0.994 | 3.9 | 0.0027 | -2.30 |
| 5 bb, best-response hero vs nit, duplicate + all-in EV | 0.993 | 6.7 | 0.0038 | -1.97 |
| 10 bb, Nash hero vs Nash, all-in EV | 0.992 | 7.5 | 0.0040 | -1.94 |
| 5 bb, best-response hero vs maniac, all-in EV | 0.996 | 2.7 | 0.0020 | -1.82 |
| 5 bb, best-response hero vs LAG, all-in EV | 1.004 | 3.0 | 0.0022 | +1.76 |
| 10 bb, best-response hero vs TAG, all-in EV | 1.007 | 7.8 | 0.0041 | +1.68 |
| 5 bb, best-response hero vs nit, all-in EV | 0.992 | 9.3 | 0.0045 | -1.67 |

## Simulated sampling distribution (4,000 samples of 100,000)

| Estimate | Observed | 2.5% | median | 97.5% | P(<= observed) | P(outside band) |
|---|---|---|---|---|---|---|
| 5 bb, Nash hero vs Nash, duplicate + all-in EV | 0.957 | 0.934 | 1.000 | 1.067 | 0.105 | 0.392 |
| 10 bb, Nash hero vs Nash, duplicate + all-in EV | 1.022 | 0.887 | 1.001 | 1.117 | 0.641 | 0.617 |

Run time 16 s.
