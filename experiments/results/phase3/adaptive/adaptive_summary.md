# P3.9: exploiting push/fold opponents that adapt

Seed 2026, commit 10aff6e60cead87b1bf4490ae179d2ffa69382d2 (dirty: False), 19097 s. 5 replicates x 30 starting bots x 4 kinds; sessions of 2,000 hands at 10 bb.

## Value and gain over equilibrium play (bb/hand, 95% t-intervals across replicates)

### Static bots (widths moved 0.00 against the equilibrium hero)

| Hero | Value | Gain over equilibrium | Loses to equilibrium | Exploitability | Fallback rate, hands 1,000+ |
|---|---|---|---|---|---|
| Equilibrium | +0.0859 [+0.0811, +0.0908] | +0.0000 [+0.0000, +0.0000] | 0% | -0.000 |  |
| Fit once, then keep | +0.1605 [+0.1508, +0.1702] | +0.0746 [+0.0670, +0.0821] | 1% | 0.131 |  |
| Refit on all hands | +0.1682 [+0.1584, +0.1779] | +0.0823 [+0.0746, +0.0899] | 0% | 0.149 | 4% |
| Refit on recent hands | +0.1675 [+0.1569, +0.1781] | +0.0815 [+0.0730, +0.0900] | 0% | 0.145 | 3% |
| Bayes best response | +0.1712 [+0.1602, +0.1823] | +0.0853 [+0.0763, +0.0944] | 1% | 0.164 | 0% |
| Clairvoyant (reference) | +0.1739 [+0.1627, +0.1851] | +0.0880 [+0.0788, +0.0972] | 0% | 0.168 |  |

### Slow adapters (widths moved 0.45 against the equilibrium hero)

| Hero | Value | Gain over equilibrium | Loses to equilibrium | Exploitability | Fallback rate, hands 1,000+ |
|---|---|---|---|---|---|
| Equilibrium | +0.0203 [+0.0172, +0.0233] | +0.0000 [+0.0000, +0.0000] | 0% | -0.000 |  |
| Fit once, then keep | -0.0300 [-0.0370, -0.0230] | -0.0503 [-0.0573, -0.0433] | 78% | 0.060 |  |
| Refit on all hands | +0.0214 [+0.0184, +0.0244] | +0.0011 [+0.0006, +0.0016] | 31% | 0.011 | 64% |
| Refit on recent hands | +0.0224 [+0.0194, +0.0254] | +0.0022 [+0.0018, +0.0025] | 16% | 0.013 | 6% |
| Bayes best response | +0.0185 [+0.0158, +0.0212] | -0.0018 [-0.0021, -0.0015] | 73% | 0.023 | 0% |
| Clairvoyant (reference) | +0.0234 [+0.0204, +0.0264] | +0.0031 [+0.0026, +0.0036] | 11% | 0.013 |  |

### Fast adapters (widths moved 0.47 against the equilibrium hero)

| Hero | Value | Gain over equilibrium | Loses to equilibrium | Exploitability | Fallback rate, hands 1,000+ |
|---|---|---|---|---|---|
| Equilibrium | +0.0213 [+0.0185, +0.0241] | +0.0000 [+0.0000, +0.0000] | 0% | -0.000 |  |
| Fit once, then keep | +0.0135 [+0.0075, +0.0195] | -0.0078 [-0.0118, -0.0038] | 53% | 0.008 |  |
| Refit on all hands | +0.0201 [+0.0171, +0.0232] | -0.0012 [-0.0015, -0.0008] | 79% | 0.003 | 49% |
| Refit on recent hands | +0.0193 [+0.0158, +0.0227] | -0.0021 [-0.0028, -0.0013] | 91% | 0.006 | 22% |
| Bayes best response | +0.0164 [+0.0126, +0.0201] | -0.0049 [-0.0060, -0.0038] | 98% | 0.012 | 0% |
| Clairvoyant (reference) | +0.0253 [+0.0221, +0.0284] | +0.0040 [+0.0035, +0.0044] | 4% | 0.017 |  |

### Adapt calls only (widths moved 0.18 against the equilibrium hero)

| Hero | Value | Gain over equilibrium | Loses to equilibrium | Exploitability | Fallback rate, hands 1,000+ |
|---|---|---|---|---|---|
| Equilibrium | +0.0432 [+0.0362, +0.0502] | +0.0000 [+0.0000, +0.0000] | 0% | -0.000 |  |
| Fit once, then keep | +0.0583 [+0.0494, +0.0672] | +0.0151 [+0.0101, +0.0201] | 14% | 0.080 |  |
| Refit on all hands | +0.0677 [+0.0606, +0.0748] | +0.0245 [+0.0228, +0.0263] | 3% | 0.090 | 4% |
| Refit on recent hands | +0.0655 [+0.0579, +0.0730] | +0.0223 [+0.0200, +0.0245] | 7% | 0.086 | 4% |
| Bayes best response | +0.0687 [+0.0613, +0.0760] | +0.0254 [+0.0237, +0.0272] | 5% | 0.102 | 0% |
| Clairvoyant (reference) | +0.0720 [+0.0647, +0.0793] | +0.0287 [+0.0268, +0.0307] | 1% | 0.108 |  |

## Predictions

* A1 (refit keeps at least 85% of the clairvoyant gain against static bots): 94%; holds.
* B1 (adaptation at least halves the equilibrium hero's value): slow ratio 0.24 [0.21, 0.27]; fast ratio 0.25 [0.22, 0.27]; holds.
* B2 (the clairvoyant gains at most 0.01 against adapters): slow +0.0031 [+0.0026, +0.0036]; fast +0.0040 [+0.0035, +0.0044]; holds.
* C1 (the Bayes best response loses against fast adapters): -0.0049 [-0.0060, -0.0038]; holds.
* C2 (fitting once loses against slow adapters): -0.0503 [-0.0573, -0.0433]; holds.
* C3 (refit within 0.005 of equilibrium against adapters): slow +0.0011 [+0.0006, +0.0016]; fast -0.0012 [-0.0015, -0.0008]; holds.
* C4 (refit gains against calls-only adapters): +0.0245 [+0.0228, +0.0263]; holds.
* D1 (refit falls back at least 30% against adapters, at most 10% against static bots): slow 64% [63%, 66%]; fast 49% [47%, 51%]; static 4% [3%, 4%]; holds.
