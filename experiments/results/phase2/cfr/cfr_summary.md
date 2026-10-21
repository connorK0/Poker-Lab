# P2.3: CFR convergence on Kuhn and Leduc

Exploitability (NashConv / 2, chips per hand) of the average strategy. Full-width
variants are deterministic; MCCFR rows are geometric means over seeds with 95%
t-intervals on the log scale. Timings (ms per iteration) are wall-clock on the
recording machine and vary between runs; every other number is reproduced exactly
by the seed.

## Kuhn poker

Exact LP game value -0.0555556 (solved in 0.01 s).

| Variant | 10 it. | 100 it. | 1,000 it. | 10,000 it. | ms / it. |
|---|---|---|---|---|---|
| CFR (vanilla) | 9.62e-02 | 2.57e-02 | 7.27e-03 | 2.32e-03 | 0.35 |
| CFR+ | 3.27e-02 | 1.19e-03 | 8.74e-05 | 9.63e-06 | 0.39 |
| Linear CFR | 2.13e-02 | 1.09e-03 | 9.35e-05 | 8.61e-06 | 0.42 |
| Discounted CFR | 2.28e-02 | 1.67e-03 | 1.47e-04 | 2.39e-05 | 0.42 |

* External-sampling MCCFR (5 seeds): after 100,000 iterations (1,457,751 nodes touched, 0.022 ms / it.) exploitability 0.00304 [0.00152, 0.00605].
* External-sampling MCCFR, linear weights (5 seeds): after 100,000 iterations (1,458,271 nodes touched, 0.023 ms / it.) exploitability 0.00329 [0.00184, 0.00588].

## Leduc hold'em

Exact LP game value -0.0856064 (solved in 0.29 s).

| Variant | 10 it. | 100 it. | 1,000 it. | 10,000 it. | ms / it. |
|---|---|---|---|---|---|
| CFR (vanilla) | 9.27e-01 | 1.73e-01 | 3.98e-02 | 1.01e-02 | 1.60 |
| CFR+ | 6.10e-01 | 1.34e-02 | 2.61e-04 | 5.67e-06 | 1.73 |
| Linear CFR | 7.21e-01 | 3.45e-02 | 5.24e-03 | 1.13e-03 | 1.85 |
| Discounted CFR | 7.79e-01 | 7.76e-03 | 1.58e-04 | 8.44e-06 | 1.89 |

* External-sampling MCCFR (5 seeds): after 300,000 iterations (13,422,943 nodes touched, 0.091 ms / it.) exploitability 0.041 [0.0377, 0.0446].
* External-sampling MCCFR, linear weights (5 seeds): after 300,000 iterations (13,426,605 nodes touched, 0.089 ms / it.) exploitability 0.0409 [0.0362, 0.0461].

## Ablation: discounted-CFR exponents on Leduc (post hoc)

alpha discounts positive regrets, beta negative regrets, and gamma sets the
averaging weight t^gamma. Linear CFR is (1, 1, 1); DCFR is (1.5, 0, 2).

| alpha | beta | gamma | 1,000 it. | 3,000 it. |
|---|---|---|---|---|
| 1 | 0 | 1 | 1.99e-03 | 6.35e-04 |
| 1 | 0 | 2 | 1.46e-03 | 5.04e-04 |
| 1 | 1 | 1 | 5.24e-03 | 2.70e-03 |
| 1 | 1 | 2 | 1.07e-02 | 4.95e-03 |
| 1.5 | 0 | 1 | 2.62e-04 | 4.48e-05 |
| 1.5 | 0 | 2 | 1.58e-04 | 2.78e-05 |
| 1.5 | 1 | 1 | 4.92e-03 | 1.82e-03 |
| 1.5 | 1 | 2 | 1.21e-02 | 3.02e-03 |
