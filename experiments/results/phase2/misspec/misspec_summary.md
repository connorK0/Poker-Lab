# P2.12: exploitation when the model is wrong

Seed 2026, commit None (dirty: None), 2883 s. 150 starting bots per kind (5 seeds x 30). Chips per hand, exact. Gain = value against the true opponent minus the equilibrium's. Means over bots, 95% t-intervals across seeds; "loses" = share of bots with a negative gain.

## How far outside the family (no hero's result is used)

| Opponent | KL (nats/hand) | Hands to 20:1 | Mean TV per decision | Its exploitability | Equilibrium's value | Available gain (oracle) |
|---|---|---|---|---|---|---|
| Family bot (control) | 0.0000 | never | 0.000 | 1.016 | +0.277 | +0.738 |
| Ignores the board | 0.0751 | 40 | 0.098 | 1.324 | +0.399 | +0.926 |
| Passive, then aggressive | 0.3986 | 8 | 0.232 | 1.162 | +0.337 | +0.825 |
| A quarter of that | 0.0247 | 121 | 0.057 | 1.030 | +0.289 | +0.740 |
| Stabs in position | 0.1950 | 15 | 0.149 | 1.130 | +0.290 | +0.840 |
| Raises only when strong | 0.1107 | 27 | 0.089 | 1.157 | +0.260 | +0.897 |
| Noise at every decision | 0.1172 | 26 | 0.128 | 1.159 | +0.277 | +0.881 |
| 75% equilibrium | 0.1588 | 19 | 0.163 | 0.305 | +0.066 | +0.239 |
| The equilibrium | 0.2948 | 10 | 0.207 | -0.000 | +0.000 | -0.000 |

Spearman correlation over off-family bots of KL and best response's shortfall at 3,000 hands: 0.37.

## Best response to the fitted model: gain by hands observed

| Opponent | 10 | 30 | 100 | 300 | 1,000 | 3,000 | Limit (infinite data) | Oracle |
|---|---|---|---|---|---|---|---|---|
| Family bot (control) | +0.591 | +0.636 | +0.700 | +0.724 | +0.734 | +0.736 | +0.738 | +0.738 |
| Ignores the board | +0.664 | +0.683 | +0.719 | +0.742 | +0.740 | +0.750 | +0.749 | +0.926 |
| Passive, then aggressive | +0.431 | +0.485 | +0.515 | +0.519 | +0.530 | +0.526 | +0.527 | +0.825 |
| A quarter of that | +0.571 | +0.632 | +0.682 | +0.703 | +0.712 | +0.715 | +0.716 | +0.740 |
| Stabs in position | +0.552 | +0.615 | +0.659 | +0.669 | +0.670 | +0.674 | +0.672 | +0.840 |
| Raises only when strong | +0.507 | +0.564 | +0.623 | +0.633 | +0.643 | +0.641 | +0.642 | +0.897 |
| Noise at every decision | +0.576 | +0.638 | +0.695 | +0.722 | +0.729 | +0.731 | +0.737 | +0.881 |
| 75% equilibrium | +0.094 | +0.095 | +0.120 | +0.141 | +0.153 | +0.156 | +0.158 | +0.239 |
| The equilibrium | -0.107 | -0.085 | -0.083 | -0.044 | -0.029 | -0.026 | -0.020 | -0.000 |

## Every response at 3,000 hands

| Opponent | Response | Gain | Share of the available gain | Loses | Exploitability |
|---|---|---|---|---|---|
| Family bot (control) | Best response | +0.736 [+0.713, +0.759] | 100% | 0% | 1.599 |
| Family bot (control) | Restricted Nash, p = 0.5 | +0.494 [+0.485, +0.502] | 67% | 0% | 0.134 |
| Family bot (control) | Restricted Nash, p = 0.9 | +0.705 [+0.685, +0.725] | 95% | 0% | 0.690 |
| Family bot (control) | Data-biased | +0.483 [+0.470, +0.495] | 65% | 0% | 0.721 |
| Family bot (control) | Best response, card-blind model | +0.207 [+0.160, +0.255] | 28% | 43% | 1.966 |
| Ignores the board | Best response | +0.750 [+0.708, +0.792] | 81% | 14% | 1.225 |
| Ignores the board | Restricted Nash, p = 0.5 | +0.616 [+0.606, +0.626] | 67% | 0% | 0.136 |
| Ignores the board | Restricted Nash, p = 0.9 | +0.750 [+0.716, +0.783] | 81% | 6% | 0.564 |
| Ignores the board | Data-biased | +0.522 [+0.513, +0.530] | 56% | 0% | 0.435 |
| Ignores the board | Best response, card-blind model | +0.168 [+0.128, +0.207] | 18% | 58% | 1.912 |
| Passive, then aggressive | Best response | +0.526 [+0.510, +0.543] | 64% | 0% | 1.382 |
| Passive, then aggressive | Restricted Nash, p = 0.5 | +0.419 [+0.414, +0.423] | 51% | 0% | 0.129 |
| Passive, then aggressive | Restricted Nash, p = 0.9 | +0.523 [+0.499, +0.548] | 63% | 1% | 0.740 |
| Passive, then aggressive | Data-biased | +0.417 [+0.403, +0.432] | 51% | 9% | 0.510 |
| Passive, then aggressive | Best response, card-blind model | +0.471 [+0.446, +0.496] | 57% | 7% | 1.606 |
| A quarter of that | Best response | +0.715 [+0.693, +0.737] | 97% | 0% | 1.487 |
| A quarter of that | Restricted Nash, p = 0.5 | +0.471 [+0.462, +0.480] | 64% | 0% | 0.125 |
| A quarter of that | Restricted Nash, p = 0.9 | +0.682 [+0.661, +0.703] | 92% | 0% | 0.659 |
| A quarter of that | Data-biased | +0.467 [+0.455, +0.479] | 63% | 0% | 0.572 |
| A quarter of that | Best response, card-blind model | +0.332 [+0.300, +0.363] | 45% | 23% | 1.860 |
| Stabs in position | Best response | +0.674 [+0.656, +0.693] | 80% | 0% | 1.421 |
| Stabs in position | Restricted Nash, p = 0.5 | +0.434 [+0.424, +0.444] | 52% | 0% | 0.129 |
| Stabs in position | Restricted Nash, p = 0.9 | +0.654 [+0.632, +0.675] | 78% | 0% | 0.649 |
| Stabs in position | Data-biased | +0.449 [+0.435, +0.464] | 54% | 0% | 0.629 |
| Stabs in position | Best response, card-blind model | +0.274 [+0.222, +0.326] | 33% | 25% | 1.939 |
| Raises only when strong | Best response | +0.641 [+0.613, +0.669] | 72% | 0% | 1.876 |
| Raises only when strong | Restricted Nash, p = 0.5 | +0.357 [+0.341, +0.373] | 40% | 0% | 0.123 |
| Raises only when strong | Restricted Nash, p = 0.9 | +0.589 [+0.565, +0.613] | 66% | 0% | 0.778 |
| Raises only when strong | Data-biased | +0.371 [+0.362, +0.380] | 41% | 0% | 0.820 |
| Raises only when strong | Best response, card-blind model | +0.317 [+0.281, +0.354] | 35% | 16% | 2.176 |
| Noise at every decision | Best response | +0.731 [+0.702, +0.760] | 83% | 0% | 1.645 |
| Noise at every decision | Restricted Nash, p = 0.5 | +0.499 [+0.486, +0.512] | 57% | 0% | 0.138 |
| Noise at every decision | Restricted Nash, p = 0.9 | +0.702 [+0.672, +0.732] | 80% | 0% | 0.708 |
| Noise at every decision | Data-biased | +0.501 [+0.482, +0.520] | 57% | 0% | 0.736 |
| Noise at every decision | Best response, card-blind model | +0.223 [+0.173, +0.272] | 25% | 36% | 1.903 |
| 75% equilibrium | Best response | +0.156 [+0.154, +0.159] | 65% | 0% | 1.473 |
| 75% equilibrium | Restricted Nash, p = 0.5 | +0.097 [+0.095, +0.100] | 41% | 0% | 0.084 |
| 75% equilibrium | Restricted Nash, p = 0.9 | +0.159 [+0.156, +0.161] | 66% | 0% | 0.471 |
| 75% equilibrium | Data-biased | +0.100 [+0.097, +0.103] | 42% | 0% | 0.751 |
| 75% equilibrium | Best response, card-blind model | -0.171 [-0.180, -0.161] | -71% | 91% | 2.177 |
| The equilibrium | Best response | -0.026 [-0.028, -0.024] |  | 100% | 1.509 |
| The equilibrium | Restricted Nash, p = 0.5 | -0.014 [-0.014, -0.014] |  | 100% | 0.059 |
| The equilibrium | Restricted Nash, p = 0.9 | -0.033 [-0.034, -0.033] |  | 100% | 0.470 |
| The equilibrium | Data-biased | -0.031 [-0.032, -0.029] |  | 100% | 1.068 |
| The equilibrium | Best response, card-blind model | -0.259 [-0.259, -0.259] |  | 100% | 2.155 |

## How often each response loses to equilibrium play, by hands observed

| Opponent | Response | 10 | 30 | 100 | 300 | 1,000 | 3,000 |
|---|---|---|---|---|---|---|---|
| Family bot (control) | Best response | 0% | 0% | 0% | 0% | 0% | 0% |
| Family bot (control) | Restricted Nash, p = 0.9 | 0% | 0% | 0% | 0% | 0% | 0% |
| Ignores the board | Best response | 9% | 9% | 11% | 11% | 16% | 14% |
| Ignores the board | Restricted Nash, p = 0.9 | 5% | 7% | 5% | 8% | 9% | 6% |
| Passive, then aggressive | Best response | 9% | 5% | 1% | 0% | 0% | 0% |
| Passive, then aggressive | Restricted Nash, p = 0.9 | 13% | 10% | 5% | 1% | 1% | 1% |
| A quarter of that | Best response | 0% | 0% | 0% | 0% | 0% | 0% |
| A quarter of that | Restricted Nash, p = 0.9 | 0% | 0% | 0% | 0% | 0% | 0% |
| Stabs in position | Best response | 0% | 1% | 0% | 0% | 0% | 0% |
| Stabs in position | Restricted Nash, p = 0.9 | 0% | 1% | 0% | 0% | 0% | 0% |
| Raises only when strong | Best response | 1% | 0% | 0% | 0% | 0% | 0% |
| Raises only when strong | Restricted Nash, p = 0.9 | 0% | 0% | 0% | 0% | 0% | 0% |
| Noise at every decision | Best response | 1% | 3% | 0% | 0% | 0% | 0% |
| Noise at every decision | Restricted Nash, p = 0.9 | 0% | 3% | 0% | 0% | 0% | 0% |
| 75% equilibrium | Best response | 19% | 18% | 9% | 3% | 0% | 0% |
| 75% equilibrium | Restricted Nash, p = 0.9 | 15% | 12% | 5% | 1% | 0% | 0% |
| The equilibrium | Best response | 100% | 100% | 100% | 100% | 100% | 100% |
| The equilibrium | Restricted Nash, p = 0.9 | 100% | 100% | 100% | 100% | 100% | 100% |

## Sessions against adapters: 2,000 hands, updates every 50

### Static family bot (control)

With the equilibrium hero, the bot moves 0.000 (TV per decision), ends 0.0000 nats/hand from the family, and its mean exploitability is 1.016.

| Hero | Value | Gain over equilibrium | Second half | Loses | Exploitability |
|---|---|---|---|---|---|
| Equilibrium | +0.277 [+0.272, +0.283] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | 0% | -0.000 |
| Fit once, best response | +0.889 [+0.870, +0.907] | +0.611 [+0.594, +0.628] | +0.719 [+0.699, +0.739] | 0% | 1.350 |
| Fit once, restricted Nash | +0.866 [+0.848, +0.883] | +0.588 [+0.574, +0.602] | +0.692 [+0.675, +0.709] | 0% | 0.597 |
| Refit, best response | +0.998 [+0.974, +1.022] | +0.721 [+0.700, +0.742] | +0.736 [+0.715, +0.757] | 0% | 1.541 |
| Refit, restricted Nash, p = 0.9 | +0.968 [+0.943, +0.993] | +0.691 [+0.669, +0.713] | +0.704 [+0.683, +0.725] | 0% | 0.678 |
| Refit, restricted Nash, p = 0.5 | +0.763 [+0.750, +0.777] | +0.486 [+0.477, +0.495] | +0.494 [+0.485, +0.503] | 0% | 0.133 |
| Refit on recent hands, best response | +0.992 [+0.968, +1.015] | +0.714 [+0.693, +0.735] | +0.726 [+0.704, +0.747] | 0% | 1.564 |
| Clairvoyant (reference) | +1.016 [+0.991, +1.040] | +0.738 [+0.717, +0.760] | +0.738 [+0.717, +0.760] | 0% | 1.580 |

### Adapts inside the family, slowly

With the equilibrium hero, the bot moves 0.263 (TV per decision), ends 0.0000 nats/hand from the family, and its mean exploitability is 0.855.

| Hero | Value | Gain over equilibrium | Second half | Loses | Exploitability |
|---|---|---|---|---|---|
| Equilibrium | +0.220 [+0.217, +0.223] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | 0% | -0.000 |
| Fit once, best response | +0.528 [+0.488, +0.567] | +0.308 [+0.270, +0.345] | +0.309 [+0.259, +0.358] | 17% | 1.471 |
| Fit once, restricted Nash | +0.504 [+0.485, +0.524] | +0.285 [+0.268, +0.302] | +0.287 [+0.263, +0.311] | 9% | 0.594 |
| Refit, best response | +0.803 [+0.797, +0.808] | +0.583 [+0.579, +0.587] | +0.614 [+0.609, +0.619] | 0% | 1.945 |
| Refit, restricted Nash, p = 0.9 | +0.739 [+0.735, +0.742] | +0.519 [+0.515, +0.523] | +0.535 [+0.529, +0.542] | 0% | 0.803 |
| Refit, restricted Nash, p = 0.5 | +0.505 [+0.502, +0.507] | +0.285 [+0.284, +0.286] | +0.259 [+0.256, +0.261] | 0% | 0.104 |
| Refit on recent hands, best response | +0.828 [+0.823, +0.833] | +0.608 [+0.606, +0.610] | +0.644 [+0.641, +0.647] | 0% | 2.158 |
| Clairvoyant (reference) | +0.853 [+0.849, +0.857] | +0.633 [+0.630, +0.636] | +0.654 [+0.652, +0.657] | 0% | 2.278 |

### Adapts inside the family, fast

With the equilibrium hero, the bot moves 0.267 (TV per decision), ends 0.0000 nats/hand from the family, and its mean exploitability is 0.884.

| Hero | Value | Gain over equilibrium | Second half | Loses | Exploitability |
|---|---|---|---|---|---|
| Equilibrium | +0.214 [+0.213, +0.216] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | 0% | -0.000 |
| Fit once, best response | +0.719 [+0.705, +0.732] | +0.504 [+0.491, +0.518] | +0.592 [+0.575, +0.609] | 1% | 1.749 |
| Fit once, restricted Nash | +0.662 [+0.648, +0.676] | +0.448 [+0.434, +0.462] | +0.525 [+0.509, +0.542] | 0% | 0.787 |
| Refit, best response | +0.840 [+0.835, +0.844] | +0.625 [+0.621, +0.630] | +0.654 [+0.650, +0.658] | 0% | 2.250 |
| Refit, restricted Nash, p = 0.9 | +0.737 [+0.734, +0.739] | +0.523 [+0.520, +0.525] | +0.533 [+0.530, +0.536] | 0% | 0.931 |
| Refit, restricted Nash, p = 0.5 | +0.467 [+0.466, +0.468] | +0.253 [+0.251, +0.254] | +0.252 [+0.248, +0.256] | 0% | 0.089 |
| Refit on recent hands, best response | +0.839 [+0.835, +0.842] | +0.624 [+0.621, +0.627] | +0.651 [+0.647, +0.655] | 0% | 2.388 |
| Clairvoyant (reference) | +0.870 [+0.869, +0.870] | +0.656 [+0.654, +0.657] | +0.660 [+0.657, +0.663] | 0% | 2.563 |

### Adapts freely, slowly

With the equilibrium hero, the bot moves 0.325 (TV per decision), ends 0.7066 nats/hand from the family, and its mean exploitability is 1.439.

| Hero | Value | Gain over equilibrium | Second half | Loses | Exploitability |
|---|---|---|---|---|---|
| Equilibrium | +0.111 [+0.109, +0.113] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | 0% | -0.000 |
| Fit once, best response | -0.073 [-0.085, -0.062] | -0.184 [-0.196, -0.173] | -0.381 [-0.397, -0.365] | 98% | 1.444 |
| Fit once, restricted Nash | -0.083 [-0.094, -0.072] | -0.194 [-0.207, -0.181] | -0.370 [-0.388, -0.352] | 98% | 0.639 |
| Refit, best response | +0.100 [+0.094, +0.106] | -0.011 [-0.016, -0.006] | -0.170 [-0.191, -0.150] | 47% | 1.902 |
| Refit, restricted Nash, p = 0.9 | +0.084 [+0.076, +0.092] | -0.027 [-0.034, -0.020] | -0.165 [-0.178, -0.152] | 63% | 0.743 |
| Refit, restricted Nash, p = 0.5 | +0.188 [+0.185, +0.191] | +0.078 [+0.074, +0.081] | -0.006 [-0.010, -0.002] | 7% | 0.102 |
| Refit on recent hands, best response | +0.190 [+0.179, +0.200] | +0.079 [+0.068, +0.090] | -0.028 [-0.037, -0.019] | 21% | 2.072 |
| Clairvoyant (reference) | +0.892 [+0.881, +0.902] | +0.781 [+0.771, +0.791] | +0.863 [+0.853, +0.873] | 0% | 2.370 |

### Adapts freely, fast

With the equilibrium hero, the bot moves 0.355 (TV per decision), ends 0.7552 nats/hand from the family, and its mean exploitability is 1.760.

| Hero | Value | Gain over equilibrium | Second half | Loses | Exploitability |
|---|---|---|---|---|---|
| Equilibrium | +0.080 [+0.076, +0.083] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | 0% | -0.000 |
| Fit once, best response | -0.197 [-0.221, -0.173] | -0.277 [-0.304, -0.251] | -0.345 [-0.372, -0.318] | 99% | 1.788 |
| Fit once, restricted Nash | -0.178 [-0.206, -0.149] | -0.257 [-0.289, -0.225] | -0.314 [-0.353, -0.275] | 100% | 0.801 |
| Refit, best response | -0.051 [-0.062, -0.041] | -0.131 [-0.143, -0.119] | -0.151 [-0.168, -0.135] | 97% | 2.117 |
| Refit, restricted Nash, p = 0.9 | -0.042 [-0.048, -0.036] | -0.122 [-0.128, -0.116] | -0.118 [-0.130, -0.107] | 100% | 0.942 |
| Refit, restricted Nash, p = 0.5 | +0.082 [+0.081, +0.084] | +0.002 [-0.002, +0.007] | -0.009 [-0.013, -0.004] | 39% | 0.089 |
| Refit on recent hands, best response | -0.026 [-0.028, -0.023] | -0.105 [-0.110, -0.101] | -0.130 [-0.140, -0.120] | 97% | 2.074 |
| Clairvoyant (reference) | +1.063 [+1.052, +1.074] | +0.983 [+0.970, +0.997] | +1.007 [+0.994, +1.020] | 0% | 2.716 |

## Pre-registered predictions

| | Prediction | Values | Held |
|---|---|---|---|
| A1 | Check: the control repeats P2.7's committed eq value, gain and exploitability of BR, RNR .5/.9 and DBR at every size | 0 | yes |
| A2 | Check: the comparison covers all 150 control bots | 150 | yes |
| A3 | Check: the control's KL to the family is at most 1e-6 | 7.499e-14 | yes |
| A4 | Check: the equilibrium hero never has negative value against any bot | 0.09046, 0.2355, 0.1661, 0.1068, 0.0807, 0.08728, 0.07556, 0.03592, 0, 0.09046, 0.1597, 0.194, 0.0552, 0.05019 | yes |
| A5 | Check: with 3,000 hands BR captures at least 97% of the control's oracle gain | 0.9968 | yes |
| B1 | Every rule kind's KL-implied hands to 20:1 (an optimistic bound) are at most 150 | 39.91, 7.516, 121.3, 15.36, 27.07, 25.55, 18.86, 10.16 | yes |
| B2 | slight_street's KL interval lies below .04 | 0.02502 | yes |
| C1 | With 3,000 hands BR captures between 60% and 90% against each off-family kind | 0.81, 0.6379, 0.8029, 0.7152, 0.8294, 0.6538 | yes |
| C2 | slight_street's BR capture at 3,000 hands is between 92% and 99% | 0.9658 | yes |
| C3 | BR's gain over equilibrium at 3,000 hands has an interval above .1 for each off-family kind | 0.7085, 0.5099, 0.6559, 0.6131, 0.7018, 0.1539 | yes |
| C4 | BR does worse than equilibrium on at most 20% of bots of each off-family kind at 3,000 | 0.14, 0, 0, 0, 0, 0 | yes |
| C5a | Against the exact equilibrium BR's gain interval at 3,000 hands lies below -.01 | -0.02364 | yes |
| C5b | Against the exact equilibrium BR does worse than equilibrium on at least 95% of bots | 1 | yes |
| C6 | The rank correlation of KL with BR's shortfall at 3,000 hands is between .2 and .55 | 0.3697 | yes |
| C7 | RNR .9 keeps at least 87% of BR's gain at 3,000 hands against each off-family kind | 0.9998, 0.9942, 0.9694, 0.9184, 0.96, 1.017 | yes |
| C8 | RNR .9's exploitability at 3,000 hands is at most 62% of BR's against every kind | 0.4319, 0.4599, 0.5359, 0.4432, 0.457, 0.4152, 0.4302, 0.3197, 0.3115 | yes |
| C9 | DBR gains at least .03 less than BR at 3,000 hands against each off-family kind | -0.1919, -0.08208, -0.2057, -0.249, -0.2094, -0.05484 | yes |
| C10 | Against near_eq and board_blind the population model beats BR fitted on 10 hands | 0.0418, 0.03692 | yes |
| C11 | The five-parameter model beats the card-blind one at 3,000 hands for each off-family kind | 0.5781, 0.03305, 0.3612, 0.2898, 0.4822, 0.3187 | yes |
| D1 | Refit BR keeps at least 85% of the clairvoyant's gain against adapters inside the family | 0.9207, 0.9539 | yes |
| D2 | Refit BR's gain over fit-once BR has an interval above 0 against all four adapters | 0.2406, 0.1086, 0.1595, 0.1291 | yes |
| D3 | Against free adapters refit BR and RNR .9 lose to equilibrium in the second half | -0.1496, -0.1353, -0.1523, -0.107 | yes |
| D4 | Against the fast free adapter refit BR, RNR .9 and recent-window BR do worse than equilibrium on at least 90% of bots | 0.9667, 1, 0.9733 | yes |
| D5 | Against free adapters RNR .9 does not protect: its gain over BR has an upper bound of at most .02 | -0.007598, 0.02354 | NO |
| D6 | Against the fast free adapter refit RNR .5's gain interval lies within .02 of 0 | -0.0022, 0.006909 | yes |
| D7 | The clairvoyant gains at least .7 against free adapters | 0.7811, 0.9833 | yes |
| D8 | Equilibrium's value against free adapters is at most half its value against fixed bots | 0.4031, 0.2981 | yes |
| D9 | Against the slow free adapter the 300-hand window beats refit on all hands | 0.08015 | yes |

Control against the committed P2.7 results: 150 bots, largest difference 0.
