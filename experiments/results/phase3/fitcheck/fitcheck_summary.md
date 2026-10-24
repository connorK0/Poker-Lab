# P3.5: model-fit checks before exploiting

Stack 10 bb. 5 seeds x 30 held-out bots per population; thresholds calibrated per seed on 300 bots simulated from the prior (flag level 5%, half per check); 400 posterior predictive replicates per check. Shares pooled over bots with 95% Wilson intervals; gains in bb/hand over equilibrium (exact, against the bot's true current strategy) with 95% t-intervals across seeds.

## Pre-registered predictions

* False alarms (standard bots flagged) at most 10% at every n: max 9.3%, holds.
* Connectors bots flagged at 100 hands at least 60%: 76.0%, holds.
* Connectors bots flagged at 300 hands at least 90%: 96.7%, holds.
* Connectors bots flagged at 1,000 hands at least 90%: 100.0%, holds.
* Connectors bots flagged at 3,000 hands at least 90%: 100.0%, holds.
* Gear bots flagged at 100 hands at least 40%: 44.0%, holds.
* Gear bots flagged at 1,000 hands at least 85%: 88.0%, holds.
* Gear bots flagged at 3,000 hands at least 85%: 98.7%, holds.
* Misspecified bots: 90% coverage lower when flagged than unflagged at every n >= 100: 100 hands 45% vs 61%, 300 hands 41% vs 64%, 1,000 hands 39% vs 54%, 3,000 hands 32% vs 53%; holds.

## Flag rates (either check)

| Hands | Standard (model right) | Pairs first | Connectors first | Gear change |
|---|---|---|---|---|
| 10 | 4.0% [1.8%, 8.5%] (6/150) | 8.0% [4.6%, 13.5%] (12/150) | 14.7% [9.9%, 21.2%] (22/150) | 7.3% [4.1%, 12.7%] (11/150) |
| 30 | 2.0% [0.7%, 5.7%] (3/150) | 8.7% [5.1%, 14.3%] (13/150) | 38.0% [30.6%, 46.0%] (57/150) | 18.7% [13.2%, 25.7%] (28/150) |
| 100 | 6.7% [3.7%, 11.8%] (10/150) | 16.7% [11.6%, 23.4%] (25/150) | 76.0% [68.6%, 82.1%] (114/150) | 44.0% [36.3%, 52.0%] (66/150) |
| 300 | 5.3% [2.7%, 10.2%] (8/150) | 30.0% [23.2%, 37.8%] (45/150) | 96.7% [92.4%, 98.6%] (145/150) | 70.0% [62.2%, 76.8%] (105/150) |
| 1,000 | 9.3% [5.6%, 15.1%] (14/150) | 43.3% [35.7%, 51.3%] (65/150) | 100.0% [97.5%, 100.0%] (150/150) | 88.0% [81.8%, 92.3%] (132/150) |
| 3,000 | 4.7% [2.3%, 9.3%] (7/150) | 71.3% [63.6%, 78.0%] (107/150) | 100.0% [97.5%, 100.0%] (150/150) | 98.7% [95.3%, 99.6%] (148/150) |

Flag rates, shown hands check alone:

| Hands | Standard (model right) | Pairs first | Connectors first | Gear change |
|---|---|---|---|---|
| 10 | 2.0% | 4.0% | 11.3% | 2.7% |
| 30 | 1.3% | 6.0% | 32.7% | 7.3% |
| 100 | 4.0% | 16.0% | 73.3% | 10.0% |
| 300 | 2.7% | 28.7% | 94.7% | 6.7% |
| 1,000 | 4.0% | 40.7% | 100.0% | 23.3% |
| 3,000 | 2.0% | 69.3% | 100.0% | 57.3% |

Flag rates, stability check alone:

| Hands | Standard (model right) | Pairs first | Connectors first | Gear change |
|---|---|---|---|---|
| 10 | 2.0% | 4.0% | 6.7% | 4.7% |
| 30 | 0.7% | 4.7% | 11.3% | 15.3% |
| 100 | 2.7% | 2.0% | 8.7% | 42.0% |
| 300 | 3.3% | 2.0% | 12.7% | 69.3% |
| 1,000 | 5.3% | 4.0% | 24.0% | 88.0% |
| 3,000 | 2.7% | 4.7% | 33.3% | 98.7% |

## Interval coverage, flagged vs unflagged (90% intervals, non-equilibrium candidates)

| Hands | Population | All | Flagged | Unflagged |
|---|---|---|---|---|
| 10 | Standard (model right) | 87.6% [85.2%, 89.6%] (788/900) | 100.0% [90.4%, 100.0%] (36/36) | 87.0% [84.6%, 89.1%] (752/864) |
| 10 | Pairs first | 82.4% [79.8%, 84.8%] (742/900) | 65.3% [53.8%, 75.2%] (47/72) | 83.9% [81.3%, 86.3%] (695/828) |
| 10 | Connectors first | 67.9% [64.8%, 70.9%] (611/900) | 43.2% [35.0%, 51.7%] (57/132) | 72.1% [68.9%, 75.2%] (554/768) |
| 10 | Gear change | 68.8% [65.7%, 71.7%] (619/900) | 95.5% [87.5%, 98.4%] (63/66) | 66.7% [63.4%, 69.8%] (556/834) |
| 30 | Standard (model right) | 88.6% [86.3%, 90.5%] (797/900) | 88.9% [67.2%, 96.9%] (16/18) | 88.5% [86.3%, 90.5%] (781/882) |
| 30 | Pairs first | 79.9% [77.1%, 82.4%] (719/900) | 64.1% [53.0%, 73.9%] (50/78) | 81.4% [78.6%, 83.9%] (669/822) |
| 30 | Connectors first | 57.0% [53.7%, 60.2%] (513/900) | 56.7% [51.4%, 61.9%] (194/342) | 57.2% [53.0%, 61.2%] (319/558) |
| 30 | Gear change | 49.2% [46.0%, 52.5%] (443/900) | 56.5% [49.0%, 63.8%] (95/168) | 47.5% [43.9%, 51.2%] (348/732) |
| 100 | Standard (model right) | 90.7% [88.6%, 92.4%] (816/900) | 70.0% [57.5%, 80.1%] (42/60) | 92.1% [90.1%, 93.8%] (774/840) |
| 100 | Pairs first | 77.6% [74.7%, 80.2%] (698/900) | 64.7% [56.7%, 71.9%] (97/150) | 80.1% [77.1%, 82.8%] (601/750) |
| 100 | Connectors first | 50.2% [47.0%, 53.5%] (452/900) | 50.7% [47.0%, 54.5%] (347/684) | 48.6% [42.0%, 55.2%] (105/216) |
| 100 | Gear change | 33.7% [30.7%, 36.8%] (303/900) | 28.5% [24.3%, 33.2%] (113/396) | 37.7% [33.6%, 42.0%] (190/504) |
| 300 | Standard (model right) | 88.0% [85.7%, 90.0%] (792/900) | 95.8% [86.0%, 98.8%] (46/48) | 87.6% [85.2%, 89.6%] (746/852) |
| 300 | Pairs first | 74.4% [71.5%, 77.2%] (670/900) | 69.6% [63.9%, 74.8%] (188/270) | 76.5% [73.0%, 79.7%] (482/630) |
| 300 | Connectors first | 49.3% [46.1%, 52.6%] (444/900) | 48.5% [45.2%, 51.8%] (422/870) | 73.3% [55.6%, 85.8%] (22/30) |
| 300 | Gear change | 22.0% [19.4%, 24.8%] (198/900) | 17.5% [14.7%, 20.6%] (110/630) | 32.6% [27.3%, 38.4%] (88/270) |
| 1,000 | Standard (model right) | 93.4% [91.6%, 94.9%] (841/900) | 91.7% [83.8%, 95.9%] (77/84) | 93.6% [91.7%, 95.1%] (764/816) |
| 1,000 | Pairs first | 68.6% [65.4%, 71.5%] (617/900) | 75.1% [70.6%, 79.2%] (293/390) | 63.5% [59.3%, 67.6%] (324/510) |
| 1,000 | Connectors first | 46.2% [43.0%, 49.5%] (416/900) | 46.2% [43.0%, 49.5%] (416/900) | n/a |
| 1,000 | Gear change | 12.2% [10.2%, 14.5%] (110/900) | 12.5% [10.4%, 15.0%] (99/792) | 10.2% [5.8%, 17.3%] (11/108) |
| 3,000 | Standard (model right) | 92.6% [90.7%, 94.1%] (833/900) | 88.1% [75.0%, 94.8%] (37/42) | 92.8% [90.8%, 94.3%] (796/858) |
| 3,000 | Pairs first | 54.7% [51.4%, 57.9%] (492/900) | 54.4% [50.5%, 58.2%] (349/642) | 55.4% [49.3%, 61.4%] (143/258) |
| 3,000 | Connectors first | 41.8% [38.6%, 45.0%] (376/900) | 41.8% [38.6%, 45.0%] (376/900) | n/a |
| 3,000 | Gear change | 6.0% [4.6%, 7.7%] (54/900) | 6.1% [4.7%, 7.9%] (54/888) | 0.0% [0.0%, 24.2%] (0/12) |

## Acting on a flag: lower-bound pick vs checked pick

| Hands | Population | Oracle gain | Lower-bound pick | Checked pick | Lower-bound loses | Checked loses |
|---|---|---|---|---|---|---|
| 10 | Standard (model right) | +0.088 [+0.079, +0.097] | +0.013 [+0.010, +0.016] | +0.012 [+0.010, +0.014] | 3.3% | 3.3% |
| 10 | Pairs first | +0.097 [+0.090, +0.104] | +0.018 [+0.016, +0.020] | +0.017 [+0.017, +0.018] | 3.3% | 2.0% |
| 10 | Connectors first | +0.110 [+0.102, +0.118] | +0.008 [+0.005, +0.011] | +0.007 [+0.003, +0.010] | 9.3% | 5.3% |
| 10 | Gear change | +0.092 [+0.065, +0.119] | +0.005 [+0.001, +0.009] | +0.004 [+0.001, +0.008] | 10.7% | 10.7% |
| 30 | Standard (model right) | +0.088 [+0.079, +0.097] | +0.050 [+0.034, +0.067] | +0.050 [+0.034, +0.066] | 2.0% | 2.0% |
| 30 | Pairs first | +0.097 [+0.090, +0.104] | +0.053 [+0.041, +0.064] | +0.049 [+0.040, +0.058] | 3.3% | 3.3% |
| 30 | Connectors first | +0.110 [+0.102, +0.118] | +0.040 [+0.028, +0.053] | +0.028 [+0.017, +0.039] | 10.0% | 4.0% |
| 30 | Gear change | +0.092 [+0.065, +0.119] | +0.012 [+0.003, +0.020] | +0.008 [+0.001, +0.015] | 22.7% | 20.0% |
| 100 | Standard (model right) | +0.088 [+0.079, +0.097] | +0.082 [+0.074, +0.091] | +0.078 [+0.072, +0.083] | 0.0% | 0.0% |
| 100 | Pairs first | +0.097 [+0.090, +0.104] | +0.079 [+0.077, +0.081] | +0.059 [+0.044, +0.074] | 0.7% | 0.7% |
| 100 | Connectors first | +0.110 [+0.102, +0.118] | +0.064 [+0.052, +0.077] | +0.018 [+0.009, +0.026] | 1.3% | 0.7% |
| 100 | Gear change | +0.092 [+0.065, +0.119] | +0.026 [+0.003, +0.049] | +0.015 [+0.000, +0.030] | 26.0% | 16.0% |
| 300 | Standard (model right) | +0.088 [+0.079, +0.097] | +0.086 [+0.077, +0.096] | +0.081 [+0.068, +0.094] | 0.0% | 0.0% |
| 300 | Pairs first | +0.097 [+0.090, +0.104] | +0.092 [+0.084, +0.099] | +0.047 [+0.034, +0.059] | 1.3% | 1.3% |
| 300 | Connectors first | +0.110 [+0.102, +0.118] | +0.091 [+0.082, +0.100] | +0.002 [-0.001, +0.005] | 0.0% | 0.0% |
| 300 | Gear change | +0.092 [+0.065, +0.119] | +0.027 [+0.015, +0.038] | +0.015 [+0.004, +0.026] | 34.0% | 8.0% |
| 1,000 | Standard (model right) | +0.088 [+0.079, +0.097] | +0.087 [+0.078, +0.097] | +0.081 [+0.070, +0.093] | 0.0% | 0.0% |
| 1,000 | Pairs first | +0.097 [+0.090, +0.104] | +0.094 [+0.087, +0.102] | +0.029 [+0.019, +0.039] | 0.7% | 0.7% |
| 1,000 | Connectors first | +0.110 [+0.102, +0.118] | +0.101 [+0.094, +0.109] | +0.000 [+0.000, +0.000] | 0.0% | 0.0% |
| 1,000 | Gear change | +0.092 [+0.065, +0.119] | +0.035 [+0.014, +0.056] | +0.008 [+0.001, +0.014] | 26.0% | 1.3% |
| 3,000 | Standard (model right) | +0.088 [+0.079, +0.097] | +0.088 [+0.079, +0.097] | +0.085 [+0.075, +0.096] | 0.0% | 0.0% |
| 3,000 | Pairs first | +0.097 [+0.090, +0.104] | +0.095 [+0.088, +0.102] | +0.014 [+0.011, +0.018] | 1.3% | 0.7% |
| 3,000 | Connectors first | +0.110 [+0.102, +0.118] | +0.104 [+0.096, +0.111] | +0.000 [+0.000, +0.000] | 0.0% | 0.0% |
| 3,000 | Gear change | +0.092 [+0.065, +0.119] | +0.036 [+0.012, +0.059] | +0.002 [-0.002, +0.005] | 30.7% | 0.0% |

## Calibrated thresholds (first seed)

| Hands | shown_hands threshold | shown_hands median p | stability threshold | stability median p |
|---|---|---|---|---|
| 10 | 0.093 | 0.596 | 0.242 | 0.496 |
| 30 | 0.079 | 0.514 | 0.186 | 0.524 |
| 100 | 0.070 | 0.496 | 0.140 | 0.521 |
| 300 | 0.076 | 0.526 | 0.191 | 0.534 |
| 1,000 | 0.075 | 0.554 | 0.186 | 0.537 |
| 3,000 | 0.048 | 0.507 | 0.186 | 0.522 |

## Example reports (first seed)

* **connectors first nit, 100 hands.** Seen 100 hands. As small blind it pushed 5 of 50; facing a push it called 7 of 30; its cards were shown 9 times. Estimated ranges: pushes 15% of hands (90% interval 10% to 23%) and calls 26% (14% to 39%); the equilibrium at 10 bb pushes 58% and calls 37%. Model check failed: the hands it showed do not match the hand order the model assumes (p = 0.000, threshold 0.070). The model's estimates cannot be trusted for this opponent, so play the equilibrium rather than the top-ranked restricted Nash response (p = 0.5) (credited with +0.030 bb/hand by the model).
* **connectors first nit, 1,000 hands.** Seen 1,000 hands. As small blind it pushed 69 of 500; facing a push it called 37 of 289; its cards were shown 66 times. Estimated ranges: pushes 15% of hands (90% interval 13% to 17%) and calls 15% (12% to 17%); the equilibrium at 10 bb pushes 58% and calls 37%. Model check failed: the hands it showed do not match the hand order the model assumes (p = 0.000, threshold 0.075). The model's estimates cannot be trusted for this opponent, so play the equilibrium rather than the top-ranked best response to the model (credited with +0.178 bb/hand by the model).
* **connectors first tag, 100 hands.** Seen 100 hands. As small blind it pushed 22 of 50; facing a push it called 4 of 25; its cards were shown 15 times. Estimated ranges: pushes 54% of hands (90% interval 42% to 65%) and calls 24% (15% to 36%); the equilibrium at 10 bb pushes 58% and calls 37%. Model check failed: the hands it showed do not match the hand order the model assumes (p = 0.007, threshold 0.070). The model's estimates cannot be trusted for this opponent, so play the equilibrium rather than the top-ranked restricted Nash response (p = 0.5) (credited with +0.043 bb/hand by the model).
* **connectors first tag, 1,000 hands.** Seen 1,000 hands. As small blind it pushed 202 of 500; facing a push it called 66 of 291; its cards were shown 137 times. Estimated ranges: pushes 42% of hands (90% interval 38% to 45%) and calls 23% (19% to 27%); the equilibrium at 10 bb pushes 58% and calls 37%. Model check failed: the hands it showed do not match the hand order the model assumes (p = 0.000, threshold 0.075). The model's estimates cannot be trusted for this opponent, so play the equilibrium rather than the top-ranked restricted Nash response (p = 0.9) (credited with +0.121 bb/hand by the model).
* **connectors first lag, 100 hands.** Seen 100 hands. As small blind it pushed 34 of 50; facing a push it called 16 of 35; its cards were shown 33 times. Estimated ranges: pushes 69% of hands (90% interval 59% to 78%) and calls 55% (41% to 68%); the equilibrium at 10 bb pushes 58% and calls 37%. Model check failed: the hands it showed do not match the hand order the model assumes (p = 0.003, threshold 0.070). The model's estimates cannot be trusted for this opponent, so play the equilibrium rather than the top-ranked restricted Nash response (p = 0.75) (credited with +0.028 bb/hand by the model).
* **connectors first lag, 1,000 hands.** Seen 1,000 hands. As small blind it pushed 319 of 500; facing a push it called 136 of 286; its cards were shown 245 times. Estimated ranges: pushes 64% of hands (90% interval 60% to 67%) and calls 48% (43% to 53%); the equilibrium at 10 bb pushes 58% and calls 37%. Model check failed: the hands it showed do not match the hand order the model assumes (p = 0.000, threshold 0.075). The model's estimates cannot be trusted for this opponent, so play the equilibrium rather than the top-ranked restricted Nash response (p = 0.9) (credited with +0.047 bb/hand by the model).
* **connectors first calling_station, 100 hands.** Seen 100 hands. As small blind it pushed 19 of 50; facing a push it called 21 of 30; its cards were shown 30 times. Estimated ranges: pushes 45% of hands (90% interval 34% to 57%) and calls 78% (67% to 87%); the equilibrium at 10 bb pushes 58% and calls 37%. Model check failed: the hands it showed do not match the hand order the model assumes (p = 0.005, threshold 0.070). The model's estimates cannot be trusted for this opponent, so play the equilibrium rather than the top-ranked restricted Nash response (p = 0.75) (credited with +0.027 bb/hand by the model).
* **connectors first calling_station, 1,000 hands.** Seen 1,000 hands. As small blind it pushed 158 of 500; facing a push it called 212 of 293; its cards were shown 277 times. Estimated ranges: pushes 32% of hands (90% interval 29% to 35%) and calls 73% (69% to 77%); the equilibrium at 10 bb pushes 58% and calls 37%. Model check failed: the hands it showed do not match the hand order the model assumes (p = 0.000, threshold 0.075). The model's estimates cannot be trusted for this opponent, so play the equilibrium rather than the top-ranked best response to the model (credited with +0.022 bb/hand by the model).
* **connectors first maniac, 100 hands.** Seen 100 hands. As small blind it pushed 47 of 50; facing a push it called 17 of 32; its cards were shown 37 times. Estimated ranges: pushes 89% of hands (90% interval 83% to 94%) and calls 72% (60% to 82%); the equilibrium at 10 bb pushes 58% and calls 37%. Model check failed: the hands it showed do not match the hand order the model assumes (p = 0.007, threshold 0.070); its push and call rates changed during the session (p = 0.095, threshold 0.140). The model's estimates cannot be trusted for this opponent, so play the equilibrium rather than the top-ranked restricted Nash response (p = 0.9) (credited with +0.096 bb/hand by the model).
* **connectors first maniac, 1,000 hands.** Seen 1,000 hands. As small blind it pushed 449 of 500; facing a push it called 153 of 296; its cards were shown 324 times. Estimated ranges: pushes 85% of hands (90% interval 83% to 87%) and calls 59% (55% to 63%); the equilibrium at 10 bb pushes 58% and calls 37%. Model check failed: the hands it showed do not match the hand order the model assumes (p = 0.000, threshold 0.075); its push and call rates changed during the session (p = 0.003, threshold 0.186). The model's estimates cannot be trusted for this opponent, so play the equilibrium rather than the top-ranked best response to the model (credited with +0.076 bb/hand by the model).
* **gear change nit -> maniac, 100 hands.** Seen 100 hands. As small blind it pushed 36 of 50; facing a push it called 11 of 25; its cards were shown 18 times. Estimated ranges: pushes 75% of hands (90% interval 66% to 83%) and calls 56% (48% to 63%); the equilibrium at 10 bb pushes 58% and calls 37%. Model check failed: the hands it showed do not match the hand order the model assumes (p = 0.060, threshold 0.070); its push and call rates changed during the session (p = 0.050, threshold 0.140). The model's estimates cannot be trusted for this opponent, so play the equilibrium rather than the top-ranked restricted Nash response (p = 0.75) (credited with +0.039 bb/hand by the model).
* **gear change nit -> maniac, 1,000 hands.** Seen 1,000 hands. As small blind it pushed 359 of 500; facing a push it called 118 of 295; its cards were shown 242 times. Estimated ranges: pushes 74% of hands (90% interval 71% to 77%) and calls 45% (41% to 49%); the equilibrium at 10 bb pushes 58% and calls 37%. Model check failed: its push and call rates changed during the session (p = 0.000, threshold 0.186). The model's estimates cannot be trusted for this opponent, so play the equilibrium rather than the top-ranked restricted Nash response (p = 0.9) (credited with +0.025 bb/hand by the model).
* **gear change tag -> nit, 100 hands.** Seen 100 hands. As small blind it pushed 18 of 50; facing a push it called 1 of 25; its cards were shown 5 times. Estimated ranges: pushes 33% of hands (90% interval 24% to 44%) and calls 15% (5% to 30%); the equilibrium at 10 bb pushes 58% and calls 37%. Recommended: the best response to the model. As small blind, push 100% of hands (equilibrium 58%): adds Q6o, Q5o, J7o, Q4o and 52 more. As big blind, call 25% of hands (equilibrium 37%): drops QJo, K9o, Q9s, JTs and 18 more. Expected gain over the equilibrium +0.169 bb/hand (90% interval +0.032 to +0.264; probability of gaining 98%), +0.167 over the population recommendation. Exploitability 0.251 bb/hand. Model check passed: the observed play looks like data the model produces.
* **gear change tag -> nit, 1,000 hands.** Seen 1,000 hands. As small blind it pushed 171 of 500; facing a push it called 70 of 294; its cards were shown 128 times. Estimated ranges: pushes 34% of hands (90% interval 31% to 37%) and calls 29% (25% to 32%); the equilibrium at 10 bb pushes 58% and calls 37%. Model check failed: its push and call rates changed during the session (p = 0.062, threshold 0.186). The model's estimates cannot be trusted for this opponent, so play the equilibrium rather than the top-ranked restricted Nash response (p = 0.9) (credited with +0.040 bb/hand by the model).
* **gear change lag -> tag, 100 hands.** Seen 100 hands. As small blind it pushed 27 of 50; facing a push it called 11 of 30; its cards were shown 16 times. Estimated ranges: pushes 54% of hands (90% interval 46% to 63%) and calls 36% (30% to 44%); the equilibrium at 10 bb pushes 58% and calls 37%. Play the equilibrium: no deviation is likely enough to gain. The one with the highest expected gain, the best response to the model, would gain +0.001 bb/hand on average but does worse than the equilibrium with probability 33%. Model check passed: the observed play looks like data the model produces.
* **gear change lag -> tag, 1,000 hands.** Seen 1,000 hands. As small blind it pushed 243 of 500; facing a push it called 74 of 304; its cards were shown 172 times. Estimated ranges: pushes 51% of hands (90% interval 48% to 54%) and calls 28% (25% to 31%); the equilibrium at 10 bb pushes 58% and calls 37%. Model check failed: its push and call rates changed during the session (p = 0.028, threshold 0.186). The model's estimates cannot be trusted for this opponent, so play the equilibrium rather than the top-ranked restricted Nash response (p = 0.9) (credited with +0.031 bb/hand by the model).
* **gear change calling_station -> tag, 100 hands.** Seen 100 hands. As small blind it pushed 11 of 50; facing a push it called 11 of 26; its cards were shown 14 times. Estimated ranges: pushes 30% of hands (90% interval 20% to 38%) and calls 50% (41% to 58%); the equilibrium at 10 bb pushes 58% and calls 37%. Recommended: the best response to the model. As small blind, push 46% of hands (equilibrium 58%): drops J8o, T7s, J6s, Q2s and 25 more. As big blind, call 22% of hands (equilibrium 37%): drops QTs, K8s, QJo, K9o and 23 more. Expected gain over the equilibrium +0.034 bb/hand (90% interval +0.015 to +0.047; probability of gaining 99%), +0.034 over the population recommendation. Exploitability 0.137 bb/hand. Model check passed: the observed play looks like data the model produces.
* **gear change calling_station -> tag, 1,000 hands.** Seen 1,000 hands. As small blind it pushed 187 of 500; facing a push it called 117 of 272; its cards were shown 183 times. Estimated ranges: pushes 36% of hands (90% interval 33% to 38%) and calls 54% (50% to 57%); the equilibrium at 10 bb pushes 58% and calls 37%. Model check failed: its push and call rates changed during the session (p = 0.013, threshold 0.186). The model's estimates cannot be trusted for this opponent, so play the equilibrium rather than the top-ranked best response to the model (credited with +0.032 bb/hand by the model).
* **gear change maniac -> nit, 100 hands.** Seen 100 hands. As small blind it pushed 35 of 50; facing a push it called 17 of 34; its cards were shown 28 times. Estimated ranges: pushes 76% of hands (90% interval 69% to 82%) and calls 60% (53% to 66%); the equilibrium at 10 bb pushes 58% and calls 37%. Model check failed: the hands it showed do not match the hand order the model assumes (p = 0.052, threshold 0.070); its push and call rates changed during the session (p = 0.122, threshold 0.140). The model's estimates cannot be trusted for this opponent, so play the equilibrium rather than the top-ranked restricted Nash response (p = 0.75) (credited with +0.047 bb/hand by the model).
* **gear change maniac -> nit, 1,000 hands.** Seen 1,000 hands. As small blind it pushed 378 of 500; facing a push it called 128 of 299; its cards were shown 271 times. Estimated ranges: pushes 81% of hands (90% interval 78% to 83%) and calls 48% (43% to 52%); the equilibrium at 10 bb pushes 58% and calls 37%. Model check failed: the hands it showed do not match the hand order the model assumes (p = 0.005, threshold 0.075); its push and call rates changed during the session (p = 0.000, threshold 0.186). The model's estimates cannot be trusted for this opponent, so play the equilibrium rather than the top-ranked restricted Nash response (p = 0.9) (credited with +0.045 bb/hand by the model).
