# P3.4: ranked exploit recommendations with confidence

Stack 10 bb. 5 seeds x 30 held-out bots per population; 2000 posterior draws; 90% intervals; candidates ranked by the interval's lower end. Gains in bb/hand over the equilibrium, exact, against each bot's true strategy (95% t-intervals across seeds). Shares pooled over bots with 95% Wilson intervals.

Check against P3.2 (mode best response, bot by bot): 2100 values, max |difference| 1.5e-13 (pass).

## Standard population

True gain by rule:

| Hands | Lower-bound pick | Bayes best response | Mode best response (P3.2) | Restricted Nash, p = 0.5 | Best candidate in hindsight | Oracle best response |
|---|---|---|---|---|---|---|
| 0 | +0.000 [-0.000, +0.001] | +0.005 [+0.004, +0.007] | -0.007 [-0.010, -0.005] | +0.004 [+0.004, +0.005] | +0.010 [+0.009, +0.011] | +0.088 [+0.079, +0.097] |
| 10 | +0.014 [+0.011, +0.017] | +0.058 [+0.038, +0.077] | +0.051 [+0.030, +0.073] | +0.028 [+0.022, +0.034] | +0.071 [+0.059, +0.083] | +0.088 [+0.079, +0.097] |
| 30 | +0.048 [+0.030, +0.066] | +0.073 [+0.062, +0.084] | +0.070 [+0.056, +0.084] | +0.042 [+0.037, +0.048] | +0.082 [+0.072, +0.091] | +0.088 [+0.079, +0.097] |
| 100 | +0.082 [+0.074, +0.091] | +0.086 [+0.076, +0.095] | +0.085 [+0.075, +0.095] | +0.049 [+0.041, +0.058] | +0.086 [+0.077, +0.096] | +0.088 [+0.079, +0.097] |
| 300 | +0.086 [+0.077, +0.096] | +0.087 [+0.077, +0.097] | +0.087 [+0.077, +0.097] | +0.051 [+0.043, +0.060] | +0.087 [+0.078, +0.097] | +0.088 [+0.079, +0.097] |
| 1,000 | +0.087 [+0.078, +0.097] | +0.088 [+0.078, +0.097] | +0.088 [+0.078, +0.097] | +0.052 [+0.045, +0.059] | +0.088 [+0.079, +0.097] | +0.088 [+0.079, +0.097] |
| 3,000 | +0.088 [+0.079, +0.097] | +0.088 [+0.079, +0.097] | +0.088 [+0.079, +0.097] | +0.052 [+0.044, +0.060] | +0.088 [+0.079, +0.097] | +0.088 [+0.079, +0.097] |

Share of bots against which the rule does worse than the equilibrium:

| Hands | Lower-bound pick | Bayes best response | Mode best response (P3.2) | Restricted Nash, p = 0.5 | Best candidate in hindsight |
|---|---|---|---|---|---|
| 0 | 0.0% | 40.0% | 72.7% | 19.3% | 0.0% |
| 10 | 2.7% | 24.7% | 26.0% | 15.3% | 0.0% |
| 30 | 2.0% | 18.7% | 20.7% | 9.3% | 0.0% |
| 100 | 0.0% | 4.7% | 4.7% | 1.3% | 0.0% |
| 300 | 0.0% | 1.3% | 1.3% | 0.0% | 0.0% |
| 1,000 | 0.7% | 0.7% | 0.7% | 0.0% | 0.0% |
| 3,000 | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |

Does the lower-bound pick win, and are the intervals honest?

| Hands | Deviates | Deviation wins | Pick loses | 90% coverage (all) | One-sided (all) | 90% coverage (pick) | ESS min / median |
|---|---|---|---|---|---|---|---|
| 0 | 28.0% [21.4%, 35.7%] (42/150) | 100.0% [91.6%, 100.0%] (42/42) | 0.0% [0.0%, 2.5%] (0/150) | 96.1% [94.6%, 97.2%] (865/900) | 97.4% [96.2%, 98.3%] (877/900) | 95.2% [84.2%, 98.7%] (40/42) | 1726 / 1766 |
| 10 | 48.0% [40.2%, 55.9%] (72/150) | 94.4% [86.6%, 97.8%] (68/72) | 2.7% [1.0%, 6.7%] (4/150) | 88.7% [86.4%, 90.6%] (798/900) | 93.7% [91.9%, 95.1%] (843/900) | 79.2% [68.4%, 86.9%] (57/72) | 172 / 1362 |
| 30 | 77.3% [70.0%, 83.3%] (116/150) | 97.4% [92.7%, 99.1%] (113/116) | 2.0% [0.7%, 5.7%] (3/150) | 87.9% [85.6%, 89.9%] (791/900) | 92.9% [91.0%, 94.4%] (836/900) | 86.2% [78.8%, 91.3%] (100/116) | 188 / 1159 |
| 100 | 93.3% [88.2%, 96.3%] (140/150) | 100.0% [97.3%, 100.0%] (140/140) | 0.0% [0.0%, 2.5%] (0/150) | 91.7% [89.7%, 93.3%] (825/900) | 95.9% [94.4%, 97.0%] (863/900) | 90.0% [83.9%, 93.9%] (126/140) | 332 / 1360 |
| 300 | 97.3% [93.3%, 99.0%] (146/150) | 100.0% [97.4%, 100.0%] (146/146) | 0.0% [0.0%, 2.5%] (0/150) | 88.2% [86.0%, 90.2%] (794/900) | 93.9% [92.1%, 95.3%] (845/900) | 89.0% [82.9%, 93.1%] (130/146) | 287 / 1619 |
| 1,000 | 100.0% [97.5%, 100.0%] (150/150) | 99.3% [96.3%, 99.9%] (149/150) | 0.7% [0.1%, 3.7%] (1/150) | 93.7% [91.9%, 95.1%] (843/900) | 96.3% [94.9%, 97.4%] (867/900) | 91.3% [85.7%, 94.9%] (137/150) | 438 / 1692 |
| 3,000 | 100.0% [97.5%, 100.0%] (150/150) | 100.0% [97.5%, 100.0%] (150/150) | 0.0% [0.0%, 2.5%] (0/150) | 92.3% [90.4%, 93.9%] (831/900) | 95.6% [94.0%, 96.7%] (860/900) | 92.7% [87.3%, 95.9%] (139/150) | 829 / 1717 |

Top pick by hands observed (share of bots):

| Hands | Equilibrium | Restricted Nash, p = 0.1 | Restricted Nash, p = 0.25 | Restricted Nash, p = 0.5 | Restricted Nash, p = 0.75 | Restricted Nash, p = 0.9 | Best response |
|---|---|---|---|---|---|---|---|
| 0 | 72% | 28% | 0% | 0% | 0% | 0% | 0% |
| 10 | 52% | 18% | 3% | 5% | 15% | 7% | 1% |
| 30 | 23% | 5% | 17% | 12% | 17% | 11% | 15% |
| 100 | 7% | 1% | 5% | 4% | 25% | 22% | 37% |
| 300 | 3% | 0% | 1% | 5% | 11% | 32% | 48% |
| 1,000 | 0% | 0% | 1% | 2% | 5% | 33% | 59% |
| 3,000 | 0% | 0% | 1% | 0% | 3% | 21% | 75% |

| Hands | Bayes minus mode BR | Regret vs hindsight | Over population (posterior) | Over population (true) |
|---|---|---|---|---|
| 0 | +0.013 [+0.010, +0.016] | +0.009 [+0.009, +0.010] | -0.000 [-0.001, +0.001] | -0.000 [-0.001, +0.001] |
| 10 | +0.006 [+0.001, +0.011] | +0.058 [+0.048, +0.067] | +0.013 [+0.011, +0.015] | +0.013 [+0.010, +0.016] |
| 30 | +0.003 [-0.002, +0.007] | +0.034 [+0.021, +0.046] | +0.045 [+0.031, +0.059] | +0.047 [+0.030, +0.065] |
| 100 | +0.000 [-0.000, +0.001] | +0.004 [+0.002, +0.005] | +0.081 [+0.071, +0.091] | +0.082 [+0.074, +0.090] |
| 300 | +0.000 [+0.000, +0.000] | +0.001 [+0.001, +0.002] | +0.086 [+0.075, +0.097] | +0.086 [+0.077, +0.095] |
| 1,000 | +0.000 [-0.000, +0.000] | +0.000 [+0.000, +0.001] | +0.087 [+0.079, +0.096] | +0.087 [+0.078, +0.096] |
| 3,000 | +0.000 [-0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.087 [+0.079, +0.096] | +0.087 [+0.079, +0.096] |

## Misspecified population

True gain by rule:

| Hands | Lower-bound pick | Bayes best response | Mode best response (P3.2) | Restricted Nash, p = 0.5 | Best candidate in hindsight | Oracle best response |
|---|---|---|---|---|---|---|
| 0 | +0.000 [-0.000, +0.001] | +0.007 [+0.005, +0.008] | -0.009 [-0.011, -0.007] | +0.005 [+0.004, +0.006] | +0.011 [+0.010, +0.013] | +0.097 [+0.090, +0.104] |
| 10 | +0.018 [+0.015, +0.021] | +0.062 [+0.043, +0.081] | +0.060 [+0.038, +0.081] | +0.031 [+0.024, +0.037] | +0.073 [+0.062, +0.085] | +0.097 [+0.090, +0.104] |
| 30 | +0.055 [+0.046, +0.064] | +0.082 [+0.072, +0.093] | +0.078 [+0.063, +0.093] | +0.045 [+0.040, +0.050] | +0.086 [+0.076, +0.095] | +0.097 [+0.090, +0.104] |
| 100 | +0.079 [+0.078, +0.081] | +0.091 [+0.081, +0.100] | +0.090 [+0.081, +0.100] | +0.051 [+0.049, +0.053] | +0.092 [+0.082, +0.101] | +0.097 [+0.090, +0.104] |
| 300 | +0.092 [+0.084, +0.099] | +0.095 [+0.087, +0.102] | +0.095 [+0.087, +0.102] | +0.055 [+0.052, +0.058] | +0.095 [+0.087, +0.102] | +0.097 [+0.090, +0.104] |
| 1,000 | +0.094 [+0.087, +0.102] | +0.095 [+0.088, +0.102] | +0.095 [+0.089, +0.102] | +0.057 [+0.055, +0.060] | +0.095 [+0.089, +0.102] | +0.097 [+0.090, +0.104] |
| 3,000 | +0.095 [+0.088, +0.102] | +0.096 [+0.089, +0.103] | +0.096 [+0.089, +0.103] | +0.058 [+0.055, +0.060] | +0.096 [+0.089, +0.103] | +0.097 [+0.090, +0.104] |

Share of bots against which the rule does worse than the equilibrium:

| Hands | Lower-bound pick | Bayes best response | Mode best response (P3.2) | Restricted Nash, p = 0.5 | Best candidate in hindsight |
|---|---|---|---|---|---|
| 0 | 5.3% | 40.7% | 72.7% | 20.7% | 0.0% |
| 10 | 4.7% | 26.0% | 24.7% | 18.7% | 0.0% |
| 30 | 4.0% | 15.3% | 18.7% | 13.3% | 0.0% |
| 100 | 0.7% | 6.7% | 8.0% | 4.0% | 0.0% |
| 300 | 0.7% | 4.7% | 4.7% | 1.3% | 0.0% |
| 1,000 | 1.3% | 1.3% | 1.3% | 1.3% | 0.0% |
| 3,000 | 1.3% | 1.3% | 1.3% | 1.3% | 0.0% |

Does the lower-bound pick win, and are the intervals honest?

| Hands | Deviates | Deviation wins | Pick loses | 90% coverage (all) | One-sided (all) | 90% coverage (pick) | ESS min / median |
|---|---|---|---|---|---|---|---|
| 0 | 33.3% [26.3%, 41.2%] (50/150) | 84.0% [71.5%, 91.7%] (42/50) | 5.3% [2.7%, 10.2%] (8/150) | 92.4% [90.5%, 94.0%] (832/900) | 94.1% [92.4%, 95.5%] (847/900) | 80.0% [67.0%, 88.8%] (40/50) | 1733 / 1764 |
| 10 | 55.3% [47.3%, 63.1%] (83/150) | 91.6% [83.6%, 95.9%] (76/83) | 4.7% [2.3%, 9.3%] (7/150) | 83.1% [80.5%, 85.4%] (748/900) | 91.2% [89.2%, 92.9%] (821/900) | 78.3% [68.3%, 85.8%] (65/83) | 96 / 1141 |
| 30 | 82.7% [75.8%, 87.9%] (124/150) | 95.2% [89.8%, 97.8%] (118/124) | 4.0% [1.8%, 8.5%] (6/150) | 79.7% [76.9%, 82.2%] (717/900) | 87.6% [85.2%, 89.6%] (788/900) | 79.8% [71.9%, 86.0%] (99/124) | 5 / 1080 |
| 100 | 91.3% [85.7%, 94.9%] (137/150) | 99.3% [96.0%, 99.9%] (136/137) | 0.7% [0.1%, 3.7%] (1/150) | 76.7% [73.8%, 79.3%] (690/900) | 88.8% [86.5%, 90.7%] (799/900) | 73.7% [65.8%, 80.4%] (101/137) | 156 / 1377 |
| 300 | 98.7% [95.3%, 99.6%] (148/150) | 99.3% [96.3%, 99.9%] (147/148) | 0.7% [0.1%, 3.7%] (1/150) | 75.2% [72.3%, 77.9%] (677/900) | 89.4% [87.3%, 91.3%] (805/900) | 70.3% [62.5%, 77.0%] (104/148) | 293 / 1618 |
| 1,000 | 100.0% [97.5%, 100.0%] (150/150) | 98.7% [95.3%, 99.6%] (148/150) | 1.3% [0.4%, 4.7%] (2/150) | 67.8% [64.7%, 70.7%] (610/900) | 85.2% [82.8%, 87.4%] (767/900) | 66.0% [58.1%, 73.1%] (99/150) | 178 / 1695 |
| 3,000 | 100.0% [97.5%, 100.0%] (150/150) | 98.7% [95.3%, 99.6%] (148/150) | 1.3% [0.4%, 4.7%] (2/150) | 54.7% [51.4%, 57.9%] (492/900) | 81.6% [78.9%, 84.0%] (734/900) | 53.3% [45.4%, 61.1%] (80/150) | 188 / 1710 |

Top pick by hands observed (share of bots):

| Hands | Equilibrium | Restricted Nash, p = 0.1 | Restricted Nash, p = 0.25 | Restricted Nash, p = 0.5 | Restricted Nash, p = 0.75 | Restricted Nash, p = 0.9 | Best response |
|---|---|---|---|---|---|---|---|
| 0 | 67% | 33% | 0% | 0% | 0% | 0% | 0% |
| 10 | 45% | 11% | 7% | 13% | 17% | 7% | 1% |
| 30 | 17% | 7% | 11% | 12% | 24% | 14% | 15% |
| 100 | 9% | 3% | 7% | 4% | 19% | 25% | 34% |
| 300 | 1% | 0% | 2% | 7% | 11% | 35% | 44% |
| 1,000 | 0% | 0% | 0% | 3% | 8% | 33% | 57% |
| 3,000 | 0% | 0% | 0% | 0% | 1% | 24% | 75% |

| Hands | Bayes minus mode BR | Regret vs hindsight | Over population (posterior) | Over population (true) |
|---|---|---|---|---|
| 0 | +0.016 [+0.014, +0.018] | +0.011 [+0.010, +0.012] | -0.000 [-0.001, +0.001] | -0.000 [-0.001, +0.001] |
| 10 | +0.003 [-0.003, +0.008] | +0.056 [+0.045, +0.066] | +0.017 [+0.016, +0.019] | +0.017 [+0.014, +0.021] |
| 30 | +0.005 [-0.001, +0.010] | +0.030 [+0.025, +0.036] | +0.051 [+0.040, +0.062] | +0.054 [+0.044, +0.065] |
| 100 | +0.000 [+0.000, +0.001] | +0.012 [+0.004, +0.020] | +0.074 [+0.071, +0.078] | +0.079 [+0.076, +0.081] |
| 300 | +0.000 [-0.000, +0.000] | +0.003 [+0.003, +0.004] | +0.084 [+0.077, +0.091] | +0.091 [+0.084, +0.098] |
| 1,000 | -0.000 [-0.000, +0.000] | +0.001 [+0.000, +0.002] | +0.089 [+0.083, +0.096] | +0.094 [+0.087, +0.101] |
| 3,000 | -0.000 [-0.000, -0.000] | +0.000 [+0.000, +0.001] | +0.091 [+0.084, +0.099] | +0.095 [+0.088, +0.102] |

## Example reports (first seed, one standard bot per archetype)

* **nit, 10 hands.** Seen 10 hands. As small blind it pushed 1 of 5; facing a push it called 0 of 2; its cards were shown 1 time. Estimated ranges: pushes 23% of hands (90% interval 6% to 48%) and calls 26% (5% to 58%); the equilibrium at 10 bb pushes 58% and calls 37%. Play the equilibrium: no deviation is likely enough to gain. The one with the highest expected gain, the best response to the model, would gain +0.078 bb/hand on average but does worse than the equilibrium with probability 29%.
* **nit, 100 hands.** Seen 100 hands. As small blind it pushed 6 of 50; facing a push it called 1 of 25; its cards were shown 5 times. Estimated ranges: pushes 15% of hands (90% interval 9% to 22%) and calls 7% (3% to 15%); the equilibrium at 10 bb pushes 58% and calls 37%. Recommended: the best response to the model. As small blind, push 100% of hands (equilibrium 58%): adds Q6o, Q5o, J7o, Q4o and 52 more. As big blind, call 10% of hands (equilibrium 37%): drops KJs, A8s, KTs, KQo and 46 more. Expected gain over the equilibrium +0.253 bb/hand (90% interval +0.171 to +0.308; probability of gaining 100%), +0.251 over the population recommendation. Exploitability 0.464 bb/hand.
* **nit, 1,000 hands.** Seen 1,000 hands. As small blind it pushed 107 of 500; facing a push it called 15 of 298; its cards were shown 57 times. Estimated ranges: pushes 20% of hands (90% interval 17% to 23%) and calls 6% (4% to 8%); the equilibrium at 10 bb pushes 58% and calls 37%. Recommended: the best response to the model. As small blind, push 100% of hands (equilibrium 58%): adds Q6o, Q5o, J7o, Q4o and 52 more. As big blind, call 13% of hands (equilibrium 37%): drops KTs, KJo, QJs, K9s and 39 more. Expected gain over the equilibrium +0.269 bb/hand (90% interval +0.245 to +0.289; probability of gaining 100%), +0.267 over the population recommendation. Exploitability 0.407 bb/hand.
* **tag, 10 hands.** Seen 10 hands. As small blind it pushed 3 of 5; facing a push it called 0 of 2; its cards were shown 1 time. Estimated ranges: pushes 52% of hands (90% interval 27% to 81%) and calls 27% (5% to 61%); the equilibrium at 10 bb pushes 58% and calls 37%. Play the equilibrium: no deviation is likely enough to gain. The one with the highest expected gain, the best response to the model, would gain +0.056 bb/hand on average but does worse than the equilibrium with probability 35%.
* **tag, 100 hands.** Seen 100 hands. As small blind it pushed 26 of 50; facing a push it called 10 of 24; its cards were shown 17 times. Estimated ranges: pushes 51% of hands (90% interval 44% to 60%) and calls 30% (22% to 39%); the equilibrium at 10 bb pushes 58% and calls 37%. Recommended: the restricted Nash response (p = 0.1). As small blind, push 59% of hands (equilibrium 58%), with the same hands played at least half the time. As big blind, call 37% of hands (equilibrium 37%), with the same hands played at least half the time. Expected gain over the equilibrium +0.001 bb/hand (90% interval +0.000 to +0.001; probability of gaining 95%), +0.000 over the population recommendation. Exploitability 0.000 bb/hand.
* **tag, 1,000 hands.** Seen 1,000 hands. As small blind it pushed 255 of 500; facing a push it called 82 of 273; its cards were shown 175 times. Estimated ranges: pushes 51% of hands (90% interval 48% to 54%) and calls 29% (27% to 31%); the equilibrium at 10 bb pushes 58% and calls 37%. Recommended: the restricted Nash response (p = 0.9). As small blind, push 82% of hands (equilibrium 58%): adds Q6o, Q5o, J7o, Q4o and 32 more. As big blind, call 35% of hands (equilibrium 37%): drops JTo, Q7s, T9s, J8s and 1 more. Expected gain over the equilibrium +0.024 bb/hand (90% interval +0.014 to +0.035; probability of gaining 100%), +0.024 over the population recommendation. Exploitability 0.056 bb/hand.
* **lag, 10 hands.** Seen 10 hands. As small blind it pushed 5 of 5; facing a push it called 1 of 2; its cards were shown 1 time. Estimated ranges: pushes 80% of hands (90% interval 56% to 96%) and calls 35% (11% to 69%); the equilibrium at 10 bb pushes 58% and calls 37%. Recommended: the restricted Nash response (p = 0.1). As small blind, push 59% of hands (equilibrium 58%): adds Q6o; drops 43s. As big blind, call 38% of hands (equilibrium 37%): adds Q6s. Expected gain over the equilibrium +0.004 bb/hand (90% interval +0.000 to +0.007; probability of gaining 95%), +0.003 over the population recommendation. Exploitability 0.000 bb/hand.
* **lag, 100 hands.** Seen 100 hands. As small blind it pushed 34 of 50; facing a push it called 14 of 26; its cards were shown 24 times. Estimated ranges: pushes 66% of hands (90% interval 58% to 74%) and calls 50% (37% to 64%); the equilibrium at 10 bb pushes 58% and calls 37%. Recommended: the restricted Nash response (p = 0.5). As small blind, push 53% of hands (equilibrium 58%): adds Q6o; drops J4s, J3s, T5s, T4s and 10 more. As big blind, call 40% of hands (equilibrium 37%): adds Q8o, Q6s, K4o. Expected gain over the equilibrium +0.012 bb/hand (90% interval +0.002 to +0.024; probability of gaining 97%), +0.011 over the population recommendation. Exploitability 0.004 bb/hand.
* **lag, 1,000 hands.** Seen 1,000 hands. As small blind it pushed 357 of 500; facing a push it called 145 of 293; its cards were shown 279 times. Estimated ranges: pushes 70% of hands (90% interval 68% to 73%) and calls 50% (46% to 53%); the equilibrium at 10 bb pushes 58% and calls 37%. Recommended: the restricted Nash response (p = 0.9). As small blind, push 48% of hands (equilibrium 58%): drops J6s, J5s, T8o, J4s and 20 more. As big blind, call 44% of hands (equilibrium 37%): adds Q8o, Q6s, J9o, Q5s and 8 more. Expected gain over the equilibrium +0.023 bb/hand (90% interval +0.017 to +0.028; probability of gaining 100%), +0.022 over the population recommendation. Exploitability 0.022 bb/hand.
* **calling_station, 10 hands.** Seen 10 hands. As small blind it pushed 3 of 5; facing a push it called 1 of 2; its cards were shown 2 times. Estimated ranges: pushes 55% of hands (90% interval 33% to 80%) and calls 48% (28% to 73%); the equilibrium at 10 bb pushes 58% and calls 37%. Play the equilibrium: no deviation is likely enough to gain. The one with the highest expected gain, the best response to the model, would gain +0.009 bb/hand on average but does worse than the equilibrium with probability 33%.
* **calling_station, 100 hands.** Seen 100 hands. As small blind it pushed 19 of 50; facing a push it called 16 of 32; its cards were shown 24 times. Estimated ranges: pushes 40% of hands (90% interval 34% to 47%) and calls 52% (47% to 59%); the equilibrium at 10 bb pushes 58% and calls 37%. Recommended: the restricted Nash response (p = 0.9). As small blind, push 46% of hands (equilibrium 58%): drops J8o, T7s, J6s, Q2s and 25 more. As big blind, call 30% of hands (equilibrium 37%): drops Q8s, J9s, Q9o, JTo and 9 more. Expected gain over the equilibrium +0.029 bb/hand (90% interval +0.019 to +0.038; probability of gaining 100%), +0.028 over the population recommendation. Exploitability 0.040 bb/hand.
* **calling_station, 1,000 hands.** Seen 1,000 hands. As small blind it pushed 167 of 500; facing a push it called 131 of 296; its cards were shown 193 times. Estimated ranges: pushes 35% of hands (90% interval 33% to 38%) and calls 46% (44% to 49%); the equilibrium at 10 bb pushes 58% and calls 37%. Recommended: the best response to the model. As small blind, push 46% of hands (equilibrium 58%): drops Q7o, J8o, J6s, Q2s and 23 more. As big blind, call 25% of hands (equilibrium 37%): drops K8s, QJo, K9o, Q9s and 19 more. Expected gain over the equilibrium +0.026 bb/hand (90% interval +0.021 to +0.031; probability of gaining 100%), +0.026 over the population recommendation. Exploitability 0.102 bb/hand.
* **maniac, 10 hands.** Seen 10 hands. As small blind it pushed 5 of 5; facing a push it called 0 of 0; its cards were shown 0 times. Estimated ranges: pushes 80% of hands (90% interval 55% to 96%) and calls 40% (6% to 80%); the equilibrium at 10 bb pushes 58% and calls 37%. Recommended: the restricted Nash response (p = 0.1). As small blind, push 59% of hands (equilibrium 58%): adds Q6o; drops 84s, 43s. As big blind, call 38% of hands (equilibrium 37%): adds Q6s. Expected gain over the equilibrium +0.005 bb/hand (90% interval +0.000 to +0.008; probability of gaining 97%), +0.003 over the population recommendation. Exploitability 0.000 bb/hand.
* **maniac, 100 hands.** Seen 100 hands. As small blind it pushed 48 of 50; facing a push it called 12 of 27; its cards were shown 26 times. Estimated ranges: pushes 91% of hands (90% interval 83% to 96%) and calls 48% (38% to 57%); the equilibrium at 10 bb pushes 58% and calls 37%. Recommended: the restricted Nash response (p = 0.75). As small blind, push 49% of hands (equilibrium 58%): drops J5s, T8o, J4s, T6s and 16 more. As big blind, call 54% of hands (equilibrium 37%): adds Q8o, Q6s, J9o, Q5s and 25 more. Expected gain over the equilibrium +0.077 bb/hand (90% interval +0.048 to +0.100; probability of gaining 100%), +0.076 over the population recommendation. Exploitability 0.051 bb/hand.
* **maniac, 1,000 hands.** Seen 1,000 hands. As small blind it pushed 470 of 500; facing a push it called 141 of 300; its cards were shown 322 times. Estimated ranges: pushes 93% of hands (90% interval 91% to 95%) and calls 50% (47% to 53%); the equilibrium at 10 bb pushes 58% and calls 37%. Recommended: the restricted Nash response (p = 0.9). As small blind, push 47% of hands (equilibrium 58%): drops J8o, J6s, Q2s, J5s and 23 more. As big blind, call 60% of hands (equilibrium 37%): adds Q8o, Q6s, J9o, Q5s and 34 more. Expected gain over the equilibrium +0.098 bb/hand (90% interval +0.088 to +0.107; probability of gaining 100%), +0.097 over the population recommendation. Exploitability 0.084 bb/hand.
