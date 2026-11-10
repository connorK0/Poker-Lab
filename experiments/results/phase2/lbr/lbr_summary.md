# P2.8b: local best response (lower bounds on exploitability)

Seed 2026, 1152 s. Values in mbb/hand (1,000 = one big blind per hand), mean over duplicate deals with a 95% interval. LBR is a lower bound: the exploitability is at least this large, not equal to it.

| Opponent | Stack | LBR actions | LBR value | Exact | Raw (no all-in adj.) | ms/hand |
|---|---|---|---|---|---|---|
| Always check or call | 100 bb | fold/call/pot/allin | 16,347 [15,339, 17,356] |  | 16,459 [15,398, 17,519] | 88 |
| Check or fold | 100 bb | fold/call/pot/allin | 750 [750, 750] |  | 750 [750, 750] | 1 |
| Uniform random | 100 bb | fold/call/pot/allin | 5,843 [5,173, 6,513] |  | 6,275 [4,897, 7,653] | 43 |
| Pot odds (play page's advisor baseline) | 100 bb | fold/call/pot/allin | 7,559 [6,911, 8,206] |  | 7,534 [6,296, 8,771] | 166 |
| Bot: nit | 100 bb | fold/call/pot/allin | 907 [720, 1,094] |  | 1,003 [754, 1,252] | 21 |
| Bot: tight-aggressive | 100 bb | fold/call/pot/allin | 1,370 [1,126, 1,614] |  | 1,432 [1,101, 1,764] | 30 |
| Bot: loose-aggressive | 100 bb | fold/call/pot/allin | 1,541 [1,187, 1,896] |  | 1,348 [832, 1,864] | 39 |
| Bot: calling station | 100 bb | fold/call/pot/allin | 5,797 [5,008, 6,587] |  | 6,201 [5,134, 7,267] | 128 |
| Bot: maniac | 100 bb | fold/call/pot/allin | 5,051 [4,403, 5,700] |  | 4,849 [3,784, 5,914] | 97 |
| Push/fold Nash | 10 bb | fold/allin | 1 [-0, 2] | -0 | -2 [-7, 4] | 2 |
| Push any two, call any two | 10 bb | fold/allin | 449 [421, 476] | 473 | 379 [276, 483] | 5 |
| Push top 50%, call top 20% | 10 bb | fold/allin | 166 [141, 191] | 151 | 194 [131, 256] | 2 |
| Push top 15%, call top 60% | 10 bb | fold/allin | 183 [160, 206] | 197 | 157 [83, 231] | 2 |
| Push/fold Nash + pot-odds fallback (LBR with all four actions) | 10 bb | fold/call/pot/allin | 299 [241, 358] |  | 297 [232, 362] | 143 |

## Ablations (paired with the main row's deals)

| Row | LBR value | Difference from the main row |
|---|---|---|
| norange_pot_odds | 3,386 [2,167, 4,605] | -3,146 [-4,329, -1,964] |
| exactflop_pot_odds | 6,736 [5,587, 7,885] | 204 [-479, 887] |
| norange_bot_tag | -642 [-1,609, 325] | -2,019 [-3,010, -1,028] |
| exactflop_bot_tag | 1,315 [868, 1,763] | -62 [-144, 20] |

## Pre-registered predictions

| | Claim | Result |
|---|---|---|
| C1 | Check: against check/fold LBR wins exactly the blinds, 750 mbb/hand (the most any strategy can win from it) | holds |
| C2 | Check: in the push/fold game (LBR folds or goes all-in) each row is within 3 standard errors of P3.1's exact exploitability | holds |
| C3 | Check: against push/fold Nash LBR wins between -10 and +10 mbb/hand | holds |
| C4 | Check: the range models never give probability 0 to an action the opponent took | holds |
| P1 | Every deep strategy other than check/fold loses at least 500 mbb/hand to LBR (lower end of the 95% interval) | holds |
| P2 | Always check/call is the most exploitable: LBR wins at least 14,000 mbb/hand | holds |
| P3 | The play page's advisor baseline (pot odds) loses 5,500 to 8,500 mbb/hand to LBR | holds |
| P4 | Uniform random loses 4,500 to 7,500 mbb/hand to LBR | holds |
| P5 | The bots order maniac > loose-aggressive > tight-aggressive > nit | holds |
| P6 | The loose, passive and wild bots (calling station, maniac) lose at least 4,000 mbb/hand; the nit, TAG and LAG at most 2,500 | holds |
| P6b | (second half of P6) | holds |
| P7 | Without range tracking LBR wins less, on the same deals (upper end of the paired interval below 0) | holds |
| P8 | Using all 1,176 flop runouts instead of 300 changes LBR's value by less than 800 mbb/hand on the same deals | holds |
| P9 | Push/fold Nash with a pot-odds fallback is exploitable at 10 bb once LBR may limp or bet the pot: at least 150 mbb/hand (lower end of the interval) | holds |
| P10 | Scoring all-ins at their expectation never increases the variance of a deep row | holds |
