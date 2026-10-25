# P3.6: importing PHH hand histories

Seed 2026, commit 3c1f25c4bc7366ee550e95d42dfcd4a891ff834a (dirty: False), 40 s. Dataset https://github.com/uoftcprg/phh-dataset.git at e47fbd5, 10088 files, content SHA-256 ebd84f50be142b5d...

## A. Round trip at scale

10,000 hands at 400 tables, each also redacted: 20,000 of 20,000 round trips identical (100.00%); 6.5 ms each. Prediction (all identical): holds.

Hands with: all_in 8,453, antes 4,350, cents 3,025, heads_up 1,250, showdown 8,201, side_pot 5,643.

## B. Fault injection

1,000 simulated and 1,000 Pluribus hands, one mutation per operator. Rejected share of effective mutations, with and without finishing stacks; silent = accepted with a changed result.

| Source | Operator | Rejected (with) | Record changed (with) | Silent (with) | Rejected (without) | Record changed (without) | Silent (without) |
|---|---|---|---|---|---|---|---|
| Simulated hands (part A) | Drop a decision | 100.0% (994/994) | 0.0% (0/994) | 0 | 100.0% (994/994) | 0.0% (0/994) | 0.0% (0/994) |
| Simulated hands (part A) | Swap two decisions | 100.0% (968/968) | 0.0% (0/968) | 0 | 100.0% (968/968) | 0.0% (0/968) | 0.0% (0/968) |
| Simulated hands (part A) | Cut the actions short | 100.0% (1000/1000) | 0.0% (0/1000) | 0 | 100.0% (1000/1000) | 0.0% (0/1000) | 0.0% (0/1000) |
| Simulated hands (part A) | Decision by another player | 100.0% (994/994) | 0.0% (0/994) | 0 | 100.0% (994/994) | 0.0% (0/994) | 0.0% (0/994) |
| Simulated hands (part A) | Delete a TOML delimiter | 100.0% (1000/1000) | 0.0% (0/1000) | 0 | 100.0% (1000/1000) | 0.0% (0/1000) | 0.0% (0/1000) |
| Simulated hands (part A) | Change a bet amount | 78.6% (726/924) | 21.4% (198/924) | 0 | 64.6% (597/924) | 21.4% (198/924) | 14.0% (129/924) |
| Simulated hands (part A) | Change a card | 56.7% (567/1000) | 43.3% (433/1000) | 0 | 49.5% (495/1000) | 43.3% (433/1000) | 7.2% (72/1000) |
| Simulated hands (part A) | Change a starting stack | 100.0% (1000/1000) | 0.0% (0/1000) | 0 | 18.6% (186/1000) | 67.1% (671/1000) | 14.3% (143/1000) |
| Simulated hands (part A) | Move chips between finishing stacks | 100.0% (1000/1000) | 0.0% (0/1000) | 0 | n/a | n/a | n/a |
| Pluribus hands | Drop a decision | 100.0% (1000/1000) | 0.0% (0/1000) | 0 | 100.0% (1000/1000) | 0.0% (0/1000) | 0.0% (0/1000) |
| Pluribus hands | Swap two decisions | 100.0% (1000/1000) | 0.0% (0/1000) | 0 | 100.0% (1000/1000) | 0.0% (0/1000) | 0.0% (0/1000) |
| Pluribus hands | Cut the actions short | 100.0% (1000/1000) | 0.0% (0/1000) | 0 | 100.0% (1000/1000) | 0.0% (0/1000) | 0.0% (0/1000) |
| Pluribus hands | Decision by another player | 100.0% (1000/1000) | 0.0% (0/1000) | 0 | 100.0% (1000/1000) | 0.0% (0/1000) | 0.0% (0/1000) |
| Pluribus hands | Delete a TOML delimiter | 100.0% (1000/1000) | 0.0% (0/1000) | 0 | 100.0% (1000/1000) | 0.0% (0/1000) | 0.0% (0/1000) |
| Pluribus hands | Change a bet amount | 51.8% (461/890) | 48.2% (429/890) | 0 | 16.5% (147/890) | 48.2% (429/890) | 35.3% (314/890) |
| Pluribus hands | Change a card | 29.3% (293/1000) | 70.7% (707/1000) | 0 | 29.1% (291/1000) | 70.7% (707/1000) | 0.2% (2/1000) |
| Pluribus hands | Change a starting stack | 100.0% (1000/1000) | 0.0% (0/1000) | 0 | 0.3% (3/1000) | 99.6% (996/1000) | 0.1% (1/1000) |
| Pluribus hands | Move chips between finishing stacks | 100.0% (1000/1000) | 0.0% (0/1000) | 0 | n/a | n/a | n/a |

* Prediction 1 (no crashes): 0 crashes; holds.
* Prediction 2 (structural rejected >= 99%): minimum 100.0%; holds.
* Prediction 3 (no silent result change with finishing stacks): 0 silent; holds.

## C. Public research data

10,001 of 10,088 files' hands accepted, 87 rejected.

| Subset | Files | Accepted | Rejected (by reason) |
|---|---|---|---|
| historical | 5 | 1 | unsupported 4 |
| pluribus | 10,000 | 10,000 | none |
| wsop | 83 | 0 | unsupported 83 |

Rejections by kind:

* wsop, unsupported: variant 'FO/8' (14)
* wsop, unsupported: variant 'F7S' (13)
* wsop, unsupported: antes differ by seat (big-blind ante) (11)
* wsop, unsupported: variant 'FR' (10)
* wsop, unsupported: variant 'PO' (7)
* wsop, unsupported: variant 'FT' (7)
* wsop, unsupported: variant 'N2L1D' (7)
* wsop, unsupported: variant 'F7S/8' (7)
* wsop, unsupported: variant 'F2L3D' (7)
* historical, unsupported: variant 'FB' (1)
* historical, unsupported: variant 'PO' (1)
* historical, unsupported: variant 'F2L3D' (1)
* historical, unsupported: variant 'NS' (1)

* Prediction 1 (PokerKit's own loader agrees on every result): 9,993 of 10,001; **does not hold**. Post hoc: 8 of the disagreements differ only in who gets a split pot's odd chip, 0 otherwise; 8 Pluribus hands record half chips.
* Prediction 2 (export then import reproduces every hand): 10,001 of 10,001; holds.

## D. Pluribus hands: per-player statistics (descriptive)

| Player | Hands | VPIP | PFR | 3-bet | C-bet | Fold to c-bet | WTSD | W$SD | AF |
|---|---|---|---|---|---|---|---|---|---|
| All humans | 50,000 | 26.8 | 17.8 | 7.9 | 51.9 | 46.2 | 30.1 | 52.3 | 2.25 |
| Pluribus | 10,000 | 26.9 | 18.0 | 7.3 | 53.7 | 45.0 | 30.6 | 48.8 | 2.33 |
| MrBlue | 9,121 | 32.1 | 19.2 | 5.9 | 46.7 | 51.5 | 28.2 | 55.8 | 1.93 |
| MrOrange | 7,512 | 21.1 | 14.5 | 7.4 | 46.0 | 40.4 | 30.9 | 54.2 | 1.99 |
| Bill | 6,713 | 27.9 | 20.2 | 10.7 | 54.5 | 38.5 | 32.4 | 46.5 | 2.12 |
| MrPink | 6,055 | 26.7 | 17.7 | 9.3 | 59.5 | 47.8 | 30.9 | 47.4 | 3.04 |
| Eddie | 5,510 | 27.7 | 17.9 | 6.2 | 61.0 | 40.2 | 29.1 | 56.7 | 2.97 |
| MrWhite | 4,483 | 26.4 | 17.4 | 7.0 | 50.3 | 51.6 | 26.9 | 54.9 | 1.92 |
| MrBlonde | 2,560 | 23.0 | 17.2 | 9.9 | 41.4 | 49.5 | 30.9 | 61.3 | 1.70 |
| Budd | 2,509 | 25.5 | 18.4 | 8.5 | 54.5 | 50.5 | 32.4 | 50.7 | 2.86 |
| Joe | 1,535 | 22.7 | 17.6 | 9.4 | 48.8 | 33.3 | 31.8 | 48.0 | 2.52 |
| MrBrown | 1,378 | 25.9 | 15.8 | 9.0 | 36.1 | 46.7 | 33.5 | 54.7 | 1.79 |
| Hattori | 1,365 | 27.1 | 18.5 | 5.7 | 43.7 | 45.3 | 33.8 | 41.6 | 1.95 |
| ORen | 771 | 27.1 | 16.7 | 7.5 | 62.7 | 73.1 | 29.8 | 48.9 | 5.13 |
| Gogo | 488 | 32.6 | 23.4 | 11.4 | 71.2 | 31.7 | 27.5 | 43.3 | 2.07 |

Phase 1 bots (measured, mean over bots):

| Archetype | VPIP | PFR | 3-bet | C-bet | Fold to c-bet | WTSD | W$SD | AF |
|---|---|---|---|---|---|---|---|---|
| nit | 10.9 | 6.8 | 2.3 | 56.8 | 54.0 | 44.3 | 59.9 | 2.34 |
| tag | 19.8 | 14.4 | 5.4 | 70.1 | 49.4 | 43.5 | 54.9 | 2.97 |
| lag | 29.4 | 21.8 | 12.5 | 79.2 | 41.1 | 47.2 | 50.2 | 3.51 |
| calling_station | 44.1 | 5.7 | 2.6 | 29.3 | 19.0 | 70.0 | 47.4 | 0.33 |
| maniac | 54.6 | 38.8 | 27.9 | 85.8 | 33.1 | 51.0 | 41.8 | 3.62 |
