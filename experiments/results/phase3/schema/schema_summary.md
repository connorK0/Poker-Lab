# P3.10: schema v2 and the migration of version 1 files

Seed 2026, commit 568cde30a69cbc7a74fc1ad2a3b6173f1175c504 (dirty: False), 62 s. Dataset https://github.com/uoftcprg/phh-dataset.git at e47fbd5.

## A. Round trips at v2 tables

10,000 hands at 400 tables; 79,995 of 80,000 checks passed; 40.5 ms per hand for all checks.

| Structure | Tables | Hands | Forced posts and consistency (find_problems) | Unmodified PokerKit replays it | JSON round trip | PHH export and import | PokerKit's own PHH loader agrees | Redacted copies (rules, JSON, PHH) |
|---|---|---|---|---|---|---|---|---|
| Blinds only | 50 | 1,250 | 1,250/1,250 | 1,250/1,250 | 1,250/1,250 | 1,250/1,250 | 1,250/1,250 (6 odd chip) | 3,750/3,750 |
| Ante for every seat | 50 | 1,250 | 1,250/1,250 | 1,250/1,250 | 1,250/1,250 | 1,250/1,250 | 1,250/1,250 (7 odd chip) | 3,750/3,750 |
| Big-blind ante | 50 | 1,250 | 1,250/1,250 | 1,250/1,250 | 1,250/1,250 | 1,250/1,250 | 1,249/1,250 (4 odd chip) | 3,750/3,750 |
| Straddle | 50 | 1,250 | 1,250/1,250 | 1,250/1,250 | 1,250/1,250 | 1,250/1,250 | 1,250/1,250 (10 odd chip) | 3,750/3,750 |
| Double straddle | 50 | 1,250 | 1,250/1,250 | 1,250/1,250 | 1,250/1,250 | 1,250/1,250 | 1,247/1,250 (20 odd chip) | 3,750/3,750 |
| Straddle on the button | 50 | 1,250 | 1,250/1,250 | 1,250/1,250 | 1,250/1,250 | 1,250/1,250 | 1,250/1,250 (10 odd chip) | 3,750/3,750 |
| Minimum bet 2 bb | 50 | 1,250 | 1,250/1,250 | 1,250/1,250 | 1,250/1,250 | 1,250/1,250 | 1,250/1,250 (7 odd chip) | 3,750/3,750 |
| Minimum bet 0.5 bb | 50 | 1,250 | 1,250/1,250 | 1,250/1,250 | 1,250/1,250 | 1,250/1,250 | 1,249/1,250 (4 odd chip) | 3,750/3,750 |

Hands with: all_in 8,601, all_in_post 1,104, antes 1,250, bb_ante_posted 1,250, cents 3,250, heads_up 1,075, showdown 8,509, side_pot 6,090, straddle_posted 3,750.

* Prediction A1 (every check passes): **does not hold**.

Failures (first 50):

* {'hand': 't3333489054-000010', 'check': 'full/loader', 'why': "loader ['-2.00', '0.6666666666666666666666666700', '0', '4.666666666666666666666666670', '4.666666666666666666666666670', '-8.00', '0'] != [-2, 2, 0, 4, 4, -8, 0]"}
* {'hand': 't3026452634-000021', 'check': 'full/loader', 'why': "loader ['-5462.00', '1434.666666666666666666666667', '5742.00', '-400', '-400', '825.6666666666666666666666670', '-2122.00', '381.6666666666666666666666667', '0'] != [-5462, 1436, 5742, -400, -400, 825, -2122, 381, 0]"}
* {'hand': 't3188062555-000024', 'check': 'full/loader', 'why': "loader ['-1.00', '-2.00', '-6.00', '-8.00', '-69.00', '34.16666666666666666666666666', '41.16666666666666666666666670', '10.66666666666666666666666667'] != [-1, -2, -6, -8, -69, 36, 40, 10]"}
* {'hand': 't1536933253-000022', 'check': 'full/loader', 'why': "loader ['-22.00', '-10.0', '21.66666666666666666666666660', '21.66666666666666666666666670', '21.66666666666666666666666670', '-33.00', '0', '0'] != [-22, -10, 23, 21, 21, -33, 0, 0]"}
* {'hand': 't3366301983-000021', 'check': 'full/loader', 'why': "loader ['-100', '-506.00', '-100', '2348.666666666666666666666667', '-6758.00', '23412.66666666666666666666667', '-599.00', '-19330.0', '1631.666666666666666666666667'] != [-100, -506, -100, 2350, -6758, 23412, -599, -19330, 1631]"}

Post hoc: every hand on which PokerKit's loader disagreed, checked again with the same hand in whole units: larger/agrees 5, odd chip/agrees 68.

## B. Version 1 files

Written by the schema v1 code at 0cf0c4f (SHA-256 of their contents f769f19bb8fb...).

| Source | v1 records | Identical to the current code |
|---|---|---|
| P3.6 part A tables (400) | 10,000 | 10,000 |
| Pluribus hands (10,000 files) | 10,000 | 10,000 |

Pluribus hands rejected: 0 by the v1 code, 0 now. The simulated hands have the same feature counts as P3.6 part A's: yes.

* Prediction B1 (every record migrates to the same hand): holds.

## C. The WSOP hold'em hands

| Subset | Files | No-limit hold'em | Imported |
|---|---|---|---|
| historical | 5 | 1 | 1 |
| wsop | 83 | 11 | 11 |

Rejected, by variant: F2L3D 8, F7S 13, F7S/8 7, FB 1, FO/8 14, FR 10, FT 7, N2L1D 7, NS 1, PO 8.

| Hand | Players | Big-blind ante (bb) | Pot (bb) | Showdown | Loader agrees | Export, import |
|---|---|---|---|---|---|---|
| dwan-ivey-2009 | 3 | 0 | 840.8 | yes | yes | same |
| wsop/2023/43/5/00-02-07 | 5 | 1.5 | 32.9 | yes | yes | same |
| wsop/2023/43/5/00-08-38 | 5 | 1.5 | 77.4 | no | yes | same |
| wsop/2023/43/5/00-15-36 | 5 | 1.5 | 18.8 | no | yes | same |
| wsop/2023/43/5/00-18-39 | 5 | 1.5 | 10.0 | no | yes | same |
| wsop/2023/43/5/02-51-10 | 5 | 1.5 | 18.2 | no | yes | same |
| wsop/2023/43/5/02-53-09 | 5 | 1.5 | 18.8 | no | yes | same |
| wsop/2023/43/5/02-54-12 | 5 | 1.5 | 19.2 | no | yes | same |
| wsop/2023/43/5/02-56-12 | 5 | 1.5 | 21.3 | no | yes | same |
| wsop/2023/43/5/02-57-27 | 5 | 1.5 | 9.7 | no | yes | same |
| wsop/2023/43/5/03-00-32 | 5 | 1.5 | 15.2 | no | yes | same |
| wsop/2023/43/5/03-02-41 | 5 | 1.5 | 46.7 | yes | yes | same |

* Prediction C1 (every no-limit hold'em hand imports, only variants rejected): 12 of 12; holds.
* Prediction C2 (PokerKit's own loader agrees): 12 of 12; holds.
* Prediction C3 (export then import reproduces every hand): 12 of 12; holds.
