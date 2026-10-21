# P2.4: Deep CFR on Leduc vs tabular MCCFR at matched traversals

5 seeds; 1,000 traversals per player per iteration; advantage networks (64, 64, 64) (11,075 parameters, 39 input features), retrained from scratch for 600 Adam steps per iteration; torch 2.14.1+cpu. Exploitability in chips per hand (geometric mean over seeds, 95% t-interval on the log scale). The uniform policy is at 2.37.

| Method | 10 it. (20,000 trav.) | 30 it. (60,000 trav.) | 100 it. (200,000 trav.) |
|---|---|---|---|
| Tabular ES-MCCFR, linear weights | 0.240 [0.211, 0.273] | 0.141 [0.133, 0.150] | 0.072 [0.069, 0.075] |
| Deep CFR with a perfect regressor | 0.563 [0.504, 0.628] | 0.180 [0.157, 0.206] | 0.087 [0.074, 0.102] |
| Deep CFR: exact average of played strategies (SD-CFR) | 0.540 [0.405, 0.720] | 0.230 [0.194, 0.272] | 0.160 [0.136, 0.188] |
| Deep CFR: strategy-memory average (Pi's target) | 0.542 [0.411, 0.715] | 0.230 [0.196, 0.271] | 0.161 [0.135, 0.192] |
| Deep CFR: policy network (its output) | 0.537 [0.408, 0.707] | 0.230 [0.190, 0.278] | 0.161 [0.134, 0.193] |
| Deep CFR: current strategy | 0.843 [0.631, 1.125] | 0.972 [0.687, 1.375] | 1.196 [0.677, 2.113] |

Deep CFR wall-clock: 5.7 s per iteration on average (both players, single thread; machine-dependent). Every other number is reproduced exactly by the seed on the same machine and thread count.
