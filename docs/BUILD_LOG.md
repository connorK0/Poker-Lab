# Build log: how this project was built

This log explains how each step of the project was done: the poker knowledge, the statistics
and machine learning, the software techniques, the decisions (and the alternatives rejected),
how each piece was verified, and the mistakes made along the way. Read it front to back and
you should be able to rebuild the project yourself; jump to a step to understand one part of
the code.

**Who it is for.** Someone comfortable with Python and basic probability (Bayes' rule,
expectation, variance) and with a first course in machine learning (logistic regression,
train/test splits). No poker knowledge and no empirical Bayes are assumed; Part A teaches both.

**How it is kept.** The log is updated in the same commit as the work it describes. Each step
links to the code and to the tests that verify it. Dates are in the
[changelog](#changelog-of-this-log).

**Math notation.** $\mathbb E$ is expectation, $\operatorname{Var}$ variance, $\hat\theta$ an
estimate of $\theta$, $\log$ the natural log, $\binom{m}{k}$ a binomial coefficient, and
$B(a,b)$ the beta function.

## Contents

* [Part A: background](#part-a-background)
  * [A.1 No-limit hold'em, precisely](#a1-no-limit-holdem-precisely)
  * [A.2 Positions and the order of action](#a2-positions-and-the-order-of-action)
  * [A.3 HUD statistics and why denominators matter](#a3-hud-statistics-and-why-denominators-matter)
  * [A.4 Player archetypes](#a4-player-archetypes)
  * [A.5 Toolchain](#a5-toolchain)
  * [A.6 Game theory in one page](#a6-game-theory-in-one-page)
* [Part B: the steps](#part-b-the-steps)
  * [Step 0: Bootstrap and working agreements](#step-0-bootstrap-and-working-agreements)
  * [Step 1: Choosing the engine and evaluator by measurement](#step-1-choosing-the-engine-and-evaluator-by-measurement)
  * [Step 2: The hand-history schema](#step-2-the-hand-history-schema-m1)
  * [Step 3: The information firewall](#step-3-the-information-firewall-m1)
  * [Step 4: The simulation engine](#step-4-the-simulation-engine-m2)
  * [Step 5: Hand strength](#step-5-hand-strength-m3)
  * [Step 6: Parameterized bots and archetypes](#step-6-parameterized-bots-and-archetypes-m3)
  * [Step 7: Stats with explicit opportunity counting](#step-7-stats-with-explicit-opportunity-counting)
  * [Step 8: Running a league](#step-8-running-a-league-parallel-reproducible-simulation)
  * [Step 9: The sanity check](#step-9-the-sanity-check-tuning-against-pre-registered-tendencies-m3)
  * [Step 10: Empirical-Bayes shrinkage](#step-10-empirical-bayes-shrinkage-m4)
  * [Step 11: The archetype classifier and calibration](#step-11-the-archetype-classifier-and-calibration-m4)
  * [Step 12: Parameter regression](#step-12-parameter-regression-m4)
  * [Step 13: Next-action prediction](#step-13-next-action-prediction-m4)
  * [Step 14: Experimental design](#step-14-experimental-design-m5)
  * [Step 15: Phase 1 results and how to read them](#step-15-phase-1-results-and-how-to-read-them-m5)
  * [Step 16: Interfaces for Phases 2-5](#step-16-interfaces-for-phases-2-5-m6)
  * Phase 2
  * [Step 17: Small games and a tree compiler](#step-17-small-games-and-a-tree-compiler-p21)
  * [Step 18: Exact evaluation and an LP solver](#step-18-exact-evaluation-best-responses-exploitability-and-an-lp-solver-p22)
  * [Step 19: CFR and its variants](#step-19-cfr-and-its-variants-p23)
  * [Step 20: Deep CFR in PyTorch](#step-20-deep-cfr-in-pytorch-p24-p210)
  * [Step 21: Leduc archetypes and modeling them](#step-21-leduc-archetypes-and-modeling-them-from-what-a-player-can-see-p26)
  * [Step 22: Exploiting the models, and how much it risks](#step-22-exploiting-the-models-and-how-much-it-risks-p27)
  * [Step 23: Contracts, exact estimates, duplicate dealing](#step-23-contracts-for-small-game-strategies-exact-exploitability-estimates-duplicate-dealing-p25-p28-p29)
  * [Step 24: Phase 2 wrap-up](#step-24-phase-2-wrap-up-one-command-and-reproduction-runs)
  * Results explorer (all phases)
  * [Step 25: The results explorer](#step-25-the-results-explorer-one-interactive-page-for-every-phase)
  * Phase 3
  * [Step 26: Heads-up push/fold solved exactly](#step-26-a-bridge-to-holdem-heads-up-pushfold-solved-exactly-p31)
  * [Step 27: Modeling and exploiting push/fold opponents](#step-27-modeling-and-exploiting-pushfold-opponents-p32)
  * [Step 28: Playing the solved game in real hands](#step-28-playing-the-solved-game-in-real-hands-behind-the-firewall-p33)
  * [Step 29: Ranked exploit recommendations with confidence](#step-29-ranked-exploit-recommendations-with-confidence-p34)
  * [Step 30: Checking the model before trusting it](#step-30-checking-the-model-before-trusting-it-p35)
  * [Step 31: Confirming a post-hoc rule on fresh data](#step-31-confirming-a-post-hoc-rule-on-fresh-data-p35b)
  * [Step 32: Importing hand histories from text files](#step-32-importing-hand-histories-from-text-files-p36)
  * [Step 33: Priors from real players](#step-33-priors-from-real-players-and-do-the-bot-trained-models-transfer-p37)
  * [Step 34: A sequence model over raw actions](#step-34-a-sequence-model-over-raw-actions-against-hand-written-stats-p38)
  * [Step 35: Schema v2](#step-35-schema-v2-big-blind-antes-straddles-and-minimum-bets-p310)
  * [Step 36: Opponents that adapt to the hero](#step-36-opponents-that-adapt-to-the-hero-p39)
  * [Step 37: Phase 3 wrap-up](#step-37-phase-3-wrap-up-one-command-reproduced)
  * Phase 2, completed after Phase 3
  * [Step 38: Variance reduction with AIVAT](#step-38-variance-reduction-with-aivat-measured-exactly-p211)
  * Performance
  * [Step 39: Faster code, identical results](#step-39-faster-code-identical-results)
  * [Step 40: A hot start, parallel replicates and a split test suite](#step-40-a-hot-start-parallel-replicates-and-a-split-test-suite)
  * Phase 2, completed after Phase 3 (continued)
  * [Step 41: Duplicate dealing at more than two seats](#step-41-duplicate-dealing-at-more-than-two-seats-p211b)
  * [Step 42: Where six copies' shortfall comes from](#step-42-where-six-copies-shortfall-comes-from-p211c)
  * [Step 43: Exploitation when the model is wrong](#step-43-exploitation-when-the-model-is-wrong-p212)
  * Tools
  * [Step 44: Playing against the model](#step-44-playing-against-the-model-a-local-web-page)
* [Part C: cross-cutting engineering](#part-c-cross-cutting-engineering)
* [Part D: mistakes and how they were caught](#part-d-mistakes-and-how-they-were-caught)
* [Part E: glossary](#part-e-glossary)
* [Part F: references](#part-f-references)
* [Changelog of this log](#changelog-of-this-log)

---

# Part A: background

## A.1 No-limit hold'em, precisely

**Cards.** A 52-card deck. Each player gets two private *hole cards*; five shared *board*
cards arrive in three stages: the *flop* (3 cards), the *turn* (1) and the *river* (1). There
are four betting rounds (*streets*): preflop, flop, turn, river. A player's hand is the best
five-card combination of their two hole cards and the five board cards.

**Forced bets.** The two seats to the left of the dealer *button* post the *small blind* (SB)
and the *big blind* (BB) before any cards are seen. Here the blinds are 1 and 2 chips, and
every amount is measured in chips or in big blinds (bb). Some games also have *antes* (a small
forced bet from everyone, or in modern tournaments one *big-blind ante* that the big blind
posts for the whole table) or *straddles* (an optional third blind, usually two big blinds).
Schema v2 holds all three ([Step 35](#step-35-schema-v2-big-blind-antes-straddles-and-minimum-bets-p310)).

**Actions.** On their turn a player may *fold* (give up), *check* (only if nothing is owed),
*call* (match the current bet), *bet* (open the betting on a street) or *raise* (increase the
current bet). "No-limit" means any size up to everything you have (*all-in*), subject to a
minimum:

* **Minimum raise.** A raise must increase the bet by at least the largest previous bet or
  raise increment on that street (preflop, at least one big blind). Example: blinds 1/2, a
  player raises *to* 6 (an increment of 4); the next raise must be *to* at least 10.
* **Incomplete all-ins do not reopen the action.** If a player goes all-in for less than a
  full raise, players who already acted may call or fold but may not re-raise.

**Side pots.** A player who is all-in can only win, from each opponent, as much as they put
in. Chips beyond that form *side pots* contested by the players who still have chips. A hand
can have several pots with different eligible players.

**Uncalled bets.** If a bet is not matched, for example everyone folds, or a shorter stack
calls all-in for less, the unmatched part is *returned* to the bettor. It is not won. This
distinction matters for statistics such as "won money at showdown" (see
[Step 2](#step-2-the-hand-history-schema-m1)).

**Showdown.** If two or more players remain after the river (or everyone is all-in), they
compare hands; the best hand wins each pot it is eligible for, and ties split. Players who
cannot win may *muck* (discard face down) without showing, which is why real hand histories
often lack opponents' cards.

**How many distinct hands are there?** Two five-card hands of the same *rank class* tie.
Counting classes gives 7,462 distinct ranks: 10 straight flushes (including the royal flush),
156 four-of-a-kind, 156 full houses, 1,277 flushes, 10 straights, 858 three-of-a-kind, 858 two
pair, 2,860 one pair and 1,277 high-card hands. Fast evaluators return a number in 1..7462
(phevaluator: 1 = royal flush, lower is stronger).

**Our simulated game.** 6 seats ("6-max"), blinds 1/2, every player starts every hand with
200 chips (100 bb). Stacks reset each hand, like a cash game where everyone tops up. There is
no rake.

## A.2 Positions and the order of action

The button moves one seat clockwise after each hand. Counting clockwise from the seat after
the button, the six positions are **SB, BB, UTG** ("under the gun"), **HJ** (hijack), **CO**
(cutoff) and **BTN** (button).

* **Preflop** action starts with UTG (left of the big blind) and ends with the big blind.
* **Postflop** action starts with the first remaining player left of the button and ends with
  the button. Acting last ("in position") is an advantage, because you see what others did
  first. That is why players open wider ranges on the button than under the gun.
* **Heads-up exception:** with two players the button posts the *small* blind, acts first
  preflop and last postflop.

In code, [`seat_order_from_button`](../poker_lab/schema/positions.py) produces the dealing
order (seat after the button first, button last). It is also the player-index order of
PokerKit, the rules engine we use, which makes the mapping between seats and engine indices a
single tuple lookup.

## A.3 HUD statistics and why denominators matter

Online players use *heads-up displays* (HUDs) that summarize opponents with a few frequencies.
Each statistic is a ratio, *successes / opportunities*. A statistic is only meaningful if the
denominator counts real opportunities. Divide 3-bets by *hands played* instead of by *times
the player faced an open raise*, and a player at a passive table looks tight just because
nobody opened in front of them.

| Stat | Success (numerator) | Opportunity (denominator) |
|---|---|---|
| VPIP (voluntarily put money in pot) | called or raised preflop | had a preflop decision (a "walk", where everyone folds to the big blind, is *not* an opportunity) |
| PFR (preflop raise) | raised preflop | had a preflop decision |
| 3-bet | re-raised when first facing exactly one raise | faced exactly one raise, had chips behind beyond the call |
| Fold to 3-bet | folded | made the first raise and then faced exactly one re-raise |
| C-bet (continuation bet) | bet the flop | was the last preflop raiser and made the first flop decision with no bet in front |
| Fold to c-bet | folded | faced the c-bet with no raise after it |
| AF (aggression factor) | postflop bets + raises | postflop calls (a *ratio*, not a proportion) |
| WTSD (went to showdown) | reached showdown | saw the flop |
| W\$SD (won money at showdown) | won (part of) a pot at showdown | reached showdown |

AF is unbounded (calls can be 0), so it cannot get a beta prior directly. We use the
*aggression share* $s = A/(A+C)$, with $A$ = bets + raises and $C$ = calls. The share is a
proportion with a natural opportunity count $A + C$. The AF is recovered as
$\mathrm{AF} = s/(1-s)$. The exact definitions used in code are in
[`profiling/stats.py`](../poker_lab/profiling/stats.py) and walked through in
[Step 7](#step-7-stats-with-explicit-opportunity-counting).

## A.4 Player archetypes

Players are often described by two axes: *loose vs tight* (how many hands they play) and
*aggressive vs passive* (how often they bet or raise rather than call).

| Archetype | Description | Typical VPIP / PFR | Postflop |
|---|---|---|---|
| Nit | very tight, plays only strong hands | 8-16% / 6-13% | folds a lot to aggression |
| TAG (tight-aggressive) | solid regular | 17-26% / 13-22% | bets when it has it, folds otherwise |
| LAG (loose-aggressive) | wide and aggressive | 26-38% / 20-32% | bluffs and pressures |
| Calling station | loose and passive | 35-60% / 3-12% | calls too much, rarely folds or raises |
| Maniac | plays most hands, raises constantly | 45-75% / 32-60% | bluffs and raises relentlessly |

These ranges were written down *before* any simulation (`INTENDED_STATS` in
[`bots/archetypes.py`](../poker_lab/bots/archetypes.py)) and later served as the yardstick in
the sanity check ([Step 9](#step-9-the-sanity-check-tuning-against-pre-registered-tendencies-m3)).

## A.5 Toolchain

| Tool | Used for |
|---|---|
| Python 3.13 (needs >= 3.11) | `StrEnum`, dataclass `slots`, modern typing |
| NumPy, SciPy | arrays, optimization (`scipy.optimize`), special functions (`betaln`, `logit`) |
| scikit-learn | logistic regression, histogram gradient boosting, grouped splits |
| matplotlib | figures |
| PokerKit 0.7.7 (MIT) | no-limit hold'em rules engine |
| phevaluator 0.6 (Apache-2.0) | fast 5-7 card hand evaluation in C |
| pytest, pytest-xdist, ruff | tests (in four processes, Step 40), lint and formatting |
| PyYAML | experiment configs |
| PyTorch 2.14.1 (CPU; optional `nn` extra) | Deep CFR networks (Phase 2) |

## A.6 Game theory in one page

Phase 2 computes equilibrium ("GTO") strategies and measures how far other strategies are from
them. This is the vocabulary it uses.

* **Extensive-form game.** A tree. An internal node is either a *decision node*, where a
  player picks an action, or a *chance node*, where nature picks an outcome with known
  probabilities (dealing a card). Leaves (*terminal nodes*) carry payoffs. A *history* $h$ is
  the path from the root to a node.
* **Information set.** A player cannot see the opponent's cards, so several histories look the
  same to them. An information set $I$ groups the histories the acting player cannot tell
  apart, and the player must act the same way at all of them. In poker,
  $I$ = (own cards, public cards, betting so far).
* **Perfect recall.** Players never forget their own actions or what they saw. Formally, all
  histories in $I$ share the same sequence of the player's own (information set, action)
  pairs. Every algorithm in Phase 2 assumes this, and the tree compiler checks it.
* **Behavior strategy.** $\sigma_i(I)$ is a probability distribution over the actions at each
  of player $i$'s information sets. A *profile* $\sigma = (\sigma_1, \sigma_2)$ fixes both
  players. The probability of reaching $h$ factors into chance's and each player's
  contributions: $\pi^\sigma(h) = \pi_c(h)\,\pi_1^\sigma(h)\,\pi_2^\sigma(h)$.
* **Zero-sum games and the value.** Player 2's payoff is minus player 1's, $u_2 = -u_1$. Von
  Neumann's minimax theorem says
  $\max_{\sigma_1}\min_{\sigma_2} u_1 = \min_{\sigma_2}\max_{\sigma_1} u_1 = v^*$, the *game
  value*. Kuhn poker has $v^* = -1/18$ for the first player and Leduc hold'em $-0.0856$:
  acting first is a disadvantage in both.
* **Best response.** $\mathrm{BR}_1(\sigma_2) = \max_{\sigma_1'} u_1(\sigma_1', \sigma_2)$, the
  most player 1 can get against a fixed opponent. Some best response is always pure (one
  action per information set).
* **Nash equilibrium.** A profile in which each strategy is a best response to the other. In
  two-player zero-sum games every equilibrium strategy guarantees at least the value against
  *any* opponent, and equilibrium strategies are interchangeable. This is what "game-theory
  optimal" means in poker.
* **NashConv and exploitability.**
  $\operatorname{NashConv}(\sigma) = \mathrm{BR}_1(\sigma_2) + \mathrm{BR}_2(\sigma_1)
  = [\mathrm{BR}_1(\sigma_2) - v^*] + [\mathrm{BR}_2(\sigma_1) + v^*]$: the total that the two
  players could gain by deviating. It is $\ge 0$, and $0$ exactly at an equilibrium.
  *Exploitability* is $\operatorname{NashConv}/2$: what a best-responding opponent wins per
  hand against $\sigma$, above the game value, averaged over both seats.
* **Equilibrium versus exploitation.** An equilibrium guarantees the value but does not punish
  mistakes as hard as possible. Against a calling station, a best response wins far more than
  the equilibrium does. Exploiting someone means deviating from equilibrium, which opens you
  to counter-exploitation. Phase 2 measures both sides of that trade-off exactly in Leduc.

**Kuhn poker** (3 cards, one bet) and **Leduc hold'em** (6 cards, two betting rounds, one
public card) are the standard small poker games of the research literature. They are small
enough to solve exactly and rich enough to contain bluffing, slow-playing and card removal.
Their rules are in [Step 17](#step-17-small-games-and-a-tree-compiler-p21).

---

# Part B: the steps

Each step lists: **goal**, **what was built**, **concepts and math**, **decisions**,
**verification**, and **pitfalls** where there were some.

## Step 0: Bootstrap and working agreements

**Goal.** A repository where every later step is traceable.

**What was built.** `git init` on a branch (`phase1`), a `PROGRESS.md` checklist created
*before any code* (a working file, kept out of the published repository),
[`pyproject.toml`](../pyproject.toml) with pytest and ruff settings, and a package skeleton.

**Concepts.**

* **A living checklist.** `PROGRESS.md` has fixed sections (Completed / In Progress / Future /
  Decisions Log / Open Questions). Every finished item carries a date and *how it was
  verified* (a test name or an experiment output). Every non-obvious choice gets one line of
  rationale in the Decisions Log. This is cheap insurance against "why did we do that?".
* **Warnings as errors.** `pyproject.toml` sets
  `filterwarnings = ["error::UserWarning:pokerkit.*"]`, so any PokerKit warning (for example
  "a card being dealt is not recommended") fails the tests instead of scrolling past.
* **Strict lint.** Ruff with pyflakes, pycodestyle, import sorting, bugbear, pyupgrade,
  simplify, comprehensions, pytest style, NumPy and Ruff-specific rules.

**Decisions.** The global Python interpreter was used instead of a virtual environment inside
the OneDrive-synced folder, which would have synced thousands of files. Exact package versions
go into every results file instead (see [Part C](#part-c-cross-cutting-engineering)).

## Step 1: Choosing the engine and evaluator by measurement

**Goal.** Decide, with numbers, how to simulate on the order of a million hands of no-limit
hold'em correctly and fast enough.

**Concepts.**

* **Profile before optimizing.** A 2,000-hand benchmark with random agents on raw PokerKit ran
  at ~150 hands/s. `cProfile` showed ~80% of the time in showdown handling: PokerKit's
  `StandardHighHand.from_game` tries all $\binom{7}{5} = 21$ five-card subsets in pure Python
  (~0.75 ms per call).
* **How fast hand evaluators work.**
  * *Naive:* rank all 21 subsets and keep the best.
  * *Cactus Kev* (a classic): give each rank a prime number; the product of a hand's five
    primes is a unique key for its rank multiset, looked up in a table, plus a bitmask check
    for flushes.
  * *Two-Plus-Two:* a ~130 MB precomputed state machine; one table lookup per card.
  * *Perfect hashing* (phevaluator): hash the 13 rank counts into an index of a compact table
    (with a separate path for flushes) that returns the rank class in $O(1)$.

  Measured on this machine: phevaluator's C core does **4.2 million** 7-card evaluations per
  second, treys (pure Python) 0.06 million.
* **Monte Carlo equity.** The *equity* of a hand is its probability of winning (ties count
  half) against an opponent hand and the unknown remaining board. Sample $N$ completions, score
  each $s_j \in \{0, \tfrac12, 1\}$, and average:

  $$\hat e = \frac{1}{N} \sum_{j=1}^{N} s_j, \qquad \operatorname{SE}(\hat e) \approx \sqrt{\frac{e(1-e)}{N}}$$

  With $N=150$ and $e \approx 0.5$ the standard error is about 0.04, good enough for coarse
  strength buckets. A preflop table built once with $N = 40{,}000$ per hand class has
  $\operatorname{SE} \approx 0.0025$.

**Decisions.**

* **Wrap PokerKit, do not rewrite the rules.** It is MIT-licensed, actively maintained, and
  peer-reviewed. Min-raise and reopening rules and side pots are exactly the kind of logic
  where a home-grown engine hides bugs.
* **Plug in speed through PokerKit's own extension point.** PokerKit games declare
  `hand_types`; we subclass `StandardHighHand` with a phevaluator-backed `from_game` and
  comparison operators ([`engine/fast_hand.py`](../poker_lab/engine/fast_hand.py)) and subclass
  the game to use it. Betting, pots and showdown order stay PokerKit's. Result: ~3x faster
  (~440 hands/s on a showdown-heavy random workload).
* **Deal our own cards.** PokerKit shuffles with Python's global `random` module, which is not
  reproducible per hand. We shuffle with a seeded `random.Random` per table and pass explicit
  cards to `deal_hole` / `deal_board` (burns are the unknown card `"??"`).

**Verification.** Differential tests compare the fast hand type with PokerKit's own on 1,500
random spots plus edge cases (wheel straights, board plays, kicker decisions), and every
simulated hand is replayed through *unmodified* PokerKit ([Step 4](#step-4-the-simulation-engine-m2)).

## Step 2: The hand-history schema (M1)

**Goal.** One canonical record of a hand that every phase produces or consumes: the simulator
now, imported histories later, and the camera plus manual capture system in Phase 4.

**What was built.** [`schema/hand_history.py`](../poker_lab/schema/hand_history.py): frozen,
slotted dataclasses `TableConfig`, `SeatInfo`, `Action`, `CardObservation`, `HoleCards`,
`ShowdownEntry`, `ChipAward`, `HandHistory`; [`schema/validation.py`](../poker_lab/schema/validation.py);
[`schema/serialization.py`](../poker_lab/schema/serialization.py) (JSON Lines).

**Concepts and math.**

* **Integers for chips.** Floating point makes "chips in = chips out" checks fragile. All
  amounts are integers in the table's smallest unit.
* **Two amounts per action.** `amount` is what the action moved from stack to pot;
  `total_bet` is the player's live bet on the street afterwards ("raise *to* 18"). Antes are
  dead money and never count toward `total_bet`. Storing both lets any consumer check
  `total_bet = previous live bet + amount` without replaying the hand.
* **Uncalled bets, separately from pot awards.** On each street let $b_{(1)} \ge b_{(2)}$ be
  the two largest live contributions. The top bettor gets $b_{(1)} - b_{(2)}$ back, and the
  rest of the pot is awarded. With contributions $c_i$, pot awards $w_i$, returns $r_i$ and
  rake $R$:

  $$\sum_i c_i = \sum_i w_i + \sum_i r_i + R, \qquad \mathrm{net}_i = w_i + r_i - c_i, \qquad \sum_i \mathrm{net}_i = -R$$

  Validation recomputes the returns independently and checks this conservation identity on
  every complete hand.
* **Validation by replay.** `find_problems` walks the actions keeping per-street live bets
  and remaining stacks. It checks: streets never go backwards, folded or all-in seats never
  act again, a check only happens when nothing is owed, a call matches the outstanding bet (or
  is a short all-in), a bet only opens a street, a raise exceeds the current bet, `is_all_in`
  is true exactly when the stack hits zero, the board has enough cards for every street with
  action, and no card appears twice.
* **Provenance per card.** `CardObservation(card, source, confidence)` with sources
  simulator/manual/camera/imported. Phase 4 can record "the camera read the turn as Jc with
  probability 0.73" in the same format the simulator uses.
* **Strict decoding.** The JSON reader rejects unknown and missing keys. A typo'd field
  becomes an error instead of being silently dropped, and nothing can be smuggled in an extra
  field (see Alexis King's "Parse, don't validate" for the general idea).

**Verification.** `tests/test_schema.py`, `tests/test_serialization.py` (dict and JSONL
round-trips, gzip, rejection cases).

**Pitfall.** A test expected a 10-chip c-bet to be 67% of a 15-chip pot. The pot was actually
19 chips (1 + 6 + 6 + 6), so the code was right and the test was wrong. Tests encode beliefs,
and beliefs can be wrong too.

## Step 3: The information firewall (M1)

**Goal.** Make it *structurally impossible* for advice-giving code to see opponents' hole
cards from the hand being played, and prove it with tests.

**Concepts.**

* **Threat model.** The realistic risk is not a malicious programmer but *accidental leakage*.
  A convenient object (the engine state, a full hand record) gets passed to decision code
  "just for now". In ML this is the classic way test labels leak into features. The defense
  is to make the wrong thing impossible to express.
* **Make illegal states unrepresentable.** A `HandHistory` whose status is `in_progress`
  *cannot* hold hole cards: the constructor and the JSON decoder both reject it.
* **Capabilities.** While a hand is live, hole cards live in a private vault inside
  [`HandHistoryStore`](../poker_lab/schema/store.py). Dealing a seat its cards returns a
  `SeatKey`, and the only read path, `store.seat_view(hand_id, key)`, returns *that seat's own*
  cards. Keys are compared by *identity* (`is`), so a key rebuilt from public fields
  (`SeatKey("h1", 3)`) does not work. This is the object-capability pattern: holding an
  unforgeable reference *is* the permission. `complete_hand` is the single place sealed cards
  are released.
* **Narrow, exact input types.** Decision code receives a `SeatView`: public state plus
  exactly one seat's two cards ([`schema/views.py`](../poker_lab/schema/views.py)). Its
  constructor checks `type(x) is ...` rather than `isinstance`, so subclasses cannot add
  fields. Hole cards must be plain `str` and a plain `tuple`, because a `str` subclass can carry
  hidden attributes. The dataclass is frozen and slotted, so nothing can be attached afterwards.
* **Proof by reachability.** Tests walk the object graph reachable from a view (Python's
  `gc.get_referents`) and assert that the only card strings found are the seat's own cards and
  the board. Types, modules and functions are skipped during the walk; otherwise the walk would
  reach module globals such as the full deck and report false leaks.

**Limit, stated plainly.** In-process Python can always defeat this with reflection (for
example `object.__getattribute__` on private attributes). The firewall stops accidental and
honest-but-buggy leakage, which is what actually happens in practice.

**Verification.** `tests/test_store_sealing.py` (forged keys, keys from another store,
reachability) and later `tests/test_advisor_firewall.py` (ten violation attempts).

## Step 4: The simulation engine (M2)

**Goal.** Play hands between arbitrary agents and emit valid hand histories, with the rules
enforced by PokerKit.

**What was built.** [`engine/simulator.py`](../poker_lab/engine/simulator.py)
(`TableSimulator`), [`engine/agent.py`](../poker_lab/engine/agent.py) (`Agent`, `Decision`),
[`engine/replay.py`](../poker_lab/engine/replay.py), simple agents in
[`engine/basic_agents.py`](../poker_lab/engine/basic_agents.py).

**Concepts.**

* **PokerKit as a state machine.** A PokerKit `State` advances by *operations* (post blind,
  deal, burn, fold, check/call, bet/raise, show/muck, push chips). Mechanical ones are
  *automated*: antes, blinds, bet collection, show-or-muck, hand killing, chip pushing and
  pulling. Dealing is done by us from the seeded deck. The engine loop is:

  ```text
  while hand not over:
      if a burn is pending:        burn "??"
      elif board cards are needed: deal the next cards from our shuffled deck
      else:                        ask the agent in seat order[actor_index] for a Decision
  ```

* **A small decision vocabulary.** Agents answer with one of fold, check/call, or bet/raise *to*
  an amount. The engine maps that to the precise schema action (check vs call, bet vs raise)
  and rejects illegal choices: folding when checking is free, or a raise outside
  `[min_raise_to, max_raise_to]`. These bounds come from PokerKit, so they follow the real
  min-raise rule.
* **Everything goes through the sealed store.** The engine records blinds, deals, actions and
  board cards into a `HandHistoryStore`. Each agent's `SeatView` is built *by the store* from
  that agent's `SeatKey`, so the engine itself never hands agents anything richer.
* **Results.** After the hand, for each seat, received $=$ final stack $-$ (start stack $-$
  contribution). Pot award $=$ received $-$ uncalled return. The engine then asserts that the
  recorded net result equals PokerKit's own payoff for every seat. That check runs on every
  hand, not just in tests.
* **Agents see redacted completed hands.** `Agent.hand_completed` receives the finished record
  with unshown hole cards removed (`redact_hole_cards`), the same information a real player
  has.

**Verification.**

* **Differential testing.** `replay_in_pokerkit` replays every recorded hand through
  *unmodified* PokerKit, with its slow reference evaluator. Every action must be legal there,
  payoffs must match, and show/muck decisions must match. Configurations: heads-up, 3-way with
  uneven stacks, 6-max, 6-max with very short stacks (side pots), 9-max with antes.
* Chip conservation and validation on 300 hands for 2, 3, 6 and 9 players; illegal decisions
  rejected; heads-up blind and acting order; same seed gives byte-identical JSON.

**Pitfalls (both caught by the validator, not by eye).**

1. *All-in flag after automatic payout.* The engine first marked a call "all-in" if the stack
   was 0 after the call. On the river, PokerKit's automations run showdown and pay the pot
   *inside the same call*, so a winning all-in caller already had chips back. Fix: decide
   all-in from the stack *before* the action.
2. *Heads-up posting order.* PokerKit posts blinds in player-index order, which heads-up puts
   the big blind first. The schema convention is antes, then SB, then BB, so posts are sorted
   before recording.
3. PokerKit's `Card.clean` does not parse strings *inside* a list (`["As", "Kd"]` stays a list
   of strings). The fast hand type normalizes its input itself.

## Step 5: Hand strength (M3)

**Goal.** Give bots a cheap, interpretable notion of "how good is my hand".

**What was built.** [`bots/hand_strength.py`](../poker_lab/bots/hand_strength.py) and the
generated table `bots/data/preflop_equity.json`.

**Concepts and math.**

* **169 starting-hand classes.** Suits are symmetric before the flop, so the
  $\binom{52}{2} = 1326$ two-card combinations collapse into 13 pairs ("TT"), 78 suited
  ("AKs") and 78 offsuit ("AKo") classes, with 6, 4 and 12 combos respectively:
  $13 \cdot 6 + 78 \cdot 4 + 78 \cdot 12 = 1326$.
* **Preflop ranking.** Each class gets its all-in equity against one uniformly random hand by
  Monte Carlo (40,000 samples per class, fixed seed). The table matches published values
  (AA 85.2%, AKs 67.0%, 72o 34.6%) within 0.006.
* **Range width as a percentile.** Sort classes by equity, strongest first, and give each
  class the cumulative share of combos at least as strong:

  $$\operatorname{pct}(c) = \frac{1}{1326} \sum_{c' \succeq c} \operatorname{combos}(c')$$

  Aces have $\operatorname{pct} = 6/1326 \approx 0.0045$. A bot with range width $w$ plays a
  hand when $\operatorname{pct} \le w$, so $w$ reads as "plays the top $w$ of hands".
* **Postflop strength.** Monte Carlo equity against one random hand ($N = 150$), computed once
  per hand and street and cached. It accounts for draws (a flush draw has real equity
  although it is not a made hand yet), which made-hand categories would miss.
* **Buckets.** Equity is cut into AIR / WEAK (weak made hands and draws) / MEDIUM / STRONG at
  0.45 / 0.62 / 0.80. When facing a bet the cut points rise to 0.55 / 0.70 / 0.85. A bet is
  *evidence* that the bettor's range is stronger than random, so equity against a random hand
  overstates our strength. Raising the bar is a crude Bayesian update. (The exact alternative,
  equity against a weighted opponent range, costs far more per decision.)

## Step 6: Parameterized bots and archetypes (M3)

**Goal.** Opponents whose tendencies are known exactly (ground truth), interpretable, and
varied, so profiling can be measured.

**What was built.** [`bots/params.py`](../poker_lab/bots/params.py) (`BotParams`),
[`bots/archetype_bot.py`](../poker_lab/bots/archetype_bot.py) (`ParametricBot`),
[`bots/archetypes.py`](../poker_lab/bots/archetypes.py),
[`bots/population.py`](../poker_lab/bots/population.py).

**The nine parameters.**

| Parameter | Meaning | Mostly moves |
|---|---|---|
| `range_width` $w$ | share of starting hands played in an unraised pot | VPIP |
| `raise_share` $r$ | P(enter with a raise rather than a limp) | PFR / VPIP |
| `three_bet_freq` $t$ | P(re-raise with a continuing hand facing one raise) | 3-bet |
| `continue_vs_raise` $c$ | facing a raise, continue with the top $w \cdot c$ | VPIP, fold to 3-bet |
| `cbet_freq` | P(c-bet the flop as preflop aggressor) | C-bet |
| `fold_to_aggression` $f$ | P(fold a weak hand facing a postflop bet) | fold to bet, WTSD |
| `bluff_freq` $b$ | P(bet or raise with air or weak hands, scaled) | AF |
| `value_aggression` $v$ | P(bet or raise with strong hands) | AF |
| `bet_size_pot` $s$ | bet size as a fraction of the pot | bet and open sizes |

**The policy** (all randomness comes from the table's seeded RNG). Let $h$ be the preflop
percentile.

* *Unopened pot:* play if $h \le w \cdot \text{pos}$, with position multipliers UTG 0.7,
  HJ 0.85, CO 1.05, BTN 1.4, SB/BB 1.0 (they average to 1). Raise with probability $r$ to
  $(2 + 2s)$ bb plus 1 bb per limper, times noise $e^{0.12 Z}$ with $Z \sim N(0,1)$; otherwise
  limp, or check in the big blind.
* *Facing one raise:* continue if $h \le w c$ (times 1.3 in the big blind, which gets a better
  price), 3-bet with probability $t$ (to 3x the bet), else call. Light 3-bet bluff with
  probability $0.08\,b$.
* *Facing a 3-bet:* continue if $h \le w c \cdot 0.5$ (opener) or $0.4$ (others); 4-bet with
  probability $\max(0.5t, 0.5 \cdot [h \le 0.02])$.
* *Facing a 4-bet or more:* only premiums, $h \le 0.02 + 0.03c$.
* *Commit rule:* if a bet would use more than half the remaining stack, go all-in.
* *Postflop, not facing a bet:* in a c-bet spot bet with probability `cbet_freq` (strong hands
  at least $v$); otherwise bet with probability $(0.5b/m,\ 0.7b/m,\ 0.6v,\ v)$ for AIR / WEAK
  / MEDIUM / STRONG. Here $m = \max(1, n_\text{opp} - 1)$ makes bluffs rarer in multiway pots.
* *Postflop, facing a bet:* with the price $\pi = \text{call}/(\text{pot} + \text{call})$ and a
  price factor $\varphi = \operatorname{clip}(0.6 + 1.2\pi,\ 0.5,\ 1.4)$:

  | Bucket | P(fold) | P(raise) |
  |---|---|---|
  | AIR | $\min(1, 2f + 0.15)\,\varphi$ | $0.5\,b/m$ |
  | WEAK | $f\,\varphi$ | $0.55\,b/m$ |
  | MEDIUM | $0.5f\,\varphi$ | $0.5\,v^2$ |
  | STRONG | 0 | $0.6\,v$ |

  and call otherwise. Bets are $s \cdot \text{pot} \cdot e^{0.12Z}$; raises go to the current
  bet plus $s \cdot (\text{pot} + \text{call}) \cdot e^{0.12Z}$, clamped to the legal range.

**Archetypes are distributions, not bots.** Each archetype draws every probability parameter
from a beta distribution with mean $\mu$ and concentration $\kappa = 30$:

$$p \sim \operatorname{Beta}\big(\mu\kappa,\ (1-\mu)\kappa\big), \qquad \mathbb E[p] = \mu, \qquad \operatorname{SD}[p] = \sqrt{\frac{\mu(1-\mu)}{\kappa+1}}$$

(SD 0.09 at $\mu = 0.5$, 0.064 at $\mu = 0.15$). Bet size is log-normal,
$s = \text{median} \cdot e^{0.2Z}$, because sizes are positive and multiplicative. Neighboring
archetypes overlap, so even infinitely many hands cannot classify perfectly; the "oracle"
classifier on true parameters measures that ceiling ([Step 14](#step-14-experimental-design-m5)).

| Mean | Nit | TAG | LAG | Station | Maniac |
|---|---|---|---|---|---|
| range_width | 0.15 | 0.26 | 0.36 | 0.48 | 0.62 |
| raise_share | 0.70 | 0.85 | 0.85 | 0.18 | 0.80 |
| three_bet_freq | 0.20 | 0.28 | 0.42 | 0.04 | 0.45 |
| continue_vs_raise | 0.50 | 0.50 | 0.60 | 0.90 | 0.80 |
| cbet_freq | 0.55 | 0.68 | 0.78 | 0.30 | 0.85 |
| fold_to_aggression | 0.82 | 0.50 | 0.33 | 0.10 | 0.18 |
| bluff_freq | 0.05 | 0.18 | 0.38 | 0.06 | 0.65 |
| value_aggression | 0.55 | 0.75 | 0.85 | 0.35 | 0.95 |
| bet_size_pot (median) | 0.50 | 0.66 | 0.75 | 0.45 | 1.00 |

**Ground truth stays separate.** Player ids are random hex tokens (`p3fa9c2e1`) and the
population is shuffled, so neither the id nor the seating says anything about the type. True
archetypes and parameters live only in `GroundTruth` (its own JSON file). A test serializes
hands and asserts that no archetype name or parameter value appears in them, because a label
hiding in an id is a classic leakage bug.

**Pitfall.** An early version made "bluff raises" even when raising was illegal (facing an
all-in), and the fallback quietly turned them into *calls* with junk hands. Bluff branches now
fold when raising is not allowed.

## Step 7: Stats with explicit opportunity counting

**Goal.** Compute every HUD stat from public information only, with exact denominators.

**What was built.** [`profiling/stats.py`](../poker_lab/profiling/stats.py): `hand_counters`
makes *one pass* over a complete hand's actions and returns, per player, a vector of 24
integer counters (`vpip_opp`, `vpip`, `three_bet_opp`, `three_bet`, ..., `bet_size_x100`).

**Concepts.**

* **A tiny state machine.** While scanning actions it tracks: the street, live bets, stacks,
  the number of preflop raises, the first and last preflop raiser, how much aggression has
  happened on the flop, whether the c-bet was made, and who already used their one-per-hand
  opportunity.
* **Corner cases that change numbers:**
  * a big-blind walk is not a VPIP opportunity (no decision was made);
  * a limper who then faces a raise *does* get a 3-bet opportunity (they could re-raise);
  * players acting after a 3-bet face *two* raises, so it is not a 3-bet opportunity for them;
  * a donk bet (a non-aggressor betting first) removes the aggressor's c-bet opportunity;
  * after someone raises the c-bet, players behind no longer "face the c-bet";
  * a player all-in preflop still "saw the flop" and reached showdown.
* **Additive counters and prefix sums.** Counters add across hands, so a player's stats over
  hands $[a, b)$ are $C_b - C_a$, where $C_j$ is the running total after $j$ hands (with
  $C_0 = 0$). One `cumsum` gives totals for *any* window in $O(1)$. That is what makes
  "accuracy after $n$ hands for every $n$" cheap.
* **Public information only.** The counter never reads hole cards, and hands are redacted
  before counting anyway, so the profiler sees what a real player at the table would.

**Verification.** `tests/test_stats.py` builds hands from a compact script (who did what on
which street) and checks each counter by hand.

## Step 8: Running a league: parallel, reproducible simulation

**Goal.** Simulate hundreds of bots for thousands of hands each, quickly and reproducibly.

**What was built.** [`bots/league.py`](../poker_lab/bots/league.py) (`run_league`).

**Concepts.**

* **Round-robin re-seating.** Each *round*, a random permutation of the population is cut
  into tables of 6, and every table plays 50 hands. With $R$ rounds every player plays exactly
  $50R$ hands against a changing mix of opponents. This averages out lucky table draws and
  keeps everyone's history the same length.
* **Seeds that do not depend on scheduling.** Each table-round is a task whose seed is derived
  from `numpy.random.SeedSequence([seed, round, table])`. Results do not depend on which worker
  runs which task or how many workers exist. `ProcessPoolExecutor.map` returns results in task
  order. A test asserts byte-identical counters with 1 and 2 workers.
* **Return arrays, not objects.** Workers return `int16` counter arrays (players x hands x
  counters) and `float32` decision rows. Pickling Python objects for 150k hands would cost more
  than simulating them.
* **Throughput.** About 2,500 hands/s with 12-15 worker processes (~200-280 per process, the
  bots' Monte Carlo equity being the main cost).

**Pitfall (Windows).** Windows starts worker processes with *spawn*. Every worker re-imports
the main module and every package `__init__` it touches. With scikit-learn and matplotlib
imported eagerly, 15 workers exhausted the machine's memory commit limit ("paging file is too
small"). Fixes:

1. package `__init__.py` files re-export names *lazily* through a module-level `__getattr__`
   (PEP 562), so a worker that only needs `stats` never imports scikit-learn;
2. heavy imports in experiment scripts were moved inside functions;
3. the default worker count is capped at 12.

## Step 9: The sanity check: tuning against pre-registered tendencies (M3)

**Goal.** Show that each archetype actually *plays* like its description.

**What was built.** [`experiments/bot_sanity.py`](../experiments/bot_sanity.py).

**Protocol (the important part).**

1. Intended HUD ranges for each archetype (A.4) were written *before* any simulation and never
   edited afterwards.
2. Policy coefficients and archetype means were tuned in six short iterations, each on a
   fresh, small population (12-48 bots per type).
3. The final check used a *new* seed (2026) and a larger population: 300 bots x 3,000 hands =
   150k hands.
4. Misses are reported, not tuned away.

**Result.** 29 of 30 archetype-stat cells fall in their intended range and all 4 intended
orderings hold (stations reach showdown more than nits and TAGs; nits win more at showdown
than stations, TAGs more than maniacs). The miss: nit fold-to-c-bet 54.0% vs 55-80%.

**Do the parameters drive the stats they should?** Spearman rank correlation $\rho$ (Pearson
correlation of ranks; robust to monotone nonlinearity) between each parameter and its target
stat:

| Parameter -> stat | $\rho$ over all bots | $\rho$ within archetypes |
|---|---|---|
| range_width -> VPIP | 0.99 | 0.96-0.99 |
| raise_share -> PFR/VPIP | 0.90 | 0.65-0.81 |
| cbet_freq -> c-bet | 0.97 | 0.72-0.93 |
| fold_to_aggression -> fold to bet | 0.96 | 0.31-0.97 |
| bluff_freq -> aggression share | 0.68 | 0.09-0.39 |
| bet_size_pot -> bet size | 1.00 | 1.00 |

**Between vs within.** A correlation over all bots mixes two sources of variation: differences
*between* archetypes and differences *within* an archetype. By the law of total covariance,

$$\operatorname{Cov}(X, Y) = \mathbb E\big[\operatorname{Cov}(X, Y \mid \text{type})\big] + \operatorname{Cov}\big(\mathbb E[X \mid \text{type}],\ \mathbb E[Y \mid \text{type}]\big)$$

`bluff_freq` correlates with aggression mostly because maniacs bluff *and* bet a lot. Within a
type its effect is faint. This predicted, before any modeling, that individual bluff
frequencies would be hard to recover; [Step 12](#step-12-parameter-regression-m4) confirmed it.

**Known gap.** Bots go to showdown far more than human regulars (WTSD ~44-51%; stations
~70%; humans ~25-35%) because their postflop logic is simple.

## Step 10: Empirical-Bayes shrinkage (M4)

**Goal.** Turn noisy small-sample stats into sensible estimates with honest uncertainty. A
player who VPIP'd 3 of 4 hands is not a 75% VPIP player.

**What was built.** [`profiling/shrinkage.py`](../poker_lab/profiling/shrinkage.py).

**The model.** Player $i$ has a true rate $p_i$ (say, their VPIP), and the population of
players has a distribution of rates, modeled as a beta distribution. We observe $k_i$
successes in $m_i$ opportunities:

$$p_i \sim \operatorname{Beta}(\alpha, \beta), \qquad k_i \mid p_i \sim \operatorname{Binomial}(m_i, p_i)$$

**Fitting the prior from the population (the "empirical" in empirical Bayes).** Integrating
$p_i$ out gives the *beta-binomial* marginal likelihood of each player's data:

$$P(k_i \mid \alpha, \beta) = \int_0^1 \binom{m_i}{k_i} p^{k_i} (1-p)^{m_i - k_i} \frac{p^{\alpha-1}(1-p)^{\beta-1}}{B(\alpha,\beta)}\,dp = \binom{m_i}{k_i} \frac{B(\alpha + k_i,\ \beta + m_i - k_i)}{B(\alpha, \beta)}$$

We maximize the log-likelihood over the *training* players (the binomial coefficient does not
depend on $\alpha, \beta$ and is dropped):

$$\ell(\alpha, \beta) = \sum_i \Big[\log B(\alpha + k_i,\ \beta + m_i - k_i) - \log B(\alpha, \beta)\Big]$$

Writing $\alpha = \mu\kappa$, $\beta = (1-\mu)\kappa$ (mean $\mu$, strength $\kappa$) and
optimizing over $(\operatorname{logit}\mu, \log\kappa)$ makes the problem unconstrained and
well scaled. `scipy.optimize.minimize` (L-BFGS-B) is started at the pooled rate with
$\kappa = 10$. Players with $m_i = 0$ contribute nothing.

**The posterior and why it "shrinks".** The beta prior is conjugate to the binomial, so

$$p_i \mid k_i \sim \operatorname{Beta}(\alpha + k_i,\ \beta + m_i - k_i), \qquad \mathbb E[p_i \mid k_i] = \frac{\alpha + k_i}{\alpha + \beta + m_i} = \frac{m_i}{m_i + \kappa} \cdot \frac{k_i}{m_i} + \frac{\kappa}{m_i + \kappa} \cdot \mu$$

The estimate is a weighted average of the player's own rate and the population mean, with
the weight on the data growing with the number of opportunities. $\kappa$ reads as "the prior
is worth $\kappa$ opportunities". Example: $\mu = 0.30$, $\kappa = 20$, a player with 6 VPIPs
in 10 opportunities gets $(6 + 6)/(20 + 10) = 0.40$, not 0.60. The 95% *credible interval* is
the 2.5% and 97.5% quantiles of the posterior beta.

**Sizes (continuous stats).** Mean bet size and open size get the normal-normal analogue:
observations $x_{ij} \sim N(\theta_i, \sigma^2)$, player means $\theta_i \sim N(\mu, \tau^2)$, so

$$\mathbb E[\theta_i \mid x_i] = \frac{\kappa \mu + \sum_j x_{ij}}{\kappa + n_i}, \qquad \kappa = \frac{\sigma^2}{\tau^2}$$

The counters store only sums and counts, so $\sigma^2$ and $\tau^2$ are estimated by the
method of moments from two disjoint *halves* A and B of each training player's hands. The
difference of the half-means cancels $\theta_i$, and the halves share only $\theta_i$:

$$\mathbb E\big[(\bar x_A - \bar x_B)^2\big] = \sigma^2 \Big(\frac{1}{n_A} + \frac{1}{n_B}\Big), \qquad \operatorname{Cov}(\bar x_A, \bar x_B) = \tau^2$$

**Verification.** `tests/test_shrinkage.py`: the fit recovers a known prior from simulated
data (mean within 0.01, strength within 20%); simulated players fall inside their 95%
credible intervals 95% +/- 1.5% of the time (coverage); the split-half estimator recovers
$\kappa = \sigma^2/\tau^2 = 9$ within 20%.

**Pitfall.** A synthetic test fixture produced more successes than opportunities for WTSD,
because `wtsd` is both a success counter (of WTSD) and the opportunity counter (of W\$SD). The
shrinker now rejects $k > m$ instead of returning nonsense.

## Step 11: The archetype classifier and calibration (M4)

**Goal.** From $n$ observed hands, output calibrated probabilities over the five archetypes.

**What was built.** [`profiling/features.py`](../poker_lab/profiling/features.py),
[`profiling/classifier.py`](../poker_lab/profiling/classifier.py).

**Features.** The 10 shrunk rates, the 2 shrunk sizes, and $\log n$ (hands observed). The
linear model sees rates on the logit scale, $z = \log\frac{p}{1-p}$ (clipped to
$[0.005, 0.995]$), and sizes on the log scale, then standardized. Differences between types
are closer to additive on those scales.

**Models.**

* *Multinomial logistic regression:*
  $P(y = k \mid x) = \exp(w_k^\top x + b_k) / \sum_j \exp(w_j^\top x + b_j)$, fit by minimizing
  cross-entropy plus an L2 penalty ($C = 1$).
* *Histogram gradient boosting:* an additive model of small trees, each fit to the gradient of
  the loss (Friedman, 2001), with features pre-binned into histograms for speed (as in
  LightGBM). Settings: 300 trees, learning rate 0.06, 15 leaves, L2 = 1.
* *One model for all sample sizes,* with $\log n$ as a feature, trained on disjoint windows of
  25, 50, ..., 2000 hands (up to 8 windows per training player and size). At prediction time
  any $n$ works.

**Calibration.** A classifier is *calibrated* if, among predictions made with confidence $q$,
a fraction $q$ is correct. Two tools measure it:

* the *reliability diagram*: bin predictions by top-class confidence and plot each bin's
  accuracy against its mean confidence; the diagonal is perfect;
* the *expected calibration error* with $B = 10$ equal-width bins:

  $$\mathrm{ECE} = \sum_{b=1}^{B} \frac{n_b}{N} \big|\operatorname{acc}_b - \operatorname{conf}_b\big|$$

*Log-loss* $-\frac1N \sum_i \log \hat p_{i, y_i}$ is a *proper scoring rule*: in expectation
it is minimized only by the true probabilities. That is why it is reported next to accuracy.
Chance level for 5 classes is $\log 5 = 1.609$ nats.

**Temperature scaling** (Guo et al., 2017). Rescale the model's log-probabilities $z$ by a
single number $T$:

$$\hat p_k = \frac{\exp(z_k / T)}{\sum_j \exp(z_j / T)}$$

$T > 1$ softens an overconfident model and $T < 1$ sharpens an underconfident one. Dividing a
row's scores by the same positive $T$ keeps their order, so *accuracy never changes*. $T$ is
fit by minimizing log-loss on held-out data. The held-out data is a *grouped* split of the
training **players** (scikit-learn's `GroupShuffleSplit`): all windows of a player go to the
same side. Windows of one player are strongly correlated, so a row-level split would calibrate
on near-copies of the training data.

**Per-size temperatures (added after the first full run, labeled post hoc).** The first run's
reliability diagrams showed the logistic model well calibrated at 50 hands but underconfident
at 200-1000 hands. One global $T$ (fitted: 0.99) cannot fix miscalibration that changes with
$n$. Fitting one $T$ per training sample size and interpolating in $\log n$ gave
$T = 1.24$ at 25 hands down to $T = 0.55$ at 2,000. That roughly halved ECE from 200 hands
upward ([Step 15](#step-15-phase-1-results-and-how-to-read-them-m5)).

**Verification.** `tests/test_profiling_models.py`: learning on synthetic clusters, argmax
invariance, recovery of a known temperature ($T = 3$ within 15%), and a constructed case where
one model is overconfident at one size and underconfident at another (per-size scaling more
than halves ECE).

## Step 12: Parameter regression (M4)

**Goal.** Recover the nine continuous tendencies, not just the archetype label, since real
players are a continuum.

**What was built.** [`profiling/regression.py`](../poker_lab/profiling/regression.py):
boosted trees (one regressor per parameter) and ridge regression (multi-output).

**Measuring it honestly.** The out-of-sample $R^2$ against the *training* mean,

$$R^2 = 1 - \frac{\sum_i (y_i - \hat y_i)^2}{\sum_i (y_i - \bar y_{\text{train}})^2},$$

looked excellent (0.78 at 25 hands, 0.92 at 2,000). But by the law of total variance,

$$\operatorname{Var}(Y) = \underbrace{\mathbb E\big[\operatorname{Var}(Y \mid \text{type})\big]}_{\text{within types}} + \underbrace{\operatorname{Var}\big(\mathbb E[Y \mid \text{type}]\big)}_{\text{between types}},$$

most of that credit comes from the between-type term, which just knowing the archetype
captures. The *within-archetype* $R^2$ replaces the baseline with the true archetype's
training mean, $\bar y_{\text{type}(i)}$. It asks how much individual variation the model
recovers *beyond the type*, and it can be negative (worse than knowing the type). A third
baseline, "classify, then predict the archetype's mean" ($\hat y = \sum_k \hat P(k \mid x)\,\bar y_k$),
shows what the classifier alone delivers.

**Result** (post hoc metric). Within-type $R^2$ averaged over parameters is $-0.86$ at 25
hands, $-0.13$ at 100, $+0.21$ at 500 and $+0.35$ at 2,000. Range width (0.90) and bet size
(0.97) are recovered individually. The exploit-relevant ones are not: fold-to-aggression 0.14,
value aggression 0.15, and bluff frequency still negative ($-0.15$) at 2,000 hands. This is
the weak result Step 9's correlations predicted.

## Step 13: Next-action prediction (M4)

**Goal.** The metric Phase 5 will reuse on real data: how much does knowing a player's
profile improve the prediction of what they do next?

**What was built.** [`profiling/situations.py`](../poker_lab/profiling/situations.py)
(decision points), [`profiling/action_model.py`](../poker_lab/profiling/action_model.py).

**Concepts.**

* **Decision points from public information.** For every voluntary action we reconstruct the
  situation *before* it: street, facing a bet, pot odds, pot size, stack-to-pot ratio,
  players left, raises on the street, position, whether the player was the preflop aggressor,
  c-bet spot, facing a c-bet, checked to, already acted this street, and more (16 features).
  The label has three classes: fold / check-call / bet-raise. Only public information is used,
  so the same code runs on real hand histories. A test checks that changing *later* actions
  never changes earlier features.
* **No peeking.** The profile used for decisions in hands $[c, c + 100)$ is built from hands
  $[0, c)$ only. Checkpoints are $c \in \{0, 25, 100, 500, 1500\}$. At $c = 0$ the profile is
  just the prior, which gives a built-in sanity check.
* **Baselines and ceilings.** The same gradient-boosting learner is trained on: the situation
  alone (the *population-average* baseline the spec asks for), situation + profile, situation
  + true archetype, and situation + true parameters (the oracle). The headline is the
  log-loss reduction against the situation-only model, plus the share of the gap to the oracle
  that the profile closes:

  $$\text{gap closed} = \frac{\mathrm{LL}_{\text{situation}} - \mathrm{LL}_{\text{profile}}}{\mathrm{LL}_{\text{situation}} - \mathrm{LL}_{\text{oracle}}}$$

**Pitfall.** The first run's windows $[c, c+100)$ overlapped (0-100 and 25-125), and the code
silently assigned each overlapping decision to the *later* checkpoint, leaving the $c = 0$
window only 25 hands long. Nothing leaked (profiles still came from earlier hands), but the
windows were unequal. Fixed by evaluating each checkpoint on its full window and duplicating
decisions that fall in two windows.

**An artifact worth stating.** With zero hands of history (a constant profile), the profile
model still beats the situation-only model by 0.3% [0.2, 0.4]. No player information exists
at $c = 0$, so this is a modelling artifact: two separately fitted models allocate capacity
differently, and the profile model can specialize on the $c = 0$ rows. It is small next to the
11-13% gains at 25 or more hands, and it sets a floor for how much of those gains could be
artifact.

## Step 14: Experimental design (M5)

**Goal.** Numbers that mean something: honest uncertainty, no leakage, reproducible from one
seed.

**What was built.** [`experiments/run_phase1.py`](../experiments/run_phase1.py) (one command),
[`phase1_data.py`](../experiments/phase1_data.py), [`phase1_analysis.py`](../experiments/phase1_analysis.py),
[`phase1_report.py`](../experiments/phase1_report.py), configs in
[`experiments/configs/`](../experiments/configs/).

**Design.**

* **Independent replicates.** Five seeds are derived from the master seed (`SeedSequence`).
  Each seed draws new populations (300 training and 150 test bots, 60 + 30 per archetype),
  simulates 2,000 hands per bot (100k + 50k hands), and refits everything: priors,
  classifiers, temperatures, regressors, action models.
* **Disjoint leagues.** Training and test populations are simulated in *separate* leagues.
  Test players never share a table with training players, never appear in a training row, and
  never touch prior fitting or calibration.
* **Intervals across seeds.** Headline intervals are Student-$t$ intervals over the $S = 5$
  seed-level results:

  $$\bar x \pm t_{0.975,\,S-1} \frac{s}{\sqrt S}, \qquad t_{0.975,\,4} = 2.776$$

  They capture every source of randomness: population, cards, split, model fitting.
* **Within-seed intervals** (kept in the per-seed JSON) use a *cluster bootstrap*: resample
  test **players** with replacement and recompute the mean of per-player means. A player's
  windows are correlated, so resampling individual windows would understate uncertainty.
* **Controls.** An *oracle* (the classifier trained on the true parameters, i.e. infinitely
  many hands) gives the ceiling. A *permutation control* shuffles archetype labels across
  training players and retrains. If anything leaked from labels into features or across the
  split, it would score above chance. It scored 19.5% [13.2, 25.8] against 20% chance.
* **Paired ablations.** Shrunk vs raw features are compared on the same seed, players and
  windows. The interval is computed on the per-seed *differences*, which removes shared
  variation and is much tighter than comparing two separate intervals.
* **"Hands needed" summaries.** For a target accuracy $a^*$, interpolate linearly in $\log n$
  between the grid points that bracket it:

  $$\log n^* = \log n_i + \frac{a^* - a_i}{a_{i+1} - a_i}\,(\log n_{i+1} - \log n_i)$$

  then summarize across seeds with the *geometric* mean (sample sizes are multiplicative) and
  the range.
* **Pre-registration vs post hoc.** The primary model (logistic on shrunk stats), the four
  variants, global temperature scaling, regression vs the population mean and action
  prediction were fixed before the reported run. Three analyses added after seeing the first
  run (per-size temperatures, the no-sizing ablation, within-archetype $R^2$) are labeled
  *post hoc* everywhere. Pre-registered metrics keep their names and reproduced exactly when
  re-run.
* **Provenance.** Every results file records the seed, the full config, the git commit, a
  dirty-tree flag (captured at the *start* of the run, so editing docs during a long run
  cannot corrupt it), package versions, and a *simulation fingerprint*. The fingerprint is a
  SHA-256 hash of the simulation source files, shipped data and PokerKit/phevaluator versions,
  and it is part of every cache key, so changed code can never silently reuse stale cached
  simulations.

## Step 15: Phase 1 results and how to read them (M5)

All numbers are means over 5 seeds with 95% t-intervals across seeds, on held-out players.
Full tables: [`experiments/results/phase1/phase1_summary.md`](../experiments/results/phase1/phase1_summary.md).

**How many hands to identify a type?** The primary model (logistic regression on shrunk
stats) reaches 73.9% accuracy after 25 hands, 81.4% after 50, 86.4% after 100, 90.1% after
200, 93.5% after 500 and 95.7% after 2,000. The ceiling with *infinite* hands (oracle on true
parameters) is 99.2%, and chance is 20%. Interpolated, 80% takes ~44 hands (40-49 across
seeds), 90% ~194 (150-264) and 95% ~934 (reached in 4 of 5 seeds within 2,000 hands).

**How much of that rests on bet sizing?** Bots size their bets by archetype with little noise,
a "tell" humans give less reliably. With the sizing features neutralized (post hoc), 80% takes
~68 hands and 90% ~334, about 1.5-1.7x more. The gap closes beyond ~500 hands. So sizing speeds
up early identification but is not what makes identification possible.

**Simple beat complex.** Boosted trees were never better than logistic regression and had
worse log-loss everywhere (90% at ~433 hands). With 13 smooth features and a few thousand
correlated training rows, the linear model's bias is an asset.

**Shrinkage helps the linear model, not the trees.** For logistic regression, shrunk features
lower log-loss at every sample size and add 2-5 accuracy points from 200 hands up. For
boosted trees the difference is indistinguishable from zero almost everywhere. That the
logistic benefit *grows* with hands was unexpected. The working hypothesis, not yet tested:
rare-opportunity stats (a calling station seldom open-raises, so it rarely faces a 3-bet) stay
noisy even at 2,000 hands, and raw rates of exactly 0 or 1 become extreme logits that a linear
model cannot ignore, while trees are insensitive to extreme but monotone values.

**Calibration.** The logistic model needs almost no global correction ($T = 0.99$), but its
miscalibration depends on sample size: overconfident at 25 hands, underconfident from 200 up.
Per-size temperatures (post hoc) cut ECE from 0.031-0.057 to 0.015-0.025 at 200+ hands. Boosted
trees are overconfident ($T = 1.50$); scaling cuts their ECE from 0.08 to 0.03 at 25 hands.

**Confusions are between neighbors.** Nit with TAG, TAG with LAG, LAG with maniac. Stations
are identified 98% of the time after 25 hands (no other type limps that much). At 2,000 hands
the remaining errors are nit->TAG (8%) and LAG->TAG (6%), where the parameter distributions
genuinely overlap.

**Parameters beyond the type are mostly not recovered** (Step 12): within-type $R^2$ +0.35 on
average at 2,000 hands; bluff frequency negative.

**Predicting the next action.** Against the population baseline (situation only, log-loss
0.771-0.774 nats), a profile built from just 25 hands cuts log-loss by 11.2% and closes 81% of
the gap to the parameter oracle. With 100 hands: 12.0% and 90%; with 500: 12.8% and 94.5%.
From 100 hands on, the profile model even beats an oracle that knows only the true archetype
(0.678 vs 0.682). Stats *are* action frequencies, so they carry within-type information useful
for prediction even when individual latent parameters are hard to pin down.

**What it means for the home game (Phase 4).** A live table deals roughly 25-35 hands an hour,
and home games often fewer. On these bots, 80% identification needs about 1.5-2 hours of play
against an opponent, 90% a full 6-8 hour session, and most of the action-prediction benefit
arrives within the first hour. Humans will be slower: no reliable sizing tells (68 and 334
hands above), non-stationary play, and a type taxonomy that is only an approximation. Profiles
must therefore accumulate across sessions (`ProfileStore`) to be useful.

## Step 16: Interfaces for Phases 2-5 (M6)

**Goal.** Typed seams that later phases plug into, each with a minimal working implementation
and *contract tests* every future implementation must pass.

* **Abstract base classes + contract tests.** Each interface (`StrategyProvider`,
  `ExploitEngine`, `Evaluator`) is an `ABC` with a documented contract. A parametrized test
  runs every implementation through the same checks: legal output for every legal view, no
  mutation, exact input types. Phase 2's CFR strategies will be added to the same parameter
  list.
* **`ActionDistribution`** ([`strategy/base.py`](../poker_lab/strategy/base.py)): a validated
  mixed strategy over concrete decisions. The total-variation distance
  $\tfrac12 \sum_d |p(d) - q(d)|$ measures how far an exploit deviates from its baseline.
* **Exploitability** ([`exploit/base.py`](../poker_lab/exploit/base.py)). In a two-player
  zero-sum game with value $v^*$ for player 1, the exploitability of player 1's strategy
  $\sigma_1$ is how far below the value a best-responding opponent can hold it:

  $$\operatorname{expl}(\sigma_1) = v^* - \min_{\sigma_2} u_1(\sigma_1, \sigma_2) \ \ge 0$$

  It is zero exactly for equilibrium strategies and is usually reported in milli-big-blinds
  per hand (mbb/hand). Exploiting a weak opponent means deviating from equilibrium, which
  raises your own exploitability; *restricted Nash responses* (Johanson et al., 2008) trade the
  two off with one parameter. `ExploitabilityEstimate` has an explicit `not_computed` state, so
  a missing number can never be mistaken for zero. Computing it is Phase 2.
* **Win rates** ([`evaluation/winrate.py`](../poker_lab/evaluation/winrate.py)). bb/100 is 100
  times the mean net result in big blinds per hand. With a per-hand standard deviation
  $\sigma$ (about 8-12 bb in no-limit), a 95% interval of half-width $h$ bb/100 needs

  $$n = \left(\frac{1.96\,\sigma}{h / 100}\right)^2 \text{ hands}$$

  For example, $\sigma = 10$ and $h = 5$ give about 154,000 hands. That is why variance
  reduction matters:
  * *Duplicate dealing:* play the same cards twice with seats swapped. If a seat's result is
    skill + luck in one seating and skill - luck in the other, the pair's mean is pure skill.
    The paired estimator is implemented; the engine hook that replays decks is a Phase 2 task.
  * *AIVAT* (Burch et al., 2018): subtract *control variates*, terms with expected value zero
    built from a value function at chance events and at the hero's own decisions:
    $\tilde X = X - \sum_t \big(V(s_{t+1}) - \mathbb E[V(s_{t+1}) \mid s_t]\big)$. The estimate
    stays unbiased, and its variance drops when $V$ tracks the luck. The hook refuses to run
    until a value function exists (Phase 2).
* **`ProfileStore`** ([`profiling/profile_store.py`](../poker_lab/profiling/profile_store.py)):
  per-player counter totals across sessions. Ingestion is *idempotent* (keyed by hand id, so
  re-importing a session does not double count), accepts completed hands only, redacts unshown
  cards, persists to JSON and can refit empirical-Bayes priors on its own population.
* **`Advisor`** ([`advisor/advisor.py`](../poker_lab/advisor/advisor.py)): the firewalled entry
  point (Step 3). It accepts exactly a `SeatView` and a `ProfileSnapshot`, refuses snapshots
  whose source hands include the live hand, and validates every strategy it returns against
  the legal actions. `AdvisorAgent` lets it play in simulation, building one snapshot per hand
  (profiles cannot change mid-hand).
* **`CaptureIngestor`** ([`capture/ingestor.py`](../poker_lab/capture/ingestor.py)) is *sensor
  fusion with an audit trail.* Per card location, a manual entry wins over the camera, the
  camera's highest-confidence read wins over others, and every disagreement, low-confidence
  read, duplicate card, partial hole-card pair, missing board card, inconsistent amount,
  illegal sequence (by replaying the log through PokerKit) or payout that contradicts a fully
  known showdown becomes a typed `Conflict` for human review. Nothing is silently fixed.

## Step 17: Small games and a tree compiler (P2.1)

**Goal.** Poker games small enough to solve *exactly*, represented so that solvers are both
fast and simple. Exact answers turn every Phase 2 question ("how exploitable is this?", "how
much does exploiting this bot gain?") into a computation instead of a noisy simulation.

**What was built.**
* [`games/base.py`](../poker_lab/games/base.py): the `Game` interface. The rules are eight
  pure functions on immutable states: initial state, is-terminal, whose turn, legal actions,
  chance outcomes, next state, payoff, and information-set key.
* [`games/kuhn.py`](../poker_lab/games/kuhn.py) and
  [`games/leduc.py`](../poker_lab/games/leduc.py). Leduc takes the deck, ante, bet sizes and
  raise cap as parameters, so larger variants can be built later to test scaling.
* [`games/tree.py`](../poker_lab/games/tree.py): `compile_tree` turns a game into a
  `GameTree` of flat NumPy arrays.
* [`games/policy.py`](../poker_lab/games/policy.py): `TabularPolicy`, a strategy for both
  players with one probability per (information set, action).

**The rules.**
* *Kuhn poker*: cards J < Q < K. Each player antes 1 and gets one card. Player 0 checks or
  bets 1. After a check, player 1 checks (showdown for the antes) or bets 1. Facing a bet, a
  player folds or calls, and a call goes to showdown.
* *Leduc hold'em*: six cards (J, Q, K in two suits). Each player antes 1 and gets one private
  card. In round 1 bets and raises are 2 chips, with at most two per round (a bet counts as the
  first). Then one public card is dealt and round 2 is played the same way with bets of 4.
  Player 0 acts first in both rounds. At showdown, a private card that pairs the public card
  wins; otherwise the higher card wins, and equal ranks split. The largest possible pot is
  13 chips from each player.

**Concepts.**

* **Describe the rules once, compile once.** Everything after the `Game` interface works on
  compiled arrays. This separates two questions that are easy to confuse when debugging. "Are
  the poker rules right?" is tested against hand-computed payoffs. "Is the algorithm right?"
  is tested against brute force and an independent LP solver (Step 18).
* **Breadth-first numbering makes tree algorithms into array operations.** Numbering nodes
  breadth first puts each depth level in a contiguous block, every node after its parent, and
  each node's children next to each other. The two passes every algorithm needs become one
  vectorized line per level:
  * *top-down (reach probabilities):* `reach[level] = reach[parent[level]] * edge[level]`;
  * *bottom-up (expected values):* `v[parent] += edge * v[child]`, summed over siblings with
    `np.bincount(parent, weights=...)`.

  Leduc has 9,457 nodes and 11 levels, so a full expected-value pass is about 20 small NumPy
  calls (well under a millisecond) instead of a recursive walk over 9,457 Python objects.
* **Sequences.** A *sequence* is an (information set, action) pair. Sequences are numbered so
  that each information set's actions are contiguous, so a whole strategy profile is one
  vector (2,184 numbers for Leduc). Per-information-set sums, for normalizing or for regret
  matching later, are a single `np.add.reduceat(x, offsets)`.
* **Counting the information sets** is a good check that you understood the game.
  * *Kuhn:* 3 cards $\times$ 4 betting points (player 0 at the start and after check-bet,
    player 1 after a check or a bet) = 12.
  * *Leduc round 1:* 6 betting points (player 0 at `""`, `kb` and `br`; player 1 at `k`, `b`
    and `kbr`) $\times$ 6 private cards = 36.
  * *Leduc round 2:* five round-1 lines reach the public card (`kk`, `bc`, `kbc`, `brc`,
    `kbrc`). Each has 6 betting points $\times$ 6 private cards $\times$ 5 public cards, so
    $5 \times 6 \times 30 = 900$.
  * *Leduc total:* $36 + 900 = 936$, the standard count. Ignoring suits (they never matter in
    Leduc) would give 288, but the standard game keeps them, and comparing with published
    numbers requires the standard game.
* **The information-set key is the small-game firewall.** A policy is a function of the key,
  and the key contains only the actor's own card and public information. A test visits all
  3,780 Leduc decision states, swaps the opponent's card for every unseen card, and checks that
  the key never changes.
* **The compiler checks its own assumptions.**
  * chance probabilities are positive and sum to 1;
  * each information set has one owner and one list of actions;
  * all of an information set's nodes sit at the same depth, which the level-by-level best
    response in Step 18 needs;
  * every node of an information set is reached by the same sequence of the owner's own
    actions (perfect recall). A test game whose player "forgets" their first action is
    rejected.

**Verification.** [`tests/test_games.py`](../tests/test_games.py):
* *Tree sizes.* Kuhn has 58 nodes, 30 leaves and 12 information sets. Leduc has 9,457 nodes,
  5,520 leaves, 936 information sets and 2,184 sequences.
* *Payoffs.* Nine hand-computed Leduc outcomes, including the 13-chip maximum, folds in both
  rounds and a split pot.
* *Betting rules.* The raise cap and illegal-action errors.
* *Vectorized values.* Expected values from the arrays equal a plain recursive evaluation to
  $10^{-12}$ for random policies in both games.
* *Structure.* Children are contiguous and point back to their parents, and leaf reach
  probabilities sum to 1.
* *Firewall and recall.* The card-swapping firewall test and the imperfect-recall rejection.
* *Policy validation.* `TabularPolicy` construction and validation.

## Step 18: Exact evaluation: best responses, exploitability and an LP solver (P2.2)

**Goal.** Measure any strategy exactly (its value, the best response to it, its
exploitability), and compute an exact equilibrium to check every later solver against.

**What was built.** [`games/evaluate.py`](../poker_lab/games/evaluate.py) (`expected_value`,
`head_to_head`, `best_response`, `exploitability_report`) and
[`solvers/lp.py`](../poker_lab/solvers/lp.py) (`sequence_form`, `solve_lp`).

**Concepts and math.**

* **Expected value** is one bottom-up pass (Step 17).
* **Best response by backward induction over information sets.** Fix the opponent's strategy.
  Let $\pi_{-i}(h)$ be the probability that chance and the opponent play to history $h$, with
  the responder's own choices counted as 1. At the responder's information set $I$, the
  *counterfactual value* of action $a$ is

  $$q(I, a) = \sum_{h \in I} \pi_{-i}(h)\, v(h \cdot a),$$

  where $v$ below $I$ already uses the responder's best choices. The responder plays
  $\arg\max_a q(I, a)$ at every node of $I$.

  Why weight by $\pi_{-i}(h)$? By Bayes' rule, the responder's belief that it is at $h$
  (rather than another history in $I$) is proportional to $\pi(h)$. Its own contribution to
  $\pi(h)$ is identical for all $h \in I$ (perfect recall), so it cancels, leaving
  $\pi_{-i}(h)$. Ties go to the first action, so the result is deterministic.
* **Why one level at a time is enough.** Every node of an information set is at the same
  depth (the compiler checks this). Processing the deepest level first therefore means every
  child value is final when $I$ is decided. Each level is vectorized:
  1. `np.bincount` sums $\pi_{-i}(h)\,v(h\cdot a)$ into $q$;
  2. `np.maximum.reduceat` finds each information set's best value;
  3. `np.minimum.reduceat` over candidate indices picks the first action that attains it.
* **Worked example: uniform random Kuhn.** Against an opponent who picks uniformly
  everywhere, these are the best responses and their values:

  | Card | Player 0's best response | Value | Player 1's best response | Value |
  |---|---|---|---|---|
  | J | bet (bluff): $\tfrac12(+1) + \tfrac12(-2)$ | $-0.5$ | fold to a bet; bet after a check | $-0.75$ |
  | Q | bet: $\tfrac12(+1) + \tfrac12 \cdot 0$ | $+0.5$ | call a bet; bet after a check | $+0.25$ |
  | K | bet or check | $+1.5$ | call; bet after a check | $+1.75$ |
  | **mean** | | $\mathrm{BR}_0 = 0.5$ | | $\mathrm{BR}_1 = 5/12$ |

  So NashConv $= 1/2 + 5/12 = 11/12$. The code reproduces this exactly. For Leduc it gives
  NashConv $= 4.747222\ldots$, the value OpenSpiel's test suite uses for the uniform policy.
  That match is good evidence that the Leduc rules are the standard ones.
* **The sequence form: equilibria by linear programming.** In a behavior strategy, the
  probability of reaching a leaf is a *product* of the player's action probabilities, which
  is not linear. The sequence form instead uses the *realization plan* $x[s]$: the probability
  that the player's own actions along the path include sequence $s$. Realization plans
  satisfy linear constraints:

  $$x[\varnothing] = 1, \qquad \sum_{a} x[I, a] = x[\text{parent sequence of } I]
  \quad \text{for every } I, \qquad x \ge 0,$$

  written $Ex = e$ for player 0 and $Fy = f$ for player 1. The expected payoff is bilinear,
  $x^\top A y$, where $A[s_0, s_1]$ sums (chance probability $\times$ payoff) over the leaves
  whose last own sequences are $s_0$ and $s_1$.

  Player 0 maximizes $\min_y x^\top A y$. For a fixed $x$ the inner minimum is itself an LP,
  and LP duality replaces it by $\max_q f^\top q$ subject to $F^\top q \le A^\top x$. The
  result is a single LP for player 0's equilibrium strategy and the game value:

  $$\max_{x, q}\ f^\top q \quad \text{s.t.} \quad F^\top q - A^\top x \le 0,\ \ Ex = e,\ \ x \ge 0.$$

  Player 1's LP is symmetric. The behavior strategy is recovered as
  $\sigma(I, a) = x[I, a] / x[\text{parent}]$. Leduc has 1,092 sequences per player, and
  SciPy's HiGHS solver handles each LP in about 0.1 s. The sequence form is *linear* in the
  size of the game tree. The alternative, listing every pure strategy, is exponential (Kuhn
  already has $2^6 = 64$ pure strategies per player; Leduc has astronomically many).

**Three independent checks agree.**
* *Published values.* The LP reproduces Kuhn's value $-1/18$ and the published Leduc value
  $-0.0856064$, and its strategies have exploitability below $10^{-9}$.
* *Closed form.* The known Kuhn equilibrium family has exploitability 0 (to $10^{-12}$) at
  four values of its parameter $\alpha \in [0, 1/3]$, and leaving the family makes it
  exploitable. Player 0 bets J with probability $\alpha$ and K with $3\alpha$, and calls a bet
  with Q with probability $\alpha + 1/3$; player 1 bluffs J and calls with Q one third of the
  time.
* *Brute force.* On Kuhn, the best-response value equals the maximum over all 64 pure
  strategies for both players. On Leduc, the reported best-response value is achieved by the
  returned policy, and no deviation at 60 randomly chosen information sets improves it.

**Verification.** [`tests/test_best_response.py`](../tests/test_best_response.py) (12 tests),
including `head_to_head`. An equilibrium never loses on average over both seats. A best
response to a fixed opponent beats it by exactly that opponent's exploitability.

**Pitfalls.**
1. *A name collision under lazy re-exports.* The function `best_response` lived in a module
   also called `best_response.py`. When Python imports a submodule, it sets an attribute of
   the same name on the package. After that, `from poker_lab.games import best_response`
   returned the *module*, and tests failed with "'module' object is not callable", but only in
   some test orders. The module is now `evaluate.py`. Rule: a submodule must never share a
   name with something its package re-exports.
2. *Renormalizing valid input changed it.* The policy constructor always divided by the
   per-information-set sums, which moved already-valid probabilities by one unit in the last
   place. An exact-equality test ("the opponent's part is untouched") caught it. The
   constructor now repairs only sums that are off by more than $10^{-12}$, so valid input is
   kept bit for bit.

## Step 19: CFR and its variants (P2.3)

**Goal.** Compute equilibria with counterfactual regret minimization (CFR), the algorithm
family behind every poker program that has beaten professionals. Measure how fast each
variant converges, and check them against the exact LP.

**What was built.**
* [`solvers/cfr.py`](../poker_lab/solvers/cfr.py): `CFRSolver`, full-width CFR with the
  variants below as configuration.
* [`solvers/mccfr.py`](../poker_lab/solvers/mccfr.py): `ExternalSamplingMCCFR`, the seeded
  Monte Carlo version.
* [`experiments/phase2_cfr.py`](../experiments/phase2_cfr.py): the convergence experiment.

**Concepts and math.**

* **Regret.** Repeat a decision $T$ times; on round $t$ you play a mixed strategy $\sigma_t$,
  and the environment reveals each action's payoff $u_t(a)$. Your *regret* for never having
  played action $a$ is

  $$R_T(a) = \sum_{t=1}^{T} \big[u_t(a) - u_t(\sigma_t)\big],$$

  where $u_t(\sigma_t) = \sum_b \sigma_t(b)\,u_t(b)$.
* **Regret matching** (Hart and Mas-Colell, 2000) plays each action in proportion to its
  positive regret:

  $$\sigma_{T+1}(a) = \frac{\max(R_T(a), 0)}{\sum_b \max(R_T(b), 0)},$$

  or uniformly if no regret is positive. It guarantees an average regret
  $R_T(a)/T \le \Delta\sqrt{|A|/T} \to 0$ against *any* sequence of payoffs, where $\Delta$ is
  the payoff range. That robustness is the point: the opponent is also learning, so their
  behavior is not a fixed distribution.
* **Why low regret gives an equilibrium.** Suppose both players' average regrets are at most
  $\varepsilon$, and let $\bar\sigma$ be the average strategies. Payoffs are linear in each
  player's (sequence-form) strategy, so player 1's best response against $\bar\sigma_2$ earns
  $\max_{\sigma'} \frac1T\sum_t u(\sigma', \sigma_2^t) \le \frac1T\sum_t u(\sigma_1^t, \sigma_2^t)
  + \varepsilon$. Symmetrically, player 2 can push player 1's payoff down to at least
  $\frac1T\sum_t u(\sigma_1^t, \sigma_2^t) - \varepsilon$. Adding the two gives
  NashConv $\le 2\varepsilon$, so the exploitability of $\bar\sigma$ is at most $\varepsilon$.
  Two consequences:
  * It is the *average* strategy that converges; the current strategy can cycle forever.
  * The average must be taken in the sequence form: iteration $t$'s strategy at $I$ is
    weighted by the player's own probability of reaching $I$, $\pi_i^{\sigma_t}(I)$.
* **Counterfactual regret** (Zinkevich et al., 2007). Running one regret matcher per
  information set works if each one is fed *counterfactual values*: Step 18's
  $q(I, a) = \sum_{h \in I} \pi_{-i}(h)\,v(h \cdot a)$. The instantaneous regret is
  $r_t(I, a) = q(I, a) - \sum_b \sigma_t(I, b)\,q(I, b)$. The theorem: a player's total regret
  is at most the sum of these local regrets, so the average regret is
  $O(\Delta\,|\mathcal I|\sqrt{|A|}/\sqrt T)$. Vanilla CFR therefore converges like
  $1/\sqrt T$.
* **Vectorized implementation.** One iteration for player $i$ takes:
  1. one backward pass for node values;
  2. two forward passes, for the reach probabilities of "everyone but $i$" and of $i$;
  3. one `np.bincount` for $q$;
  4. one `np.add.reduceat` for $\sum_b \sigma q$.

  On Leduc this is about 0.9 ms per player.
* **The variants** differ only in how regrets and averages are accumulated:

  | Variant | Regret update | Average weight | Updates |
  |---|---|---|---|
  | CFR | $R \leftarrow R + r$ | 1 | simultaneous |
  | CFR+ | $R \leftarrow \max(R + r, 0)$ | $t$ | alternating |
  | Linear CFR | iteration $t$'s regret weighted by $t$ | $t$ | alternating |
  | Discounted CFR | positive $R$ $\times\, t^{1.5}/(t^{1.5}+1)$, negative $R$ $\times\, \tfrac12$ | $t^2$ | alternating |

  * *CFR+'s floor* (regret matching+): an action with very negative regret is played again as
    soon as it becomes good. It does not have to first "pay back" its debt.
  * *Weighted averages* forget the poor early iterations.
  * *Alternating updates* let player 1 respond to player 0's already-updated strategy.
  * *Discounted CFR* generalizes linear CFR with three exponents: $\alpha$ (positive regrets),
    $\beta$ (negative regrets) and $\gamma$ (average weight $t^\gamma$). Multiplying the
    running sum by $t/(t+1)$ every iteration makes iteration $t$'s final weight
    $\prod_{s=t}^{T} \frac{s}{s+1} = \frac{t}{T+1} \propto t$. That is how "weight by $t$" is
    implemented without storing history.
* **External-sampling MCCFR** (Lanctot et al., 2009). Each iteration runs one *traversal* per
  player:
  * the traverser explores all of their own actions;
  * chance outcomes are sampled from the chance probabilities;
  * the opponent's actions are sampled from their current strategy.

  *Unbiased.* A traversal for player $i$ reaches $h$ with probability $\pi_c(h)\,\pi_{-i}(h)$,
  so the expected sampled regret at $I$ is
  $\sum_{h\in I}\pi_c(h)\pi_{-i}(h)\,[v(h\cdot a) - v(h)]$. This is exactly the
  counterfactual regret (sampled values below $h$ are themselves unbiased). An iteration costs
  tens of nodes instead of the whole tree, at the price of noise.

  *Averaging.* The average strategy is accumulated at the *opponent's* information sets. They
  are visited with probability proportional to the opponent's own reach, which is the weight
  the average needs.
* **Measuring convergence fairly.**
  * *Exact evaluation.* Exploitability is evaluated exactly (Step 18) at log-spaced
    checkpoints.
  * *Work unit.* Methods are compared by *tree nodes touched*, because "iterations" mean very
    different amounts of work for full-width and sampled CFR.
  * *Seeds only where there is randomness.* Full-width CFR is deterministic, so seeds would
    repeat an identical curve.
  * *MCCFR summaries.* Five seeds are summarized on the log scale: a geometric mean and a
    t-interval of $\log_{10}$ exploitability. Values span orders of magnitude, and the log
    makes their spread closer to symmetric.

**Results** ([`cfr_summary.md`](../experiments/results/phase2/cfr/cfr_summary.md), commit
`53eb199`). Exploitability in chips per hand; the uniform policy starts at 0.458 (Kuhn) and
2.37 (Leduc).

| Leduc, iterations | 10 | 100 | 1,000 | 10,000 |
|---|---|---|---|---|
| CFR | 0.93 | 0.17 | 0.040 | 0.010 |
| CFR+ | 0.61 | 0.013 | 2.6e-4 | 5.7e-6 |
| Linear CFR | 0.72 | 0.035 | 5.2e-3 | 1.1e-3 |
| Discounted CFR | 0.78 | 7.8e-3 | 1.6e-4 | 8.4e-6 |

1. *Vanilla CFR follows its theory.* It runs parallel to the $1/\sqrt T$ reference line.
   CFR+ and discounted CFR go faster than $1/T$ on Leduc and end 1,200 to 1,800 times less
   exploitable than vanilla CFR after 10,000 iterations.
2. *Linear CFR is fast on Kuhn but stalls on Leduc* (200 times worse than CFR+ at 10,000
   iterations). A post-hoc ablation over the exponents $(\alpha, \beta, \gamma)$ explains most
   of it (Leduc, 1,000 iterations):
   * *Negative regrets (the biggest factor).* Keeping them ($\beta = 1$) makes discounted CFR
     77 times worse (0.012 vs 1.6e-4).
   * *Positive-regret discount.* Moving $\alpha$ from 1 to 1.5 gains about 8 times when
     $\beta = 0$.
   * *Averaging weight.* $\gamma$ matters least.

   Linear CFR has $\alpha = \beta = 1$, so it loses on the two factors that matter most.
3. *Sampling is a bad deal in a game this small.* At equal work (13.4 million nodes),
   external-sampling MCCFR on Leduc reaches 0.041 [0.038, 0.045] (5 seeds), about 80 times
   more exploitable than CFR+ at the same work. To get below 0.05 it needs about 10 times
   more node visits. Linear weighting does not change MCCFR measurably (0.0409). Sampling
   pays off when a single full traversal is unaffordable, which is the regime Deep CFR
   (next step) is built for.
4. *Cost.* Vectorized full-width CFR costs 1.6-1.9 ms per Leduc iteration (both players),
   and the LP solves Leduc exactly in 0.3 s. At this size the LP is the fastest exact method;
   CFR's advantage is that its memory and time grow gracefully with game size, while LP
   solvers do not.

**Verification.** [`tests/test_cfr.py`](../tests/test_cfr.py) (10 tests):
* *Textbook equality.* Vectorized CFR equals a plain recursive textbook CFR after three
  iterations on Kuhn *and* Leduc: regrets to $10^{-10}$, average strategies after
  normalization.
* *Regression thresholds* on exploitability after 1,000 iterations for every variant and game.
* *LP agreement.* CFR+'s average value is within $10^{-3}$ of the LP value.
* *MCCFR reproducibility* from its seed.
* *Unbiasedness.* A statistical test: over 40,000 traversals, the mean sampled regret is within
  four standard errors of the exact counterfactual regret for every player-0 sequence of Kuhn.

**Pitfalls.**
1. *Two correct implementations, different raw numbers.* The recursion adds the
   own-reach-weighted strategy once per *history*; the vectorized code adds it once per
   *information set*. The raw sums differ by a constant factor per information set, so the
   differential test compares normalized averages.
2. *Seeds without randomness.* The original plan asked for five seeds and intervals on every
   convergence curve. Deterministic algorithms would just repeat the same curve, so seeds are
   used only for MCCFR (logged as a decision).
3. *Checkpoints that missed powers of 10.* They left table cells empty in the first smoke run;
   `log_checkpoints` now aligns them.

## Step 20: Deep CFR in PyTorch (P2.4, P2.10)

**Goal.** Replace CFR's regret table with neural networks, the step that lets CFR scale to
games too large to enumerate, and measure honestly what that costs in a game small enough to
check exactly.

**What was built.**
* [`solvers/deep_cfr.py`](../poker_lab/solvers/deep_cfr.py): `DeepCFR`, `InfosetEncoder`,
  `Reservoir`.
* [`experiments/phase2_deep_cfr.py`](../experiments/phase2_deep_cfr.py): the experiment.
* PyTorch 2.14.1 (CPU) as the `nn` extra. Its version is recorded in every results file.

**Concepts and math.**

* **Why networks.** A table needs a regret for every information set, and NLHE has about
  $10^{160}$ of them. A network that maps information-set *features* to regrets generalizes:
  similar situations get similar regrets without being visited. The rank-only features here
  merge Leduc's 936 information sets into 288 classes (the suits never matter), so even this
  small network shares statistics across suit-isomorphic situations. A test checks that count.
* **The algorithm** (Brown, Lerer, Gross, Sandholm 2019). On iteration $t$, for each player
  $p$:
  1. *Traverse.* Run $K$ external-sampling traversals (Step 19). At $p$'s information sets,
     play regret matching on the *predicted* advantages $V_p(I)$. If none is positive, play the
     largest. Store $(I, t, \tilde r)$ in $p$'s advantage memory, where
     $\tilde r(a) = v(a) - \sum_b \sigma(b)\,v(b)$ are the sampled regrets. Store
     $(I, t, \sigma(I))$ in the strategy memory at the opponent's information sets.
  2. *Fit.* Re-initialize $V_p$ and train it from scratch on the whole memory, minimizing
     $\sum_i t_i \,\lVert V_p(I_i) - \tilde r_i \rVert^2$ over legal actions.

  At the end, a policy network $\Pi$ is fit to the strategy memory (cross-entropy, weights
  $t$).
* **Why that regression target is right.** For a fixed $I$, the minimizer of
  $\sum_i t_i (V - \tilde r_i)^2$ is the weighted mean $\sum t_i \tilde r_i / \sum t_i$. That is
  proportional to linear CFR's cumulative regret, estimated from samples, and regret matching
  ignores the proportionality constant. Likewise $\Pi$'s minimizer is linear CFR's average
  strategy. A network with enough capacity, trained to convergence, would therefore reproduce
  sampled linear CFR. Everything Deep CFR loses comes from finite data, finite capacity and
  imperfect training.
* **Reservoir sampling** (Vitter's algorithm R). Keep the first $C$ items. For the $n$-th
  item ($n > C$), overwrite a random slot with probability $C/n$. By induction every item seen
  so far is in memory with probability exactly $C/n$, so the memory is always a uniform sample
  of everything generated, using fixed space. A statistical test checks this.
* **A small-game shortcut that changes nothing.** Within an iteration the networks are frozen,
  so the strategy is a fixed function of the information set. Evaluating the network once on
  all 936 information sets and traversing with that table is therefore *identical* to querying
  the network at every visited node, just faster.
* **Exact evaluation of four policies.** Because everything can be enumerated:
  1. the policy network (Deep CFR's output);
  2. the strategy-memory average ($\Pi$'s regression target);
  3. the *exact* reach- and $t$-weighted average of the strategies the advantage networks
     played, which is what Single Deep CFR (Steinberger 2019) plays;
  4. the current strategy.

  Comparing them separates the error of fitting $\Pi$ from everything else.
* **A controlled ablation: the perfect regressor.** Replace each network by the exact minimizer
  of its loss (per-information-set weighted means), keeping traversals and memories identical.
  The comparisons then read:
  * *tabular MCCFR → perfect regressor:* the cost of updating the strategy only once per $K$
    traversals;
  * *perfect regressor → Deep CFR:* the cost of function approximation.
* **Fair comparisons.**
  * *Matched traversals.* Tabular external-sampling MCCFR with linear weights gets $KT$
    iterations of 2 traversals each, the same budget.
  * *Tuning off the reporting seeds.* The training budget was tuned on development seed 99,
    never on a reporting seed, by how well the network fits a realistic memory snapshot
    ($R^2$ against the per-information-set targets):

    | Steps | Learning rate | $R^2$ | Time |
    |---|---|---|---|
    | 300 | $10^{-3}$ constant | 0.875 | 1.0 s |
    | 600 | $10^{-3}$ constant | 0.929 | 1.9 s |
    | 600 | $3\cdot10^{-3}$ cosine | 0.943 | 1.9 s |
    | 1,000 | $3\cdot10^{-3}$ cosine | 0.954 | 3.2 s |

    600 steps with cosine decay was chosen.

**Results** ([`deep_cfr_summary.md`](../experiments/results/phase2/deep_cfr/deep_cfr_summary.md),
commit `d09ad62`; 5 seeds; geometric means with 95% intervals; uniform policy at 2.37):

| Method | 20k traversals | 60k | 200k |
|---|---|---|---|
| Tabular ES-MCCFR, linear weights | 0.240 [0.211, 0.273] | 0.141 [0.133, 0.150] | 0.072 [0.069, 0.075] |
| Deep CFR with a perfect regressor | 0.563 [0.504, 0.628] | 0.180 [0.157, 0.206] | 0.087 [0.074, 0.102] |
| Deep CFR, policy network | 0.537 [0.408, 0.707] | 0.230 [0.190, 0.278] | 0.161 [0.134, 0.193] |
| Deep CFR, exact average (SD-CFR) | 0.540 [0.405, 0.720] | 0.230 [0.194, 0.272] | 0.160 [0.136, 0.188] |
| Deep CFR, current strategy | 0.843 | 0.972 | 1.196 |

1. *Deep CFR works but plateaus.* It falls from 2.37 to 0.16 chips per hand. From iteration 50
   to 100 it barely moves (0.18 → 0.16), while the perfect regressor keeps improving
   (0.12 → 0.087). This is the signature of an approximation-error floor: more data does not
   fix errors the network makes near zero advantage, where regret matching is most sensitive.
2. *The cost decomposes cleanly.* At 200k traversals, batching costs a factor of 1.2
   (0.072 → 0.087) and function approximation a factor of 1.85 (0.087 → 0.161). Early on,
   batching dominates: with only 10 strategy updates, the perfect regressor is 2.3 times worse
   than tabular MCCFR.
3. *Fitting the policy network is not the bottleneck.* $\Pi$, the memory average and the exact
   average agree to three decimals. In this game Single Deep CFR (no $\Pi$) would give the same
   answer.
4. *The current strategy does not converge* (0.84 → 1.20). Only averages carry CFR's
   guarantee.
5. *Cost.* About 5.7 s per iteration (single thread, 8 runs in parallel), against microseconds
   per tabular iteration. In Leduc, Deep CFR loses to a table on every axis. That is expected:
   its purpose is games where no table fits. The value of this step is a correct, measured
   implementation whose error sources are known before it is pointed at larger games.

**Verification.** [`tests/test_deep_cfr.py`](../tests/test_deep_cfr.py) (5 tests; skipped
without PyTorch):
* the encoder gives 288 distinct rows and correct legal masks;
* the reservoir keeps a uniform sample;
* the perfect-regressor variant converges on Kuhn;
* the network variant is reproducible from its seed, and its trajectory does not depend on
  when it is evaluated;
* the config is validated.

**Pitfalls.**
1. *Evaluation changed the experiment.* Training the policy network at a checkpoint drew new
   random weights from PyTorch's global generator. That shifted the initialization of every
   later advantage network, so the same seed gave 0.39 or 0.44 depending on the evaluation
   schedule. Every network now gets a seed derived from (seed, iteration, role) inside
   `torch.random.fork_rng`, and a test checks that extra checkpoints change nothing.
2. *A tie broken badly.* Before training, all advantages are zero. The "play the largest"
   rule then picked the first legal action (fold or check) with probability 1. Untrained
   players now play uniformly.
3. *A wrong test, not a wrong algorithm.* I expected both players to have played uniformly
   after iteration 1. With alternating updates, player 0's network is retrained *before* player
   1 traverses, exactly as in the paper's Algorithm 1. The test was corrected.
4. *A diagnostic that silently failed.* A scratch script outside the repo could not import
   `poker_lab` (Python puts the *script's* directory on the path, not the working directory),
   and a polling loop waited ten minutes for output that would never come. Set `PYTHONPATH`, and
   check a background job's first lines before waiting on it.

## Step 21: Leduc archetypes and modeling them from what a player can see (P2.6)

**Goal.** Bring Phase 1's opponent modeling into a game where its consequences for
exploitation can be computed *exactly*.

**What was built.** [`games/leduc_bots.py`](../poker_lab/games/leduc_bots.py): bots, simulation,
likelihood, fitting, and two controls (public visit counts and a card-blind model).

**Concepts and math.**

* **A bot is a behavior strategy.** In a small game the whole bot fits in a `TabularPolicy`
  (both seats). Five interpretable parameters generate it: `bluff`, `medium_aggression` and
  `value` (bet/raise probability with a weak, medium or strong hand), and `fold_weak` and
  `fold_medium` (fold to a bet; strong hands never fold). Not facing a bet, the bot bets with
  its bucket's aggression $a$. Facing one, it folds with the bucket's $f$; otherwise it raises
  with probability $a$ if raising is allowed, and calls.
* **Hand-strength buckets by equity.** In round 1 the bucket is the rank (J weak, Q medium,
  K strong). In round 2 it is the showdown equity against a random unseen card:
  * a pair wins 100%: strong;
  * an unpaired card higher than the third rank (e.g. K on a Q board) wins 62.5%: medium;
  * anything else wins 12.5%: weak.

  So Q on a K board is "medium", like K on a Q board. A test pins this down.
* **Archetypes as distributions.** Each parameter is drawn from
  $\text{Beta}(\mu\kappa, (1-\mu)\kappa)$ with $\kappa = 30$, as in Phase 1. The five means range
  from the nit (bluff 0.05, folds weak hands 90%) to the maniac (bluff 0.65, value 0.95).
  The archetypes are very exploitable: about 1.0 chips per hand on average, while the
  equilibrium itself already wins 0.28 per hand against them.
* **Seeing only what a player sees.** `simulate_hands` records the hero's card, the board and
  every action, but the bot's card only if the hand reaches showdown (46% of hands here). This
  is the small-game version of Phase 1's redaction.
* **The likelihood with a hidden card.** The bot's actions depend on its card, which is often
  unknown. Given the hero's card $h$ and board $b$, the bot's card is uniform over the unseen
  cards $C$ (independent of the hero's own actions), so

  $$P(\text{bot's actions} \mid \theta) = \frac{1}{|C|} \sum_{c \in C} \prod_t \sigma_\theta(a_t \mid I_t(c)).$$

  Computed in log space with a per-hand log-sum-exp over the candidates. The candidate rows
  of all hands are flattened into arrays, so a log-likelihood for thousands of hands is two
  `reduceat` calls. A test checks it against brute force.
* **Fitting: the posterior mode, in the right parameterization.** The prior is an independent
  Beta per parameter, fit by moments to a separate *training* population (never to the test
  bots). The estimate maximizes log-likelihood + log-prior over $z = \operatorname{logit}\theta$,
  *including the Jacobian* $\sum \log \theta(1-\theta)$. That is the mode of
  $\text{Beta}(a+1, b+1)$, which sits exactly at the prior mean $a/(a+b)$: with no data the
  estimate is the population mean, and data pull it away. The mean absolute parameter error
  falls from 0.12 (10 hands) to 0.011 (3,000 hands).

**Verification.** [`tests/test_leduc_exploit.py`](../tests/test_leduc_exploit.py):
* parameter semantics and buckets;
* observations hide unshown cards, and the mean payoff matches the exact value;
* the marginal likelihood equals brute force;
* parameters are recovered to within 0.06 from 3,000 hands;
* zero hands give exactly the prior mean, and with 20 hands the posterior mode is closer to the
  prior mean than the maximum-likelihood estimate.

**Pitfall: a mode pulled to the boundary.** The first fit maximized the posterior density in
$\theta$ itself. The population prior is diffuse ($a + b \approx 2$), and several shape
parameters were below 1, so the prior density is infinite at 0. The fold parameter of a
20-hand fit landed at 0.0001 instead of near the population mean, an artifact of the
parameterization. A test written to show shrinkage failed and exposed it. Two lessons:
* a posterior mode is not invariant to reparameterization (the posterior mean is);
* Beta priors with $a < 1$ make $\theta$-space MAP estimates collapse onto the boundary.

## Step 22: Exploiting the models, and how much it risks (P2.7)

**Goal.** The question this whole project builds toward, answered exactly in a small game:
*after watching an opponent for $n$ hands, how much more do you win by exploiting them, and
how exposed does that make you?*

**What was built.**
* [`solvers/robust.py`](../poker_lab/solvers/robust.py): `RobustResponder`, restricted Nash and
  data-biased responses as sequence-form LPs.
* [`experiments/phase2_exploit.py`](../experiments/phase2_exploit.py): the experiment.

**Concepts and math.**

* **Best response to a model** maximizes $x^\top A y_{\text{fix}}$. It is optimal if the model is
  right and arbitrarily bad if it is wrong.
* **Restricted Nash response** (Johanson, Zinkevich, Bowling 2008). With probability $p$ the
  opponent plays the model; otherwise it plays adversarially, and the hero does not know which.
  The hero maximizes $p\,x^\top A y_{\text{fix}} + (1-p)\min_y x^\top A y$. Dualizing the inner
  minimum exactly as in the equilibrium LP (Step 18) gives *one* LP:

  $$\max_{x,q}\ p\,(A y_{\text{fix}})^\top x + (1-p)\,f^\top q \quad\text{s.t.}\quad
  F^\top q - A^\top x \le 0,\ Ex = e,\ x \ge 0.$$

  $p = 0$ is an equilibrium and $p = 1$ a best response. Sweeping $p$ traces the frontier of
  gain against exploitability.
* **Data-biased response** (Johanson and Bowling 2009). Trust the model per information set:
  the opponent must play the model at $I$ with probability $c_I$. In the sequence form that is
  a *linear* constraint, $y[I,a] \ge c_I\,\sigma_{\text{fix}}(I,a)\,y[\text{parent}(I)]$. It
  dualizes into an extra non-negative variable block $r$:
  $F^\top q + G^\top r - A^\top x \le 0$. Here
  $c_I = n_I/(n_I + 10)$, where $n_I$ counts how often the bot was seen acting at $I$'s
  public state.
* **Why LPs rather than CFR on a modified game.** Both robust responses are LPs over the same
  polytopes as the equilibrium LP. HiGHS solves each exactly in about 0.15 s on Leduc, so the
  experiment needs no convergence tuning. Tests check the endpoints ($p = 0, 1$; $c = 0, 1$),
  monotonicity of the frontier, and that no strategy beats the response on its own objective.
* **The experiment** (pre-registered in the script's docstring):
  * *Replicates.* 5 seeds, each with its own training population (for the prior) and 30
    held-out bots.
  * *Observation.* The hero plays the exact equilibrium against each bot for 3,000 hands,
    seeing the bot's card only at showdown.
  * *Models.* One model from the first $n \in \{10, \ldots, 3000\}$ hands.
  * *Responses.* Best response, RNR for $p \in \{0.25, 0.5, 0.75, 0.9\}$, and DBR.
  * *Metrics, both exact.* Gain against the *true* bot over the equilibrium, and the
    response's exploitability.

**Results** ([`exploit_summary.md`](../experiments/results/phase2/exploit/exploit_summary.md),
commit `963cee6`; chips per hand; 95% intervals across seeds):

| Hands observed | 10 | 100 | 3,000 | Oracle |
|---|---|---|---|---|
| Best response: gain | +0.591 | +0.700 | +0.736 | +0.738 |
| Best response: exploitability | 1.48 | 1.61 | 1.60 | 1.58 |
| RNR $p = 0.9$: gain | +0.581 | +0.675 | +0.705 | +0.706 |
| RNR $p = 0.9$: exploitability | 0.70 | 0.71 | 0.69 | 0.69 |
| RNR $p = 0.5$: gain | +0.438 | +0.486 | +0.494 | +0.494 |
| RNR $p = 0.5$: exploitability | 0.13 | 0.14 | 0.13 | 0.13 |
| DBR: gain | +0.001 | +0.098 | +0.483 | |

1. *Exploitation pays off almost immediately here.* A best response to a 10-hand model gains
   +0.591 [0.563, 0.619] per hand over the equilibrium, and no bot in any seed made it lose.
2. *Robustness is cheap.* RNR with $p = 0.9$ keeps 96-98% of the best response's gain at
   about 45% of its exploitability. With $p = 0.5$ it keeps two thirds of the gain at under a
   tenth of the risk. This is the classic RNR result, reproduced exactly.
3. *DBR is conservative by construction.* Confidence grows only where the bot has been seen
   acting, so DBR starts at the equilibrium and approaches the best response with data
   (+0.48 at 3,000 hands).

**Then the results looked too good, so two post-hoc controls were added (and labeled so).**

* **The population model, 0 hands.** A best response to the prior-mean bot, with no
  observation at all, already gains **+0.512** [0.503, 0.522]. The 10-hand result was mostly
  the *population prior*: these bots share systematic weaknesses that knowledge of the
  population captures. The honest "value of watching this opponent" is the gain *above* the
  population model:

  | Hands observed | 10 | 30 | 100 | 300 | 3,000 | Oracle |
  |---|---|---|---|---|---|---|
  | Gain above the population model | +0.079 | +0.124 | +0.187 | +0.211 | +0.224 | +0.226 |

  The value is concentrated in archetypes far from the population average. Against nits,
  the population model gains only +0.14, while the oracle gains +0.66. This echoes Phase 1:
  the population and the archetype carry most of the information; individual detail adds the
  rest slowly.
* **A misspecified model.** A card-blind, HUD-style model uses three action frequencies that
  ignore hand strength, fit from the same hands. Its best response:
  * gains only +0.15 to +0.22;
  * *does not improve with data* (misspecification error does not shrink with $n$);
  * loses to plain equilibrium play against **33-44% of the bots**.

  Restricted Nash responses to the *same wrong model* gain +0.34 to +0.40 at $p = 0.5$-$0.75$,
  nearly twice the best response, against the true bot. On the frontier plot the card-blind
  curve bends *down*: trusting a wrong model more costs money. Robust responses are not just
  safer; under misspecification they also exploit better.

**Limits of these numbers.** The modeling is *correctly specified* (the five-parameter model is
the bots' own family), and the bots are far more exploitable than strong humans. Real
opponents are neither from a known family nor stationary. Treat these as upper bounds on what
modeling buys. The card-blind control shows the downside when the model family is wrong.

## Step 23: Contracts for small-game strategies, exact exploitability estimates, duplicate dealing (P2.5, P2.8, P2.9)

**What was built.**
* [`exploit/small_games.py`](../poker_lab/exploit/small_games.py): `PolicyExploitEngine`
  (best response, restricted Nash, data-biased) returning `PolicyExploitResult` with an
  *exact* `ExploitabilityEstimate`.
* `ExploitabilityEstimate` now has `value` and `unit`: mbb/hand for no-limit, chips/hand for
  the small games.
* [`engine/duplicate.py`](../poker_lab/engine/duplicate.py): `duplicate_match`, with
  `TableSimulator.play_hand(deck=...)` as the engine hook.

**Concepts.**

* **One strategy type per game family.** No-limit hold'em strategies answer one `SeatView` at
  a time (`StrategyProvider`). In a small game every solver returns a complete `TabularPolicy`,
  which can be evaluated exactly. Forcing one interface onto the other would need an action
  and observation abstraction that does not exist yet. That is Phase 3's job.
  Contract tests ([`tests/test_small_game_contracts.py`](../tests/test_small_game_contracts.py))
  hold every source (LP, CFR+, MCCFR, Deep CFR) and every policy exploit engine to the same
  rules:
  * the result is a valid policy on the given tree;
  * it is reproducible from its seed;
  * it beats uniform play;
  * engines report exact exploitability in chips per hand, and never do worse than the
    equilibrium against the model (for BR and RNR).
* **Duplicate dealing.** Deal $i$ is played at two tables with the same deck order and button.
  The players swap seats at the second table, so each holds each side of the deal once. Each
  player's two results share the card luck with opposite signs, so the pair's mean removes
  most of it.
  * *Agent isolation.* Agents at the two tables are separate instances (factories), so nothing
    learned in the first play can leak into the second.
  * *Separate randomness.* Each table has its own decision RNG, so only the cards, not the
    agents' coin flips, are mirrored.
  * *The decisive test:* two identical always-call agents. Every pot is then decided by the
    cards alone, so each pair cancels *exactly*. The duplicate interval has zero width while
    the naive one is over 10 bb/100 wide.

**Verification.** `tests/test_small_game_contracts.py` (18 tests) and
[`tests/test_duplicate.py`](../tests/test_duplicate.py) (3 tests): same cards in the same
seats at both tables, boards from one run-out, exact cancellation, deck validation.

## Step 24: Phase 2 wrap-up: one command and reproduction runs

**What was built.** [`experiments/run_phase2.py`](../experiments/run_phase2.py) runs the three
Phase 2 experiments (`cfr`, `deep_cfr`, `exploit`) from one master seed, like `run_phase1.py`
does for Phase 1. Two details matter on Windows:

* The experiment modules are imported inside `run()`, not at the top of the file. Worker
  processes are *spawned*: each one re-imports the main module, and a top-level import would
  make every worker load matplotlib, scikit-learn and PyTorch.
* Deep CFR is skipped with a message when PyTorch is missing
  (`importlib.util.find_spec("torch") is None`), so the base install still runs everything
  else.

**The reproduction protocol.** "Reproducible" is a claim to test, not to assume. After the
Phase 2 code was committed:

1. Check out a clean commit (here `348c4d5`) and run both phases from seed 2026.
2. Compare each new JSON file with its committed version, field by field. Skip the provenance
   block (`meta`) and wall-clock fields (any key mentioning seconds or time). A small script
   walks both trees and lists every other difference.
3. Expect zero differences. Any difference is either a bug (hidden global state, an unseeded
   RNG, an order-dependent reduction) or an undocumented change in behavior.

**Results.**

* Phase 1: zero differences, and identical summary tables and figures. The simulation
  fingerprint still changed (`7799e1e5ceac` to `dbec7cdce9d0`). It hashes the source files of
  the simulation packages, and Phase 2 edited two of them (the deck hook in
  `engine/simulator.py` and the line-ending fix in the bot writers). The fingerprint is
  conservative by design. It says "the simulation *may* have changed, so do not reuse the
  cache"; the rerun shows it did not. The rerun's outputs were committed, so the recorded
  fingerprint matches the current code.
* Phase 2: zero differences in all three experiments against their original runs at three
  different commits. The PNG figures were byte-identical too: matplotlib's PNG writer embeds no
  timestamp, so the same data and library versions give the same bytes.

**A provenance slip.** The Phase 2 rerun recorded `dirty: true`, because README.md was being
edited while it ran. Each step records the dirty flag when it starts (see Step 14), and the
README edits came first. The numbers were unaffected (README is not code), but the committed
originals have clean provenance, so they were kept. The rerun counts only as verification. The
rule: during a run whose provenance matters, edit only untracked or scratch files. Better
still, run it from a separate checkout (`git worktree add`).

## Step 25: The results explorer: one interactive page for every phase

**Why.** Static figures answer the questions the author thought to ask. An explorer lets
the reader ask their own. For example: how much does a restricted Nash response gain against
TAGs alone, after 30 hands, with a misspecified model? The user asked for it as a standing part
of the project: every phase adds sections as it produces numbers.

**What was built.** [`experiments/explorer/`](../experiments/explorer/) turns the committed
results JSON into one self-contained page, `experiments/results/explorer.html`, which opens
offline with a double-click.

```
results JSON --phase1.py / phase2.py--> section specs --build.py--> explorer.html
                                        (controls, views, charts)   (data + app.js + app.css)
```

* `spec.py` has constructors for four chart types (line, dot plot, heatmap, scatter), plus
  controls, sections and stat tiles.
* `phase1.py` and `phase2.py` each turn one phase's results into sections: 8 + 8 today.
  `build.py` assembles them. Planned phases appear in the navigation as empty pages.
* `static/app.js` is a dependency-free renderer of about 1,000 lines. It handles routing,
  controls, SVG charts, legends, tooltips and table views.

**Concepts.**

1. **Precompute views; keep the browser dumb.** A section has controls (say metric × model
   × archetype = 4 × 2 × 6). Every combination of control values is a *view* keyed by the
   values joined with `|`, built in Python. That is 48 small chart specs. The alternative,
   shipping raw data and aggregating in JavaScript, would duplicate the statistics code in a
   second language, where it would be untested. Precomputing keeps every number on the
   Python side, where the tests are. The cost is page size (825 KB here). Scatter plots of
   individual bots reference columns of a shared per-section table instead of repeating
   points in every view.
2. **Aggregate at the level of the independent unit.** Bots within one seed share a training
   population and a fitted prior, so they are not independent. Every curve is therefore a
   mean over bots *within* each seed, followed by a t-interval across the 5 seeds (the
   independent replicates). A ratio metric such as the share of the oracle's gain is a ratio
   of means within each seed, $\bar g_s / \bar g^{\text{oracle}}_s$, never a mean of
   per-bot ratios (those explode when a bot's oracle gain is near 0).
3. **Charts from first principles.** A scale maps data to pixels.
   * Linear: $p = r_0 + (v - d_0)/(d_1 - d_0)\,(r_1 - r_0)$.
   * Log: the same formula with $\log_{10} v$ in place of $v$.
   * *Nice ticks.* Divide the span by the desired tick count. Write the result as
     $e \cdot 10^k$ with $1 \le e < 10$, round $e$ to 1, 2, 5 or 10, and step from the first
     multiple of that size. Log axes tick at powers of ten (adding 2× and 5× when the range
     covers few decades).
   * One notation per axis. As soon as a log axis goes below 0.001, every tick is written
     $10^{k}$ or $m\times10^{k}$, so the reader never mixes "0.001" with "10⁻⁴".
4. **Interaction.**
   * When all series share x positions, a crosshair snaps to the nearest one and the tooltip
     lists every series there. Otherwise the tooltip shows the nearest point by squared
     pixel distance, as in sampled curves with different x grids.
   * Hit areas are larger than the marks: dots get a 12 px transparent circle, and scatter
     points answer within 30 px.
   * Arrow keys move the crosshair, and every chart has a table view, so no value depends on
     a mouse or on color.
   * Text from the data is inserted with `textContent`, never `innerHTML`.
5. **Render at true size.** An SVG drawn with `viewBox = 0 0 W H` is scaled to its box. If $W$
   is measured before the layout settles, all text shrinks with it. That happened here: the
   first charts were measured at full width, then placed in a half-width grid column, and
   their 11 px labels rendered at about 5 px. A `ResizeObserver` per figure redraws whenever
   its width really changes. That one mechanism also handles window resizes and scrollbars.
6. **Color.**
   * The palette is the validated one from the PNG figures (C.4), as CSS custom properties
     with separate light and dark values.
   * The validator passes both modes. It warns that three light-mode slots are below 3:1
     contrast against the background, and the required relief is the legends and table
     views.
   * Heatmaps use a single-hue ramp quantized to 8 steps, and the cell text switches between
     dark and light ink by step.
   * A color follows its entity everywhere. Archetypes always use the same slots, and also
     get distinct marker shapes in scatter plots, since only three categorical slots survive
     an all-pairs colorblindness check.
7. **Labels that stay out of the data.** A reference-line label goes at whichever end of
   the line covers fewer data points (points and segment midpoints counted inside the label's
   box). It also gets a halo, a stroke in the background color painted under the text with
   `paint-order: stroke`.
8. **Embedding data in HTML safely.** The JSON goes inside `<script type="application/json">`,
   with every `</` written as `<\/`. Otherwise a string containing `</script>` would end the
   element early. The three placeholders (CSS, JS, data) are substituted in *one* regex pass,
   so text inserted for one placeholder is never scanned for another.
9. **Staying current is enforced, not remembered.**
   [`tests/test_explorer.py`](../tests/test_explorer.py) fails in four cases:
   * a results file has no section reading it;
   * a control combination has no view;
   * a chart is malformed (non-finite values, non-positive values on a log axis, an interval
     that excludes its mean);
   * the committed page differs from a fresh build.

   The build is deterministic (no timestamps, no current commit) for that last check. The
   phase runners rebuild the page after full runs.
10. **Testing a UI without a UI.**
    [`browser_check.py`](../experiments/explorer/browser_check.py) runs headless Chrome or
    Edge on a copy of the page with a test script ([`smoke.js`](../experiments/explorer/static/smoke.js))
    injected.
    * The script visits all 321 views. On every chart it hovers, toggles a legend entry and
      opens the table.
    * It then appends a JSON report to the DOM, which `--dump-dom` prints.
    * `--virtual-time-budget` lets the page finish without real waiting.
    * Screenshots came from `--screenshot`, with `--blink-settings=preferredColorScheme=0|1`
      forcing the color scheme.
    * Chrome on Windows will not lay out narrower than about 500 px. Phone widths were
      therefore checked inside a 390 px `<iframe>`.
11. **Narrative numbers come from the data.** Each section's "What it shows" bullets are
    formatted from the results in the builder, not typed. Two claims first written as
    guesses were checked this way.
    * With the card-blind model, best-response losses do concentrate: 93% of TAGs and 83-87%
      of LAGs, but no calling stations.
    * Fold parameters are the slowest to learn.

    The check also surfaced a new fact: restricted Nash with p = 0.5 still loses against
    27-33% of nits.

**Verification.** Seven tests in `tests/test_explorer.py`. The browser check ran 321 views
and 555 charts with no errors: tooltips appeared on every hover except 22 scatter views with
no point near the chart's center, which is correct. Screenshots were reviewed in light, dark
and phone layouts.

## Step 26: A bridge to hold'em: heads-up push/fold solved exactly (P3.1)

**Why this game.** Phase 3 has to move the Phase 2 machinery (LP, CFR, best responses,
exploit engines) from toy games to hold'em. Heads-up push/fold is the smallest hold'em game
that is still real poker: it is played whenever stacks are short, and its solution
appears in published charts. That gives an independent check, the same role the
published Leduc game value played in Phase 2.

**The game.** Blinds are 0.5 and 1 bb, there is no ante, and the effective stack is $S$ bb
before posting.

* The small blind folds (losing 0.5) or goes all-in.
* Facing the all-in, the big blind folds (losing 1) or calls.
* After a call, the board is dealt and the better hand wins $2S$.

Two ideas make it tractable.

1. **Hand classes with exact card removal.** Suits only matter for flushes and for which
   cards are gone, so the 1,326 starting hands collapse into 169 classes (13 pairs with 6
   combos each, 78 suited with 4, 78 offsuit with 12). Dealing two hands picks a *pair of
   classes* $(i, j)$ with probability $n_{ij}/(1326 \cdot 1225)$. Here $n_{ij}$ counts the
   combo pairs that share no card, which is how card removal enters. AA vs AA has
   $n = 6$, while AA vs 72o has $6 \times 12 = 72$. The test recounts several $n_{ij}$ by
   brute force over all hands.
2. **No board in the tree.** Once both players are all-in, nobody acts again. The called
   all-in can therefore be replaced by its expectation $S(2e_{ij} - 1)$ for the small blind,
   where $e_{ij}$ is the class matchup equity (ties count half). The tree is one chance node
   with 28,561 class pairs, then two decisions. That makes 142,806 nodes and 338 information
   sets (169 per player), compiled by the Phase 2 `compile_tree` unchanged.

**The matchup table.** $e_{ij}$ is the average over compatible combo pairs (equally
likely given the classes) of the probability of winning over all boards. Exact enumeration
would take about $10^{10}$ hand evaluations. Instead, Monte Carlo runs 50,000 boards per class
pair, *stratified*: every compatible combo pair gets the same number of boards.

* Stratification removes the variance from which combos were dealt. The standard error is
  $\sqrt{\sum_p \hat\sigma_p^2/m}\,/\,k$ for $k$ strata of $m$ boards each, at most 0.0022 here.
* Every class pair gets its own child of one `SeedSequence`, so the table does not depend on
  how many worker processes ran (5 minutes on 16 threads).
* Symmetry gives the rest for free: $e_{ji} = 1 - e_{ij}$, and the diagonal is exactly 0.5.
* Cross-check: each class's equity against a random hand, $\sum_j n_{ij}e_{ij}/\sum_j n_{ij}$,
  agrees with Phase 1's independently computed table (different code, different seed) to
  0.0067 at worst.

**Solve once, rescale.** Only the called-all-in payoffs depend on $S$, and they are linear in
it. The tree is compiled once at $S = 1$, and every other stack multiplies those leaves by
$S$ (`dataclasses.replace` on the frozen tree). A test checks the result equals a direct
compile at $S = 7.3$. That makes an exact LP at 381 stacks (every 0.05 bb from 1 to 20 bb)
take under a minute.

**An equilibrium check that shares no code with the solver.** In a Nash equilibrium of this
game, every hand that pushes must gain by pushing rather than folding, given the opponent's
calls, and every hand that folds must not. Written out:

$$\Delta_{SB}(i) = \sum_j P(j \mid i)\,[c_j\,S(2e_{ij}-1) + (1 - c_j)] + 0.5,$$

$$\Delta_{BB}(j) = \frac{\sum_i n_{ij}\,p_i\,S(1 - 2e_{ij})}{\sum_i n_{ij}\,p_i} + 1,$$

where $p$ and $c$ are the push and call probabilities.
[`hand_gains`](../poker_lab/games/push_fold.py) computes both from the matchup table alone.
The test confirms that the LP's solution satisfies every one of these 338 conditions at three
stacks. CFR+ reaches the same push and call decision for all 169 hands at 5, 10 and 20 bb.

**Checking against a published chart.** HoldemResources publishes, for each hand, the
largest stack at which Nash pushes (or calls). The values were read from the page
automatically and stored with source and date in
[`experiments/references/hune_no_ante.json`](../experiments/references/hune_no_ante.json).
Our sweep gives the same quantity: the stacks at which a hand plays with probability at
least 0.5.

* Calls: all 169 hands agree to within 0.1 bb, the chart's own resolution.
* Pushes: 151 hands agree within 0.1 bb and 158 within 0.25 bb.
* The remaining disagreements were priced in expected value (*post hoc*). At every stack
  where the charts differ, the hand gains at most 0.014 bb by pushing rather than folding,
  or the reverse. Following the published chart everywhere it differs costs at most 0.21
  mbb per hand.

These are hands that are nearly indifferent, and Monte Carlo noise of 0.002 in $e_{ij}$ is
enough to flip them. Non-monotonic hands such as 63s and 43s appear in both solutions. They
push at very short stacks and again at medium stacks, but not in between.

**Transcription errors are data errors.** The first automated read of the chart was wrong in
five cells. Pusher K9o and Q9o were not 20+ in that read. The read had also attached the
10%-ante footnotes to the no-ante caller chart, which made three small suited hands
"call up to 15 bb". The tell was that the solver disagreed by 2 to 12 bb exactly there, on
hands whose neighbors agreed to 0.1 bb. A second read, focused on those cells, fixed them.
The corrections are logged in the file. The rule: before treating a disagreement with a
reference as a finding, re-check the reference value itself.

**Simple rules, priced.** Exploitability (a best response's gain) of push-the-top-x%
rules, ranking hands by equity against a random hand:

* With Nash's own range *sizes* at each stack, the ranking costs at most 0.013 bb per hand.
  Hand order is nearly right; what matters is how many hands.
* Fixed sizes, such as "push 20%, call 10%" at every stack, cost up to 0.56 bb per hand.
  Each fixed rule is least bad at the stack its sizes happen to fit.

**Verification.** `tests/test_push_fold.py` has 20 tests:
* card-removal counts against brute force;
* published equities (AA vs KK 0.819, KK vs AKs 0.661, AKo vs QQ 0.432, AA vs 72o 0.88);
* agreement with the Phase 1 equity table;
* rescaled tree equals a direct compile;
* exact LP, CFR+, the per-hand equilibrium conditions;
* ranges tighten from 2 to 16 bb;
* four published thresholds bracketed within ±0.3 bb.

Experiment at commit `3aa83c2` (clean): `python -m experiments.run_phase3 --seed 2026`.

## Step 27: Modeling and exploiting push/fold opponents (P3.2)

**The question.** Phase 2 found that in Leduc most of the exploitation gain came from the
population prior, before watching the individual at all. Does that hold in a hold'em game,
and how much do the cards you happen to see at showdown add over plain action frequencies?

**Bots with interpretable ranges.** A push/fold bot plays hand class $i$ with probability

$$P(\text{push } i) = \sigma\!\left(\frac{w_{push} - \text{pct}_i}{T}\right), \qquad
P(\text{call } i) = \sigma\!\left(\frac{w_{call} - \text{pct}_i}{T}\right),$$

where $\sigma$ is the logistic function.

* $\text{pct}_i$ is the hand's strength percentile: the share of the 1,326 combos ranked above
  it, plus half its own.
* The *width* $w$ is roughly the share of hands played.
* The *temperature* $T$ softens the edge of the range.

Archetypes are distributions over $(w_{push}, w_{call}, T)$: Beta for the widths (concentration
30, as before) and log-normal for $T$. The archetypes sit on both sides of the 10 bb
equilibrium (push about 58%, call about 37%):

| Archetype | Push width | Call width |
|---|---|---|
| Nit | 0.20 | 0.10 |
| TAG | 0.40 | 0.25 |
| LAG | 0.65 | 0.40 |
| Calling station | 0.30 | 0.70 |
| Maniac | 0.90 | 0.55 |

**What the hero sees.** The hero plays the equilibrium while watching. It sees its own class,
the bot's actions, and the bot's class *only when a push is called*: a fold never shows
cards. The simulation deals class pairs with the game's own probabilities, so observation and
exact evaluation share one model of the deal. A test scrambles every hidden class and checks
that the observations do not change, the same firewall test used for the real store.

**Two models of the same hands.**

1. *Likelihood.* Shown hands contribute $\log P(a \mid i)$. Hidden hands contribute the
   marginal over the classes the bot could hold given the hero's own class:

   $$\log \sum_i P(i \mid h)\,P(a \mid i), \qquad P(i \mid h) = \frac{n_{hi}}{\sum_k n_{hk}}.$$

   This is the card-removal weight from Step 26. The fit is the posterior mode over
   (logit $w_{push}$, logit $w_{call}$, $\log T$), with Beta priors on the widths (mode in logit
   space, as in Step 21) and a normal prior on $\log T$, fitted to a separate training
   population. Because the data enter only through counts, the whole likelihood is six dot
   products. A test checks it against a hand-by-hand sum.
2. *Frequency* (the HUD model). It takes only the push rate as small blind and the call rate
   facing a push, each shrunk toward the population with 10 pseudo-hands. It then solves for
   the widths that reproduce them at the population's temperature (Brent's method). It never
   looks at a shown card.

**Responses and evaluation.** Responses come from the Phase 2 engines, unchanged, on the
push/fold tree:

* best response;
* restricted Nash with $p \in \{0.5, 0.9\}$;
* data-biased response, with confidence $n/(n + 10)$ per seat.

Gain is exact: the value against the true bot minus the equilibrium's, over both seats. The
design follows Phase 2:

* 5 seeds × 30 held-out bots;
* 10 to 3,000 hands observed;
* oracle and population-model references.

The pre-registered control is a population that ranks hands "pairs first", so both models
are wrong about *which* hands a bot plays.

**Results** (commit `a8ea075`):

* *The population prior is worthless here.* A best response to the average bot gains
  −0.007 bb/hand: it loses to equilibrium play. The archetypes straddle the equilibrium, so
  their average is near it, and a best response to a near-equilibrium model is a poor guess
  against any individual. This is the opposite of Leduc, where every archetype was
  exploitable in the same direction. Watching the individual is what pays: with the
  likelihood model, the best response reaches 58% of the oracle's gain after 10 hands, 80%
  after 30 and 97% after 100.
* *Shown cards help early.* After 10 hands the likelihood model's push-width error is 0.106
  against 0.157 for the frequency model, and its best response gains +0.051 against +0.038
  bb/hand. From about 300 hands the two are equal. Frequencies pin down the range sizes, and
  the sizes are what matters.
* *Hedging pays while the model is uncertain.* After 10 hands the best response loses to
  equilibrium play against 26% of bots. Restricted Nash with p = 0.9 gains as much (+0.052)
  at 55% of the exploitability.
* *The misspecification control turned out mild.* Against pairs-first bots, the best response
  still gets 99% of the oracle's gain by 3,000 hands. Before reporting that as robustness,
  the overlap was measured: the two orders have rank correlation 0.95, and at equal width
  their ranges differ by only 4–12% of combos. So this result supports "get the range size
  right", not "misspecification is harmless". A harsher test belongs with the model-fit
  checks (P3.5).

**Verification.** `tests/test_push_fold_bots.py` has 9 tests:

* percentiles form a CDF in both orders;
* a sharp threshold plays its width;
* hidden classes never reach the observations;
* the simulated push rate matches the strategy;
* the likelihood equals a hand-by-hand sum;
* zero hands give the population;
* recovery from 3,000 hands for both models.

Experiment: `python -m experiments.run_phase3 --seed 2026 --only exploit` (4 minutes).

---

## Step 28: Playing the solved game in real hands, behind the firewall (P3.3)

**The question.** Steps 26 and 27 work inside the abstract game: 169 hand classes, two
actions per player, payoffs from a matchup table. A real hand is a `SeatView`: seats, stacks,
blinds, an action log and two specific cards. Can the solved strategy play real hands, and how
would we know it plays them *correctly*?

**From a view to an information set.** `PushFoldStrategy` (`poker_lab/strategy/push_fold.py`)
reads only the view. It accepts a spot when all of these hold:

* the street is preflop, the blinds are in the game's 1 : 2 ratio and there is no ante or
  straddle;
* every seat other than the two blinds has folded (heads-up there are none);
* the hero is the small blind making its first decision, or the big blind facing a
  small-blind raise that puts it all-in, with nothing in between;
* the effective stack is in (1, 20] bb.

The *effective stack* is the smaller of the two blinds' stacks at the start of the hand,
$S = \min(s_{SB}, s_{BB}) / \text{bb}$: chips beyond the shorter stack can be neither won nor
lost, so a 6 bb player against a 20 bb player plays the 6 bb game. The information set is then
(role, class of the two cards), and the provider returns the policy's probabilities for it. A
push is a raise to the most the hero can bet; any excess over the opponent's stack comes back
as an uncalled bet. The equilibrium for each stack comes from the LP of Step 26 (about 0.1 s,
cached per stack). LP round-off such as $10^{-12}$ is snapped to 0 or 1, so pure choices stay
pure.

**Off the tree.** A limp, a min-raise, a 25 bb stack, a table with antes: the abstract game has
no information set for these. The provider hands such views to a `fallback` provider, or
raises `OffTree` when it has none. It does not *translate* off-tree actions onto the tree. In
large games, an opponent's bet size that is not in the abstraction is mapped onto the
neighboring sizes that are (the pseudo-harmonic mapping of Ganzfried and Sandholm, 2013). Here
the abstraction has a single bet size, all-in, so there is nothing to interpolate between: a
min-raise read as a push would just be a different, unchecked game.

Blind against blind after the other seats fold is accepted as an approximation. The abstract
game deals the two hands from a full deck, but at a real table the folded players hold cards
too, and their folds make those cards weaker on average. Heads-up tables are exact.

**The firewall, restated as a property.** The provider is a function of the view alone:
$a = f(\text{view})$. The hand-history store builds a view from public state and the hero's own
cards. So two live hands that differ only in the opponent's sealed cards produce *equal*
views, and any $f$ must give them the same advice. The new test builds three such hands
(opponent AA, 72o, T9o), checks that the views are equal and the advice identical, and walks
the advisor's object graph: no card string is reachable from it, since the cached policies are
arrays of probabilities. An end-to-end test plays 80 heads-up hands with the hero behind
`AdvisorAgent`.

**The bridge check.** The strongest test of a mapping is to use it. If the provider maps real
hands correctly, then playing real hands must win, on average, exactly what the abstract game
says: the head-to-head value of the two policies. Anything wrong makes the two disagree:

* a wrong class (suits swapped, ranks transposed);
* a wrong role, or the wrong blinds;
* card removal handled differently from the real deck;
* chip accounting for all-ins and uncalled bets.

The experiment (`experiments/phase3_bridge.py`) plays heads-up hands through the real engine:
PokerKit's rules, phevaluator showdowns and the sealed store. Blinds are 50/100, both stacks
reset to $S$ bb every hand, and the hero plays behind `AdvisorAgent`. It covers:

* stacks of 5, 10 and 20 bb;
* the hero playing the equilibrium, or the exact best response to the opponent's true ranges;
* opponents: the five archetypes of Step 27 at their mean parameters, and the equilibrium;
* 100,000 duplicate deals per matchup (200,000 hands), 6.6 million hands in all.

**Four estimators of one win rate.** Card luck dominates poker results: at 10 bb the result
of a single hand has a standard deviation of 2.6 to 6.1 bb (more against opponents whose
all-ins get called more often), while the effects being measured are hundredths of a big
blind. The experiment compares four estimators on the same hands.

1. *Naive*: the plain mean over hands at one table.
2. *Duplicate* (Step 23): each deal is played twice with seats swapped, and the unit is the
   mean of the two plays. If $X_1, X_2$ have variance $\sigma^2$ and correlation $\rho$,
   $$\operatorname{Var}\!\left(\tfrac{X_1 + X_2}{2}\right) = \frac{\sigma^2 (1 + \rho)}{2}.$$
   A pair costs two hands, so per hand the variance is $\sigma^2 (1 + \rho)$ against
   $\sigma^2$ for naive. The more the two plays mirror each other ($\rho \to -1$), the more
   luck cancels.
3. *All-in EV*: a called all-in for $c$ chips each is scored by its expected value
   $c\,(2e_{ij} - 1)$ instead of the realized $c\,(2W - 1)$, where $W \in \{0, \tfrac12, 1\}$
   is the showdown result and $e_{ij}$ the class matchup equity. This is a *control variate*:
   the estimator subtracts $X - Y$, whose expectation is zero. Given the two specific combos,
   $E[W] = e_{\text{combo}}$. Given the classes, the compatible combo pairs are equally likely,
   and $e_{ij}$ is exactly their average, so $E[Y] = E[X]$: no bias, and the luck of the board
   (and of the combo within the class) is gone. Poker tracking software reports the same
   quantity as "all-in adjusted winnings".
4. *Both together*.

Each estimate is compared with the exact value by
$z = (\hat\theta - \theta) / \widehat{SE}$. The pre-registered rule: if the bridge is right,
the $z$ behave like standard normal draws, so about 5% exceed 1.96 and none exceeds 4, and a
t-test finds the mean $z$ consistent with zero ($p \ge 0.01$). Every hand must also end
preflop: a postflop decision would mean a spot left the tree. Each comparison's *minimum
detectable effect* is $2.8\,\widehat{SE}$, the discrepancy it would catch with 80% power at a
two-sided 5% level ($1.96 + 0.84 = 2.8$).

Two of the estimators score all-ins with the same matchup table that defines the exact game,
so they test the mapping and the accounting but not the table. The naive and duplicate
estimators use the boards that actually came, so they test the table against real dealing as
well, with wider intervals.

**An amendment, made before any result was seen.** The first full run crashed after 20
minutes, in aggregation, dividing by a standard error of exactly zero. At 20 bb the
equilibrium has no mixed hands. When both players use it, the second play of every deal
mirrors the first: the hero holds the opponent's cards in the opponent's seat and makes the
opponent's choices, so every pair sums to zero. The estimate is exact, and $z$ is $0/0$. The
rule was amended: an estimate whose standard deviation is below $10^{-9}$ bb has no $z$; it
passes only if it equals the exact value, and it is left out of the $z$ statistics. The
quick configuration now includes 20 bb, so the smoke test covers the case.

**Engineering: a profile, then a one-line fix.** The first pilot ran at 94 hands/s per
process, against about 1,000 for the bare engine. `cProfile` put 72% of the time in
`ProfileStore.snapshot`: before every decision the advisor's profile store recomputed each
profile's 95% credible intervals through `scipy.stats.beta.ppf`, whose argument checking
costs about 86 µs per call. `scipy.special.betaincinv` evaluates the same function (the
inverse regularized incomplete beta) in 3.4 µs, and gave bit-identical results on 400,000
random $(a, b)$ pairs. Measure before optimizing: the push/fold code itself was never the
bottleneck.

**Results** (commit `9bdab59`, 6.6 million hands, 20 minutes on 12 processes):

* *The bridge holds.* All 6.6 million hands ended preflop. Of the 130 z-scores, 6 exceed 1.96
  (4.6%, against about 5% expected) and the largest is 2.57. Per estimator the mean $z$ lies
  between −0.10 and +0.14 (t-test $p \ge 0.38$). The two zero-variance estimates (Nash against
  Nash at 20 bb, duplicate with and without all-in EV) equal the exact value exactly.
* *The check is sensitive.* The smallest discrepancy each matchup against a bot would detect
  (80% power) is 0.013–0.099 bb/hand for the naive estimator and 0.003–0.015 bb/hand for
  duplicate + all-in EV. For scale: the gains measured in Step 27 are 0.05–0.09 bb/hand.
* *Variance reduction, measured.* For a ±0.05 bb/hand 95% interval at 10 bb against the bots,
  the median number of hands needed is 32,073 naive, 10,733 with duplicate dealing, 4,339 with
  all-in EV and 1,572 with both: 20× fewer. Needs grow with the stack (naive: 11,627 at 5 bb,
  91,551 at 20 bb), since a called all-in risks more.
* *Duplicate dealing helps most against opponents who play like the hero.* At 10 bb Nash needs
  5.2× fewer hands against the LAG (ranges close to Nash's) but only 1.4× fewer against the
  nit. The closer the strategies, the closer $\rho$ gets to −1. Nash against Nash is the
  limit: 91 hands with duplicate dealing against 32,302 naive at 10 bb, and exactly zero at
  20 bb.

These numbers plan Phase 5: a live home game will not reach tens of thousands of hands, so a
live win-rate claim needs variance reduction. Duplicate dealing is impossible at a real table,
and all-in EV covers only all-ins, which leaves AIVAT-style control variates over every chance
node.

**Verification.**

* `tests/test_push_fold_strategy.py` (11 tests): the equilibrium is exact (exploitability
  < $10^{-7}$) and solved once per stack; in engine-played hands every view maps to the right
  role, class and stack, with the right probabilities and all-in sizes; uneven stacks use the
  effective stack; fixed ranges; off-tree cases (limp, min-raise, 25 bb and 1 bb stacks,
  antes, odd blinds, postflop) fall back or raise; blind against blind after folds.
* The strategy contract suite now includes the provider (100 bb deep, so always the
  fallback).
* `tests/test_advisor_firewall.py`: identical views and advice whatever the opponent's sealed
  cards; no card reachable from the advisor; an end-to-end `AdvisorAgent` match.
* Experiment: `python -m experiments.run_phase3 --seed 2026 --only bridge` (20 minutes);
  smoke test with `--quick` (20 s).

## Step 29: Ranked exploit recommendations with confidence (P3.4)

**The question.** Step 27 measured how much a best response to a fitted model gains, on
average, over many bots. A player facing one opponent needs something else: a recommendation
for *this* opponent, with an honest statement of how sure it is. After 10 hands a best
response might gain 0.1 bb/hand or lose 0.03; after 1,000 hands the same advice is nearly
certain. The recommender should say which, choose accordingly, and be checked against the
truth.

**From hand histories to a profile.** `push_fold_observations`
(`poker_lab/profiling/push_fold.py`) turns completed hand histories into the sufficient
statistics of Step 27's likelihood: the opponent's push or fold as small blind, its call or fold
facing the hero's push as big blind, its class when its cards were shown, and otherwise the
hero's own class (which sets the card-removal term). Two firewall rules: an in-progress hand
raises `IncompleteHandError`, and every hand is first redacted to the hero's view, so an
opponent's unshown cards cannot reach the model even if the caller passes full simulator
records. A test forges every unshown opponent card and gets identical observations. Hands
the push/fold game does not describe (a limp, a different stack, a pot the opponent and the
hero are not the blinds of, a duplicate hand id) are skipped and counted by reason. The table
rules are shared with Step 28's live mapping (`blind_pot`), so "a push/fold hand" means the
same thing to the advisor and to the profiler.

**A full posterior, not just a mode.** Step 27 fitted the mode of the posterior over a bot's
parameters $\theta$ = (push width, call width, temperature). A statement like "this response
gains with 95% probability" needs the whole posterior
$p(\theta \mid D) \propto p(D \mid \theta)\,p(\theta)$. With three parameters, two cheap tools
are enough.

1. *Laplace approximation.* Work in $z = (\operatorname{logit} w_p, \operatorname{logit} w_c,
   \log T)$, where the posterior is smooth and unbounded. Near its mode $\hat z$, the negative
   log posterior is about quadratic, $-\log p(z \mid D) \approx c + \tfrac12 (z - \hat z)^\top
   H (z - \hat z)$, so the posterior is about normal with covariance $H^{-1}$. $H$ is the
   Hessian at the mode, here by central differences (19 evaluations, step $5 \times
   10^{-3}$).
2. *Importance sampling* corrects the approximation. Draw $z_k$ from a proposal $q$ (a
   Student-t with 4 degrees of freedom, centered at $\hat z$ with scale $H^{-1}$) and weight
   each draw by $w_k \propto p(D \mid z_k)\,p(z_k) / q(z_k)$, normalized to sum to one.
   Weighted averages $\sum_k w_k f(z_k)$ converge to posterior expectations whatever the
   proposal, as long as it covers the posterior. Its tails must be at least as heavy as the
   posterior's, or a few draws get huge weights; that is why a t is used rather than a normal.
   Normalizing the weights cancels the unknown constant in $p(D)$.

How good is the weighted sample? The Kish *effective sample size*
$\text{ESS} = 1 / \sum_k w_k^2$ equals the number of draws when all weights are equal and 1
when one draw has all the weight. Over this step's 2,100 fits it ranged from 5 to 1,795 of
2,000 draws (median 1,602). A brute-force check integrates the posterior on a
$41^3$ grid: the importance-sampling means agree within five Monte Carlo standard errors.
Weighted quantiles (the smallest value whose cumulative weight reaches $q$) give the credible
intervals.

**Scoring a strategy against every draw, exactly.** The recommender needs the gain of each
candidate against each of 2,000 draws: 14,000 head-to-head values per recommendation. On the
143,000-node tree that would take minutes. In closed form it takes milliseconds. Let
$P_{ij}$ be the probability of dealing class $i$ to the small blind and $j$ to the big
blind, $r_i = \sum_j P_{ij}$, and $C_{ij} = S(2e_{ij} - 1)$ the small blind's payoff when
called. Write $s$ for a small blind's per-class push probabilities and $c$ for a big blind's
call probabilities. Then the small blind's expected value is

$$u(s, c) = \sum_{ij} P_{ij}\big[(1 - s_i)(-\tfrac12) + s_i(1 - c_j) + s_i c_j C_{ij}\big]
= r \cdot (1.5\,s - 0.5) + s^\top D\, c, \qquad D = P \circ (C - 1),$$

and the hero's value against an opponent, both seats averaged, is
$\tfrac12[u(s_h, c_o) - u(s_o, c_h)]$: a bilinear form, so a batch of hero strategies against a
batch of draws is two matrix products. Exploitability has a closed form too. A best-responding
opponent calls the hero's pushes with class $j$ exactly when $(s^\top D)_j < 0$, and pushes
class $i$ against the hero's calls when $1.5 r_i + (D c)_i > 0$, so

$$\text{expl}(s, c) = \tfrac12\Big[-r \cdot (1.5 s - 0.5) - \sum_j \min\big(0, (s^\top D)_j\big)
+ \sum_i \big(\max(0, 1.5 r_i + (Dc)_i) - 0.5\,r_i\big)\Big].$$

Both equal the tree computations of Step 18 to $10^{-14}$ in the tests.

**The candidates, and the Bayes action.** The opponent acts at most once per hand, so the
hero's value is *linear* in the opponent's per-class probabilities ($c$ appears once in
$s^\top D c$). The posterior expected value of any hero strategy is therefore its value
against the posterior *mean* strategy $\bar s = \sum_k w_k s(\theta_k)$, $\bar c$. It
follows that the best response to $(\bar s, \bar c)$ maximizes the posterior expected gain
over *all* strategies: it is the Bayes action. (Step 27's best response to the posterior
*mode* is not; the mode of $\theta$ does not give the mean of $s(\theta)$, since $s$ is
nonlinear.) The candidates are the equilibrium, restricted Nash responses (Step 22) for
$p \in \{0.1, 0.25, 0.5, 0.75, 0.9\}$ against $(\bar s, \bar c)$, and that best response
($p = 1$). For each candidate $m$ and draw $k$ the gain is
$G_{mk} = v(m, \theta_k) - v(\text{eq}, \theta_k)$, which gives a posterior mean, a 90%
credible interval and $P(G > 0)$. Exploitability is exact, and so is the gain over the
*population recommendation* (the top pick with no hands seen): what watching this opponent
is worth.

**Ranking by a lower confidence bound.** Candidates are ranked by the lower end of their 90%
interval, the 5% posterior quantile $L_m$, with ties going to the smaller $p$. The
equilibrium's gain is exactly zero, so $L_{\text{eq}} = 0$. A deviation tops the list only if
$L_m > 0$, which means less than 5% of the posterior weight sits at or below zero: the
recommendation deviates only when it gains with posterior probability above 95%. This is how
$p$ is "set by evidence": with ten hands only small, safe deviations clear the bar; with
hundreds, the best response does. A maximin-flavored rule like this suits a player who
cannot afford a losing adjustment. A player who only cares about the long-run average should
take the Bayes best response, and the experiment prices the difference.

Does choosing the best of seven by their intervals spoil the chosen interval (a *winner's
curse*)? Not if the prior is right. A posterior probability is conditional on the data, and
the choice is a function of the data, so $P(G_m < L_m \mid D) \le 5\%$ holds for whichever
$m$ the data selected. With a misspecified prior the argument fails, and the experiment
measures by how much.

**The report.** `RecommendationReport.explanation()` writes what a player would read:

> Seen 100 hands. As small blind it pushed 34 of 50; facing a push it called 14 of 26; its
> cards were shown 24 times. Estimated ranges: pushes 66% of hands (90% interval 58% to 74%)
> and calls 50% (37% to 64%); the equilibrium at 10 bb pushes 58% and calls 37%. Recommended:
> the restricted Nash response (p = 0.5). As small blind, push 53% of hands (equilibrium 58%):
> adds Q6o; drops J4s, J3s, T5s, T4s and 10 more. As big blind, call 40% of hands
> (equilibrium 37%): adds Q8o, Q6s, K4o. Expected gain over the equilibrium +0.012 bb/hand
> (90% interval +0.002 to +0.024; probability of gaining 97%), +0.011 over the population
> recommendation. Exploitability 0.004 bb/hand.

(A LAG bot from the experiment; its true gain was +0.016.)

**The experiment** (`experiments/phase3_recommend.py`, pre-registered at commit `a433bc5`
before the first run) reuses Step 27's game, seeds and held-out bots: 5 seeds × 30 bots, 10 bb,
the standard population and the pairs-first (misspecified) one, the same hands. Step 27's
sampling and observation code was factored into shared functions, and its quick
configuration's output stayed byte-identical. The best response to the posterior mode also
reproduces Step 27's per-bot gains: 2,100 values, largest difference $1.5 \times 10^{-13}$.
For every bot, sample size (0 to 3,000 hands) and candidate it records the posterior summary
and the exact true gain against the bot's real strategy. Five ways of choosing are compared:
the lower-bound pick, the Bayes best response, the mode best response, a fixed $p = 0.5$
response, and the best candidate in hindsight (not achievable).

**Results** (standard bots; gains in bb/hand; the oracle best response gains +0.088):

| Hands | Lower-bound pick | Bayes BR | Mode BR | Pick loses to eq. | Bayes BR loses | 90% coverage |
|---|---|---|---|---|---|---|
| 0 | +0.000 | +0.005 | −0.007 | 0.0% | 40.0% | 96.1% |
| 10 | +0.014 | +0.058 | +0.051 | 2.7% | 24.7% | 88.7% |
| 30 | +0.048 | +0.073 | +0.070 | 2.0% | 18.7% | 87.9% |
| 100 | +0.082 | +0.086 | +0.085 | 0.0% | 4.7% | 91.7% |
| 1,000 | +0.087 | +0.088 | +0.088 | 0.7% | 0.7% | 93.7% |
| 3,000 | +0.088 | +0.088 | +0.088 | 0.0% | 0.0% | 92.3% |

* *The ranking keeps its promise.* The pre-registered prediction was that the top pick loses
  to the equilibrium against at most 5% of standard bots at every sample size. The worst is
  2.7% (4 of 150 after 10 hands), and no loss exceeds 0.031 bb/hand. When the pick deviates,
  the deviation really wins 94-100% of the time. The Bayes best response loses to a quarter
  of the bots after 10 hands, and the mode best response to 26%.
* *Safety costs gain while evidence is thin.* After 10 hands the pick earns 16% of the
  oracle's gain against 65% for the Bayes best response. After 30 hands it is 55% against 83%,
  after 100 94% against 97%, and from 300 hands they are within about 0.001 bb/hand. The evidence moves $p$: the
  equilibrium is the top pick for 52% of bots after 10 hands, 23% after 30 and none after
  1,000, and the full best response for 1%, 15% and 59%.
* *The intervals are about right.* The prediction was 85-95% coverage of the 90% intervals
  at every sample size. It holds from 10 hands on (88-94%) and misses at 0 hands: 96%,
  slightly too wide, because the prior is a product of independent Betas fitted by moments to
  a mixture of five archetypes and is broader than the real population. The true gain is
  above the interval's lower end 93-96% of the time (nominal 95%).
* *A small winner's curse.* For the top pick alone the lower end holds only 88% of the time
  after 10 hands (nominal 95%): choosing the largest bound favors bounds that came out too
  high, which the approximate prior does not fully prevent. It washes out by 30 hands
  (94-96% from then on).
* *Bayes beats mode while evidence is thin.* The best response to the posterior mean gains
  +0.013 bb/hand more than the best response to the mode with no hands, and +0.006 [+0.001,
  +0.011] after 10. From 100 hands the difference is below 0.001.
* *Misspecified bots: right decisions, wrong intervals.* Against pairs-first bots, 90%
  coverage *falls* as evidence grows, from 92% with no hands to 55% after 3,000. The model
  ranks hands by equity, so with more data the posterior concentrates more tightly on a
  slightly wrong answer. The decisions survive because the error is small next to the gains:
  the pick loses to equilibrium against at most 5.3% of these bots (the population
  recommendation at 0 hands), at most 1.3% from 100 hands on, and earns 98% of the oracle's
  gain by 3,000 hands. Intervals that shrink around the wrong value are the signature of
  misspecification, and the reason P3.5 checks a model's fit before trusting it.
* *Diagnostics.* Only 3 of 2,100 posterior fits had an ESS below 100, all against pairs-first
  bots. In the worst (ESS 5, a pairs-first nit after 30 hands) the recommender chose the
  equilibrium. With no hands, the $p = 0.1$ response's lower bound lies within 0.0002 bb/hand of
  zero, so Monte Carlo noise decides between it and the equilibrium. That is why the
  population recommendation differs between bots of the same seed (each bot has its own
  random stream).

**Engineering.** A recommendation takes about 0.55 s, almost all of it in the ten restricted
Nash LPs (five values of $p$, two seats). Posterior draws and closed-form scoring take about
60 ms. The full experiment (2,100 recommendations) ran in 164 s on 12 processes.

**Verification.**

* `tests/test_recommend.py` (15 tests): closed-form values and exploitability equal the tree
  (two stacks); the vectorized log posterior equals the scalar likelihood plus prior; weighted
  quantiles equal NumPy's inverted-CDF quantiles with equal weights; with no hands the
  posterior is the prior; importance sampling matches grid integration; the report is ranked by
  the lower bound, with the equilibrium at exactly zero, and a deviating pick has
  $P(G > 0) > 95\%$; the best response to the posterior mean has the highest posterior mean
  gain, and along the restricted Nash family both gain and exploitability grow with $p$; the
  closed-form best response matches the tree's; reproducibility per seed; strong evidence
  recovers over 95% of the oracle's gain; settings are validated.
* `tests/test_push_fold_profile.py` (6 tests): observations match an independent count from
  the full records; forged unshown cards change nothing; in-progress hands are refused;
  skipped hands are counted by reason (duplicate, off the tree, stack, not the blinds,
  including three-handed tables); end to end, from completed hands to observations, a ranked
  recommendation and advised play through `AdvisorAgent`.
* Experiment: `python -m experiments.run_phase3 --seed 2026 --only recommend` (3 minutes; it
  reads Step 27's results for the cross-check); smoke test with `--quick`.

## Step 30: Checking the model before trusting it (P3.5)

**The question.** Step 29's intervals were honest against bots the model describes, but against
pairs-first bots their coverage fell to 55% by 3,000 hands: with more data the posterior
concentrated, ever more confidently, on a slightly wrong answer. A Bayesian posterior answers
"which parameter values fit?", never "does the model fit at all?". That needs a separate
check, run before the recommendations are trusted.

**Two harsher ways to be wrong.** Pairs-first bots turned out to be a mild test, since their
hand order has rank correlation 0.95 with equity. Two new populations push harder.

* *Connectors-first* bots score a hand by pairs, suitedness and connectedness before high cards
  ($4\cdot\text{pair} + 2\cdot\text{suited} + \text{connectedness} + \text{high}/14 +
  \text{low}/28$, with connectedness 2, 1 or 0.5 for gaps of 0, 1 or 2). 54s then ranks above
  KQo and A9o sits near the median. Rank correlation with equity is 0.48, and at a 40% width only
  half of the bot's combos are ones the model expects.
* *Gear-change* bots play one archetype and then switch to another at a random hand in the
  middle half of the session. The model assumes one stationary strategy. The recommendation is
  judged against the second gear, the one the bot plays from now on.

**Data in order.** The checks need the opponent's decisions in the order they were made, which
the sufficient statistics of Step 27 throw away. An `ActionLog` keeps, for each decision, the
seat, the action, the hero's own class and the bot's class when shown. The hand-history
profiler now builds one (hands in the order given), and its `Observations` are derived from it,
so the likelihood and the checks always see the same data.

**Posterior predictive checks.** Choose a discrepancy $T(y, \theta)$ that measures how badly
data $y$ fit parameters $\theta$. Draw $\theta_r$ from the posterior and a replicate $y_r$ from
the model with $\theta_r$, keeping fixed what the bot did not choose (which decisions it faced,
the hero's own class each time). The posterior predictive p-value is

$$p = \Pr\big(T(y_r, \theta_r) \ge T(y, \theta_r) \mid y\big),$$

estimated from 400 draws. A small $p$ says the real data are more extreme than the fitted model
produces (Rubin 1984; Gelman, Meng and Stern 1996). Two discrepancies, one per assumption.

1. *Shown hands (the hand order).* At a showdown the bot's class $i$ is revealed. Given the
   hero's own class $h$ and that the bot pushed, the model gives
   $$q(i \mid h, \theta) = \frac{P(i \mid h)\, s_\theta(i)}{\sum_{i'} P(i' \mid h)\, s_\theta(i')},$$
   card removal times the bot's range ($c_\theta$ for calls). Conditioning on $h$ is exact and
   needs no model of the hero. The hero's decision to call depends only on $h$, so given $h$ it
   carries no information about $i$: the check is valid whatever the hero plays. The classes go
   into bins (pair, suited, offsuit × equity quartile; 10 are non-empty, since every pair is in
   the top half), and $T$ is Pearson's chi-square $\sum_b (O_b - E_b)^2 / E_b$, with expected
   counts $E_b = \sum_{\text{shown}} q(b \mid h_t, \theta)$, summed over both seats. A bot that
   shows 54s and 76s where the model expects A9o and KTo fills the wrong bins.
2. *Stability (stationarity).* For each decision, the residual $x_t - p_t$ compares the action
   (1 aggressive, 0 fold) with the model's probability given only the hero's class,
   $p_t = \sum_i P(i \mid h_t)\, s_\theta(i)$. Under the model the partial sums
   $S_k = \sum_{t \le k}(x_t - p_t)$ wander like a random walk. A gear change makes them drift
   one way and then the other. $T = \max_k |S_k| / \sqrt{\sum_t p_t(1 - p_t)}$, a CUSUM
   (Page 1954), taken per seat, and the larger of the two is used.

Ties are frequent with few decisions (many replicates give the same small CUSUM), so the
p-values are *mid-p*: ties count half (Lancaster 1961).

**Calibration: why the threshold is not 0.05.** Posterior predictive p-values are conservative.
The posterior was fitted to the same data, so the replicates resemble the data more than fresh
data would, and under a correct model the p-values pile up toward 0.5 instead of spreading
uniformly (Bayarri and Berger 2000). A fixed 0.05 cut-off would almost never fire. The fix is
to calibrate on data the model describes by construction (Hjort, Dahl and Steinbakk 2006).
Draw 300 bots from the fitted prior, play each for 3,000 hands with the same hero, compute the
p-values at every sample size, and set each check's threshold at the 2.5% quantile. A bot the
model describes then fails each check with probability 2.5%, and fails at least one with
probability at most 5% (the union bound). Only the prior and the hero are needed, both available
in real use: no test bot, and no ground truth.

**What to do on a flag.** `PushFoldRecommender` takes the `ActionLog` and a calibrated
`FitChecker`, runs both checks on the same posterior it ranks with, and adds them to the
report. When a check fails, `RecommendationReport.advised` is the equilibrium instead of the
top of the ranking, and the explanation says which assumption broke and what the model would
have recommended. That was the pre-registered action. The results below show it is right for
one kind of misfit and wrong for the other.

**Designing on development seeds.** Two pilots on seeds 98 and 99 (never reporting seeds) fixed
the design before the pre-registration.

* The log-likelihood of the shown classes flagged connectors-first bots sooner than chi-square
  (93% vs 60% at 100 hands) but never flagged pairs-first bots. Chi-square caught those
  eventually, so chi-square it is.
* The first pilot, with 120 calibration bots, gave 10-15% false alarms at some sample sizes.
  The second showed why: a 2.5% quantile estimated from 120 values is about the third-smallest
  p-value, which is noisy. With 300 calibration bots per seed false alarms fell to 3-7%, and
  whether the calibration bots come from the prior or from the training population, or the
  p-values are mid-p or not, barely mattered.

**The experiment** (`experiments/phase3_fitcheck.py`, pre-registered at commit `c4c15d0`; 5
seeds × 30 bots per population, 10 bb, 400 replicates per check, 7 minutes on 12 processes).
The standard and pairs-first bots and hands are Step 27's, so the same bots are now seen by
three experiments.

**Results.** Share of bots flagged by either check (95% Wilson intervals in the summary file):

| Hands | Standard (model right) | Pairs first | Connectors first | Gear change |
|---|---|---|---|---|
| 10 | 4.0% | 8.0% | 14.7% | 7.3% |
| 30 | 2.0% | 8.7% | 38.0% | 18.7% |
| 100 | 6.7% | 16.7% | 76.0% | 44.0% |
| 300 | 5.3% | 30.0% | 96.7% | 70.0% |
| 1,000 | 9.3% | 43.3% | 100.0% | 88.0% |
| 3,000 | 4.7% | 71.3% | 100.0% | 98.7% |

* *All pre-registered predictions hold.* False alarms reach 9.3% at most (prediction: at most
  10%), though that is above the calibrated 5%. Connectors-first bots are flagged 76% at 100
  hands and 97-100% from 300 (predictions 60% and 90%). Gear-change bots are flagged 44% at 100
  hands and 88-99% from 1,000 (predictions 40% and 85%).
* *Each check catches its own failure.* The shown-hands check catches the wrong hand orders
  (connectors-first 95% by 300 hands; pairs-first slowly, 69% by 3,000), and the stability
  check catches the gear changes (69% by 300 hands, 99% by 3,000).
* *Flags mark bad intervals, mostly by population.* Pooled over the three misspecified
  populations, flagged bots' 90% intervals cover the true gain less often than unflagged bots'
  (45% vs 61% at 100 hands, 32% vs 53% at 3,000): the pre-registered prediction holds. Within a
  single population the gap is small or mixed, and for pairs-first bots after 1,000 hands it
  reverses (75% vs 64%). So most of the pooled gap comes from *which populations* get flagged,
  a composition effect (Simpson's paradox in miniature), not from a sharper per-bot signal.
* *Gear-changers break Step 29's promise, and the fallback restores it.* Against bots that
  changed gear, trusting the model loses to equilibrium play against 34% of them after 300
  hands and 31% after 3,000, far from the 5% the ranking promises when the model is right.
  Falling back on a flag cuts that to 8% at 300 hands, 1.3% at 1,000 and 0% at 3,000. The
  average gain falls too (+0.027 to +0.015 at 300 hands, +0.036 to +0.002 at 3,000), because
  trusting the model wins big against some gear-changers while losing to others.
* *A wrong hand order is a different story.* Against connectors-first bots the model's
  recommendation earns +0.091 bb/hand after 300 hands (the oracle: +0.110) and loses to none
  of them, although its intervals cover only 49%. The flag is right that the intervals are
  wrong, but the recommendation's *direction* is still right. A bot with a strange order is
  still too loose or too tight, and that is what the best response exploits. Falling back
  throws almost everything away (+0.002 at 300 hands, 0 from 1,000). Pairs-first bots: +0.094
  vs +0.029 after 1,000 hands.
* *Post hoc: act on the stability flag only.* Falling back only when the stability check fails
  keeps +0.083 (standard), +0.092 (pairs-first) and +0.065 (connectors-first) bb/hand after
  1,000 hands, while matching the full fallback against gear-changers (1.3% lost to). It still
  costs against connectors-first bots, because the stability check fires on 24-33% of them
  from 1,000 hands: a wrong hand order also leaves structure in the action residuals. This rule
  was chosen after seeing the results, so the library default stays as pre-registered until
  fresh seeds confirm it (they did: Step 31).

The lesson generalizes beyond poker: a model-fit check answers "can I trust the numbers?",
not "is the decision wrong?". Which failures should change the decision depends on how the
decision depends on the model. Here it depends on the opponent's range *sizes*, which the model
still gets right under a wrong order, and on stationarity, which a gear change breaks.

**Verification.**

* `tests/test_fit_checks.py` (11 tests): the `ActionLog` reproduces the hand log's
  observations at every prefix; bins split type and strength; mid-p counts ties half; a bot
  the model describes is rarely flagged; the shown-hands check catches a connectors-first bot
  and the stability check a gear change (and not a stationary bot); the shown-hands check is
  unaffected by how the hero plays (a hero who calls far tighter than equilibrium); empty data
  give p = 1; calibration and threshold lookup; the recommender falls back, and says why, when
  a check fails.
* `tests/test_push_fold_profile.py`: the hand-history profile keeps hand order.
* Experiment: `python -m experiments.run_phase3 --seed 2026 --only fitcheck` (7 minutes);
  smoke test with `--quick`.

## Step 31: Confirming a post-hoc rule on fresh data (P3.5b)

**The question.** Step 30 ended with a rule nobody had planned: fall back to the equilibrium only
when the *stability* check fails, and treat a failed shown-hands check as a warning about the
intervals. It looked better than the pre-registered rule on every population. But it was chosen
*after* looking at the results, among several rules one could have tried (fall back on either
check, on shown hands only, on stability only, on both, never). Picking the best-looking of
several options on the same data overstates how good it is. Statisticians call this the garden
of forking paths (Gelman and Loken 2013): each choice made after seeing the data is a
comparison that was never counted. The cure is cheap here because data come from a simulator:
run the experiment again on data the choice never saw.

**Exploratory, then confirmatory.** The project already pre-registers each experiment. P3.5b
adds the other half of that discipline: a finding that came out of one experiment's analysis
becomes the *hypothesis* of the next, with its own written predictions.

1. *Make the policy a parameter.* `PushFoldRecommender(fallback_on=...)` names the checks whose
   failure means playing the equilibrium. The default stays as pre-registered in P3.5 (either
   check). A failed check outside `fallback_on` adds one sentence to the explanation: the
   intervals are unreliable, the recommendation stands.
2. *Fresh seeds.* The replicate seeds come from `SeedSequence([2026, 35])` instead of
   `SeedSequence(2026)`. That gives new training populations and priors, new test bots in all
   four populations, new hands and new calibration bots, while every other setting is P3.5's.
   The code asserts that no seed is shared with the earlier experiments.
3. *Predictions with numbers, and a decision rule.* (1) Against gear-changers, stability only
   loses to the equilibrium against at most 10% of bots at 1,000 and 3,000 hands. (2) Against
   pairs-first and connectors-first bots at 300, 1,000 and 3,000 hands, it earns more than the
   any-check fallback, judged by a paired comparison: per seed, the mean over the same bots of
   (stability-only gain minus any-check gain), with the 95% t-interval across seeds above zero.
   (3) Against those bots it loses to the equilibrium against at most 5% from 100 hands. If all
   three hold, the default becomes stability only; if any fails, it stays.

The paired comparison matters. Both rules are scored on the *same* bots and hands, so most of
the variation between bots (some are easy to exploit, some hard) cancels in the difference, and
a small, consistent advantage shows up with five seeds where comparing two separate averages
would drown it.

**Results** (`experiments/phase3_confirm.py`, pre-registered at commit `7f50502`; 5 fresh seeds,
600 bots, 7 minutes). Paired differences are stability only minus any check, in bb/hand, with
95% t-intervals across seeds:

| Prediction | Result | Verdict |
|---|---|---|
| Gear-changers lost to, at most 10% (1,000 / 3,000 hands) | 0.7% / 0.0% | holds |
| Pairs-first: gain kept at 300 / 1,000 / 3,000 hands | +0.028 [+0.023, +0.032], +0.053 [+0.042, +0.065], +0.074 [+0.065, +0.083] | holds |
| Connectors-first: gain kept at 300 / 1,000 / 3,000 hands | +0.080 [+0.070, +0.089], +0.069 [+0.054, +0.084], +0.052 [+0.044, +0.061] | holds |
| Wrong-order bots lost to from 100 hands, at most 5% | at most 2.0% (each population) | holds |

All three hold, so the decision rule applies: `PushFoldRecommender` now falls back on the
stability check alone (`DEFAULT_FALLBACK = ("stability",)`). A failed shown-hands check still
appears in the report and the explanation, as a warning that the intervals are unreliable.

What the rule does not fix, measured on the same fresh seeds:

* Against connectors-first bots it still gives up part of the gain once enough hands are seen
  (+0.069 vs +0.102 for trusting the model after 1,000 hands; +0.052 vs +0.105 after 3,000).
  A wrong hand order also distorts the action residuals, so the stability check fires on some
  of these bots.
* Against gear-changers it cannot act before the stability check has the data: after 100
  hands it still loses to 23% of them (trusting the model: 27%).

The fresh seeds also replicate Step 30's flag rates. False alarms are 4-8% (Step 30: 2-9%);
connectors-first bots are flagged 77% after 100 hands and 94% after 300 (76%, 97%); gear-change
bots 31% after 100 hands and 91% after 1,000 (44%, 88%). The checks behave the same on data
they have never seen.

**Verification.**

* `tests/test_fit_checks.py`: under the default policy a failed shown-hands check only warns
  (the recommendation stands and the explanation says why), while `fallback_on=CHECKS` falls
  back; an unknown check name is rejected. The fallback test of Step 30 still passes, since it
  fails both checks.
* `phase3_fitcheck.run` was split into `evaluate` and `run` for reuse; its quick-configuration
  records are identical to before.
* Experiment: `python -m experiments.run_phase3 --seed 2026 --only confirm` (7 minutes).

---

## Step 32: Importing hand histories from text files (P3.6)

**The question.** Everything so far ran on hands this project simulated. Real hands arrive as
text files written by someone else, with their own conventions and their own mistakes. P3.6
adds a batch importer for completed hands. Two questions matter. Does a hand survive the
trip into the schema exactly? And when a file is wrong, does the importer notice, or does a
wrong hand slip into the profiles?

**Which format.** PHH (Poker Hand History; Kim 2024) is an open, TOML-based text format from the
University of Toronto group that also wrote PokerKit. Public research data uses it, for example
the 10,000 hands that the Pluribus AI played against professionals (Brown and Sandholm 2019). A
hand looks like this:

```toml
variant = 'NT'                      # no-limit Texas hold'em
antes = [0, 0, 0, 0, 0, 0]
blinds_or_straddles = [50, 100, 0, 0, 0, 0]
min_bet = 100
starting_stacks = [10000, 10000, 10000, 10000, 10000, 10000]
actions = ['d dh p1 TcQc', ..., 'p3 f', 'p4 cbr 210', 'p5 f', 'p6 f', 'p1 cc', 'p2 f',
           'd db 7d5h9d', 'p1 cc', 'p4 cc', ..., 'p1 cbr 230', 'p4 f']
players = ['MrBlue', 'MrBlonde', 'MrWhite', 'MrPink', 'MrBrown', 'Pluribus']
finishing_stacks = [10310, 9900, 10000, 9790, 10000, 10000]
```

`d dh` deals hole cards (`????` when unknown), `d db` deals the board, `f` folds, `cc` checks or
calls, `cbr` bets or raises *to* an amount, and `sm` shows or mucks at showdown. Players are
listed in dealing order: small blind first and button last (heads-up, big blind first).
Online poker sites have their own formats, and PokerKit can parse several of them. Those
parsers are deliberately not used: a batch importer for an online site's files is one step from
a HUD, and this project is offline research only. Watching a folder for new hands is also left
out, since that is how real-time HUDs work. The importer reads the files it is given, once.

**Parse, don't repair.** PokerKit's own loader replays a PHH file, and when the next action
does not apply it "repairs" the hand: it inserts the fold, check or showdown muck that would
make the action legal. That suits a viewer and is wrong for data: a hand with a missing action
would be imported with an invented one. The importer drives PokerKit's state machine itself and
applies the file's actions one by one. An action that does not apply rejects the hand. Only two
steps may stay implicit, because neither adds information: dealing *unknown* hole cards, and
burning cards. A hand is accepted only if:

* every action applies, in order, under PokerKit's rules (betting order, minimum raises,
  all-ins, side pots);
* the actions run to the end of the hand, so an in-progress or truncated hand is rejected. This
  keeps the information firewall simple: an import never produces a hand in progress;
* every known card appears once, and a show at showdown is of the cards that were dealt;
* the converted record passes the schema's own `find_problems` (chip conservation, positions,
  uncalled bets recomputed independently);
* the file's finishing stacks, when it has them, agree with the replay.

Every rejection carries a typed reason: `parse`, `unsupported`, `illegal_action`,
`incomplete`, `result_mismatch`, `unknown_result`, `inconsistent` or `duplicate`. A batch
import returns the accepted hands and a list of rejections (file, position in the file,
reason, detail), so nothing is dropped silently.

**The mapping, with its small pieces of arithmetic.**

* *Results.* PokerKit reports each player's payoff $\pi_i$ (final stack minus starting stack).
  The schema needs pot awards $w_i$, which exclude uncalled bets $u_i$. With contributions
  $c_i$ (everything a seat put in), $w_i = \pi_i + c_i - u_i$, and the uncalled bets are
  recomputed from the actions by the schema's own function (on each street, the excess of the
  largest bet over the second largest goes back to its owner).
* *Finishing stacks as a checksum.* A recorded finishing stack $f_i$ gives
  $\hat w_i = f_i - s_i + c_i - u_i$. The importer requires $0 \le \hat w_i \le w_i$ for every
  seat: the record may pay winners *less* than the replay (the difference is the rake,
  $\text{rake} = \sum_i c_i - \sum_i \hat w_i - \sum_i u_i \ge 0$) but never more, and a seat the
  replay pays nothing must receive nothing. With `allow_rake=False` (tournaments, research data)
  the rake must be zero, so the record must equal the replay exactly.
* *Units.* Amounts may be decimals (dollars and cents). The importer finds the smallest $k \le 4$
  with $10^k x \in \mathbb{Z}$ for every amount $x$ in the hand, multiplies through, and records
  the unit in the currency: `USD/100` means hundredths of a dollar.
* *Seats.* PHH lists players in dealing order, which is exactly PokerKit's player order and the
  schema's `seat_order_from_button` order (Step 4). With a `seats` field, player $k$ sits at
  physical seat `seats[k] - 1` and the last player has the button. The importer checks that
  these seat numbers really go clockwise from the button.

Schema v1 cannot hold straddles, antes that differ by seat (the big-blind ante of modern
tournaments), a minimum bet other than the big blind, more than one runout, or variants other
than no-limit hold'em. Such hands are rejected as `unsupported`, never approximated.

**Two PokerKit behaviors the importer has to guard against.** Probing PokerKit with hand-made
files turned up two places where it is more lenient than a data importer can be. It only *warns*
when a card is dealt twice (and not at all when the repeat is inside one deal). And it accepts a
show of cards other than those dealt and pays the pot to the shown cards: a player dealt `9h9s`
who "shows" `AhAs` wins with aces. The importer checks both itself.

**An exporter, for round trips.** `hand_to_phh` writes a schema hand as PHH. Where a board deal
or a showdown decision goes is PokerKit's business (after an all-in, PokerKit asks for the
showdown *before* dealing the rest of the board), so the exporter drives a PokerKit state and
writes each step where the state asks for it. It records finishing stacks, so a re-import
checks the result. Metadata that PHH has no field for travels in a user-defined field
(`_poker_lab_metadata`, JSON). Two properties then become testable:

1. *Round trip:* for a simulated hand $h$, `import(export(h))` equals $h$ up to card provenance
   (`as_imported(h)`: cards become `source=imported`, the hand `source="imported"`, and every seat
   gets a hole-card entry).
2. *Fixed point:* for a hand $g$ imported from someone else's file, `import(export(g)) == g`. The
   export is not the original text (comments and formatting are gone), but nothing the schema
   holds is lost.

**The experiment** (`experiments/phase3_import.py`, pre-registered at commit `582ac7c`; one seed,
40 seconds). The pipeline was first debugged on the quick configuration with a different seed,
and that output was discarded.

* *A. Round trip at scale.* 400 tables with random setups (2 to 9 players, gaps between seat
  numbers, antes, stacks from 1 to 400 big blinds, amounts in chips or cents) played 25 hands
  each between random, calling, all-in and Phase 1 archetype bots. Every hand and its redacted
  copy was exported and imported. Prediction: all identical.
* *B. Fault injection.* This is a version of mutation testing (DeMillo, Lipton and Sayward 1978)
  aimed at data rather than code, and a cousin of fuzzing (Miller, Fredriksen and So 1990):
  damage valid inputs on purpose and see what the checks catch. 1,000 simulated and 1,000
  Pluribus hands each got one random mutation per operator, and each damaged hand was imported
  twice, with and without its finishing stacks. *Structural* operators: drop a decision, swap
  two adjacent decisions of different players, cut the actions short, give a decision to another
  player, delete a TOML delimiter. *Value* operators: change a bet amount (times 0.5 to 2), a
  card, a starting stack (times 0.5 to 1.5), or move chips between two finishing stacks.
  Outcomes: rejected; *unchanged* (the mutation had no effect, left out of the rates); *record
  changed* (accepted, every player's result the same); *result changed* (accepted, a result
  wrong: a silent error). Predictions: (1) nothing raises except typed rejections; (2) every
  structural operator is rejected at least 99% of the time; (3) with finishing stacks, no
  silent errors.
* *C. Public research data.* The PHH dataset (MIT license) at a pinned commit: Pluribus (10,000
  hands), the 2023 WSOP Event #43 final table (83), five historical hands. Its HandHQ subset,
  scraped from online sites in 2009, is never fetched: a sparse git clone downloads only the
  three subsets (12 MB) into a local cache outside the project, and the results record the
  commit and a SHA-256 over every file used. Predictions: (1) PokerKit's own loader, with its
  standard evaluator, replays every accepted hand to the same result; (2) export and re-import
  reproduce every accepted hand. The import counts themselves had been seen in a development
  run, so they were reported, not predicted.
* *D. Descriptive.* Per-player statistics of the Pluribus hands, next to the Phase 1 bots.

**The first run falsified prediction B1.** 40 of the 33,540 damaged imports crashed instead of
being rejected, all from the card operator when it dealt a card twice. Inside one deal (`9h9h`)
PokerKit says nothing and the schema's `HoleCards` raised `ValueError`; between the board and a
hand PokerKit warns, carries on, and later fails an internal assertion while pushing chips.
Diagnosing those crashes found a worse bug in the importer's own check that a show matches the
deal. It compared *sets* of cards, so in a deal of `9h9h` the duplicate collapsed and its slot
counted as an unknown card, which the show `9h9s` then "filled". Eight damaged hands were
accepted this way. Their records equaled the originals, so the run counted them as unchanged
and the fault stayed invisible. The fix (commit `3c1f25c`):
* all known cards in the deals and shows are checked for repeats before the replay;
* the show-versus-deal check counts multisets;
* a PokerKit assertion on input it accepted is a typed rejection.

Regression tests cover each case, and the experiment was rerun with the same seed. Both runs
are reported.

**Results of the rerun** (commit `3c1f25c`):

| Prediction | Result | Verdict |
|---|---|---|
| A: all round trips identical | 20,000 of 20,000 (10,000 hands, each also redacted; 8,453 with an all-in, 5,643 with side pots, 3,025 in cents) | holds |
| B1: no crashes | 0 (first run: 40) | holds after the fix |
| B2: structural operators rejected ≥ 99% | 100% for every operator, source and condition | holds |
| B3: no silent errors with finishing stacks | 0 of 7,814 effective value mutations | holds |
| C1: PokerKit's own loader agrees on every accepted hand | 9,993 of 10,001 | **does not hold** |
| C2: export and re-import reproduce every accepted hand | 10,001 of 10,001 | holds |

Without finishing stacks the value operators tell the story of what a file alone can and cannot
prove (shares of effective mutations, Pluribus / simulated hands):

| Operator | Rejected | Accepted, record changed | Accepted, a result wrong |
|---|---|---|---|
| Change a bet amount | 16.5% / 64.6% | 48.2% / 21.4% | 35.3% / 14.0% |
| Change a card | 29.1% / 49.5% | 70.7% / 43.3% | 0.2% / 7.2% |
| Change a starting stack | 0.3% / 18.6% | 99.6% / 67.1% | 0.1% / 14.3% |

With finishing stacks the "result wrong" column is zero everywhere and the rejections take
its place. The "record changed" column cannot shrink, because no check can see those changes:
the size of a bet nobody called, a folded player's hole card, a starting stack deeper than
anything bet. Those changes do matter to a profiler (a bet size is a tell), so imported data is
only as good as its source. The checks guarantee consistency, not truth.

**Why C1 failed, post hoc.** All 8 disagreements are split pots with an odd chip. The Pluribus
files record half chips: a 1,349-chip pot split two ways gives each winner 674.5, so a
finishing stack reads `10112.5`. The importer scales those 8 hands to tenths of a chip and
reproduces the file exactly. PokerKit's own loader works in whole chips and gives the odd chip
to one winner (675 and 674). Both are consistent with the rules; they disagree on a
convention. The prediction is reported as failed, with this explanation labeled post hoc.

**What the data cover.** 10,001 of 10,088 files were imported: all 10,000 Pluribus hands, the
Dwan–Ivey hand of the five historical ones (the others are badugi, pot-limit Omaha, triple-draw
lowball and short-deck), and none of the 83 WSOP hands. That event is a mixed-game championship
(72 hands in other variants), and its 11 hold'em hands use a big-blind ante.

**What the real hands show (descriptive).** Imported hands go straight into the Phase 1 stats
code (redacted to public information):

| | VPIP | PFR | 3-bet | C-bet | Fold to c-bet | WTSD | W\$SD | AF |
|---|---|---|---|---|---|---|---|---|
| 13 professionals (50,000 player-hands) | 26.8 | 17.8 | 7.9 | 51.9 | 46.2 | 30.1 | 52.3 | 2.25 |
| Pluribus (10,000) | 26.9 | 18.0 | 7.3 | 53.7 | 45.0 | 30.6 | 48.8 | 2.33 |
| Phase 1 TAG bots | 19.8 | 14.4 | 5.4 | 70.1 | 49.4 | 43.5 | 54.9 | 2.97 |
| Phase 1 LAG bots | 29.4 | 21.8 | 12.5 | 79.2 | 41.1 | 47.2 | 50.2 | 3.51 |

Preflop the professionals sit between the TAG and LAG bots. After the flop they differ more:
they c-bet about 52% of the time against 70–79% for those bots, and they reach showdown after
seeing the flop 30% of the time, while every Phase 1 archetype averages 43–70%. Phase 1 logged
that its bots go to showdown too often (Decisions Log, 2026-10-20); real data now put a number on
it. A classifier trained on the bots would meet real players outside its training range on
exactly these statistics, which is what P3.7 (priors fitted to real data) has to deal with.

**Verification.**

* `tests/test_phh_import.py` (35 tests): round trips of simulated hands covering heads-up play,
  antes, gaps between seats, short stacks and all-ins, each also redacted; file round trips
  through `.phh` and `.phhs`; a capture-style hand with camera cards, a muck and metadata; a
  hand-written foreign file in dollars and cents with physical seats, comments and a
  user-defined field (mapping checked field by field, then the fixed point); one test per
  rejection reason; a hand PokerKit's loader repairs but the importer refuses; shows of other
  cards; cards dealt twice (6 cases, 4 of them found by the experiment); rake and unknown
  results; the batch report; the command-line tool; imported push/fold hands giving the
  profiler exactly the observations of the originals.
* The simulator's mapping of PokerKit's blind and ante posts became one function shared with
  the importer (`engine.simulator.forced_posts`); the engine tests are unchanged.
* Experiment: `python -m experiments.run_phase3 --seed 2026 --only import` (40 seconds; the
  first run fetches the dataset). Command-line import:
  `python -m poker_lab.importers DIR --out hands.jsonl.gz --no-rake`.

## Step 33: Priors from real players, and do the bot-trained models transfer? (P3.7)

**The question.** Phase 1 fit its empirical-Bayes priors, its classifier and its action
models to bots. P3.6 brought in real hands. Two questions follow. Do priors fit to real players
estimate a new real player's statistics better than the defaults? And do the models trained on
bots still mean what they say when they meet real players?

**Which data.** A search for public no-limit hold'em hand histories (October 2026) found only
one in-scope source of multi-way human play: the Pluribus hands (13 professionals, 10,000
hands, 6-max, 100 big blinds). The others were heads-up only (the DeepStack study's logs, about
45,000 hands, in the Science supplement), bot against bot (the Annual Computer Poker
Competition logs), or collected in ways the project's "no scraping" rule excludes: HandHQ
(scraped from real-money sites in 2009) and the IRC Poker Database (play-money games logged by
an observer bot in 1995-2001). The user chose Pluribus only. Thirteen elite professionals are
a small and unusual population; everything below is about them, not about a home game.

**Empirical Bayes, revisited.** Step 10 models each player's true rate $p_i$ for a binomial
statistic as a draw from a population prior $\mathrm{Beta}(\alpha, \beta)$. Write the prior as
a mean $\mu = \alpha/(\alpha+\beta)$ and a concentration $\kappa = \alpha + \beta$ (in
pseudo-opportunities). The concentration says how alike the players are: the between-player
standard deviation of the true rate is

$$\mathrm{sd}(p) = \sqrt{\frac{\mu(1-\mu)}{\kappa + 1}}.$$

The posterior mean after $k$ successes in $m$ opportunities, $(\alpha + k)/(\kappa + m)$, gives
the prior the weight of $\kappa$ extra opportunities. A homogeneous population (large
$\kappa$) shrinks hard; a varied one (small $\kappa$) lets the data speak sooner.

**Scoring a prior without knowing anyone's true rate.** For a held-out player, observe a window
of $n$ hands and form the estimate $e$ (the posterior mean). The target is the player's rate
$r'$ in all their *other* hands. $r'$ is noisy, but its noise is independent of the window, so
with $p$ the player's true rate and $E[r' \mid p] = p$,

$$E[(e - r')^2] = E[(e - p)^2] + E[(r' - p)^2],$$

because the cross term $E[(e - p)(r' - p)]$ vanishes. The second term is the same for every
prior, so *differences* in this score between priors are unbiased for differences in mean
squared error against the true rate. For the RMSE shown, an unbiased estimate of the noise
term, $r'(1 - r')/(m' - 1)$ with $m'$ held-out opportunities, is subtracted. A test checks the
whole estimator on simulated players whose true rates are known: the corrected score lands
within 5% of the true error.

**Opportunities weigh players unequally.** The beta-binomial marginal likelihood
(Step 10) counts every opportunity: a player with 300 c-bet chances moves the fit more than one
with 3. That is the maximum-likelihood answer, but when how often a player gets a chance is
related to how they use it, the fit is pulled toward the players with many chances. Among the
bots, calling stations rarely raise before the flop, so they get few c-bet chances, and they
c-bet least. A development run found that a c-bet prior fit to all 300 training bots is worse
than the uniform prior for estimating held-out bots, for this reason. The rule for how many
players a refit needs therefore uses the mean error over the ten statistics, and per-statistic
failures are reported.

**A support check.** A classifier returns probabilities over its five archetypes for any input,
including a player unlike all of them, and its calibration was measured only on bots.
`SupportCheck` asks first whether a profile resembles any archetype it was trained on. For each
training sample size it fits one Gaussian per archetype (mean $\mu_k$, covariance $S_k$) in the
classifier's feature space (logits of rates, logs of sizes) and scores a profile $x$ by its
smallest squared Mahalanobis distance,

$$d^2(x) = \min_k\, (x - \mu_k)^\top S_k^{-1} (x - \mu_k).$$

The threshold is *split-conformal* (Vovk, Gammerman and Shafer 2005): 20% of the training
players are held out (all of a player's windows together), and with $m$ held-out windows of the
same size,

$$p(x) = \frac{1 + \#\{\text{held-out windows with } d^2 \ge d^2(x)\}}{m + 1}.$$

A profile is flagged when $p(x) \le \alpha$. If a new window and the held-out ones are
exchangeable (drawn the same way), it is flagged with probability at most $\alpha$, whatever
the archetypes' true shapes: the Gaussians only order the profiles. One consequence decided
$\alpha$. The smallest possible p-value is $1/(m+1)$, and Phase 1 bots have only one
2,000-hand window each, so 60 held-out bots give $m = 60$ at that size: at $\alpha = 1\%$
nothing could ever be flagged there. The check uses $\alpha = 5\%$, the level of P3.5's checks.

**A label-free calibration check.** Real players have no true archetype, but a calibrated
classifier still makes a testable promise. Its top probability after $n$ hands should match how
often its label agrees with the label it gives a later, disjoint block of 1,000 hands of the same
player. The *gap* is mean top probability minus that agreement rate. It is not zero even for
bots (the reference label is itself sometimes wrong), and it depends on the archetype: about 0
for calling stations, about $-0.1$ for TAG bots at 1,000 hands. So each player is compared with
bots that got the *same reference label*: a player's expected gap is the bots' gap for their
reference label, averaged over their 2,000-hand chunks. The verdict, fixed in advance, uses the
95% interval of (gap minus expected gap), averaged over players:

* *transfers* if every interval lies within $\pm 0.10$ (an equivalence test: the data must show
  the difference is small, not merely fail to show it is large);
* *fails* if some interval excludes 0;
* *inconclusive* otherwise.

**A control with the real data's shape.** Thirteen players with 488 to 9,121 hands each, at
tables that always include Pluribus, are not the population the checks were calibrated on. The
control replays the real seating (which players sat in which of the 92 sessions, and for how
many hands) with each professional replaced by a Phase 1 bot and Pluribus by a TAG bot, and runs
every check on it. A check that fails on the professionals and passes on the control fails
because of the players, not the pipeline or the amount of data.

**Developed on stand-ins, then run once.** The whole pipeline was built and debugged on
simulated stand-ins only: the quick configuration and two full-size runs on development seeds
98 and 99, with 13 Phase 1 bots playing in random sessions as the "players". The real hands were
loaded once before the plan was committed, to check that the loader reproduces P3.6's
per-player hand counts. The stand-in runs changed five things before the pre-registration
(commit `9fce42a`): the support check's $\alpha$ (1% could never flag at 2,000 hands); the
consistency check's comparison with label-matched bots (the raw gap depends on the archetype);
its verdict on intervals instead of point estimates (with 8 players, noise alone could decide a
point estimate); the min-players rule on the mean over statistics (the c-bet prior above); and
two bounds for the control. The predictions themselves are about the professionals, whose
statistics P3.6 had already shown, so the ones that depend on how far they sit from the bots
are not blind. Part A has no predictions for that reason.

**The experiment** (`experiments/phase3_priors.py`; seed 2026; 9 minutes; the five Phase 1
replicates come from the simulation cache, re-simulated under the current code and identical
to the old cache bit for bit):

* *A.* Priors fit to the 13 professionals, next to the bots' priors.
* *B.* Leave one professional out: priors fit to the other twelve against the uniform default
  and the bots' prior, on windows of 25 to 1,000 hands. *B3* turns the professionals' prior on
  the bots. *B4* refits on $k$ players to set `refit_priors`' default.
* *C.* Phase 1's classifier on windows of the professionals' hands: the support check, the
  labels, the consistency check.
* *D.* Next-action log-loss on the professionals' decisions (the Phase 5 metric), for models
  trained on bots and models trained on the other professionals (leave one out).
* *Control.* Parts A-D on Phase 1 bots replaying the real seating.

**Results** (commit `9fce42a`):

| Prediction | Result | Verdict |
|---|---|---|
| B1: the professionals' prior beats uniform, every statistic, 25-200 hands | all 10 at 25, 50, 100 hands; 7 of 10 at 200 | **does not hold** |
| B2: it beats the bots' prior, same criterion | 28 of 40 statistic-size pairs | **does not hold** |
| B3: on bots, it is worse than uniform for at least 5 of 10 statistics at 100 and 500 hands | 8 and 8 | holds |
| C1a: held-out bots flagged at most 7% | at most 5.8% | holds |
| C1b: professionals flagged at least 80% at 500-2,000 hands | 100% | holds |
| C3: consistency verdict *fails* | *fails* (overconfident at 25-100 hands) | holds |
| D1: the bot-trained model predicts worse | by 0.08-0.12 nats per decision | holds |
| D2: the profile helps by less than 3% | yes, except the bot-trained model at 1,500 hands (3.8%) | **does not hold** |
| D3: the bot-trained model is worse calibrated | yes at every checkpoint, but the interval excludes 0 only at 0 and 25 hands | **does not hold** |
| Control C1: flagged at most 25% | 2-6% up to 200 hands, 24-56% at 500-2,000 | **does not hold** |
| Control C3: not *fails* | *inconclusive* | holds |
| Control D1: bot-trained model not worse | yes | holds |
| Control D2: the profile helps by at least 3% | 13-16% | holds |
| B4 decision: `min_players` | 2 for professionals, 8 for bots: **8** (was 30) | rule fired |

**A. The professionals are alike.** Their priors are concentrated (the concentration is 36 to
1,551 pseudo-opportunities, against 5 to 56 for the bots), so the between-player spread is
small:

| | VPIP | PFR | 3-bet | C-bet | Fold to c-bet | WTSD | W\$SD |
|---|---|---|---|---|---|---|---|
| Professionals: mean | 26.5% | 17.9% | 8.2% | 51.6% | 45.6% | 30.3% | 51.9% |
| Professionals: SD between players | 2.9 | 1.6 | 1.5 | 7.3 | 5.0 | 1.2 | 3.3 |
| Phase 1 bots: mean | 31.9% | 17.8% | 10.0% | 65.3% | 38.9% | 50.7% | 50.1% |
| Phase 1 bots: SD between players | 17.1 | 12.3 | 9.6 | 19.2 | 15.3 | 11.1 | 6.6 |

(SD in percentage points.) The bots differ from each other 1.2 to 9.5 times as much as the
professionals do. The professionals also size differently: post-flop bets of 0.64 of the pot
and opens to 2.4 big blinds, against 0.82 and 4.0 for the bots. Pluribus sits at the
professionals' center on almost every statistic.

**B. A prior from the right population is worth hundreds of hands.** RMSE against the true
rate, percentage points, mean over the ten statistics:

| Hands | Uniform | Bots' prior | Professionals' prior (leave one out) |
|---|---|---|---|
| 25 | 13.8 | 10.2 | 4.8 |
| 100 | 10.8 | 8.1 | 4.6 |
| 200 | 8.5 | 6.9 | 4.4 |
| 1,000 | 4.9 | 4.5 | 3.1 |

After 25 hands the right prior is better than the default after 1,000. For WTSD the
professionals differ so little that the prior alone is as good as anything the hands add (the
corrected error is 0.0: below what the held-out noise lets us resolve). B1 fails only at 200
hands, where three intervals (VPIP, PFR, fold to c-bet) reach below 0: by then the data carry
those estimates whatever the prior. B2 fails more widely: the bots' prior is close for W\$SD
(both centered near 50%) and fold to c-bet.

*B3, the other side.* On the bots, the professionals' prior is the worst of the three: 14.8
points after 100 hands against 11.1 for uniform, worse for 8 of the 10 statistics at 100 and at
500 hands. A concentrated prior from one population is confidently wrong about another. A home
game is not a table of professionals, so these priors are not shipped as a default: refit on
your own players.

*B4.* A refit helps from 2 professionals on, but needs 8 bots: with a varied population, a
handful of players gives a prior that is confidently wrong about the rest. The pre-registered
rule took the larger value, so `ProfileStore.refit_priors` now refits from 8 players (was 30).

**C. The classifier does not transfer.** The support check flags 52% of the professionals'
25-hand windows, 95% at 100 hands and all of them from 200 hands, while held-out bots stay at
2.8-5.8%. Pluribus is flagged 77-100%. The classifier still answers: it calls the
professionals TAG (64% of windows at 2,000 hands) or LAG (33%), with lower confidence than on
bots (0.71 against 0.91). And the confidence it does give is too high. Compared with bots that
got the same label, its top probability exceeds its agreement with a later 1,000-hand look by
0.14, 0.13 and 0.12 at 25, 50 and 100 hands (intervals above 0): the verdict is *fails*. The
labels are not a description of these players, and their probabilities are not calibrated for
them.

*The control's surprise.* Bots replaying the professionals' seats were flagged at the nominal
rate up to 200 hands (2-6%) but 24%, 28% and 56% at 500, 1,000 and 2,000 hands. The
pre-registered bound of 25% failed. The development stand-ins, seated at random, stayed at
9-18%. The likely reason, not tested here: statistics depend on the opponents, and in the
real seating each player meets the same few opponents for thousands of hands, while Phase 1's
league redraws its tables every 50 hands. So at large samples the support check also reacts
to the table, not only to the player. At small samples, where the professionals are already
flagged 52-95%, the control shows the flags come from the players.

**D. Real players need a model trained on real players.** A next-action model trained on bots
is 0.08 to 0.12 nats per decision worse on the professionals than one trained on the other
professionals (0.726 against 0.620 with no history), and worse calibrated (error 0.038-0.047
against 0.013-0.027). Adding the player's profile helps almost nothing: 0.0% to -2.3% for the
model trained on professionals (it makes the log-loss slightly worse), 1.0% to 3.8% for the
bot-trained one, against 11-13% on the bots in Phase 1 (13-16% on the control). With players
this alike, knowing the population is what predicts their actions; for Phase 5 that population
model has to come from real hands.

**What changed in the library.** `ProfileStore` keeps split halves (format 2, loads format 1),
so a refit also fits the size priors; `refit_priors` refits from 8 players; `SupportCheck` is
available next to the classifier. Nothing is shipped from the professionals' data.

**Limitations.** Thirteen elite professionals, in a lab setting, all of whose tables included
Pluribus; whatever these results say about real players, they say it about this group. The
consistency check covers the 8 professionals with at least 2,000 hands. The control explains
less than hoped at large samples, because a fixed table changes the statistics.

**Verification.**

* `tests/test_profile_store.py` (split halves sum to totals and follow the hand id; a refit fits
  the size priors from halves and keeps them for a format 1 file; the new default),
  `tests/test_support_check.py` (held-out split by player; false alarms near alpha on new
  players; shifted players flagged; nearest size in log hands; pooled covariance for small
  classes), `tests/test_priors_experiment.py` (the corrected held-out error within 5% of the
  true error on simulated players with known rates; the decision rules; session order; seating
  blocks).
* The re-simulated Phase 1 cache is identical to the old one (all counters and decision rows).
* Experiment: `python -m experiments.run_phase3 --seed 2026 --only priors` (9 minutes with the
  simulation cache).

## Step 34: A sequence model over raw actions, against hand-written stats (P3.8)

**The question.** Phase 1 identifies a player's archetype from twelve numbers: ten rates with
carefully counted opportunities (VPIP, c-bet, fold to a c-bet, ...) and two sizes, shrunk
toward population priors and fed to a logistic regression. Every one of those numbers was
chosen by hand. The spec's optional extension asks whether a neural network that reads the raw
hands (every decision, in order) does better on the same splits and metrics. It is the classic
question of hand-made features against learned ones, asked on a problem where the hand-made
features are good.

**Hands as tokens.** `profiling.sequences.hand_tokens` turns one complete hand into, for each
seated player, a short sequence:

* one token per decision the player made, in order: the public situation just before it (the
  16 `SITUATION_FEATURES` of Step 13: street, facing a bet, pot odds, pot size, stack-to-pot
  ratio, players left, raises this street, position, ...), the action as one of five
  (fold, check, call, bet, raise), and for a bet or raise its size, as
  $\log(1 + r)$ with $r$ = the raise over the call divided by the pot after calling, and as the
  log of the player's total street bet in big blinds;
* one closing token per hand: saw the flop, reached showdown, won at showdown (what the stats
  counters know of the outcome).

A hand where the player made no decision (a walk) is the closing token alone, so hands can be
counted. Only public information is used, not even shown cards: the tokens carry what the
counters carry, as events rather than totals. A test recovers each bot's size parameter from
its tokens (median postflop bet within 10% of `bet_size_pot`).

**Re-simulating Phase 1, bit for bit.** Phase 1's cache kept decisions only inside its
action-prediction windows, so the five replicates were simulated again with a league extractor
that records every token (about 90 seconds each; 1.6 million training tokens per replicate).
Same seeds, same code: the run checks that players, labels, parameters, all per-hand counters
and all of Phase 1's decision rows come out identical, and stops otherwise. The tokens are
stored compressed-row style (`HandSequences`): player $p$'s hands are a slice of a hand
array, hand $h$'s tokens a slice of a token array.

**The network** (`profiling.sequence_model.SequenceClassifier`, PyTorch):

1. *Within a hand, order matters.* Each token passes through a small MLP; a GRU (a recurrent
   network with gates that decide what to keep, Cho et al. 2014) reads a hand's tokens in
   order and its last state describes the hand. "Raise, bet, check" can then mean more than
   its parts.
2. *Across hands, it does not.* The bots are stationary, so a player's hands are exchangeable:
   any reordering is equally likely. The summary of a window should then be a symmetric
   function of its hands; Zaheer et al. (2017) show that sums of per-item features, followed
   by any function, cover the symmetric functions ("deep sets").
3. *Which symmetric function.* The first version averaged the hand embeddings. On the
   development seeds it plateaued at about 92% accuracy at 2,000 hands, where the logistic
   model reaches 95%. An average of embeddings has to learn ratios (c-bets per c-bet chance)
   from only 240 training players. The fix builds Phase 1's recipe into the network: each hand
   emits, for each of $K = 16$ learned "stats", a soft opportunity $c \in [0,1]$ and the share
   $t \in [0,1]$ of it taken, that is soft successes $ct$ and failures $c(1-t)$. A window sums
   them, $S$ and $F$, and reads each stat as the logit of its posterior mean under a learned
   $\mathrm{Beta}(a, b)$ prior:

   $$x_k = \log(S_k + a_k) - \log(F_k + b_k) = \operatorname{logit}\frac{S_k + a_k}{S_k + F_k + a_k + b_k}.$$

   A short window is pulled toward the prior and a long one converges to the rate: shrinkage
   (Step 10), with stats the network chooses.
4. *Head.* An MLP reads the $K$ shrunk rates and $\log n$ and gives five logits.

*Training cheaply.* A window's sums are differences of cumulative sums over the player's hand
outputs, so one pass of the encoder over a batch of players' 2,000 hands gives every window of
every size at once: a hand is encoded once per epoch however many windows contain it. Each
epoch draws, per training player and size, as many windows as Phase 1 used (8 up to 200
hands, then 4, 2, 1) at random starts. Adam with a cosine-decayed learning rate and clipped
gradients; 30 epochs; the 20% of training bots that Phase 1's classifier held out for its
temperature (the same players) pick the epoch and fit the temperatures. Results are
bit-identical for a seed and a thread count.

**Two ablations.** *Order dropped* (`bag`): the same network with a hand's token embeddings
summed instead of read by the GRU. *Counters as input* (`counters`): the same order-free
network, each hand being its stats-counter vector instead of its tokens: the information of
Phase 1's stats, with a learned summary. Together they split a difference from Phase 1 into
order, raw information and the learner.

**Part C, more players.** A development check suggested the networks were short of training
players rather than information, so the plan adds a test: 1,200 more training bots per
replicate (240 per archetype, drawn and simulated like Phase 1's pool, in a league of their
own), and every model refit on all 1,500.

**Part B, real players.** The professionals' transfer check from Step 33, repeated for the
sequence model on the same windows, with a new control draw.

**Developed on two other seeds, then run once.** Everything was built on development seeds 98
and 99 (full-size replicates with new populations) and the quick configuration; their test
windows were looked at. That is where mean pooling failed and was replaced, where the schedule,
clipping, batch size and number of rates were set (compared on seed 98, differences of 0.01
to 0.02 nats, inside the noise), and where, with those settings frozen, seed 99 showed every
network 0.02 to 0.07 nats worse than the logistic model. A check on seed 99 with 1,500
training bots closed that gap, which is why part C exists. The plan and its eight predictions
were committed (`c4578ea`) before any network saw Phase 1's five replicates; the real hands
were not loaded during development.

**The experiment** (`experiments/phase3_sequence.py`; seed 2026, commit `c4578ea`, 8 torch
threads). Every replicate came out identical to Phase 1, and the refit logistic model
reproduced Phase 1's published accuracy and log-loss exactly (largest difference 0). The run
took 3.8 hours, but about 2.4 of them were one network's training stalled while the machine
slept; a replicate otherwise takes 17 minutes.

| Prediction | Result | Verdict |
|---|---|---|
| A1: the sequence model beats the logistic model at no sample size | gains -0.044 to -0.003 nats, no interval above 0 | holds |
| A2: order within a hand changes log-loss by under 0.02 | -0.012 to +0.009 | holds |
| A3: tokens beat the stats counters by under 0.02 | counters - tokens -0.006 to +0.012 | holds |
| B1: on the professionals, the sequence model is overconfident at 25-100 hands | +0.17, +0.15, +0.16, every interval above 0 | holds |
| B2: on the control, neither model is overconfident at 25-100 hands | no interval above 0 | holds |
| C1: with 1,500 training bots the logistic model changes by under 0.02 | +0.006 to +0.012 | holds |
| C2: the token network improves by at least 0.02 everywhere | +0.033 to +0.062, every interval above 0 | holds |
| C3: with 1,500 bots the token network is within 0.03 of the logistic model | ahead by +0.008 to +0.019; interval upper ends reach +0.041 | **does not hold** |

**A. At Phase 1's scale the hand-written stats win.** Held-out bots, Phase 1's windows, means
over the five replicates:

| Hands | Logistic: accuracy / log-loss | Sequence model | Order dropped | Counters network |
|---|---|---|---|---|
| 25 | 73.9% / 0.580 | 74.5% / 0.602 | 74.0% / 0.615 | 73.5% / 0.627 |
| 100 | 86.4% / 0.322 | 86.0% / 0.325 | 85.9% / 0.327 | 85.8% / 0.325 |
| 500 | 93.5% / 0.189 | 91.9% / 0.218 | 92.3% / 0.209 | 92.1% / 0.206 |
| 2,000 | 95.7% / 0.152 | 92.5% / 0.196 | 93.5% / 0.187 | 93.7% / 0.181 |

The sequence model ties the logistic model at 50 to 100 hands and falls behind with more: 90%
accuracy takes 228 hands against 194 (and one replicate never gets there). At 25 hands it is a
little more accurate but worse calibrated (calibration error 0.063 against 0.043), so its
log-loss is worse; per-size temperatures shrink that gap from 0.022 to 0.009 nats without
closing it. The two ablations
explain the deficit. Order inside a hand adds nothing measurable (A2), probably because each
token's situation already says where in the hand it was made. Raw tokens add nothing over
the stats counters fed to the same network (A3). The gap is the learner, not the information:
twelve well-chosen numbers and a logistic regression are hard to beat with 240 players to
learn from.

**C. With five times the players, the flexible models pull ahead.** Refit with 1,500 training
bots (log-loss before minus after):

| Hands | Logistic | Boosted trees | Token network | Counters network |
|---|---|---|---|---|
| 25 | +0.010 | +0.053 | +0.062 | +0.067 |
| 200 | +0.007 | +0.053 | +0.035 | +0.030 |
| 2,000 | +0.007 | +0.072 | +0.050 | +0.046 |

The logistic model is limited by its features, not its data (C1). The networks gain 0.03 to
0.06 nats (C2) and end up ahead of it by 0.008 to 0.019 nats, clearly so at 100 and 200 hands;
boosted trees, which trailed the logistic model in Phase 1, become the best from 500 hands (0.023 to
0.039 ahead, intervals above 0). C3 failed in the direction nobody minds: the networks did
better than "within 0.03" allowed. The lesson is the textbook one about flexible models: their
ranking against a simple model depends on how much data there is, so "the neural network lost"
is a statement about 300 training players.

**B. On real players the sequence model is no better, and the two models disagree.** On the
professionals, the sequence model's confidence exceeds its agreement with a later 1,000-hand
look by 0.15 to 0.17 more than for bots of the same label at 25 to 100 hands (the logistic
model: 0.12 to 0.14). Both verdicts are *fails*. And the two bot-trained classifiers give a
professional the same archetype in only 33% to 48% of windows, against 83% to 94% on held-out
bots: outside their training range they extrapolate in different directions. On the control
neither is overconfident at small samples (B2). The sequence model's verdict there is still
*fails*, because it is *under*confident at 500 and 1,000 hands; the plan had warned that with
13 players and six sample sizes the verdict rule raises false alarms (it did so on a
development stand-in too), which is why B1 and B2 were stated on the small-sample intervals.

**What changed.** The library gains `profiling.sequences` (hand tokens and a league extractor)
and `profiling.sequence_model` (`HandSequences`, `SequenceClassifier`, with mean or learned-rate
pooling, global or per-size temperatures). No default changes: Phase 1's logistic model stays
the classifier, and the reasons not to trust any bot-trained classifier on real players
(Step 33) apply to the network with more force.

**Limitations.** One architecture family, set on two development seeds; the GRU was run only
in part A, with the order-free network standing in for it in part C (A2 says order does not
matter). The bots are stationary and stereotyped (sizing nearly fixed per archetype), so the
things a sequence model could see and statistics cannot (a player changing gears, timing,
multi-street lines that depend on the board) do not exist in this data. Shown cards were left
out to keep the comparison with the stats fair.

**Verification.**

* `tests/test_sequence_model.py`: tokens are each player's decisions in order plus a closing
  token whose outcome flags equal the counters; bot sizes recovered from tokens; rows rebuild
  hands in any arrival order; window logits equal the head applied to the mean or to the shrunk
  rates; the GRU separates two classes that differ only in the order of a hand's tokens while
  the order-free network cannot; fits are bit-identical for a seed; the validation players are
  the logistic model's temperature players.
* `tests/test_sequence_experiment.py`: windows match Phase 1's row by row; the consistency
  chunks; the stand-in loader keeps counters and tokens in step.
* Each replicate re-simulated identical to Phase 1 (checked in the run); the refit logistic
  model reproduces Phase 1's published numbers exactly.
* Experiment: `python -m experiments.run_phase3 --seed 2026 --only sequence` (about 1.4 hours
  on 16 threads, most of it simulating and training on the extra bots of part C).

## Step 35: Schema v2: big-blind antes, straddles and minimum bets (P3.10)

**The problem.** Schema v1 described a table's forced bets with three numbers: small blind,
big blind, and an ante that every seat pays. Three common rules did not fit:

* the *big-blind ante* of modern tournaments: one player, the big blind, antes for the whole
  table (the 2023 WSOP final table in the PHH dataset antes 1.5 bb this way);
* *straddles*: a voluntary third (and fourth) blind posted before the cards, usually twice the
  big blind, by the player after the big blind or, in some games, by the button;
* a *minimum bet* other than the big blind (some home games and published records).

The importer of Step 32 rejected such hands by name, so the dataset's 11 WSOP hold'em hands
never got in. Approximating them (say, spreading the big-blind ante over every seat) would
have changed who wins what, so the schema had to grow.

**The design.** `TableConfig` gains `big_blind_ante`, `straddles` and `min_bet`.

* *Dealing order.* Forced bets are listed the way PokerKit and PHH list them: small blind, big
  blind, then `straddles[0]` for the first player after the big blind, `straddles[1]` for the
  next, with zeros for players who do not straddle. A button straddle four-handed is
  `straddles = (0, 4)`. `TableConfig.blinds_or_straddles` returns the whole list.
* *Semantic antes.* PHH stores one ante per player. The schema stores the two kinds that exist
  in practice, `ante` (every seat) and `big_blind_ante` (one seat), and refuses a table with
  both. Anything else (a button ante, say) is rejected on import rather than stored as a list
  nobody downstream knows how to read.
* *Canonical values.* `min_bet` defaults to the big blind and `straddles` has no trailing
  zeros, so two descriptions of the same rules compare equal. A straddle must exceed the bet
  before it; a table cannot have more straddles than seats after the blinds.
* *One translation to PokerKit.* `engine.pokerkit_rules(table, n)` turns a table into
  PokerKit's `(ante_trimming_status, antes, blinds_or_straddles, min_bet)`. The simulator, the
  PokerKit replay and the PHH exporter all use it, so they cannot drift apart. A big-blind ante
  is never *trimmed*: PokerKit would otherwise treat the one ante larger than everyone else's
  as an unmatched bet and hand it back. It is dead money.

**Checking forced posts against the rules.** The action log already recorded each post with
its amount; nothing checked that the posts were the ones the rules require. Validation now
recomputes them independently of PokerKit (`schema.expected_forced_posts`). For the player in
dealing position $k$ with starting stack $S$, nominal ante $A_k$ (the ante, plus the big-blind
ante for the big blind) and nominal blind or straddle $B_k$:

$$a_k = \min(A_k, S), \qquad b_k = \min(B_k,\; S - a_k).$$

The ante comes first, as in PokerKit, so a short big blind may post its ante and only part of
its blind. Heads-up the dealing order starts with the big blind while the button posts the
small blind, so positions 0 and 1 swap roles. Every simulated hand passes this check as it is
recorded, which cross-checks the engine's posts against a second implementation on every hand.

**Version 2 and migration.** `SCHEMA_VERSION` is now 2 and records are always written in it.
`migrate_record` reads a version 1 record. It first checks the record against version 1's own
keys (a version 1 record carrying a `min_bet` key is an error, not a hint), then fills in what
version 1 implied: no big-blind ante, no straddles, a minimum bet of one big blind.
`hand_from_dict` and `read_jsonl` migrate transparently; `migrate_jsonl` rewrites a file. How do
we know the migration is right? A fixture (`tests/data/schema_v1_hands.jsonl.gz`, 40 hands)
was written by the version 1 code itself at commit `0cf0c4f`, from a function kept in
`tests/schema_fixture.py`. The test reads the file with the new code and compares it with what
the same function makes today. Old code's output, new code's reading, identical.

**What else changed.** The push/fold strategy (Step 28) treats any ante or straddle as off its
tree. Profiling needs nothing new: a straddle is a forced post, so a straddler who checks its
option is not counted as voluntarily entering the pot (a test checks this), and the first raise
over a straddle is the open.

**The experiment** (`experiments/phase3_schema.py`, pre-registered at commit `4c4e373`):

* *A. Round trips at v2 tables.* 400 random tables in the style of Step 32 (2 to 9 players,
  gaps between seats, stacks from 1 to 400 bb, chips or cents, five kinds of agents), 50 for
  each of eight structures: blinds only, an ante for every seat, a big-blind ante, a straddle,
  a double straddle, a button straddle, a minimum bet of 2 bb and one of 0.5 bb. Each of the
  10,000 hands, and its redacted copy, went through: the rules check, a replay through
  unmodified PokerKit, a JSON round trip, a PHH round trip, and PokerKit's own PHH loader on
  the exported text.
* *B. Version 1 files at scale.* The version 1 code, checked out in a temporary git worktree,
  wrote what it makes in Step 32: the 10,000 hands of its round-trip tables and the 10,000
  Pluribus hands. The current code read those files and made the same hands itself.
* *C. The WSOP hold'em hands*, and the historical hands, at the pinned dataset commit.

Developing on the quick configuration showed that PokerKit's loader sometimes splits a pot's
odd cent differently from the engine (Step 32 had seen the same in the Pluribus files). Before
the real run the plan therefore counted a difference of less than one unit per player as an
*odd chip*, reported separately.

| Prediction | Result |
|---|---|
| A1: every check passes on every hand, for every structure | **does not hold**: 79,995 of 80,000; 5 loader checks fail |
| B1: every version 1 record migrates to the hand the current code makes | holds: 20,000 of 20,000 |
| C1: every no-limit hold'em hand imports; only other variants are rejected | holds: 11 of 11 WSOP hands, 1 of 1 historical |
| C2: PokerKit's own loader agrees on every imported hand | holds: 12 of 12 |
| C3: export then import reproduces every imported hand | holds: 12 of 12 |

**What the failures were.** All five failing checks are the loader on pots split three or more
ways in cents, where the difference reaches 1.33 units for one player. PokerKit gives a split
pot's *whole* remainder to the first winner in dealing order: with three winners and 2 cents
left over, one winner gets both. Reading the cents as decimal dollars, PokerKit's loader splits
the pot exactly (0.67 cents each). The odd-chip allowance of less than one unit assumed one
leftover unit; with $w$ winners the remainder can be up to $w - 1$ units. A post-hoc re-check
confirms that this is the only cause: each of the 73 hands on which the loader disagreed at all
(the 5 failures and 68 one-unit odd chips) was written again in whole units, and then the loader
agrees exactly on all 73. The verdict stays *does not hold*; the explanation is labeled post
hoc. (Whether PokerKit's rule matches house rules is a question for Phase 4's home game, which
will have to record how its dealer splits odd chips.)

**Other results.** The engine, unmodified PokerKit, JSON and PHH agree on every hand at every
structure: 1,104 hands had a post that put a short stack all-in, 1,250 a big-blind ante and
3,750 a straddle. Part B's simulated hands have exactly the feature counts Step 32 published
for its tables, so they are the same tables. The WSOP hands are five-handed with a big-blind
ante of 1.5 bb; the 76 files still rejected are other games (stud, razz, draw, Omaha, short-deck and
fixed-limit hold'em).
Separately from the experiment, Phase 1 was re-simulated from scratch with schema v2 (828 s):
every results file and summary is identical to the committed ones.

**Mistakes.**

* The pre-registered odd-chip allowance was too narrow (above).
* The record of the version 1 files hashed their gzip bytes, which carry a timestamp, so two
  identical runs recorded different hashes. The hash now covers the decompressed content.

**Limitations.** The schema holds two kinds of ante, not arbitrary per-seat antes. A straddle
is described by the table, as if it were posted every hand; in a live game a straddle is a
choice each hand, which a capture app can record by giving that hand its own `TableConfig`.
Profiling's situation features still flag only the two blinds as "posted a blind", not a
straddler. Dead blinds (a player returning to the table posts a missed blind) are not modeled.

**Verification.**

* `tests/test_schema_v2.py`: the version 1 fixture reads as the hands the current code makes;
  `migrate_jsonl`; each version's keys checked strictly; the new table rules are canonical and
  checked; `expected_forced_posts` by hand (heads-up, short stacks, a button straddle); a
  changed table is caught by the forced-post check; hands at eight v2 tables replay through
  unmodified PokerKit; a straddle is not voluntary.
* `tests/test_phh_import.py`: a big-blind-ante hand imports, replays and round-trips; minimum
  bets other than the big blind import; button antes and big-blind antes with trimming are
  rejected; the round-trip fixture includes v2 tables.
* `tests/test_schema_experiment.py`: each structure's tables have it; a round-trip task runs
  every check; the loader check tells an odd chip from a real difference.
* Experiment: `python -m experiments.run_phase3 --seed 2026 --only schema` (about a minute on
  16 threads).

## Step 36: Opponents that adapt to the hero (P3.9)

**The question.** Every exploit so far was judged against a bot that keeps its strategy, which
is why every value could be exact: the hero's strategy against the bot's, in closed form (Steps
27 and 29). A person at the table does not keep still. Push very wide against someone who folds
too much and they start calling wider. The exploit itself changes the opponent, so its value can
only be measured by playing it out. How much of the exploitation gain survives that, and do the
safeguards built in Steps 29 and 30 (rank by a lower bound, fall back when the stability check
fails) help?

**Bots that adapt.** `games.push_fold_adaptive.AdaptiveBot` starts from a P3.2 bot (two widths
and a temperature) and watches the hero the way a player with a HUD would. It sees how often
the hero pushes as the small blind and how often it calls a push as the big blind, never its
cards. Every 25 hands it updates in four steps.

1. *Rates with a memory.* For pushes, keep a weighted count of events $x$ and of chances $n$.
   Every hand both decay by $d = 2^{-1/h}$, and a hand in which the hero was the small blind
   adds its chance and, if the hero pushed, its event:

   $$x_t = d\,x_{t-1} + c_t\,\mathbb{1}[\text{hero pushed}],\qquad n_t = d\,n_{t-1} + c_t,\qquad \hat r = x_t / n_t,$$

   with $c_t = 1$ when the hero had the chance and 0 otherwise. An event $h$ hands old counts
   half as much as a new one (the *half-life*). The counts start at $m = 10$ hands at the
   equilibrium's rate, a prior that keeps the first estimates sane. Calls are counted the same
   way over the pushes the hero faced.
2. *A model of the hero.* The hero is modeled as a soft threshold in the equity order, at the
   population's typical temperature, with the width that covers the share $\hat r$ of hands:
   the "HUD stats" model of Step 27, pointed the other way.
3. *Best response.* For each hand class, push if pushing beats folding against the modeled
   hero's calls, and call if calling beats folding against its pushes (`hand_gains`, Step 27).
   The target widths are the widths, at the bot's own temperature, that cover the same shares.
4. *A step.* $w \leftarrow w + \eta\,(w^\* - w)$ for each width the bot adapts.

Three kinds of adapter: *slow* ($\eta = 0.1$, $h = 200$), *fast* ($\eta = 0.5$, $h = 50$), and
*calls only* (fast, but only the call width moves: the player who calls wider after the hero
pushes a lot but keeps their own pushing habits). The bot stays inside the family the hero's
model assumes at every moment, since only its widths move. The model is wrong in one thing only:
it assumes the bot keeps still.

These updates are a version of *fictitious play* (Brown 1951): best-respond to the opponent's
observed frequencies. Robinson (1951) proved that in two-player zero-sum games, when both players
play this way, the frequencies converge to an equilibrium. Here only the bot does, and against an
equilibrium hero a simpler fact applies. A hand the equilibrium always pushes is worth pushing
against the equilibrium's calls, a hand it always folds is worth folding, and likewise for calls;
only the few hands it mixes are indifferent. So the best response to the equilibrium plays almost
exactly the equilibrium's range sizes (within 2 points, checked in the tests), and a bot that
best-responds to an equilibrium hero drifts toward equilibrium ranges itself. (It best-responds to
its soft-threshold model of the hero, not to the equilibrium itself, so it gets close rather than
exact.) That fact turns out to drive the results.

**Scoring a session.** `play_session` plays a hero and a bot in blocks of 25 hands. Before each
block the hero picks its strategy from a *redacted* log of the completed hands, with the bot's
cards present only where they were shown at a showdown. The block is dealt and played, and then
the bot observes and adapts. Each block is scored by the exact expected value of the two
strategies in force (`match_values`, both seats averaged), not by the chips that changed hands:

$$V = \frac{1}{N} \sum_b n_b\, u(\sigma^{\text{hero}}_b, \sigma^{\text{bot}}_b).$$

This is the expected result *given* the path the session took, so the luck of the cards
disappears while the randomness that matters stays: which actions were sampled, and therefore
how the bot adapted. (Replacing a random quantity by its conditional expectation never adds
variance; it is the Rao-Blackwell idea.) Every hero meets a fresh copy of the same bot with the
same random numbers. The deals and the uniform draws behind each action do not depend on the
strategies, so two heroes see the same cards and differ only through what they do (*common
random numbers*). A hero's *gain* is its value minus the equilibrium hero's against the same
bot.

**The heroes.** *Equilibrium* always. *Fit once*: watch 200 hands at equilibrium, then play
Step 29's advised response (best by the 90% lower bound, the equilibrium when the stability
check fails) for the rest of the session. *Refit*: the advised response recomputed every 50
hands on all hands so far. *Refit recent*: the same on the last 300 hands. *Bayes best
response*: recomputed every 50 hands, with no lower bound and no fallback. *Clairvoyant*: the
best response to the bot's current strategy, read from the bot itself. Nobody can play that;
it is the ceiling, and against a static bot it is Step 27's oracle.

**What theory says before the run.** A strategy with exploitability $\varepsilon$ can lose at
most $\varepsilon$ per hand (relative to the game's value) to an opponent who best-responds to it.
So a fixed exploit with high exploitability is a standing invitation to an adaptive opponent,
and a restricted Nash response (Step 22) caps the invitation. A hero that refits keeps
re-exploiting, but its model weighs every hand ever seen, so it reacts more slowly than a bot
with a 50-hand memory.

**Development.** The design was set on seeds 98 and 99, with 10 starting bots each. The first
pilot had only slow and fast adapters, and both drifted to near-equilibrium ranges against
every hero, which left almost nothing to exploit. The calls-only adapter was added so that one
kind keeps a leak the hero can find. The pilots also showed that against static bots about 2-7%
of refits fell back to equilibrium. The stability check runs 40 times per session against a
threshold calibrated for one look, so D1 allows up to 10%.

**The experiment** (`experiments/phase3_adaptive.py`, pre-registered at commit `10aff6e`). Five
replicates, each with Step 27's prior and its 30 standard test bots as starting points, so the
static control is Step 29's population. Every starting bot is played as each of the four kinds
against each of the six heroes: 3,600 sessions of 2,000 hands at 10 bb, and 72,600
recommendations, each with both model-fit checks. The checks are calibrated per replicate on 300
bots drawn from the prior (Step 30). A hero's gain is averaged over the 30 bots of a replicate,
with 95% t-intervals across the five replicates; "loses to" is the share of the 150 bots against
which its gain is negative.

| Prediction | Result |
|---|---|
| A1: against static bots, refitting keeps at least 85% of the clairvoyant's gain | holds: 94% |
| B1: against slow and fast adapters, the equilibrium hero's value is at most half its value against the same bots static | holds: 0.24 [0.21, 0.27] and 0.25 [0.22, 0.27] |
| B2: against them even the clairvoyant gains at most 0.01 bb/hand | holds: +0.0031 [+0.0026, +0.0036] and +0.0040 [+0.0035, +0.0044] |
| C1: against fast adapters the Bayes best response loses to equilibrium play | holds: −0.0049 [−0.0060, −0.0038] |
| C2: against slow adapters fitting once loses to equilibrium play | holds: −0.0503 [−0.0573, −0.0433] |
| C3: against slow and fast adapters refitting stays within 0.005 bb/hand of equilibrium play | holds: +0.0011 [+0.0006, +0.0016] and −0.0012 [−0.0015, −0.0008] |
| C4: against calls-only adapters refitting gains | holds: +0.0245 [+0.0228, +0.0263] |
| D1: refitting falls back on at least 30% of its refits from hand 1,000 on against slow and fast adapters, and at most 10% against static bots | holds: 64% [63%, 66%], 49% [47%, 51%] and 4% [3%, 4%] |

**Results** (bb/hand; the last four columns are gains over equilibrium play against the same
bots; intervals in the summary file):

| Bots | Equilibrium hero's value | Fit once | Refit | Bayes best response | Clairvoyant |
|---|---|---|---|---|---|
| Static | +0.086 | +0.0746 | +0.0823 | +0.0853 | +0.0880 |
| Slow adapters | +0.020 | −0.0503 | +0.0011 | −0.0018 | +0.0031 |
| Fast adapters | +0.021 | −0.0078 | −0.0012 | −0.0049 | +0.0040 |
| Calls only | +0.043 | +0.0151 | +0.0245 | +0.0254 | +0.0287 |

* *Adaptation takes the money off the table, for everyone.* Against slow and fast adapters the
  equilibrium hero wins a quarter of what it wins from the same bots static, and nothing is
  left to take: even perfect knowledge of the bot's current strategy gains under 0.005
  bb/hand, against +0.088 when the bots keep still. The reason is the one above. Measured post
  hoc against equilibrium play, the total distance of a bot's push and call widths from the
  equilibrium's range sizes falls from 0.43 at the start to 0.08 for slow and 0.15 for fast
  adapters by the end of the session, and 95% and 85% of the bots end closer.
* *What survives is the leak the bot does not fix.* Calls-only adapters fix their calling (the
  equilibrium hero's value halves, to +0.043) but push as before. Refitting gains +0.0245
  against them, 85% of the clairvoyant's +0.0287, and loses to 3% of them.
* *The safeguards hold the recommender at equilibrium level.* Against slow and fast adapters
  refitting gains +0.0011 and −0.0012 bb/hand. It loses to 31% and 79% of the bots, but by
  little: both intervals lie within 0.002 of zero. It gets there by giving up. Step 30's
  stability check reads the bot's drift as a change in its play, and from hand 1,000 on 64%
  (slow) and 49% (fast) of refits fall back to equilibrium, against 4% for static and calls-only
  bots. Against slow adapters the timeline is clear: the recommender profits while the bot still
  plays its old leaks (+0.013 in the first 250 hands), pays some back as the bot adapts (−0.006
  in the next 250, when 37% of refits fall back) and then sits at equilibrium play (60-66% of
  refits fall back from hand 500 on).
* *The costly mistakes are the ones without safeguards.* An exploit fitted once and kept gains
  −0.050 bb/hand against slow adapters (losing to 78% of them), and −0.067 per hand in the last
  250 hands: its strategy has exploitability 0.060 averaged over the session, and the bot moves
  toward the best response to it while the hero never looks again. The Bayes best response gains
  −0.0049 against fast adapters (losing to 98% of them). Its strategies are the most exploitable of the heroes anyone
  can play (0.012 bb/hand, against 0.003 for refitting), and the loss is largest at the start
  (−0.0135 in the first 250 hands) but lasts, −0.002 to −0.008 per 250 hands after that.
  Without adaptation the same heroes are fine: against static bots the best response earns
  97% of the clairvoyant's gain and fitting once 85%.
* *Static bots replay Step 29.* Refitting keeps 94% of the clairvoyant's gain and loses to none
  of the 150 bots. It falls back on 2-4% of refits per 250 hands, but over 40 refits 19% of the
  static bots see at least one fallback: the price of a threshold calibrated for one look and
  applied 40 times, as the pilots warned.
* *Post hoc: a shorter memory is not the fix.* Refitting on the last 300 hands instead of all
  hands changes the gain by +0.0010 [+0.0007, +0.0013] bb/hand against slow adapters, −0.0009
  [−0.0013, −0.0004] against fast and −0.0023 [−0.0030, −0.0016] against calls-only adapters
  (differences paired by replicate; −0.0007 [−0.0039, +0.0025] against static bots). On 300
  hands the stability check fails less often (6% of refits from hand 1,000 on against slow
  adapters instead of 64%, 22% instead of 49% against fast ones), so the window hero keeps
  exploiting: a little more profitably against slow adapters, a little less against fast ones,
  where it loses to 91% of the bots. Fewer hands presumably also mean wider posteriors and
  smaller exploits, which costs against calls-only bots, whose leak never moves. These are
  differences of 0.001-0.002 bb/hand. So the theory section's worry, that a model weighing every
  hand reacts too slowly, does not point to the fix: against the fastest bots a shorter memory
  did worse, not better.
* *Post hoc: what the example trajectories show.* For the first bot of each archetype in the
  first replicate, the fast adapters never settle: from hand 500 on their push widths still
  range from 45% to 96%, against equilibrium play and against a best response alike. What the
  best response changes is the hero's value. From hand 500 on it varies 1.9 times as much from
  block to block as equilibrium play's (median over the 15 adapters), and in 9 of the 15 it
  drops below zero in some block, to as low as −0.24 bb/hand. Equilibrium play cannot lose to
  any bot over both seats (the game is symmetric, so its value is zero), and its value never
  falls below +0.010. The slow nit shows a chase. The best response pushes every hand from hand
  50; the bot answers by calling 44% of hands (from 4%), and by hand 325 the hero's value has
  fallen to −0.091. The bot keeps widening its pushes, to 93%, the hero calls 47% in turn, and
  its value swings back to +0.082 before both settle near +0.010 from hand 1,000.

**What it means.** The safeguards built for a wrong model (Step 30) also cover an opponent who
adapts: the stability check notices the drift, and falling back gives back little. For a player
the lessons are plain. Re-check an exploit rather than fitting it once and keeping it. Against
opponents who adjust, the money is in what they do not adjust. And equilibrium play is not
merely safe here: against slow and fast adapters it is within 0.005 bb/hand of a hero who can
read the bot's strategy.

**Limitations.** The bots adapt by one fixed rule that sees only two frequencies, and they never
leave the family the hero's model assumes. No hero modeled the adaptation itself (weighting
recent hands in the likelihood, or best-responding to the bot's update rule). The game is
heads-up push/fold at one stack size, where the best response to the equilibrium nearly is the
equilibrium; whether adapters settle as close to equilibrium in deeper games is untested.

**Engineering.** The results record 19,097 s (5.3 hours) on 12 processes, but that is wall time:
the laptop was closed, and asleep, for most of the run (the pitfall of C.3). The development
pilots put the work at about 1.8 hours; the Phase 3 wrap-up measured 1.6 (Step 37). The results file is 18.7 MB, since it keeps every refit's
record. The figures were redrawn from it after the run: a legend moved off the data, the panels'
separate scales stated, and one color per meaning in the value figure (no number changed).

**Mistakes.**

* The write-up first reported the run's 5.3 hours of wall time as its run time, three times the
  pilots' estimate, although the laptop had been closed for most of it. C.3 already warned
  about exactly this; the owner caught it.
* Two explorer sentences claimed more than the data show. One said that against a best response
  an adapter's "ranges chase the hero's": in the example trajectories fast adapters wander as
  much against equilibrium play, and only the slow nit shows a chase. The other explained the best response's loss by the hero's long
  memory, which the post-hoc window comparison does not support. Both now state what was
  measured, computed from the results file.
* A figure legend covered the first point of a curve, and the value figure used one color for
  two meanings (a hero in one panel, a kind of bot in the other).

**Verification.**

* `tests/test_push_fold_adaptive.py` (6 tests): a static bot never changes; the rate estimates
  follow what the hero does (a hero that always pushes and never calls is estimated above 99%
  and below 1%); bots move toward the best response, and a calls-only bot moves its call width
  alone, by as much as a full adapter does; the best response to the equilibrium plays the
  equilibrium's range sizes within 2 points; sessions are scored by exact values block by block,
  and the hero sees only completed hands, with unshown classes hidden; two heroes facing copies
  of the same bot from the same seed get the same deals and differ only in their actions.
* `tests/test_adaptive_experiment.py` (4 tests): the refitting heroes decide on their schedule,
  and fitting once decides once and then keeps its strategy; a window limits the hands used; the
  clairvoyant reads the bot; a small run of the tasks aggregates into all eight predictions, with
  the equilibrium hero's gain exactly zero and the static bots' widths unmoved.
* Experiment: `python -m experiments.run_phase3 --seed 2026 --only adaptive` (1.6 hours on 12
  processes, measured in the Phase 3 wrap-up, Step 37; the pre-registered run's 5.3 hours of
  wall time include the laptop sleeping); smoke test with `--quick`.

## Step 37: Phase 3 wrap-up: one command, reproduced

Step 24 set the rule for ending a phase: rerun every experiment from one command and a clean
commit, and compare the new results with the committed ones. Phase 3 has eleven steps, written
over five days, and several later steps changed code that earlier steps use. The rerun answers
two questions: does the code still produce what the repository says it produced, and, where it
does not, why?

**What was built.** Two small tools for Step 24's protocol.

* `run_phase3 --out DIR` writes a run to another folder. Without it, a reproduction run
  overwrites the committed results in place, and every step after the first records
  `dirty: True`. Each step reads the dirty flag when it starts (Step 14), and by then the
  earlier steps have rewritten tracked files. With `--out` pointing at an untracked folder, a
  run from a clean checkout stays clean from first step to last. The explorer is rebuilt only
  from the default folder.
* [`experiments/compare_runs.py`](../experiments/compare_runs.py) turns Step 24's comparison
  script into a tested module. It walks two results folders and lists every difference outside
  provenance and timing:
  * JSON files field by field, floats exactly. It skips the top-level `meta` block (commit,
    timestamp, package versions) but requires `meta.config` to match. It skips wall-clock keys,
    matched as whole words after splitting the key at underscores (`seconds`, `elapsed`,
    `time`, `ms` and a few more). A substring match would have skipped `fallback_second_half`,
    the data behind one of Step 36's predictions, because it contains "second".
  * Markdown summaries line by line. It skips provenance lines, the cells of timing columns in
    tables, and times in milliseconds inside sentences ("6.5 ms each"). Only the time itself is
    masked, so a changed count on the same line is still reported.
  * PNG figures byte for byte, then pixel by pixel, because two encoders can write the same
    image as different bytes.

**The run.** `python -m experiments.run_phase3 --seed 2026 --out DIR` ran all eleven steps from
commit `98449f0`. It ran in a git worktree outside the synced OneDrive folder, so work in the
main checkout could not touch it.

**Results.**

* **Eight steps reproduced exactly**: every number, summary line and figure of pushfold,
  exploit, bridge, recommend, priors, sequence, schema and adaptive. That includes the two
  longest, the sequence models (PyTorch, five replicates) and Step 36's 18.7 MB of
  refit-by-refit records.
* **Three steps differ, each through a later change that was made on purpose:**
  * *import* (Step 32) now accepts 10,012 of the 10,088 public files instead of 10,001. Schema
    v2 (Step 35) can hold a big-blind ante, so the 11 WSOP hold'em hands that Step 32 rejected
    for that reason now import, as Step 35 reports. Both prediction verdicts are unchanged.
    PokerKit's loader still disagrees on 8 hands, every one a split pot's odd chip, and every
    accepted hand still survives export and import.
  * *fitcheck* (Step 30) and *confirm* (Step 31) keep every number, but some example
    explanations now begin "Recommended: ..." where they said "Model check failed: ... play the
    equilibrium". Step 31 made the stability check the recommender's only fallback by default,
    so a failed hand-order check now only warns that the intervals are unreliable. The examples
    use the default. fitcheck also reports the stability-only rule, which joined its list of
    rules when Step 31 was planned.
* **Rerun at their own commits, the three match.** To show that those changes are the whole
  story, the three steps were run again at the commits their results record (`3c1f25c`,
  `c4c15d0` and `7f50502`). Every JSON field and summary line is identical. The figures are
  not, and the reason is a gap in provenance. In Steps 30 and 32, the figure code was fixed after
  the run (a legend moved off the data, an axis un-inverted), and the figures were redrawn from
  the saved JSON in the results commit. So the committed figures come from newer code than the
  commit their JSON records. They match the figures drawn at `98449f0` exactly.

The committed results were kept, as in Step 24: they are the record of the pre-registered runs,
with clean provenance. The rerun counts as verification.

**How long it takes.** The run took 10.5 hours of wall time (14:01 to 00:31), and the laptop's
lid was closed for 6.5 of them. The Windows System log records each closure (see C.3), so the
time awake can be measured, not guessed:

| Step | This run, awake (s) | Committed record (s) |
|---|---|---|
| pushfold | 72 | 48 |
| exploit | 240 | 217 |
| bridge | 1,283 | 1,168 |
| recommend | 177 | 164 |
| fitcheck | 445 | 445 |
| confirm | 424 | 442 |
| import | 51 | 40 |
| priors | 841 | 529 |
| sequence | 4,919 (28,318 of wall time) | 13,657 |
| schema | 86 | 62 |
| adaptive | 5,868 | 19,097 |
| **all** | **14,406 (4.0 hours)** | |

* **A closed lid pauses the run almost entirely.** The sequence step's five replicates took
  979 and 985 s when nothing interrupted them, and 4,232, 5,786 and 16,306 s across three lid
  closures. Subtract each closure and they come back to between 930 and 1,000 s.
* **A screen turned off by the idle timeout did not slow it.** Through 35 minutes of that,
  each of adaptive's 12 worker processes used as many CPU seconds as wall seconds.
* **Two committed records are wall time across sleep too:** sequence's 13,657 s (3.8 hours;
  Step 34 already says most of it was a stall) and adaptive's 19,097 s (5.3 hours; Step 36).
  Measured awake, they take 1.4 and 1.6 hours.
* **Two differences are not explained.** Priors took 841 s against 529, with no standby
  recorded; the laptop was recharging from a 7% battery at the time, which some laptops
  throttle for. Pushfold and exploit ran during a screen-off standby that began a minute
  before the run.

**What it means.** A one-command reproduction tests the repository, not just the code. Results
are a snapshot taken at one commit, and the code moves on. Here every difference traced to a
change that was intended and written up where it was made, and none was a bug. The figure gap
is the lesson to carry forward: a figure redrawn after a run should say so in a place a reader
of the results will see. Here only the commit messages did.

**Mistakes.**

* The first check of whether the laptop had slept looked only for Kernel-Power events 42 and
  107, which Windows writes for S3 sleep and hibernation. This laptop uses Modern Standby,
  which writes events 506 and 507 instead, with a reason such as "Lid". The check found
  nothing, and a test run that took 20 minutes instead of about 2.5 was blamed on the low
  battery. The lid had been closed for 18 of those minutes.
* The first comparison script matched wall-clock keys as substrings, so it would have skipped
  `fallback_second_half`. The tool's own test caught it before the run.
* Times in milliseconds (`ms_per_hand`, "6.5 ms each") were not recognized as timing at first,
  so schema and import showed differences that were only timing. Recognizing them took one key
  word and one pattern, both tested.

**Verification.**

* `tests/test_compare_runs.py` (2 tests). The tool must skip provenance, timing keys, timing
  columns, milliseconds inside sentences, and a PNG written with a different compression
  level. It must report a float changed in its last bit, a changed config, a changed data key
  whose name contains "second", a changed table cell, a changed count on a line that also holds
  a time, a changed pixel, and an extra file.
* Reproduction: `python -m experiments.run_phase3 --seed 2026 --out DIR`, then
  `python -m experiments.compare_runs experiments/results/phase3 DIR`. It exits 1 if anything
  is listed.

## Step 38: Variance reduction with AIVAT, measured exactly (P2.11)

**The question.** Phase 5 will measure a win rate at a live home game. Step 28 showed what that
costs: at 10 bb, a ±0.05 bb/hand interval takes tens of thousands of hands by counting chips, and
the two remedies it measured fall short at a real table. Duplicate dealing needs a dealer who can
repeat every deal, and all-in EV covers only all-ins. AIVAT (Burch, Schmid, Moravcik, Morrill and
Bowling 2018) is the general method: a control variate at every step whose probabilities the
evaluator knows. This step asks three questions in Leduc and in push/fold, where everything can be
computed exactly. How much variance does AIVAT remove? How good must its value function be? And how
much is left when the evaluator sees only what the hero saw, so that folded cards stay hidden, as at
a home game?

**Control variates along a hand.** A hand is a path $h_0, h_1, \dots, h_T = z$ through the game
tree, and it pays the hero $u(z)$. After each step the evaluator knows some things and not
others. Call $S_t$ its *information class* after step $t$: the set of histories it cannot tell
apart (`observer_classes` groups the tree's nodes this way, from `Game.observer_key`). Some
steps are taken by an actor whose probabilities the evaluator knows: chance, and the hero, whose
strategy is ours. For any function $V$ of the class, the *value function*, such a step gets the
term

$$\delta_t = V(S_{t+1}) - \mathbb E\big[V(S_{t+1}) \mid S_t\big]
           = V(S_{t+1}) - \sum_{S'} P(S' \mid S_t)\, V(S'),$$

and the estimate subtracts all of them from the result:

$$\hat u = u(z) - \sum_{t \in K} \delta_t,$$

where $K$ holds the steps with known probabilities. The opponent's steps get no term: their
probabilities are exactly what we do not know. The hero's seat is a fair coin the evaluator
sees, so it is a chance step too: its term is the value of the seat dealt minus the average of
the two seats' values.

*Unbiased, whatever $V$ is.* Each term has mean zero given what came before,
$\mathbb E[\delta_t \mid S_t] = 0$, by construction. The running sums
$M_t = \sum_{s < t,\, s \in K} \delta_s$ therefore form a *martingale* with respect to the
evaluator's information: given $S_0, \dots, S_t$ (each class refines the one before), the
expected next value is the current one, and the first is 0. The condition is on the classes,
not on the full history: the public card's odds, for one, depend on the hero's hidden card. By the tower property,
$\mathbb E[\delta_t] = \mathbb E\big[\mathbb E[\delta_t \mid S_t]\big] = 0$, so
$\mathbb E[\hat u] = \mathbb E[u]$. Nothing in this asks $V$ to be good. A poor $V$ only fails
to remove variance, or adds some (see the results).

*Low variance, when $V$ is good.* Let $V^\*(S) = \mathbb E[u \mid S]$ be the true expected result
given what the evaluator knows, and suppose the result is a function of the last class (true
when the evaluator sees everything, and arranged by imaginary observations, below). Any result
is its prediction at the start plus every change of prediction along the way; the sum
telescopes:

$$u(z) = V^\*(S_0) + \sum_{t=0}^{T-1} \big[V^\*(S_{t+1}) - V^\*(S_t)\big].$$

At a known step $V^\*(S_t) = \mathbb E[V^\*(S_{t+1}) \mid S_t]$, so with $V = V^\*$ the bracket
*is* $\delta_t$ and the correction removes it exactly:

$$\hat u = V^\*(S_0) + \sum_{t \notin K} \big[V^\*(S_{t+1}) - V^\*(S_t)\big].$$

What is left is the prediction before the deal plus the changes at the opponent's decisions. The
changes of a martingale are uncorrelated, so the variance splits step by step,

$$\operatorname{Var}(\hat u) = \mathbb E\Big[\sum_{t \notin K} \big(V^\*(S_{t+1}) - V^\*(S_t)\big)^2\Big],$$

and with every step corrected it is zero: $\hat u = V^\*(S_0) = \mathbb E[u]$ on every hand. In
practice $V$ is the value of some strategy profile (the equilibrium's, say), not of the profile
being played, and the cancellation is partial. All-in EV (Step 28) is a special case, with terms
only at the cards dealt after a called all-in.

The evaluator computes $P(S' \mid S)$ from reach probabilities. Let $w(h)$ be chance's
probability of reaching $h$ times the hero's, leaving out the opponent's. When the opponent's
reach is the same at every history of a class, it cancels from every ratio, and
$P(S' \mid S) = W(S') / W(S)$ with $W(S) = \sum_{h \in S} w(h)$. That holds whenever the
evaluator sees the opponent's cards: a class then fixes the opponent's cards and every action,
which is all the opponent's reach depends on.

**Imaginary observations.** The hero's own cards are luck too. An evaluator who sees the
opponent's cards and every action, but not the hero's cards, can replace the result by its
average over every private hand the hero could have held, weighted by how likely chance was to
deal it and the hero to play the observed actions with it:

$$\bar u(S_T) = \frac{\sum_{z \in S_T} w(z)\, u(z)}{\sum_{z \in S_T} w(z)}.$$

The opponent's reach is the same for all of them (it cannot see the hero's cards), so these weights
are the exact posterior and $\bar u(S_T) = \mathbb E[u \mid S_T]$. That makes it unbiased (tower
property) and, as a conditional expectation, never more variable than $u$ (Rao-Blackwell, Step 36).
It removes the luck of the hero's own deal, given how the hero played. The value function is
averaged the same way: $V(S)$ is the $w$-weighted average of a value for each full history in $S$.
Chance terms, hero terms and imaginary observations together are what Burch et al. call AIVAT. The
evaluator needs two things a live game may not give it: the hero's strategy (for $w$ and the hero's
terms) and the opponent's cards after every hand.

**Why self-play at an equilibrium leaves nothing.** Burch et al.'s Table 1 evaluates a Leduc
equilibrium against itself. Here, with the equilibrium's own values (exact, since that is the
profile being played), AIVAT's standard deviation is exactly zero, and the decomposition above
says why. The only changes left are at the opponent's decisions. There the evaluator sees what
the opponent sees (its card, the public card and the actions) and averages over what it does
not (the hero's card), so $V^\*$ is the value of the opponent's information set. An equilibrium
strategy is a best response, so at every information set the play reaches, each action it plays
with positive probability has the same value; otherwise moving probability to the better action
would gain. Every change at an opponent decision is zero, every other step is corrected, and the
estimate is the same number on every hand. The paper's standard deviations were small but not
zero: 0.00643 chips/hand for AIVAT, and 0.00377 for the version that also corrects the
opponent's steps with its strategy known. With exact values that second version has no variance
for any profile (a test checks this), so the paper's values were presumably not exactly those of
the strategies played.

**Exact, not sampled.** A small game needs no sampled hands to measure an estimator. Its value on
a hand is fixed by the terminal history: the terms along the path and the class average are all
functions of $z$. So for the profile played with the hero in seat $s$, with reach probabilities
$\pi_s(z)$, the mean and second moment are sums over terminals (`terminal_moments`),

$$m_s = \sum_z \pi_s(z)\, x_s(z), \qquad q_s = \sum_z \pi_s(z)\, x_s(z)^2,$$

and with the seat a fair coin the variance per hand is $\tfrac12 (q_0 + q_1) - \mu^2$, where
$\mu = \tfrac12 (m_0 + m_1)$. One pass down the tree and one up give every estimator's value at
every terminal.

Duplicate dealing takes one more idea. Let $D$ be a deal: every chance outcome, here both private
cards and the public card. The first table plays it with the hero in seat 0, the second with the
hero in seat 1, and the unit is the pair's mean $Y = (X_1 + X_2)/2$. Each table draws its actions
with its own random numbers (Step 23), so given the deal the two results are independent. Write
$m_s(D)$ and $q_s(D)$ for seat $s$'s first and second moments given the deal; `deal_moments`
computes them for every deal, with its probability $p_D$. The *law of total variance* splits the
pair's variance into the spread within a deal and the spread between deals:

$$\operatorname{Var}(Y) = \mathbb E_D\big[\operatorname{Var}(Y \mid D)\big]
  + \operatorname{Var}_D\big(\mathbb E[Y \mid D]\big).$$

Independence given the deal makes the first part a sum of the two tables' variances,
$\operatorname{Var}(Y \mid D) = \tfrac14 \big[(q_0 - m_0^2) + (q_1 - m_1^2)\big]$ (the argument
$D$ dropped), and $\mathbb E[Y \mid D] = \tfrac12 (m_0 + m_1)$, so

$$\operatorname{Var}(Y) = \sum_D p_D \Big[\tfrac14 \big(q_0 - m_0^2 + q_1 - m_1^2\big)
  + \tfrac14 (m_0 + m_1)^2\Big] - \mu^2.$$

A pair costs two hands: $n$ hands make $n/2$ pairs, whose mean has variance
$\operatorname{Var}(Y) / (n/2)$, so the variance per hand is $2 \operatorname{Var}(Y)$. The same
algebra shows where Step 28's correlation $\rho$ comes from: with the tables independent given
the deal, $\operatorname{Cov}(X_1, X_2) = \operatorname{Cov}\big(m_0(D), m_1(D)\big)$. Pairing
cancels only the luck the deal fixes. A hand that ends before the public card fits every public
card, so `deal_moments` forces each public card in turn and weights it by its probability given
the private cards (Leduc has $6 \times 5 \times 4 = 120$ deals). It supports one later chance node
per path: Leduc's public card fits, hold'em's flop, turn and river would not.

**The hero's view.** At a home game where folded cards stay hidden, the evaluator knows what the
hero knows: its own cards, the actions and the public card. Its classes now group histories that
differ in the *opponent's* cards, and the posterior over those depends on the opponent's
strategy, which is unknown. So $P(S' \mid S)$ is known only at steps whose probabilities do not
depend on the opponent's hidden card. `known_transitions` finds them:

* the hero's own decisions;
* a chance step at which the opponent has not yet acted anywhere in the class (its reach is 1
  everywhere, so the posterior is chance's alone);
* a chance step that leads to each next class with the same probability from every history of
  its class.

In Leduc that leaves the hero's decisions and the private deal. The public card does not
qualify: it is dealt after the opponent has acted, and its probabilities depend on the
opponent's card (if the hero holds no king, a king comes with probability 1/4 when the opponent
holds one and 2/4 when it does not). Imaginary observations are out too, since the average
would need the opponent's reach. The *hero's view* estimator corrects only the known steps. Its
$V$ averages over the opponent's possible cards with the reach of the value function's own
profile, a model of the opponent; any weights keep it unbiased.

The *approximate hero's view* also corrects the public card, with probabilities computed as if
the opponent's card were uniform over the cards the hero has not seen. The opponent's actions
say something about its card, so those probabilities are wrong, and the terms no longer have
mean zero. In a small game the bias can be computed exactly; at a real table it cannot.

Neither version uses cards shown at a showdown: `observer_key` hides the opponent's cards in
every state. Whether a hand reaches a showdown is decided after the steps being corrected, so a
term that used those cards only on the hands that showed them would no longer have mean zero.
(This step does not test other ways of using them.)

**The experiment** (`experiments/phase2_aivat.py`, pre-registered at commit `695cb26`).

* *Leduc.* 50 bots, 10 from each of Phase 2's five archetypes (Step 21), drawn from the seed.
  Two heroes: the equilibrium (the LP of Step 18), and an exploiting hero, the restricted Nash
  response with $p = 0.9$ to the bot's true policy (Step 22). Also the equilibrium in self-play,
  the setting of Burch et al.'s Table 1.
* *Value functions.* The self-play values of a profile: the exact equilibrium, uniform play, and
  CFR+'s average strategy after 1, 3, 10, 30, 100, 300 and 1,000 iterations (Step 19). Also the
  *oracle*: the true values of this hero against this bot, which nobody has in practice (the
  ceiling). Results use the equilibrium's values unless stated.
* *Metrics*, all exact: the bias; the *ratio*, the plain result's variance per hand over the
  estimator's, which is how many times fewer hands the estimator needs for the same interval (a
  duplicate pair counts as two hands); and the hands needed for a 95% interval of ±0.1 chips,
  $n = \lceil (1.96\,\sigma / 0.1)^2 \rceil$.
* *A sampled check.* For the first 2 bots of each archetype and both heroes, 20,000 duplicate
  deals were played hand by hand. Each hand was recorded as its edge labels and replayed to its
  terminal with `GameTree.follow`, and every estimator was scored on those hands. Each sample
  mean is compared with that estimator's exact mean, $z = (\bar x - m) / \widehat{SE}$, so the
  check tests the exact moments, including a biased estimator's bias.
* *Push/fold.* P3.3's 33 matchups (5, 10 and 20 bb; Nash and best-response heroes; the five
  archetypes at their mean parameters, and Nash), in the abstract game of Step 26, whose plain
  result is P3.3's all-in EV. Hands needed are for ±0.05 bb. P3.3 measured all-in EV and
  duplicate + all-in EV on 100,000 real duplicate deals per matchup, which the exact numbers
  must reproduce.

The nine estimators:

| Estimator | Terms at | Averages over the hero's cards | The evaluator needs |
|---|---|---|---|
| Chips (the plain result) | nothing | no | nothing |
| Duplicate dealing | nothing; each deal played in both seats | no | a dealer who repeats deals |
| Chance terms only (MIVAT) | chance steps, valued at full histories | no | every hand's cards |
| Chance terms + imaginary observations | chance steps | yes | the opponent's cards; the hero's strategy |
| AIVAT | chance and hero steps | yes | the opponent's cards; the hero's strategy |
| AIVAT + duplicate dealing | as AIVAT, on duplicate deals | yes | both |
| AIVAT, opponent's strategy known too | every step | yes | also the opponent's strategy (a reference) |
| Hero's view | the hero's deal and decisions | no | what the hero saw; its strategy |
| Hero's view + public-card terms | also the public card, as if the opponent's card were uniform | no | the same; biased |

**Development.** The design and thresholds were set on seeds 98 and 99, on Leduc. There AIVAT
gave 26× for the equilibrium hero and 4.4-4.5× for the exploiting hero (medians), the unbiased
hero's view 1.04-1.05×, and push/fold was computed only at 3 bb, the quick configuration's stack
(AIVAT 13× for the Nash hero). The stacks of the real run, 5, 10 and 20 bb, were computed for the
first time in it. Two changes made in development are disclosed in the plan: the sampled check
first compared every estimator with the true value (see Mistakes), and an estimator with no
variance left is reported as such ("no variance", 0 hands).

**The run.** `python -m experiments.run_phase2 --seed 2026 --only aivat`, on 2026-10-29 from a
clean tree at `695cb26`.

| Prediction | Result | Verdict |
|---|---|---|
| A1: every estimator but the approximate hero's view is unbiased (bias below $10^{-9}$), in every matchup and with every value function | largest absolute bias $1.4 \times 10^{-13}$ | holds |
| A2: sampled hands agree with the exact moments: per estimator, no z beyond ±4, mean-z t-test p ≥ 0.01, median sampled/exact SD in [0.97, 1.03] | largest absolute z 3.42; smallest p 0.31; SD ratios 0.999-1.006 | holds |
| B1: AIVAT, equilibrium hero: median ratio ≥ 15× and every bot ≥ 5× | median 27×, lowest 10× | holds |
| B2: AIVAT, exploiting hero: median ratio between 3× and 8× | 4.30× | holds |
| B3: oracle values beat equilibrium values for the equilibrium hero by ≥ 1.4× (ratio of the median ratios) | 1.66× (45× against 27×) | holds |
| C1: hero's view: median ratio < 1.15× for both heroes | 1.05× and 1.04× | holds |
| C2: approximate hero's view: median ratio in [1.4×, 2.5×] for both heroes, and its bias reaches at least 0.02 chips/hand against some bot (exploiting hero) | 1.85× and 1.54×; bias up to 0.058 | holds |
| D1: uniform-play values still give AIVAT a median ratio ≥ 4× (equilibrium hero) | 9.12× | holds |
| D2: from 30 CFR+ iterations on, AIVAT's median ratio is within 20% of the equilibrium values' (equilibrium hero) | 29×, 30×, 30×, 30× against 27× | holds |
| E1: duplicate dealing adds < 25% on top of AIVAT (equilibrium hero) | 1.02× | holds |
| F1: push/fold: exact SDs of all-in EV and duplicate + all-in EV within 3% of P3.3's sampled SDs in every matchup | 66 estimates, sampled/exact 0.957 to 1.022 | **does not hold** |
| F2: push/fold, Nash hero: AIVAT needs ≥ 4× fewer hands than all-in EV at every stack (median over bots) | 29× at 5 bb, 70× at 10 bb, 22× at 20 bb | holds |
| F3: push/fold: hero's view gains < 1.3× over all-in EV at every stack, both heroes | 1.14× to 1.20× | holds |

Twelve of the thirteen hold. F1 is taken up after the results.

**Results: Leduc self-play** (SD per hand, chips):

| Estimator | Here (exact) | Burch et al., Table 1 |
|---|---|---|
| Chips (the plain result) | 3.497 | 3.513 |
| Duplicate dealing | 2.070 | |
| Chance terms only (MIVAT) | 1.463 | 2.327 |
| Chance terms + imaginary observations | 1.933 | 1.928 |
| AIVAT | 0 | 0.00643 |
| AIVAT + duplicate dealing | 0 | |
| AIVAT, opponent's strategy known too | 0 | 0.00377 |
| Hero's view | 3.400 | |
| Hero's view + public-card terms (biased) | 2.449 (bias +0.006) | |

Chips and chance terms with imaginary observations agree with the paper to within 0.5%; AIVAT's
zeros are the argument above. Only the chance-terms-only estimator departs, 1.463 against
2.327, so it is probably not quite the paper's MIVAT; this step did not find where they differ.
The hero's view, even with exact values, takes the SD only from 3.497 to 3.400: what it cannot
correct (the opponent's card, its decisions as the hero sees them, and the public card) carries
most of the luck.

**Results: Leduc bots.** Ratios against the plain result, median [interquartile range] over the
50 bots:

| Estimator | Equilibrium hero | Exploiting hero | Equilibrium hero, oracle values | Exploiting hero, oracle values |
|---|---|---|---|---|
| Duplicate dealing | 2.05× [1.82, 2.32] | 2.95× [2.11, 3.50] | 2.05× | 2.95× |
| Chance terms only (MIVAT) | 3.62× [3.28, 3.92] | 3.40× [2.79, 3.78] | 4.58× | 7.04× |
| Chance terms + imaginary observations | 2.26× [1.90, 2.88] | 1.86× [1.76, 2.22] | 2.28× | 2.29× |
| AIVAT | 27× [19, 32] | 4.30× [4.00, 4.65] | 45× [35, 57] | 9.34× [7.06, 12] |
| AIVAT + duplicate dealing | 28× [19, 33] | 5.33× [4.62, 5.80] | 45× | 9.34× |
| AIVAT, opponent's strategy known too | 52× [36, 61] | 5.28× [4.43, 7.33] | no variance | no variance |
| Hero's view | 1.05× [1.05, 1.05] | 1.04× [1.04, 1.05] | 1.09× | 1.05× |
| Hero's view + public-card terms (biased) | 1.85× [1.74, 1.94] | 1.54× [1.35, 1.74] | 2.31× | 1.77× |

* *With every hand's cards shown, AIVAT removes most of the luck for an equilibrium hero.* A
  ±0.1 chips/hand interval needs a median of 282 hands instead of a median of 6,332 (separate
  medians over the bots, so their quotient is not the median ratio), and no bot gives less
  than 10×. Calling stations give the least (post hoc: an archetype median of 11×, against
  20-33× for the other four).
* *For an exploiting hero it removes much less*: 4.30× (1,907 hands instead of 8,074). With
  the oracle's exact values, what remains is exactly the variance of the opponent's decisions
  (the decomposition above), so the medians say how much of the plain result's variance those
  decisions carry: about 1/45 against the equilibrium hero and 1/9.3 against the exploiting
  hero. Correcting the opponent's steps as well removes the rest, as the "no variance" column
  shows (post hoc reading).
* *The value function matters, but a poor one still helps* (curve below). Oracle values beat the
  equilibrium's by 1.66× for the equilibrium hero.
* *Duplicate dealing adds little on top of AIVAT*: 1.02× for the equilibrium hero (E1), and
  1.24× for the exploiting hero (post hoc, the ratio of the two medians). With the oracle's values it adds exactly nothing: AIVAT and AIVAT + duplicate
  have the same variance in every matchup. Post hoc, the reason: with exact values the expected
  estimate given the deal is the same for every deal (the deal is corrected exactly, and the
  opponent's changes have mean zero whatever the deal), so the between-deal part of the
  variance is zero and pairing has nothing to cancel. With the equilibrium's values duplicate
  dealing is sometimes worse than AIVAT alone, for 5 of the 50 bots with the equilibrium hero
  and 2 with the exploiting hero (post hoc, from `aivat.json`): when the two seats' remaining
  errors are positively correlated, $\sigma^2 (1 + \rho)$ exceeds $\sigma^2$.
* *The hero's view gains 4-5%* (1.05× and 1.04×; 6,036 hands instead of 6,332 for the
  equilibrium hero). Even the oracle's values give only 1.09×. With uniform-play values it is
  worse than the plain result against 38% of the bots for the equilibrium hero (as low as
  0.96×) and 20% for the exploiting hero (post hoc, from `aivat.json`): unbiased does not mean
  helpful.
* *The approximate hero's view gains more, 1.85× and 1.54×, and is biased.* Its largest bias
  over the bots is 0.019 chips/hand for the equilibrium hero and 0.058 for the exploiting hero
  with the equilibrium's values; with the oracle's values, 0.040 and 0.21. For scale (post
  hoc), the exploiting hero's exact win rates range from 0.51 to 1.66 chips/hand (median 1.07),
  and 0.058 is more than half the ±0.1 interval's half-width.
* *The sampled check passes.* No z-score reaches 4, every per-estimator t-test has p ≥ 0.31,
  and the sampled SDs match the exact ones (medians 0.999-1.006). The largest z, 3.42, sits in
  one matchup, the exploiting hero against one calling station (bot `b005`), where all nine
  estimators have z from 1.99 to 3.42: they score the same hands, so a sample that runs high
  runs high for all of them. That matchup holds 9 of the 16 z-scores beyond 1.96, out of 180
  (post hoc, from `aivat.json`).

**How good must the value function be?** Median AIVAT ratio by the profile the values come from
(exploitability in chips/hand; CFR+'s average after one iteration is uniform play):

| Values from | Exploitability | Equilibrium hero | Exploiting hero |
|---|---|---|---|
| Uniform play (CFR+, 1 iteration) | 2.3736 | 9.12× | 3.95× |
| CFR+, 3 iterations | 1.7935 | 8.52× | 4.11× |
| CFR+, 10 | 0.6104 | 13× | 6.18× |
| CFR+, 30 | 0.0818 | 29× | 5.09× |
| CFR+, 100 | 0.0134 | 30× | 4.65× |
| CFR+, 300 | 0.0023 | 30× | 4.52× |
| CFR+, 1,000 | 0.0003 | 30× | 4.37× |
| Equilibrium (LP) | 0 | 27× | 4.30× |

For the equilibrium hero, values from 30 CFR+ iterations are as good as the exact equilibrium's,
and slightly better (29-30× against 27×). For the exploiting hero the curve is not monotone: it
peaks at 10 iterations (6.18×) and falls toward the LP's 4.30×. Both fit one reading (post hoc,
not a test): the best value function is the true one for this hero against this bot (the
oracle), and closeness to an equilibrium is only a proxy for it, a poor one for a hero who is
not playing the equilibrium. The hero's view stays at 1.00-1.06× with every value function.

**Results: push/fold.** Hands for a ±0.05 bb/hand interval, Nash hero, median over the five
archetype bots (median ratio to all-in EV in parentheses):

| Stack | All-in EV (the plain result) | Duplicate + all-in EV | AIVAT | AIVAT + duplicate | Hero's view |
|---|---|---|---|---|---|
| 5 bb | 1,737 | 578 (3.01×) | 60 (29×) | 53 (33×) | 1,508 (1.14×) |
| 10 bb | 4,329 | 1,209 (3.26×) | 45 (70×) | 47 (67×) | 3,401 (1.17×) |
| 20 bb | 10,796 | 4,260 (2.25×) | 206 (22×) | 215 (25×) | 8,654 (1.16×) |

* *AIVAT works on top of all-in EV.* For the Nash hero it needs 22-70× fewer hands, for the
  best-response hero 13× (5 bb), 12× (10 bb) and 6.95× (20 bb). The hero's view gains 1.14-1.20×
  for both heroes.
* *Duplicate dealing on top of AIVAT helps little and sometimes hurts:* at 10 bb its median is
  slightly worse than AIVAT's alone. Post hoc, from `aivat.json`: with the equilibrium's values
  it adds variance in 8 of the 33 matchups (up to 14%, Nash against the nit at 5 bb), and with
  the oracle's values it changes nothing in any.
* *The exact numbers agree with P3.3's within sampling noise* (post hoc, below). Over the 66
  estimates P3.3 sampled, sampled/exact SD ratios run from 0.957 to 1.022. Step 28 quoted
  medians at 10 bb over both heroes of 4,339 hands for all-in EV and 1,572 for duplicate +
  all-in EV; the exact medians over the same bot matchups are 4,354.5 and 1,571.5 (post hoc).

**F1, the failed prediction.** F1 predicted that every exact SD of all-in EV and of duplicate +
all-in EV would be within 3% of P3.3's sampled SD. One of the 66 is not: at 5 bb, Nash against
Nash, duplicate + all-in EV, P3.3's sampled SD is 0.957 of the exact 0.0637 bb/hand (7 hands for
±0.05 bb exactly, 6 by P3.3's sample). The prediction fails as written.

*Post hoc: how noisy is a sampled SD?* (Computed after the run.) For a mean, 100,000 pairs feel
like plenty. For a standard deviation, it depends on the tails. With *kurtosis*
$\kappa = \mathbb E[(X - \mu)^4] / \sigma^4$, the sample variance $s^2$ of $n$ values has

$$\operatorname{Var}(s^2) \approx \frac{(\kappa - 1)\,\sigma^4}{n}, \qquad
  \frac{\operatorname{SE}(s)}{\sigma} \approx \sqrt{\frac{\kappa - 1}{4n}},$$

the second by the delta method: $s = \sqrt{s^2}$, so a small change in $s^2$ moves $s$ by
$1/(2\sigma)$ times as much. For normal data $\kappa = 3$ and the relative error is
$1/\sqrt{2n}$: 0.22% at $n = 100{,}000$, which puts a ±3% band more than 13 standard errors
away. This matchup is far from normal. Both players use the equilibrium, so the two tables of a
pair cancel exactly (Step 28) unless a mixed decision goes differently at the two, which happens
in only 0.58% of pairs. The variance sits in those rare pairs: kurtosis 475, so the
sampled SD's relative standard error is $\sqrt{(\kappa - 1) / (4n)} = 0.034$ and the miss is
$z = -1.25$. Drawing P3.3's sample 4,000 times from the exact pair distribution, 95% of
the sampled/exact ratios fall between 0.934 and 1.067, 10.5% are at or below the
observed 0.957, and a ±3% band is missed 39% of the time. For the same estimate at
10 bb, which passed at 1.022, it is missed 62% of the time. (At 20 bb the
Nash-against-Nash duplicate has no variance at all.) Treating the two as independent, F1 as
written had about a 23% chance of holding even with the exact model right.

P3.3's tables also meet the exact model's assumption: `duplicate_match` draws a separate action
seed for each table, so the two plays are independent given the deal, and no systematic gap is
expected. Scored with this kurtosis-aware standard error, the 65 estimates with variance look
like noise: the largest |z| is 2.92 (20 bb, Nash against the TAG, duplicate), 5% lie beyond
1.96, and the z-scores have SD 1.03 and mean -0.28. Were they independent, that mean would be
2.2 standard errors below zero; estimates in one matchup share deals, so they are not, and how
much that matters was not computed. Most estimates were never at risk: the median relative standard error is
0.43%, and only the two Nash-against-Nash duplicates have a 95% sampling interval wider
than F1's band. The miss is what sampling noise predicts for that estimate; the threshold, not
the exact model, was too tight. The verdict stays *does not hold*. The check is
`experiments/phase2_aivat_posthoc.py` (`python -m experiments.phase2_aivat_posthoc --seed 2026`,
written after the run; results in `aivat_posthoc.json`), and the explorer's "Post hoc: was F1's
miss sampling noise?" shows every estimate with its sampling interval. It simplifies one thing:
P3.3's hero alternates seats rather than tossing a coin, so for all-in EV its sample is
stratified and the standard error above slightly overstates the noise; duplicate pairs play both
seats and are unaffected.

**What it means.** The question was what a live game can expect, and the answer has two parts
that are far apart.

* *If every hand's cards are shown afterwards* and the hero's strategy is known, AIVAT needs 27×
  fewer hands than the plain result for an equilibrium hero in Leduc, 4.3× fewer for an
  exploiting hero, and 22-70× fewer for the Nash hero in push/fold, on top of all-in EV. A poor
  value function keeps much of that: uniform play's values still give 9.12×.
* *If folded cards stay hidden*, the unbiased hero's view gains 1.04-1.20× (Leduc and push/fold,
  both heroes): nearly as many hands as counting chips. The biased version gains 1.5-1.9× in
  Leduc with the equilibrium's values, at a bias a live game cannot measure; here it reached
  0.058 chips/hand with those values and 0.21 with the true ones.

Duplicate dealing is impossible at a real table, and on top of AIVAT it adds little. So what a
live evaluation can claim depends mostly on whether the opponents' cards are seen after every
hand. That is a question for the owner, not a decision this step can make: Phase 4 will record
the home game, so would the players table every hand, folded ones included, to the camera?

**Engineering.** Every estimator is a number per terminal, built with passes over the tree, and
every moment is a dot product, so the whole step took 22 s (19 s inside the experiment) on 10
worker processes. The plan had expected minutes. After the run the Leduc figure was redrawn from
`aivat.json` with fixed figure code (below; no number changed).

**Mistakes.**

* F1's tolerance was a fixed 3% on sampled SDs, which ignores kurtosis (above). The prediction
  is reported as failed.
* The Leduc figure's rows were upside down. Its two panels share the y axis, and each called
  `invert_yaxis()`, which inverted the shared axis twice: the mistake of Step 32 (Part D),
  repeated. The axis is now inverted once.
* In development the sampled check compared every estimator's sample mean with the true value,
  so the biased estimator failed it as well as the bias prediction: one fault counted twice. The
  check now compares each sample with that estimator's own exact mean, bias included, and bias
  is judged exactly, by A1 and C2.

**Limitations.**

* AIVAT needs the hero's strategy, mixing included. A human hero who deviates from the advice
  biases the hero terms.
* Hold'em has no value function yet; `AIVATEvaluator` still raises without a correction.
* `deal_moments` supports one later chance node per path, and `known_transitions` is tested only
  on Leduc.
* The exploiting hero responds to the bot's true policy, not a fitted model, and the bots do not
  adapt.
* A2's nine t-tests have no multiple-testing correction; with the smallest p at 0.31, one could
  not change the verdict.
* The sampled check covers 10 of the 50 bots.
* Why the chance-terms-only estimator differs from the paper's MIVAT is not known.

**Verification.**

* `tests/test_aivat.py` (24 tests, on Kuhn and Leduc): unbiased for random value functions and
  random profiles, with chance terms, AIVAT, every step and the hero's view; AIVAT's and the
  hero's view's estimates do not depend on the opponent's strategy; self-play with equilibrium
  values has no variance, and exact values with every step corrected leave none, with the root
  value as the mean; the hero's-view corrections are equal at terminals the hero cannot tell
  apart; `known_transitions` in Leduc (the hero's decisions and the private deal, never the
  public card or the opponent's decisions); `deal_moments` against brute-force duplicate pairs
  in Kuhn, and in Leduc that its 120 deals have the right probabilities and their moments add
  up to the terminal moments; observer classes reject a mismatched game;
  `GameTree.follow`; `AIVATEvaluator` end to end on 6,000 sampled Leduc hands (its interval
  covers the exact value and is less than half as wide as the plain one).
* Experiment: `python -m experiments.run_phase2 --seed 2026 --only aivat` (22 s); smoke test with
  `--quick`.
* `tests/test_aivat_posthoc.py` (14 tests): the exact one-hand and pair distributions match the
  experiment's own moments and the head-to-head value; Nash-against-Nash pairs cancel whenever
  both tables play alike; the pair distribution matches brute-force duplicate pairs in Kuhn;
  the relative standard error of a sample SD matches simulation for normal and two-point data;
  the simulated ratios are centred on 1, reproducible, and the same drawn in blocks.
* Post hoc check: `python -m experiments.phase2_aivat_posthoc --seed 2026`.

## Step 39: Faster code, identical results

**The question.** Two steps take three quarters of a Phase 3 rerun: adaptive (Step 36) and
sequence (Step 34), 1.6 and 1.4 of the wrap-up run's 4.0 awake hours (Step 37). The owner asked
for faster code under one hard rule: every committed result must come out the same, bit for bit.
A speedup that moved a probability in its last digit would make every later comparison with the
committed results a judgment call; under the rule, `compare_runs` (Step 37) decides. Each changed
step is rerun with `python -m experiments.run_phase3 --seed 2026 --only <step> --out DIR` and
compared with `python -m experiments.compare_runs experiments/results/phase3 DIR <step>`, which
skips only provenance and wall-clock keys. Anything but "0 differences" sends the change back
(for two steps the reference is Step 37's rerun of the old code; see Results).

**Profile first.** As in Step 1, measurement came before any change (`cProfile` and timers). In
adaptive, the restricted Nash response LPs (HiGHS through SciPy's `linprog`, Step 22) took 74-75%
of the time spent on one bot in the full configuration (bot 0, against fast and static
adapters) and 82% in the quick configuration. In sequence, training took 61% of the step,
re-simulating the replicates with an empty cache about 31%, and prediction 6.6%. The profiles
gave a ranked plan; the owner approved it and chose the top three items.

**Amdahl's law.** If a fraction $f$ of a program's time is made $s$ times faster and the rest is
untouched, the time becomes

$$T' = T\Big[(1 - f) + \frac{f}{s}\Big], \qquad
  \frac{T}{T'} = \frac{1}{(1 - f) + f/s} \le \frac{1}{1 - f}.$$

The bound is the limit $s \to \infty$: the part removed entirely. Amdahl (1967) made the point
about parallel computers, where the part that cannot be split limits the gain however many
processors there are; the algebra holds for speeding up any part. For adaptive at full size,
with $f = 0.745$, LPs that cost nothing would still leave $1 - f = 0.255$ of the time: at most
3.9× faster (5.6× in the quick configuration, $f = 0.82$). In sequence, the change below
touches prediction only, 6.6% of the time, so the most it can save is 6.6% (1.07×). Both
bounds were computed post hoc, from the profiles taken before the changes. The law says where
to look before a change and, after it, whether a measured speedup can be the change's at all.

**1. Compute only the branch that is used.** Each refit of a P3.9 hero called `recommend`, which
builds Step 29's whole ranked report: the equilibrium, a restricted Nash response for each $p$
(one LP per seat and $p$: ten LPs with the default five values) and the best response, each
scored against every posterior draw. The hero kept one entry. Two cases never need the LPs:

* the *Bayes best-response* hero plays the best response to the posterior mean, in closed form
  (`hand_gains`, Step 26);
* when the stability check fails, the advice is the equilibrium, computed once in advance. Step
  36 counted fallbacks on 64% and 49% of the refitting hero's refits from hand 1,000 on, against
  slow and fast adapters.

`PushFoldRecommender.choose(data, rule=..., seed=...)` returns only the advice (a `Choice`) and
solves the LPs only when the candidates must be ranked. Computing a value only when it is needed
is *lazy evaluation*. Skipping work is exact when the skipped work is *pure* as far as later work
can tell: the ranking (and the population recommendation it computes once) draws only from
random generators it seeds itself and changes no state that a later result reads (the caches it
fills return what a fresh computation would), so leaving it out cannot change anything after
it. The work that the advice itself needs (the posterior with the given seed, the fit checks
with that seed extended by 1) runs in both paths, in the same order. Both paths call the same
functions (`_evidence`, then `_rank` when ranking), so they cannot drift apart.

**2. Reuse LP solutions.** *Memoization* (Michie 1968) stores a function's results under its
arguments and answers a repeated call from memory. It is exact when two things hold: the key
determines the input completely, and the function gives the same output for the same input.
`_solve` keeps, per process, the solutions of the last `SOLVED_SIZE = 2048` LPs, keyed on the
exact bytes of everything the LP is built from: SHA-1 digests of the sparse matrices $A$, $E$ and
$F$ (shape, dtypes, and the `data`, `indices` and `indptr` arrays that store a sparse matrix row
by row), the objective's float64 bytes, the weight $1 - p$ as float64 bytes, and a digest of the
data-biased response's extra rows $G$ (Step 22) when there are any. HiGHS is deterministic, so
the same bytes give the same solution, and a hit returns exactly what a solve would have.

* *A key may be stricter than the input, never looser.* $0.0$ and $-0.0$ are equal numbers with
  different bytes, and one matrix can be stored in two sparse layouts; either costs a miss, not a
  wrong answer. An objective rounded to 12 digits would be a looser key and could return the
  answer to a different LP.
* *Collisions.* Among $n$ different inputs, two share a SHA-1 digest with probability about
  $n^2 / 2^{161}$: about $2^{-121}$ for a million. (SHA-1 is no longer safe against collisions
  built on purpose, which do not arise here.)
* *What would break it:* a solver that is not a function of its input (a time limit, so the answer
  depends on the machine's speed; randomness without a fixed seed; a parallel sum whose order
  depends on which thread finishes first); another solver version or options; and changing a form's
  matrices or a stored solution in place. Options matter at this precision: in the profile, the same
  LPs solved with HiGHS's presolve turned off gave solutions up to $1.28 \times 10^{-11}$ away. The
  memo lives and dies with one process, so it never meets another version or other options; a cache
  on disk would need them in its key, as Step 14's simulation cache needs the code fingerprint.
  Every caller who asks for the same LP gets the same array, so one caller's edit would reach the
  next. The arrays are stored read-only, and an edit raises an error.

The memo is a *least-recently-used* (LRU) cache: an `OrderedDict` in which a hit moves its entry
to the end, a new entry goes at the end, and above 2,048 entries the first, the one used longest
ago, is dropped; each step costs $O(1)$. Recency is the usual bet: what was used recently tends to
be used again soon. The bound caps memory. A push/fold solution has 339 entries (169 classes × 2
actions, plus the empty sequence); with the objective in the key, an entry takes about 5.4 kB,
about 11 MB in all. (Since Step 40 a sweep is one entry holding a solution per $p$, and the
bound counts solutions.)

The same commit moves one computation out of a loop. The LP of Step 22 maximizes
$p\,(A y_{\text{fix}})^\top x + (1-p)\,f^\top q$, and $A y_{\text{fix}}$ does not depend on $p$.
`restricted_nash_sweep` computes the model's realization plan and $A y_{\text{fix}}$ once per seat
instead of once per $p$. The same routine on the same inputs gives the same bytes, and each
objective is then formed exactly as before.

**3. Keep embeddings between predictions.** A fitted `SequenceClassifier` predicts many sets of
windows of the same players: every window size and calibration in part A, then the transfer
checks (Step 34). Each call encoded every hand again. Now the classifier keeps each hand's
embedding between predictions, keyed by the identity of the `HandSequences` object and the exact
tuple of players in the block of 16 that was encoded together.

*Why the block, not the player.* Floating-point addition is not associative: every operation
rounds, so the order of a sum can change its last bit. In double precision
$(0.1 + 0.2) + 0.3 = 0.6000000000000001$, while $0.1 + (0.2 + 0.3) = 0.6$. A matrix product is
many such sums, and the libraries PyTorch calls choose how to split and order them from the
matrices' shapes and the number of threads (blocks sized for the processor's cache, vector lanes,
a share per thread). So a batched product is not guaranteed to equal, bit for bit, the same rows
computed in another batch. The profile found one such case: predicting every window size in one
call (the same blocks of players, but more windows in each product of the output layer) gave
probabilities that were not bit-identical to one call per size, while reusing embeddings from
the same blocks gave identical bytes. Whether a hand's embedding changes with its block was not
tested; keying by the block makes the question moot. The network computes in 32-bit floats, for
which one *ulp* (unit in the last place) just above 1 is $2^{-23} \approx 1.2 \times 10^{-7}$;
`compare_runs` compares floats exactly and would report a one-ulp change. Keyed by the block, a
reused embedding is always one computed in exactly the batch that would compute it again. The
thread count must stay fixed for the same reason (Step 34 already required it for reproducible
fits).

*Why identity, and what it assumes.* The LP memo reads the objective's bytes and hashes $G$ on every
call; each form's matrices are hashed once, when the form is built, which assumes they are not
changed in place afterwards. The embedding memo keys on the object instead. In CPython `id()` is the
object's address, which a new object can take once the old one is freed, so the classifier holds
every `HandSequences` it keeps embeddings for. An identity key cannot see a change of content, so
the module docstring states the preconditions: the weights change only through `fit` (which clears
the memo), the thread count stays fixed, and sequences are not changed in place. Nothing is freed
before the next `fit`, so repeated predictions should pass the same object (a new one only misses
and costs memory). A pickle or deep copy starts with an empty memo (`__getstate__`): its keys are
the ids of the original's objects, which the copy does not hold. The price is memory: about
0.27-0.38 GB more at the peak, from about 4.1 GB.

**How it was done.** The changes were made on a branch in a git worktree outside OneDrive, with
separate worktrees for profiling, the test runs and the reruns. A reviewer subagent read each
diff before its commit. None of its findings changed a result in the experiments (no caller
modifies the returned arrays); its fixes are under Mistakes, plus documentation (`falls_back`,
the memo's preconditions). The suite went
from 444 to 457 tests, all passing. The reruns ran one at a time. Commits: `90ae418` (`choose`,
the LP memo and the sweep), `90f18f9` (their tests), `34eea0e` (the embedding memo) and
`eea6815` (its tests).

**Results.** Both changed steps reproduced the committed results exactly: `compare_runs` reported
"adaptive: 0 differences" and "sequence: 0 differences" ("total: 0 differences outside provenance
and timing"). The other steps that call the changed code were rerun the same way. Phase 3's exploit
and recommend and Phase 2's exploit and aivat match the committed results (aivat's two post hoc
files come from a separate script, so the rerun does not have them). Fitcheck and confirm match
Step 37's rerun of the old code exactly; against the committed results they show the same 53 and
23 differences that Step 37 found and explained.

| Measurement | Old code | New code |
|---|---|---|
| adaptive, quick configuration, one after the other | 54 s | 28 s (48% less) |
| sequence, quick configuration, alternating, 3 rounds (the first with an empty simulation cache; post hoc) | 40.1 s, then 21.2 and 21.0 s | 37.6 s, then 18.5 and 18.6 s (rounds 2-3: about 12% less) |
| sequence: 40 `predict_proba` calls on a real replicate's pools (1 training epoch, 8 threads), identical bytes | GRU model 29.3 s, counters network 4.4 s | 2.5 s and 1.2 s |
| adaptive, seed 2026 | 5,868 s awake (Step 37's run) | 2,492 s (58% less) |
| sequence, seed 2026 | 4,919 s awake (Step 37's run) | about 3,230 s awake (34% less; mostly not the change, below) |

The first three rows compare old and new code in one session; the last two compare runs on
different days.

**Run times, awake only** (C.3).

* The sequence run took 13,032 s of wall time. For about 9,800 s of it (about 13:04 to 15:47) its
  process was suspended at the owner's request (`NtSuspendProcess`, then `NtResumeProcess`). A
  suspended process, like a laptop with its lid closed, gets no processor time while the clock
  runs, so that interval is subtracted: about 3,230 s awake.
* Later in that run the laptop entered Modern Standby with the reason "Idle Timeout" (screen
  off). Step 37 found that this standby does not slow a step's workers, and here replicates 2-5
  took 628-648 s each.
* The adaptive run spent most of its time (about 37 of its 42 minutes) in the same idle-timeout
  standby. As in Step 37 that counts as awake, so its 2,492 s compare directly with Step 37's
  5,868 s.
* Apart from the three alternating quick-configuration rounds of the sequence step, every time
  here is from a single run; none has a confidence interval.

**Against Amdahl's bound** (post hoc: computed for this step, after the runs).

*sequence's full-step gain is mostly not the change.* The memo can save at most 6.6% of the step;
the step took 34% less time (1.52×). Training, which the change does not touch (`fit` neither
fills nor reads the memo, and the training code is unchanged since Step 34's run), took 27-39%
less time, fit by fit (seconds per replicate):

| Fit | Committed run | This run |
|---|---|---|
| Sequence model (GRU) | 201-216 | 145-148 |
| Order dropped | 42-46 | 28-29 |
| Counters network | 22.5-24 | 15.2-15.7 |
| Part C, order dropped | 209-217 | 136-156 |
| Part C, counters network | 115-119 | 76-81 |

Left out: a committed counters fit of 8,705.6 s, during which the laptop slept, and this run's
first GRU fit, which spans the suspension. In Step 37's two uninterrupted replicates training took
601 s of 979 and 985; here it took 403-424 s of 628-648. So about 190 of the roughly 350 s each
replicate saved came from training, on a machine that ran faster that day; of the rest, about
60 s is the memo's (the profile's 300 s over five replicates) and about 100 s other work, such
as simulation, that also ran faster. Why the machine was faster was not found. The old code's
two smaller part-A fits have run at this speed before: in Step 37's third replicate they took
29.7 and 16.3 s. Load from other programs, the laptop's power and thermal state, and the
screen being off for replicates 2-5 are candidates, not tested. The memo's own effect is the
same-session rows of the table: with old and new code alternating in one session, training
took the same time (8.5-8.7 s per run, all ten fits), and the new code saved 2.4-2.7 s per run,
about 12% of a warm run's 21 s. That is more than the full-size bound because the quick
configuration trains a smaller network for 8 epochs instead of 30, so prediction is a larger
share of its time. At full size the profile put the saving at about 300 s of the step, about
6%, inside the bound.

*adaptive stays inside it.* In the quick configuration, old and new run one after the other, the
new time is $28/54 = 0.519$ of the old; with that configuration's share, solving
$0.519 = 0.178 + 0.822/s$ gives $s \approx 2.4$: the part the changes act on took about 2.4× less
time ($f$ counts the LPs only; `choose` also skips realization plans and the ranking's scoring,
a few percent more). The full step's $2{,}492/5{,}868 = 0.425$ gives $s \approx 4.4$ with the
full-size share. The two need not agree: a full session is 2,000 hands with five values of $p$,
a quick one 300 hands with three, so the share of refits that fall back, and skip the LPs, may
differ (Step 36 measured it from hand 1,000 on); and the full-step run came from the same faster
day as the sequence run. The quick configuration's 48%, from one same-session pair, is the clean
number for that configuration. The two adaptive changes were measured together, and the code
does not count cache hits, so how much each gave is not known.

**What was left.**

* Transfer reads the test bots through `bots.inputs`, a different `HandSequences` object from part
  A's `inputs` for the same players, so the memo misses and those players are encoded once more
  per model. Fixed in [Step 40](#step-40-a-hot-start-parallel-replicates-and-a-split-test-suite):
  part A now predicts through the view that transfer reads, with the same bytes.
* Refitting a `SequenceClassifier` keeps its old `history` (new losses are appended to it) and,
  when the refit fits no per-size temperatures, its old `size_temperatures`. This is older than
  this step and has nothing to do with speed; the experiments fit a new classifier every time, so
  no result is affected.

**What it means.** An exact speedup is argued first and tested second. A cache is exact when its
key determines the input and the function is deterministic; skipping work is exact when the work
is pure. The tests check the facts those arguments rest on (the same bytes, the expensive call
made or not), and `compare_runs` checks each step whole. The profile said where the time was
worth attacking (three quarters of adaptive's time was LPs); Amdahl's law, applied after the
runs (post hoc), said which measured gain could be the change's. Adaptive's could; most of
sequence's 34% could not, and a comparison across days would have credited the memo with a
faster machine. A speedup is measured with old and new code in one session.

**Mistakes.**

* `choose` first returned the recommender's own equilibrium arrays on a fallback, so a caller
  that changed them would have changed every later recommendation. The reviewer caught it; it now
  returns copies.
* `restricted_nash_sweep` first iterated over its `p` values once per seat, so a generator, which
  can be read only once, would have failed. The reviewer caught it; it now makes a tuple first.
* A pickle or deep copy of a classifier would have carried the embedding memo, keyed by the ids
  of the original's objects. The reviewer caught it; `__getstate__` drops the memo.
* The first report of the sequence rerun credited the change with the step's 34%. Amdahl's bound,
  computed while drafting this step, says the change can give at most 6.6% (post hoc); the
  same-session comparison was run after that.

**Limitations.**

* The times are single runs on a laptop whose speed varied between days by more than some of the
  effects; the clean comparisons are the same-session ones.
* The adaptive LP share at full size comes from one bot's sessions, and LP cache hits are not
  counted.
* The embedding memo's preconditions are documented, not checked by the code.

**Verification.**

* `tests/test_recommend.py` (1 new test): `choose` equals the report's advice and its best
  response, for data whose advice is the equilibrium, a restricted Nash response and the best
  response; an unknown rule raises.
* `tests/test_fit_checks.py` (4 new tests): with checks that fail, only warn and pass, `choose`
  equals the report's advice with the same numbers; spies on `_solve` and `linprog` see no LP on a
  fallback or for the best response, and at least as many as seats times $p$ values for a
  ranking.
* `tests/test_leduc_exploit.py` (4 new tests): a repeated LP is answered from memory, also for a
  responder compiled again, with HiGHS's own read-only solution (a fresh solve gives the same
  bytes); LPs one ulp apart in an objective entry or the weight, with other $G$ values, for the
  other seat or for another tree's payoffs do not share an entry; least-recently-used eviction;
  the sweep equals the single responses bit for bit, with the memo warm and cold (replaced in
  Step 40, when later $p$ became hot-started; only the first $p$ is still bit for bit).
* `tests/test_sequence_model.py` (4 new tests): repeated predictions encode each block once and
  give the bytes of predictions from scratch; two `HandSequences` of the same players keep their
  own embeddings, and a new object never inherits a freed one's; a refit forgets the old
  embeddings; `fit` encodes its held-out players afresh every epoch and leaves nothing behind; a
  deep copy starts empty and predicts the same bytes. `tests/test_adaptive_experiment.py`'s
  window test now spies on `choose`.
* Reruns: `python -m experiments.run_phase3 --seed 2026 --only adaptive --out DIR`, then
  `python -m experiments.compare_runs experiments/results/phase3 DIR adaptive`; the same for
  sequence, exploit, recommend, fitcheck and confirm (fitcheck and confirm also against Step 37's
  rerun), and for Phase 2's exploit and aivat through `run_phase2.run(..., out_dir=DIR)`.
* Same-session timing: `python -m experiments.run_phase3 --quick --seed 2026 --only sequence
  --out DIR` from a worktree of the old commit (`f3d628c`) and one of the new (`eea6815`),
  alternating, with `compare_runs` between each pair.

## Step 40: A hot start, parallel replicates and a split test suite

**The question.** After Step 39 the owner asked for three more changes: "implement the hot
start, split the test suite and run two sequence replications at once. Make the rest of the
decisions." Two of them can keep every result bit for bit, as Step 39's changes did. The hot
start cannot. It starts each LP of a sweep from the previous one's optimal basis, and a
different path through the simplex method rounds differently and can stop at a different
optimal answer, so some numbers move in their last digits. The project's rule is that a change
which alters floating-point results needs the owner's sign-off, and asking for the hot start
was that sign-off. The rule exists for the reason Step 39 gave: `compare_runs` compares floats
exactly, so after such a change every rerun lists thousands of differences, and each comparison
with the committed results needs an argument that they are rounding and nothing more. A
sign-off accepts that cost on purpose, and this step owes the argument: Results classifies all
83,784 differences. A new development dependency, pytest-xdist, came with "split the test
suite". A fourth change, in the same commit as the parallel replicates, fixes the first item of
Step 39's "What was left".

**1. A hot start for the restricted Nash sweeps.**

*Bases and vertices.* Write an LP with its inequalities turned into equalities by slack
variables: minimize $c^\top z$ subject to $Mz = b$, $z \ge 0$, with $m$ rows. ($M$ is the
constraint matrix here; $A$ stays the payoff matrix of Steps 18 and 22.) A *basis* is a set $B$
of $m$ columns whose square matrix $M_B$ is invertible. Setting every other variable to zero
leaves one solution, $z_B = M_B^{-1} b$; if it is non-negative it is a vertex of the feasible
set, and every vertex arises this way. If the LP has an optimum, one is at a vertex. The
*simplex method* (Dantzig 1963) walks from vertex to adjacent vertex. At each basis it prices
the rows, $y = M_B^{-\top} c_B$, and computes the *reduced cost* of each variable outside the
basis,

$$d_j = c_j - M_j^\top y,$$

the change in the objective per unit of $z_j$ brought in. If every $d_j \ge 0$, no neighbor is
cheaper and the basis is optimal. Otherwise a variable with $d_j < 0$ enters, the basic variable
that would first turn negative leaves (one *pivot*), and the walk goes on. Real solvers, HiGHS among
them, also handle bounds other than zero and keep a factorization of $M_B$ instead of its inverse;
the logic is the same. HiGHS runs the *dual* simplex method here (`linprog`'s option), which keeps
the reduced costs non-negative and pivots toward feasibility; how it continues after a change of
costs, from a basis that is feasible but no longer optimal, was not examined. The tests check only
that it starts from that basis (no presolve) and reaches an optimum.

*What a change of costs keeps.* Whether a basis is feasible, $M_B^{-1} b \ge 0$, involves $M$
and $b$ but not $c$. Whether it is optimal involves $c$. The restricted Nash LP of Step 22,

$$\max_{x,q}\ p\,(A y_{\text{fix}})^\top x + (1-p)\,f^\top q \quad\text{s.t.}\quad
F^\top q - A^\top x \le 0,\ Ex = e,\ x \ge 0,$$

has the same constraints for every $p$; only the objective moves. So the optimal basis for one
$p$ is a feasible basis for the next. The walk can start there, and when the optimum moves
little, a few pivots finish it. A solve from scratch must first find a feasible basis, then walk
to the optimum. Starting from an earlier solve is called a *warm start* when a starting point,
such as a basis, is handed to a solver, and a *hot start* when the solver instance also keeps
its working data from the last solve. The code does the second.

*How the code does it.* `linprog` cannot keep a solver between calls: each call builds a new HiGHS
instance. So the LPs now reach HiGHS through SciPy's own bindings (`scipy.optimize._highspy._core`,
a private module of SciPy 1.18.0), built exactly as `linprog(method="highs")` builds them: the rows
of $A_{ub}$ then those of $A_{eq}$, the matrix stored column by column, $-\infty$ as the lower bound
of each inequality row, and the options `linprog` passes (presolve on, the dual simplex method).
`restricted_nash_sweep` poses, for each seat, one LP per $p$ below 1; the recommender (Step 29),
which recommend, fitcheck, confirm and adaptive use, sweeps $p$ = 0.1, 0.25, 0.5, 0.75 and 0.9 in
that order. A new HiGHS instance per seat solves the first $p$ from scratch. For each later $p$ the
code changes the costs (`changeColsCost`) and runs the same instance again, which starts from the
basis it holds. An LP solved from scratch is built and solved as `linprog` would solve it, so it
gives `linprog`'s bytes (a test compares every array passed to HiGHS, and the result). Single
responses, data-biased responses and the first $p$ of every sweep are therefore unchanged. The
restricted Nash LP without its costs is built once per form and kept (an LRU of 8 forms), since only
the costs depend on the model and $p$. The hot-start agent measured a sweep about 3.7× faster on the
development seeds.

*Why presolve is skipped.* A solve from scratch first runs *presolve*. It simplifies the LP
(removes redundant rows and fixed columns, tightens bounds; some reductions depend on the
costs), solves the smaller LP and maps the answer back (*postsolve*). A basis of the original LP
is not a basis of the reduced one. So when HiGHS holds a valid basis, it solves the original LP
from that basis and does not presolve. A test reads HiGHS's presolve status: `kNotPresolved` for
every hot-started solve and for no solve from scratch.

*Why the digits change.* A hot start computes its answer by another route even when it ends at
the same optimal basis as a solve from scratch: the original LP, a factorization built up over
other pivots, no postsolve. Every operation rounds, so the last bits differ. Step 39 saw the
same effect with presolve turned off (solutions up to $1.28 \times 10^{-11}$ apart). Here the
later $p$ usually agree with solves from scratch to about $10^{-11}$.

*Alternative optima, and why some differ by more.* An LP can have several optimal vertices. The
simplex method stops at the first basis it finds optimal, so two starting points can end at
different ones. And optimality is tested within a tolerance: HiGHS's primal and dual
feasibility tolerances are $10^{-7}$ by default, so a reduced cost above $-10^{-7}$ counts as
non-negative. An illustration, made up for this log (it is also the LP the code's start-up check
solves, below): minimize $c_1 z_1 + c_2 z_2$ subject to $z_1 + z_2 \ge 1$, $z \ge 0$. The
vertices are $(1, 0)$ and $(0, 1)$. At the basis $\{z_1\}$, the vertex $(1, 0)$, the row's
price is $y = c_1$, so the reduced cost of $z_2$ is $d_2 = c_2 - c_1$ (the surplus variable's is
$c_1 > 0$).

| Costs $c$ | $d_2$ at $(1, 0)$ | A hot start from $(1, 0)$ |
|---|---|---|
| $(1, 2)$ | $1$ | stays: $(1, 0)$ is optimal |
| $(2, 1)$ | $-1$ | $(1, 0)$ is still feasible (the constraint did not change); one pivot reaches $(0, 1)$ |
| $(1, 1)$ | $0$ | stays at $(1, 0)$; $(0, 1)$ and every point between cost the same |
| $(1, 1 - 10^{-8})$ | $-10^{-8}$, inside the tolerance | stays at $(1, 0)$, though $(0, 1)$ is cheaper by $10^{-8}$ |

In the last two rows a solve from scratch has no reason to start at $(1, 0)$ and can end at
$(0, 1)$. The "strategies" then differ by 1 in each coordinate while the objective values agree
exactly, or within $10^{-8}$. A zero reduced cost at an optimum, as in the third row, is the sign of
such ties: the LP is *dual degenerate*, and unless the vertex is also primal degenerate it has
*alternative optima*. (*Primal* degeneracy is the other kind: one vertex described by several bases,
with a basic variable at zero; it can make the simplex method pivot without moving.) Some of the
project's LPs have such ties, exact or within the tolerance. In the hot-start agent's development
checks the hot path sometimes stopped at another optimal vertex, with strategies differing by up to
about $3 \times 10^{-4}$ and objective values within the tolerance. Two strategies with the same
objective can still score differently on what the experiments report. The objective mixes the gain
against the model and the worst case (weights $p$ and $1 - p$), and another optimal vertex can trade
one for the other. That is the likely source of the largest differences in Results; they were not
traced one by one.

*Still deterministic.* A hot-started sweep is not the same function as five single solves, but
it is a function. Each sweep makes its own HiGHS instance and makes the same calls in the same
order, so its result depends only on its inputs (the form, $A y_{\text{fix}}$ and the $p$
values in order), never on what the process solved before. That matters in a process pool. The
steps that sweep run their tasks in pools of 12 worker processes, and which worker gets which
task depends on timing. Suppose instead a worker kept one HiGHS instance and hot-started each
sweep from the last one's basis. A result would then depend on what that worker happened to
solve before, and a rerun could differ from itself. A test runs the same sweep fresh, after
other LPs and sweeps in two orders (another model, another tree, the same $p$ in reverse order,
the LPs one at a time), and in a spawned process, and requires the same bytes every time. In the
quick-configuration reruns, new code against new code gave 0 differences.

*The memo keeps the two kinds apart.* Step 39's memo is exact when the key determines the input and
the function is deterministic. A hot-started solution at $p = 0.5$ is not the one a solve from
scratch gives, so the LP alone no longer determines the answer: how it was solved does too. A sweep
of two or more LPs is therefore stored under its own key (a sweep of one LP is that LP solved from
scratch, under the single LP's key): the form, $A y_{\text{fix}}$, the $p$ values in order, and
whether the bindings are in use. The two kinds of key never meet. Otherwise
`both_seats("rnr", model, p=0.5)` would return different bytes depending on whether a sweep had run
earlier in the process: a memo that changes results. The key holds $p$ as Python floats and
$A y_{\text{fix}}$ as float64, and the LPs are built from those same values. So equal inputs of
different dtypes share an entry, and the key determines the LPs exactly. The memo's bound (2,048)
now counts solutions, since a sweep's entry holds one per $p$. Like the memo, the stored models
belong to one process and are not safe to share between threads; the experiments use processes.

*When the fast path fails.* Three guards keep it from being less safe than `linprog`:

* A solution is accepted on `linprog`'s terms. HiGHS must report an optimal status, then the
  solution must pass `linprog`'s own check of the result (SciPy's `_check_result`): no NaN, and the
  bounds and constraints met within $\sqrt{10^{-9}} \times 10 \approx 3.2 \times 10^{-4}$. HiGHS's
  $10^{-7}$ tolerances decide when the simplex method stops; this looser check only rejects answers
  that are plainly wrong. So a solution from the bindings must pass the same check as one from
  `linprog`.
* If a hot start fails, that LP is solved from scratch on a new instance, and the later ones are
  hot-started from there. The failure is itself a function of the inputs, so the result still
  is. A failure from scratch raises an error that names the $p$ and how the LP was solved.
* The bindings are private and may change in any SciPy release. At first use each process tries
  them on the illustration's LP: costs $(1, 2)$ from scratch must give $(1, 0)$, then costs
  $(2, 1)$ hot-started must give $(0, 1)$. If the import or this check fails, every LP falls
  back to `linprog`, solved from scratch, with one `RuntimeWarning` per process. After a
  successful check, a failure raises. A test requires the bindings to be in use with the
  installed SciPy, so an upgrade that breaks them fails a test instead of quietly making
  everything slower.

**2. Two replicates at once: processes, threads and throughput.** The sequence step (Step 34)
trains five Phase 1 replicates one after another, each with 8 PyTorch threads at full size.
`run_phase3 --sequence-jobs N` (`phase3_sequence --replicate-jobs N`) trains up to N at once.

*Processes, not threads.* Each replicate runs in its own process from a `ProcessPoolExecutor`.
Processes share no memory, so two replicates cannot touch each other's caches or settings. Each
worker sets the configured torch thread count when it starts (`_init_worker`). A fit's bytes
depend on its number of threads (Steps 34 and 39), so the obvious alternative, one replicate
with 16 threads, would change results. Running whole replicates side by side, each with its
8 threads, uses more of the processor and keeps the bytes.

*Spawned workers.* Windows cannot copy a running process (it has no `fork`). Python starts each
worker with *spawn*: a fresh interpreter that imports the main module under another name
(`__mp_main__`) and receives its task by pickling. This has three consequences, met in Steps 8
and 24 and again here:

* A script that starts a pool keeps its top-level work under `if __name__ == "__main__":`.
  Otherwise each worker would try to start the run again, and multiprocessing stops that with an
  error.
* The function a worker runs must be importable by name from a module (`_replicate_from_cache`;
  in the tests, `_sweep_bytes_in_this_process`).
* Each worker starts with empty caches and default settings, hence the thread count in the
  initializer.

Starting a worker also takes time, for example to import PyTorch (not measured separately). In
the quick configuration on the development seeds 98 and 99 (two replicates, 4 threads each), two
replicates at once were slower than one after another (38.6 s against 33.6-34.3 s), probably
because process start-up outweighs the gain when the models are small.

*The order of work.* With N > 1 the parent first makes sure every replicate's hands are in the
simulation cache. It goes one replicate after another, simulating what is missing with the
simulation's own worker processes. Only then do N training processes start, and they read cache
files only. So two simulations never run at once, and no training process starts a pool of its
own (two replicates each simulating with 12 workers would put far more processes than
processors on the machine). The populations that part B needs reach each worker once, through
the pool's initializer, not with every task.

*Results in a fixed order.* Replicates finish in whatever order they finish. The parent stores
each result under its replicate number and aggregates them in replicate order. Aggregates are
sums over replicates, and floating-point sums depend on their order (Step 39), so aggregating in
completion order would make the last digits depend on which replicate finished first. A test
makes the last replicate's results arrive first and requires the same files. The number of jobs
is recorded as `meta.replicate_jobs`, outside `meta.config`. It says how a run was executed, not
what it computed, so `compare_runs`, which requires the configs to match, treats one-job and
two-job runs as the same experiment. Both full-size runs gave "sequence: 0 differences".

*Lazy submission.* Submitting all five replicates at once would queue them in the pool. A failure in
one would surface only when its result was read, after the others had finished training (up to about
half an hour later at full size), and queued replicates would keep starting. Instead the parent
keeps at most N replicates in flight and waits for the first to finish
(`wait(..., FIRST_COMPLETED)`). A failure prints at once and no new replicate starts; the running
ones finish, and the first error is raised with the worker's output attached as a note. A worker's
printed lines are captured and printed when it finishes, so two replicates' lines never interleave.

*Throughput and latency.* *Latency* is how long one job takes; *throughput* is how many jobs finish
per hour. Running jobs side by side trades one for the other. If one replicate alone takes $T$ and
each of two running together takes $sT$ (the slowdown $s \ge 1$ comes from sharing the machine),
then two back to back take $2T$ and two together $sT$. The pair finishes sooner whenever $s < 2$,
and throughput rises by $2/s$, while each replicate takes $s$ times as long. Here $s$ was 1.65-1.71
(Results), so throughput rose by about $2/1.7 \approx 1.18$. With five replicates and two slots the
fifth runs alone, so training takes $(2s + 1)T$ instead of $5T$: with $s = 1.7$, $4.4/5 = 0.88$, 12%
less. Measured (post hoc check): about 1,746 s against 2,003 s, a ratio of 0.87. With an even number
of replicates the ratio would be $s/2 \approx 0.85$ (with many replicates it approaches that).

*Why the slowdown is about 1.7, not 1 (a likely explanation, not measured).* The laptop's processor,
an AMD Ryzen 9 8945HS, has 16 *logical processors*: 8 physical cores with two hardware threads each
(*simultaneous multithreading*, SMT; Intel calls it hyper-threading). Two threads on one core share
its execution units, caches and memory bandwidth, so the second thread adds much less than a second
core would. One replicate's 8 threads can each have a core to themselves; two replicates' 16 threads
fill both hardware threads of every core. The two also share memory bandwidth and the L3 cache, and
a laptop processor runs at a lower clock with every core busy; how much each contributed was not
measured. With 16 threads busy, the machine did about 1.18 times the training work per second that
it did with 8. Running more threads than logical processors (*oversubscription*) would make the
operating system time-slice them. Parallel numeric code, whose threads wait for each other at every
step, slows down sharply when one of its threads is switched out. The step warns when jobs × threads
exceeds the logical processors; here $2 \times 8 = 16$, exactly the limit.

*Memory.* A full-size replicate peaks at about 4.4-4.5 GB in its own process (Step 39's profile:
about 4.1 GB plus 0.27-0.38 GB for the embedding memo), rounded up to `GB_PER_JOB = 5`. Two at once
need about 10 GB, on a 15 GB laptop that the owner's applications share. The step warns when free
physical memory is below 5 GB per job, and it did at the start of the two-job run ("4.7 GB of
physical memory available; 2 full-size replicates at once need about 10 GB and may page"). Free
memory then fell to 1.42 GB (Results). When physical memory runs out, Windows moves pages to disk
and back, and a process waiting on that runs far slower. Whether it happened here was not measured.

*The decision* (post hoc, after the full-size runs; made under the owner's "make the rest of the
decisions"). The default stays one job. About 12% of a full-size step is not worth taking free
memory down to 1.4 GB on a laptop shared with the owner's applications, and the project runs one
heavy job at a time. The option stays for machines with more memory.

**3. Splitting the test suite across processes.**

*How pytest-xdist schedules.* `-n 4` starts four worker processes. Each collects the suite and
runs the tests the controller hands it. The `--dist` mode decides how:

* `load` hands out tests in batches as workers become free.
* `loadfile` hands out whole files. Every test of a file runs in one worker, so a module-scoped
  fixture (built once per module in each process) is built once.
* `worksteal` gives each worker an equal, contiguous share of the tests up front. A worker that
  runs out takes some of the pending tests from the end of a busy worker's queue (*work
  stealing*, Blumofe and Leiserson 1999).

`loadfile` was chosen first, to keep module fixtures built once per file. On a quiet machine, four
processes took the 457-test suite from 64.1 s to 33.2 s; eight took 28.2 s, with twice as many
worker processes (so presumably about twice the memory; not measured).

*Makespan.* The suite's time is its *makespan*: the time until the last worker finishes. On $m$
workers with jobs of lengths $t_1, \dots, t_n$, no schedule can beat

$$C \ \ge\ \max\Big(\max_i t_i,\ \frac{1}{m}\sum_i t_i\Big),$$

since the longest job runs on one worker and the total work is shared by at most $m$. Graham
(1966) showed that giving each job, in any order, to the first free worker (*list scheduling*)
is never worse than $2 - 1/m$ times the best schedule. The bad cases put a long job last: it
starts when the others are nearly done and runs while they sit idle. Sorting the jobs longest
first (*LPT*, longest processing time first) brings the guarantee down to $4/3 - 1/(3m)$ times
the best (Graham 1969), 1.25 for $m = 4$.

*What happened.* The new test `test_two_replicate_jobs_write_what_one_writes` runs the tiny sequence
step twice from empty caches, once with one job and once with two spawned workers. It takes 31.8-33
s alone and 44.2 s under the load of the rest of the suite. Its file (10 tests) took 44.6 s
serially. Under `loadfile` that file is one job. xdist 3.8 hands out files by their number of tests,
so a 10-test file is not started first, and most likely it finished last (the order was not logged).
The suite went from 33-36 s to 65-67 s, although the 471 tests' total serial time is only 98.7 s.
`load` (69.9 s) and `worksteal` alone (64.0 s) did no better, presumably because neither started
that test early.

*The fix (post hoc).* Tests marked `long` (one so far) move to the front of the collection
(`tests/conftest.py`). The sort is stable, so every other test keeps its order. The suite now uses
`--dist worksteal`. The first worker's share begins with the long test; the other three work through
their shares, then take tests from the end of the first worker's queue. Three runs took 53.5, 53.9
and 53.6 s, with the same tests and results: LPT applied to the one job that matters. The bound:
$98.7/4 = 24.7$ s of work per worker, but the longest test takes 44 s under the load of the rest of
the suite, so while the rest runs beside it, about 44 s plus the workers' start-up is the floor.
(The bound assumes fixed job lengths; here a test's length depends on what runs beside it.) The
measured 53.5-53.9 s is within about 10 s of it. Going lower would take a shorter test;
`-m "not slow"` skips it (it is also marked `slow`). Work stealing has a price: a file's tests can
now run in several workers, each building that file's module fixtures. The times above include that
cost.

*One process when needed.* `-n 0` runs everything in the main process: for a single file
(`python -m pytest -n 0 tests/test_leduc_exploit.py`: 19 passed in 5.0 s), and for the debugger,
since xdist refuses `--pdb` when it distributes tests.

**4. Transfer reuses part A's embeddings.** Step 39's embedding memo keys on the identity of a
`HandSequences` object and the block of players encoded together. Part A predicted the held-out
bots through `neural_inputs(data.test, kind)`, one object. Transfer (part B) read the same
players through `PoolView(data.test).inputs(kind)`, another object. So the memo missed and those
players were encoded once more per model (Step 39's first "What was left" item). Now `replicate`
makes one `PoolView` of the test bots, part A predicts through it (`test_view`), and transfer
reads the same view. The objects are identical, so the memo hits. The result is bit-identical,
for Step 39's reason: a reused embedding is one computed in exactly the block that would compute
it again. A test counts the encodings of the 30 held-out bots: two blocks of 16, once each,
before; none after. The full-size runs gave "sequence: 0 differences". The time saved was not
measured separately; it is a part of prediction, which Step 39 put at 6.6% of the step. The same
commit releases part A's models before part C (`del models, bots`) to lower peak memory; that
effect was not measured separately either.

**How it was done.** Each change was made by a subagent in its own git worktree outside OneDrive
(`poker-lab-xdist`, `poker-lab-hotstart`, `poker-lab-seqpar`). A reviewer subagent read each diff
before its commit, and its findings were fixed before the commits (Mistakes). The commits were
cherry-picked onto a branch (`perf2`) on top of `1137e8a`: `80f2152` (pytest-xdist), `03b3b4f` (the
hot start) and `3fa49aa` (parallel replicates and the transfer reuse). Two more commits, `505e16a`
and `9f877e5`, change the help text to "checked at full size with two", record the full-size
measurement in a docstring, start tests marked `long` first and switch to work stealing. They change
no code path of an experiment, so the verification at `3fa49aa` covers them. The suite went from 457
to 471 tests, all passing. The verification ran from a clean worktree at `3fa49aa`, one step at a
time, while a sampler logged free physical memory every 10 s.

**Results.**

*Reruns on seed 2026, full configuration* (perf2 at `3fa49aa`):

| Step | Compared with | `compare_runs` |
|---|---|---|
| sequence, two jobs | committed results | 0 differences |
| sequence, one job | committed results | 0 differences |
| exploit (Phase 3) | committed results | 0 differences |
| Phase 2 exploit | committed results | 0 differences |
| Phase 2 aivat | committed results | 0 differences, apart from its two post hoc files, which a rerun does not make (as in Step 39) |
| adaptive | committed results | 7,255 differences |
| recommend | committed results | 20,939 differences |
| fitcheck | Step 37's rerun of the old code | 27,915 differences |
| confirm | Step 37's rerun of the old code | 27,675 differences |

Exploit and Phase 2's steps pose single LPs only, which keep `linprog`'s bytes. Recommend,
fitcheck, confirm and adaptive sweep. As in Step 39, fitcheck and confirm are compared with
Step 37's rerun, because the committed files already differ from the current code by Step 37's
53 and 23 wording differences.

*What kind of differences* (classified for this step, post hoc, by a script that walks both
files):

| | adaptive | recommend | fitcheck | confirm |
|---|---|---|---|---|
| Float differences | 7,255 | 20,939 | 27,915 | 27,675 |
| Other differences (strings, integers, booleans, list lengths) | 0 | 0 | 0 | 0 |
| Relative gap below $10^{-12}$ | 7,197 (99.2%) | 20,841 (99.5%) | 27,662 (99.1%) | 27,453 (99.2%) |
| Largest absolute gap | $2.6 \times 10^{-7}$ | $2.0 \times 10^{-6}$ | $7.3 \times 10^{-5}$ | $1.7 \times 10^{-6}$ |
| Largest relative gap | $7.0 \times 10^{-6}$ | $6.9 \times 10^{-4}$ | $4.9 \times 10^{-3}$ | $2.1 \times 10^{-3}$ |
| Float differences under `aggregate` | 316 | 182 | 231 | 233 |
| Summary lines that differ | 0 | 0 | 0 | 0 |
| Figures | pixel-identical | pixel-identical | pixel-identical | pixel-identical |

* No string, integer, boolean or list length differs, so no choice, count, flag or prediction
  verdict changed. The floats that differ under adaptive's and confirm's `predictions` (12 and
  7) are measured values; the verdicts are booleans and did not change.
* The largest absolute gap is fitcheck's exploitability of a restricted Nash response at
  $p = 0.75$ for one bot at 30 hands: 0.0741832 against 0.0742564. The largest relative gaps are
  on values near zero, such as a lower bound of $-2.5559 \times 10^{-5}$ against
  $-2.5435 \times 10^{-5}$ (0.49%).
* Aggregates differ too, but every summary line rounds to the same text.
* In recommend, no difference falls under $p = 0.1$, the first value of each sweep, solved from
  scratch. The per-bot differences fall under 0.25 (6,839), 0.5 (7,655), 0.75 (4,630) and 0.9
  (1,633), and 182 under aggregates (counted for this step, post hoc). This is what the design
  says: a sweep's first LP keeps `linprog`'s bytes.

The committed results stay the original runs, as after Step 37; replacing them would be the
owner's decision. A rerun with today's code reproduces sequence, exploit and Phase 2 exactly, and
adaptive, recommend, fitcheck and confirm to the digits above.

*Same-session timing, quick configuration* (`run_phase3 --quick --seed 2026 --only adaptive`;
old is `1137e8a`, new is `3fa49aa`; alternating, four rounds; "step" is the step's own
whole-second time, "process" the whole command's wall time):

| Round | Old: step, process | New: step, process |
|---|---|---|
| 1 | 27 s, 32.1 s | 16 s, 18.6 s |
| 2 | 27 s, 30.0 s | 16 s, 18.7 s |
| 3 | 27 s, 29.9 s | 16 s, 18.7 s |
| 4 | 27 s, 30.0 s | 16 s, 18.7 s |

The step took 41% less time. Old against old and new against new gave 0 differences: both are
deterministic. Old against new gave 552 float differences and no summary line. The largest
absolute gap was $2.3 \times 10^{-7}$, the largest relative gap $6.8 \times 10^{-5}$, and 532 of
the 552 were below $10^{-12}$. Step 39 measured this step at 54 s before its changes and 28 s
after, in another session; today the old code took 27 s.

Post hoc, Amdahl's law (Step 39) links this to the sweep's speedup. If sweeps took a fraction
$f$ of the step and became 3.7× faster, $16/27 = 0.593 = (1 - f) + f/3.7$ gives $f \approx 0.56$.
No profile of the code after Step 39 measured that share, so this says only that the two
measurements fit together if sweeps were a little over half of the quick step.

*Full-size sequence step* (training times, ratios and whole-step estimates computed post hoc from
the logs). A caveat comes first. The two-job run went first, in a new worktree, with an empty
simulation cache: its "hands ready" lines took 199-204 s per replicate, 1,008 s for five. The
one-job run went second and reused that cache. So the whole-step times (2,782 s with two jobs, 2,023
s with one) cannot be compared; the training times can:

| Replicate | One job (s) | Two jobs (s) | Ratio |
|---|---|---|---|
| 1 | 403 | 688, with 2 | 1.71 |
| 2 | 401 | 686, with 1 | 1.71 |
| 3 | 399 | 659, with 4 | 1.65 |
| 4 | 399 | 659, with 3 | 1.65 |
| 5 | 401 | 399, alone | 1.00 |
| Training, all five | 2,003 | about 1,746 | 0.87 |

| Fit (seconds per replicate) | Alone (one job) | Paired (two jobs, replicates 1-4) |
|---|---|---|
| Sequence model (GRU) | 140-142 | 234-255 |
| Order dropped | 26-27 | 44-45 |
| Counters network | 15 | 23-24 |
| Part C, order dropped | 130-131 | 218-222 |
| Part C, counters network | 71-73 | 114-117 |

Training took 13% less with two jobs. For the whole step, add the same simulation time to both
(an assumption: the one-job path simulates the same hands with the same workers). With a warm
cache, about 12% less (2,782 − 1,008 = 1,774 s against 2,023 s). With an empty cache, about 8%
less (2,782 s against 1,008 + 2,023 = 3,031 s). The second pair ran about 4% faster than the
first, for reasons not examined.

*Memory* (post hoc statistics of free physical memory from `Win32_OperatingSystem` every 10 s, with
the owner's applications open; 4.97 GB free at the start):

| Run | Lowest free | Mean free |
|---|---|---|
| sequence, two jobs (simulation and training) | 1.42 GB | 3.44 GB |
| sequence, one job | 3.00 GB | 4.49 GB |
| adaptive | 3.69 GB | 4.13 GB |

In the quick configuration on the development seeds 98 and 99 (a development benchmark on
2026-10-31), two jobs raised the peak private memory of the whole process tree from 3.32 to 4.2 GB.
They took 38.6 s twice, against 34.3 and 33.6 s for one job, with 0 differences.

*Test suite* (pytest's own times; the 457-test settings were run twice, the first time on a
busier machine):

| Tests | Options | Time (s) |
|---|---|---|
| 457 | `-n 0` (one process) | 115.7, 64.1 |
| 457 | `-n 2 --dist loadfile` | 60.7, 51.2 |
| 457 | `-n 4 --dist loadfile` | 36.2, 33.2 |
| 457 | `-n 8 --dist loadfile` | 28.5, 28.2 |
| 457 | `-n 4 --dist load` | 36.1, 34.0 |
| 471 | `-n 0` | 98.7 |
| 471 | `-n 4 --dist loadfile` | 66.6, 67.0, 67.4, 65.1 |
| 471 | `-n 4 --dist load` | 69.9 |
| 471 | `-n 4 --dist worksteal` | 64.0 |
| 471 | `-n 4 --dist worksteal`, `long` tests first (committed) | 53.5, 53.9, 53.6 |

On the quiet round, four processes ran the 457 tests 1.9× faster than one, and eight 2.3×
faster. Candidates for the gap to 4× and 8×, not measured separately: worker start-up, fixtures
built in each process, the longest file, and tests that use several threads or processes
themselves.

*Full-size step times of the verification run* (single runs):

| Step | Seconds |
|---|---|
| sequence, two jobs (empty cache) | 2,782 |
| sequence, one job (warm cache) | 2,023 |
| adaptive | 1,633 (Step 39's run, on another day: 2,492) |
| exploit | 134 |
| recommend | 53 |
| fitcheck | 214 |
| confirm | 214 |
| Phase 2 exploit | 95 |
| Phase 2 aivat | 19 |

Adaptive took 34% less time than in Step 39's run, but on another day. Step 39 found that the
machine's speed varies between days by more than some effects, so the same-session 41% of the
quick configuration is the clean number.

**Run times, awake only** (C.3).

* The verification ran from 14:38:59 to 16:38:39 on 2026-11-01: 7,180 s of wall time, of which
  the steps account for 7,167 s. Every Modern Standby interval in that window had the reason Idle
  Timeout (screen off), with a keep-awake request held. The memory sampler wrote 699 samples,
  never more than 11 s apart, so the machine ran throughout. As in Steps 37 and 39, idle-timeout
  standby counts as awake.
* The quick-configuration comparison ran right after, between 16:40 and 16:44.
* Apart from the four quick rounds and the repeated suite timings, every time here is from a
  single run; none has a confidence interval.

**What was left.**

* The committed results are still the original runs. A rerun now matches adaptive, recommend,
  fitcheck and confirm to the digits in Results, not bit for bit; replacing the committed results
  is the owner's decision.
* The hot start's gain at full size was not measured in one session; 1,633 s against Step 39's
  2,492 s spans two days.
* Two jobs stay off by default. On a machine with more memory and cores they may pay more; that
  was not measured.
* `refit_recent` got no memo of its own: Step 39's LP memo already answers its repeated LPs.
* Step 39's second item (a refit keeps the old `history` and `size_temperatures`) still stands.

**What it means.** Step 39 asked whether a change moves any result. This step had to ask how far one
moves, and answer as exactly. A hot start reaches the optimum by another path, so it rounds
differently, and where an LP has several optimal vertices it may stop at another one. Both are
properties of the simplex method, not bugs. Neither reached a summary, a figure or a verdict here:
99% of the changed floats moved by less than $10^{-12}$ relative, and the rest by at most
$7.3 \times 10^{-5}$ in absolute value. What made the change acceptable is that the result is still
a function of its inputs, the same in any process, whatever the process solved before; without that,
a rerun could not even reproduce itself. The other three changes keep the bytes, for Step 39's
reasons: each fit's thread count stays fixed, results are combined in a fixed order, and a reused
embedding is one the same block would compute. Parallel work pays only as far as the parallel pieces
do not share. Two replicates on 16 hardware threads that are really 8 cores ran 1.7 times slower
each and saved 13% together. And a test suite on four processes is never faster than its longest
test, and slower still unless that test starts first.

**Mistakes.**

* A failed hot start first aborted the whole sweep. The reviewer caught it; that LP is now solved
  from scratch on a new instance, and the sweep continues.
* The first draft accepted a solution from the bindings on HiGHS's status alone, so it could
  pass where `linprog` fails. The reviewer caught it; `linprog`'s own check now runs too.
* The sweep memo first hashed $p$ and $A y_{\text{fix}}$ as given, so equal values of different
  dtypes got different keys. The reviewer caught it; both are converted (Python floats, float64)
  before hashing, and the LPs are built from the converted values.
* The first tests did not check that a sweep's later LPs are really hot-started. The reviewer
  caught it; a test now reads HiGHS's presolve status.
* The first parallel draft reported a failing replicate only after the others had finished training,
  up to about half an hour later at full size. The reviewer caught it; submission is now lazy.
* The same draft defaulted to two jobs without a full-size measurement and did not consider
  memory. The reviewer caught it; the default is one, and the step warns about threads and
  memory.
* The first comparison of the two full-size runs set 2,782 s against 2,023 s, which made two
  jobs look about 38% slower. The two-job run had gone first and filled the empty simulation
  cache (1,008 s) that the one-job run then reused. The training times in the logs are the fair
  comparison. When runs share a cache, fill it first or plan the order, as Step 39's first quick
  round, with an empty cache, already showed.
* The new two-replicate test doubled the suite's time (65-67 s against 33-36 s), because a
  whole-file scheduler started its file late. It was found by timing the suite before the commit
  and fixed, post hoc, by starting `long` tests first under work stealing (53.5-53.9 s).

**Limitations.**

* Full-size times are single runs, and the hot start's full-size gain spans two days.
* Three numbers come from the hot-start agent's development checks, not from this step's runs:
  the 3.7× per sweep, the usual agreement to about $10^{-11}$, and the strategy differences of up
  to about $3 \times 10^{-4}$.
* Which optimal vertex a hot start reaches is deterministic but otherwise arbitrary. The
  differences seen are bounded by measurement, not by proof; another LP could differ more, with
  its objective still within tolerance.
* The fast path depends on a private SciPy module. It is guarded by the start-up check, a test
  and the fallback, not by any promise from SciPy.
* Memory was measured for the whole system, with the owner's applications open. Paging was not
  measured, and the per-replicate peak comes from Step 39's profile.
* Parallel replicates were checked bit for bit at full size on one seed and in the tiny and
  quick configurations, and only on Windows, where workers are spawned.
* The suite's time depends on the machine's load; the long test alone sets a floor of about 44 s
  plus start-up.

**Verification.**

* `tests/test_leduc_exploit.py` (11 → 19 tests; Step 39's sweep test replaced):
  * An LP solved from scratch through the bindings gets exactly the arrays, options and (empty)
    integrality that `linprog` passes to HiGHS, and gives `linprog`'s bytes. This holds for both
    seats, for several costs in a row on one form (its model built once), and for a data-biased
    LP.
  * A sweep's first LP per seat equals the single response bit for bit. The later ones are
    hot-started (HiGHS reports `kNotPresolved` for exactly those) and agree with LPs solved from
    scratch: the objective within $10^{-7}$, and for these LPs the strategy within $10^{-9}$.
    Every strategy is valid, the best response stands at $p = 1$, and invalid $p$ raise.
  * A failed hot start is solved from scratch on a new instance (the single LP's bytes), and the
    next LP is hot-started from it; the result is still a function of the inputs. A failure from
    scratch raises with the $p$ and how the LP was solved, and an infeasible LP's message names
    HiGHS's status.
  * `linprog`'s check accepts a bound, the inequality and the equality at 0.9 times the
    tolerance and rejects them at 1.1 times; NaN is rejected.
  * float32 values of $p$ and the same values as Python floats share one entry; the float64 values
    nearest 0.1 and 0.9 are other LPs (0.5 is exact in both), so that sweep is another entry.
  * The same sweep gives the same bytes fresh, after other LPs and sweeps in two orders, and in
    a spawned process.
  * Single LPs and sweeps never answer for each other. Eviction counts solutions, and an entry
    larger than the bound is kept alone until the next one.
  * The bindings are in use with the installed SciPy, without a warning.
  * Without them there is one warning per process, and every LP, a sweep's included, is
    `linprog`'s, solved from scratch.
* `tests/test_fit_checks.py`: the spy now counts `_solve_sweep`, `_run` and `linprog`. A ranking
  poses at least one sweep per seat and solves at least seats × $p$ LPs.
* `tests/test_sequence_experiment.py` (4 → 10 tests):
  * `--sequence-jobs` reaches the sequence step only, and 0 is rejected.
  * One job, or one replicate, runs the old loop in this process; more jobs than replicates are
    capped at the number of replicates.
  * The step warns about threads above the logical processors and memory below about 5 GB per
    job, and never warns for one job.
  * Transfer encodes none of the 30 held-out bots again (two blocks before, none after), and
    every number equals the old wiring's, float for float.
  * (`slow`, `long`) With two replicate processes, and the last replicate's results delivered
    first, the step writes the same JSON, summary and figures as with one job, each run starting
    from an empty cache. The parallel path simulates the same cache files, each worker has the
    configured thread count, and `meta` records the jobs.
  * A worker's error reaches the parent with the worker's output as a note. At most `jobs`
    replicates are in flight, and none starts after a failure.
* `tests/conftest.py` puts tests marked `long` first; `pyproject.toml` registers the marker and
  sets `-n 4 --dist worksteal`.
* Reruns: `python -m experiments.run_phase3 --seed 2026 --only sequence --sequence-jobs 2 --out
  DIR`, the same with `--sequence-jobs 1`, then `python -m experiments.compare_runs
  experiments/results/phase3 DIR sequence`. The same for adaptive, exploit and recommend, and for
  fitcheck and confirm against Step 37's rerun. Phase 2's exploit and aivat ran through
  `run_phase2.run(..., out_dir=DIR)`.
* Same-session timing: `python -m experiments.run_phase3 --quick --seed 2026 --only adaptive
  --out DIR`, from a worktree at `1137e8a` and one at `3fa49aa`, alternating, with
  `compare_runs` between runs.
* Suite: `python -m pytest` (471 passed, 53.5-53.9 s) and `python -m pytest -n 0
  tests/test_leduc_exploit.py` (19 passed in 5.0 s).

## Step 41: Duplicate dealing at more than two seats (P2.11b)

**The question.** Step 23 plays each deal twice with the seats swapped, and Step 28's push/fold
hands show what that saves heads-up: a median over the matchups of 2.06× fewer hands than
independent dealing. The next item after AIVAT was to generalize this to n players by cyclic seat
rotation and to measure the variance reduction in a 6-max simulation with Phase 1's archetype bots,
under its own plan and pre-registration. So: how much variance per hand played does rotating the
seats remove at 6-max no-limit hold'em, and how does that depend on the opponents (copies of one
bot, bots of one archetype, a random mix) and on the number of players (2, 3, 6)? A live table
cannot deal the same cards twice, so this is a tool for simulated evaluations only. No committed
result had to move: with two players the new code plays exactly the old hands (section 2), so
nothing needed the owner's sign-off.

**1. The variance of a mean of correlated results.**

*The formula.* Fix a player and a deal. Let $X_1, \dots, X_n$ be the player's results at the $n$
tables, each with variance $\sigma^2$, and let $U = \frac{1}{n} \sum_r X_r$ be their mean. The
variance of a sum is the sum of all the covariances, so

$$\operatorname{Var}(U) = \frac{1}{n^2} \sum_{r=1}^{n} \sum_{r'=1}^{n}
  \operatorname{Cov}(X_r, X_{r'})
  = \frac{1}{n^2} \big( n\sigma^2 + n(n-1)\,\rho\,\sigma^2 \big)
  = \frac{\sigma^2 \big(1 + (n-1)\rho\big)}{n}.$$

The $n$ diagonal terms are the variances. The other $n(n-1)$ are covariances, and $\rho$ is by
definition the mean correlation over those ordered pairs of tables. With $n = 2$ this is Step 28's
$\sigma^2 (1 + \rho) / 2$. Survey sampling knows the factor $1 + (m-1)\rho$ as the *design effect*
of a cluster of $m$ units with *intraclass correlation* $\rho$ (Kish 1965): there $\rho$ is usually
positive, and a cluster is worth less than $m$ separate units. Monte Carlo calls runs built to be
negatively correlated *antithetic variates* (Hammersley and Morton 1956). Duplicate dealing is that
idea applied to seats.

*What a deal costs.* A deal at $n$ tables is $n$ hands. So $N$ hands make $N/n$ deals, whose mean
has variance $n \operatorname{Var}(U) / N$. Independent dealing gives $\sigma^2 / N$ for the same
$N$. The quotient is this step's metric,

$$\text{ratio} = \frac{\sigma^2}{n \operatorname{Var}(U)} = \frac{1}{1 + (n-1)\rho},$$

how many times fewer hands duplicate dealing needs for an interval of the same width. With
$\rho = 0$ the ratio is 1: the $n$ plays are just $n$ hands. With $\rho = 1$, the same result at
every table, it is $1/n$: $n$ hands for the information of one. So duplicate dealing pays only when
$\rho < 0$. An interval's width goes as $1/\sqrt{N}$, so on a fixed number of hands it narrows by
only the square root of the ratio: 1.30× fewer hands is a 1.14× narrower interval.

*The bound.* A variance cannot be negative, so $1 + (n-1)\rho \ge 0$, that is

$$\rho \ge -\frac{1}{n-1}.$$

That is $-1$ at two tables, $-1/2$ at three and $-1/5$ at six. Two results can be exact opposites;
six results cannot all be opposite to one another. At the bound $U$ never varies, and as $\rho$
nears it the ratio grows without limit. So $\rho$ alone cannot compare table sizes: $-0.05$ is a
quarter of the way to the bound at six seats and a twentieth of the way at two. The comparable
quantity, chosen post hoc, is $s = -(n-1)\rho$, the share of perfect cancellation reached: 0 for
none, 1 at the bound. A share $s$ gives the ratio $1/(1-s)$ at any $n$.

*Skill plus or minus luck.* Step 23 said a player's two results share the card luck with opposite
signs: $X_1 = a + \ell$ and $X_2 = a - \ell$, with mean $a$. For $n$ tables write $X_r = a + \ell_r$
with $\sum_r \ell_r = 0$: the mean is $a$ whatever the luck. If the $\ell_r$ are exchangeable with
variance $v$ and covariance $c$, then $0 = \operatorname{Var}(\sum_r \ell_r) = nv + n(n-1)c$, so
their correlation is $-1/(n-1)$. Luck that sums to zero sits exactly at the bound. Step 23's
"removes most of it" describes that ideal; Results show how far these bots fall short.

**2. Cyclic rotation as a design.**

*The seating.* At table $r$ player $j$ sits in seat $(j + r) \bmod n$. All tables of a deal use the
same deck order and the same button seat, and cards are dealt by position, so a seat holds the same
cards and position at every table. For three players:

| | Table 0 | Table 1 | Table 2 |
|---|---|---|---|
| Player 0 | seat 0 | seat 1 | seat 2 |
| Player 1 | seat 1 | seat 2 | seat 0 |
| Player 2 | seat 2 | seat 0 | seat 1 |

Every row holds every seat once, so each player holds each seat's cards and position once per deal.
Every column does too, so each table is a full table. An $n \times n$ array of $n$ symbols with that
property is a *Latin square*, the classic design for balancing two factors at once (Fisher 1935).
This one is cyclic: for fixed $j$ the map $r \mapsto (j + r) \bmod n$ is one-to-one, and so is the
map $j \mapsto (j + r) \bmod n$ for fixed $r$, which is the whole proof. With $n = 2$ it is Step
23's seat swap.

*The sum-zero argument.* Let $w^{(r)}_q$ be seat $q$'s result at table $r$. The simulator takes no
rake, so $\sum_q w^{(r)}_q = 0$ at every table. Player $j$'s mean over a deal is

$$u_j = \frac{1}{n} \sum_{r=0}^{n-1} w^{(r)}_{(j+r) \bmod n}.$$

If a seat's result were the same at every table, $w^{(r)}_q = w_q$, then as $r$ runs over the tables
the seat runs over all the seats, and $u_j = \frac{1}{n} \sum_q w_q = 0$, for every player on every
deal: the bound of section 1. With different agents the plays differ from table to table, a seat's
result depends on $r$, and only part of the luck cancels.

*When it is exact.* For copies of one deterministic agent that ignores player ids. Each word
matters. Copies: the same agent must play each seat at every table. Deterministic: each table has
its own decision RNG (Step 23), so an agent's coin flips are not mirrored. Ignores ids: agents can
see them, and one that treats `p3` differently plays differently after the rotation. The odd chip of
a split pot does not break it, because it goes by position, not by player: three always-call agents
with an ante of 1 have 9-chip pots whose two-way splits leave an odd chip, and each seat's result is
still the same at every table (a test). `CallingAgent` is the exact case the experiment checks.

*What it does not balance.* Since $(j + 1 + r) - (j + r) = 1$, player $j + 1$ sits in the next seat
after player $j$ at every table. The rotation never changes who sits to whose left. Different
players have $(n-1)!$ cyclic orders: 1 at two seats, 2 at three, 120 at six. A match plays one of
them, so it measures that seating order. The experiment seats each lineup in the order drawn,
records it as `seating`, and its aggregates average over the orders drawn. No lineup was replayed in
another order, so how much the order matters was not measured.

*Heads-up stays byte-identical.* The committed bridge results (Step 28) were built on the old
two-player code, so $n = 2$ must give the same hands. The new code draws its seeds with
`SeedSequence(seed).generate_state(n + 1)`, which at $n = 2$ is the old code's `generate_state(3)`,
and it calls the factories table by table in seat order, which for two players is hero, villain,
villain, hero, as before (the bridge's hero factory draws a seed per call). A test rebuilds the old
algorithm of commit `2131bb4` inline and compares every hand's serialized form. The bridge step, the
only other experiment that calls `duplicate_match`, was rerun on seed 2026 with the new code:
"bridge: 0 differences".

**3. Estimating the ratio and $\rho$.** One player of a lineup gives $x_{r,i}$, the result of deal
$i$ at table $r$ in big blinds, for $D$ deals. With $\bar x$ the mean of all $nD$ results and
$u_i = \frac{1}{n} \sum_r x_{r,i}$,

$$\hat\sigma^2 = \frac{\sum_{r,i} (x_{r,i} - \bar x)^2}{nD - 1}, \qquad
  s_u^2 = \frac{\sum_i (u_i - \bar x)^2}{D - 1}, \qquad
  \widehat{\text{ratio}} = \frac{\hat\sigma^2}{n\, s_u^2}.$$

*Same estimate, another interval.* The mean of the $u_i$ is the mean of all the hands, so the point
estimate of the win rate does not change. Only the interval does. `DuplicateDealEvaluator` with
`group_size=n` puts a Student-t interval over the $D$ per-deal means, which are independent. A
t-interval over the $nD$ hands would be wrong here, because a deal's hands are not independent: it
takes the mean's variance to be $\hat\sigma^2 / (nD)$ when it is $\sigma^2 (1 + (n-1)\rho) / (nD)$.
With $\rho < 0$ it is too wide, by the square root of the ratio. The results store it for display
only. Post hoc, in the real run's random mix over 12,000 hands, the median half-widths are 44.3
bb/100 for that interval and 38.6 for the per-deal one.

*Correlation by offset.* With $z_{r,i} = x_{r,i} - \bar x$ and $S = \sum_{r,i} z_{r,i}^2$,

$$c_k = \frac{1}{S} \sum_{r,i} z_{r,i}\, z_{(r+k) \bmod n,\, i}, \qquad
  \hat\rho = \frac{1}{n-1} \sum_{k=1}^{n-1} c_k.$$

Offset $k$ compares a player's result at one table with its result $k$ seats further round. $c_k$ is
the ordinary Pearson correlation of the $nD$ pairs: both coordinates are the same numbers in another
order, so they have the same mean and variance, and the denominator reduces to $S$. Offsets $k$ and
$n - k$ are the same pairs (put $r' = r + k$ and the two factors swap), so $c_{n-k} = c_k$. At six
seats $\hat\rho = (2c_1 + 2c_2 + c_3)/5$, and only $k = 1, 2, 3$ are reported; at three and at two
seats $\hat\rho = c_1$.

*The exact identity.* Since $u_i - \bar x = \frac{1}{n} \sum_r z_{r,i}$,

$$(D-1)\, s_u^2 = \frac{1}{n^2} \sum_i \Big( \sum_r z_{r,i} \Big)^2
  = \frac{1}{n^2} \sum_{k=0}^{n-1} \sum_{r,i} z_{r,i}\, z_{(r+k) \bmod n,\, i}
  = \frac{S}{n^2} \Big( 1 + \sum_{k=1}^{n-1} c_k \Big)
  = \frac{S \big(1 + (n-1)\hat\rho\big)}{n^2}.$$

The second step groups the $n^2$ pairs of tables by their offset, and $k = 0$ gives $S$. With
$\hat\sigma^2 = S / (nD - 1)$,

$$\widehat{\text{ratio}} = \frac{n (D-1)}{(nD - 1)\big(1 + (n-1)\hat\rho\big)}.$$

The first factor counts degrees of freedom: 0.99958 at $n = 6$, $D = 2{,}000$. This is an identity,
not an approximation: over the real run's 1,900 players (post hoc) the largest relative gap is
$1.1 \times 10^{-15}$. It also gives the bound in the sample: $1 + (n-1)\hat\rho \ge 0$ always.

*Hands needed.* For a 95% interval of $\pm h$ bb per hand: $(1.96\,\hat\sigma / h)^2$ independent
and $n (1.96\, s_u / h)^2$ duplicate. Their quotient is the ratio, up to rounding up. A player with
$s_u = 0$ would get `RATIO_CAP` and the flag `no_variance_left`; no player in the real run did.

**4. A cluster bootstrap over lineups.** An aggregate here is a median over players. Its interval is
a percentile bootstrap (Efron and Tibshirani 1993): draw lineups with replacement, pool the drawn
lineups' players, take the median, repeat 2,000 times and read the 2.5% and 97.5% quantiles.

*Why lineups, not players.* Players at a table share cards and opponents, and their results sum to
zero hand by hand, so they are not independent draws. Heads-up is the extreme: the two players'
results are opposite, so their ratios are equal (in all 200 heads-up lineups, checked post hoc).
Resampling players would count one number as two independent ones. Resampling whole groups is a
*cluster bootstrap*. A test requires that pairs of equal values give exactly the interval of single
values. Post hoc, resampling players instead would have given intervals 1.2 to 1.5 times narrower:
[1.284, 1.311] against [1.278, 1.319] for the six-bot random mix. For the same reason a heads-up
lineup counts twice among the players and once in the bootstrap, and a heads-up ratio by archetype
describes the pair.

*Few clusters.* A percentile interval is too narrow when clusters are few. Worked out post hoc:
for the mean of $G$ normal values with sample SD $\hat\tau$, the bootstrap distribution has
variance $\frac{G-1}{G} \hat\tau^2 / G$, so, reading it as normal, its 95% interval has half-width
about $1.96 \sqrt{(G-1)/G}\; \hat\tau / \sqrt{G}$, where the exact interval has the 97.5% quantile
of $t_{G-1}$ in place of $1.96 \sqrt{(G-1)/G}$. Its coverage is then
$P\big(|t_{G-1}| < 1.96 \sqrt{(G-1)/G}\big)$:

| Clusters $G$ | 5 | 8 | 10 | 20 | 50 | 100 |
|---|---|---|---|---|---|---|
| Coverage of a nominal 95% interval, mean of normal data | 84.6% | 89.1% | 90.4% | 92.9% | 94.2% | 94.6% |

Development had met this: 2 lineups per archetype gave a CI of [1.84×, 1.85×] inside an IQR of
[1.80×, 1.88×]. So the experiment leaves a CI blank below `MIN_CI_LINEUPS = 8`, and no prediction
uses a CI. Is ten still crude? A post hoc check on the real run's lineups says yes. Take one kind's
lineups as the truth, draw $G$ of them with replacement, compute the interval (500 resamples),
repeat 2,000 times (Monte Carlo error about 0.7 points) and count how often it covers the kind's
median:

| Lineup kind | $G$ = 8 | $G$ = 10 | $G$ = 20 | $G$ = 50 |
|---|---|---|---|---|
| Six copies of one bot | 88.2% | 89.4% | 93.0% | 94.2% |
| Six bots of one archetype | 89.2% | 90.5% | 93.1% | 93.6% |
| Six bots, random mix | 89.6% | 90.5% | 93.0% | 94.0% |
| Three bots, random mix | 90.0% | 91.8% | 94.0% | 93.8% |
| Two bots, random mix | 91.5% | 93.5% | 94.4% | 94.0% |

The same lineups are both the data and the truth, so this shows the size of the effect, not the
exact coverage on new lineups. Read the per-archetype CIs of the copies and one-archetype lineups,
which rest on 10 lineups each, as roughly 90% intervals: normal theory gives 90.4% at $G = 10$,
and the table pools archetypes, so it does not test one archetype's own 10 lineups. The other CIs
rest on 42 to 200 lineups.
Cameron, Gelbach and Miller (2008) study a related problem for regressions with few clusters, where
standard tests reject too often; their remedy, a cluster bootstrap-t, is not used here.

*A percentile interval need not contain the median.* Its ends are quantiles of the resampled
medians, not the sample median plus or minus anything, and nothing forces the sample median between
them. Figures that draw `median - lower` as an error bar must allow for that. In the real run and in
every development output each reported CI does contain its median (checked post hoc).

*Conditional on the populations.* A bot can sit in several mixed lineups of a replicate: the ten
`mixed_6` lineups of a replicate seat 35 to 43 distinct bots in 60 seats (post hoc). Resampling
lineups treats them as independent, so these CIs are conditional on the populations and a little
narrow.

**5. Seeds and stream tags.** Every random stream comes from a `SeedSequence` over a short list of
integers:

| Stream | Entropy |
|---|---|
| A replicate's population | `[replicate seed, 1]` |
| A kind's lineup draws | `[replicate seed, 2, kind]` |
| A chunk of a lineup's deals | `[replicate seed, 3, kind, lineup, chunk]` |
| The calling check | `[master seed, 4, chunk]` |
| The bootstrap | `[master seed, 5, kind]` and `[master seed, 5, kind, archetype]` |

A `SeedSequence` mixes a pool of four 32-bit words, and shorter entropy is padded with zeros. So
`[5]`, `[5, 0]`, `[5, 0, 0]` and `[5, 0, 0, 0]` give the same numbers, while `[5, 0, 0, 0, 0]` does
not (checked for this step on NumPy 2.4.6). With `[seed]` for one stream and `[seed, kind]` for
another, kind 0 would silently reuse the first. Hence the second word, a tag from 1 to 5 that
differs per stream and is never 0. Archetypes in the bootstrap's entropy count from 1, so the first
archetype's stream is not the kind's own. The full plan has 4,001 worker tasks with 4,001 distinct
seeds (checked post hoc). The state words also form a prefix: `generate_state(3)` is the start of
any longer request. $n = 2$ does not need that, since it asks for exactly the old three words; it
means a larger table shares its deck seed and first two table seeds with a heads-up match of the
same seed. Results are aggregated in task order, so they do not depend on the number of workers (a
test runs one and two).

**The plan and predictions** (`experiments/phase2_duplicate.py`, pre-registered at commit
`0fb6e53`).

*The simulation.* Per replicate, a population of 12 bots per archetype (60 bots: Phase 1's
`make_population`, `ParametricBot` with 150 Monte Carlo equity samples). Blinds 1/2, 100 bb stacks
reset every hand, table size $n$. Each lineup plays 2,000 deals in chunks of 250, each chunk one
`duplicate_match` run as one worker task. A chunk restarts the button at seat 0, which does not
matter, since every player holds every position once per deal.

| Lineup kind | Who sits | Lineups per replicate | Lineups in all | Hands per replicate |
|---|---|---|---|---|
| Six copies of one bot (`clones_6`) | one bot drawn per archetype, six copies | 5 | 50 | 60,000 |
| Six bots of one archetype (`mirror_6`) | six different bots, one lineup per archetype | 5 | 50 | 60,000 |
| Six bots, random mix (`mixed_6`) | drawn from the population, without replacement within a lineup | 10 | 100 | 120,000 |
| Three bots, random mix (`mixed_3`) | the same, at a table of three | 10 | 100 | 60,000 |
| Two bots, random mix (`mixed_2`) | the same, at a table of two | 20 | 200 | 80,000 |
| Six always-call agents (`calling_6`) | a check, once per run, 200 deals | | 1 | 1,200 in the run |

Ten replicates make 500 lineups and 3.8 million hands, plus the check's 1,200.

*From pilot to thresholds.* Pre-registration here had four stages:

1. *Pilot* on seed 97: 1 replicate, 250 deals per lineup, 26,300 hands, 15 s. It sized the next
   runs.
2. *Development* on seeds 98 and 99: 2 replicates, 1,500 deals per lineup, 5, 5, 6, 6 and 6 lineups:
   378,000 hands plus the check, 137-139 s each on 12 workers. The outputs are not committed; what
   they showed is in the docstring as "Seen in development".
3. *Sizing and thresholds*, on development data only. A subagent's precision analysis found that the
   median ratio did not drift with the deals between 250 and 1,500 and that the spread between
   lineups dominates the CI. So the full configuration adds lineups, not deals. Against the draft
   configuration it has 10 replicates, not 5, and 20 heads-up lineups per replicate, not 10, since a
   heads-up lineup is one value and had the widest CI (±9% projected). It keeps the draft's 10
   lineups of each other mixed kind (development had 6) and 2,000 deals per lineup (development had
   1,500). All of these are pre-registered in `experiments/configs/phase2.yaml` at `0fb6e53`; the
   docstring states only the 10 and the 20. Each threshold was set from what the two seeds showed.
4. *Commit, then run.* A reviewer read the plan, its findings were fixed (Mistakes), and `0fb6e53`
   was committed at 21:13:38 on 2026-11-03. The real run started from it, on a clean tree, at
   21:13:52.

*The predictions.* Twelve, in five families: A, the calling check is exact; B, the size of the gain,
that it falls from copies to one archetype to a random mix, and that almost no player loses; C, nits
gain least; D, the correlation barely depends on the offset; E, the gain stays flat with $n$. The
table under Results has them all.

*What a threshold can and cannot show.* A1 can fail only through a bug, and the run stops there if
it does. B1, B2 and E1 are bands, wide on purpose: B1's runs from 1.15 to 1.5, 8.5 times the width
of the final CI (post hoc). They fail only if the two development seeds misled badly, and holding
says nothing about where inside the band the value lies; the CI does that. The narrowest are B3, C1
and C2, orderings that could have come out otherwise. Development also shows what not to predict:
nits came lowest among one-archetype lineups only on seed 98 (1.17× | 1.35×, with TAG at 1.34× on
seed 99), so that was left out. And all twelve hold on both development seeds by construction, since
the thresholds came from them. A threshold also belongs to a run size. Evaluated post hoc on their
saved outputs, the pilot's 250 deals fail C2, D1, D2 and D3, and the quick configuration's 100 deals
(seed 98) fail B4, D1, D2 and D3, so `--quick` is only a smoke test.

**How it was done.** The work ran in a git worktree outside OneDrive (`poker-lab-dup`). `8a5f983`
generalizes the library: `duplicate_match` takes $n \ge 2$ players, `DuplicateMatch` keeps its
tables (`first` and `second` remain) and adds `grouped_results` and `evaluate`, and
`DuplicateDealEvaluator` takes `group_size`; the bridge rerun checked it. `0fb6e53` adds the
experiment, its plan and predictions and a new last step of `run_phase2`. `00f8f7e` holds the real
run's results and a new explorer section (`duplicate-n`, "Duplicate dealing at more than two seats")
with a tile on the Phase 2 page. `c7f0029` adds `experiments/phase2_duplicate_posthoc.py`, which
reads the committed JSON and plays no hands, and redraws one figure; `8cdd759` holds its results,
from a clean tree, and a post hoc explorer section (`duplicate-posthoc`); `4150aac` corrects two
explorer sentences. A reviewer subagent read the library diff and the experiment before their
commits, and its findings were fixed first (Mistakes). The suite went from 471 tests to 479
(library), 511 (experiment) and 517 (post hoc). The real run was
`python -m experiments.phase2_duplicate --seed 2026`; `run_phase2 --only duplicate` calls the same
function, and a test covers that wiring.

**Results** (seed 2026, commit `0fb6e53`, clean tree;
[`duplicate_summary.md`](../experiments/results/phase2/duplicate/duplicate_summary.md); the figures
`duplicate_ratio`, `duplicate_offsets` and `duplicate_players` are in the same folder).

*By lineup kind.* The ratio is a median over players, pooled over lineups and replicates:

| Lineup kind | Lineups | Players | Ratio: median [IQR] | 95% CI of the median | Players with a ratio below 1 | Median $\rho$ | Median $c_k$, $k$ = 1, 2, 3 | Median SD per hand (bb) | Median hands for ±5 bb/100: independent | duplicate |
|---|---|---|---|---|---|---|---|---|---|---|
| Six copies of one bot | 50 | 300 | 1.53× [1.42, 1.64] | [1.49, 1.57] | 0 | -0.069 | -0.067, -0.071, -0.067 | 19.27 | 570,490 | 388,778 |
| Six bots of one archetype | 50 | 300 | 1.44× [1.37, 1.52] | [1.43, 1.47] | 1 | -0.061 | -0.059, -0.064, -0.062 | 19.92 | 610,030 | 421,662 |
| Six bots, random mix | 100 | 600 | 1.30× [1.21, 1.40] | [1.28, 1.32] | 1 | -0.046 | -0.040, -0.040, -0.041 | 24.74 | 940,565 | 713,764 |
| Three bots, random mix | 100 | 300 | 1.25× [1.16, 1.35] | [1.22, 1.28] | 2 | -0.100 | -0.100 | 14.42 | 319,506 | 276,703 |
| Two bots, random mix | 200 | 400 | 1.29× [1.17, 1.44] | [1.24, 1.32] | 4 | -0.224 | -0.224 | 8.29 | 105,680 | 86,536 |

The hands are separate medians over the players, so their quotient is not the median ratio (940,565
/ 713,764 = 1.32, against 1.30). Six always-call agents cancel exactly: over 200 deals every
per-deal mean is 0 chips, while a single hand has an SD of 2.19 bb. (Post hoc check of that number:
six callers put in 1 bb each, the winner nets 5 bb, so without ties the variance is
$\frac16 \cdot 25 + \frac56 = 5$, an SD of 2.24 bb; split pots lower it.)

*By the player's archetype.* Median ratio [95% CI], then median SD per hand in bb. Each cell of the
first two columns rests on 10 lineups, so its CI is about a 90% interval (section 4):

| Archetype | Six copies of one bot | Six bots of one archetype | Six bots, random mix |
|---|---|---|---|
| Nit | 1.34× [1.27, 1.42], 3.50 | 1.31× [1.26, 1.40], 4.78 | 1.18× [1.16, 1.20], 13.54 |
| TAG | 1.65× [1.55, 1.72], 13.64 | 1.43× [1.37, 1.49], 10.87 | 1.27× [1.24, 1.29], 17.91 |
| LAG | 1.53× [1.50, 1.59], 20.49 | 1.45× [1.42, 1.49], 20.25 | 1.34× [1.32, 1.36], 25.13 |
| Calling station | 1.57× [1.47, 1.67], 24.47 | 1.52× [1.49, 1.58], 27.67 | 1.43× [1.40, 1.46], 38.11 |
| Maniac | 1.51× [1.48, 1.56], 49.83 | 1.46× [1.43, 1.49], 47.58 | 1.30× [1.28, 1.34], 41.19 |

*By the number of players* (random mixes):

| Players $n$ | Median ratio | 95% CI | Median $\rho$ | Bound $-1/(n-1)$ | Median SD per hand (bb) | Median hands for ±5 bb/100: independent | duplicate |
|---|---|---|---|---|---|---|---|
| 2 | 1.29× | [1.24, 1.32] | -0.224 | -1 | 8.29 | 105,680 | 86,536 |
| 3 | 1.25× | [1.22, 1.28] | -0.100 | -0.5 | 14.42 | 319,506 | 276,703 |
| 6 | 1.30× | [1.28, 1.32] | -0.046 | -0.2 | 24.74 | 940,565 | 713,764 |

The heads-up push/fold reference (Step 28's matchups against the bots, 30 of them, the median of
each matchup's naive / duplicate hands) is 2.06×: 2.07× at 5 bb, 2.30× at 10 bb and 2.01× at 20 bb.

*Predictions against results.* Seeds 98 and 99 are the development runs the thresholds came from:

| | Prediction | Seed 98 | Seed 99 | Seed 2026 | Verdict |
|---|---|---|---|---|---|
| A1 | Calling check: every per-deal mean is exactly 0 chips | 0 | 0 | 0 | holds |
| B1 | Six bots, random mix: median ratio in [1.15, 1.5] | 1.33 | 1.26 | 1.30 | holds |
| B2 | Six copies of one bot: median ratio in [1.35, 1.9] | 1.60 | 1.54 | 1.53 | holds |
| B3 | The median ratio falls: copies, one archetype, random mix | 1.60, 1.46, 1.33 | 1.54, 1.45, 1.26 | 1.53, 1.44, 1.30 | holds |
| B4 | At most 5% of the players have a ratio below 1, in each six-player kind | 0, 1.7%, 0 | 0, 0, 1.4% | 0, 0.33%, 0.17% | holds |
| C1 | Random mix: nits have the lowest median ratio of the five archetypes | 1.20, next 1.32 | 1.13, next 1.28 | 1.18, next 1.27 | holds |
| C2 | Copies of one bot: nits have the lowest median ratio of the five archetypes | 1.35, next 1.56 | 1.34, next 1.49 | 1.34, next 1.51 | holds |
| D1 | Copies of one bot: medians at offsets 1-3 within 0.03 | range 0.009 | range 0.010 | range 0.005 | holds |
| D2 | One archetype: medians at offsets 1-3 within 0.03 | range 0.004 | range 0.013 | range 0.005 | holds |
| D3 | Random mix: medians at offsets 1-3 within 0.03 | range 0.012 | range 0.010 | range 0.001 | holds |
| E1 | Random mixes: median ratio in [1.1, 1.6] at $n$ = 2, 3 and 6 | 1.35, 1.23, 1.33 | 1.40, 1.35, 1.26 | 1.29, 1.25, 1.30 | holds |
| E2 | Random mixes: median $\rho$ at most -0.02 at $n$ = 2, 3 and 6 | -0.260, -0.094, -0.050 | -0.286, -0.130, -0.041 | -0.224, -0.100, -0.046 | holds |

All twelve hold. That is a replication with the same simulation and bots, on more lineups and deals,
not a strong test: the thresholds came from two seeds, and several are wide on purpose. The
orderings B3, C1 and C2 are the parts that could have gone otherwise. Post hoc, the CIs in them do
not overlap: [1.49, 1.57], [1.43, 1.47], [1.28, 1.32] for B3; nits [1.16, 1.20] against TAG [1.24,
1.29] for C1; nits [1.27, 1.42] against maniacs [1.48, 1.56] for C2 (at 10 lineups, so about 90%
intervals).

What differed from development, none of it predicted:

* Among copies TAG has the highest ratio on seed 2026 (1.65×). Calling stations were highest on
  seeds 98 (1.84×) and 99 (1.58×).
* In the random mix calling stations are highest on seeds 98 and 2026, LAG on seed 99.
* Among one-archetype lineups nits are lowest on seed 2026 (1.31×, next TAG at 1.43×), as on seed 98
  and not on seed 99.
* Two bots gain less than in either development run: 1.29× against 1.35× and 1.40×, with $\rho$ at
  -0.224 against -0.260 and -0.286. The share of players below 1 is 1% against 8% (2 of 24 players
  on each development seed, so one lineup).

*Post hoc: the share of perfect cancellation.* Computed after the run, from `duplicate.json`, by
`experiments/phase2_duplicate_posthoc.py` ([`duplicate_posthoc_summary.md`](../experiments/results/phase2/duplicate/duplicate_posthoc_summary.md)),
which also gives the coverage tables of section 4 and most of the readings below:

| Lineup kind | $n$ | Median $\rho$ | Bound | Share $s = -(n-1)\rho$: median [IQR] | $1/(1-s)$ | Measured median ratio |
|---|---|---|---|---|---|---|
| Six copies of one bot | 6 | -0.069 | -0.2 | 34.5% [29.7, 39.1] | 1.526 | 1.525 |
| Six bots of one archetype | 6 | -0.061 | -0.2 | 30.7% [27.0, 34.3] | 1.444 | 1.443 |
| Six bots, random mix | 6 | -0.046 | -0.2 | 23.0% [17.3, 28.4] | 1.299 | 1.298 |
| Three bots, random mix | 3 | -0.100 | -0.5 | 20.0% [13.6, 26.1] | 1.251 | 1.250 |
| Two bots, random mix | 2 | -0.224 | -1 | 22.4% [14.6, 30.4] | 1.289 | 1.288 |

* *The share is flat, and the ratio with it.* $\rho$ is smaller at six seats, and so is its bound;
  measured against the bound the three random mixes reach about the same share, 20% to 23%. Since
  the ratio is $1/(1-s)$, this restates the flat ratio in comparable units; why the share stays flat
  was not examined. The algebra does not require it: $\rho$ could have stayed near -0.05 at every
  $n$, which would give about 1.05× at two seats. It is a measurement. (The last two columns differ
  by the degrees-of-freedom factor and by how a median interpolates.) Development had already shown
  $(n-1)\rho$ about -0.2 to -0.3 at every $n$; the share's values are post hoc.
* *Copies stay well below perfect cancellation.* Six copies reach about 35%. By section 2 the bots'
  own randomness must be the cause: `ParametricBot` reads no player ids, all its randomness comes
  from the table's RNG, and each table draws its own numbers. What was NOT tested is how the
  shortfall divides between mixed actions, bet-size noise and Monte Carlo equity, or that the plays
  of a deal part early in the hand. A test would switch those sources off one at a time; with all of
  them off, section 2 makes the cancellation exact, as for `CallingAgent`.
* *Depth, not seats?* Heads-up at 100 bb these bots gain 1.29×. Heads-up push/fold (5-20 bb, every
  hand all-in or fold, Step 28) gains 2.06×. Same $n$, a different game. This is not an experiment:
  the bots and the heroes differ too, so it is only a suggestion.
* *Who plays alike tends to cancel more, with exceptions.* Heads-up lineups of one archetype have a
  median of 1.46× (33 lineups), mixed pairs 1.26× (167); maniac against nit is lowest, at 1.07× (19
  lineups). But the two highest pairs are mixed, calling station with maniac and with LAG, at 1.49×
  and 1.48× (18 and 8 lineups), and nit against nit is only 1.25× (8). Grouped after seeing the
  results. It fits B3 and Step 28's note that duplicate dealing helps most against opponents who
  play like the hero.
* *The ratio does not simply follow the SD.* Per-hand SD runs from 3.50 bb (nit copies) to 49.83 bb
  (maniac copies). Among copies TAGs (13.64 bb) gain most (1.65×), while maniacs gain 1.51×. Over
  players within a kind the rank correlation of ratio and SD is only 0.15 to 0.52. A nit's own SD
  depends on its company: 3.50 bb among copies, 13.54 bb in the random mix. Why nits gain least was
  not examined.
* *Medians do not average.* In the random mix the median $\rho$, -0.046, lies below all three offset
  medians (-0.040, -0.040, -0.041). Each offset's distribution over players is skewed to the low
  side (skewness -0.84, -0.31, -0.97), so its mean lies below its median (-0.045, -0.043, -0.051),
  and $\rho$, their weighted mean, has mean -0.045 and little skew. Offset 3 is also wider (IQR
  [-0.080, -0.014] against [-0.063, -0.021] at offset 1); it has half as many distinct pairs, $3D$
  rather than $6D$.
* *How noisy one player's ratio is.* Six copies of one bot in a rotated match differ only by
  sampling noise, and within a lineup their ratios have a median relative SD of 6.6% (7.0% and 7.4%
  on the development seeds). That is about the size Step 38's kurtosis formula gives: the relative
  standard error of the variance $s_u^2$ is about $\sqrt{(\kappa - 1)/D}$, and on the development
  seeds' per-hand arrays the median kurtosis of per-deal means in the six-player kinds is 9 to 17
  (14 to 16 among copies), which at $D = 1{,}500$ gives 7% to 10% (9% to 10% among copies). A
  lineup's six per-deal means sum to zero, so their variances share noise, which may be why the
  spread within a lineup comes in lower (not tested). The ratio inherits that noise, so the
  quartiles and the share below 1 include it. Between bots the spread is wider: among copies, lineup
  medians run from 1.15× (a nit) to 2.47× (a calling station). One heads-up lineup, TAG against nit,
  reached 6.08× ($\rho = -0.836$, SD 3.25 bb); the next highest is 2.33×.
* *It does not make win rates cheap.* For ±5 bb/100 in a random mix of six, the median player needs
  940,565 independent hands and 713,764 duplicate ones.

**Run times, awake only** (C.3).

* The real run took 1,455 s (`elapsed_seconds` 1,454.7, about 24 minutes) on 12 workers, plugged
  in: 21:13:52 to 21:38:17 on 2026-11-03 by the shell's clock. That is 3,801,200 hands, about 2,600
  hands/s.
* The machine stayed awake. The System log shows Modern Standby entered at 21:07:31 with the reason
  Idle Timeout, before the start and with a keep-awake request held, and no later Kernel-Power 506
  or 507 event until after the end. A memory sampler wrote 140 samples through the whole run, 10-13
  s apart. As in Steps 37, 39 and 40, that standby counts as awake.
* Memory was tight, with the owner's applications open: free physical memory was 0.11 to 1.59 GB,
  mean 0.75 GB, and the commit charge (virtual memory in use) 50.1 to 53.9 GB, up from 40.0 GB at
  the first sample and back to 38.7 GB at the last. Just before the run, nearly idle, the machine
  paged about 2,100 pages/s; paging was not measured during it. All this can change only the run
  time, and the single real run has nothing to compare with. The development runs did 2,730-2,760
  hands/s on another day, on battery, with fewer lineups of each mixed kind.
* The bridge rerun's wall time spans a closed lid, so it is not quoted.
* Phase 2 as a whole: the step times recorded in the committed results add up to 2,326 s, 39 minutes
  (cfr 46.3, deep_cfr 620.6, exploit 185.5, aivat 19.3, duplicate 1,454.7). They come from different
  days and code versions (Step 40's verification ran Phase 2's exploit in 95 s), and `run_phase2`
  was not run whole with the new step.
* Test suite. 517 tests took 127.2 s on 2026-11-03 with the laptop plugged in (no memory reading
  for that run), and 123.1 and 103.8 s on 2026-11-04 on battery with 5.6-5.8 GB
  available and no paging. A third run that day took 300.1 s, while the laptop went from battery to
  plugged in, and a fourth, plugged in with about 3.4 GB available, 117.7 s. Before this step the
  suite took 53.5-53.9 s (471 tests) and 61.6 s (479). The two-replicates test (Step 40's
  `test_two_replicate_jobs_write_what_one_writes`) took 86-87 s in the battery runs and 119 s in a
  511-test run of 158.7 s on the paging evening, against 44 s under the suite's load in Step 40, and
  it sets a floor under the suite's time. The new tests add little (the new end-to-end test takes
  12-15 s inside a run), so most of the rise since Step 40 is that one test, and why it now takes
  about twice as long, plugged in too, was **not examined**. The end-to-end test is marked `slow`,
  not `long`, since two long tests would queue on one worker.

**What was left.**

* How the copies' shortfall divides among the bots' sources of randomness: rerun `clones_6` with
  each switched off. ([Step 42](#step-42-where-six-copies-shortfall-comes-from-p211c) does this.)
* Depth against seats: the same bots at several stack depths, instead of comparing with Step 28.
* The seating order: replay some lineups in other cyclic orders.
* Control variates on top: this step scores chips only. Step 28's all-in EV and Step 38's AIVAT were
  not combined with the rotation, and hold'em still has no AIVAT value function.
* Why the two-replicates test takes about twice as long as in Step 40, and a whole `run_phase2` with
  the new step.

**What it means.** Rotating the seats works, modestly. At 6-max with these bots it needs 1.30× fewer
hands, about the same at three and two seats (1.25×, 1.29×), and even six copies of one bot reach
only 1.53×. The useful lesson is in the algebra: a deal at $n$ tables costs $n$ hands, so everything
rests on the correlation between tables, and that correlation has a floor of $-1/(n-1)$. A small
$\rho$ at six seats does not by itself mean a weak effect, because six results cannot all oppose
each other. Measured against the floor the random mixes cancel about a fifth of what could cancel,
at every table size tried. For simulated evaluations this is a modest gain worth taking: the code
exists, no extra hands are needed and the point estimate does not change. It does not make 6-max win
rates cheap (713,764 hands for ±5 bb/100), and it does nothing at a live table. All twelve
predictions hold, which says the two development seeds were not misleading; it does not explain
anything. Why the share stays flat, how the bots' randomness divides the copies' shortfall and
whether depth explains the gap to push/fold are untested.

**Mistakes.**

* The plan first cited Step 27 for the push/fold measurement; it is Step 28 (P3.3). It also quoted
  "about 3x", a ratio of two medians (32,073 / 10,733 = 2.99), while the code compares the median
  over matchups of each matchup's naive / duplicate (2.07× at 5 bb, 2.30× at 10 bb, 2.01× at 20 bb,
  2.06× overall). The reviewer caught both.
* `predictions()` ran before `save_json`, and a prediction's path stopping at a dict would have
  raised `TypeError` after the whole simulation, before anything was saved. The reviewer caught it.
  A non-number now counts as a failed prediction with the value None, and a test evaluates every
  prediction on made-up results of the real shape.
* Per-archetype CIs for copies and one-archetype lineups rest on one lineup per replicate, and in
  development 2 lineups gave a CI of [1.84×, 1.85×] inside an IQR of [1.80×, 1.88×]. The reviewer
  caught it. CIs are now blank below `MIN_CI_LINEUPS = 8`, no prediction uses them, and the full run
  has 10 lineups per archetype there. Ten is still crude: about 90% coverage (section 4, post hoc).
* A figure's error bars could have been negative, since a percentile CI need not contain the median,
  and matplotlib 3.11 refuses a negative error. The reviewer caught it; the code clips them. A stale
  heads-up fallback was removed, and two labels were fixed ("Median hands", "rho: median").
* The library's docstring claimed exactness for "identical agents whose play depends only on their
  cards", which is too narrow and incomplete: agents can see player ids. It now says copies of one
  deterministic agent that ignores player ids, and a test with an ante covers odd chips. The
  reviewer caught it.
* `DuplicateDealEvaluator()` on a 6-table match would have averaged tables two at a time with no
  error. The reviewer caught it. `paired_results` now refuses any number of tables but two, and
  `DuplicateMatch.evaluate` groups by the number of tables.
* Also from the library's review: the rotation's fixed neighbors are now documented,
  `sd_bb_per_hand` is documented as the raw per-hand SD and not the spread behind the interval, and
  a stale TODO (a Phase 5 engine hook to replay decks) was removed, since this is simulation only.
* The plan's docstring says copies of one bot "spread about 2% in development". Over all copies
  lineups the median within-lineup relative SD is 7.0% and 7.4% on those seeds (post hoc). No
  prediction used the figure, and the committed plan stays as it was.
* The reviewer of this step, tracing every number and claim, caught these in the draft. The bots'
  randomness was offered as a likely reason copies stop short, though section 2 makes it the only
  possible one (what is untested is how it divides), and the proposed test, deterministic bots,
  could only have confirmed the proof. A flat share was said to explain a flat ratio, though the
  ratio is $1/(1-s)$: one fact, stated twice. "Who plays alike cancels more" was too strong, since
  the two highest heads-up pairs are mixed. A spread was rounded twice (7.0% and 7.4%), a test's 118
  s was attached to the wrong run, and heads-up's seeds were credited to a property it does not use.
  The first two claims were also in the explorer (`4150aac`) and the README.
* The `duplicate_ratio` figure's legend sat on the two dots of the 6.08× heads-up lineup, found
  while drafting this step. The legend moved and the figure was redrawn from the committed JSON (no
  number changed; the other figures are pixel-identical). The explorer's "independent dealing"
  label, which sat on the first dot, became "no gain".

**Limitations.**

* One real seed, and Phase 1's parametric bots at 100 bb stacks. The bots come from one family, and
  nothing here says what people would do.
* Each lineup measures one seating order, of 120 cyclic orders at six seats. The aggregates average
  over the orders drawn, not over all of them.
* Mixed lineups reuse bots within a replicate, so their CIs are conditional on the populations and a
  little narrow. Per-archetype CIs at 10 lineups are about 90% intervals.
* $\hat\sigma^2$ is pooled over positions. An independent table with a moving button also balances
  positions, so the variance that matters there leaves out the variance between the positions' mean
  results, which $\hat\sigma^2$ includes. The real run did not record it. On the development seeds'
  per-hand arrays (post hoc) that part is a median of 0.03% to 0.12% of $\hat\sigma^2$ within a
  kind, and removing it moves a kind's median ratio by at most 0.0024.
* The comparison with push/fold is across games, bots and heroes, and the reason offered for the gap
  is untested; so is how the bots' randomness divides the copies' shortfall.
* Nothing here scores all-in EV or control variates, so the hands needed are for chips alone.
* The run time is a single run on a machine short of memory.

**Verification.**

* `tests/test_duplicate.py` (3 → 10 tests):
  * With two players the code gives the hands of the old algorithm, rebuilt inline from commit
    `2131bb4`: every hand's serialized form is equal, and the factories are called in the same
    order.
  * With six players, and three at a six-seat table: same cards per seat and same button at all
    tables, player $j$ in seat $(j + r) \bmod n$, each player in each seat once per deal, one shared
    run-out.
  * Six always-call agents: each seat's result is the same at all six tables, every player's results
    over a deal sum to zero, and the duplicate interval has zero width while the naive one is over
    10 bb/100.
  * Three always-call agents with an ante of 1: odd-chip two-way splits occur, and they still
    cancel.
  * `grouped_results` runs deal by deal, refuses a player who missed hands, and `paired_results`
    refuses anything but two tables. Bad player lists are rejected before any agent is built.
* `tests/test_evaluator.py` (5 → 6 tests): `group_size=3` with luck that sums to zero gives an
  interval under 0.2 times the naive one and the right method name; results that do not fill a group
  and a `group_size` of 1, 0 or 2.0 are refused.
* `tests/test_duplicate_experiment.py` (32 tests, counting parameter cases):
  * The metric on constructed arrays. At $n$ = 2, 3 and 6: identical tables give $\rho = 1$ and
    ratio $(D-1)/(nD-1)$; perfectly cancelling tables give $\rho = -1/(n-1)$, `RATIO_CAP` with the
    flag and valid JSON. At $n = 6$: independent tables give about 1. Neighbors sharing a term give
    offsets (0.5, 0, 0), $\rho = 0.2$ and ratio 0.5, each offset equal to `np.corrcoef`; the ratio
    equals the identity from $\rho$; an alternating pattern at $n = 4$ gives offsets (-1, 1). Hands
    needed follow the formulas.
  * The bootstrap resamples lineups, drops empty ones and is seeded; pairs of equal values give the
    interval of single values; bounds are NaN below `MIN_CI_LINEUPS`.
  * Predictions: the table's tests (including ten cases of `decreasing`, `first_lowest`,
    `range_at_most` and an empty list), a path that stops at a dict fails instead of raising, an
    unknown test is refused, and every prediction reads numbers from made-up results of the real
    shape.
  * Lineups are drawn as specified, replicates have different populations, chunk seeds are distinct,
    the plan is reproducible, a kind's lineups do not depend on the other kinds' counts, and bad
    configs are refused.
  * A worker task's array matches a `duplicate_match` hand by hand and sums to zero per hand, its
    win rates match the library's evaluators, and the same seed gives the same bytes. The calling
    check is exact, and it detects a broken array.
  * `run_phase2` runs `duplicate` last and passes its config; both configs are valid.
  * (`slow`) A tiny run writes the JSON, summary and figures at $n$ = 2, 3 and 6, with one worker
    and with two: `compare_step` finds no difference.
* `tests/test_duplicate_posthoc.py` (6 tests): the normal-theory coverage against a simulation and
  its rise to 95%; lineup coverage below nominal with few lineups; heads-up pairs count a lineup
  once; the share gives the ratio at any table size; a run on the committed results, with few
  trials, is JSON-safe, with the identity from $\rho$ exact and the implied ratios within 1%.
* `tests/test_explorer.py` covers the new sections.
* Post hoc: `python -m experiments.phase2_duplicate_posthoc` (73 s, from the committed JSON).
* Experiment: `python -m experiments.phase2_duplicate --seed 2026` (1,455 s), or
  `python -m experiments.run_phase2 --seed 2026 --only duplicate`; smoke test with `--quick` (this
  step: 6.1 s on 2026-11-02 on battery, 17.4 s on 2026-11-03 plugged in).
* Heads-up unchanged: `python -m experiments.run_phase3 --seed 2026 --only bridge --out DIR`, then
  `python -m experiments.compare_runs experiments/results/phase3 DIR bridge` ("bridge: 0
  differences").
* Suite: `python -m pytest` (517 tests). Explorer: `python -m experiments.explorer`.

## Step 42: Where six copies' shortfall comes from (P2.11c)

**The question.** Step 41 rotated the seats at six tables and found that even six copies of one bot
reach only 1.53×, about 35% of the way to perfect cancellation. Its argument said the bots' own
randomness is the only possible cause, and it left open how the shortfall divides between the three
sources: mixed actions (a bot choosing between actions at random), bet-size noise (a random factor
on a raise) and Monte Carlo equity (a sampled estimate of hand strength). This step plays Step 41's
`clones_6` lineups again with those sources changed one at a time. The code is
`experiments/phase2_randomness.py`. `poker_lab/bots` and `experiments/phase2_duplicate.py` are not
edited, so the simulation fingerprint and every committed result stay where they were.

**1. What the shortfall measures, for copies.**

Take a deal with cards $C$. Let $Y_j$ be the result of whoever sits in seat $j$ of one table. The
bots are copies, and `ParametricBot` reads no player ids (Step 41), so given $C$ the six tables are independent runs of the same
experiment. A player's mean over the tables, $u = \frac{1}{n} \sum_r Y_{j(r),\,r}$, holds each seat
exactly once. Two facts follow.

*The mean given the cards is zero.* $\operatorname{E}[u \mid C] = \frac{1}{n} \sum_j
\operatorname{E}[Y_j \mid C]$, and a hand without rake is zero-sum, so the seats' expected results
add to 0. The cards alone leave nothing in $u$.

*The variance given the cards is a sum.* The tables are independent given $C$, so the variances add:

$$\operatorname{Var}(u) = \operatorname{E}\big[\operatorname{Var}(u \mid C)\big]
  = \frac{1}{n^2} \sum_j \operatorname{E}\big[\operatorname{Var}(Y_j \mid C)\big]
  = \frac{1}{n} \cdot \frac{1}{n} \sum_j \operatorname{E}\big[\operatorname{Var}(Y_j \mid C)\big].$$

The first equality is the law of total variance with its other term, $\operatorname{Var}(
\operatorname{E}[u \mid C])$, equal to zero. With $\sigma^2 = \operatorname{Var}(Y)$ per hand and
Step 41's ratio $= \sigma^2 / (n \operatorname{Var}(u))$,

$$\frac{1}{\text{ratio}} = R = \frac{\operatorname{E}[\operatorname{Var}(Y \mid C)]}{\operatorname{Var}(Y)},$$

averaged over the seats. The law of total variance also says $\operatorname{Var}(Y) =
\operatorname{E}[\operatorname{Var}(Y \mid C)] + \operatorname{Var}(\operatorname{E}[Y \mid C])$. So
$R$ is the share of a hand's variance that comes from the bots' draws, and $1 - R$ the share the
cards (with the seat) explain. The rotation cancels all of $1 - R$ and none of $R$. That changes how
Step 41's number reads: 35% is not a limit of the rotation. It says how much of a hand's result the
cards decide between these bots.

The argument can be tested directly. $\operatorname{E}[\operatorname{Var}(Y \mid C)]$ has its own
estimate, the sample variance of a seat's results across the tables of a deal, and it should equal
$n s_u^2$. They differ by at most 0.8% in every condition but the two exact checks
(`within_deal_variance`, prediction A3).

*Two ways to count.* Step 41's "about 35%" is $-(n-1)\rho$ at the median $\rho$. This step's
metric is the mean over lineups of $1/\text{ratio}$, because means of $1/\text{ratio}$ add and ratios
do not. It gives 33.8% cancelled. The bots and cards are the same; only the summary differs.

**2. Sharing a source instead of switching it off.**

The obvious experiment replaces a draw with a constant, and that changes the bots. Decide every
mixed action with $u = 0.5$ and a bot always takes the branch that holds 0.5, so a bot that bluffs
30% of the time may never bluff. A smaller shortfall would then say nothing about the original bots.

So the main design *shares* the draws: each table still draws at random, but all six use the same
numbers. This is the idea of common random numbers (glossary), applied between the tables of a deal.
A shared draw is a hash:

```text
draw = blake2b(chunk seed | deal | seat | street | pairing key | source)
```

The table number is not in it, so a seat's draw is the same at each table. A cryptographic hash's
output behaves as independent uniforms that do not depend on the situation, so one table on its own
plays hands from the same distribution as the default. Only the correlation between tables changes.
This rests on the construction; predictions B1 to B4 are coarse checks of it (SD and seven playing
stats). `SwitchedBot` subclasses `ParametricBot` and changes only where the draws come from: with
every source live it plays exactly as its parent, draw for draw, which a test in
`tests/test_duplicate_randomness.py` checks. Equity needs
no special case, because the parent already caches its estimate per street, so a street's estimate
is one draw in both.

Why does sharing cancel? Take a made-up decision that bluffs with probability $p$ when $u < p$.
Suppose equity differs a little between two tables, so $p_1 = 0.30$ and $p_2 = 0.35$. With separate
draws the tables disagree with probability $p_1(1 - p_2) + (1 - p_1)p_2 = 0.195 + 0.245 = 0.44$.
With one $u$ they disagree only when $0.30 \le u < 0.35$, probability 0.05. This *coupling* depends
on the two thresholds sitting at the same end of $[0, 1]$. Had one table's bluff branch been laid out
at the other end, one $u$ would give disagreement 0.65, worse than independent draws. Section 4
returns to this.

The identity of section 1 still holds with some sources shared. Given $C$ and the shared draws, the
tables are independent again, the seats' expected results still add to 0, and each seat is held
once. So the remaining share in a condition with live set $S$ is

$$R(S) = \frac{\operatorname{E}[\operatorname{Var}(Y \mid C, \text{the shared sources})]}{\operatorname{Var}(Y)}.$$

**3. Splitting the shortfall into terms.**

*The decomposition.* Under one pairing rule $Y$ is a function of independent inputs: the cards $C$
and the fields of draws $M$, $Z$ and $E$ (mixed, noise, equity). Any square-integrable function of
independent inputs can be written as a sum of orthogonal pieces, one per subset of the inputs, so its
variance splits into one non-negative term per subset (Hoeffding 1948, Sobol' 1993). Group the terms
by which sources they involve, with or without $C$, and call the group for source set $B$ its $W_B$.
Then $W_\varnothing$ is what the cards alone explain, $1 - R$, and

$$R(S) = \sum_{B \,:\, B \cap S \ne \varnothing} W_B,$$

because holding the shared sources fixed removes exactly the terms that involve only them. In
sensitivity analysis $R(S)$ is the *total-effect index* of the group $S$ (Homma and Saltelli 1996).

*Getting the terms: Moebius inversion.* Eight conditions, one per subset of three sources, give the
seven $W_B$. Let $G(T) = R(\text{all}) - R(\text{all but } T)$, the drop from sharing the sources in
$T$. By the sum above $G(T) = \sum_{B \subseteq T} W_B$, a sum over subsets, and such a sum inverts
as $W_B = \sum_{T \subseteq B} (-1)^{|B| - |T|} G(T)$. With this step's decision-key numbers, as
shares of a hand's variance (computed from unrounded values, so the displayed sums can be
off by 0.1):

* $W_{M} = R(\text{all}) - R(\text{mixed shared}) = 66.2\% - 15.9\% = 50.3\%$.
* $W_{Z} = 66.2\% - 65.8\% = 0.3\%$, $W_{E} = 66.2\% - 64.8\% = 1.4\%$.
* $G(\{M,Z\}) = R(\text{all}) - R(\text{only equity live}) = 66.2\% - 13.7\% = 52.5\%$, so
  $W_{MZ} = 52.5\% - 50.3\% - 0.3\% = 1.9\%$.
* The terms add back: $R(\text{only mixed live}) = W_M + W_{MZ} + W_{ME} + W_{MZE} = 50.3\% + 1.9\% +
  10.9\% + 2.0\% = 65.1\%$. This checks the arithmetic only, since that condition is one of the eight
  the terms come from.

No order of removal is assumed, which is the point: "remove mixed first" and "remove mixed last"
give 50.3% and 65.1%, and every term involving mixed is positive here, so those are the extremes. A true term cannot be negative;
a sample term can, by sampling error, and noise x equity comes to -0.6% here.

*One number per source.* Readers want a single share, so each source also gets its $W$ plus equal
parts of the interactions it is in: mixed is $50.3\% + (1.9\% + 10.9\%)/2 + 2.0\%/3 = 57.3\%$ of a
hand's variance, 86.7% of the shortfall. That is the Shapley value of the game $R$ (Shapley 1953;
Owen 2014 for Sobol' indices). Equal parts is a convention, and the seven terms are the evidence.

The three "only live" conditions overlap and do not add: 65.1%, 3.6% and 13.7% sum to 82.4% against
a shortfall of 66.2%, because an interaction counts in every source it involves.

**4. The pairing is a choice, and it moves the split.**

Sharing needs a rule for which decisions at two tables use one draw once the tables have parted.
Two natural rules:

* `decision`: the seat's k-th decision on the street (for equity, the street). Tables keep the same
  draws after they part.
* `history`: the whole public history so far, amounts included. Tables use one draw only while
  nothing public has differed, so any difference in a bet's size makes all later draws of a shared
  source fresh, at every seat.

Each table's own distribution is the default's under both, so both are legitimate. The ends are the
same: nothing shared is the baseline and everything shared cancels exactly. Between them the terms
differ, and by a lot:

| Term | Decision key: share of a hand's variance | History key |
|---|---|---|
| Mixed actions alone | 50.3% [48.2%, 52.3%] | 10.3% [8.7%, 12.1%] |
| Bet-size noise alone | 0.3% [-0.5%, 1.2%] | -0.1% [-0.9%, 0.8%] |
| Monte Carlo equity alone | 1.4% [0.4%, 2.6%] | 0.4% [-0.5%, 1.3%] |
| Mixed x noise | 1.9% [0.5%, 3.2%] | 40.2% [37.6%, 42.7%] |
| Mixed x equity | 10.9% [9.5%, 12.4%] | -0.5% [-2.0%, 1.0%] |
| Noise x equity | -0.6% [-2.0%, 0.5%] | 0.1% [-1.3%, 1.5%] |
| All three together | 2.0% [0.4%, 3.8%] | 15.7% [13.4%, 18.2%] |
| **Sum: the shortfall** | **66.2%** | **66.2%** |

| Share of the shortfall, interactions split equally | Decision key | History key |
|---|---|---|
| Mixed actions | 86.7% [85.0%, 88.1%] | 53.5% [52.4%, 54.5%] |
| Bet-size noise | 2.4% [1.8%, 3.0%] | 38.2% [37.1%, 39.3%] |
| Monte Carlo equity | 10.9% [9.4%, 12.6%] | 8.3% [7.3%, 9.3%] |

The mechanism is in section 2. Under the history key a tiny size difference at one table ends the
coupling of every later mixed action, so sharing mixed actions alone achieves little (ratio 1.78×)
unless noise is shared too, and the two sources look entwined (mixed x noise 40.2%). Under the
decision key the later draws stay coupled, so a size difference does little (noise 2.4%). The
question "how much is noise?" has no answer free of the pairing, because a draw has no identity
across tables once they have parted. Other rules would give other splits, and these two are not shown
to bound them. How the policy lays out its branches on $[0, 1]$ can matter too, as section 2's
made-up example shows; no condition here varies the layout, so that was not tested on the bots.

What can be read without this qualification are the levels that come out alike under both keys:

| Condition | Decision key | History key |
|---|---|---|
| Only mixed actions live, share of the default's shortfall | 98.4% | 99.3% |
| Only equity live, share of a hand's variance | 13.7% | 15.7% |
| Noise shared alone, median ratio | 1.54× | 1.52× |
| Equity shared alone, median ratio | 1.53× | 1.54× |

So with the other two sources perfectly coupled, mixed actions alone still produce about the whole
shortfall, and sharing noise or equity alone gains nothing visible. Only noise live is where the keys
part: 3.6% against 55.9% of a hand's variance.

**5. Each condition over its own variance.**

All shared conditions estimate one $\operatorname{Var}(Y)$, so a review of the plan proposed one
pooled denominator per lineup, which looks more efficient. Development rejected it. A rare big pot
enters one condition's numerator and denominator together and mostly cancels there; with a pooled
denominator it stays in the numerator alone. On development seed 99 pooling took nits' cancelled share
from 26.8% to 0.8% and only-mixed over the baseline from 0.97 to 0.89. Here, on that seed, the
"obvious" variance reduction made things worse. Each condition therefore uses its own $\sigma^2$, as the
committed ratio does.

**6. Switching off outright, reported beside the split.**

The plain experiment is still run, because it shows what the plain reading would have said and why it
misleads: `off_mixed` (every mixed action decided by $u = 0.5$), `off_noise` (size factor 1) and
`off_all` (both, with equity shared, as a check). There is no `off_equity`: exact equity needs about a
million evaluations per flop decision, and a frozen estimate is what `shared_equity` already is.

| Archetype | Median SD, default | Mixed off | VPIP, default | Mixed off | Median ratio, default | Mixed off |
|---|---|---|---|---|---|---|
| Nit | 3.46 | 6.38 | 0.112 | 0.107 | 1.34× | 4.60× |
| TAG | 13.62 | 20.00 | 0.229 | 0.217 | 1.65× | 5.30× |
| LAG | 20.56 | 28.53 | 0.310 | 0.295 | 1.53× | 7.35× |
| Calling station | 24.11 | 1.27 | 0.436 | 0.410 | 1.57× | no variance |
| Maniac | 49.76 | 73.91 | 0.496 | 0.518 | 1.51× | 14× |

Median ratio 7.81× looks like a big win, but these are other bots: the c-bet rate moves +0.219, fold
to a bet -0.113, and the calling station's SD falls about 20-fold. The station's median ratio has no
variance left. The two stat changes are means over the lineups with opportunities (40 for c-bet,
41 for fold to a bet): under `off_mixed` nearly every station lineup has none. Noise off, by
contrast, leaves the bots about where they were: no stat moves more than 0.0019 and the ratio stays 1.53×.

**Results.** Seed 2026, commit `2b02bed`: 50 lineups, 2,000 deals, 17 conditions, 10,200,000 hands.

| Condition | Median ratio [95% CI] | Remaining share (mean) | Median SD (bb) |
|---|---|---|---|
| Default: all three live | 1.53× [1.49, 1.57] | 66.2% | 19.30 |
| Mixed actions shared | 6.26× [5.89, 7.40] | 15.9% | 18.89 |
| Bet-size noise shared | 1.54× [1.49, 1.56] | 65.8% | 19.40 |
| Equity shared | 1.53× [1.50, 1.58] | 64.8% | 19.40 |
| Only mixed actions live | 1.54× [1.52, 1.59] | 65.1% | 19.53 |
| Only bet-size noise live | 31× [27, 36] | 3.6% | 18.77 |
| Only equity live | 7.72× [6.76, 9.11] | 13.7% | 18.92 |
| All three shared (check) | no variance | 0.0% | 18.66 |
| Mixed actions shared, history key | 1.78× [1.71, 1.86] | 55.9% | 18.91 |
| Only bet-size noise live, history key | 1.77× [1.72, 1.84] | 55.9% | 18.80 |
| Only equity live, history key | 6.61× [6.17, 7.29] | 15.7% | 19.43 |
| Mixed actions off (changes the bots) | 7.81× [6.64, 9.56] | 12.3% | 20.25 |
| Bet-size noise off | 1.53× [1.50, 1.57] | 65.9% | 19.09 |

**Predictions: all 33 hold.** The thresholds came from seeds 98 and 99, and the shares moved by a few
points between seeds (noise's small decision-key share most, in relative terms), so this shows the
numbers repeat on a third seed. It is not a strong test.

| Reading | Seed 98 | Seed 99 | Seed 2026 |
|---|---|---|---|
| Cancelled | 36.9% | 33.7% | 33.8% |
| Decision key: mixed, equity, noise | 87.4, 10.7, 1.9 | 83.4, 12.2, 4.4 | 86.7, 10.9, 2.4 |
| History key: mixed, noise, equity | 53.3, 38.6, 8.1 | 51.4, 40.5, 8.1 | 53.5, 38.2, 8.3 |

* A1: the default repeats Step 41's `clones_6`: 300 players, largest ratio difference 0.
* A2: with all three shared or off, every per-deal sum is exactly 0 chips.
* A3: direct within-deal variance against $n s_u^2$: at most 0.8%.
* B1 to B4: sharing a source moves the median SD at most 1.0% and any stat at most 0.0054; by
  archetype, SD factors 0.85 to 1.08 and stats at most 0.028.
* C1 to C3: the levels table of section 4.
* D1 to D6: mixed shared alone 6.26×; split 86.7%, 10.9%, 2.4%; mixed x equity 10.9% above equity
  alone 1.4%; maniacs lowest on equity at 4.9% (others 8.3% to 16.5%).
* H1 to H7: only noise live 3.6% and 55.9%; history mixed shared 1.78×; split 53.5%, 38.2%, 8.3%;
  mixed x noise 40.2%; mixed above noise under both.
* E1 to E3: the table of section 6.

The committed `randomness_division` figure draws sample terms below zero as zero, so a bar can pass
100% slightly (LAG here).

*By archetype* (10 lineups each, so CIs a little narrow, Step 41 section 4):

| Archetype | Cancelled | Mixed (decision key) | Noise | Equity | Mixed, noise, equity (history key) |
|---|---|---|---|---|---|
| Nit | 25.5% [21.4%, 29.6%] | 81.8% | 1.7% | 16.5% | 54.3%, 36.5%, 9.2% |
| TAG | 37.8% [34.8%, 40.5%] | 84.2% | 2.9% | 12.9% | 50.8%, 38.5%, 10.7% |
| LAG | 35.0% [33.2%, 37.1%] | 86.7% | 2.2% | 11.1% | 50.7%, 40.6%, 8.7% |
| Calling station | 36.8% [31.5%, 43.0%] | 88.1% | 3.5% | 8.3% | 58.3%, 33.2%, 8.5% |
| Maniac | 34.1% [32.2%, 36.1%] | 93.1% | 2.0% | 4.9% | 53.2%, 42.4%, 4.3% |

**How to read these numbers.**

* *Cards explain about a third of a hand's result here.* Between copies of these bots, the deal fixes
  33.8% of a hand's variance and the draws move the rest. That is a statement about these bots, whose
  policy acts on equity buckets (`equity_bucket`). Other strategies would give another share, not
  measured here.
* *Equity works mostly through mixed actions, under the decision key.* Mixed x equity is 10.9%,
  equity alone 1.4%. That fits section 2's picture, in which a different estimate moves a probability
  and so a threshold, and the table then acts differently only if the draw falls between the two
  thresholds. That mechanism was not tested separately.
* *Why nits cancel least and why maniacs have the lowest equity share were not examined.* D6 was
  predicted from development, not explained.
* *What this does not divide.* The identity is for copies. Different bots differ for other reasons
  than draws, so nothing here says how much of the 1.30× shortfall of a random mix is randomness.
* *What this is not.* A hand's draws are the bots', not people's. Sharing draws is possible only in
  simulation, and a live table offers neither repeated cards nor shared randomness.

**Run times, awake only** (C.3).

* The real run took 4,311 s, about 72 minutes, on 12 workers, plugged in: started 14:47:30 on
  2026-11-05 by the shell's clock. That is about 2,370 hands/s (Step 41: about 2,600, another day, and
  a single run has nothing to compare with). The conditions took 243-271 s each.
* The System log shows no Kernel-Power 506 or 507 event during the run, and the memory sampler wrote
  70 samples about a minute apart, so the machine stayed awake. A keep-awake request for the session
  was declined by the desktop app's permission check, so nothing held the machine awake.
* Free physical memory was 1.81 to 4.02 GB, mean 2.33 GB. Paging averaged 117 pages/s, with a peak of
  3,003. This can change only the run time.
* The development runs took 1,067 and 990 s on 8 workers, with 1.9 GB free at the start. They do less
  work than the real run (10 lineups of 1,500 deals, against 50 of 2,000).
* Test suite: 543 tests took 132.6 s, plugged in with about 2.7 GB free.

**To reproduce.**

* `python -m experiments.phase2_randomness --seed 2026` or
  `python -m experiments.run_phase2 --seed 2026 --only randomness`; smoke test with `--quick`.
* Results: `experiments/results/phase2/randomness/` (`randomness.json` with every lineup's record,
  `randomness_summary.md`, two figures). Explorer: section `duplicate-randomness` in Phase 2.
* Suite: `python -m pytest` (543 tests).

**What was left.**

* Other pairing rules, and whether any pair of them bounds the split.
* Whether sharing draws helps a real evaluation: seat-keyed draws in a mix of different bots might
  raise the 1.30× of a random mix. This step does not test that, and exact cancellation is not
  expected there.
* From Step 41, still open: depth against seats, the seating order, control variates on top of the
  rotation, and a whole `run_phase2` (which now includes this step).

**What it means.** The copies' shortfall is the bots' draws, all of it: Step 41's argument says so, and
it vanishes to the chip when the draws are shared, as that argument requires. Mixed actions are the largest source under both rules tried, and on their
own they keep about the whole shortfall. Monte Carlo equity alone leaves about a seventh of a hand's
variance. But the tidy "87% mixed actions" holds only under one pairing rule: the other gives 53%,
with noise at 38%. Noise matters to the extent that a size difference unhooks later draws, and that
depends on how one defines two tables' draws as "the same". The lesson is a general one about
attribution: when a system's parts interact through a shared path, "how much is each part?" may not
have an answer until the coupling is specified, and the terms and the invariant levels are worth more
than a single percentage. It also shows why a source cannot simply be switched off to measure it:
that gives 7.81×, for other bots.

**Mistakes.**

* The plan first had two predictions that could not fail once A1 held, since A1 ties the cancelled
  shares to committed numbers (C1, D1 of that draft). The reviewer caught it; they were dropped and
  those numbers are stated as known.
* The plan's first account of what $u = 0.5$ does to the bots was wrong. The reviewer caught it, and
  the text was corrected before the commit.
* The first version of the plan gave one split, without saying it depends on the pairing. The reviewer
  caught it, and that is why the history key exists: 6 more conditions, and section 4.
* The stats checks were first too coarse to see an archetype's change; B3 and B4 were added, and the
  plan now says the claim rests on the construction. An average of a stat over lineups with no
  opportunities was unclear; `behaviour_change_lineups` now counts them. The reviewer caught both.
* A pooled denominator, proposed by the review, was tried and rejected on development evidence
  (section 5).
* A tiny-run test compared results with `==` and failed: `NaN != NaN`, and a stat with no
  opportunities is NaN. It compares `repr`, and means skip NaN.
* The first commit message for the results said 34 predictions; a script counted 33, and the unpushed
  message was corrected. That results commit also went in after ruff only; the suite ran before the
  next commit.
* The explorer's first subtitle said the calling station "barely plays a pot any more" under
  `off_mixed`. Its VPIP is still 0.410; what falls is the spread of its results. Caught reading the
  built page, before it was committed.

## Step 43: Exploitation when the model is wrong (P2.12)

**The question.** Steps 21 and 22 built the Leduc exploitation results: fit five parameters to a bot
from the hands it shows, then best-respond, play a restricted Nash response, or a data-biased
response. Every bot there came from the same five-parameter family the model fits. That is the
kindest case for a model, and the conclusions might be artefacts of it. This step builds opponents
the family cannot express, and opponents that model the hero and change, then measures what each
hero keeps. The code is `poker_lab/games/leduc_misspec.py` (opponents, the projection, adaptive
bots, sessions) and `experiments/phase2_misspec.py` (the experiment). Nothing under `poker_lab/bots`,
`poker_lab/engine` or the P2.7 code is edited, and a control repeats P2.7's numbers exactly.

**1. How wrong is a model? Projecting onto the family.**

A model family $\{p_\theta\}$ that does not contain the truth $q$ still has a best member. Maximum
likelihood does not converge to "the truth"; it converges to the *pseudo-true* parameter

$$\theta^* = \arg\max_\theta \operatorname{E}_q[\log p_\theta(X)] = \arg\min_\theta
\operatorname{KL}(q \,\|\, p_\theta),$$

because $\operatorname{KL}(q\|p_\theta) = \operatorname{E}_q[\log q(X)] - \operatorname{E}_q[\log
p_\theta(X)]$ and the first term does not depend on $\theta$ (White, 1982). So the KL divergence at
$\theta^*$ is the natural measure of misspecification: the expected log-likelihood per observation
that the best family member loses to the truth.

Here an observation $X$ is one hand *as the hero sees it*: the public actions, and the bot's card only
if the hand reaches showdown. A folded hand's likelihood therefore averages over the cards the bot
could hold:

$$p_\theta(x) = \sum_c P(c \mid \text{hero's card, board}) \prod_{t} \pi_\theta(a_t \mid c, h_t).$$

Leduc is small, so the expectation over $q$ is exact. `simulate_views` is not used for it: the
projection enumerates every deal and every action sequence, weights each by its probability under
the true bot and the equilibrium hero, and groups them into the hero's observation classes.
`FamilyProjection.closest` then maximises the expected log-likelihood over the five rates from two
starting points, and `measure` reports $\theta^*$, the KL in nats per hand, and a distance in
actions (section 2).

*Hands to evidence.* With $n$ independent hands the log-likelihood ratio of the truth against
$\theta^*$ has mean $n \cdot \operatorname{KL}$. It reaches $\log 20$, 20:1 odds, on average at

$$n_{20} = \frac{\log 20}{\operatorname{KL}}.$$

This is an optimistic bound, for two reasons. It assumes the alternative is known exactly; a real
test has to search over alternatives and pays for it. And it is a mean: the ratio's spread around
$n \cdot \operatorname{KL}$ is large for small $n$. A kind needing 7 hands by this count is clearly
outside the family. One needing 120 may in practice look like a family member for many hundreds.

**2. A distance in actions: reach-weighted total variation.**

KL counts surprise in nats, which is hard to picture. The second measure asks how differently the
bot acts. At an information set $I$ the total variation between two action distributions is
$\tfrac12 \sum_a |\pi(a\mid I) - \pi^*(a\mid I)|$, the probability mass that would have to move.
Averaged with the weight $w(I)$, the probability that play reaches $I$:

$$\operatorname{TV} = \frac{\sum_I w(I)\, \tfrac12 \sum_a |\pi(a\mid I) - \pi^*(a\mid I)|}{\sum_I w(I)}.$$

An infoset nobody reaches cannot matter, and the weighting says so.

**3. The opponents.**

A family bot gives every information set in a hand-strength bucket the same aggression $a$ and fold
rate $f$. It ignores its seat, the street and how many bets it faces. A rule bot starts from a family
bot's five parameters and gives each information set its own $a$ and $f$:

* `board_blind` buckets by its own rank on both streets, ignoring the board: a pair of jacks plays as
  a weak hand, an unpaired king as a strong one.
* `street` is passive and sticky on street 1 ($a/4$, $f/2$), aggressive and honest on street 2
  ($1-(1-a)^2$, $1-(1-f)^2$). `slight_street` mixes a quarter of it with the family bot at every
  information set: the same error at a lower dose.
* `positional`, with no bet to face, opens tight in seat 0 ($a/2$) and stabs in seat 1 ($1-(1-a)^3$).
* `honest_raiser` bets as usual, raises only with a strong hand, and folds more to a raise
  ($1-(1-f)^3$).
* `noisy` draws each information set's action probabilities from a Dirichlet distribution with
  concentration 5 around the family bot's.
* `near_eq` is three quarters equilibrium and a quarter family bot at every information set: an
  opponent with little to give.
* `equilibrium` is the equilibrium itself. Nothing can be won from it, so it shows only what a wrong
  model costs.

These were the distances before the real run (development seeds, means of two seeds; median bot in
brackets). Seed 2026 reproduced them to within 0.006 nats:

| Kind | KL (nats/hand) | Hands to 20:1 | TV per decision |
|---|---|---|---|
| Family (control) | 0 | never | 0 |
| A quarter of `street` | 0.025 | 118 (125) | 0.057 |
| Ignores the board | 0.075 | 40 (47) | 0.097 |
| Noise at every decision | 0.112 | 27 (27) | 0.124 |
| Raises only when strong | 0.115 | 26 (35) | 0.091 |
| 75% equilibrium | 0.160 | 19 (19) | 0.163 |
| Aggression depends on seat | 0.196 | 15 (15) | 0.150 |
| The equilibrium | 0.295 | 10 (10) | 0.207 |
| Passive, then aggressive | 0.403 | 7 (7) | 0.232 |

No kind is barely distinguishable by this count. The closest, the quarter dose, needs about 120
hands even under the optimistic bound. The sessions in section 6 last 2,000 hands, and the static
fits use up to 3,000.

Each kind starts from the same P2.7 bot (same seeds, same parameters), so the control and the rule
kinds differ only in the rule. The control is the P2.7 bot itself and repeats P2.7's numbers to the
last digit (check A1).

**4. The limit of infinite data.**

With unlimited hands, the fitted parameters converge to $\theta^*$. So best response to
$\theta^*$, called *limit* here, is the best a hero with this model can ever do against that bot.
That splits the shortfall from the clairvoyant oracle in two:

$$\underbrace{G_{\text{oracle}} - G_{\text{fit}}(n)}_{\text{shortfall}} =
\underbrace{G_{\text{oracle}} - G_{\text{limit}}}_{\text{bias of the model}} +
\underbrace{G_{\text{limit}} - G_{\text{fit}}(n)}_{\text{too little data}}.$$

Inside the family the bias term is zero and only data matters. Outside it, the first term stays
whatever $n$ is.

**5. Bots that adapt.**

An adaptive bot plays sessions of 2,000 hands. Every 50 hands it fits the same five-parameter model
to the hero from its last `window` hands, with a prior centred on the equilibrium's closest family
member worth 10 hands, and moves.

* *Inside the family* (`family_slow`, `family_fast`): it computes its best response to that model,
  reads off that response's five family rates, and steps its own rates a fraction `speed` of the way
  there. It never leaves the family.
* *Freely* (`tabular_slow`, `tabular_fast`): it steps towards its full best response, a strategy with
  a separate probability at each of the 936 infosets. Mixing two behaviour strategies infoset by
  infoset is not the same as mixing the strategies, because a probability deep in the tree depends on
  the path. The fix is the *realization plan*: $x(\sigma)$ is the product of a player's own action
  probabilities along the sequence $\sigma$. Plans are linear in the mixture, so
  $x' = (1 - s)\,x + s\,x_{\text{BR}}$ is exactly the strategy "play $x$ with probability $1-s$, else
  $x_{\text{BR}}$", and by Kuhn's theorem the behaviour strategy $\pi'(a \mid I) = x'(\sigma a) /
  x'(\sigma)$ plays the same. Stepping a fraction towards a best response to the opponent's empirical
  play is fictitious play with a fixed step.

**6. Sessions and how they are scored.**

Eight heroes meet each adaptive bot, each in its own session against a fresh copy of the bot:

* *equilibrium*, the reference every gain is measured from;
* *fit once*: equilibrium for 300 hands, then a fit and a best response (or restricted Nash, $p =
  0.9$) kept to the end, which is the static P2.7 hero dropped into a moving game;
* *refit*: a new fit every 100 hands on all hands so far, from the first hand (the population
  model's response before any hand), with best response, restricted Nash at $p = 0.9$, or at $p =
  0.5$;
* *refit on recent hands*: best response to a fit on the last 300 hands only;
* the *clairvoyant*: best response to the bot's actual current strategy. It reads the bot, so it is a
  reference, not a hero anyone could build.

The hands are simulated only to produce what each side learns from: the actions, and a card only at
showdown. A block of 50 hands is scored by the exact seat-averaged value of the two strategies in
force during it (`head_to_head`), so the luck of the cards does not enter the score, and the path
the adaptation took does. A session's value is the mean of its 40 block values.

Each block's random numbers come from a stream keyed by (seed, bot, kind, block), and every hero
uses the same streams. That is common random numbers at the start. Once two heroes play
differently, the same numbers deal different hands, so the numbers are only partly common.

*Gain*, *capture*, *loses*, *exploitability*: as in Step 22, with "loses" a Wilson interval on the
share of 150 bots where the hero's value is below the equilibrium hero's against the same bot.

**Results.** Seed 2026, commit `8601332`, clean tree: 5 seeds, 150 bots per kind, 9 static kinds and
5 session kinds, every value exact. Gains are chips per hand above equilibrium play, means over the
5 seeds.

*Opponents that do not adapt, after 3,000 hands:*

| Kind | Oracle gain | BR gain | BR capture | Limit capture | RNR .9 gain | DBR gain | BR loses | Expl. BR / RNR .9 |
|---|---|---|---|---|---|---|---|---|
| Family (control) | +0.738 | +0.736 | 100% | 100% | +0.705 | +0.483 | 0% | 1.60 / 0.69 |
| A quarter of `street` | +0.740 | +0.715 | 97% | 97% | +0.682 | +0.467 | 0% | 1.49 / 0.66 |
| Ignores the board | +0.926 | +0.750 | 81% | 81% | +0.750 | +0.522 | 14% | 1.23 / 0.56 |
| Noise at every decision | +0.881 | +0.731 | 83% | 84% | +0.702 | +0.501 | 0% | 1.64 / 0.71 |
| Raises only when strong | +0.897 | +0.641 | 72% | 72% | +0.589 | +0.371 | 0% | 1.88 / 0.78 |
| 75% equilibrium | +0.239 | +0.156 | 65% | 66% | +0.159 | +0.100 | 0% | 1.47 / 0.47 |
| Aggression depends on seat | +0.840 | +0.674 | 80% | 80% | +0.654 | +0.449 | 0% | 1.42 / 0.65 |
| Passive, then aggressive | +0.825 | +0.526 | 64% | 64% | +0.523 | +0.417 | 0% | 1.38 / 0.74 |
| The equilibrium | 0 | −0.026 | n/a | n/a | −0.033 | −0.031 | 100% | 1.51 / 0.47 |

Capture is mean gain over mean oracle gain. *Limit* is best response to the closest family bot.
*Loses* is the share of bots on which the hero does worse than equilibrium play. Best response's gain
after 10, 100 and 3,000 hands: against the board-blind bot +0.664, +0.719, +0.750; against the street
bot +0.431, +0.515, +0.526; against the near-equilibrium bot +0.094, +0.120, +0.156. Best response to
the population model, with no hands, gains +0.734, +0.437 and +0.160 against those three. The rank
correlation of KL with best response's shortfall, over all off-family bots, is 0.37.

*Sessions of 2,000 hands*, gain over the equilibrium hero (whole session; share of bots below
equilibrium play in brackets):

| Hero | Fixed bot | Family, slow | Family, fast | Free, slow | Free, fast |
|---|---|---|---|---|---|
| Fit once, BR | +0.611 (0%) | +0.308 (17%) | +0.504 (1%) | −0.184 (98%) | −0.277 (99%) |
| Fit once, RNR .9 | +0.588 (0%) | +0.285 (9%) | +0.448 (0%) | −0.194 (98%) | −0.257 (100%) |
| Refit, BR | +0.721 (0%) | +0.583 (0%) | +0.625 (0%) | −0.011 (47%) | −0.131 (97%) |
| Refit, RNR .9 | +0.691 (0%) | +0.519 (0%) | +0.523 (0%) | −0.027 (63%) | −0.122 (100%) |
| Refit, RNR .5 | +0.486 (0%) | +0.285 (0%) | +0.253 (0%) | +0.078 (7%) | +0.002 (39%) |
| Refit on last 300, BR | +0.714 (0%) | +0.608 (0%) | +0.624 (0%) | +0.079 (21%) | −0.105 (97%) |
| Clairvoyant | +0.738 | +0.633 | +0.656 | +0.781 | +0.983 |
| *Equilibrium's own value* | +0.277 | +0.220 | +0.214 | +0.111 | +0.080 |

Against the free adapters the second half is worse than the whole: refit best response −0.170 and
−0.151, restricted Nash $p = 0.9$ −0.165 and −0.118, restricted Nash $p = 0.5$ −0.006 and −0.009.
The bots moved 0.26 to 0.36 in total variation per decision. Against the equilibrium hero, the free
adapters ended 0.71 and 0.76 nats per hand from the family.

**Predictions: all 5 checks and 22 of 23 predictions hold; D5 fails.**

| Prediction | Threshold | Seed 2026 |
|---|---|---|
| A1, A2: the control repeats P2.7 | max difference ≤ 1e-9, on 150 bots | 0, on 150 |
| A3: control KL | ≤ 1e-6 | 7.5e-14 |
| A4: equilibrium hero never negative | ≥ −1e-6 | lowest 0 (the equilibrium kind) |
| A5: control capture | ≥ 97% | 99.7% |
| B1: KL-implied hands to 20:1 | ≤ 150 | 8 to 121 |
| B2: quarter dose KL, upper | ≤ 0.04 | 0.025 |
| C1: BR capture, off-family | 60% to 90% | 64% to 83% |
| C2: quarter dose capture | 92% to 99% | 96.6% |
| C3: BR gain lower bound, off-family | ≥ 0.1 | 0.154 to 0.708 |
| C4: BR loses, off-family | ≤ 20% | 0% to 14% |
| C5a, C5b: equilibrium opponent | BR upper ≤ −0.01; loses ≥ 95% | −0.024; 100% |
| C6: Spearman, KL against shortfall | 0.2 to 0.55 | 0.37 |
| C7: RNR .9's share of BR's gain | ≥ 87% | 92% to 102% |
| C8: RNR .9's exploitability over BR's | ≤ 62% | 31% to 54% |
| C9: DBR minus BR, upper | ≤ −0.03 | −0.055 to −0.249 |
| C10: population minus 10-hand BR, lower | ≥ 0 | 0.042, 0.037 |
| C11: hand-reading minus card-blind, lower | ≥ 0 | 0.033 to 0.578 |
| D1: refit BR's share of the clairvoyant, family adapters | ≥ 85% | 92%, 95% |
| D2: refit minus fit once, lower | ≥ 0 | 0.109 to 0.241 |
| D3: free adapters, second half, upper | ≤ 0 | −0.107 to −0.152 |
| D4: fast free adapter, share below equilibrium | ≥ 90% | 97% to 100% |
| **D5: RNR .9 minus BR, free adapters, upper** | **≤ 0.02** | **−0.008, +0.024: fails** |
| D6: RNR .5 against the fast free adapter, interval | within ±0.02 | [−0.002, +0.007] |
| D7: clairvoyant, free adapters | ≥ 0.7 | 0.78, 0.98 |
| D8: equilibrium's value ratio, upper | ≤ 0.5 | upper ends below 0.5 (means 0.40, 0.29) |
| D9: recent minus refit, slow free adapter, lower | ≥ 0 | 0.080 |

*D5.* The claim was that restricted Nash at $p = 0.9$ does not protect against a free adapter: it
should do no better than best response by more than 0.02. Against the fast free adapter its gain
over best response is +0.009 [−0.005, +0.024]. The upper end passes the bound. In development it was
+0.013 at most, so the margin was 0.007, too thin for a third seed. The substance rests on other
numbers: restricted Nash $p = 0.9$ still does worse than equilibrium play on 100% of those bots,
ending at −0.122. But the prediction as written failed, and a gain of up to 0.024 over best response
is not excluded.

The checks were known in advance: P2.7's committed numbers fix A1, A2 and A5, and A3 and A4 follow
from the construction and a theorem. The other 23 thresholds came from seeds 98 and 99, and most
numbers moved by about 0.01 between seeds. So passing them shows that the development picture repeats
on a third seed, not that it was hard to predict.

**How to read these numbers.**

* *A wrong model's cost is bias, not data.* Against every off-family kind, best response after 3,000
  hands keeps the same share as best response to the closest family bot, to within about 0.7 points.
  More hands would not help; a richer model might.
* *A wrong model still wins here.* Apart from the near-equilibrium bot, these bots are exploitable by
  1.03 to 1.32 chips per hand. So even a model that misreads them finds nearly two thirds or more of the gain,
  and beats equilibrium play on at least 86% of bots. Against the near-equilibrium bot, with little to
  give (0.31), the share drops to 65%. There, as against the board-blind bot, a 10-hand fit gains less
  than best response to the population prior (C10); against the positional, honest-raiser and
  noisy bots it gains more.
* *KL is a weak guide to the cost.* The street rule is the easiest to detect (8 hands on seed 2026) and the most
  costly (64% kept), but the near-equilibrium bot costs almost as much at 40% of its KL, and the
  board-blind bot needs 40 hands to detect and keeps 81%, about as much as the positional bot that
  needs 15. The rank correlation is 0.37.
* *Phase 2's orderings hold off the family.* Restricted Nash $p = 0.9$ keeps 92% to 102% of best
  response's gain at a third to a half of its exploitability. The data-biased response gains less.
  Reading the cards shown at showdown beats ignoring them.
* *Adapting is what breaks it.* A bot that adapts inside the family is a moving target the model can
  still describe, and a hero that refits keeps 92% to 95% of what is available. A bot that adapts
  freely puts every best-response hero behind equilibrium play, on average, in the second half of
  a session, at both speeds, and restricted Nash at $p = 0.9$ is not enough protection. At $p = 0.5$ the hero
  breaks about even: what the restriction buys here is not losing, not winning.
* *What this is not.* Leduc only, four rule kinds plus three others, two adaptation schemes, one
  hero model. Nothing here is about people.

**Run times, awake only** (C.3).

* The real run took 2,883 s, about 48 minutes, on 10 workers (`--workers 10`; the
  config's default, recorded in the results, is 12): 648 s for the static part and 2,234 s
  for the sessions, started 20:19:19 on 2026-11-09 by the shell's clock. The memory sampler wrote 96
  samples, never more than 31 s apart, so the machine did not pause. The System log shows a
  Kernel-Power 506 at 20:51:40 and no 507 before the run ended, while sampling went on.
* It started on battery at 43% with 1.2 GB free, and was plugged in at 20:35 (Kernel-Power 105).
  Free memory fell to 0.02 GB at 20:22 and averaged 1.39 GB. This can change only the run time.
* The development runs took 3,033 and 3,000 s on 10 workers.
* Test suite: 604 tests took 229 s before the edits that followed the review. The run after them took
  5,001 s, including a sleep from 19:08:53 to 20:00:14 (Kernel-Power 506 and 507). That leaves about
  1,920 s awake, and the slowdown was not explained.

**To reproduce.**

* `python -m experiments.phase2_misspec --seed 2026` or
  `python -m experiments.run_phase2 --seed 2026 --only misspec`; smoke test with `--quick`.
* Results: `experiments/results/phase2/misspec/` (`misspec.json`, `misspec_summary.md`, two figures).
  Explorer: section `misspecification` in Phase 2.
* Suite: `python -m pytest` (604 tests).

**What was left.**

* A hero whose model can grow: a tabular or per-street model, used once the five-parameter fit is
  rejected (Step 30's fit check, applied here).
* A stop-loss or change detector for a hero facing a free adapter. Restricted Nash $p = 0.5$ stops the
  loss, but by giving up the gain from the first hand.
* Other adaptation rules: a bot that best-responds outright, or one that adapts to other features
  of the hero.

**What it means.** The Phase 2 exploitation results hold up better than feared against fixed opponents
the model cannot describe. Best response loses a share of the gain, between about a sixth and a third here (3% for
the quarter dose),
and that share does not shrink with more data. But it still wins clearly, and its ordering against
restricted Nash and the data-biased response stays the same. The real weakness is opponents who move.
When the opponent adapts in ways the model cannot follow, a hero that keeps best-responding to its
model is exploited in turn: in the second half of a session it does worse than equilibrium play
on average, and against the faster adapter it ends the session below equilibrium play on 97% to
100% of bots. The general lesson: a model's error matters most when it is about how the
opponent reacts to you, not about how the opponent plays.

**Mistakes.**

* The first development design was changed after seeing seed 98: the restricted Nash $p = 0.5$ hero
  and the quarter-dose kind were added. Both came before the plan was written, and seed 98 was run
  again on the final code.
* The review of the plan found 16 problems, fixed before the commit. Among them: the
  equilibrium-opponent prediction could not fail (no strategy gains against an equilibrium); the KL
  correlation prediction was one-sided; D5 tested the mean only; two claims went beyond their numbers;
  checks were counted as predictions; and restricted Nash, the data-biased response, the population
  prior and exploitability had no prediction at all. C7 to C10 and D1 were added then, with thresholds
  from the development means only, because those runs predate the fields.
* D5's bound was set with a margin of 0.007 on two seeds, and the third seed went past it.
* A script checking that every prediction path resolves had no `__main__` guard and broke the worker
  pool on Windows. Adding the guard fixed it.
* The real run started on battery with 1.2 GB free.
* The explorer's first text said the hand-reading model beats the card-blind one "by at least" the
  smallest *mean* difference, while the smallest lower bound is lower. It was caught reading the text
  before the commit.
* The `misspec_by_hands` figure first had a wide gap between its two panels: its legend hung under the
  left panel, and `tight_layout` counted it as part of that panel. It was redrawn from the committed
  results with a legend under both panels, in a commit of its own; no number changed.

## Step 44: Playing against the model (a local web page)

**The request.** Before any home-game capture, the owner wanted to play against the model themselves,
so it can learn from their hands and look for exploits, and wanted a no-limit hold'em framework that a
future exploit engine can plug into. Their choices: heads-up push/fold first (5-20 bb), because the
whole learn-and-exploit pipeline already exists for it; full no-limit hold'em playable now, with an
empty-or-heuristic exploit slot; a local web page served by Python's standard library; the model's
read of them behind a switch that is off by default, and a report after every session. This is a
tool, not a research result: nothing here is pre-registered, and no number from it is reported.
The code is `poker_lab/play/` (new); nothing outside it changes except the package data list in
`pyproject.toml`, so every committed result and every existing default stays as it was.

Start it with `python -m poker_lab.play` and open `http://127.0.0.1:8790/`.

**1. A person inside a synchronous engine: a condition variable.**

`TableSimulator` asks each seat for a decision with an ordinary function call, `agent.act(view,
rng)`, and cannot go on until it returns. A bot returns at once. A person answers seconds later,
through a different thread: the web server's. One standard way to join the two is a *condition
variable*: a lock plus a waiting room. A thread holding the lock can `wait()`, which releases the
lock and sleeps until another thread calls `notify()`; it then takes the lock back and rechecks
what it was waiting for (waits can wake spuriously, so the check runs in a loop).

`DecisionChannel` (in `channel.py`) has one lock and four fields: the pending `SeatView`, the
answer, a turn token, and a closed flag.

* The engine thread, inside `HumanAgent.act`, stores the view, increments the token, and waits
  until an answer or the closed flag appears.
* A server thread shows `pending()` to the page. When the page posts `{token, action, amount}`,
  `submit` checks that the token is the current one, that no answer is already there, and that the
  decision is legal (`check_decision` against `view.legal`, plus the mode's own rule: in push/fold
  only push or fold, and call or fold facing a push). Only then does it store the answer and
  notify.
* `close()` sets the flag and notifies; the waiting engine thread raises `SessionClosed`, which
  unwinds the hand. The unfinished hand never completes, so it is never saved or learned from.

The token matters. Without it a double click, or a page that polled just before the engine moved
on, could answer the *next* decision with a click meant for this one. With it, a stale click is
refused (HTTP 409) and nothing reaches the engine.

Everything else in a session (results, the hand log, the read) is guarded by the same lock, which
is re-entrant, so the server's `state()` can call `pending()` while holding it.

**2. The firewall, again.**

The model must learn only what it could see at a real table: completed hands, with your cards only
when you showed them. Three layers already enforce that, and the play code uses them rather than
adding its own:

1. The simulator calls `hand_completed` with the hand passed through `redact_hole_cards` for that
   seat, and only after the hand is complete.
2. `PushFoldModel.observe` redacts again (to the model's seat) before keeping the hand, so even a
   caller that passed a full record could not leak your cards; earlier sessions loaded from disk go
   through the same function.
3. `push_fold_observations` redacts a third time and reads your class only where `shown` is true.
   In hold'em, `ProfileStore.ingest` redacts to public information and keeps counters, never hands.

Two tests check the result rather than the code path. One plays 60 hands and walks every object
reachable from the model (history, posterior, recommender, the read): the only card strings in it
are the model's own cards, the boards and the cards you showed. The other is a counterfactual: it
gives two models the same hands, except that in every hand where you did not show, your hole cards
are replaced by others, and checks that the two reads are identical, number for number. If an
unshown card could reach the model by any path, some number would differ.

Your hands themselves are saved in full, as JSON Lines, in a folder outside the repository
(`POKER_LAB_PLAY_DIR`, default `%LOCALAPPDATA%/poker_lab/play`); the code refuses a folder inside
the repository. They are the owner's own data: using them in a research result needs their OK.

**3. Push/fold: the P3.4 pipeline with you as the opponent.**

The model plays the hero's side of P3.4, and you are the opponent it models:

* *Prior.* `PushFoldPrior.fit` on 20 push/fold bots per archetype (P3.2's population, a fixed
  seed). Before any hand, the model expects you to be one of those bots.
* *Posterior and ranking.* Exactly `PushFoldRecommender.recommend` on your decisions in hand
  order: posterior draws, the equilibrium, restricted Nash responses at $p \in \{0.1, 0.25, 0.5,
  0.75, 0.9\}$ and the best response, ranked by the lower end of a 90% interval on the gain over
  the equilibrium against you. The equilibrium's gain is 0, so the model deviates only when some
  response gains with posterior probability of at least 95%. (The play code calls the
  recommender's two internal steps itself, so it can keep the posterior for the read; the numbers
  are the ones `recommend` returns.)
* *Fit checks and fallback.* P3.5's checks, with thresholds calibrated for the session's stack on
  200 bots drawn from the prior and played by the equilibrium, at 10 to 800 hands, then cached in
  the data folder (about 0.5 s per bot on this laptop, so a minute or two with four worker
  processes, once per stack). A failed stability check means the model plays the equilibrium
  (P3.5b's rule).
* *Refit schedule: after every completed hand, on all your hands at this stack.* Two facts set it.
  A refit is cheap (about 0.15 s here), so there is no reason to wait. And P3.9 compared the
  options: the *refit* hero kept 94% of the clairvoyant's gain against static bots and stayed
  within 0.002 bb/hand of equilibrium play against bots that adapt, and refitting on the last 300
  hands instead of all of them changed the gain by about 0.001 bb/hand either way (post hoc there).
  In a short session a window would also discard most of the evidence. Your earlier sessions at
  the same stack count too, unless you start a fresh model.

One caveat goes in every report. The checks were calibrated for a single look at the data; here
they run after every hand, so the chance that some check fails by accident during a session is
well above 5% (in P3.9, 19% of static bots saw at least one fallback over 40 refits). A false fallback costs little (the equilibrium cannot lose against any opponent in
expectation), so the rule is kept as it is and the report says so.

*What the read shows* (the switch, or the report): your estimated push and call ranges as 13x13
grids (the posterior mean probability per hand class, with the equilibrium's range outlined); the
estimated range sizes with 90% intervals; the response chosen, its rank and its gain interval;
P(gain > 0); the fit checks and whether it fell back; and two exact values from the closed form
$v(s,c)$ of P3.4: the model's strategy against your estimated strategy, and the equilibrium's. Your
estimated exploitability (how much a best response would win from you, if the estimate were right)
comes from the same closed form. Because the hero's value is linear in your per-class
probabilities, its value against the posterior mean equals its posterior expected value.

**4. All-in EV: taking the cards' luck out of your results.**

When two players are all in before the river, the remaining cards decide the pot but say nothing
about the decisions. With $P$ the pot once uncalled chips are returned, $c$ your contribution and
$e$ your equity (win probability, ties counted half) given both hands and the board at the moment
of the all-in:

$$\text{all-in EV} = e P - c, \qquad \text{actual} = (\text{what you won}) - c.$$

The expected difference between the two is zero, and summed over a session the luck of the
run-outs drops out. The two players' all-in EVs sum to zero, as their actual results do (a test
checks both).

The equity is exact from the flop on: $\binom{45}{2} = 990$ run-outs from the flop, 44 from the
turn. Preflop there are $\binom{48}{5} = 1{,}712{,}304$ boards, too many to evaluate in Python
each hand, so the code averages over 20,000 random boards. Each board's outcome is a value in
$\{0, \tfrac12, 1\}$ with variance at most $\tfrac14$, so the standard error of the mean is at most
$\sqrt{1/4 \div 20{,}000} \approx 0.0035$: about 0.14 bb in a 40 bb pot. The random numbers are
seeded with the hand id, so a report reproduces exactly. Pots with three or more players, all-ins
on the river and hands without an all-in keep their actual result.

**5. No-limit hold'em: profiling and an exploit slot.**

You sit at a heads-up or six-handed table, 1/2 blinds, 100 bb, stacks topped up each hand. The
opponents are Phase 1's archetype bots (drawn at random; the report reveals which) or advisors: the
pot-odds baseline adjusted by the exploit slot.

*The read* is Phase 1's profile: shrunk stats with 95% intervals and the calibrated logistic
classifier's archetype probabilities. Its training rows come from a small league (120 bots, 600
hands each, windows of 25 to 600 hands), simulated once and cached as plain arrays; the classifier
is refit from them at each session start. Both the priors and the classifier were built on
six-handed bot tables, so the read says which bot your stats resemble, not how a person plays, and
heads-up it is outside its training range. The page says both.

*The exploit slot* (`exploit_slot.py`) is a registry of `ExploitEngine`s, each with a `kind`:
`heuristic` (a rule of thumb whose gain was never measured), `computed` (a response computed
against a model of the opponent, with a stated method) or `none`. The page shows the kind next to
the name. It ships with `FoldEquityHeuristic` (the default, labeled as a heuristic everywhere) and
the identity. The contract a future engine must meet is the existing one: take exactly a
`SeatView`, a baseline `ActionDistribution` and a `ProfileSnapshot` built from completed hands;
return a strategy that is legal under `view.legal`, an exploitability estimate (or `not_computed`
with a reason) and a one-line rationale; and be deterministic given its inputs. `Advisor` checks
the types and the legality. A new engine needs one `register(...)` call and no change to the page.

**6. Local only, by construction.**

The server is `http.server.ThreadingHTTPServer`, and it refuses any address but 127.0.0.1. That
alone does not stop a web page elsewhere from talking to it: a page on another site can make the
browser send requests to `127.0.0.1`. Three checks close that:

* *Host.* Every request must name `127.0.0.1:<port>` or `localhost:<port>`. A *DNS rebinding*
  attack points an attacker's domain at 127.0.0.1, so the browser treats the attacker's page as
  the same site as the server; but the request then carries the attacker's domain as its `Host`,
  and is refused.
* *Content type.* A `POST` must be `application/json`. A cross-site HTML form can send only
  `text/plain`, URL-encoded or multipart bodies; a script on another site that sets
  `application/json` triggers a CORS *preflight*, which this server never approves. So another
  site cannot submit a decision or start a session.
* *Origin and Fetch Metadata.* When the browser sends `Origin` or `Sec-Fetch-Site`, they must say
  same-origin.

The page itself is vanilla HTML, CSS and JS with a Content-Security-Policy of `'self'`: it can load
nothing from anywhere else, and the server makes no outgoing request. Bodies are capped at 4 KB,
every JSON field is checked for exact type (a `true` is not a stack size) and range, and unknown
fields are refused. The tests post each kind of bad input and check the status code.

**7. The page.** Cards, the board, the pot and stacks in big blinds, the legal actions with sizes
(raise presets at half pot, three quarters and pot), the hand log, the session's results in chips
and all-in EV, the switch and the report. It works from the keyboard (F fold, C check or call,
P push, A all-in, R raise then Enter, N next hand, M the read, E twice to end) and at phone width
(checked at 375 px: no element wider than the screen). It polls `GET /api/state` every 0.3 s
while the engine works and every 2 s while it waits for you (a decision or the next hand).

**What was left.**

* *A computed exploit for hold'em.* The slot's `computed` kind is empty. P2.8b (a local best
  response over an action abstraction) is the natural first entry; it would also give the
  advisors an exploitability estimate.
* *A model of a person in hold'em.* The read is a bot classifier. Hands from real players (Phase 4,
  only on the owner's go) are what a human model needs.
* *Tournament stacks.* Push/fold resets the stacks every hand, so the effective stack never moves;
  a sit-and-go with changing stacks would need one recommender per stack.

**Mistakes.**

* The first smoke test failed with a stale-decision error on a legal click. Between the click and
  the engine thread waking up, the old turn still looked pending, so the next poll offered it
  again. `pending()` now hides a turn as soon as it is answered.
* Inline `style` attributes broke under the page's own Content-Security-Policy; the page sets the
  heat-map shading through the DOM instead.
* A first draft confirmed "end session" with the browser's blocking `confirm()` dialog; it became
  a second key press, which works the same from the keyboard and cannot freeze the page.
* Two code reviews found what the first tests missed: turn tokens restarted at 0 in every session
  (a stale tab could answer a new session's first decision), two kinds of odd JSON dropped the
  connection, caches were written in place (a crash mid-write broke every later session), one cut
  line in a hands file stopped the next push/fold session, a failed report write left the page
  stuck, and ending during the first hold'em setup waited on the uninterruptible bot league while
  holding the server's lock. Each was fixed with a test: tokens start at random, caches are
  written to a temporary file and renamed, unreadable lines are skipped and counted, `end()` is
  idempotent, waits at most 5 s outside the lock and always ends the session, and a stalled
  client times out after 10 s.

---

# Part C: cross-cutting engineering

## C.1 Testing strategy

| Kind | Example | Why |
|---|---|---|
| Construction invariants | `HandHistory` rejects hole cards while in progress | bugs fail at the source, not downstream |
| Hand-crafted scenarios | `tests/test_stats.py` scripts hands street by street | exact expected counts for every corner case |
| Differential testing | replay every simulated hand through unmodified PokerKit | a reference implementation is the best oracle |
| Independent solvers | LP equilibrium vs closed-form Kuhn family vs brute-force best response | three methods that share no code must agree |
| Statistical recovery | simulate from a known prior, check recovery and 95% coverage | tests the estimator, not one number |
| Contract tests | every `StrategyProvider` and `ExploitEngine` through the same checks | future implementations inherit the guarantees |
| Adversarial tests | ten attempts to smuggle opponent cards into the Advisor | a firewall is only as good as its attack tests |
| Fault injection | 33,540 deliberately damaged hand-history files (Step 32) | measures what a validator catches, and found two bugs its unit tests missed |
| Estimators against known truth | the held-out prior error, corrected for noise, within 5% of the true error on players with known rates (Step 33) | an evaluation metric is code too, and can be wrong |
| Simulated controls | the real data's seating replayed by bots, run through every check (Step 33) | separates what the players do from what the pipeline and the data's shape do |
| Bit-for-bit re-simulation | five replicates simulated again with a new extractor: every counter and decision row identical, the refit classifier reproduces the published numbers exactly (Step 34) | a new model is compared on exactly the old data |
| Ablations that prove themselves | the GRU separates two classes that differ only in the order of a hand's tokens; the order-free network cannot (Step 34) | an ablation must remove what it claims to |
| The old code as the oracle | the schema v1 code, checked out in a git worktree, writes 20,000 hands; the new code must read them back as exactly the hands it makes itself (Step 35) | a migration is tested against what the old version really wrote, not against what we remember it wrote |
| Shared randomness | two heroes facing copies of one adaptive bot from the same seed get identical deals and differ only in their actions (`tests/test_push_fold_adaptive.py`, 6 tests); each session block's value equals the exact head-to-head value; the experiment's heroes refit on schedule and on their window (`tests/test_adaptive_experiment.py`, 4 tests) (Step 36) | a paired comparison is only as good as its pairing |
| Exact moments against sampled play | every estimator's mean and variance are sums over the game tree; 20,000 duplicate deals per sampled matchup, each hand recorded as labels and replayed with `GameTree.follow`, must agree with them (z-scores, SD ratios), and per-deal moments must match brute-force duplicate pairs in Kuhn (Step 38) | the exact formulas check the sampler, and the sampler checks the formulas |
| Properties for any input | AIVAT's estimate is unbiased for a value function of random numbers and does not depend on the opponent's strategy (property tests in `tests/test_aivat.py`, Step 38) | a guarantee that holds for every value function is tested with one that is useless |
| Closed-form reference cases | the metric on constructed arrays: identical tables give $\rho = 1$ and ratio about $1/n$, perfectly cancelling tables $\rho = -1/(n - 1)$ with no variance left, neighbors sharing a term offsets (0.5, 0, 0), and every ratio equals the identity from $\rho$ (Step 41) | a statistic is checked where algebra gives the answer, before any poker runs through it |
| Report code on made-up results | every pre-registered prediction is evaluated on made-up results of the real shape, and a quantity that is not a number fails its prediction instead of raising (Step 41) | a crash in the analysis after a 24-minute simulation would lose the run |
| A design's exact case | six always-call agents: every per-deal mean is exactly 0 chips, in the tests and at the start of the real run, which stops if not (Step 41) | a variance-reduction design is checked where it must be perfect |
| A switch that must do nothing | `SwitchedBot` with every source live plays draw for draw as its parent, and the real run's baseline repeats the committed ratios with difference 0 (Step 42) | an experiment's wrapper cannot quietly change the bots |
| A decomposition on known terms | `decompose` recovers terms built by hand, and they sum to the whole (Step 42) | the algebra is checked before any poker runs through it |
| Round trips and fixed points | export then import 20,000 simulated hands; import, export, import 10,001 real ones | an encoder and decoder that agree on everything lose nothing |
| Reproducibility | same seed gives identical JSON; 1 vs 2 workers give identical arrays | catches hidden global state |
| Whole-repository reproduction | one command reruns every step from a clean commit, and `compare_runs` lists every difference outside provenance and timing; each difference is traced to a commit by rerunning at the commit the results record (Steps 24, 37) | results committed days apart can drift from the code that is supposed to produce them |
| Bit-for-bit optimization | each faster path checked against the slow one exactly: `choose` against the full report (by value), repeated predictions against predictions from scratch, a cached LP against a fresh HiGHS solve (Step 39); an LP solved from scratch through SciPy's HiGHS bindings against `linprog`, array by array; two replicate processes against one (Step 40); each changed step rerun and compared by `compare_runs` | a speedup that moves a result is a change of method |
| Changes signed off as moving digits | a hot-started sweep against LPs solved from scratch: the first $p$ bit for bit, later ones with the same objective within HiGHS's tolerance; every `compare_runs` difference classified by kind and size, with summary lines, figures and verdicts unchanged (Step 40) | a change allowed to move digits must still be shown to move nothing else |
| History independence | the same LP sweep fresh, after other LPs and sweeps in two orders, and in a spawned process gives the same bytes; two replicate processes with the last replicate's results delivered first write the same files (Step 40) | a result that depends on what a process did before depends on scheduling |
| Counting the expensive calls | spies wrap the LP calls and the encoder: no LP on a fallback or for the best response, an evicted LP solved again, each block of players encoded once (Step 39); a sweep's later LPs hot-started, with HiGHS reporting that it did not presolve them, and a failed hot start solved again from scratch (Step 40) | checks that the work is skipped, not only that the answer is right |
| End-to-end | simulate -> advise -> profile -> evaluate -> round-trip | proves the seams fit |

Statistical tests use fixed seeds and tolerances wide enough to be stable and tight enough to catch
real errors (for example coverage within +/- 1.5 points of 95% over 5,000 simulated players). The
suite has 624 tests (the Deep CFR and sequence-model tests skip without PyTorch). `python -m pytest`
runs them in four processes with pytest-xdist (`-n 4 --dist worksteal`, set in `pyproject.toml`).
Measured times are 53.5-53.9 s for 471 tests and 61.6 s for 479; 517 took 104-127 s, and once 300 s,
in varying machine states (Step 41); 604 took 214 to 229 s (Step 43). `tests/conftest.py` starts the tests marked `long` first, so
the longest test does not start late and finish last (Step 40). `-m "not slow"` skips the slowest
tests, and `-n 0` runs everything in one process, for a single file or for `--pdb`. The explorer's
browser behavior has its own check (`python -m experiments.explorer.browser_check`, Step 25), since
it needs Chrome or Edge.

## C.2 Reproducibility and provenance

* One master seed; every other seed is derived with `numpy.random.SeedSequence`.
* `SeedSequence` pads entropy shorter than its four-word pool with zeros, so `[s]`, `[s, 0]` and
  `[s, 0, 0]` are one seed. Give each stream a nonzero tag word of its own, and count sub-streams
  from 1 where the shorter list is also a stream (Step 41). The words of `generate_state(k)` are the
  start of any longer request, so asking for more words keeps the earlier ones.
* Each table owns a seeded `random.Random` that deals the cards and drives every agent's
  randomness, so a table replays exactly whatever the agents do internally.
* Results record seed, config, git commit, dirty flag (captured at the start of the run),
  package versions and the simulation fingerprint.
* Simulations are cached by config plus fingerprint (`experiments/cache/`, git-ignored),
  so analyses can be re-run in minutes and stale data cannot be reused.
* Real data are analysed once. An experiment on real hands is developed on simulated
  stand-ins with the same code path (P3.7: `real_source: standin`) and on development seeds,
  so the pre-registered run is the first time the analysis sees the real data.
* External data are pinned: the PHH dataset is fetched at a fixed git commit into a cache
  outside the project, and results record that commit and a SHA-256 over every file read.
* A reproduction run writes to an untracked folder (`--out`) from a git worktree, so every
  step records a clean tree. Figures carry no provenance of their own: two were redrawn after
  their run by newer code than the commit their JSON records (Step 37).
* Caches inside a process must never change a result. The LP memo keys on its inputs' exact
  bytes; the embedding memo keys on objects it holds, under preconditions its docstring states.
  Both die with their process; a cache on disk would also need the code and solver versions in
  its key, as the simulation cache has its fingerprint (Step 39).
* Parallel work must not change a result. Worker processes get their seeds from the task
  (Step 8), each fit its configured thread count (Steps 34 and 40), and nothing from the process's
  history: a hot-started LP sweep makes its own HiGHS instance, so its result depends only on its
  inputs, whichever worker runs it. Results are combined in task order, never completion order,
  since floating-point sums depend on their order (Steps 39 and 40). The number of jobs is
  recorded (`meta.replicate_jobs`) outside `meta.config`: it is how a run was executed, not what
  it computed.
* The committed results stay the original runs. Since Step 40's hot start, a rerun of adaptive,
  recommend, fitcheck and confirm matches them in every summary line, figure and verdict, but not
  bit for bit (99% of the changed floats within $10^{-12}$ relative); the other steps still
  reproduce exactly. Replacing committed results is the owner's decision.

## C.3 Platform lessons (Windows, PowerShell, OneDrive)

* *Spawned workers re-import modules:* keep package imports lazy (PEP 562) and heavy imports
  inside functions; cap worker counts.
* *PowerShell quoting:* `"...$SD..."` expands `$SD` as a variable (a commit message lost its
  `"W$SD"`); use single-quoted here-strings (`@' ... '@`) for commit messages.
* *Encodings:* Windows PowerShell 5.1's `Set-Content -Encoding utf8` writes a byte-order mark;
  rewrite files with Python when the exact bytes matter.
* *Line endings:* `Path.write_text` translates `\n` to `\r\n` on Windows; scripts that edit
  source files use `write_bytes` (or `newline="\n"`).
* *Exit codes:* git writes some output to stderr, which PowerShell can report as a failure even
  when the commit succeeded; check `git log` rather than the exit code.
* *Synced folders:* a virtual environment inside OneDrive syncs thousands of files; prefer one
  outside it, or record package versions as done here.
* *Scripted edits of documents:* a find-and-replace pattern that occurs exactly once can still
  be in the wrong place. A glossary insertion keyed on a short pattern landed inside the
  mistakes table and went unnoticed for several commits. Anchor replacements on enough
  surrounding context to be unambiguous *by meaning*, and re-read the result.
* *Quoting inside generated code:* a backslash-n written inside a shell heredoc that feeds a
  Python script can arrive as a real line break, so an exact-match replacement silently stops
  matching (this happened twice, and once to this very sentence). Edit such files with an
  editor tool, or use raw strings.
* *Sleep inside long runs:* a laptop that sleeps stretches a run's wall time (one network's
  fit took 8,706 s instead of 23 s in Step 34, one step took 2 hours in Step 33, Step 36's
  run recorded 5.3 hours with the laptop closed for most of it, and the Phase 3 wrap-up spent
  6.5 of its 10.5 hours with the lid closed). Results are unaffected, but read the per-step
  times before reporting how long anything takes, and know whether the machine slept. On a
  Modern Standby laptop the System log shows sleep as Kernel-Power events 506 and 507 (entering
  and leaving standby, with a reason such as Lid or Idle Timeout); events 42 and 107 appear
  only for S3 sleep and hibernation. A closed lid paused the wrap-up run almost entirely; a
  screen turned off by the idle timeout did not slow it (Step 37).
* *Speed varies between days:* unchanged training code took 27-39% less time on 2026-10-30 than
  in the committed Step 34 run, for reasons not found (Step 39). Measure a speedup with old and new
  code in one session, and check a step-level comparison against Amdahl's bound. A process
  suspended on purpose (`NtSuspendProcess`) is subtracted like a closed lid.
* *Spawned workers need a guarded main module and start slowly:* on Windows every worker process
  starts as a new interpreter that imports the main module (Steps 8 and 24). So a script that starts
  a pool keeps its top-level work under `if __name__ == "__main__":`, the function a worker runs is
  defined at module level, and each worker sets per-process state (torch's thread count) itself.
  Start-up takes time in every worker: in the quick configuration two replicate processes were
  slower than one, probably for that reason (Step 40).
* *Logical processors are not cores:* this laptop's 16 logical processors are 8 cores with two
  hardware threads each (an AMD Ryzen 9 8945HS with SMT). Two trainings of 8 threads each ran
  1.65-1.71× slower each (Step 40). Before running jobs side by side, count jobs × threads against
  the logical processors, and the memory (15 GB, shared with the owner's applications; two full-size
  replicates took free memory down to 1.42 GB).
* *Idle-timeout standby while a keep-awake request is held:* during Step 40's two-hour verification
  every Modern Standby interval had the reason Idle Timeout, and a memory sampler that logged every
  10-11 s never skipped. So that time counts as awake, as in Steps 37 and 39. A sampler writing on a
  fixed interval is a simple check that a machine kept running.
* *A full machine slows runs, not results:* with the owner's applications open, free physical memory
  stayed between 0.11 and 1.59 GB through Step 41's real run and the commit charge reached 53.9 GB.
  A test that took 44 s under the suite's load (Step 40) took 119 s on such an evening, and 86-87 s
  the next day on battery with memory free, so a time says little without the machine's state. Log
  free memory with a sampler next to any time quoted, and say whether the laptop was plugged in.

## C.4 Figures

Every figure is rendered in a light and a dark variant (the README picks one with
`<picture>`). Colors come from a palette checked by a validator for colorblind separation
between adjacent series. Lines are 2 px with ringed markers and gridlines are hairlines.
Interval bands are clipped to the metric's natural range for display only. Every chart has a
numeric table next to it, so no number depends on reading a color.

Axes shared between panels are one object: with `sharey=True`, calling `invert_yaxis()` on each
panel inverts the shared axis once per panel, so two panels undo each other (Steps 32 and 38).
Invert once, and look at every figure before committing it.

---

# Part D: mistakes and how they were caught

| Mistake | Caught by | Fix | Step |
|---|---|---|---|
| All-in flag computed after PokerKit paid out the pot | schema validator on hand 285 of a smoke run | use the stack before the action | [4](#step-4-the-simulation-engine-m2) |
| Heads-up blinds recorded BB first | heads-up order test | sort forced posts | [4](#step-4-the-simulation-engine-m2) |
| Strings inside lists not parsed by `Card.clean` | evaluator equivalence test | normalize input in the fast hand type | [4](#step-4-the-simulation-engine-m2) |
| Bluff raise when raising was illegal turned into a junk call | code review of the policy | fold in bluff branches | [6](#step-6-parameterized-bots-and-archetypes-m3) |
| Wrong expected pot size (19, not 15) | the test failed against correct code | fixed the test | [2](#step-2-the-hand-history-schema-m1) |
| Synthetic counters with more successes than opportunities | shrinkage test returned impossible intervals | shrinker rejects $k > m$ | [10](#step-10-empirical-bayes-shrinkage-m4) |
| 15 workers exhausted memory on Windows | crash ("paging file is too small") | lazy imports, worker cap | [8](#step-8-running-a-league-parallel-reproducible-simulation) |
| Overlapping action windows silently merged | reading the code while writing up results | equal-length windows with duplication | [13](#step-13-next-action-prediction-m4) |
| Underconfidence at large samples hidden by one temperature | reliability diagrams of the first run | per-size temperatures (post hoc) | [11](#step-11-the-archetype-classifier-and-calibration-m4) |
| Plain $R^2$ flattered by between-type variance | reviewing "too good" regression numbers | within-archetype $R^2$ (post hoc) | [12](#step-12-parameter-regression-m4) |
| Cache key ignored code version and equity samples | provenance review | code fingerprint in every cache key | [14](#step-14-experimental-design-m5) |
| Provenance read at the end of a long run | provenance review | capture at the start | [14](#step-14-experimental-design-m5) |
| Function and submodule both named `best_response` under lazy re-exports | "'module' object is not callable" in some test orders | renamed the module to `evaluate.py` | [18](#step-18-exact-evaluation-best-responses-exploitability-and-an-lp-solver-p22) |
| Policy constructor renormalized valid input by one ulp | exact-equality test | repair only sums off by more than $10^{-12}$ | [18](#step-18-exact-evaluation-best-responses-exploitability-and-an-lp-solver-p22) |
| Plan asked for seeds on deterministic CFR curves | thinking about what a seed would change | seeds only for sampled methods | [19](#step-19-cfr-and-its-variants-p23) |
| Checkpoints missed powers of 10 | empty cells in the smoke-run table | decade-aligned `log_checkpoints` | [19](#step-19-cfr-and-its-variants-p23) |
| Evaluating the policy network shifted later network initializations | same seed, different results with a different checkpoint schedule | derived per-network seeds in `fork_rng` | [20](#step-20-deep-cfr-in-pytorch-p24-p210) |
| Argmax rule on all-zero advantages played fold/check with certainty | first-iteration strategy inspection | uniform play until first trained | [20](#step-20-deep-cfr-in-pytorch-p24-p210) |
| Scratch diagnostic could not import the package; wait loop never noticed | empty output after ten minutes | `PYTHONPATH`, check first lines early | [20](#step-20-deep-cfr-in-pytorch-p24-p210) |
| MAP in theta-space collapsed onto 0 under a diffuse Beta prior | a shrinkage test failed | posterior mode in logit space (with Jacobian) | [21](#step-21-leduc-archetypes-and-modeling-them-from-what-a-player-can-see-p26) |
| A 10-hand model "captured 80% of the oracle gain" | too good to be true | population-model and misspecified-model controls (post hoc) | [22](#step-22-exploiting-the-models-and-how-much-it-risks-p27) |
| A scripted doc edit matched a unique but wrong spot (a row of this table starting with "Bluff", not the glossary) | sorting the glossary later showed a missing entry | anchor replacements on surrounding context, then re-read the result | [C.3](#c3-platform-lessons-windows-powershell-onedrive) |
| Editing README.md during a reproduction run marked its results dirty | reading the rerun's metadata | kept the clean originals; edit only scratch files (or use a worktree) during such runs | [24](#step-24-phase-2-wrap-up-one-command-and-reproduction-runs) |
| Template placeholders split in the wrong order | `ValueError` on the first build | one regex pass over all placeholders | [25](#step-25-the-results-explorer-one-interactive-page-for-every-phase) |
| Charts measured before the grid settled, so their text rendered at half size | reviewing the first screenshot | a `ResizeObserver` redraws each figure when its width changes | [25](#step-25-the-results-explorer-one-interactive-page-for-every-phase) |
| Smoke-test report parser matched the injected script's own comment | a JSON decode error | reworded the comment; take the last match | [25](#step-25-the-results-explorer-one-interactive-page-for-every-phase) |
| A "390 px" screenshot was laid out about 500 px wide (Chrome's minimum window width on Windows) | text cut off at the right edge | check phone widths inside a 390 px iframe | [25](#step-25-the-results-explorer-one-interactive-page-for-every-phase) |
| Five cells of the published reference chart were misread by the automated page reader | the solver disagreed by 2-12 bb on exactly those hands, while their neighbors agreed to 0.1 bb | a second, focused read; corrections logged in the reference file | [26](#step-26-a-bridge-to-holdem-heads-up-pushfold-solved-exactly-p31) |
| A global `*.npz` ignore rule silently excluded shipped package data | `git add` refused the file | a targeted `!poker_lab/games/data/*.npz` exception | [26](#step-26-a-bridge-to-holdem-heads-up-pushfold-solved-exactly-p31) |
| A log axis for a quantity that can be exactly 0 (a rule that equals Nash) | the explorer's well-formed-chart test | linear axis; exact values in the table view | [26](#step-26-a-bridge-to-holdem-heads-up-pushfold-solved-exactly-p31) |
| An f-string reused its outer quotes (valid only from Python 3.12; the project supports 3.11) | ruff's syntax check for the target version | build the inner string first | [26](#step-26-a-bridge-to-holdem-heads-up-pushfold-solved-exactly-p31) |
| Profile credible intervals through `scipy.stats.beta.ppf` took 72% of an advisor-driven simulation | `cProfile`, after a pilot ran 10x slower than the bare engine | `scipy.special.betaincinv` (bit-identical, ~25x faster) | [28](#step-28-playing-the-solved-game-in-real-hands-behind-the-firewall-p33) |
| Aggregation divided by a standard error that was exactly zero (Nash vs Nash, duplicate dealing, pure equilibrium at 20 bb) | a crash after 20 minutes of simulation, before any output | zero-variance estimates have no z and must equal the exact value (amendment committed before the rerun); the quick config covers the case | [28](#step-28-playing-the-solved-game-in-real-hands-behind-the-firewall-p33) |
| A scratch script's output path was a literal `%s` (a shell substitution had failed) | a stray folder in the repository root | moved the files out; scratch outputs go to the scratch directory | [28](#step-28-playing-the-solved-game-in-real-hands-behind-the-firewall-p33) |
| Test tables reused the default hand-id prefix, so the profiler counted their hands as duplicates | a failing skip-count test (the code was right, the test wrong) | a distinct `hand_id_prefix` per table | [29](#step-29-ranked-exploit-recommendations-with-confidence-p34) |
| A 2.5% threshold estimated from 120 calibration bots (about the third-smallest p-value) gave 10-15% false alarms in a pilot | a second pilot that varied the calibration | 300 calibration bots per seed, decided on development seeds before the pre-registration | [30](#step-30-checking-the-model-before-trusting-it-p35) |
| A test assumed the first hand always gives the opponent a decision (it does not when the hero folds its small blind) | the test failed against correct code | the order test compares a forward and a reversed history instead | [30](#step-30-checking-the-model-before-trusting-it-p35) |
| A takeaway said Bayes and mode best responses differ "until 100 hands", though the interval already included zero at 30 | re-reading the sentence against the table | the sentence is now computed from the intervals | [29](#step-29-ranked-exploit-recommendations-with-confidence-p34) |
| A card dealt twice crashed the importer (PokerKit only warns, then fails an assertion) | the pre-registered fault injection: 40 crashes falsified its first prediction | every known card checked for repeats before the replay; assertions become typed rejections | [32](#step-32-importing-hand-histories-from-text-files-p36) |
| The show-versus-deal check compared sets, so a deal of `9h9h` "repaired" by a show of `9h9s` was accepted | diagnosing those crashes; the 8 cases had been counted as unchanged | multisets | [32](#step-32-importing-hand-histories-from-text-files-p36) |
| File keys relative to each folder collided across the Pluribus folders (`100/0` and `101/0` both became `0`) | a fixed-point check on two folders rejected half the hands as duplicates | keys relative to the common parent (or an explicit `root`) | [32](#step-32-importing-hand-histories-from-text-files-p36) |
| `invert_yaxis()` on every panel of a shared-y figure flipped it back (an even number of panels) | looking at the figure: rows bottom-up, legend over the data | invert once | [32](#step-32-importing-hand-histories-from-text-files-p36) |
| A pilot run wrote into the committed results folder (the script's default output ignores the config's name) | `git status` before the pre-registration commit | deleted the pilot output; quick runs go through `run_phase3 --quick` | [32](#step-32-importing-hand-histories-from-text-files-p36) |
| Seating grouped by the players' order: PHH lists players in dealing order, which rotates with the button, so every hand became its own session | loader check (10,000 blocks where 92 session folders were expected) | group a session's hands by the set of players | [33](#step-33-priors-from-real-players-and-do-the-bot-trained-models-transfer-p37) |
| A support check at alpha = 1% could never flag a 2,000-hand profile: 60 held-out windows make the smallest p-value 1/61 | reasoning about the conformal p-value's floor during development | alpha = 5% | [33](#step-33-priors-from-real-players-and-do-the-bot-trained-models-transfer-p37) |
| Comparing players' consistency gap with all bots mixes miscalibration with archetype mix (the gap is about 0 for calling stations and -0.1 for TAG bots) | per-archetype gaps in a development run | compare each player with bots of the same reference label | [33](#step-33-priors-from-real-players-and-do-the-bot-trained-models-transfer-p37) |
| A min-players rule over every statistic would never fire: even 300 bots give a c-bet prior worse than uniform | quick stand-in run | rule on the mean over statistics; per-statistic failures reported | [33](#step-33-priors-from-real-players-and-do-the-bot-trained-models-transfer-p37) |
| Patch scripts wrote CRLF line endings (Python's text mode on Windows) | git's line-ending warning | write bytes; normalize before committing | [33](#step-33-priors-from-real-players-and-do-the-bot-trained-models-transfer-p37) |
| Averaging hand embeddings over a window plateaued at about 92% accuracy at 2,000 hands (logistic 95%): a mean must learn ratios such as c-bets per c-bet chance from 240 players | development seeds 98 and 99, test windows | learned shrunk rates (soft successes and failures, a Beta prior per rate) | [34](#step-34-a-sequence-model-over-raw-actions-against-hand-written-stats-p38) |
| The check against Phase 1's published numbers matched seeds but not the configuration: the quick run's two seeds are the first two of the full run's | a KeyError in the quick run (no 400-hand row) | compare the results file's whole configuration first | [34](#step-34-a-sequence-model-over-raw-actions-against-hand-written-stats-p38) |
| An experiment helper named `test_windows`, imported into a test module, was collected by pytest as a test | the first test run (fixture 'pool' not found) | renamed `held_out_windows` | [34](#step-34-a-sequence-model-over-raw-actions-against-hand-written-stats-p38) |
| An odd-chip allowance of less than one unit, pre-registered for PokerKit's loader, assumed one leftover unit; three winners can leave two, and PokerKit gives both to one winner | the real run (5 of 80,000 checks failed) | reported as a failed prediction; a post-hoc re-check in whole units explains all 73 differences | [35](#step-35-schema-v2-big-blind-antes-straddles-and-minimum-bets-p310) |
| A content hash over gzip files changed between identical runs (gzip headers carry a timestamp) | comparing two runs' results field by field | hash the decompressed content | [35](#step-35-schema-v2-big-blind-antes-straddles-and-minimum-bets-p310) |
| A first design with only slow and fast adapters left almost nothing to exploit: both drifted to near-equilibrium ranges against every hero | the development pilot on seeds 98 and 99 | a calls-only adapter that keeps a leak, added before the pre-registration | [36](#step-36-opponents-that-adapt-to-the-hero-p39) |
| A run's wall time (5.3 hours, the laptop closed for most of it) was reported as its run time, three times the pilots' estimate | the owner, who knew the laptop had been closed | report times only from runs the machine stayed awake through; the Phase 3 wrap-up run measured 1.6 hours | [36](#step-36-opponents-that-adapt-to-the-hero-p39) |
| Two explorer takeaways claimed more than the data show (an adapter's ranges "chase the hero's"; the best response loses because the hero's model remembers every hand) | checking each sentence against the plotted trajectories and the results file | rewritten as numbers computed from the results file, labeled post hoc | [36](#step-36-opponents-that-adapt-to-the-hero-p39) |
| A legend covered a curve's first point; one figure used one color for a hero in one panel and a kind of bot in the other | looking at the figures | legend moved, colors reassigned, figures redrawn from the results file | [36](#step-36-opponents-that-adapt-to-the-hero-p39) |
| A check for sleep looked only for Kernel-Power events 42 and 107; the laptop uses Modern Standby, which writes 506 and 507, so an 18-minute lid closure was missed and a slow test run blamed on the battery | listing every Kernel-Power event during the wrap-up run | check both pairs (C.3) | [37](#step-37-phase-3-wrap-up-one-command-reproduced) |
| A comparison script matched wall-clock keys as substrings, so it would have skipped `fallback_second_half`, a data field | the tool's own test | keys split at underscores and matched as whole words | [37](#step-37-phase-3-wrap-up-one-command-reproduced) |
| Figures redrawn in a results commit by newer code than the commit their JSON records, said only in the commit message | a rerun at the recorded commits, whose figures differed | noted in C.2 and Step 37 | [37](#step-37-phase-3-wrap-up-one-command-reproduced) |
| A pre-registered ±3% tolerance on sampled SDs ignored kurtosis: for Nash against Nash with duplicate dealing (kurtosis 475 at 5 bb) a sampled SD's relative standard error is 3.4%, and in simulation the band is missed 39% of the time at 5 bb and 62% at 10 bb | the real run: F1 failed at 0.957 | reported as failed; post hoc, z-scores with a kurtosis-aware standard error $\sqrt{(\kappa - 1)/(4n)}$ (largest 2.92); set a tolerance on a sampled SD from its own standard error | [38](#step-38-variance-reduction-with-aivat-measured-exactly-p211) |
| `invert_yaxis()` on both panels of a shared-y figure, again: the Leduc figure's rows were upside down (Step 32's mistake, repeated) | looking at the figure after the run | invert once; the figure redrawn from `aivat.json` (no number changed); C.4 | [38](#step-38-variance-reduction-with-aivat-measured-exactly-p211) |
| The sampled check compared every estimator with the true value, so the biased estimator failed it as well as the bias prediction (one fault counted twice) | development runs on seeds 98 and 99 | compare each sample with that estimator's own exact mean; judge bias exactly (A1, C2); disclosed in the plan | [38](#step-38-variance-reduction-with-aivat-measured-exactly-p211) |
| `choose` returned the recommender's own equilibrium arrays on a fallback; a caller that changed them would have changed every later recommendation | the reviewer subagent's reading of the diff | return copies | [39](#step-39-faster-code-identical-results) |
| `restricted_nash_sweep` iterated over its `p` values once per seat, so a generator would have failed | the reviewer subagent | make a tuple first | [39](#step-39-faster-code-identical-results) |
| A pickle or deep copy of a `SequenceClassifier` would have carried its embedding memo, keyed by the ids of the original's objects | the reviewer subagent | `__getstate__` drops the memo; a test checks that a copy starts empty and predicts the same bytes | [39](#step-39-faster-code-identical-results) |
| The first report of the sequence rerun credited the embedding memo with the step's 34%; unchanged training code took 27-39% less time that day | Amdahl's bound from the profile: the memo can save at most 6.6% (post hoc) | a same-session comparison of old and new code; C.3 | [39](#step-39-faster-code-identical-results) |
| A failed hot-started LP aborted the whole restricted Nash sweep | the reviewer subagent's reading of the diff | that LP solved from scratch on a new HiGHS instance, the sweep continues; a test forces the failure | [40](#step-40-a-hot-start-parallel-replicates-and-a-split-test-suite) |
| Solutions from SciPy's HiGHS bindings were accepted on HiGHS's status alone, so they could pass where `linprog` fails | the reviewer subagent | `linprog`'s own check (no NaN; bounds and constraints within about $3.2 \times 10^{-4}$); a test at 0.9 and 1.1 times the tolerance | [40](#step-40-a-hot-start-parallel-replicates-and-a-split-test-suite) |
| The sweep memo hashed $p$ and $A y_{\text{fix}}$ as given, so equal values of different dtypes got different keys | the reviewer subagent | convert to Python floats and float64 before hashing, and build the LPs from the converted values; a float32 test | [40](#step-40-a-hot-start-parallel-replicates-and-a-split-test-suite) |
| No test checked that a sweep's later LPs were really hot-started | the reviewer subagent | a test reads HiGHS's presolve status (`kNotPresolved` for exactly the hot-started solves) | [40](#step-40-a-hot-start-parallel-replicates-and-a-split-test-suite) |
| A failing replicate was reported only after the others finished training (up to about half an hour later at full size) | the reviewer subagent | lazy submission: at most N in flight; the failure prints at once and no new replicate starts | [40](#step-40-a-hot-start-parallel-replicates-and-a-split-test-suite) |
| Parallel replicates defaulted to two jobs without a full-size measurement, and memory was not considered | the reviewer subagent | default 1 before the full-size measurement, kept after it; warnings for threads above logical processors and free memory below about 5 GB per job | [40](#step-40-a-hot-start-parallel-replicates-and-a-split-test-suite) |
| One and two jobs compared by whole-step time (2,782 s against 2,023 s), though the two-job run had filled the empty simulation cache (1,008 s) that the one-job run reused | the two-job log's "hands ready" times, 199-204 s per replicate | compare training times (2,003 s against about 1,746 s); add the simulation to both for whole-step estimates | [40](#step-40-a-hot-start-parallel-replicates-and-a-split-test-suite) |
| A new 33 s test doubled the suite's time (65-67 s against 33-36 s): a whole-file scheduler started its file late | timing the suite before the commit | tests marked `long` start first, under `--dist worksteal` (post hoc): 53.5-53.9 s | [40](#step-40-a-hot-start-parallel-replicates-and-a-split-test-suite) |
| The plan cited Step 27 for the push/fold measurement (it is Step 28) and quoted "about 3x", a ratio of two medians, while the code compares the median of each matchup's naive / duplicate (2.06×) | the reviewer subagent's reading of the plan | corrected before the pre-registration commit; the docstring says which statistic it uses | [41](#step-41-duplicate-dealing-at-more-than-two-seats-p211b) |
| `predictions()` ran before `save_json`, so a path stopping at a dict would have raised `TypeError` after the whole simulation, with nothing saved | the reviewer subagent | a non-number fails its prediction with the value None; a test evaluates every prediction on made-up results of the real shape | [41](#step-41-duplicate-dealing-at-more-than-two-seats-p211b) |
| Per-archetype bootstrap CIs rested on 2 lineups in development (a CI of [1.84×, 1.85×] inside an IQR of [1.80×, 1.88×]) | the reviewer subagent | CIs blank below `MIN_CI_LINEUPS = 8`, and no prediction uses them; post hoc, 10 lineups still give about 90% coverage | [41](#step-41-duplicate-dealing-at-more-than-two-seats-p211b) |
| A figure's error bars could be negative, since a percentile CI need not contain the median, and matplotlib 3.11 refuses a negative error | the reviewer subagent | clip them at 0 | [41](#step-41-duplicate-dealing-at-more-than-two-seats-p211b) |
| The exactness claim said "identical agents whose play depends only on their cards", though agents can see player ids | the reviewer subagent | copies of one deterministic agent that ignores player ids; a test with an ante covers odd chips | [41](#step-41-duplicate-dealing-at-more-than-two-seats-p211b) |
| `DuplicateDealEvaluator()` on a 6-table match would have averaged tables two at a time with no error | the reviewer subagent | `paired_results` refuses other than two tables; `DuplicateMatch.evaluate` groups by the number of tables | [41](#step-41-duplicate-dealing-at-more-than-two-seats-p211b) |
| The plan's docstring says copies of one bot "spread about 2% in development"; over all copies lineups the median within-lineup relative SD is 7.0% and 7.4% on those seeds | checking the sentence against the development outputs while writing the step | reported in Step 41; no prediction used it, and the committed plan stays as it was | [41](#step-41-duplicate-dealing-at-more-than-two-seats-p211b) |
| The `duplicate_ratio` figure's legend sat on the 6.08× heads-up lineup | reading the figure while drafting the step | the legend moved, the figure redrawn from the committed JSON; no number changed | [41](#step-41-duplicate-dealing-at-more-than-two-seats-p211b) |
| Step 41's draft said the bots' randomness was a likely reason copies stop short (the library's own argument makes it the only possible one) and that a flat share explains a flat ratio (the ratio is $1/(1-s)$) | the reviewer subagent, tracing every claim before the commit | both restated, in the log, the explorer and the README; what stays untested is how the randomness divides and why the share is flat | [41](#step-41-duplicate-dealing-at-more-than-two-seats-p211b) |
| The plan had two predictions that could not fail once A1 held, since A1 ties the cancelled shares to committed numbers | the reviewer subagent | dropped; the numbers are stated as known | [42](#step-42-where-six-copies-shortfall-comes-from-p211c) |
| The plan reported one split of the shortfall, without saying it depends on which decisions at two tables share a draw | the reviewer subagent | a second pairing rule, 6 more conditions; the split moves from 87% to 53% for mixed actions | [42](#step-42-where-six-copies-shortfall-comes-from-p211c) |
| One pooled variance per lineup, proposed by a review as more efficient, took nits' cancelled share on a development seed from 26.8% to 0.8% | trying it on the development seeds | each condition keeps its own variance: a rare big pot then cancels | [42](#step-42-where-six-copies-shortfall-comes-from-p211c) |
| A test compared results with `==`, and `NaN != NaN` for a stat with no opportunities | the test failing | it compares `repr`, and means skip NaN | [42](#step-42-where-six-copies-shortfall-comes-from-p211c) |
| A commit message said 34 predictions (33), and an explorer subtitle said the calling station barely plays a pot under `off_mixed` (its VPIP is 0.410; the spread of its results is what falls) | a script's count; reading the built page | corrected before anything was pushed | [42](#step-42-where-six-copies-shortfall-comes-from-p211c) |
| The plan's prediction against the equilibrium opponent could not fail (no strategy gains against an equilibrium), the KL correlation was one-sided, D5 tested a mean, and restricted Nash, the data-biased response, the population prior and exploitability had no prediction | the reviewer subagent | all fixed before the commit; checks counted apart from predictions | [43](#step-43-exploitation-when-the-model-is-wrong-p212) |
| D5's bound left a margin of 0.007 on two development seeds | the run on seed 2026 (+0.024 against +0.02) | reported as failed; the substance rests on other numbers | [43](#step-43-exploitation-when-the-model-is-wrong-p212) |
| A path-check script without a `__main__` guard broke the worker pool on Windows | the run failing | the guard | [43](#step-43-exploitation-when-the-model-is-wrong-p212) |
| The `misspec_by_hands` figure had a wide gap between its panels: `tight_layout` counted the legend under the left panel as part of it | reading the figure while drafting the step | a legend under both panels, redrawn from the committed JSON; no number changed | [43](#step-43-exploitation-when-the-model-is-wrong-p212) |
| A click answered the next decision: an answered turn still looked pending until the engine thread woke | stale-decision error in the first smoke test | `pending()` hides a turn once answered; turn tokens | [44](#step-44-playing-against-the-model-a-local-web-page) |

---

# Part E: glossary

| Term | Meaning |
|---|---|
| Ablation | a version of a model with one ingredient removed (here: the order within a hand, or the raw tokens), to measure what that ingredient contributes |
| AIVAT | a win-rate estimator for a player whose strategy is known: control-variate terms at chance and hero steps plus imaginary observations; unbiased for any value function, but it needs the opponent's cards after every hand (Burch et al. 2018; Step 38) |
| All-in | betting every remaining chip |
| All-in EV | a called all-in scored by its equity instead of the board that came; unbiased, removes board luck |
| Alternative optima | several optimal solutions of one LP: when the objective is parallel to a face of the feasible set, every point of that face is optimal, and the simplex method stops at whichever optimal vertex it reaches first (Step 40) |
| Amdahl's law | if a fraction $f$ of the time is made $s$ times faster, the whole is $1/((1 - f) + f/s)$ times faster, never more than $1/(1 - f)$ (Amdahl 1967; Step 39) |
| Antithetic variates | Monte Carlo runs built to be negatively correlated, so their mean has less variance than the mean of independent runs (Hammersley and Morton 1956); duplicate dealing is an instance (Step 41) |
| Basis (of an LP) | as many columns as the LP has rows, with an invertible matrix; setting every other variable to its bound leaves one solution, a vertex of the feasible set if it also satisfies the bounds. Feasibility depends on the constraints only, not on the costs (Step 40) |
| Bayes action | the decision with the highest posterior expected payoff |
| bb, bb/100 | big blind; win rate in big blinds per 100 hands |
| Best response | a strategy that maximizes payoff against a fixed opponent |
| Big-blind ante | an ante the big blind posts for the whole table, instead of every seat posting its own (modern tournaments) |
| Button (BTN) | dealer position; acts last after the flop |
| C-bet | continuation bet: the preflop raiser bets the flop |
| Calibrated p-value | a p-value whose threshold is set from its distribution on data simulated from the model, so the false-alarm rate is what it says |
| Calibration | predicted probabilities match observed frequencies |
| Capability | an unforgeable reference whose possession grants a permission |
| CFR | counterfactual regret minimization: one regret matcher per information set |
| Cluster bootstrap | resampling whole clusters (players, or a table's lineup), not rows |
| Common random numbers | giving the alternatives being compared the same random inputs, so their difference reflects only how they differ (here: the same deals and draws for every hero) |
| Concentration (of a Beta prior) | $\kappa = \alpha + \beta$, the prior's weight in pseudo-opportunities; large when the players are alike |
| Condition variable | a lock plus a waiting room: a thread holding the lock can wait (releasing it) until another thread notifies; used to let the engine wait for a person's click (Step 44) |
| Confirmatory analysis | testing a hypothesis fixed in advance on data it was not derived from; the counterpart of exploratory analysis |
| Conformal p-value | the rank of a new score among held-out scores, $(1 + \#\{\text{held-out} \ge \text{new}\})/(m + 1)$; valid for exchangeable data whatever the model (Vovk et al. 2005) |
| Conjugate prior | a prior whose posterior has the same family (beta-binomial) |
| Content-Security-Policy | a response header telling the browser where a page may load scripts, styles and data from; `'self'` allows only the page's own server (Step 44) |
| Control variate | a term with known zero mean subtracted from an estimator to cut its variance (all-in EV, Step 28; AIVAT, Step 38) |
| CORS preflight | the request a browser sends before a cross-site request it cannot send freely (for example a JSON `POST`); a server that does not approve it never receives the real request (Step 44) |
| Counterfactual value | an action's value at an information set, weighted by the probability that chance and the opponent reach it |
| Coupling | running two random experiments on shared random numbers, so they disagree less often than independent runs would; two thresholds $p_1$, $p_2$ on one uniform disagree with probability $|p_1 - p_2|$ |
| Coverage | the share of intervals that contain the true value; a 90% interval should cover it 90% of the time |
| Credible interval | interval containing the parameter with a given posterior probability |
| CUSUM | cumulative sum of residuals; its largest excursion detects a change in level (Page 1954) |
| Cyclic rotation (of seats) | at table $r$ player $j$ sits in seat $(j + r) \bmod n$, so each player holds each seat once per deal and keeps the same neighbors (Step 41) |
| Data-biased response | robust response that trusts the model per information set by data count |
| Dead money | chips in the pot that no player has to match, such as antes; they are won, never returned as an uncalled bet |
| Deep CFR | CFR with neural networks in place of the regret and average-strategy tables |
| Deep set | a model of a set that applies the same network to each item, sums, and reads the sum; any function that ignores order can be written this way (Zaheer et al. 2017) |
| Degenerate LP | primal degenerate: one vertex described by several bases (a basic variable at zero), so the simplex method can pivot without moving; dual degenerate: a zero reduced cost of a nonbasic variable at an optimum; then there are alternative optima unless the vertex is also primal degenerate (Step 40) |
| Delta method | the variance of a smooth function of an estimate, from its slope: $\operatorname{Var}(g(\hat\theta)) \approx g'(\theta)^2 \operatorname{Var}(\hat\theta)$ |
| Design effect | the variance of a cluster's mean over that of the same number of independent units, $1 + (m-1)\rho$ for clusters of $m$ with intraclass correlation $\rho$ (Kish 1965); duplicate dealing's ratio is its inverse (Step 41) |
| Digest (hash) | a short fixed-size fingerprint of a byte string (here SHA-1, 160 bits); equal inputs give equal digests, and different inputs share one with negligible probability unless built to by an attacker (Step 39) |
| Dirty flag | whether the git checkout had uncommitted changes to tracked files when a run started; recorded with every result, so a number can be traced to exact code |
| DNS rebinding | pointing an attacker's domain at 127.0.0.1 so the browser treats the attacker's page as the same site as a local server; defeated by checking the `Host` header (Step 44) |
| Duplicate dealing | playing each deal at n tables with the n players' seats rotated (heads-up: twice with seats swapped), so card luck cancels in part; simulation only (Steps 23 and 41) |
| ECE | expected calibration error |
| Effective sample size (ESS) | for weighted draws, $1 / \sum_k w_k^2$: how many equally weighted draws the sample is worth |
| Effective stack | the smaller of two players' stacks: the most either can win or lose against the other |
| Empirical Bayes | estimating the prior from the population itself |
| Equity | probability of winning (ties count half) |
| Equivalence test | a test that the difference is small (its interval inside a margin), not merely not shown to be large (Schuirmann 1987) |
| Exchangeable | a sequence whose probability does not change when it is reordered; a stationary player's hands are |
| Exploitability | how much a best response wins against a strategy (NashConv / 2) |
| Fault injection | damaging valid inputs on purpose to measure what a checker catches; mutation testing aimed at data |
| Fictitious play | each player best-responds to the observed frequencies of the opponent's past play (Brown 1951); in two-player zero-sum games the frequencies converge to an equilibrium (Robinson 1951) |
| Fixed point (of a round trip) | data that come back unchanged after export and re-import |
| Game value | what the first player gets when both play an equilibrium (minimax theorem) |
| Garden of forking paths | the unplanned choices made after seeing data, each an uncounted comparison that makes the chosen result look better than it is |
| GRU | gated recurrent unit: a recurrent network that reads a sequence one item at a time, with gates that decide what to keep (Cho et al. 2014) |
| Half-life (of a weighted average) | the age at which an observation's weight has fallen to half; with decay $d$ per step it is $\log 2 / \log(1/d)$ steps, so $d = 2^{-1/h}$ |
| Hand class | one of the 169 strategically distinct starting hands (pairs, suited, offsuit), e.g. AKs |
| Hands to evidence | $\log 20 / \mathrm{KL}$: the hands at which the expected log-likelihood ratio against the closest model reaches 20:1; optimistic, since it assumes the alternative is known (Step 43) |
| Headless browser | a browser run without a window, driven from the command line (screenshots, DOM dumps, scripted checks) |
| Hero's view (evaluation) | an evaluator that knows only what the hero saw (its cards, the actions, the public cards), never the opponent's cards; it can correct only steps whose probabilities do not depend on them (Step 38) |
| Hot start (warm start) | starting a solver from an earlier solve instead of from scratch: a warm start hands it a starting point such as a basis; a hot start keeps the solver instance and its working data and changes only the data, here the costs (Step 40) |
| HUD | heads-up display of opponent statistics |
| Imaginary observations | replacing a hand's result by its average over every private hand the hero could have held, weighted by chance's probability times the hero's probability of playing as it did; exact when the opponent cannot see the hero's cards (Burch et al. 2018) |
| Importance sampling | estimating averages under a target distribution from draws of an easier proposal, each weighted by target / proposal |
| Information class (of an evaluator) | the histories an evaluator cannot tell apart at a point of a hand; AIVAT's terms and averages run over them |
| Information set | the histories a player cannot tell apart when acting |
| Intraclass correlation | the mean correlation between units of one group, here a player's results at two tables of one deal; at least $-1/(m-1)$ for groups of $m$, since a variance cannot be negative (Step 41) |
| Kullback-Leibler divergence | $\mathrm{KL}(q \| p) = \mathrm{E}_q[\log q(X) - \log p(X)]$: the expected log-likelihood per observation that $p$ loses to the truth $q$; zero only when they agree (Kullback and Leibler 1951) |
| Kurtosis | $\kappa = \mathbb E[(X - \mu)^4] / \sigma^4$, 3 for a normal distribution; large when rare extreme values carry most of the variance, which makes a sample SD noisy: its relative standard error is about $\sqrt{(\kappa - 1)/(4n)}$ |
| Laplace approximation | a normal approximation to a posterior at its mode, with the inverse Hessian of the negative log posterior as covariance |
| Latin square | an $n \times n$ array of $n$ symbols in which each symbol occurs once in every row and every column; here the rows are players, the columns tables and the symbols seats (Fisher 1935; Step 41) |
| Law of total variance | $\operatorname{Var}(Y) = \mathbb E[\operatorname{Var}(Y \mid D)] + \operatorname{Var}(\mathbb E[Y \mid D])$: the spread within groups plus the spread between them |
| Lazy evaluation | computing a value only when it is needed; skipping unneeded work is exact when that work is pure (changes no state that later work reads, a shared random generator included) (Step 39) |
| Leave one player out | fit on every player but one, score the one; repeated for each, so no player is in its own training data |
| Linear program (LP) | optimizing a linear function subject to linear equalities and inequalities; the feasible set is a polyhedron, and if an optimum exists and the feasible set has a vertex (always so when every variable is bounded below, as in standard form), one optimum is at a vertex |
| Log-loss | mean negative log-probability of the true class |
| Logical processor | a hardware thread that the operating system can schedule; with SMT, two per physical core |
| Lower confidence bound | the low end of an interval; ranking by it prefers what is surely good over what is good on average |
| LPT scheduling | longest processing time first: sort jobs longest first and give each to the first free worker; the makespan is at most $4/3 - 1/(3m)$ times the best on $m$ workers (Graham 1969; Step 40) |
| LRU cache | a cache of bounded size that, when full, drops the entry used longest ago (Step 39) |
| Mahalanobis distance | distance scaled by a covariance, $(x - \mu)^\top S^{-1}(x - \mu)$: how many standard deviations away, in every direction at once |
| Makespan | the time from the start of a set of jobs to the end of the last; on $m$ workers at least $\max(\text{longest job}, \text{total work}/m)$ (Step 40) |
| Martingale | a sequence whose expected next value, given everything so far, is its current value; its steps have mean zero and are uncorrelated |
| mbb/hand | milli-big-blinds per hand |
| MCCFR | Monte Carlo CFR: CFR on sampled parts of the tree |
| Memoization | storing a function's results under its arguments and answering repeated calls from memory; exact when the key determines the input and the function is deterministic (Michie 1968; Step 39) |
| Mid-p value | a p-value that counts ties with the observed statistic as half, for discrete statistics |
| Migration (of a schema) | reading records written in an older version of a format as the current version, filling in what the old version implied |
| Minimum bet | the smallest bet allowed, which is also the smallest raise increment before anyone raises; usually one big blind |
| Minimum detectable effect | the smallest true difference a test would detect with given power ($2.8\,SE$ for 80% at a two-sided 5% level) |
| Misspecification | the true distribution is outside the model family, so no parameter value is right; measured here by the KL divergence to the closest member (Step 43) |
| MIVAT | control-variate terms at chance steps only, valued at full histories: the chance-only method Burch et al. compare AIVAT with, as implemented here; it may differ from the paper's, whose Leduc self-play SD it does not reproduce (Step 38) |
| Modern Standby | the sleep mode of many recent Windows laptops (S0 low-power idle): the system stays nominally on, and programs run slowly or not at all; logged as Kernel-Power events 506 and 507 |
| Moebius inversion | recovering the terms of a sum over subsets, $G(T) = \sum_{B \subseteq T} W_B$, as $W_B = \sum_{T \subseteq B} (-1)^{|B| - |T|} G(T)$ |
| Muck | discard a hand without showing it |
| Nash equilibrium | a profile where each strategy is a best response to the other |
| NashConv | sum of both players' best-response gains; 0 at equilibrium |
| Non-associativity (floating point) | $(a + b) + c$ and $a + (b + c)$ can differ in the last bit because every operation rounds, so a sum depends on its order and a batched computation on its batch (Step 39) |
| Odd chip | the chip left over when a pot does not split evenly; rules give it to one winner, some records split it in halves |
| Off-tree | a real spot the abstract game has no information set for (e.g. a min-raise in push/fold) |
| Oracle | a model given the true answer's inputs, to measure the ceiling |
| Oversubscription | more runnable threads than logical processors: the operating system time-slices them, and parallel numeric code whose threads wait for each other slows down sharply (Step 40) |
| Paired comparison | two methods scored on the same units, so variation between units cancels in the difference |
| Percentile bootstrap | an interval from the quantiles of a statistic's resampled values; too narrow with few clusters (about 90% coverage for a nominal 95% at 10 clusters), and it need not contain the sample's statistic (Step 41) |
| Perfect recall | players never forget their own actions or observations |
| Permutation control | retrain on shuffled labels; above-chance results mean leakage |
| PHH | Poker Hand History, an open TOML-based text format for recording hands (Kim 2024) |
| Posterior predictive check | compares data with replicates drawn from the fitted model; a small p-value says the model does not fit |
| Pre-registration | committing the plan, the metric and the predictions with their thresholds before the run that tests them; a threshold set from development seeds can show that a result replicates, not why (Steps 14 and 41) |
| Presolve | an LP solver's first pass: simplify the LP (redundant rows, fixed columns, tighter bounds), solve the smaller LP, map the answer back (postsolve); HiGHS skips it when it starts from a basis (Step 40) |
| Process pool | a fixed set of worker processes that run submitted tasks (`ProcessPoolExecutor`); each process has its own memory, so tasks share only what is sent to them (Steps 8 and 40) |
| Profiler | a tool that measures where a program spends its time; Python's is `cProfile` (Steps 1, 39) |
| Pseudo-true parameter | the parameter maximum likelihood converges to under misspecification: the family member closest to the truth in KL divergence (White 1982) |
| Push/fold | a short-stack strategy space: go all-in or fold, and call or fold facing an all-in |
| Range | the set of hands a player plays in a spot |
| Rao-Blackwell | replacing an estimator by its conditional expectation given some information never increases its variance (here: scoring each block by its exact expected value instead of the chips won) |
| Reach-weighted total variation | the mean over information sets, weighted by how often play reaches them, of half the summed absolute difference between two action distributions (Step 43) |
| Realization plan | sequence-form strategy: probability of each own action sequence |
| Reduced cost | the change in an LP's objective per unit of a variable brought into the basis, $d_j = c_j - M_j^\top y$; a basis is optimal when none is negative (for minimization), within the solver's tolerance (Step 40) |
| Regret matching | play actions in proportion to their positive cumulative regret |
| Reservoir sampling | keeping a uniform random sample of a stream in fixed memory |
| Restricted Nash response | best response to a model, hedged against an adversary with probability 1-p |
| SeedSequence | NumPy's seed mixer: a pool of four 32-bit words, with shorter entropy padded with zeros, so `[s]` and `[s, 0, 0]` are one seed; hence a nonzero tag word per stream (Step 41) |
| Sequence | an (information set, action) pair |
| Shapley value | a fair division of a game's total: each player's contribution averaged over the orders of joining; for variance terms, a source's own term plus equal parts of its interactions (Shapley 1953) |
| Share of perfect cancellation | $-(n-1)\rho$ for a player's results at $n$ tables of one deal: 0 for none, 1 at the bound $\rho = -1/(n-1)$; a share $s$ gives duplicate dealing the ratio $1/(1-s)$ at any $n$ (Step 41) |
| Shrinkage | pulling noisy estimates toward a population mean |
| Side pot | pot that an all-in player for less is not eligible for |
| Simplex method | the classic LP algorithm: move from vertex to adjacent vertex of the feasible set, one basis change (pivot) at a time, while the objective improves; stop when no reduced cost is negative (Dantzig 1963; Step 40) |
| Simpson's paradox | a pooled comparison that differs from (or reverses) the comparisons within groups, because the groups differ in size or kind |
| Simultaneous multithreading (SMT) | one physical core runs two hardware threads that share its execution units and caches, so the second thread adds much less than a second core; Intel calls it hyper-threading. This laptop's AMD Ryzen 9 8945HS has 8 cores, 16 logical processors (Step 40) |
| Sobol-Hoeffding decomposition | writing a function of independent inputs as orthogonal pieces, one per subset of inputs, so its variance splits into one non-negative term per subset (Hoeffding 1948, Sobol' 1993) |
| Spawn (start method) | starting a worker process as a fresh interpreter that imports the main module and receives its task by pickling; the only start method on Windows, hence the `if __name__ == "__main__":` guard (Steps 8 and 40) |
| Stable sort | a sort that keeps equal elements in their original order; Python's is stable (Step 40) |
| Stationarity | the data-generating process does not change over time |
| Straddle | a voluntary blind posted before the cards, usually twice the big blind, by the player after the big blind (or the button); the straddler acts last preflop |
| Stratified sampling | splitting a Monte Carlo estimate into strata (here: combo pairs) sampled equally, removing the variance between strata |
| Street | betting round: preflop, flop, turn, river |
| Support check | flags inputs unlike the training data, so a model's output is not trusted where it was never trained |
| Temperature scaling | dividing log-probabilities by $T$ to fix confidence |
| Throughput and latency | how many jobs finish per unit of time, and how long one job takes; two jobs side by side that each run $s$ times slower raise throughput by $2/s$ while each takes longer (Step 40) |
| Token (of a hand) | one of a player's decisions in a hand (situation, action, size), or the hand's closing outcome, as a vector a sequence model reads |
| Tolerance (LP solver) | how far a solver lets constraints (primal) and reduced costs (dual) miss and still calls a solution feasible and optimal; $10^{-7}$ by default in HiGHS, so a vertex whose reduced costs miss by less than that can be returned as optimal, though another is slightly better (Step 40) |
| Total-effect index | the share of the variance left when everything except a group of inputs is held fixed: all the terms that involve the group (Homma and Saltelli 1996) |
| Tower property | $\mathbb E\big[\mathbb E[X \mid S]\big] = \mathbb E[X]$: averaging a conditional expectation over what it conditions on gives the plain expectation |
| ulp | unit in the last place: the gap between a floating-point number and the next; just above 1 it is $2^{-52}$ for 64-bit floats and $2^{-23}$ for 32-bit floats (Step 39) |
| Uncalled bet | the part of a bet nobody matched, returned to the bettor |
| Value function (for control variates) | any number per history or class used to build control-variate terms; the estimate stays unbiased whatever it is, and loses more variance the closer it is to the true expected value |
| VPIP / PFR / WTSD / W\$SD | see [A.3](#a3-hud-statistics-and-why-denominators-matter) |
| Winner's curse | the best-looking of several noisy estimates tends to be an overestimate |
| Work stealing | scheduling in which an idle worker takes pending work from the end of a busy worker's queue (Blumofe and Leiserson 1999); pytest-xdist's `--dist worksteal` (Step 40) |
| Worktree (git) | a second checkout of the same repository at another commit, in its own folder, used here to run old code next to new |
| z-score | (estimate - true value) / standard error; about standard normal when the estimator is unbiased |

---

# Part F: references

Poker, games and engines

* Kim, J. (2023). *PokerKit: A Comprehensive Python Library for Fine-Grained Multi-Variant
  Poker Game Simulations.* arXiv:2308.07327.
* Lee, H. *PokerHandEvaluator* (phevaluator), github.com/HenryRLee/PokerHandEvaluator:
  the perfect-hash evaluation algorithm.
* Billings, D., Davidson, A., Schaeffer, J., Szafron, D. (2002). *The challenge of poker.*
  Artificial Intelligence 134. Hand strength and potential, opponent modeling.
* Southey, F. et al. (2005). *Bayes' Bluff: Opponent Modelling in Poker.* UAI. Also introduces
  Leduc hold'em.
* Ganzfried, S., Sandholm, T. (2011). *Game theory-based opponent modeling in large
  imperfect-information games.* AAMAS.
* Ganzfried, S., Sandholm, T. (2013). *Action Translation in Extensive-Form Games with Large
  Action Spaces: Axioms, Paradoxes, and the Pseudo-Harmonic Mapping.* IJCAI. (Mapping
  off-tree bet sizes onto an abstraction; Step 28)
* Kuhn, H. W. (1950). *A simplified two-person poker.* Contributions to the Theory of Games 1.
* Lanctot, M. et al. (2019). *OpenSpiel: A Framework for Reinforcement Learning in Games.*
  arXiv:1908.09453. Source of the uniform-policy NashConv used as a check.

Equilibrium, exploitation, evaluation (Phase 2 and 5)

* Koller, D., Megiddo, N., von Stengel, B. (1996). *Efficient Computation of Equilibria for
  Extensive Two-Person Games.* Games and Economic Behavior 14. (The sequence form)
* Huangfu, Q., Hall, J. A. J. (2018). *Parallelizing the dual revised simplex method.*
  Mathematical Programming Computation 10. (HiGHS, SciPy's LP solver; hot-started in Step 40)
* Hart, S., Mas-Colell, A. (2000). *A Simple Adaptive Procedure Leading to Correlated
  Equilibrium.* Econometrica 68. (Regret matching)
* Zinkevich, M., Johanson, M., Bowling, M., Piccione, C. (2007). *Regret Minimization in Games
  with Incomplete Information.* NeurIPS. (CFR)
* Lanctot, M., Waugh, K., Zinkevich, M., Bowling, M. (2009). *Monte Carlo Sampling for Regret
  Minimization in Extensive Games.* NeurIPS. (MCCFR, external sampling)
* Neller, T., Lanctot, M. (2013). *An Introduction to Counterfactual Regret Minimization.*
  EAAI model AI assignment. The gentlest introduction, with code.
* Tammelin, O. (2014). *Solving Large Imperfect Information Games Using CFR+.* arXiv:1407.5042.
* Brown, N., Sandholm, T. (2019). *Solving Imperfect-Information Games via Discounted Regret
  Minimization.* AAAI. (Linear and discounted CFR)
* Brown, N., Lerer, A., Gross, S., Sandholm, T. (2019). *Deep Counterfactual Regret
  Minimization.* ICML.
* Steinberger, E. (2019). *Single Deep Counterfactual Regret Minimization.* arXiv:1901.07621.
* Vitter, J. S. (1985). *Random Sampling with a Reservoir.* ACM Transactions on Mathematical
  Software 11. (Reservoir sampling)
* Johanson, M., Zinkevich, M., Bowling, M. (2008). *Computing Robust Counter-Strategies.*
  NeurIPS. (Restricted Nash response)
* Johanson, M., Bowling, M. (2009). *Data Biased Robust Counter Strategies.* AISTATS.
* Southey, F. et al. (2005). *Bayes' Bluff* (above) also introduces Bayesian opponent
  modeling with hidden cards in Leduc, the idea behind Step 21's marginal likelihood.
* Burch, N., Schmid, M., Moravcik, M., Morrill, D., Bowling, M. (2018). *AIVAT: A New Variance
  Reduction Technique for Agent Evaluation in Imperfect Information Games.* AAAI. (Step 38;
  Table 1's Leduc self-play numbers)
* White, M., Bowling, M. (2009). *Learning a Value Analysis Tool for Agent Evaluation.* IJCAI.
  (MIVAT as we understand it; not checked against the paper; Step 38)
* Brown, G. W. (1951). *Iterative Solution of Games by Fictitious Play.* In T. C. Koopmans (ed.),
  Activity Analysis of Production and Allocation. Wiley. (Fictitious play; Step 36)
* Robinson, J. (1951). *An Iterative Method of Solving a Game.* Annals of Mathematics 54.
  (Fictitious play converges in two-player zero-sum games; Step 36)

Statistics and machine learning

* Efron, B., Morris, C. (1975). *Data Analysis Using Stein's Estimator and Its
  Generalizations.* JASA. Classic empirical Bayes on batting averages.
* Robinson, D. (2017). *Introduction to Empirical Bayes: Examples from Baseball Statistics.*
  The beta-binomial approach used here, explained gently.
* Friedman, J. (2001). *Greedy Function Approximation: A Gradient Boosting Machine.* Annals of
  Statistics.
* Ke, G. et al. (2017). *LightGBM: A Highly Efficient Gradient Boosting Decision Tree.*
  NeurIPS. (Histogram-based boosting.)
* Guo, C., Pleiss, G., Sun, Y., Weinberger, K. (2017). *On Calibration of Modern Neural
  Networks.* ICML. (Temperature scaling)
* Cho, K. et al. (2014). *Learning Phrase Representations using RNN Encoder-Decoder for
  Statistical Machine Translation.* EMNLP. (The GRU; Step 34)
* Zaheer, M. et al. (2017). *Deep Sets.* NeurIPS. (Models of sets; Step 34)
* Kingma, D., Ba, J. (2015). *Adam: A Method for Stochastic Optimization.* ICLR.
* Loshchilov, I., Hutter, F. (2017). *SGDR: Stochastic Gradient Descent with Warm
  Restarts.* ICLR. (Cosine learning-rate schedules)
* Naeini, M. P., Cooper, G., Hauskrecht, M. (2015). *Obtaining Well Calibrated Probabilities
  Using Bayesian Binning.* AAAI. (ECE)
* Gneiting, T., Raftery, A. (2007). *Strictly Proper Scoring Rules, Prediction, and
  Estimation.* JASA.
* Efron, B., Tibshirani, R. (1993). *An Introduction to the Bootstrap.* Chapman & Hall.
* Ojala, M., Garriga, G. (2010). *Permutation Tests for Studying Classifier Performance.*
  JMLR.
* Kaufman, S., Rosset, S., Perlich, C. (2012). *Leakage in Data Mining: Formulation,
  Detection, and Avoidance.* ACM TKDD.
* Gelman, A., Carlin, J., Stern, H., Dunson, D., Vehtari, A., Rubin, D. (2013). *Bayesian Data
  Analysis*, 3rd ed. CRC Press. Chapters 4 (normal approximation at the mode) and 10
  (importance sampling). (Step 29)
* Kish, L. (1965). *Survey Sampling.* Wiley. (The effective sample size of weighted samples; the
  design effect of clusters, Step 41)
* Fisher, R. A. (1935). *The Design of Experiments.* Oliver and Boyd. (Latin squares; Step 41)
* Hammersley, J. M., Morton, K. W. (1956). *A new Monte Carlo technique: antithetic variates.*
  Mathematical Proceedings of the Cambridge Philosophical Society 52(3). (Step 41)
* Cameron, A. C., Gelbach, J. B., Miller, D. L. (2008). *Bootstrap-Based Improvements for Inference
  with Clustered Errors.* Review of Economics and Statistics 90(3). (Few clusters; Step 41)
* Hoeffding, W. (1948). *A class of statistics with asymptotically normal distribution.* Annals of
  Mathematical Statistics 19(3). (The orthogonal decomposition; Step 42)
* Sobol', I. M. (1993). *Sensitivity estimates for nonlinear mathematical models.* Mathematical
  Modeling and Computational Experiment 1(4). (Step 42)
* Homma, T., Saltelli, A. (1996). *Importance measures in global sensitivity analysis of nonlinear
  models.* Reliability Engineering and System Safety 52(1). (Total-effect indices; Step 42)
* Shapley, L. S. (1953). *A value for n-person games.* In Contributions to the Theory of Games II,
  Princeton University Press. (Step 42)
* Owen, A. B. (2014). *Sobol' indices and Shapley value.* SIAM/ASA Journal on Uncertainty
  Quantification 2(1). (Step 42)
* Kullback, S., Leibler, R. A. (1951). *On Information and Sufficiency.* Annals of Mathematical
  Statistics 22(1). (Step 43)
* White, H. (1982). *Maximum Likelihood Estimation of Misspecified Models.* Econometrica 50(1).
  (Pseudo-true parameters; Step 43)
* Kuhn, H. W. (1953). *Extensive Games and the Problem of Information.* In Contributions to the
  Theory of Games II, Princeton University Press. (Kuhn's theorem; Step 43)
* Rubin, D. (1984). *Bayesianly Justifiable and Relevant Frequency Calculations for the
  Applied Statistician.* Annals of Statistics 12. (Posterior predictive checks)
* Gelman, A., Meng, X.-L., Stern, H. (1996). *Posterior Predictive Assessment of Model
  Fitness via Realized Discrepancies.* Statistica Sinica 6. (Discrepancies that depend on the
  parameters; Step 30)
* Bayarri, M. J., Berger, J. (2000). *P Values for Composite Null Models.* JASA 95. (Why
  posterior predictive p-values are conservative)
* Hjort, N. L., Dahl, F. A., Steinbakk, G. H. (2006). *Post-Processing Posterior Predictive
  p Values.* JASA 101. (Calibrating them)
* Page, E. S. (1954). *Continuous Inspection Schemes.* Biometrika 41. (CUSUM)
* Gelman, A., Loken, E. (2013). *The garden of forking paths: Why multiple comparisons can be a
  problem, even when there is no "fishing expedition" or "p-hacking".* Working paper,
  Columbia University. (Step 31)
* Lancaster, H. O. (1961). *Significance Tests in Discrete Distributions.* JASA 56. (Mid-p)
* Wilson, E. B. (1927). *Probable Inference, the Law of Succession, and Statistical
  Inference.* JASA 22. (Wilson intervals)
* Kim, J. (2024). *Recording and Describing Poker Hands* (first version: *Poker Hand History
  File Format Specification*). IEEE Conference on Games; arXiv:2312.11753. (PHH, Step 32)
* Brown, N., Sandholm, T. (2019). *Superhuman AI for multiplayer poker.* Science 365. (The
  Pluribus hands)
* Kim, J. (2025). *A Dataset of Poker Hand Histories*, v3. Zenodo,
  doi:10.5281/zenodo.10796885. CC BY 4.0. (The survey in Step 33)
* Moravcik, M. et al. (2017). *DeepStack: Expert-level artificial intelligence in heads-up
  no-limit poker.* Science 356. (Its supplement holds about 45,000 hands against 33 players)
* Vovk, V., Gammerman, A., Shafer, G. (2005). *Algorithmic Learning in a Random World.*
  Springer. (Conformal prediction; the support check's p-values)
* Mahalanobis, P. C. (1936). *On the Generalised Distance in Statistics.* Proceedings of the
  National Institute of Sciences of India 2.
* Schuirmann, D. J. (1987). *A Comparison of the Two One-Sided Tests Procedure and the Power
  Approach for Assessing the Equivalence of Average Bioavailability.* Journal of
  Pharmacokinetics and Biopharmaceutics 15. (Equivalence tests)
* DeMillo, R. A., Lipton, R. J., Sayward, F. G. (1978). *Hints on Test Data Selection: Help for
  the Practicing Programmer.* Computer 11(4). (Mutation testing)
* Miller, B. P., Fredriksen, L., So, B. (1990). *An Empirical Study of the Reliability of UNIX
  Utilities.* CACM 33(12). (Fuzzing)
* Owen, A. B. (2013). *Monte Carlo theory, methods and examples*, chapter 9 (importance
  sampling). statweb.stanford.edu/~owen/mc.

Software design

* Miller, M. S. (2006). *Robust Composition: Towards a Unified Approach to Access Control and
  Concurrency Control.* PhD thesis. (Object capabilities)
* King, A. (2019). *Parse, don't validate.* Blog post.

Performance and numerics

* Amdahl, G. M. (1967). *Validity of the Single Processor Approach to Achieving Large Scale
  Computing Capabilities.* AFIPS Spring Joint Computer Conference. (Amdahl's law; Step 39)
* Michie, D. (1968). *"Memo" Functions and Machine Learning.* Nature 218. (Memoization; Step 39)
* Goldberg, D. (1991). *What Every Computer Scientist Should Know About Floating-Point
  Arithmetic.* ACM Computing Surveys 23(1). (Rounding, ulps, non-associativity; Step 39)
* Dantzig, G. B. (1963). *Linear Programming and Extensions.* Princeton University Press. (The
  simplex method; Step 40)
* Bertsimas, D., Tsitsiklis, J. N. (1997). *Introduction to Linear Optimization.* Athena
  Scientific. (Bases, degeneracy, and what a change of costs keeps: chapters 3 and 5; Step 40)
* Graham, R. L. (1966). *Bounds for Certain Multiprocessing Anomalies.* Bell System Technical
  Journal 45(9). (List scheduling within $2 - 1/m$ of the best; Step 40)
* Graham, R. L. (1969). *Bounds on Multiprocessing Timing Anomalies.* SIAM Journal on Applied
  Mathematics 17(2). (Longest processing time first within $4/3 - 1/(3m)$; Step 40)
* Blumofe, R. D., Leiserson, C. E. (1999). *Scheduling Multithreaded Computations by Work
  Stealing.* Journal of the ACM 46(5). (Step 40)
* Tullsen, D. M., Eggers, S. J., Levy, H. M. (1995). *Simultaneous Multithreading: Maximizing
  On-Chip Parallelism.* ISCA. (Simultaneous multithreading, AMD's SMT and Intel's hyper-threading;
  Step 40)
* pytest-xdist, github.com/pytest-dev/pytest-xdist: the `--dist` modes `load`, `loadfile` and
  `worksteal`. (Step 40)
* Python documentation, *multiprocessing*: "Contexts and start methods" and "Safe importing of
  main module", docs.python.org/3/library/multiprocessing.html. (Spawned workers; Step 40)
* NumPy documentation, *numpy.random.SeedSequence*, numpy.org/doc/stable/reference/random/ (the pool
  of four words), and pull request 16551, github.com/numpy/numpy/pull/16551, "Ensure SeedSequence
  0-padding does not collide with spawn keys", which describes the implicit zero padding of short
  entropy. (Step 41)

---

# Changelog of this log

* 2026-10-20: Log created at the user's request and backfilled for Steps 0-16 (M0-M6 and the
  Phase 1 results). Results quoted are from the final run at commit `98c6dab` (simulation
  fingerprint `7799e1e5ceac`), cross-checked against `phase1_summary.md`.
* 2026-10-21: Phase 2 begun. Added A.6 (game theory), Step 17 (small games and the tree
  compiler) and Step 18 (best responses, exploitability, sequence-form LP).
* 2026-10-21: Step 19 (CFR, CFR+, linear/discounted CFR, external-sampling MCCFR). Results
  from commit `53eb199`, cross-checked against `experiments/results/phase2/cfr/cfr_summary.md`.
* 2026-10-21: Step 20 (Deep CFR). Results from commit `d09ad62`, cross-checked against
  `experiments/results/phase2/deep_cfr/deep_cfr_summary.md`.
* 2026-10-21: Steps 21-23 (Leduc archetypes and likelihood fitting; robust exploitation with
  post-hoc controls; small-game contracts and duplicate dealing). Exploitation results from
  commit `963cee6`, cross-checked against `experiments/results/phase2/exploit/exploit_summary.md`.
* 2026-10-22: Step 24 (one-command Phase 2 runner; reproduction runs of both phases at commit
  `348c4d5`, which matched every committed number exactly).
* 2026-10-23: Step 25 (the results explorer). Its takeaways are computed from the committed
  results; the card-blind per-archetype loss shares were computed for this step and are new.
* 2026-10-23: Phase 3 begun. Step 26 (heads-up push/fold). Results from commit `3aa83c2`,
  cross-checked against `experiments/results/phase3/pushfold/pushfold_summary.md`.
* 2026-10-23: Step 27 (push/fold opponent models). Results from commit `a8ea075`,
  cross-checked against `experiments/results/phase3/exploit/exploit_summary.md`; the
  hand-order overlap figures were computed for this step.
* 2026-10-24: Step 28 (the push/fold provider in real hands). Results from commit `9bdab59`,
  cross-checked against `experiments/results/phase3/bridge/bridge_summary.md`; per-hand
  standard deviations and detectable effects were computed from `bridge.json` for this step.
* 2026-10-24: Step 29 (ranked exploit recommendations with confidence). Results from commit
  `a433bc5`, cross-checked against `experiments/results/phase3/recommend/recommend_summary.md`;
  the ESS range, the 0-hand lower bounds and the timing were computed from `recommend.json` and
  a scratch benchmark for this step.
* 2026-10-24: Step 30 (model-fit checks). Results from commit `c4c15d0`, cross-checked against
  `experiments/results/phase3/fitcheck/fitcheck_summary.md`; the post-hoc stability-only
  rule and the within-population coverage split were computed from `fitcheck.json` for this
  step.
* 2026-10-24: Step 31 (the post-hoc fallback rule confirmed on fresh seeds). Results from
  commit `7f50502`, cross-checked against
  `experiments/results/phase3/confirm/confirm_summary.md`.
* 2026-10-24: Step 32 (importing PHH hand histories). Results from the rerun at commit
  `3c1f25c`, cross-checked against `experiments/results/phase3/import/import_summary.md`; the
  first run's numbers (commit `582ac7c`) come from its saved summary, and the mutation totals
  were computed from both runs' `import.json` for this step.
* 2026-10-25: Step 33 (priors from real players; transfer checks). Results from commit
  `9fce42a`, cross-checked against `experiments/results/phase3/priors/priors_summary.md`; the
  between-player SDs, per-statistic errors and consistency rates were computed from
  `priors.json` for this step. The figures were redrawn from that file after the run with
  clearer labels (no number changed).
* 2026-10-25: Step 34 (a sequence model over raw actions). Results from commit `c4578ea`,
  cross-checked against `experiments/results/phase3/sequence/sequence_summary.md`; the
  development numbers (seeds 98 and 99) come from development logs, not from a committed
  results file.
* 2026-10-26: Step 35 (schema v2). Results from commit `568cde3`, cross-checked against
  `experiments/results/phase3/schema/schema_summary.md`; the pre-registered run was at
  `4c4e373` (same counts), and the Phase 1 re-simulation was compared field by field with
  the committed Phase 1 results.
* 2026-10-28: Step 36 (opponents that adapt to the hero). Results from the pre-registered run at
  commit `10aff6e`, cross-checked against
  `experiments/results/phase3/adaptive/adaptive_summary.md`; the paired comparison of the two
  refitting heroes, the bots' distance from the equilibrium's widths and the example-trajectory
  statistics were computed from `adaptive.json` for this step (post hoc). The figures were
  redrawn from that file after the run (legend placement and colors; no number changed). The
  method text was drafted before the run; the rate formula and the reason the bots drift toward
  equilibrium were corrected while writing it up.
* 2026-10-28: Step 36's run time corrected: the 5.3 hours the results record are wall time, with
  the laptop closed for most of the run (the pilots' estimate was about 1.8 hours).
* 2026-10-29: Step 37 (Phase 3 wrap-up). Reproduction run at `98449f0`, 2026-10-28 14:01 to
  2026-10-29 00:31, compared with `compare_runs`; the three differing steps rerun at their
  recorded commits. Times come from the run's log, with the lid-closed intervals taken from the
  Windows System log (Kernel-Power 506 and 507). Step 36's run time updated to the measured 1.6
  hours.
* 2026-10-29: Step 38 (AIVAT, P2.11). Results from the pre-registered run at commit `695cb26`
  (results commit `0916441`), cross-checked against
  `experiments/results/phase2/aivat/aivat_summary.md`. Computed from `aivat.json` for this step
  (post hoc): the archetype medians, the hero's view with uniform-play values, the exploiting
  hero's win rates, the matchups where duplicate dealing adds variance on top of AIVAT, the
  z-scores of the matchup that ran high, and the exact 10 bb medians compared with Step 28's.
  F1's post hoc check was written after the run (`experiments/phase2_aivat_posthoc.py`, results
  commit `317e8cc`, cross-checked against `aivat_posthoc_summary.md`). The `aivat_leduc`
  figure was redrawn from `aivat.json` after the run with fixed figure code (its two panels each
  inverted the shared y axis, so the rows were upside down); no number changed. The step was
  drafted by a subagent from the results files and checked against them before it was added.
* 2026-10-30: Step 39 (performance: `choose`, the LP memo and sweep, the embedding memo;
  commits `90ae418`, `90f18f9`, `34eea0e` and `eea6815`). Reruns on seed 2026 compared with
  `compare_runs`: adaptive, sequence, exploit, recommend and Phase 2's exploit and aivat match
  the committed results; fitcheck and confirm match Step 37's rerun of the old code. Times come
  from the runs' logs, with the suspended interval from its suspend and resume times and
  standby from the Windows System log (Kernel-Power 506 and 507). Computed for this step (post
  hoc): the Amdahl bounds, the fit-by-fit training times of the committed, wrap-up and new
  sequence runs, the wrap-up run's fast third replicate, and the same-session quick-configuration
  comparison of the sequence step. C.1's test count updated (457). The step was drafted by a
  subagent from the diffs, run logs and results files and checked against them before it was
  added.
* 2026-11-01: Step 40 (the HiGHS hot start, signed off by the owner as changing floating-point
  results; two sequence replicates at once, default one; the test suite in four processes; transfer
  reusing part A's embeddings; commits `80f2152`, `03b3b4f`, `3fa49aa`, `505e16a` and `9f877e5`).
  Reruns on seed 2026 at `3fa49aa` compared with `compare_runs`: sequence (one and two jobs),
  exploit and Phase 2's exploit and aivat match the committed results; adaptive and recommend differ
  from them, and fitcheck and confirm from Step 37's rerun of the old code, in floats only. Times
  come from the runs' logs, with standby from the Windows System log (Kernel-Power 506 and 507,
  reason Idle Timeout) and a memory sampler's 10-second log. Computed for this step (post hoc): the
  classification of every difference by kind and size, recommend's differences by $p$, the
  training-only times and per-replicate slowdowns of the two sequence runs, the whole-step estimates
  with the simulation added to both, the memory statistics, the Amdahl check of the quick
  comparison, and the suite's makespan bound. C.1 (471 tests), C.2, C.3, A.5 and Parts D, E and F
  updated; Step 39's first "What was left" item marked fixed, and its memo bound and sweep test
  annotated. The step was drafted by a subagent from the diffs, run logs and results files and
  checked against them before it was added.
* 2026-11-04: Step 41 (duplicate dealing for n players, P2.11b; commits `8a5f983`, `0fb6e53`,
  `00f8f7e`, `c7f0029`, `8cdd759` and `4150aac`). Results from the pre-registered run at `0fb6e53`
  (results commit `00f8f7e`), cross-checked against
  `experiments/results/phase2/duplicate/duplicate_summary.md`. The bridge step, rerun on seed 2026
  with the new code, matches the committed results. Times come from the results file and the run's
  logs, with standby from the Windows System log (Kernel-Power 506 and 507, reason Idle Timeout) and
  a memory sampler's 10-second log. Computed for this step (post hoc): the share of perfect
  cancellation by lineup kind, the coverage of the percentile bootstrap on the real run's lineups
  and in normal theory, intervals from resampling players, heads-up lineups by archetype pair, the
  rank correlation of ratio and SD, the skew of the offsets, the spread within lineups of copies,
  the final predictions on the pilot's and the quick run's outputs, the development columns of the
  predictions table (the saved development outputs hold placeholder predictions) and, from the
  development seeds' per-hand arrays (not committed), the variance between positions and the
  kurtosis of per-deal means. Most of those come from `experiments/phase2_duplicate_posthoc.py`,
  whose results are committed. C.1 (517 tests), C.2, C.3 and Parts D, E and F updated. The step was
  drafted by a subagent from the diffs, run logs and results files and checked against them before
  it was added.
* 2026-11-05: Step 42 (P2.11c: how six copies' shortfall divides between the bots' randomness),
  from the pre-registered plan in `experiments/phase2_randomness.py` and its run on seed 2026. C.1
  (543 tests), Parts D, E and F updated; Step 41's open item points here. Numbers checked by script
  against `randomness.json`.
* 2026-11-09: Step 43 (P2.12: exploitation when the opponent model is wrong), from the
  pre-registered plan in `experiments/phase2_misspec.py` (`8601332`) and its run on seed 2026
  (results `e2b8c12`, explorer `9d94750`). C.1 (604 tests), Parts D, E and F updated. Numbers
  checked by script against `misspec.json`.
* 2026-11-09: Step 44 (playing against the model: a local web page, push/fold and hold'em, the
  exploit slot), a tool rather than a result. C.1 (624 tests), Parts D and E updated. The P3.9
  numbers it quotes were checked against Step 36.
