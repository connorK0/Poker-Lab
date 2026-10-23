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
forced bet from everyone) or *straddles* (an optional third blind); the schema supports both.

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
| pytest, ruff | tests, lint and formatting |
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

**What was built.** `git init` on a branch (`phase1`), [`PROGRESS.md`](../PROGRESS.md) created
*before any code*, [`pyproject.toml`](../pyproject.toml) with pytest and ruff settings, and a
package skeleton.

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
  how many worker processes ran (5 minutes on 16 cores).
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
| Reproducibility | same seed gives identical JSON; 1 vs 2 workers give identical arrays | catches hidden global state |
| End-to-end | simulate -> advise -> profile -> evaluate -> round-trip | proves the seams fit |

Statistical tests use fixed seeds and tolerances wide enough to be stable and tight enough to
catch real errors (for example coverage within +/- 1.5 points of 95% over 5,000 simulated
players). The suite has 242 tests and runs in about a minute (the Deep CFR tests skip
without PyTorch). The explorer's browser behavior has its own check
(`python -m experiments.explorer.browser_check`, Step 25), since it needs Chrome or Edge.

## C.2 Reproducibility and provenance

* One master seed; every other seed is derived with `numpy.random.SeedSequence`.
* Each table owns a seeded `random.Random` that deals the cards and drives every agent's
  randomness, so a table replays exactly whatever the agents do internally.
* Results record seed, config, git commit, dirty flag (captured at the start of the run),
  package versions and the simulation fingerprint.
* Simulations are cached by config plus fingerprint (`experiments/cache/`, git-ignored),
  so analyses can be re-run in minutes and stale data cannot be reused.

## C.3 Platform lessons (Windows, PowerShell, OneDrive)

* *Spawned workers re-import modules:* keep package imports lazy (PEP 562) and heavy imports
  inside functions; cap worker counts.
* *PowerShell quoting:* `"...$SD..."` expands `$SD` as a variable (a commit message lost its
  `"W$SD"`); use single-quoted here-strings (`@' ... '@`) for commit messages.
* *Encodings:* Windows PowerShell 5.1's `Set-Content -Encoding utf8` writes a byte-order mark;
  rewrite files with Python when the exact bytes matter.
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

## C.4 Figures

Every figure is rendered in a light and a dark variant (the README picks one with
`<picture>`). Colors come from a palette checked by a validator for colorblind separation
between adjacent series. Lines are 2 px with ringed markers and gridlines are hairlines.
Interval bands are clipped to the metric's natural range for display only. Every chart has a
numeric table next to it, so no number depends on reading a color.

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

---

# Part E: glossary

| Term | Meaning |
|---|---|
| All-in | betting every remaining chip |
| bb, bb/100 | big blind; win rate in big blinds per 100 hands |
| Best response | a strategy that maximizes payoff against a fixed opponent |
| Button (BTN) | dealer position; acts last after the flop |
| C-bet | continuation bet: the preflop raiser bets the flop |
| Calibration | predicted probabilities match observed frequencies |
| Capability | an unforgeable reference whose possession grants a permission |
| CFR | counterfactual regret minimization: one regret matcher per information set |
| Cluster bootstrap | resampling whole clusters (players), not rows |
| Conjugate prior | a prior whose posterior has the same family (beta-binomial) |
| Counterfactual value | an action's value at an information set, weighted by the probability that chance and the opponent reach it |
| Credible interval | interval containing the parameter with a given posterior probability |
| Data-biased response | robust response that trusts the model per information set by data count |
| Deep CFR | CFR with neural networks in place of the regret and average-strategy tables |
| Duplicate dealing | playing each deal twice with seats swapped so card luck cancels |
| ECE | expected calibration error |
| Empirical Bayes | estimating the prior from the population itself |
| Equity | probability of winning (ties count half) |
| Exploitability | how much a best response wins against a strategy (NashConv / 2) |
| Game value | what the first player gets when both play an equilibrium (minimax theorem) |
| Hand class | one of the 169 strategically distinct starting hands (pairs, suited, offsuit), e.g. AKs |
| Headless browser | a browser run without a window, driven from the command line (screenshots, DOM dumps, scripted checks) |
| HUD | heads-up display of opponent statistics |
| Information set | the histories a player cannot tell apart when acting |
| Log-loss | mean negative log-probability of the true class |
| mbb/hand | milli-big-blinds per hand |
| MCCFR | Monte Carlo CFR: CFR on sampled parts of the tree |
| Muck | discard a hand without showing it |
| Nash equilibrium | a profile where each strategy is a best response to the other |
| NashConv | sum of both players' best-response gains; 0 at equilibrium |
| Oracle | a model given the true answer's inputs, to measure the ceiling |
| Perfect recall | players never forget their own actions or observations |
| Permutation control | retrain on shuffled labels; above-chance results mean leakage |
| Push/fold | a short-stack strategy space: go all-in or fold, and call or fold facing an all-in |
| Range | the set of hands a player plays in a spot |
| Realization plan | sequence-form strategy: probability of each own action sequence |
| Regret matching | play actions in proportion to their positive cumulative regret |
| Reservoir sampling | keeping a uniform random sample of a stream in fixed memory |
| Restricted Nash response | best response to a model, hedged against an adversary with probability 1-p |
| Sequence | an (information set, action) pair |
| Shrinkage | pulling noisy estimates toward a population mean |
| Side pot | pot that an all-in player for less is not eligible for |
| Stratified sampling | splitting a Monte Carlo estimate into strata (here: combo pairs) sampled equally, removing the variance between strata |
| Street | betting round: preflop, flop, turn, river |
| Temperature scaling | dividing log-probabilities by $T$ to fix confidence |
| Uncalled bet | the part of a bet nobody matched, returned to the bettor |
| VPIP / PFR / WTSD / W\$SD | see [A.3](#a3-hud-statistics-and-why-denominators-matter) |

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
* Kuhn, H. W. (1950). *A simplified two-person poker.* Contributions to the Theory of Games 1.
* Lanctot, M. et al. (2019). *OpenSpiel: A Framework for Reinforcement Learning in Games.*
  arXiv:1908.09453. Source of the uniform-policy NashConv used as a check.

Equilibrium, exploitation, evaluation (Phase 2 and 5)

* Koller, D., Megiddo, N., von Stengel, B. (1996). *Efficient Computation of Equilibria for
  Extensive Two-Person Games.* Games and Economic Behavior 14. (The sequence form)
* Huangfu, Q., Hall, J. A. J. (2018). *Parallelizing the dual revised simplex method.*
  Mathematical Programming Computation 10. (HiGHS, SciPy's LP solver)
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
  Reduction Technique for Agent Evaluation in Imperfect Information Games.* AAAI.

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
* Naeini, M. P., Cooper, G., Hauskrecht, M. (2015). *Obtaining Well Calibrated Probabilities
  Using Bayesian Binning.* AAAI. (ECE)
* Gneiting, T., Raftery, A. (2007). *Strictly Proper Scoring Rules, Prediction, and
  Estimation.* JASA.
* Efron, B., Tibshirani, R. (1993). *An Introduction to the Bootstrap.* Chapman & Hall.
* Ojala, M., Garriga, G. (2010). *Permutation Tests for Studying Classifier Performance.*
  JMLR.
* Kaufman, S., Rosset, S., Perlich, C. (2012). *Leakage in Data Mining: Formulation,
  Detection, and Avoidance.* ACM TKDD.

Software design

* Miller, M. S. (2006). *Robust Composition: Towards a Unified Approach to Access Control and
  Concurrency Control.* PhD thesis. (Object capabilities)
* King, A. (2019). *Parse, don't validate.* Blog post.

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
