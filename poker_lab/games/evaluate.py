"""Exact expected values, best responses and exploitability on a compiled game tree.

Notation (two-player zero-sum; ``u`` is player 0's payoff, player 1 receives ``-u``):

* ``value(sigma)``: player 0's expected payoff when both players follow the profile ``sigma``.
* ``BR_0(sigma_1)``: the most player 0 can expect against player 1's part of ``sigma``;
  ``BR_1(sigma_0)`` likewise for player 1 (in player 1's own payoff).
* ``NashConv(sigma) = BR_0(sigma_1) + BR_1(sigma_0)``. With game value ``v*`` this is
  ``(BR_0 - v*) + (BR_1 + v*)``: the total each player could gain by deviating. It is >= 0, and
  0 exactly at a Nash equilibrium.
* ``exploitability = NashConv / 2``: what a best-responding opponent wins per hand against
  ``sigma``, averaged over the two seats, above the game value.

The best response is computed by backward induction over the responder's information sets.
Fix the other player's strategy and chance; let ``pi_-i(h)`` be the probability that they (not
the responder) play to node ``h``. At information set ``I`` the responder picks the action with
the largest *counterfactual value* ``q(I, a) = sum_{h in I} pi_-i(h) * v(h.a)``, where ``v`` is
already computed below ``I`` with the responder's best choices. Every node of ``I`` then takes
that action. Because nodes are processed one depth level at a time (deepest first) and every
information set sits on a single level, all ``v(h.a)`` are final when ``I`` is processed.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from poker_lab.games.policy import TabularPolicy
from poker_lab.games.tree import GameTree

_TIE_TOL = 1e-12


def expected_value(policy: TabularPolicy) -> float:
    """Player 0's expected payoff under the profile ``policy``."""
    tree = policy.tree
    return float(tree.backward(tree.edge_probabilities(policy.probs))[0])


def head_to_head(hero: TabularPolicy, villain: TabularPolicy) -> float:
    """Hero's expected payoff per hand against ``villain``, averaged over both seats."""
    as_p0 = expected_value(hero.combine(villain, 0))
    as_p1 = -expected_value(hero.combine(villain, 1))
    return 0.5 * (as_p0 + as_p1)


@dataclass(frozen=True)
class BestResponse:
    player: int
    value: float  # the responder's expected payoff
    policy: TabularPolicy  # pure best response at `player`'s information sets, input elsewhere


def best_response(policy: TabularPolicy, player: int) -> BestResponse:
    """Exact best response of ``player`` to the other player's part of ``policy``."""
    if player not in (0, 1):
        raise ValueError("player must be 0 or 1")
    tree: GameTree = policy.tree
    sign = 1.0 if player == 0 else -1.0
    edge = tree.edge_probabilities(policy.probs)
    responder_edge = tree.edge_player == player
    reach_others = tree.reach(np.where(responder_edge, 1.0, edge))
    v = sign * tree.utility
    probs = policy.probs.copy()
    n_seq = tree.num_sequences
    starts = tree.seq_offset[:-1]

    for d in range(tree.max_depth, 0, -1):
        lv, up = tree.level(d), tree.level(d - 1)
        par = tree.parent[lv] - up.start
        mine = responder_edge[lv]
        # Chance and the other player: expectation over their edge probabilities.
        others = np.where(mine, 0.0, edge[lv] * v[lv])
        v[up] += np.bincount(par, weights=others, minlength=up.stop - up.start)
        if not mine.any():
            continue
        # Responder: counterfactual value of each action, then the best action per infoset.
        kids = np.flatnonzero(mine) + lv.start
        seqs = tree.edge_seq[kids]
        q = np.bincount(seqs, weights=reach_others[tree.parent[kids]] * v[kids], minlength=n_seq)
        best_q = np.maximum.reduceat(q, starts)[tree.seq_infoset]
        candidates = np.where(q >= best_q - _TIE_TOL, np.arange(n_seq), n_seq)
        best_seq = np.minimum.reduceat(candidates, starts)  # first best action per infoset
        chosen = seqs == best_seq[tree.seq_infoset[seqs]]
        v[up] += np.bincount(
            par[mine][chosen], weights=v[kids[chosen]], minlength=up.stop - up.start
        )
        infosets_here = np.unique(tree.seq_infoset[seqs])
        here = np.isin(tree.seq_infoset, infosets_here)
        probs[here] = 0.0
        probs[best_seq[infosets_here]] = 1.0

    return BestResponse(player, float(v[0]), TabularPolicy(tree, probs))


@dataclass(frozen=True)
class ExploitabilityReport:
    value: float  # player 0's expected payoff under the profile
    br_values: tuple[float, float]  # (BR_0(sigma_1), BR_1(sigma_0)), each in the responder's payoff

    @property
    def nash_conv(self) -> float:
        return self.br_values[0] + self.br_values[1]

    @property
    def exploitability(self) -> float:
        return 0.5 * self.nash_conv

    @property
    def gains(self) -> tuple[float, float]:
        """How much each player gains by switching to a best response."""
        return self.br_values[0] - self.value, self.br_values[1] + self.value


def exploitability_report(policy: TabularPolicy) -> ExploitabilityReport:
    return ExploitabilityReport(
        value=expected_value(policy),
        br_values=(best_response(policy, 0).value, best_response(policy, 1).value),
    )


def exploitability(policy: TabularPolicy) -> float:
    """NashConv / 2 in the game's payoff units (chips per hand)."""
    return exploitability_report(policy).exploitability
