"""Exact equilibria by linear programming in the sequence form (Koller, Megiddo, von Stengel 1996).

A behavior strategy is awkward for optimization: the probability of reaching a leaf is a
*product* of the player's action probabilities. The sequence form uses the *realization plan*
instead: ``x[s]`` is the probability that the player's own actions along the path include
sequence ``s`` (an information set and an action there). Realization plans are linear:

    x[empty] = 1,  and for every information set I:  sum_a x[I, a] = x[parent sequence of I],

written ``E x = e`` (player 0) and ``F y = f`` (player 1), with ``x, y >= 0``. The expected payoff
is bilinear, ``x^T A y``, where ``A[s0, s1]`` sums ``chance probability * payoff`` over the
leaves whose last sequences are ``s0`` and ``s1``.

Player 0 maximizes ``min_y x^T A y``. For fixed ``x`` the inner minimum is an LP; replacing it
by its dual (``max_q f^T q`` subject to ``F^T q <= A^T x``) gives one LP for player 0's
equilibrium strategy and the game value:

    maximize f^T q   subject to   F^T q - A^T x <= 0,   E x = e,   x >= 0,   q free.

Player 1's LP is symmetric. Behavior strategies are recovered as ``x[I, a] / x[parent of I]``
(uniform where the information set is unreachable). Sizes stay small: Leduc has 1,092 sequences
per player, and HiGHS solves each LP in well under a second.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import sparse
from scipy.optimize import linprog

from poker_lab.games.policy import TabularPolicy
from poker_lab.games.tree import TERMINAL, GameTree


@dataclass(frozen=True)
class SequenceForm:
    """Sequence-form matrices; local index 0 of each player is the empty sequence."""

    payoff: sparse.csr_matrix  # A: (n0, n1) player 0's payoff
    constraints: tuple[sparse.csr_matrix, sparse.csr_matrix]  # E, F
    local_index: tuple[np.ndarray, np.ndarray]  # global sequence -> local index (or -1)
    global_index: tuple[np.ndarray, np.ndarray]  # local index (>= 1) -> global sequence


def sequence_form(tree: GameTree) -> SequenceForm:
    local, glob = [], []
    for p in (0, 1):
        mine = np.flatnonzero(tree.sequences_of(p))
        idx = np.full(tree.num_sequences, -1, dtype=np.int64)
        idx[mine] = np.arange(1, len(mine) + 1)
        local.append(idx)
        glob.append(mine)

    def loc(p: int, seq: np.ndarray) -> np.ndarray:
        return np.where(seq < 0, 0, local[p][np.maximum(seq, 0)])

    leaves = np.flatnonzero(tree.player == TERMINAL)
    chance_reach = tree.reach(tree.chance_prob)
    a = sparse.coo_matrix(
        (
            chance_reach[leaves] * tree.utility[leaves],
            (loc(0, tree.own_seq[0, leaves]), loc(1, tree.own_seq[1, leaves])),
        ),
        shape=(len(glob[0]) + 1, len(glob[1]) + 1),
    ).tocsr()  # duplicate (row, col) entries are summed

    mats = []
    for p in (0, 1):
        infosets = np.flatnonzero(tree.infoset_player == p)
        rows, cols, vals = [0], [0], [1.0]
        for r, i in enumerate(infosets, start=1):
            rows.append(r)
            cols.append(int(loc(p, np.array([tree.infoset_parent_seq[i]]))[0]))
            vals.append(-1.0)
            for s in range(tree.seq_offset[i], tree.seq_offset[i + 1]):
                rows.append(r)
                cols.append(int(local[p][s]))
                vals.append(1.0)
        mats.append(
            sparse.coo_matrix((vals, (rows, cols)), shape=(len(infosets) + 1, len(glob[p]) + 1))
        )
    return SequenceForm(a, (mats[0].tocsr(), mats[1].tocsr()), tuple(local), tuple(glob))


@dataclass(frozen=True)
class LPSolution:
    value: float  # game value for player 0
    policy: TabularPolicy  # an exact equilibrium (to solver tolerance)
    realization: tuple[np.ndarray, np.ndarray]  # x, y over local sequence indices


def _solve_side(sf: SequenceForm, player: int) -> tuple[float, np.ndarray]:
    """Equilibrium realization plan of ``player`` and the game value (player 0's payoff)."""
    a = sf.payoff if player == 0 else -sf.payoff.T  # the solver maximizes its own payoff
    own, other = sf.constraints[player], sf.constraints[1 - player]
    n_own, n_q = own.shape[1], other.shape[0]
    rhs_own = np.zeros(own.shape[0])
    rhs_own[0] = 1.0
    # variables: [x (n_own), q (n_q)]; maximize q[0]  <=>  minimize -q[0]
    c = np.zeros(n_own + n_q)
    c[n_own] = -1.0
    a_ub = sparse.hstack([-a.T, other.T]).tocsr()
    a_eq = sparse.hstack([own, sparse.csr_matrix((own.shape[0], n_q))]).tocsr()
    bounds = [(0, None)] * n_own + [(None, None)] * n_q
    res = linprog(
        c,
        A_ub=a_ub,
        b_ub=np.zeros(a_ub.shape[0]),
        A_eq=a_eq,
        b_eq=rhs_own,
        bounds=bounds,
        method="highs",
    )
    if res.status != 0:
        raise RuntimeError(f"sequence-form LP failed: {res.message}")
    value = -res.fun
    return (value if player == 0 else -value), res.x[:n_own]


def solve_lp(tree: GameTree) -> LPSolution:
    """Exact equilibrium of a two-player zero-sum game via two sequence-form LPs."""
    sf = sequence_form(tree)
    v0, x = _solve_side(sf, 0)
    v1, y = _solve_side(sf, 1)
    if abs(v0 - v1) > 1e-6 * max(1.0, abs(v0)):
        raise RuntimeError(f"LP values disagree: {v0} vs {v1}")
    weights = np.zeros(tree.num_sequences)
    for p, plan in ((0, x), (1, y)):
        weights[sf.global_index[p]] = np.clip(plan[1:], 0.0, None)
    return LPSolution(0.5 * (v0 + v1), TabularPolicy.from_weights(tree, weights), (x, y))
