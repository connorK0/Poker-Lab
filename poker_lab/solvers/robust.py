"""Robust counter-strategies by linear programming: restricted Nash and data-biased responses.

The hero (player ``h``) faces an opponent whose strategy has been *modeled* as ``sigma_fix``.
Notation follows ``poker_lab.solvers.lp``: realization plans ``x`` (hero, ``E x = e``) and ``y``
(opponent, ``F y = f``), hero payoff ``x^T A y``, and ``y_fix`` the model's realization plan.

* **Best response to the model** maximizes ``x^T A y_fix``. It wins the most if the model is
  right and can lose badly if it is wrong.
* **Restricted Nash response** (Johanson, Zinkevich, Bowling 2008). With probability ``p`` the
  opponent must play ``sigma_fix``; otherwise it plays freely (adversarially), and the hero does
  not know which. The hero solves ``max_x p x^T A y_fix + (1 - p) min_y x^T A y``. Dualizing the
  inner minimum exactly as in the equilibrium LP gives

      max_{x,q}  p (A y_fix)^T x + (1 - p) f^T q
      s.t.  F^T q - A^T x <= 0,   E x = e,   x >= 0.

  ``p = 0`` gives an equilibrium strategy, ``p = 1`` a best response to the model. Sweeping
  ``p`` traces the best achievable trade-off between exploiting the model and being
  exploitable.
* **Data-biased response** (Johanson and Bowling 2009). Trust the model separately at each of
  the opponent's information sets: the opponent must play ``sigma_fix(I)`` with probability
  ``c_I`` and chooses the rest freely, ``sigma(I) = c_I sigma_fix(I) + (1 - c_I) sigma_free(I)``.
  In the sequence form this is the linear constraint ``y[I, a] >= c_I sigma_fix(I, a)
  y[parent(I)]``, written ``G y >= 0``. The inner minimum over this smaller polytope dualizes to

      max_{x,q,r}  f^T q   s.t.  F^T q + G^T r - A^T x <= 0,   E x = e,   x >= 0,   r >= 0.

  ``c = 0`` everywhere gives an equilibrium; ``c = 1`` everywhere a best response to the model.
  Typically ``c_I`` grows with the number of observations of the opponent at ``I``.

Both are exact (HiGHS) and fast for Leduc (well under a second per solve).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import sparse
from scipy.optimize import linprog

from poker_lab.games.evaluate import best_response
from poker_lab.games.policy import TabularPolicy
from poker_lab.games.tree import GameTree
from poker_lab.solvers.lp import SequenceForm, sequence_form


@dataclass(frozen=True)
class HeroForm:
    """Sequence-form matrices seen from the hero's side."""

    payoff: sparse.csr_matrix  # hero payoff, (hero sequences, opponent sequences)
    own: sparse.csr_matrix  # E (hero constraints)
    opponent: sparse.csr_matrix  # F (opponent constraints)
    hero_global: np.ndarray  # local hero index (>= 1) -> global sequence
    opp_local: np.ndarray  # global sequence -> local opponent index (or -1)


def hero_form(sf: SequenceForm, hero: int) -> HeroForm:
    a = sf.payoff if hero == 0 else -sf.payoff.T
    return HeroForm(
        payoff=a.tocsr(),
        own=sf.constraints[hero],
        opponent=sf.constraints[1 - hero],
        hero_global=sf.global_index[hero],
        opp_local=sf.local_index[1 - hero],
    )


def realization_plan(tree: GameTree, policy: TabularPolicy, player: int) -> np.ndarray:
    """``player``'s realization plan of ``policy``: own reach of each sequence (global index)."""
    own = tree.edge_player == player
    reach = tree.reach(np.where(own, tree.edge_probabilities(policy.probs), 1.0))
    plan = reach[tree.infoset_node][tree.seq_infoset] * policy.probs
    return np.where(tree.sequences_of(player), plan, 0.0)


def _local_plan(form: HeroForm, tree: GameTree, global_plan: np.ndarray) -> np.ndarray:
    n_opp = form.opponent.shape[1]
    y = np.zeros(n_opp)
    y[0] = 1.0
    mine = form.opp_local >= 0
    y[form.opp_local[mine]] = global_plan[mine]
    return y


def _confidence_rows(tree: GameTree, form: HeroForm, model: TabularPolicy, hero: int, c):
    """Rows of G: y[I, a] - c_I * sigma_fix(I, a) * y[parent(I)] >= 0 for c_I * sigma > 0."""
    opp = 1 - hero
    rows, cols, vals = [], [], []
    r = 0
    for i in np.flatnonzero(tree.infoset_player == opp):
        if c[i] <= 0:
            continue
        parent = tree.infoset_parent_seq[i]
        parent_col = 0 if parent < 0 else int(form.opp_local[parent])
        for s in range(tree.seq_offset[i], tree.seq_offset[i + 1]):
            coef = c[i] * model.probs[s]
            if coef <= 0:
                continue
            rows += [r, r]
            cols += [int(form.opp_local[s]), parent_col]
            vals += [1.0, -coef]
            r += 1
    return sparse.coo_matrix((vals, (rows, cols)), shape=(r, form.opponent.shape[1])).tocsr()


def _solve(form: HeroForm, objective_x: np.ndarray, weight_q: float, g=None) -> np.ndarray:
    """Maximize objective_x . x + weight_q * q[0] over the (possibly restricted) LP."""
    n_x, n_q = form.own.shape[1], form.opponent.shape[0]
    n_r = 0 if g is None else g.shape[0]
    c = np.concatenate([-objective_x, np.zeros(n_q), np.zeros(n_r)])
    c[n_x] -= weight_q
    blocks = [-form.payoff.T, form.opponent.T]
    if n_r:
        blocks.append(g.T)
    a_ub = sparse.hstack(blocks).tocsr()
    a_eq = sparse.hstack([form.own, sparse.csr_matrix((form.own.shape[0], n_q + n_r))]).tocsr()
    b_eq = np.zeros(form.own.shape[0])
    b_eq[0] = 1.0
    bounds = [(0, None)] * n_x + [(None, None)] * n_q + [(0, None)] * n_r
    b_ub = np.zeros(a_ub.shape[0])
    res = linprog(c, A_ub=a_ub, b_ub=b_ub, A_eq=a_eq, b_eq=b_eq, bounds=bounds, method="highs")
    if res.status != 0:
        raise RuntimeError(f"robust-response LP failed: {res.message}")
    return res.x[:n_x]


def _to_policy(tree: GameTree, form: HeroForm, x: np.ndarray, model: TabularPolicy, hero: int):
    weights = np.where(tree.sequences_of(hero), 0.0, model.probs)
    weights[form.hero_global] = np.clip(x[1:], 0.0, None)
    # opponent part: keep the model's probabilities exactly (normalize leaves them unchanged)
    return TabularPolicy.from_weights(tree, weights)


class RobustResponder:
    """Restricted Nash / data-biased responses against a model, for either hero seat."""

    def __init__(self, tree: GameTree) -> None:
        self.tree = tree
        sf = sequence_form(tree)
        self.forms = (hero_form(sf, 0), hero_form(sf, 1))

    def restricted_nash(self, model: TabularPolicy, hero: int, p: float) -> TabularPolicy:
        """Hero's ``p``-restricted Nash response (other player's part: the model)."""
        if not 0.0 <= p <= 1.0:
            raise ValueError("p must be in [0, 1]")
        if p == 1.0:
            return best_response(model, hero).policy
        form = self.forms[hero]
        y_fix = _local_plan(form, self.tree, realization_plan(self.tree, model, 1 - hero))
        x = _solve(form, p * (form.payoff @ y_fix), 1.0 - p)
        return _to_policy(self.tree, form, x, model, hero)

    def data_biased(self, model: TabularPolicy, hero: int, confidence) -> TabularPolicy:
        """Hero's data-biased response; ``confidence[I]`` in [0, 1] per information set."""
        c = np.asarray(confidence, dtype=float)
        if c.shape != (self.tree.num_infosets,) or c.min() < 0 or c.max() > 1:
            raise ValueError("confidence must be in [0, 1], one value per information set")
        form = self.forms[hero]
        g = _confidence_rows(self.tree, form, model, hero, c)
        x = _solve(form, np.zeros(form.own.shape[1]), 1.0, g)
        return _to_policy(self.tree, form, x, model, hero)

    def both_seats(self, method: str, model: TabularPolicy, **kwargs) -> TabularPolicy:
        """A full hero strategy: the response as player 0 and as player 1, combined."""
        fn = {"rnr": self.restricted_nash, "dbr": self.data_biased}[method]
        seat0 = fn(model, 0, **kwargs)
        seat1 = fn(model, 1, **kwargs)
        return seat0.combine(seat1, 0)
