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

Both are exact (HiGHS) and fast for Leduc (well under a second per solve). The LPs reach
HiGHS through SciPy's bindings (``scipy.optimize._highspy._core``, a private module) rather
than ``linprog``, built exactly as ``linprog(method="highs")`` builds them: the same rows in the
same order, the matrix stored column by column, the same bounds and the options ``linprog``
passes. HiGHS is deterministic, so an LP solved from scratch gives ``linprog``'s bytes. A
solution is accepted on the same terms as ``linprog``'s: an optimal status from HiGHS, then
``linprog``'s own check (no NaN; bounds and constraints met within about 3.2e-4); where
``linprog`` would report a failure, an error is raised, as before. Only the costs of a
restricted Nash LP depend on the model and ``p``; the rest is built once per form.

A sweep (``restricted_nash_sweep``) poses, for each seat, one LP per ``p`` below 1, with the
same constraints and costs that change with ``p``. A new HiGHS instance per seat solves the
first from scratch, then, for each next ``p``, changes the costs and runs again. Holding an
optimal basis for the same constraints, HiGHS skips presolve and restarts the simplex method
from that basis (a *hot start*): it is still feasible, and a few pivots usually reach the new
optimum. The path differs, so the rounding does: the later ``p`` usually agree with the LPs
solved from scratch to about 1e-11, not bit for bit. And the simplex method stops at the first
basis it finds optimal within HiGHS's tolerances (1e-7 by default), so where two vertices both
pass, the two paths can stop at different ones: strategies that differ by more than rounding,
with objective values within that tolerance. Should a hot start fail, that LP is solved from
scratch on a new instance and the next ones are hot-started from there. A sweep is
deterministic all the same: it makes its own instances and the same calls in the same order, so
its result depends only on its inputs (the form, ``A y_fix`` and the ``p`` in order), never on
what the process solved before, which in a pool of workers depends on how tasks were shared.
Its first LP is solved from scratch and equals the single response bit for bit; single
responses (``restricted_nash``, ``data_biased``) give ``linprog``'s bytes as before.

Each process keeps its last ``SOLVED_SIZE`` LP solutions, keyed on their exact inputs: a single
LP on the form, objective, weight and ``G``; a sweep on the form, ``A y_fix`` and the ``p`` in
order. Refits and tests often pose the same LP again, and HiGHS returns the same solution for
the same input, so a repeat is answered from memory without changing any result. The two kinds
of key never meet: a single ``p`` never gets a hot-started solution, and a sweep never gets a
single LP's solution for a later ``p``. If SciPy's bindings are missing or fail a test LP at
first use (after a SciPy upgrade, say), every LP falls back to ``linprog``, solved from
scratch, with one warning per process; a failure after that raises an error. A test checks
that the bindings are in use.
"""

from __future__ import annotations

import hashlib
import importlib
import warnings
from collections import OrderedDict
from dataclasses import dataclass
from typing import NamedTuple

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
    key: bytes  # digest of payoff, own and opponent (the LP cache's key)


def _matrix_key(m) -> bytes:
    m = m.tocsr()
    h = hashlib.sha1(repr((m.shape, m.dtype.str, m.indices.dtype.str)).encode())
    for part in (m.data, m.indices, m.indptr):
        h.update(np.ascontiguousarray(part).tobytes())
    return h.digest()


def hero_form(sf: SequenceForm, hero: int) -> HeroForm:
    a = (sf.payoff if hero == 0 else -sf.payoff.T).tocsr()
    own, opponent = sf.constraints[hero], sf.constraints[1 - hero]
    return HeroForm(
        payoff=a,
        own=own,
        opponent=opponent,
        hero_global=sf.global_index[hero],
        opp_local=sf.local_index[1 - hero],
        key=b"".join(_matrix_key(m) for m in (a, own, opponent)),
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


SOLVED_SIZE = 2048  # LP solutions kept per process
_SOLVED: OrderedDict[tuple, tuple[np.ndarray, ...]] = OrderedDict()


def _recall(key) -> tuple[np.ndarray, ...] | None:
    xs = _SOLVED.get(key)
    if xs is not None:
        _SOLVED.move_to_end(key)
    return xs


def _remember(key, xs: tuple[np.ndarray, ...]) -> tuple[np.ndarray, ...]:
    """Keep ``xs`` (made read-only) under ``key`` as the newest entry, then drop the entries
    used longest ago until at most ``SOLVED_SIZE`` solutions are kept (a sweep's entry holds one
    per ``p``; the newest entry is kept even if it alone holds more)."""
    for x in xs:
        x.flags.writeable = False
    _SOLVED.pop(key, None)
    _SOLVED[key] = xs
    kept = sum(map(len, _SOLVED.values()))
    while kept > SOLVED_SIZE and len(_SOLVED) > 1:
        kept -= len(_SOLVED.popitem(last=False)[1])
    return xs


def _solve(
    form: HeroForm, objective_x: np.ndarray, weight_q: float, g=None, name: str = "LP"
) -> np.ndarray:
    """Maximize objective_x . x + weight_q * q[0] over the (possibly restricted) LP, solved
    from scratch: the hero part of the solution, read-only, from memory if the same LP was
    solved recently. ``name`` is for error messages."""
    key = (
        form.key,
        np.ascontiguousarray(objective_x, dtype=float).tobytes(),
        np.float64(weight_q).tobytes(),
        None if g is None else _matrix_key(g),
    )
    xs = _recall(key)
    if xs is None:
        xs = _remember(key, _solve_in_order(form, [(objective_x, weight_q, name)], g))
    return xs[0]


def _solve_sweep(form: HeroForm, payoff_y: np.ndarray, ps) -> tuple[np.ndarray, ...]:
    """The hero parts of the ``p``-restricted Nash LPs (``A y_fix = payoff_y``) for each ``p``
    in ``ps`` (all below 1), in that order: the first solved from scratch, each later one
    hot-started from the one before. Read-only, from memory if the same sweep (the same ``ps``
    in the same order) was solved recently. A sweep of one LP is that LP solved from scratch,
    so it is ``_solve``'s, memory included. With several, ``ps`` and ``payoff_y`` are taken as
    float64, the values the key holds, so the key determines the LPs exactly."""
    ps = tuple(ps)
    if len(ps) == 1:
        return (_solve(form, ps[0] * payoff_y, 1.0 - ps[0], name=f"p = {ps[0]:g}"),)
    ps = tuple(float(p) for p in ps)
    payoff_y = np.ascontiguousarray(payoff_y, dtype=float)
    key = (
        "sweep",
        _fast_highs() is not None,  # the fallback solves every LP from scratch
        form.key,
        payoff_y.tobytes(),
        np.array(ps).tobytes(),
    )
    xs = _recall(key)
    if xs is None:
        lps = [(p * payoff_y, 1.0 - p, f"p = {p:g}") for p in ps]
        xs = _remember(key, _solve_in_order(form, lps))
    return xs


def _constraints(form: HeroForm, g):
    """``linprog``'s constraints over ``(x, q, r)``: ``A_ub z <= b_ub``, ``A_eq z = b_eq`` and
    the bounds (``r``, the dual of ``G``'s rows, only with ``g``)."""
    n_x, n_q = form.own.shape[1], form.opponent.shape[0]
    n_r = 0 if g is None else g.shape[0]
    blocks = [-form.payoff.T, form.opponent.T]
    if n_r:
        blocks.append(g.T)
    a_ub = sparse.hstack(blocks).tocsr()
    a_eq = sparse.hstack([form.own, sparse.csr_matrix((form.own.shape[0], n_q + n_r))]).tocsr()
    b_eq = np.zeros(form.own.shape[0])
    b_eq[0] = 1.0
    bounds = [(0, None)] * n_x + [(None, None)] * n_q + [(0, None)] * n_r
    return a_ub, np.zeros(a_ub.shape[0]), a_eq, b_eq, bounds


def _cost(form: HeroForm, objective_x: np.ndarray, weight_q: float, n_cols: int) -> np.ndarray:
    """The cost vector ``linprog`` minimizes: the objective with its sign flipped."""
    n_x = form.own.shape[1]
    c = np.concatenate([-objective_x, np.zeros(n_cols - n_x)])
    c[n_x] -= weight_q
    return c


def _highs(form: HeroForm, objective_x: np.ndarray, weight_q: float, g) -> np.ndarray:
    """One LP solved by ``scipy.optimize.linprog``: the fallback, and the reference that a solve
    from scratch through the bindings reproduces bit for bit."""
    a_ub, b_ub, a_eq, b_eq, bounds = _constraints(form, g)
    c = _cost(form, objective_x, weight_q, a_ub.shape[1])
    res = linprog(c, A_ub=a_ub, b_ub=b_ub, A_eq=a_eq, b_eq=b_eq, bounds=bounds, method="highs")
    if res.status != 0:
        raise RuntimeError(f"robust-response LP failed: {res.message}")
    return res.x[: form.own.shape[1]].copy()


def _solve_in_order(form: HeroForm, lps, g=None) -> tuple[np.ndarray, ...]:
    """Solve the LPs ``(objective_x, weight_q, name)`` of ``form`` (and ``g``), which differ only
    in their costs, in order: the first from scratch on a new HiGHS instance, each later one
    hot-started from the optimal basis of the one before. If a hot start fails, that LP is
    solved from scratch on a new instance, and the next ones are hot-started from there; the
    result stays a function of the inputs. The hero part of each solution."""
    fast = _fast_highs()
    if fast is None:
        return tuple(_highs(form, objective_x, weight_q, g) for objective_x, weight_q, _ in lps)
    core, options = fast
    model = _form_model(core, form) if g is None else _model(core, form, g)
    n_x, n_cols = form.own.shape[1], model.cols.size
    xs, highs = [], None
    for objective_x, weight_q, name in lps:
        c = _cost(form, objective_x, weight_q, n_cols)
        if highs is not None:  # the same model with new costs; HiGHS keeps the last basis
            _check(core, highs.changeColsCost(n_cols, model.cols, c), "changeColsCost")
            try:
                xs.append(_run(core, highs, model, name, hot=True)[:n_x].copy())
                continue
            except RuntimeError:
                pass  # solved from scratch below
        # what linprog does: a new instance, this model with this cost, solved from scratch
        highs = core._Highs()
        _check(core, highs.passOptions(options), "passOptions")
        model.lp.col_cost_ = c
        _check(core, highs.passModel(model.lp), "passModel")
        xs.append(_run(core, highs, model, name, hot=False)[:n_x].copy())
    return tuple(xs)


_TOL = np.sqrt(1e-9) * 10  # linprog's _check_result with its default tol: about 3.2e-4


def _run(core, highs, model, name: str, hot: bool) -> np.ndarray:
    """Run HiGHS on its model: the whole solution, or an error unless it is optimal and passes
    ``linprog``'s check (``_feasible``), so the bindings fail where ``linprog`` fails."""
    ran = highs.run()
    status = highs.getModelStatus()
    how = f"{name}, {'hot-started' if hot else 'solved from scratch'}"
    if ran == core.HighsStatus.kError or status != core.HighsModelStatus.kOptimal:
        raise RuntimeError(
            f"robust-response LP failed ({how}): HiGHS model status "
            f"{highs.modelStatusToString(status)}"
        )
    solution = highs.getSolution()
    z = np.array(solution.col_value)
    rows = np.array(solution.row_value)
    if not _feasible(model, z, rows, highs.getInfo().objective_function_value):
        raise RuntimeError(
            f"robust-response LP failed ({how}): the solution misses the bounds or constraints "
            f"by more than {_TOL:.2e}"
        )
    return z


def _feasible(model, z: np.ndarray, rows: np.ndarray, fun: float) -> bool:
    """``linprog``'s check of an optimal solution (``_check_result``): no NaN, and the bounds,
    the inequalities and the equalities met within ``_TOL``."""
    n_ub = model.b_ub.size
    slack = model.b_ub - rows[:n_ub]
    con = model.b_eq - rows[n_ub:]
    if np.isnan(z).any() or np.isnan(fun) or np.isnan(slack).any() or np.isnan(con).any():
        return False
    bounds = np.all((z >= model.lower - _TOL) & (z <= model.upper + _TOL))
    return bool(bounds and not (slack < -_TOL).any() and not (np.abs(con) > _TOL).any())


def _check(core, status, call: str) -> None:
    if status == core.HighsStatus.kError:
        raise RuntimeError(f"robust-response LP failed: HiGHS {call} returned an error")


class _Model(NamedTuple):
    """An LP for HiGHS without its cost, and what ``_run`` needs to check a solution."""

    lp: object  # HighsLp
    cols: np.ndarray  # column indices 0 .. n - 1, for changeColsCost
    lower: np.ndarray  # column bounds, +-inf where unbounded
    upper: np.ndarray
    b_ub: np.ndarray  # right-hand sides of the inequality rows, then of the equality rows
    b_eq: np.ndarray


MODELS_SIZE = 8  # restricted Nash LP models (without costs) kept per process
_MODELS: OrderedDict[bytes, _Model] = OrderedDict()


def _form_model(core, form: HeroForm) -> _Model:
    """The restricted Nash LP of ``form``, built once per form. Its cost is set before each use
    (``passModel`` copies the model into the HiGHS instance), so, like ``_SOLVED``, it is not
    safe to share between threads; the experiments run in processes."""
    model = _MODELS.get(form.key)
    if model is None:
        model = _MODELS[form.key] = _model(core, form, None)
        if len(_MODELS) > MODELS_SIZE:
            _MODELS.popitem(last=False)
    else:
        _MODELS.move_to_end(form.key)
    return model


def _model(core, form: HeroForm, g) -> _Model:
    """The LP (without its cost) that ``linprog(method="highs")`` builds for ``form`` (and
    ``g``), as SciPy 1.18 builds it (``_clean_inputs``, ``_linprog_highs``): the rows of
    ``A_ub`` then those of ``A_eq``, stored column by column; row bounds ``(-inf, b_ub]`` and
    ``[b_eq, b_eq]``; ``None`` bounds infinite."""
    a_ub, b_ub, a_eq, b_eq, bounds = _constraints(form, g)
    a = sparse.csc_array(
        sparse.vstack(
            (
                sparse.coo_array(a_ub, dtype=float, copy=True),
                sparse.coo_array(a_eq, dtype=float, copy=True),
            )
        )
    )
    bounds = np.array(bounds, dtype=float)  # None -> nan
    bounds[np.isnan(bounds[:, 0]), 0] = -np.inf
    bounds[np.isnan(bounds[:, 1]), 1] = np.inf
    col_lower, col_upper = bounds.T.copy()
    row_lower = np.concatenate([np.full(b_ub.size, -np.inf), b_eq])
    row_upper = np.concatenate([b_ub, b_eq])
    lp = _highs_lp(core, a.indptr, a.indices, a.data, row_lower, row_upper, col_lower, col_upper)
    cols = np.arange(a.shape[1], dtype=np.int32)
    return _Model(lp, cols, col_lower, col_upper, b_ub, b_eq)


def _highs_lp(core, start, index, value, row_lower, row_upper, col_lower, col_upper):
    """A ``HighsLp`` filled as SciPy's ``_highs_wrapper`` fills it, without a cost; infinite
    bounds become HiGHS's infinity, as in ``_linprog_highs``."""

    def bound(v):
        v = np.array(v, dtype=float)
        inf = np.isinf(v)
        v[inf] = np.sign(v[inf]) * core.kHighsInf
        return v

    n_rows, n_cols = len(row_upper), len(col_upper)
    lp = core.HighsLp()
    lp.num_col_ = n_cols
    lp.num_row_ = n_rows
    lp.a_matrix_.num_col_ = n_cols
    lp.a_matrix_.num_row_ = n_rows
    lp.a_matrix_.format_ = core.MatrixFormat.kColwise
    lp.col_lower_ = bound(col_lower)
    lp.col_upper_ = bound(col_upper)
    lp.row_lower_ = bound(row_lower)
    lp.row_upper_ = bound(row_upper)
    lp.a_matrix_.start_ = start
    lp.a_matrix_.index_ = index
    lp.a_matrix_.value_ = value
    return lp


_HIGHS_MODULE = "scipy.optimize._highspy._core"  # SciPy's HiGHS bindings (private)
_FAST: list = []  # the probe's result once probed: [(bindings, options)] or [None]


def _fast_highs():
    """SciPy's HiGHS bindings and the options ``linprog(method="highs")`` passes them, or
    ``None`` (with one warning per process) if they are missing or fail a test LP at first use;
    after that, a failure raises."""
    if not _FAST:
        _FAST.append(_probe_highs())
    return _FAST[0]


def _probe_highs():
    try:
        core = importlib.import_module(_HIGHS_MODULE)
        options = core.HighsOptions()
        # what linprog(method="highs") sets with its default arguments (SciPy 1.18), in order
        options.presolve = "on"
        options.highs_debug_level = core.HighsDebugLevel.kHighsDebugLevelNone
        options.log_to_console = False
        options.output_flag = False
        options.simplex_strategy = core.simplex_constants.SimplexStrategy.kSimplexStrategyDual
        # min x0 + 2 x1 s.t. x0 + x1 >= 1, x >= 0: x = (1, 0); with costs (2, 1): x = (0, 1)
        lp = _highs_lp(core, [0, 1, 2], [0, 0], [1.0, 1.0], [1.0], [np.inf], [0, 0], [np.inf] * 2)
        lp.col_cost_ = np.array([1.0, 2.0])
        highs = core._Highs()
        _check(core, highs.passOptions(options), "passOptions")
        _check(core, highs.passModel(lp), "passModel")
        cold = _probe_run(core, highs)
        cols = np.arange(2, dtype=np.int32)
        _check(core, highs.changeColsCost(2, cols, np.array([2.0, 1.0])), "changeColsCost")
        hot = _probe_run(core, highs)
        if not (np.allclose(cold, [1.0, 0.0]) and np.allclose(hot, [0.0, 1.0])):
            raise RuntimeError(f"a test LP gave {cold} and {hot}")
    except Exception as e:  # anything unexpected: use linprog
        warnings.warn(
            f"SciPy's HiGHS bindings ({_HIGHS_MODULE}) are missing or have changed ({e!r}): "
            "robust responses fall back to linprog, slower, with every LP solved from scratch",
            RuntimeWarning,
            stacklevel=2,
        )
        return None
    return core, options


def _probe_run(core, highs) -> np.ndarray:
    """``_run`` for the probe (which the tests' spies on ``_run`` do not count)."""
    ran = highs.run()
    if ran == core.HighsStatus.kError or highs.getModelStatus() != core.HighsModelStatus.kOptimal:
        raise RuntimeError("a test LP was not solved")
    return np.array(highs.getSolution().col_value)


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
        return self._restricted_nash(model, hero, (p,))[0]

    def restricted_nash_sweep(self, model: TabularPolicy, ps) -> list[TabularPolicy]:
        """Restricted Nash responses (both seats, combined) for each ``p`` in ``ps``, the best
        response at ``p = 1``. The model's realization plan is computed once per seat, and each
        seat's LPs (the ``p`` below 1, in the order given) are solved by one HiGHS instance,
        hot-started after the first. The first such ``p`` gives exactly ``both_seats("rnr",
        model, p=p)``; later ones agree with their single responses, usually to about 1e-11;
        where two strategies are both optimal within HiGHS's tolerances they can differ more
        (see the module docstring)."""
        ps = tuple(ps)
        seats = [self._restricted_nash(model, hero, ps) for hero in (0, 1)]
        return [s0.combine(s1, 0) for s0, s1 in zip(*seats, strict=True)]

    def _restricted_nash(self, model: TabularPolicy, hero: int, ps) -> list[TabularPolicy]:
        ps = tuple(ps)
        if any(not 0.0 <= p <= 1.0 for p in ps):
            raise ValueError("p must be in [0, 1]")
        form = self.forms[hero]
        lps = tuple(p for p in ps if p != 1.0)  # p = 1: the best response, no LP
        xs: tuple[np.ndarray, ...] = ()
        if lps:
            plan = realization_plan(self.tree, model, 1 - hero)
            payoff_y = form.payoff @ _local_plan(form, self.tree, plan)  # A y_fix
            xs = _solve_sweep(form, payoff_y, lps)
        out, solved = [], 0
        for p in ps:
            if p == 1.0:
                out.append(best_response(model, hero).policy)
            else:
                out.append(_to_policy(self.tree, form, xs[solved], model, hero))
                solved += 1
        return out

    def data_biased(self, model: TabularPolicy, hero: int, confidence) -> TabularPolicy:
        """Hero's data-biased response; ``confidence[I]`` in [0, 1] per information set."""
        c = np.asarray(confidence, dtype=float)
        if c.shape != (self.tree.num_infosets,) or c.min() < 0 or c.max() > 1:
            raise ValueError("confidence must be in [0, 1], one value per information set")
        form = self.forms[hero]
        g = _confidence_rows(self.tree, form, model, hero, c)
        x = _solve(form, np.zeros(form.own.shape[1]), 1.0, g, name="data-biased response")
        return _to_policy(self.tree, form, x, model, hero)

    def both_seats(self, method: str, model: TabularPolicy, **kwargs) -> TabularPolicy:
        """A full hero strategy: the response as player 0 and as player 1, combined."""
        fn = {"rnr": self.restricted_nash, "dbr": self.data_biased}[method]
        seat0 = fn(model, 0, **kwargs)
        seat1 = fn(model, 1, **kwargs)
        return seat0.combine(seat1, 0)
