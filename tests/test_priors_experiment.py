"""P3.7 experiment helpers: the held-out error estimator, the decision rules, the loaders."""

import numpy as np

from experiments.phase3_priors import (
    UNIFORM,
    Players,
    assemble,
    consistency_verdict,
    player_errors,
    session_key,
    smallest_k,
)
from poker_lab.profiling.stats import BINOMIAL_STATS, N_COUNTERS, C

VPIP = [s.name for s in BINOMIAL_STATS].index("vpip")


def _players(rates, hands, rng) -> Players:
    counters = []
    for p in rates:
        c = np.zeros((hands, N_COUNTERS), dtype=np.int32)
        c[:, C["hands"]] = 1
        c[:, C["vpip_opp"]] = 1
        c[:, C["vpip"]] = rng.random(hands) < p
        counters.append(c)
    ids = [f"p{i}" for i in range(len(rates))]
    halves = [np.zeros(hands, dtype=np.int8) for _ in rates]
    return Players(ids, counters, halves, np.zeros((0, 20), np.float32), {})


def test_held_out_error_minus_noise_estimates_the_error_against_the_true_rate():
    rng = np.random.default_rng(0)
    rates = rng.beta(20, 60, size=400)
    players = _players(rates, 400, rng)
    prior = UNIFORM.copy()
    prior[:, VPIP] = (20, 60)
    priors = np.stack([UNIFORM, prior])
    n = 50
    held_out, truth = [], []
    for i, p in enumerate(rates):
        err, noise = player_errors(players, i, n, cap=4, priors=priors)
        held_out.append(err[:, VPIP] - noise[VPIP])
        cum = players.cum[i]
        starts = n * np.unique(np.linspace(0, 400 // n - 1, 4).round().astype(int))
        k = (cum[starts + n] - cum[starts])[:, C["vpip"]]
        a, b = priors[:, 0, VPIP, None], priors[:, 1, VPIP, None]
        truth.append((((a + k) / (a + b + n) - p) ** 2).mean(axis=1))
    held_out, truth = np.mean(held_out, axis=0), np.mean(truth, axis=0)
    assert np.allclose(held_out, truth, rtol=0.15)
    assert held_out[1] < held_out[0]  # the right prior beats the uniform one
    # stats without opportunities are undefined, not zero
    err, _ = player_errors(players, 0, n, cap=4, priors=priors)
    assert np.isnan(err[:, [j for j in range(len(BINOMIAL_STATS)) if j != VPIP]]).all()


def test_smallest_k_needs_every_larger_k_to_pass():
    good, bad = np.array([1.0, -0.5]), np.array([-1.0, 0.2])
    assert smallest_k({2: {25: bad}, 5: {25: good}, 12: {25: good}}) == 5
    assert smallest_k({2: {25: good}, 5: {25: bad}, 12: {25: good}}) == 12
    assert smallest_k({2: {25: good}, 5: {25: good}, 12: {25: bad}}) is None
    assert smallest_k({2: {25: good, 50: bad}, 5: {25: good, 50: good}}) == 5


def test_consistency_verdict():
    def d(lo, hi):
        return {"difference": {"value": (lo + hi) / 2, "lower": lo, "upper": hi}}

    assert consistency_verdict({"25": d(-0.05, 0.08)})["verdict"] == "transfers"
    assert consistency_verdict({"25": d(-0.05, 0.08), "50": d(0.01, 0.2)})["verdict"] == "fails"
    assert consistency_verdict({"25": d(-0.15, 0.08)})["verdict"] == "inconclusive"
    assert consistency_verdict({})["verdict"] == "no data"


def test_session_key_orders_the_pluribus_files():
    files = ["pluribus/100b/7.phh", "pluribus/30/12.phh", "pluribus/100/9.phh", "pluribus/30/2.phh"]
    assert sorted(files, key=session_key) == [
        "pluribus/30/2.phh",
        "pluribus/30/12.phh",
        "pluribus/100/9.phh",
        "pluribus/100b/7.phh",
    ]


def test_assemble_orders_each_players_hands_and_groups_the_seating():
    row = [0] * N_COUNTERS
    hands = [
        ("s1", [("a", row), ("b", row)], [("a", (0.5,), 1)], 0),
        ("s1", [("b", row), ("a", row)], [("b", (0.1,), 2), ("a", (0.2,), 0)], 1),
        ("s2", [("a", row), ("c", row)], [("a", (0.3,), 2)], 0),
    ]
    players, seating = assemble(hands, {})
    assert players.ids == ["a", "b", "c"]
    assert list(players.hands) == [3, 2, 1]
    assert seating == [(("a", "b"), 2), (("a", "c"), 1)]  # the seat order rotates in a session
    seq_of_a = players.decisions[players.decisions[:, 0] == 0][:, 1]
    assert list(seq_of_a) == [0, 1, 2]
    assert list(players.halves[0]) == [0, 1, 0]
