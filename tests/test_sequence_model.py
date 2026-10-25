"""Raw action sequences (tokens) and the sequence classifier of P3.8."""

import numpy as np
import pytest
import torch
from sklearn.model_selection import GroupShuffleSplit

from poker_lab.bots import Archetype, ParametricBot, sample_params
from poker_lab.engine import SeatedPlayer, TableSimulator
from poker_lab.profiling import decision_points, hand_counters
from poker_lab.profiling.sequence_model import HandSequences, SequenceClassifier, _Net
from poker_lab.profiling.sequences import (
    ACTIONS,
    N_TOKEN_FEATURES,
    TOKEN_FEATURES,
    SequenceRowExtractor,
    hand_tokens,
)
from poker_lab.profiling.situations import SITUATION_FEATURES
from poker_lab.profiling.stats import C
from poker_lab.schema import TableConfig, redact_hole_cards

COL = {name: i for i, name in enumerate(TOKEN_FEATURES)}


@pytest.fixture(scope="module")
def table():
    rng = np.random.default_rng(0)
    archetypes = list(Archetype)
    params = [sample_params(archetypes[i % 5], rng) for i in range(6)]
    players = [SeatedPlayer(i, f"p{i}", ParametricBot(params[i])) for i in range(6)]
    sim = TableSimulator(TableConfig("t", 6, 1, 2), players, seed=1, starting_stack=200)
    return params, [redact_hole_cards(h) for h in sim.play(300)]


def test_tokens_are_the_decisions_then_a_closing_token(table):
    _, hands = table
    for hand in hands:
        tokens = dict(hand_tokens(hand))
        points = decision_points(hand)
        kinds = [a.action_type for a in hand.actions if a.action_type.is_decision]
        counters = dict(hand_counters(hand))
        for pid, toks in tokens.items():
            mine = [(f, k) for (p, f, _), k in zip(points, kinds, strict=True) if p == pid]
            assert len(toks) == len(mine) + 1
            for tok, (feats, kind) in zip(toks, mine, strict=False):
                assert tok[0] == 0.0
                assert tok[1 : 1 + len(SITUATION_FEATURES)] == feats
                one_hot = [tok[COL[a.value]] for a in ACTIONS]
                assert one_hot == [float(a is kind) for a in ACTIONS]
                if not kind.is_aggressive:
                    assert tok[COL["log1p_raise_pot"]] == tok[COL["log_total_bb"]] == 0.0
            end = toks[-1]
            assert end[0] == 1.0
            c = counters[pid]
            assert end[COL["saw_flop"]] == float(c[C["saw_flop"]] > 0)
            assert end[COL["showdown"]] == float(c[C["wtsd"]] > 0)
            assert end[COL["won_showdown"]] == float(c[C["wsd"]] > 0)


def test_bet_sizes_recover_the_bots_size_parameter(table):
    params, hands = table
    sizes = {f"p{i}": [] for i in range(6)}
    for hand in hands:
        for pid, toks in hand_tokens(hand):
            for tok in toks[:-1]:
                if tok[COL["street"]] >= 1 and tok[COL["bet"]] == 1.0:
                    sizes[pid].append(np.expm1(tok[COL["log1p_raise_pot"]]))
    for i, p in enumerate(params):
        s = np.array(sizes[f"p{i}"])
        if len(s) >= 15:  # bots bet bet_size_pot * pot with small multiplicative noise
            assert np.median(s) == pytest.approx(p.bet_size_pot, rel=0.1)


def test_extractor_rows_rebuild_every_hand(table):
    _, hands = table
    extractor = SequenceRowExtractor()
    pop = {f"p{i}": i for i in range(6)}
    blocks = [np.array(extractor(h, pop, seq), dtype=np.float32) for seq, h in enumerate(hands)]
    rows = np.concatenate(blocks)
    assert rows.shape[1] == 2 + N_TOKEN_FEATURES
    seqs = HandSequences.from_rows(rows, 6)
    # hands may arrive in any order (the league collects tables as they finish)
    reordered = HandSequences.from_rows(np.concatenate(blocks[::-1]), 6)
    np.testing.assert_array_equal(reordered.tokens, seqs.tokens)
    np.testing.assert_array_equal(reordered.hand_ptr, seqs.hand_ptr)
    assert seqs.n_players == 6
    assert (seqs.hands() == len(hands)).all()
    first = dict(hand_tokens(hands[0]))["p2"]
    h = seqs.player_ptr[2]
    np.testing.assert_array_equal(
        seqs.tokens[seqs.hand_ptr[h] : seqs.hand_ptr[h + 1]], np.array(first, dtype=np.float32)
    )
    with pytest.raises(ValueError, match="closing"):
        HandSequences.from_rows(rows[:-1], 6)


def test_window_logits_pool_the_mean_of_hand_embeddings():
    torch.manual_seed(0)
    net = _Net(4, 8, "bag", 3, 0.0).eval()
    emb = torch.randn(30, 8)
    starts, n = torch.tensor([0, 5, 12]), torch.tensor([10, 7, 18])
    expected = torch.stack(
        [
            net.head(torch.cat([emb[s : s + k].mean(0), torch.log(k.float())[None]]))
            for s, k in zip(starts, n, strict=True)
        ]
    )
    torch.testing.assert_close(net.logits(emb, starts, n), expected, rtol=1e-5, atol=1e-5)


def test_rate_pooling_is_a_shrunk_rate_of_soft_counts():
    """With ``pool="rates"`` each hand gives soft successes s and failures f per rate; a window's
    feature is logit((S + a) / (S + F + a + b)) = log(S + a) - log(F + b)."""
    torch.manual_seed(1)
    net = _Net(4, 8, "bag", 3, 0.0, pool="rates", n_rates=2).eval()
    with torch.no_grad():
        net.log_prior.copy_(torch.log(torch.tensor([[2.0, 0.5], [3.0, 1.0]])))
    counts = net.hands(torch.randn(12, 4), torch.full((6,), 2))
    assert counts.shape == (6, 4)
    assert (counts >= 0).all()
    assert (counts[:, :2] + counts[:, 2:] <= 1 + 1e-6).all()  # successes + failures = chance
    S, F = counts[1:5, :2].sum(0), counts[1:5, 2:].sum(0)
    a, b = torch.tensor([2.0, 0.5]), torch.tensor([3.0, 1.0])
    rate = (S + a) / (S + F + a + b)
    x = torch.cat([torch.logit(rate), torch.log(torch.tensor([4.0]))])
    torch.testing.assert_close(
        net.logits(counts, torch.tensor([1]), torch.tensor([4]))[0],
        net.head(x),
        rtol=1e-4,
        atol=1e-5,
    )


def _order_toy(n_players, hands, rng):
    """Class 0 plays A then B in every hand, class 1 plays B then A: the same tokens, in a
    different order."""
    a, b, end = [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]
    rows, labels = [], []
    for p in range(n_players):
        k = p % 2
        labels.append(k)
        for h in range(hands):
            first, second = (a, b) if k == 0 else (b, a)
            noise = rng.normal(0, 0.05, 2)
            rows += [
                (p, h, first[0], first[1] + noise[0], first[2]),
                (p, h, second[0], second[1] + noise[1], second[2]),
                (p, h, *end),
            ]
    return HandSequences.from_rows(np.array(rows, dtype=np.float32), n_players), np.array(labels)


def test_the_gru_reads_order_and_the_bag_does_not():
    torch.set_num_threads(2)
    seqs, labels = _order_toy(40, 40, np.random.default_rng(0))
    test, test_labels = _order_toy(20, 40, np.random.default_rng(1))
    windows = np.array([(p, 0, 10) for p in range(20)])
    kw = {"hidden": 16, "epochs": 30, "patience": 30, "players_per_batch": 4, "random_state": 0}
    gru = SequenceClassifier("gru", **kw).fit(seqs, labels, [10])
    bag = SequenceClassifier("bag", **kw).fit(seqs, labels, [10])
    assert (gru.predict(test, windows) == test_labels).mean() == 1.0
    assert bag.predict_proba(test, windows).max(axis=1).max() < 0.75  # no better than a guess


def test_fit_is_reproducible_and_probabilities_are_valid():
    torch.set_num_threads(2)
    seqs, labels = _order_toy(20, 30, np.random.default_rng(2))
    windows = np.array([(p, s, n) for p in range(20) for s, n in ((0, 5), (3, 20), (0, 30))])
    kw = {"hidden": 8, "epochs": 3, "players_per_batch": 4, "random_state": 7}
    p1 = SequenceClassifier("gru", **kw).fit(seqs, labels, [5, 20]).predict_proba(seqs, windows)
    p2 = SequenceClassifier("gru", **kw).fit(seqs, labels, [5, 20]).predict_proba(seqs, windows)
    np.testing.assert_array_equal(p1, p2)
    np.testing.assert_allclose(p1.sum(axis=1), 1.0)
    clf = SequenceClassifier("bag", **kw).fit(seqs, labels, [5, 20])
    assert clf.size_temperatures is None  # 4 validation players: too few windows per size
    np.testing.assert_array_equal(
        clf.predict_proba(seqs, windows, calibration="by_size"), clf.predict_proba(seqs, windows)
    )
    with pytest.raises(ValueError, match="calibration"):
        clf.predict_proba(seqs, windows, calibration="isotonic")


def test_held_out_players_match_the_phase1_classifier_split():
    """The players that pick the epoch and fit the temperature are the ones the Phase 1
    classifier holds out for its temperature (same grouped split and seed)."""
    torch.set_num_threads(2)
    seqs, labels = _order_toy(30, 10, np.random.default_rng(3))
    clf = SequenceClassifier("bag", hidden=4, epochs=1, random_state=11).fit(seqs, labels, [5])
    groups = np.repeat(np.arange(30), 4)  # Phase 1 rows: several windows per player
    split = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=11)
    _, cal = next(split.split(groups, groups, groups))
    assert set(clf.val_players.tolist()) == set(groups[cal].tolist())


def test_vectors_make_one_token_per_hand():
    vectors = [np.ones((5, 3)), np.zeros((7, 3))]
    seqs = HandSequences.from_vectors(vectors)
    assert seqs.hands().tolist() == [5, 7]
    assert np.diff(seqs.hand_ptr).tolist() == [1] * 12
    with pytest.raises(ValueError, match="encoder"):
        SequenceClassifier("lstm")
