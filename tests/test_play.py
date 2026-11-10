"""Play against the model (``poker_lab.play``): firewall, session loop, HTTP validation, bind."""

from __future__ import annotations

import dataclasses
import http.client
import json
import random
import time

import numpy as np
import pytest

from poker_lab.engine import SeatedPlayer, TableSimulator
from poker_lab.engine.agent import IllegalActionError
from poker_lab.games.preflop import CLASS_INDEX, load_matchups
from poker_lab.play import holdem, pushfold
from poker_lab.play.allin_ev import all_in_result, equity
from poker_lab.play.channel import StaleDecision
from poker_lab.play.data import REPO_ROOT, DataDir
from poker_lab.play.server import PlayServer
from poker_lab.play.session import Session, Settings
from poker_lab.schema import FULL_DECK, CardObservation, HandHistory, HoleCards
from poker_lab.strategy import PushFoldStrategy, StrategyAgent, fixed_ranges
from tests.helpers import reachable_card_strings

CAL = pushfold.CalibrationConfig(bots=6, sizes=(10, 25), draws=40, posterior_draws=300, workers=1)
CLF = holdem.ClassifierConfig(
    per_archetype=6, hands=100, sizes=(25, 50, 100), windows_max=3, workers=1
)
KW = {"calibration": CAL, "classifier": CLF}
YOU, MODEL = 0, 1


@pytest.fixture(scope="module")
def thresholds():
    return pushfold.calibrate_thresholds(10, CAL)


@pytest.fixture(scope="module")
def cache_dir(tmp_path_factory):
    """A data folder whose classifier rows and thresholds are computed once for the module."""
    root = tmp_path_factory.mktemp("play")
    data = DataDir(root)
    holdem.training_rows(CLF, data.cache)
    return root


def _wide_human() -> StrategyAgent:
    """A scripted 'person': pushes 60% and calls 40% of hands (a loose player)."""
    ranges = fixed_ranges(np.full(169, 0.6), np.full(169, 0.4))
    return StrategyAgent(PushFoldStrategy(ranges))


def _pf_hands(thresholds, n=60, seed=3):
    model = pushfold.PushFoldModel(10, thresholds, draws=300, check_draws=40)
    players = [
        SeatedPlayer(YOU, pushfold.HUMAN_ID, _wide_human()),
        SeatedPlayer(MODEL, pushfold.MODEL_ID, pushfold.ModelAgent(model)),
    ]
    sim = TableSimulator(pushfold.table(), players, seed=seed, starting_stack=1000)
    return model, list(sim.play(n))


# ---------------------------------------------------------------------------- the firewall


def test_model_never_holds_your_unshown_cards(thresholds):
    model, hands = _pf_hands(thresholds)
    assert len(model.history) == len(hands) == 60
    unshown = 0
    visible: set[str] = set()
    for full, seen in zip(hands, model.history, strict=True):
        assert seen.hand_id == full.hand_id
        mine = seen.hole_cards_of(YOU)
        if full.hole_cards_of(YOU).shown:
            assert mine.cards == full.hole_cards_of(YOU).cards
            visible |= {o.card for o in mine.cards}
        else:
            unshown += 1
            assert mine.cards is None  # the model's copy has no cards for you
        visible |= {o.card for o in full.hole_cards_of(MODEL).cards} | set(full.board_cards())
    assert unshown > 10
    # Nothing reachable from the model (history, posterior, recommender, read) holds a card
    # string it could not have seen.
    assert reachable_card_strings(model) <= visible
    assert reachable_card_strings(model.read()) <= visible


def _swap_your_cards(hand: HandHistory) -> HandHistory:
    """The same hand with different cards for you (only meaningful if you did not show)."""
    used = {o.card for hc in hand.hole_cards for o in hc.cards} | set(hand.board_cards())
    spare = [c for c in FULL_DECK if c not in used][:2]
    hole = tuple(
        HoleCards(hc.seat, tuple(CardObservation(c) for c in spare), hc.shown)
        if hc.seat == YOU
        else hc
        for hc in hand.hole_cards
    )
    return dataclasses.replace(hand, hole_cards=hole)


def test_your_unshown_cards_cannot_change_the_model(thresholds):
    _, hands = _pf_hands(thresholds, n=40, seed=8)
    altered = [h if h.hole_cards_of(YOU).shown else _swap_your_cards(h) for h in hands]
    assert altered != hands
    a = pushfold.PushFoldModel(10, thresholds, history=hands, draws=300, check_draws=40)
    b = pushfold.PushFoldModel(10, thresholds, history=altered, draws=300, check_draws=40)
    assert json.dumps(a.read()) == json.dumps(b.read())


def test_holdem_profile_uses_public_information_only(cache_dir):
    rows = holdem.training_rows(CLF, DataDir(cache_dir).cache)
    model = holdem.HoldemModel(rows)
    bots = holdem.opponents("bots", 6, seed=1, engine="none")  # the first one plays your seat
    players = [SeatedPlayer(YOU, holdem.HUMAN_ID, bots[0][1])]
    players += [SeatedPlayer(i, pid, a) for i, (pid, a, _) in enumerate(bots) if i]
    sim = TableSimulator(holdem.table(6), players, seed=2)
    for hand in sim.play(30):
        model.observe(hand)
    assert model.store.hands_ingested == 30
    assert reachable_card_strings(model) == set()  # counters only, never hands
    read = model.read(6)
    assert abs(sum(a["probability"] for a in read["archetypes"]) - 1) < 1e-9


# ---------------------------------------------------------------------------- all-in EV


def test_equity_is_exact_from_the_flop_and_zero_sum():
    a, b, flop = ("As", "Ah"), ("Kd", "Kc"), ("2s", "7d", "9c")
    e1, m1 = equity(a, b, flop, "x")
    e2, _ = equity(b, a, flop, "x")
    assert m1 == "exact"
    assert e1 + e2 == pytest.approx(1.0, abs=1e-12)


def test_preflop_equity_is_seeded_monte_carlo_close_to_the_table():
    e1, m1 = equity(("As", "Ah"), ("Kd", "Kc"), (), "hand-1")
    e2, _ = equity(("As", "Ah"), ("Kd", "Kc"), (), "hand-1")
    assert m1 == "monte_carlo"
    assert e1 == e2
    m = load_matchups()
    assert e1 == pytest.approx(m.equity[CLASS_INDEX["AA"], CLASS_INDEX["KK"]], abs=0.015)


def test_all_in_ev_is_zero_sum_and_unadjusted_without_an_all_in(thresholds):
    _, hands = _pf_hands(thresholds, n=60, seed=4)
    adjusted = 0
    for h in hands:
        r0, r1 = all_in_result(h, YOU), all_in_result(h, MODEL)
        assert r0.actual + r1.actual == 0
        if r0.adjusted:
            adjusted += 1
            assert r0.ev + r1.ev == pytest.approx(0.0, abs=1e-6)
        else:
            assert r0.ev == r0.actual
    assert adjusted > 5


# ---------------------------------------------------------------------------- the session loop


def _wait(session, pred, timeout=60.0):
    t0 = time.time()
    while time.time() - t0 < timeout:
        st = session.state()
        assert st["session"]["status"] != "error", st["session"]["error"]
        if pred(st):
            return st
        time.sleep(0.01)
    raise AssertionError("timed out")


def _play(session, hands, choose):
    played = 0
    while True:
        st = _wait(session, lambda s: s["turn"] or s["session"]["status"] == "hand_over")
        if st["turn"]:
            session.submit({"token": st["turn"]["token"], **choose(st["turn"])})
            continue
        played += 1
        if played == hands:
            return
        session.next_hand()


def test_push_fold_session_loop(cache_dir, thresholds):
    data = DataDir(cache_dir)
    s = Session(Settings.parse({"mode": "pushfold", "stack_bb": 10, "seed": 5}), data, **KW)
    rng = random.Random(0)

    def choose(turn):
        assert turn["your_cards"]
        assert len(turn["your_cards"]) == 2
        actions = [b["action"] for b in turn["buttons"]]
        assert actions in (["fold", "allin"], ["fold", "call"])
        return {"action": rng.choice(actions)}

    st = _wait(s, lambda x: x["turn"] or x["session"]["status"] == "hand_over")
    skipped = 0
    while not st["turn"]:  # the model folded its small blind: no decision for you
        skipped += 1
        s.next_hand()
        st = _wait(s, lambda x: x["turn"] or x["session"]["status"] == "hand_over")
    t = st["turn"]
    if t["legal"]["can_raise"]:  # push/fold only: no smaller raise, no limp
        with pytest.raises(IllegalActionError):
            s.submit({"token": t["token"], "action": "raise", "amount": t["legal"]["min_raise_to"]})
        with pytest.raises(IllegalActionError):
            s.submit({"token": t["token"], "action": "call"})
    with pytest.raises(StaleDecision):
        s.submit({"token": t["token"] + 1, "action": "fold"})
    _play(s, 15, choose)
    report = s.end()
    assert report["results"]["hands"] == 15 + skipped
    assert sum(x["net_bb"] for x in s.log) == pytest.approx(report["results"]["bb"])
    assert report["model"]["advised"]["rank"] >= 1
    lines = data.hands_path(s.id).read_text().splitlines()
    assert len(lines) == 15 + skipped
    assert json.loads(data.report_path(s.id).read_text())["session_id"] == s.id
    assert s.state()["read"] is None  # the read is off by default
    # A new session at the same stack starts from these hands unless it is fresh.
    s2 = Session(Settings.parse({"mode": "pushfold", "stack_bb": 10, "seed": 6}), data, **KW)
    st = _wait(s2, lambda x: x["turn"] or x["session"]["status"] == "hand_over")
    s2.set_read(True)
    assert s2.state()["read"]["earlier_hands"] >= 15
    s2.end()


@pytest.mark.parametrize(("opponents", "players"), [("bots", 6), ("advisor", 2)])
def test_holdem_session_loop(cache_dir, opponents, players):
    data = DataDir(cache_dir)
    settings = Settings.parse(
        {"mode": "holdem", "players": players, "opponents": opponents, "seed": 9}
    )
    s = Session(settings, data, **KW)

    def choose(turn):
        legal = turn["legal"]
        if legal["can_raise"] and turn["street"] == "preflop":
            return {"action": "raise", "amount": legal["min_raise_to"]}
        return {"action": "check" if legal["can_check"] else "call"}

    _play(s, 5, choose)
    s.set_read(True)
    read = s.state()["read"]
    assert read["hands"] == 5
    report = s.end()
    assert len(report["opponents"]) == players - 1
    if opponents == "advisor":
        assert report["model"]["exploit"]["kind"] == "heuristic"
    else:
        assert all("archetype" in o for o in report["opponents"])


def test_settings_validation():
    ok = Settings.parse({"mode": "holdem", "players": 2, "opponents": "advisor", "engine": "none"})
    assert ok.players == 2
    for bad in (
        [],
        {"mode": "omaha"},
        {"mode": "pushfold", "stack_bb": 4},
        {"mode": "pushfold", "stack_bb": 21},
        {"mode": "pushfold", "stack_bb": True},
        {"mode": "pushfold", "stack_bb": 10.0},
        {"mode": "pushfold", "fresh": 1},
        {"mode": "holdem", "players": 3},
        {"mode": "holdem", "opponents": "humans"},
        {"mode": "holdem", "engine": "magic"},
        {"mode": "holdem", "engine": [1]},
        {"mode": "holdem", "engine": {}},
        {"mode": "pushfold", "seed": -1},
        {"mode": "pushfold", "extra": 1},
    ):
        with pytest.raises(ValueError, match=r"must|unknown"):
            Settings.parse(bad)


def test_data_folder_must_be_outside_the_repository(tmp_path):
    with pytest.raises(ValueError, match="inside the repository"):
        DataDir(REPO_ROOT / "my-hands")
    DataDir(tmp_path)  # anywhere else is fine


# ---------------------------------------------------------------------------- HTTP


@pytest.fixture
def server(cache_dir):
    srv = PlayServer(0, DataDir(cache_dir), **KW)
    import threading

    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    yield srv
    srv.end_session()
    srv.shutdown()
    srv.server_close()


def _req(srv, method, path, body=None, *, ctype="application/json", headers=None, raw=None):
    port = srv.server_address[1]
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=30)
    h = {"Host": f"127.0.0.1:{port}"} | (headers or {})
    data = raw if raw is not None else (None if body is None else json.dumps(body).encode())
    if data is not None:
        h["Content-Type"] = ctype
    conn.request(method, path, body=data, headers=h)
    res = conn.getresponse()
    payload = res.read()
    conn.close()
    try:
        return res.status, json.loads(payload), res
    except ValueError:
        return res.status, payload, res


def test_server_binds_loopback_only(cache_dir):
    with pytest.raises(ValueError, match="binds 127"):
        PlayServer(0, DataDir(cache_dir), host="0.0.0.0")
    srv = PlayServer(0, DataDir(cache_dir))
    try:
        assert srv.server_address[0] == "127.0.0.1"
    finally:
        srv.server_close()


def test_static_files_and_headers(server):
    status, body, res = _req(server, "GET", "/")
    assert status == 200
    assert b'<script src="/app.js">' in body
    assert "default-src 'none'" in res.getheader("Content-Security-Policy")
    for path in ("/app.js", "/app.css"):
        status, body, _ = _req(server, "GET", path)
        assert status == 200
        assert b"http://" not in body
        assert b"https://" not in body
    assert _req(server, "GET", "/../poker_lab/play/server.py")[0] == 404
    assert _req(server, "GET", "/static/app.js")[0] == 404
    assert _req(server, "GET", "/", headers={"Host": "evil.example:80"})[0] == 403


def test_post_validation(server):
    port = server.server_address[1]
    good = {"mode": "pushfold", "stack_bb": 10, "seed": 1}
    assert _req(server, "POST", "/api/new", good, ctype="text/plain")[0] == 415
    assert _req(server, "POST", "/api/new", raw=b"{not json")[0] == 400
    assert _req(server, "POST", "/api/new", raw=b"\xff\xfe")[0] == 400
    assert _req(server, "POST", "/api/new", raw=b" " * 5000)[0] == 413
    assert _req(server, "POST", "/api/new", raw=b"[" * 4000)[0] == 400  # too deep for json
    assert _req(server, "POST", "/api/new", {"mode": "holdem", "engine": [1]})[0] == 400
    assert _req(server, "POST", "/api/new", {"mode": "pushfold", "stack_bb": "10"})[0] == 400
    assert _req(server, "POST", "/api/new", {"mode": "pushfold", "x": 1})[0] == 400
    evil = {"Origin": "http://evil.example"}
    assert _req(server, "POST", "/api/new", good, headers=evil)[0] == 403
    cross = {"Sec-Fetch-Site": "cross-site"}
    assert _req(server, "POST", "/api/new", good, headers=cross)[0] == 403
    assert _req(server, "POST", "/api/new", good, headers={"Host": "evil.example"})[0] == 403
    assert _req(server, "POST", "/api/act", {"token": 1, "action": "fold"})[0] == 409  # no session
    assert _req(server, "POST", "/api/nope", {})[0] == 404
    same = {"Origin": f"http://localhost:{port}", "Host": f"localhost:{port}"}
    status, st, _ = _req(server, "POST", "/api/new", good, headers=same)
    assert status == 200
    assert st["session"]["status"] in ("preparing", "ready", "playing")
    session = server.session
    st = _wait(session, lambda x: x["turn"] or x["session"]["status"] == "hand_over")
    if st["turn"]:
        token = st["turn"]["token"]
        for bad in (
            {"token": token, "action": "shove"},
            {"token": str(token), "action": "fold"},
            {"token": token, "action": "fold", "amount": 5},
            {"token": token, "action": "raise"},
            {"token": token, "action": "raise", "amount": True},
            {"token": token, "action": "fold", "why": "x"},
            [token, "fold"],
        ):
            assert _req(server, "POST", "/api/act", bad)[0] == 400, bad
        assert _req(server, "POST", "/api/act", {"token": token + 7, "action": "fold"})[0] == 409
        assert _req(server, "POST", "/api/next", {})[0] == 409  # mid-hand
        assert _req(server, "POST", "/api/act", {"token": token, "action": "fold"})[0] == 200
    assert _req(server, "POST", "/api/read", {"on": 1})[0] == 400
    assert _req(server, "POST", "/api/read", {"on": True, "x": 0})[0] == 400
    status, st, _ = _req(server, "POST", "/api/read", {"on": True})
    assert status == 200
    assert st["session"]["read_on"] is True
    assert _req(server, "POST", "/api/end", {"now": True})[0] == 400
    status, st, _ = _req(server, "POST", "/api/end", {})
    assert status == 200
    assert st["report"]["session_id"] == session.id
    assert _req(server, "POST", "/api/next", {})[0] == 409  # ended
