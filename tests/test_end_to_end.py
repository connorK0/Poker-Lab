"""The whole pipeline runs end to end: simulate -> sealed store -> firewalled advisor
(strategy + exploit + profiles from completed hands) -> hand histories -> profiles ->
evaluator with confidence intervals -> JSON Lines round trip."""

from poker_lab.advisor import Advisor, AdvisorAgent
from poker_lab.bots import Archetype, ParametricBot, mean_params
from poker_lab.engine import SeatedPlayer, TableSimulator
from poker_lab.evaluation import WinRateEvaluator, hand_results
from poker_lab.exploit import FoldEquityHeuristic
from poker_lab.profiling.profile_store import ProfileStore
from poker_lab.schema import TableConfig, read_jsonl, validate_hand_history, write_jsonl
from poker_lab.strategy import PotOddsStrategy


def test_end_to_end(tmp_path):
    hero = AdvisorAgent(Advisor(PotOddsStrategy(seed=1), FoldEquityHeuristic(), seed=2))
    bots = {f"bot_{a.value}": ParametricBot(mean_params(a), equity_samples=60) for a in Archetype}
    players = [SeatedPlayer(0, "hero", hero)] + [
        SeatedPlayer(i + 1, pid, bot) for i, (pid, bot) in enumerate(bots.items())
    ]
    sim = TableSimulator(TableConfig("e2e", 6, 1, 2), players, seed=99)
    hands = list(sim.play(150))

    # every record is complete and internally consistent, and survives serialization
    for hand in hands:
        validate_hand_history(hand)
    path = tmp_path / "session.jsonl.gz"
    write_jsonl(hands, path)
    assert list(read_jsonl(path, validate=True)) == hands

    # the advisor's own profiles grew from completed hands only
    assert hero.store.hands_ingested == 150
    assert set(hero.store.players()) == {"hero", *bots}
    assert all(a.exploitability.method for a in hero.advice_log)

    # an independent profiler rebuilt from the saved session agrees
    offline = ProfileStore()
    offline.ingest_many(read_jsonl(path))
    for pid in bots:
        assert offline.profile(pid).counters == hero.store.profile(pid).counters

    # win rate with a confidence interval
    est = WinRateEvaluator().evaluate(hand_results(hands, "hero"))
    assert est.n_hands == 150
    assert est.lower < est.bb_per_100 < est.upper
