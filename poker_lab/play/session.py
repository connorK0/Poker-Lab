"""A play session: the engine thread, the state the page polls, and the end-of-session report.

``Session`` owns a ``TableSimulator`` and runs it in its own thread. Your seat is a
``HumanAgent``; when the engine needs your decision the thread sleeps in the
``DecisionChannel`` until ``submit`` answers it (see ``channel``). Between hands the thread waits
for ``next_hand``. Every completed hand is appended to the session's JSON Lines file in the data
folder, scored (chips and all-in EV) and, in hold'em, profiled.

The model's read is computed on the engine thread after each hand and kept as plain JSON, so the
server threads never touch the model while it refits. ``state()`` includes it only while the
read is switched on (off by default); the report always includes it.
"""

from __future__ import annotations

import dataclasses
import secrets
import sys
import threading
import time
import traceback
from dataclasses import dataclass
from typing import Any

from poker_lab.engine import SeatedPlayer, TableSimulator
from poker_lab.engine.agent import Decision, DecisionKind, IllegalActionError
from poker_lab.play import holdem, pushfold
from poker_lab.play.allin_ev import all_in_result
from poker_lab.play.channel import DecisionChannel, HumanAgent, SessionClosed
from poker_lab.play.data import DataDir
from poker_lab.play.exploit_slot import DEFAULT, ENGINES
from poker_lab.play.present import bb, describe_hand, describe_view
from poker_lab.schema import HandHistory, SeatView

MODES = ("pushfold", "holdem")
OPPONENTS = ("bots", "advisor")
ACTIONS = ("fold", "check", "call", "raise", "allin")
YOU = 0  # your seat


@dataclass(frozen=True)
class Settings:
    mode: str
    stack_bb: int = 10  # push/fold
    players: int = 6  # hold'em: 2 or 6
    opponents: str = "bots"  # hold'em
    engine: str = DEFAULT  # hold'em advisors' exploit engine
    fresh: bool = False  # push/fold: ignore your earlier sessions
    seed: int | None = None

    @classmethod
    def parse(cls, obj: Any) -> Settings:
        """Strictly validated settings from the page's JSON (``ValueError`` on anything else)."""
        if not isinstance(obj, dict):
            raise ValueError("settings must be a JSON object")
        allowed = {f.name for f in dataclasses.fields(cls)}
        if unknown := set(obj) - allowed:
            raise ValueError(f"unknown settings: {sorted(unknown)}")
        mode = obj.get("mode")
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")

        def integer(name: str, default: int | None, lo: int, hi: int) -> int | None:
            v = obj.get(name, default)
            if v is None and default is None:
                return None
            if type(v) is not int or not lo <= v <= hi:
                raise ValueError(f"{name} must be an integer in [{lo}, {hi}]")
            return v

        stack = integer("stack_bb", 10, pushfold.MIN_STACK, pushfold.MAX_STACK)
        players = integer("players", 6, 2, 6)
        if players not in (2, 6):
            raise ValueError("players must be 2 or 6")
        opponents = obj.get("opponents", "bots")
        if opponents not in OPPONENTS:
            raise ValueError(f"opponents must be one of {OPPONENTS}")
        engine = obj.get("engine", DEFAULT)
        if not isinstance(engine, str) or engine not in ENGINES:
            raise ValueError(f"engine must be one of {sorted(ENGINES)}")
        fresh = obj.get("fresh", False)
        if type(fresh) is not bool:
            raise ValueError("fresh must be true or false")
        seed = integer("seed", None, 0, 2**31 - 1)
        return cls(mode, stack, players, opponents, engine, fresh, seed)


def parse_decision(obj: Any, view: SeatView) -> tuple[int, Decision]:
    """``(token, Decision)`` from ``{"token", "action", "amount"?}``; ``ValueError`` if malformed
    (legality is checked later, against the pending view)."""
    if not isinstance(obj, dict) or set(obj) - {"token", "action", "amount"}:
        raise ValueError("expected {token, action, amount?}")
    token, action = obj.get("token"), obj.get("action")
    if type(token) is not int:
        raise ValueError("token must be an integer")
    if action not in ACTIONS:
        raise ValueError(f"action must be one of {ACTIONS}")
    amount = obj.get("amount")
    if action == "raise":
        if type(amount) is not int or amount <= 0:
            raise ValueError("a raise needs a positive integer amount (chips, the street total)")
    elif amount is not None:
        raise ValueError("only a raise takes an amount")
    legal = view.legal
    assert legal is not None
    if action == "fold":
        return token, Decision.fold()
    if action in ("check", "call"):
        if (action == "check") != legal.can_check:
            raise IllegalActionError("check is free here" if legal.can_check else "you must call")
        return token, Decision.check_call()
    if action == "allin":
        if not legal.can_bet_or_raise:
            raise IllegalActionError("betting or raising is not allowed here")
        return token, Decision.bet_raise(legal.max_raise_to)
    return token, Decision.bet_raise(amount)


def push_fold_only(view: SeatView, decision: Decision) -> None:
    """Push/fold mode: push or fold as the small blind, call or fold facing a push."""
    legal = view.legal
    assert legal is not None
    if legal.can_bet_or_raise:
        ok = decision.kind is DecisionKind.FOLD or (
            decision.kind is DecisionKind.BET_RAISE and decision.to_amount == legal.max_raise_to
        )
    else:
        ok = decision.kind in (DecisionKind.FOLD, DecisionKind.CHECK_CALL)
    if not ok:
        raise IllegalActionError("push/fold: push or fold, and call or fold facing a push")


class Conflict(RuntimeError):
    """The request does not fit the session's current state (e.g. next hand mid-hand)."""


class Session:
    def __init__(
        self,
        settings: Settings,
        data: DataDir,
        *,
        calibration: pushfold.CalibrationConfig | None = None,
        classifier: holdem.ClassifierConfig | None = None,
        start: bool = True,
    ) -> None:
        self.settings = settings
        self.data = data
        self.calibration = calibration or pushfold.CalibrationConfig()
        self.classifier = classifier or holdem.ClassifierConfig()
        self.mode = settings.mode
        self.id = data.new_session_id(self.mode)
        self.seed = settings.seed if settings.seed is not None else secrets.randbits(31)
        self.channel = DecisionChannel()
        self.lock = self.channel.cond  # one lock (re-entrant) for all session state
        self.status = "preparing"
        self.message = "Starting"
        self.error: str | None = None
        self.started = time.strftime("%Y-%m-%d %H:%M:%S")
        self.hands = 0
        self.chips = 0
        self.ev = 0.0
        self.adjusted = 0
        self.net_by_player: dict[str, int] = {}
        self.log: list[dict[str, Any]] = []
        self.last_hand: dict[str, Any] | None = None
        self.read_on = False
        self.read: dict[str, Any] | None = None
        self.report: dict[str, Any] | None = None
        self.big_blind = 1
        self._deal = True
        self._your_hand: HandHistory | None = None
        self.agents: list[Any] = []
        self.revealed: dict[str, dict[str, str]] = {}
        self.thread = threading.Thread(target=self._run, name=f"session-{self.id}", daemon=True)
        if start:
            self.thread.start()

    # ------------------------------------------------------------------ engine thread
    def _progress(self, text: str) -> None:
        with self.lock:
            if self.channel.closed:
                raise SessionClosed("session closed")
            self.message = text

    def _prepare(self) -> None:
        s = self.settings
        human = HumanAgent(self.channel, self._on_your_hand)
        if self.mode == "pushfold":
            cal = self.calibration
            name = cal.cache_name(s.stack_bb)
            cached = self.data.read_cache(name)
            if cached is None:
                self._progress(f"Calibrating the model-fit checks for {s.stack_bb} bb (once)")
                th = pushfold.calibrate_thresholds(
                    s.stack_bb,
                    cal,
                    lambda k, n: self._progress(
                        f"Calibrating the model-fit checks for {s.stack_bb} bb (once): {k}/{n}"
                    ),
                )
                self.data.write_cache(name, {str(k): v for k, v in th.items()})
            else:
                th = {int(k): v for k, v in cached.items()}
            self._progress("Calibration ready")  # stops here if ended meanwhile
            history = []
            if not s.fresh:
                self._progress("Loading your earlier hands at this stack")
                history = [
                    h
                    for h in self.data.earlier_hands("pf", exclude=self.id)
                    if h.metadata.get("stack_bb") == str(s.stack_bb)
                ]
            self._progress("Fitting the model")
            self.model = pushfold.PushFoldModel(s.stack_bb, th, history=history)
            model_agent = pushfold.ModelAgent(self.model)
            self.agents = [model_agent]
            table = pushfold.table()
            players = [
                SeatedPlayer(YOU, pushfold.HUMAN_ID, human),
                SeatedPlayer(1, pushfold.MODEL_ID, model_agent),
            ]
            stack = s.stack_bb * pushfold.BIG_BLIND
            meta = {"mode": "pushfold", "stack_bb": str(s.stack_bb)}
        else:
            self._progress("Preparing the archetype classifier (the first time: a bot league)")
            rows = holdem.training_rows(self.classifier, self.data.cache)
            self._progress("Fitting the archetype classifier")  # stops here if ended meanwhile
            self.model = holdem.HoldemModel(rows)
            opps = holdem.opponents(
                s.opponents, s.players - 1, self.seed, s.engine, self.model.shrinker
            )
            self.agents = [a for _, a, _ in opps]
            self.revealed = {pid: info for pid, _, info in opps}
            table = holdem.table(s.players)
            players = [SeatedPlayer(YOU, holdem.HUMAN_ID, human)] + [
                SeatedPlayer(i + 1, pid, agent) for i, (pid, agent, _) in enumerate(opps)
            ]
            stack = holdem.STACK
            meta = {"mode": "holdem", "players": str(s.players), "opponents": s.opponents}
        self.big_blind = table.big_blind
        self.sim = TableSimulator(
            table,
            players,
            seed=self.seed,
            starting_stack=stack,
            hand_id_prefix=f"{self.id}-",
            metadata=meta,
        )
        self._update_read()

    def _update_read(self) -> None:
        if self.mode == "pushfold":
            read = self.model.read()
        else:
            read = self.model.read(self.settings.players)
            read["exploit"] = holdem.exploit_summary(self.agents, self.settings.engine)
        with self.lock:
            self.read = read

    def _on_your_hand(self, hand: HandHistory) -> None:
        self._your_hand = hand  # redacted to your seat by the simulator

    def _record(self, hand: HandHistory) -> None:
        self.data.append_hand(self.id, hand)
        mine = self._your_hand
        assert mine is not None
        assert mine.hand_id == hand.hand_id
        result = all_in_result(mine, YOU)
        if self.mode == "holdem":
            self.model.observe(hand)
        summary = describe_hand(mine, YOU)
        summary |= {
            "number": self.hands + 1,
            "ev": round(result.ev, 2),
            "adjusted": result.adjusted,
            "equity": result.equity,
            "method": result.method,
        }
        self._update_read()
        with self.lock:
            self.hands += 1
            self.chips += result.actual
            self.ev += result.ev
            self.adjusted += result.adjusted
            for s in hand.seats:
                pid = s.player_id
                self.net_by_player[pid] = (
                    self.net_by_player.get(pid, 0) + hand.net_results()[s.seat]
                )
            self.last_hand = summary
            self.log.append(
                {
                    "number": summary["number"],
                    "hand_id": hand.hand_id,
                    "your_cards": summary["your_cards"],
                    "net_bb": bb(result.actual, self.big_blind),
                    "ev_bb": bb(result.ev, self.big_blind),
                    "adjusted": result.adjusted,
                }
            )

    def _run(self) -> None:
        try:
            self._prepare()
            while True:
                with self.lock:
                    self.status, self.message = "hand_over" if self.hands else "ready", ""
                    while not self._deal and not self.channel.closed:
                        self.lock.wait()
                    if self.channel.closed:
                        return
                    self._deal = False
                    self.status = "playing"
                hand = self.sim.play_hand()
                self._record(hand)
        except SessionClosed:
            pass
        except Exception as exc:  # shown on the page; the traceback goes to the console
            traceback.print_exc(file=sys.stderr)
            with self.lock:
                self.status, self.error = "error", f"{type(exc).__name__}: {exc}"

    # ------------------------------------------------------------------ server threads
    def submit(self, obj: Any) -> None:
        pending = self.channel.pending()
        if pending is None:
            raise Conflict("no decision is waiting")
        token, decision = parse_decision(obj, pending[1])
        check = push_fold_only if self.mode == "pushfold" else None
        self.channel.submit(token, decision, check)

    def next_hand(self) -> None:
        with self.lock:
            if self.status not in ("hand_over", "ready") or self.channel.closed:
                raise Conflict("a hand is in progress or the session has ended")
            self._deal = True
            self.status = "playing"
            self.lock.notify_all()

    def set_read(self, on: bool) -> None:
        with self.lock:
            self.read_on = on

    def results(self) -> dict[str, Any]:
        with self.lock:
            n, big = self.hands, self.big_blind
            return {
                "hands": n,
                "chips": self.chips,
                "bb": bb(self.chips, big),
                "allin_ev_bb": bb(self.ev, big),
                "bb_per_hand": round(self.chips / big / n, 3) if n else None,
                "allin_ev_bb_per_hand": round(self.ev / big / n, 3) if n else None,
                "adjusted_hands": self.adjusted,
            }

    def state(self) -> dict[str, Any]:
        with self.lock:
            pending = self.channel.pending()
            turn = None
            if pending is not None:
                turn = describe_view(pending[1], pending[0], self.mode)
            return {
                "session": {
                    "id": self.id,
                    "mode": self.mode,
                    "settings": dataclasses.asdict(self.settings),
                    "seed": self.seed,
                    "status": "ended" if self.report else self.status,
                    "message": self.message,
                    "error": self.error,
                    "read_on": self.read_on,
                    "data_dir": str(self.data.root),
                },
                "turn": turn,
                "last_hand": self.last_hand,
                "results": self.results(),
                "log": self.log[-40:],
                "read": self.read if self.read_on else None,
                "report": self.report,
            }

    def end(self, timeout: float = 30.0) -> dict[str, Any]:
        """Stop the session (an unfinished hand is abandoned), write and return the report."""
        with self.lock:
            if self.report is not None:
                return self.report
        self.channel.close()
        if self.thread.is_alive():
            self.thread.join(timeout)
        report = self._build_report()
        if self.hands:
            report["report_file"] = str(self.data.write_report(self.id, report))
        with self.lock:
            self.report = report
            self.status = "ended"
        return report

    def _build_report(self) -> dict[str, Any]:
        with self.lock:
            read = self.read
            report: dict[str, Any] = {
                "session_id": self.id,
                "mode": self.mode,
                "settings": dataclasses.asdict(self.settings),
                "seed": self.seed,
                "started": self.started,
                "ended": time.strftime("%Y-%m-%d %H:%M:%S"),
                "hands_file": str(self.data.hands_path(self.id)) if self.hands else None,
                "results": self.results(),
                "model": read,
                "error": self.error,
            }
            notes = [
                "All-in EV replaces the result of a heads-up all-in before the river by its "
                "expectation over the cards still to come (exact from the flop, a 20,000-board "
                "Monte Carlo preflop), so luck is separated from play.",
            ]
            if self.mode == "pushfold" and read is not None:
                timeline = getattr(self.model, "timeline", [])
                counts: dict[str, int] = {}
                for _, name, _ in timeline:
                    counts[name] = counts.get(name, 0) + 1
                report["model_choices"] = counts
                report["fallbacks"] = sum(f for *_, f in timeline)
                notes.append(
                    "The model's numbers assume you play like its family of push/fold players "
                    "(ranges ordered by equity, with noise); the fit checks test that. The "
                    "checks run after every hand against thresholds calibrated for one look, "
                    "so some fallbacks are false alarms."
                )
            if self.mode == "holdem":
                report["opponents"] = [
                    {
                        "player": pid,
                        **info,
                        "net_bb": bb(self.net_by_player.get(pid, 0), self.big_blind),
                    }
                    for pid, info in self.revealed.items()
                ]
                notes.append(
                    "The hold'em read is Phase 1's profile and archetype classifier, trained on "
                    "bots. Any exploit in hold'em is a heuristic, not a computed exploit."
                )
            report["notes"] = notes
            return report
