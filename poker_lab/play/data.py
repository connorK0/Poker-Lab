"""Where your hands, reports and the model's caches live: a local folder outside the repository.

The folder is ``POKER_LAB_PLAY_DIR`` if set, otherwise ``%LOCALAPPDATA%/poker_lab/play`` on
Windows and ``~/.local/share/poker_lab/play`` elsewhere. A folder inside the repository is
refused, so your hands can never be committed by accident. They are your own hands: using them
in a research result needs your OK (the project's real data is the Pluribus hands only).

Layout::

    sessions/<session id>.jsonl         every completed hand, full record (your own cards too)
    sessions/<session id>-report.json   the end-of-session report
    cache/                               calibrated fit-check thresholds, classifier training rows
"""

from __future__ import annotations

import json
import os
import re
import secrets
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from poker_lab.schema import HandHistory, read_jsonl, write_jsonl

REPO_ROOT = Path(__file__).resolve().parents[2]
SESSION_ID = re.compile(r"^(pf|nl)-\d{8}-\d{6}-[0-9a-f]{4}$")


def default_dir() -> Path:
    if env := os.environ.get("POKER_LAB_PLAY_DIR"):
        return Path(env)
    if local := os.environ.get("LOCALAPPDATA"):
        return Path(local) / "poker_lab" / "play"
    return Path.home() / ".local" / "share" / "poker_lab" / "play"


class DataDir:
    def __init__(self, root: str | Path | None = None) -> None:
        self.root = Path(root or default_dir()).resolve()
        if self.root == REPO_ROOT or REPO_ROOT in self.root.parents:
            raise ValueError(f"{self.root} is inside the repository; your hands must stay outside")
        self.sessions = self.root / "sessions"
        self.cache = self.root / "cache"
        self.sessions.mkdir(parents=True, exist_ok=True)
        self.cache.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def new_session_id(mode: str) -> str:
        prefix = {"pushfold": "pf", "holdem": "nl"}[mode]
        return f"{prefix}-{time.strftime('%Y%m%d-%H%M%S')}-{secrets.token_hex(2)}"

    def hands_path(self, session_id: str) -> Path:
        if not SESSION_ID.match(session_id):
            raise ValueError("bad session id")
        return self.sessions / f"{session_id}.jsonl"

    def report_path(self, session_id: str) -> Path:
        return self.hands_path(session_id).with_name(f"{session_id}-report.json")

    def append_hand(self, session_id: str, hand: HandHistory) -> None:
        write_jsonl([hand], self.hands_path(session_id), append=True)

    def write_report(self, session_id: str, report: dict[str, Any]) -> Path:
        path = self.report_path(session_id)
        path.write_text(json.dumps(report, indent=1), encoding="utf-8")
        return path

    def earlier_hands(self, prefix: str, exclude: str | None = None) -> Iterator[HandHistory]:
        """Hands of earlier sessions of one mode, oldest session first, in play order."""
        for path in sorted(self.sessions.glob(f"{prefix}-*.jsonl")):
            if path.stem != exclude and SESSION_ID.match(path.stem):
                yield from read_jsonl(path)

    def read_cache(self, name: str) -> Any | None:
        path = self.cache / name
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))

    def write_cache(self, name: str, value: Any) -> None:
        (self.cache / name).write_text(json.dumps(value), encoding="utf-8")
