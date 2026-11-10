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
import sys
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from poker_lab.schema import HandHistory, loads_hand, write_jsonl

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
        self.skipped_lines = 0  # unreadable lines met by the last ``earlier_hands``

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
        write_atomic(path, json.dumps(report, indent=1))
        return path

    def earlier_hands(self, prefix: str, exclude: str | None = None) -> Iterator[HandHistory]:
        """Hands of earlier sessions of one mode, oldest session first, in play order.

        A line that does not parse (say, the last one, cut short when the program was killed
        mid-write) is skipped and counted in ``skipped_lines``, so one bad line cannot stop
        every later session."""
        self.skipped_lines = 0
        for path in sorted(self.sessions.glob(f"{prefix}-*.jsonl")):
            if path.stem == exclude or not SESSION_ID.match(path.stem):
                continue
            with path.open(encoding="utf-8", errors="replace") as fh:
                for lineno, line in enumerate(fh, start=1):
                    if not line.strip():
                        continue
                    try:
                        hand = loads_hand(line)
                    except (ValueError, TypeError, KeyError) as exc:
                        self.skipped_lines += 1
                        print(f"skipped {path.name}:{lineno}: {exc}", file=sys.stderr)
                        continue
                    yield hand

    def read_cache(self, name: str) -> Any | None:
        """A cached value, or None if missing or unreadable (then it is recomputed)."""
        path = self.cache / name
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            return None

    def write_cache(self, name: str, value: Any) -> None:
        write_atomic(self.cache / name, json.dumps(value))


def write_atomic(path: Path, text: str) -> None:
    """Write ``text`` to a temporary file next to ``path``, then rename it into place, so a
    reader sees either the old file or the whole new one, never half of it."""
    tmp = path.with_name(f"{path.name}.{os.getpid()}.{secrets.token_hex(3)}.tmp")
    try:
        tmp.write_text(text, encoding="utf-8")
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)
