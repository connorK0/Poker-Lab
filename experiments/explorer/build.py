"""Build the results explorer: one self-contained HTML page for every phase's results.

    python -m experiments.explorer                     # writes experiments/results/explorer.html
    python -m experiments.explorer --out somewhere.html

The page is deterministic: the same results files and static sources give the same bytes
(no build timestamp, no current commit), so ``tests/test_explorer.py`` can check that the
committed page is current.
"""

from __future__ import annotations

import argparse
import json
import re
from collections.abc import Callable
from pathlib import Path
from typing import Any

from experiments.common import REPO_ROOT, RESULTS_ROOT

STATIC = Path(__file__).resolve().parent / "static"
DEFAULT_OUT = RESULTS_ROOT / "explorer.html"
MARKERS = ("/*__CSS__*/", "/*__JS__*/", "__DATA__")

# Phases without results yet: shown in the navigation so the page grows with the project.
PLANNED = (
    (
        "phase3",
        "Phase 3",
        "General framework: hand histories to exploit recommendations",
        "Import real hand histories, fit priors on real data, and rank exploit recommendations "
        "by expected gain and confidence, starting from an abstracted hold'em game solved "
        "exactly with the Phase 2 tools.",
    ),
    (
        "phase4",
        "Phase 4",
        "Home-game capture",
        "Camera card recognition plus manual action entry into the sealed hand-history store.",
    ),
    (
        "phase5",
        "Phase 5",
        "Live evaluation",
        "Action-prediction accuracy and win rate with variance reduction on real home games.",
    ),
)


class Loader:
    """Reads results JSON (repo-relative paths) and remembers which files were read."""

    def __init__(self, root: Path = REPO_ROOT) -> None:
        self.root = root
        self.read: list[str] = []
        self._cache: dict[str, Any] = {}

    def __call__(self, rel: str) -> Any:
        if rel not in self._cache:
            with open(self.root / rel, encoding="utf-8") as fh:
                self._cache[rel] = json.load(fh)
            self.read.append(rel)
        return self._cache[rel]


def phase_builders() -> list[Callable[[Loader], dict[str, Any]]]:
    # Imported here so that importing this module stays cheap.
    from experiments.explorer import phase1, phase2, phase3

    return [phase1.build, phase2.build, phase3.build]


def payload(load: Loader | None = None) -> dict[str, Any]:
    load = load or Loader()
    phases = [build(load) for build in phase_builders()]
    built = {p["id"] for p in phases}
    for pid, label, title, summary in PLANNED:
        if pid not in built:
            phases.append(
                {
                    "id": pid,
                    "label": label,
                    "title": title,
                    "status": "planned",
                    "summary": summary,
                    "tiles": [],
                    "provenance": [],
                    "sections": [],
                }
            )
    return {
        "title": "Poker opponent modeling: results explorer",
        "intro": (
            "Every experiment in this project, phase by phase. Pick a phase, then a section; "
            "the buttons above each chart switch models, games, sample sizes and simulation "
            "settings. All numbers come from the committed results files listed at the bottom "
            "of each page, measured on held-out data across independent seeds."
        ),
        "phases": phases,
    }


def render(data: dict[str, Any]) -> str:
    template = (STATIC / "template.html").read_text(encoding="utf-8")
    for marker in MARKERS:
        if template.count(marker) != 1:
            raise ValueError(f"template must contain {marker!r} exactly once")
    css = (STATIC / "app.css").read_text(encoding="utf-8")
    js = (STATIC / "app.js").read_text(encoding="utf-8")
    blob = json.dumps(data, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
    blob = blob.replace("</", "<\\/")  # never close the <script> element early
    parts = dict(zip(MARKERS, (css, js, blob), strict=True))
    # One pass, so text inserted for one marker is never scanned for another.
    return re.sub("|".join(map(re.escape, MARKERS)), lambda m: parts[m.group(0)], template)


def build(out: Path = DEFAULT_OUT) -> Path:
    html = render(payload())
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8", newline="\n")
    return out


def refresh(out_dir: Path) -> None:
    """Rebuild the explorer after a run that wrote canonical results (not a --quick run).

    Called by the one-command phase runners. A failure here never loses the experiment's own
    results, which are already written, so it is reported instead of raised.
    """
    if out_dir.resolve().parent != RESULTS_ROOT.resolve() or out_dir.name.endswith("_quick"):
        return
    try:
        path = build()
    except (KeyError, ValueError, TypeError, OSError) as exc:
        print(
            f"WARNING: explorer not rebuilt ({exc!r}); fix it, then: python -m experiments.explorer"
        )
        return
    print(f"explorer: rebuilt {path}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Build the interactive results explorer.")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT, help="output HTML file")
    args = ap.parse_args()
    out = build(args.out)
    size = out.stat().st_size / 1024
    print(f"wrote {out} ({size:,.0f} KB); open it in a browser")


if __name__ == "__main__":
    main()
