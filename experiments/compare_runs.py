"""Compare a reproduction run with the committed results, file by file.

    python -m experiments.compare_runs experiments/results/phase3 <rerun>/phase3 [steps...]

The protocol of build log Step 24. JSON results are compared field by field, skipping the
provenance block (``meta``, apart from its config, which must match) and wall-clock fields (keys
with the word seconds, elapsed, time or ms); every other difference is listed, floats exactly.
Markdown summaries are compared line by line, apart from provenance lines (``Seed ..., commit
...``, ``Run time ...``), the cells of timing columns in tables and times in milliseconds
(``6.5 ms each``). PNG figures are compared byte for byte, then pixel by pixel. A reproduction
passes when nothing is listed.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

# whole words only: "fallback_second_half" is data, "elapsed_seconds" is not
CLOCK = {"seconds", "elapsed", "time", "timing", "duration", "timestamp", "ms"}
PROVENANCE = ("Seed ", "Run time ")  # summary lines with the commit or the run time
MILLISECONDS = re.compile(r"\d[\d,.]*(?= ms\b)")  # "6.5 ms each": a timing inside a sentence


@dataclass(frozen=True)
class Difference:
    file: str
    where: str
    what: str
    gap: float | None = None  # |old - new| for two floats


def _clock(key: str) -> bool:
    return bool(CLOCK & set(key.lower().replace("_", " ").split()))


def _walk(a: Any, b: Any, where: str, out: list[tuple[str, str, float | None]]) -> None:
    if isinstance(a, dict) and isinstance(b, dict):
        for k in sorted(set(a) | set(b)):
            if (where == "" and k == "meta") or _clock(k):
                continue
            if k not in a or k not in b:
                out.append((f"{where}.{k}", "only in " + ("new" if k not in a else "old"), None))
            else:
                _walk(a[k], b[k], f"{where}.{k}", out)
    elif isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            out.append((where, f"length {len(a)} vs {len(b)}", None))
        else:
            for i, (x, y) in enumerate(zip(a, b, strict=True)):
                _walk(x, y, f"{where}[{i}]", out)
    elif isinstance(a, float) and isinstance(b, float):
        if not (a == b or (math.isnan(a) and math.isnan(b))):
            out.append((where, f"{a!r} vs {b!r}", abs(a - b)))
    elif a != b:
        out.append((where, f"{str(a)[:80]!r} vs {str(b)[:80]!r}", None))


def compare_json(old: Path, new: Path) -> list[Difference]:
    a, b = (json.loads(p.read_text(encoding="utf-8")) for p in (old, new))
    found: list[tuple[str, str, float | None]] = []
    _walk(a, b, "", found)
    if a.get("meta", {}).get("config") != b.get("meta", {}).get("config"):
        found.append(("meta.config", "the configs differ", None))
    return [Difference(old.name, w, what, gap) for w, what, gap in found]


def _cells(row: str) -> list[str]:
    return [c.strip() for c in row.strip().strip("|").split("|")]


def compare_summary(old: Path, new: Path) -> list[Difference]:
    """Line by line, apart from provenance lines, timing columns and times in milliseconds."""
    a, b = (p.read_text(encoding="utf-8").splitlines() for p in (old, new))
    if len(a) != len(b):
        return [Difference(old.name, "", f"{len(a)} lines vs {len(b)}")]
    out, clock_cols = [], set()
    for i, (x, y) in enumerate(zip(a, b, strict=True)):
        if x.startswith("|") and (i == 0 or not a[i - 1].startswith("|")):  # a table header
            clock_cols = {k for k, c in enumerate(_cells(x)) if _clock(c)}
        elif not x.startswith("|"):
            clock_cols = set()
        if x == y or (x.startswith(PROVENANCE) and y.startswith(PROVENANCE)):
            continue
        if MILLISECONDS.sub("#", x) == MILLISECONDS.sub("#", y):
            continue
        cx, cy = _cells(x), _cells(y)
        if clock_cols and len(cx) == len(cy):
            same = [c == d for c, d in zip(cx, cy, strict=True)]
            if all(s for k, s in enumerate(same) if k not in clock_cols):
                continue
        out.append(Difference(old.name, f"line {i + 1}", f"{x[:100]!r} vs {y[:100]!r}"))
    return out


def compare_png(old: Path, new: Path) -> list[Difference]:
    if old.read_bytes() == new.read_bytes():
        return []
    from PIL import Image

    a, b = (np.asarray(Image.open(p)) for p in (old, new))
    if a.shape == b.shape and np.array_equal(a, b):
        return []
    return [Difference(old.name, "", "the pixels differ")]


COMPARE = {".json": compare_json, ".md": compare_summary, ".png": compare_png}


def compare_step(old_dir: Path, new_dir: Path) -> list[Difference]:
    """Every difference between the files of one step, including files only one run has."""
    old = {p.name for p in old_dir.iterdir() if p.suffix in COMPARE}
    new = {p.name for p in new_dir.iterdir() if p.suffix in COMPARE}
    out = [Difference(n, "", "missing from the new run") for n in sorted(old - new)]
    out += [Difference(n, "", "only in the new run") for n in sorted(new - old)]
    for name in sorted(old & new):
        out += COMPARE[Path(name).suffix](old_dir / name, new_dir / name)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("old", type=Path, help="committed results, e.g. experiments/results/phase3")
    ap.add_argument("new", type=Path, help="the reproduction run's folder")
    ap.add_argument("steps", nargs="*", help="steps to compare (default: all in the new run)")
    args = ap.parse_args()
    steps = args.steps or sorted(p.name for p in args.new.iterdir() if p.is_dir())
    total = 0
    for step in steps:
        diffs = compare_step(args.old / step, args.new / step)
        total += len(diffs)
        print(f"{step}: {len(diffs)} differences")
        for d in diffs[:10]:
            gap = "" if d.gap is None else f" (gap {d.gap:.3g})"
            print(f"  {d.file} {d.where}: {d.what}{gap}")
    print(f"total: {total} differences outside provenance and timing")
    raise SystemExit(1 if total else 0)


if __name__ == "__main__":
    main()
