"""The reproduction check (``experiments/compare_runs.py``): what it compares and what it skips."""

import json

import numpy as np
from PIL import Image

from experiments.compare_runs import compare_step

RESULT = {
    "meta": {"seed": 2026, "commit": "aaa", "config": {"hands": 100}},
    "elapsed_seconds": 12.5,
    "aggregate": {"gain": [0.1, 0.25], "fallback_second_half": 0.64, "fit": {"seconds": 3.0}},
    "round_trip": {"identical": 20000, "ms_per_hand": 6.5},
}
SUMMARY = """# A step

Seed 2026, commit aaa (dirty: False), 12 s.

| Model | Gain | Seconds |
|---|---|---|
| refit | +0.0011 | 31 |

20,000 of 20,000 round trips identical; 6.5 ms each.

Run time 529 s.
"""


def _write(folder, result, summary, pixels):
    folder.mkdir()
    (folder / "step.json").write_text(json.dumps(result), encoding="utf-8")
    (folder / "step_summary.md").write_text(summary, encoding="utf-8")
    Image.fromarray(pixels).save(folder / "step.png", compress_level=1)


def _pixels():
    return np.arange(48, dtype=np.uint8).reshape(4, 4, 3)


def test_provenance_and_timing_are_skipped(tmp_path):
    _write(tmp_path / "old", RESULT, SUMMARY, _pixels())
    new = json.loads(json.dumps(RESULT))
    new["meta"]["commit"] = "bbb"
    new["elapsed_seconds"] = 99.0
    new["aggregate"]["fit"]["seconds"] = 7.0
    new["round_trip"]["ms_per_hand"] = 6.2
    summary = (
        SUMMARY.replace("commit aaa", "commit bbb")
        .replace("| 31 |", "| 45 |")
        .replace("6.5 ms", "6.2 ms")
        .replace("529 s", "601 s")
    )
    (tmp_path / "new").mkdir()
    (tmp_path / "new" / "step.json").write_text(json.dumps(new), encoding="utf-8")
    (tmp_path / "new" / "step_summary.md").write_text(summary, encoding="utf-8")
    # the same pixels, encoded differently
    old_png, new_png = tmp_path / "old" / "step.png", tmp_path / "new" / "step.png"
    Image.fromarray(_pixels()).save(new_png, compress_level=9)
    assert old_png.read_bytes() != new_png.read_bytes()
    assert compare_step(tmp_path / "old", tmp_path / "new") == []


def test_every_other_difference_is_found(tmp_path):
    _write(tmp_path / "old", RESULT, SUMMARY, _pixels())
    new = json.loads(json.dumps(RESULT))
    new["meta"]["config"]["hands"] = 200
    new["aggregate"]["gain"][1] = 0.25000000000000006
    new["aggregate"]["fallback_second_half"] = 0.65  # data, although its name says "second"
    pixels = _pixels()
    pixels[0, 0, 0] += 1
    new["round_trip"]["identical"] = 19999
    summary = SUMMARY.replace("+0.0011", "+0.0012").replace("20,000 of", "19,999 of")
    _write(tmp_path / "new", new, summary, pixels)
    (tmp_path / "new" / "extra.png").write_bytes((tmp_path / "new" / "step.png").read_bytes())
    found = {(d.file, d.where) for d in compare_step(tmp_path / "old", tmp_path / "new")}
    assert found == {
        ("extra.png", ""),
        ("step.json", ".aggregate.gain[1]"),
        ("step.json", ".aggregate.fallback_second_half"),
        ("step.json", "meta.config"),
        ("step.json", ".round_trip.identical"),
        ("step_summary.md", "line 7"),
        ("step_summary.md", "line 9"),  # a timing on the line does not hide the count
        ("step.png", ""),
    }
    gaps = [d.gap for d in compare_step(tmp_path / "old", tmp_path / "new") if d.gap]
    assert min(gaps) < 1e-15  # floats are compared exactly
