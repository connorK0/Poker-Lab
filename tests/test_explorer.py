"""The results explorer: coverage of every results file, well-formed views, a current page.

The browser side (every view renders, hovers, legend toggles, tables) is checked separately
with ``python -m experiments.explorer.browser_check``, which needs Chrome or Edge.
"""

import json
import math
from itertools import product

import pytest

from experiments.common import REPO_ROOT, RESULTS_ROOT
from experiments.explorer import build as build_module
from experiments.explorer.build import DEFAULT_OUT, Loader, payload, render

COLORS = {f"c{i}" for i in range(1, 9)} | {"muted"}
CHART_TYPES = {"line", "dots", "heatmap", "scatter"}
# Results files that need no section of their own, with the reason.
EXEMPT = {
    "per_seed": "per-seed copies of numbers already in phase1_results.json",
}


@pytest.fixture(scope="module")
def built():
    load = Loader()
    return payload(load), load


def _results_files():
    for path in sorted(RESULTS_ROOT.glob("phase*/**/*.json")):
        rel = path.relative_to(RESULTS_ROOT).parts
        if rel[0].endswith("_quick") or any(part in EXEMPT for part in rel):
            continue
        yield path.relative_to(REPO_ROOT).as_posix()


def _sections(data):
    for phase in data["phases"]:
        for sec in phase["sections"]:
            yield phase, sec


def test_every_results_file_feeds_a_section(built):
    """New experiments must add an explorer section (the standing rule for every phase)."""
    data, load = built
    cited = {src for _, sec in _sections(data) for src in sec["sources"]}
    files = list(_results_files())
    assert files, "no results found"
    missing = [f for f in files if f not in cited or f not in load.read]
    assert not missing, f"results without an explorer section: {missing}"


def test_done_phases_have_sections_and_planned_phases_do_not(built):
    data, _ = built
    ids = [p["id"] for p in data["phases"]]
    assert len(ids) == len(set(ids))
    for phase in data["phases"]:
        if phase["status"] == "done":
            assert phase["sections"]
            assert phase["tiles"]
            assert phase["provenance"]
        else:
            assert not phase["sections"]


def _finite(v):
    return isinstance(v, (int, float)) and math.isfinite(v)


def _check_axis(ax, values, where):
    assert ax["label"], where
    if ax["scale"] == "log":
        bad = [v for v in values if v <= 0]
        assert not bad, f"{where}: non-positive values on a log axis"
    if ax["scale"] == "point":
        assert all(0 <= v < len(ax["categories"]) for v in values), where


def _check_chart(chart, tables, where):
    assert chart["type"] in CHART_TYPES, where
    assert chart["title"], where
    if chart["type"] == "line":
        assert chart["series"], where
        xs, ys = [], []
        for s in chart["series"]:
            assert s["color"] in COLORS, where
            assert s["points"], f"{where}: empty series {s['name']}"
            for p in s["points"]:
                assert len(p) in (2, 4), f"{where}: {p}"
                assert all(_finite(v) for v in p), f"{where}: {p}"
                xs.append(p[0])
                ys.extend(p[1:2] if len(p) == 2 else [p[1]])
                if len(p) == 4:
                    assert p[2] <= p[1] + 1e-9 <= p[3] + 2e-9, f"{where}: CI excludes the mean"
        _check_axis(chart["x"], xs, where)
        _check_axis(chart["y"], ys, where)
    elif chart["type"] == "dots":
        for s in chart["series"]:
            assert len(s["values"]) == len(chart["rows"]), where
            vals = [v[0] for v in s["values"] if v]
            assert vals, where
            assert all(_finite(v) for v in vals), where
            _check_axis(chart["x"], vals, where)
        if chart["ranges"] is not None:
            assert len(chart["ranges"]) == len(chart["rows"]), where
    elif chart["type"] == "heatmap":
        assert len(chart["values"]) == len(chart["rows"]), where
        assert all(len(r) == len(chart["cols"]) for r in chart["values"]), where
    elif chart["type"] == "scatter":
        for g in chart["groups"]:
            assert g["color"] in COLORS, where
            if "table" in g:
                t = tables[g["table"]]
                for col in (g["x"], g["y"], g["label"], *(g.get("where") or [])[:1]):
                    assert col in t, f"{where}: missing column {col}"
            else:
                assert g["points"], where


def test_every_control_combination_has_well_formed_views(built):
    data, _ = built
    for phase, sec in _sections(data):
        for t in sec["tables"].values():
            assert len({len(col) for col in t.values()}) == 1, sec["id"]
        options = [[o["value"] for o in c["options"]] for c in sec["controls"]]
        keys = ["|".join(k) for k in product(*options)] if options else [""]
        assert set(keys) == set(sec["views"]), sec["id"]
        for c in sec["controls"]:
            assert c["default"] in [o["value"] for o in c["options"]]
        for key in keys:
            for i, chart in enumerate(sec["views"][key]["charts"]):
                _check_chart(chart, sec["tables"], f"{phase['id']}/{sec['id']}[{key}] #{i}")
        assert sec["takeaways"], sec["id"]
        assert all(s.startswith("experiments/results/") for s in sec["sources"])


def test_committed_page_is_current(built):
    data, _ = built
    assert DEFAULT_OUT.exists(), "run: python -m experiments.explorer"
    committed = DEFAULT_OUT.read_text(encoding="utf-8")
    assert committed == render(data), (
        "experiments/results/explorer.html is stale: run python -m experiments.explorer"
    )


def test_build_is_deterministic(built):
    data, _ = built
    again = payload(Loader())
    assert json.dumps(again, sort_keys=True) == json.dumps(data, sort_keys=True)


def test_refresh_rebuilds_only_after_canonical_runs(monkeypatch):
    calls = []
    monkeypatch.setattr(build_module, "build", lambda: calls.append(1) or DEFAULT_OUT)
    build_module.refresh(RESULTS_ROOT / "phase2_quick")
    build_module.refresh(REPO_ROOT / "elsewhere" / "phase2")
    assert not calls
    build_module.refresh(RESULTS_ROOT / "phase2")
    assert calls == [1]


def test_render_never_closes_the_data_script_early():
    data = {"phases": [], "title": "</script><script>alert(1)</script>", "intro": ""}
    html = render(data)
    blob = html.split('id="explorer-data">', 1)[1].split("</script>", 1)[0]
    assert json.loads(blob)["title"] == data["title"]
