"""Small constructors for the explorer's JSON chart specs.

The page renders four chart types from plain dictionaries (see ``static/app.js``):

* ``line``: series of ``[x, y]`` or ``[x, y, lower, upper]`` points; log or linear axes;
  optional bands, dashes, marker-only series and horizontal reference lines.
* ``dots``: categorical rows (one dot per series per row, optional whiskers and an
  intended-range bar behind each row).
* ``heatmap``: a matrix with row and column labels.
* ``scatter``: points per group (optionally connected) or columns of a shared table.

Colors are tokens ``c1``..``c8`` (the validated categorical slots, light and dark) or
``muted``. A color follows its entity everywhere: the families below map names to slots once.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from typing import Any

import numpy as np

from poker_lab.evaluation.intervals import t_interval

ARCHETYPES = ("nit", "tag", "lag", "calling_station", "maniac")
ARCHETYPE_LABELS = {
    "nit": "Nit",
    "tag": "TAG",
    "lag": "LAG",
    "calling_station": "Calling station",
    "maniac": "Maniac",
}
ARCHETYPE_COLORS = {a: f"c{i}" for i, a in enumerate(ARCHETYPES, start=1)}
# Secondary encoding for scatter plots with more than three groups (palette all-pairs cap).
ARCHETYPE_SHAPES = {
    "nit": "circle",
    "tag": "square",
    "lag": "triangle",
    "calling_station": "diamond",
    "maniac": "cross",
}


def sig(x: Any, digits: int = 4) -> float | None:
    """Round to significant digits (keeps the page small); None for missing or non-finite."""
    if x is None:
        return None
    x = float(x)
    if not math.isfinite(x):
        return None
    if x == 0:
        return 0.0
    return float(f"{x:.{digits}g}")


def ci(values: Iterable[float]) -> tuple[float | None, float | None, float | None]:
    """Mean and 95% t-interval across independent replicates (seeds)."""
    iv = t_interval(np.asarray(list(values), dtype=float))
    return sig(iv.estimate), sig(iv.lower), sig(iv.upper)


def axis(
    label: str,
    *,
    scale: str = "linear",
    fmt: str = "num",
    lo: float | None = None,
    hi: float | None = None,
) -> dict[str, Any]:
    """``fmt``: int, num, pct (fraction shown as %), sci, signed (+0.123), x (ratio)."""
    out: dict[str, Any] = {"label": label, "scale": scale, "format": fmt}
    if lo is not None:
        out["min"] = lo
    if hi is not None:
        out["max"] = hi
    return out


def series(
    name: str,
    color: str,
    points: Sequence[Sequence[float | None]],
    *,
    dash: bool = False,
    marker: bool = True,
    line: bool = True,
    hidden: bool = False,
    shape: str = "circle",
    note: str = "",
) -> dict[str, Any]:
    pts = []
    for p in points:
        q = [sig(v) for v in p]
        if q[0] is None or q[1] is None:
            continue
        while len(q) > 2 and q[-1] is None:
            q.pop()
        pts.append(q)
    out: dict[str, Any] = {"name": name, "color": color, "points": pts}
    if dash:
        out["dash"] = True
    if not marker:
        out["marker"] = False
    if not line:
        out["line"] = False
    if hidden:
        out["hidden"] = True
    if shape != "circle":
        out["shape"] = shape
    if note:
        out["note"] = note
    return out


def ref(value: float, label: str, *, axis_name: str = "y") -> dict[str, Any]:
    return {"axis": axis_name, "value": sig(value), "label": label}


def line_chart(
    title: str,
    x: dict[str, Any],
    y: dict[str, Any],
    series_list: list[dict[str, Any]],
    *,
    subtitle: str = "",
    refs: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    return {
        "type": "line",
        "title": title,
        "subtitle": subtitle,
        "x": x,
        "y": y,
        "series": series_list,
        "refs": refs or [],
    }


def dots_chart(
    title: str,
    rows: list[str],
    x: dict[str, Any],
    series_list: list[dict[str, Any]],
    *,
    subtitle: str = "",
    ranges: list[Sequence[float] | None] | None = None,
    range_label: str = "",
    refs: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """``series_list`` items: {name, color, values: [[v, lo, hi] | [v] | None per row]}."""
    return {
        "type": "dots",
        "title": title,
        "subtitle": subtitle,
        "rows": rows,
        "x": x,
        "series": series_list,
        "ranges": ranges,
        "rangeLabel": range_label,
        "refs": refs or [],
    }


def dot_series(
    name: str, color: str, values: Sequence[Sequence[float | None] | None]
) -> dict[str, Any]:
    vals = []
    for v in values:
        if v is None or v[0] is None:
            vals.append(None)
        else:
            vals.append([sig(u) for u in v])
    return {"name": name, "color": color, "values": vals}


def heatmap(
    title: str,
    rows: list[str],
    cols: list[str],
    values: Sequence[Sequence[float]],
    *,
    subtitle: str = "",
    row_label: str = "",
    col_label: str = "",
    fmt: str = "pct",
    vmax: float | None = None,
    counts: Sequence[Sequence[float]] | None = None,
    labels: Sequence[Sequence[str]] | None = None,
) -> dict[str, Any]:
    """``labels``: text drawn in each cell instead of its value (the value moves to the
    tooltip), e.g. hand names on a 13 x 13 chart."""
    out = {
        "type": "heatmap",
        "title": title,
        "subtitle": subtitle,
        "rows": rows,
        "cols": cols,
        "values": [[sig(v) for v in row] for row in values],
        "rowLabel": row_label,
        "colLabel": col_label,
        "format": fmt,
        "max": vmax,
    }
    if counts is not None:
        out["counts"] = [[int(c) for c in row] for row in counts]
    if labels is not None:
        out["labels"] = [list(row) for row in labels]
    return out


def scatter_chart(
    title: str,
    x: dict[str, Any],
    y: dict[str, Any],
    groups: list[dict[str, Any]],
    *,
    subtitle: str = "",
    refs: list[dict[str, Any]] | None = None,
    diagonal: bool = False,
) -> dict[str, Any]:
    """``groups``: from ``scatter_group`` (inline points) or ``table_group`` (shared table)."""
    return {
        "type": "scatter",
        "title": title,
        "subtitle": subtitle,
        "x": x,
        "y": y,
        "groups": groups,
        "refs": refs or [],
        "diagonal": diagonal,
    }


def scatter_group(
    name: str,
    color: str,
    points: Sequence[Sequence[Any]],
    *,
    connect: bool = False,
    dash: bool = False,
    shape: str = "circle",
) -> dict[str, Any]:
    """Points are ``[x, y]`` or ``[x, y, label]``; the label shows in the tooltip."""
    pts = []
    for p in points:
        xv, yv = sig(p[0]), sig(p[1])
        if xv is None or yv is None:
            continue
        pts.append([xv, yv, *p[2:3]])
    out: dict[str, Any] = {"name": name, "color": color, "points": pts}
    if connect:
        out["connect"] = True
    if dash:
        out["dash"] = True
    if shape != "circle":
        out["shape"] = shape
    return out


def table_group(
    name: str,
    color: str,
    table: str,
    x: str,
    y: str,
    *,
    where: tuple[str, str] | None = None,
    label: str = "label",
    shape: str = "circle",
) -> dict[str, Any]:
    """Points read from a section-level table: columns ``x`` and ``y``, rows where
    ``table[where[0]] == where[1]``."""
    out: dict[str, Any] = {
        "name": name,
        "color": color,
        "table": table,
        "x": x,
        "y": y,
        "label": label,
    }
    if where:
        out["where"] = list(where)
    if shape != "circle":
        out["shape"] = shape
    return out


def tile(label: str, value: str, note: str = "") -> dict[str, str]:
    return {"label": label, "value": value, "note": note}


def control(cid: str, label: str, options: Sequence[tuple[str, str]]) -> dict[str, Any]:
    return {
        "id": cid,
        "label": label,
        "options": [{"value": v, "label": lab} for v, lab in options],
    }


def section(
    sid: str,
    title: str,
    lead: str,
    *,
    sources: list[str],
    controls: list[dict[str, Any]] | None = None,
    views: dict[str, dict[str, Any]],
    takeaways: list[str] | None = None,
    method: str = "",
    tables: dict[str, dict[str, list[Any]]] | None = None,
    post_hoc: bool = False,
    defaults: dict[str, str] | None = None,
) -> dict[str, Any]:
    """``views`` maps the selected control values, joined by ``|`` in control order, to
    ``{"charts": [...], "notes": [...]}``. A section without controls has the single key ``""``.
    Every combination of control values must have a view. ``defaults`` overrides the initial
    value of some controls (otherwise the first option).
    """
    controls = [dict(c) for c in controls or []]
    for c in controls:
        c["default"] = (defaults or {}).get(c["id"], c["options"][0]["value"])
    combos = [""]
    for c in controls:
        combos = [f"{k}|{o['value']}" if k else o["value"] for k in combos for o in c["options"]]
    missing = [k for k in combos if k not in views]
    if missing:
        raise ValueError(f"{sid}: no view for control values {missing[:3]}")
    for key, view in views.items():
        parts = key.split("|") if controls else [key]
        if controls and len(parts) != len(controls):
            raise ValueError(f"{sid}: view key {key!r} does not match {len(controls)} controls")
        if not controls and key != "":
            raise ValueError(f"{sid}: a section without controls has one view, keyed ''")
        if not view.get("charts"):
            raise ValueError(f"{sid}: view {key!r} has no charts")
    return {
        "id": sid,
        "title": title,
        "lead": lead,
        "controls": controls,
        "views": views,
        "takeaways": takeaways or [],
        "method": method,
        "tables": tables or {},
        "sources": sources,
        "postHoc": post_hoc,
    }


def view(*charts: dict[str, Any], notes: Sequence[str] = ()) -> dict[str, Any]:
    return {"charts": list(charts), "notes": list(notes)}


def provenance(path: str, data: Any) -> dict[str, Any]:
    """Where a results file came from, as recorded in its ``meta`` block."""
    meta = data.get("meta", {})
    git = meta.get("git") or {}
    out = {
        "file": path,
        "seed": meta.get("seed"),
        "commit": (git.get("commit") or "")[:7],
        "dirty": git.get("dirty"),
    }
    if meta.get("simulation_fingerprint"):
        out["fingerprint"] = meta["simulation_fingerprint"]
    return out
