"""Figure styling shared by all experiment plots.

Every figure is rendered twice, ``<name>.png`` on a light surface and ``<name>-dark.png`` on a
dark one, so the README can serve the right one with ``<picture>``. Colors are the validated
reference palette (categorical slots in fixed order; light and dark steps checked with the
palette validator: adjacent-pair CVD dE >= 8.4, normal-vision dE >= 19.3). Light-mode aqua,
yellow and magenta are below 3:1 contrast on the surface, so plots that use them carry direct
labels, and the README carries the numbers as tables.

Mark specs: 2 px lines, markers with a surface-colored ring, hairline solid gridlines, muted
axes, text in ink tokens (never series colors), a legend whenever there are 2+ series.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

THEMES: dict[str, dict[str, object]] = {
    "light": {
        "surface": "#fcfcfb",
        "ink": "#0b0b0b",
        "ink2": "#52514e",
        "muted": "#898781",
        "grid": "#e1e0d9",
        "axis": "#c3c2b7",
        "series": ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6"],
        "wash": "#cde2fb",  # sequential blue step 100: shaded reference bands
        "seq": ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"],
        "neutral": "#c3c2b7",
    },
    "dark": {
        "surface": "#1a1a19",
        "ink": "#f0efec",
        "ink2": "#c3c2b7",
        "muted": "#898781",
        "grid": "#2c2c2a",
        "axis": "#383835",
        "series": ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#008300", "#9085e9"],
        "wash": "#253445",  # recessive desaturated blue: reference bands behind marks
        "seq": ["#0d366b", "#104281", "#184f95", "#1c5cab", "#2a78d6", "#5598e7", "#86b6ef"],
        "neutral": "#5d5c58",
    },
}

LINE_WIDTH = 2.0
MARKER_SIZE = 6.5  # points; ~13 px at 150 dpi
DPI = 150


def style_axes(ax: plt.Axes, theme: dict[str, object], *, grid_axis: str = "y") -> None:
    ax.set_facecolor(theme["surface"])
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(theme["axis"])
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(colors=theme["muted"], labelcolor=theme["ink2"], labelsize=8.5, length=3)
    if grid_axis:
        ax.grid(axis=grid_axis, color=theme["grid"], linewidth=0.8, linestyle="-")
        ax.set_axisbelow(True)
    ax.title.set_color(theme["ink"])
    ax.xaxis.label.set_color(theme["ink2"])
    ax.yaxis.label.set_color(theme["ink2"])


def line(ax: plt.Axes, theme: dict[str, object], x, y, color, label=None, *, marker="o", **kw):
    return ax.plot(
        x,
        y,
        color=color,
        linewidth=LINE_WIDTH,
        marker=marker,
        markersize=MARKER_SIZE,
        markeredgecolor=theme["surface"],
        markeredgewidth=1.5,
        solid_capstyle="round",
        solid_joinstyle="round",
        label=label,
        **kw,
    )


def band(ax: plt.Axes, x, lo, hi, color, alpha: float = 0.14) -> None:
    ax.fill_between(x, lo, hi, color=color, alpha=alpha, linewidth=0)


def legend(ax: plt.Axes, theme: dict[str, object], **kw) -> None:
    leg = ax.legend(frameon=False, fontsize=8.5, labelcolor=theme["ink2"], **kw)
    for handle in leg.legend_handles:
        if hasattr(handle, "set_markeredgecolor"):
            handle.set_markeredgecolor(theme["surface"])


def new_figure(theme: dict[str, object], *args, **kwargs) -> tuple[plt.Figure, object]:
    """``plt.subplots`` on the theme's surface. ``save_both`` closes the figure."""
    fig, axes = plt.subplots(*args, **kwargs)
    fig.set_facecolor(theme["surface"])
    return fig, axes


def save_both(
    draw: Callable[[dict[str, object]], plt.Figure], out_dir: str | Path, name: str
) -> list[Path]:
    """Render ``draw(theme)`` for both themes; returns the written paths."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for mode, theme in THEMES.items():
        fig = draw(theme)
        suffix = "" if mode == "light" else "-dark"
        path = out_dir / f"{name}{suffix}.png"
        fig.savefig(path, dpi=DPI, facecolor=theme["surface"], bbox_inches="tight")
        plt.close(fig)
        paths.append(path)
    return paths


def title(
    fig: plt.Figure, theme: dict[str, object], text: str, subtitle: str = "", top: float = 0.92
) -> None:
    """Left-aligned title + subtitle, then lay out the axes below ``top`` (figure fraction)."""
    fig.text(
        0.01, 0.995, text, ha="left", va="top", fontsize=12, color=theme["ink"], fontweight="bold"
    )
    if subtitle:
        fig.text(0.01, 0.955, subtitle, ha="left", va="top", fontsize=9, color=theme["ink2"])
    fig.tight_layout(rect=(0, 0, 1, top))
