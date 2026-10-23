"""Smoke-test the built explorer in headless Chrome or Edge (a development check, not pytest).

    python -m experiments.explorer.browser_check

Injects ``static/smoke.js`` into a temporary copy of the page, which visits every view of every
section and hovers, toggles and tabulates every chart, then reads the JSON report it writes
into the DOM. Exits non-zero on any failure. Needs a Chromium-based browser installed;
``pytest`` covers the data side without one (``tests/test_explorer.py``).
"""

from __future__ import annotations

import html as html_lib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from experiments.explorer.build import DEFAULT_OUT, STATIC

CANDIDATES = (
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
)


def find_browser() -> str | None:
    for name in ("google-chrome", "chromium", "chromium-browser", "chrome", "msedge"):
        path = shutil.which(name)
        if path:
            return path
    return next((c for c in CANDIDATES if os.path.exists(c)), None)


def run(page: Path = DEFAULT_OUT) -> dict:
    browser = find_browser()
    if browser is None:
        raise SystemExit("no Chrome/Chromium/Edge found")
    html = page.read_text(encoding="utf-8")
    smoke = (STATIC / "smoke.js").read_text(encoding="utf-8")
    html = html.replace("</body>", f"<script>{smoke}</script>\n</body>", 1)
    with tempfile.TemporaryDirectory() as tmp:
        test_page = Path(tmp) / "explorer_smoke.html"
        test_page.write_text(html, encoding="utf-8")
        out = subprocess.run(
            [
                browser,
                "--headless=new",
                "--disable-gpu",
                "--window-size=1400,1000",
                "--virtual-time-budget=120000",
                "--dump-dom",
                test_page.as_uri(),
            ],
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=600,
        ).stdout
    found = re.findall(r'<pre id="smoke-result">(.*?)</pre>', out, re.S)
    if not found:
        raise SystemExit("the smoke script did not report (did the page fail to load?)")
    return json.loads(html_lib.unescape(found[-1]))


def main() -> None:
    report = run()
    failures = report.pop("failures")
    # A scatter hovered at its center shows nothing when no point is within reach: expected.
    quiet = report.pop("noTooltip", [])
    print(", ".join(f"{k} {v}" for k, v in report.items()))
    if quiet:
        print(f"{len(quiet)} hovers found no point near the chart center, e.g. {quiet[0]}")
    for f in failures[:40]:
        print("FAIL", f)
    if failures:
        print(f"{len(failures)} failures")
        sys.exit(1)
    print("ok")


if __name__ == "__main__":
    main()
