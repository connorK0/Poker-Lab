"""Public hand-history data used by the experiments, fetched once at a pinned commit.

The PHH dataset (Kim 2024; github.com/uoftcprg/phh-dataset, MIT license) collects hands in the
PHH format. Only research and broadcast subsets are used: the 10,000 hands of Pluribus against
professionals (Brown and Sandholm 2019, supplementary data), the 83 televised hands of the 2023
WSOP Event #43 final table, and five historical hands. The repository's HandHQ subset was
scraped from online poker sites and is never fetched (offline-research constraint).

A sparse, blob-less git clone fetches only those paths (about 12 MB). It lives outside the
project folder (``POKER_LAB_DATA``, default ``%LOCALAPPDATA%/poker_lab/data`` on Windows or
``~/.cache/poker_lab/data``), so the 10,088 small files are neither committed nor synced.
Results record the commit and a SHA-256 over every file used.
"""

from __future__ import annotations

import hashlib
import os
import subprocess
from collections.abc import Iterable
from pathlib import Path

PHH_DATASET_URL = "https://github.com/uoftcprg/phh-dataset.git"
PHH_DATASET_COMMIT = "e47fbd5816372360bade4de5d712346fe1bb70f6"
#: Sparse-checkout paths: Pluribus and WSOP; cone mode adds the files directly in ``data/``
#: (the historical hands). Never ``data/handhq``.
PHH_DATASET_PATHS = ("data/pluribus", "data/wsop")


def data_home() -> Path:
    if env := os.environ.get("POKER_LAB_DATA"):
        return Path(env)
    if local := os.environ.get("LOCALAPPDATA"):
        return Path(local) / "poker_lab" / "data"
    return Path.home() / ".cache" / "poker_lab" / "data"


def _git(*args: str, cwd: Path | None = None) -> str:
    out = subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True, check=True)
    return out.stdout.strip()


def phh_dataset(commit: str = PHH_DATASET_COMMIT) -> Path:
    """The dataset's ``data`` directory at ``commit``, cloning it on first use."""
    repo = data_home() / "phh-dataset"
    if not (repo / ".git").exists():
        repo.parent.mkdir(parents=True, exist_ok=True)
        _git("clone", "-q", "--filter=blob:none", "--no-checkout", "--sparse", PHH_DATASET_URL,
             str(repo))  # fmt: skip
        _git("sparse-checkout", "set", *PHH_DATASET_PATHS, cwd=repo)
    if _git("rev-parse", "HEAD", cwd=repo) != commit:
        _git("fetch", "-q", "origin", commit, cwd=repo)
        _git("-c", "advice.detachedHead=false", "checkout", "-q", commit, cwd=repo)
    if _git("rev-parse", "HEAD", cwd=repo) != commit:
        raise RuntimeError(f"{repo} is not at commit {commit}")
    return repo / "data"


def content_sha256(files: Iterable[tuple[Path, str]]) -> str:
    """SHA-256 over (key, SHA-256 of the bytes) of every file, in key order."""
    total = hashlib.sha256()
    for path, key in sorted(files, key=lambda fk: fk[1]):
        total.update(key.encode())
        total.update(hashlib.sha256(path.read_bytes()).digest())
    return total.hexdigest()
