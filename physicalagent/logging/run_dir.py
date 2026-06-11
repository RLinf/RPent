"""Per-run log directory helpers (RLinf-style naming convention).

Creates ``logs/YYYYmmdd-HH:MM:SS_<tag>/`` directories. The REPL driver
writes its outputs (images/, depths/, states.json, episode.mp4, …)
directly into this directory — there is no separate workdir to gather.
"""
from __future__ import annotations

import datetime
from pathlib import Path

from physicalagent.config import get_repo_root, get_logs_dir


def make_log_dir(
    *,
    suite: str,
    task: int,
    seed: int,
    repo_root: str | Path | None = None,
    timestamp: str | None = None,
) -> Path:
    """Create and return a per-run ``logs/`` directory.

    Returns ``<repo_root>/logs/<YYYYmmdd-HH:MM:SS>_<suite>_t<task>_s<seed>/``.
    The directory is created (``parents=True``).  Identifying metadata is
    encoded in the directory name itself.
    """
    if repo_root is None:
        repo_root = get_repo_root()
    repo_root = Path(repo_root)

    if timestamp is None:
        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H:%M:%S")
    tag = f"{suite}_t{task}_s{seed}"
    run_dir = get_logs_dir() if repo_root == get_repo_root() else repo_root / "logs"
    run_dir = run_dir / f"{timestamp}_{tag}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir