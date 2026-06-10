"""Per-run log directory helpers (RLinf-style naming convention).

Creates ``logs/YYYYmmdd-HH:MM:SS_<tag>/`` directories and provides
utilities to gather traces, transcripts, recipes, audits, images, and
videos into a unified run folder.
"""
from __future__ import annotations

import datetime
import json
import shutil
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
    The directory is created (``parents=True``) and a ``run.json`` stub is
    written with the identifying metadata.
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

    # Write a lightweight run manifest so other tools can discover runs.
    manifest = {
        "suite": suite,
        "task": task,
        "seed": seed,
        "timestamp": timestamp,
        "run_dir": str(run_dir),
    }
    (run_dir / "run.json").write_text(json.dumps(manifest, indent=2))
    return run_dir


def gather_workdir_into(
    run_dir: str | Path,
    workdir: str | Path,
    symlink: bool = False,
) -> None:
    """Copy (or symlink) the REPL workdir contents into ``run_dir/repl/``.

    Called after the agent loop finishes so every state, log, image, and
    depth file is preserved alongside the transcript, recipe, and audit.
    """
    run_dir = Path(run_dir)
    workdir = Path(workdir)
    if not workdir.exists():
        return

    dest = run_dir / "repl"
    dest.mkdir(parents=True, exist_ok=True)

    for item in workdir.iterdir():
        src = workdir / item.name
        dst = dest / item.name
        if symlink:
            if dst.exists() or dst.is_symlink():
                dst.unlink()
            dst.symlink_to(src.resolve())
        else:
            if src.is_dir():
                if dst.exists():
                    shutil.rmtree(dst)
                shutil.copytree(src, dst)
            else:
                shutil.copy2(src, dst)