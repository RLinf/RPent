"""Path resolution and environment-variable configuration."""
from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path

# ============================================================================
# Repository / package roots
# ============================================================================

def get_repo_root() -> Path:
    """Return the RPent repository root directory.

    Resolution: ``RPENT_REPO_ROOT`` env var, then the parent of
    the ``rpent/`` package directory.
    """
    env = os.environ.get("RPENT_REPO_ROOT")
    if env:
        return Path(env).expanduser().resolve()
    # config.py lives at <repo>/rpent/utils/config.py
    return Path(__file__).resolve().parents[2]


# ============================================================================
# Paths derived from the repo root  (callable so tests can override)
# ============================================================================

def get_resources_dir(robot_name: str) -> Path:
    """Return the per-robot resources directory (memory + reference corpora)."""
    return get_repo_root() / "resources" / robot_name


def get_memory_dir(robot_name: str) -> Path:
    """Return the persistent, cross-run memory directory for a robot."""
    return get_resources_dir(robot_name) / "memory"


def get_pi05_checkpoint_path() -> str:
    return os.environ.get("PI05_CHECKPOINT_PATH", "")


def get_libero_type() -> str:
    return os.environ.get("LIBERO_TYPE", "pro")


def get_rlinf_repo_path() -> Path | None:
    """Return the configured RLinf checkout path, or *None*."""
    env = os.environ.get("RLINF_REPO_PATH")
    if env:
        return Path(env).expanduser().resolve()
    return None


def build_rpent_subprocess_env(
    *,
    rlinf_root: str | Path | None = None,
    base_env: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Build a subprocess environment with an optional RLinf source override."""
    env = dict(base_env or os.environ)
    root = (
        Path(rlinf_root).expanduser().resolve()
        if rlinf_root
        else get_rlinf_repo_path()
    )
    python_paths = [str(get_repo_root())]
    if root is not None:
        if not (root / "rlinf" / "__init__.py").is_file():
            raise ValueError(
                f"RLinf source override must contain rlinf/__init__.py: {root}"
            )
        env["RLINF_REPO_PATH"] = str(root)
        python_paths.append(str(root))
    existing = env.get("PYTHONPATH")
    if existing:
        python_paths.append(existing)
    env["PYTHONPATH"] = os.pathsep.join(python_paths)
    return env
