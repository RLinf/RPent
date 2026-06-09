"""Helpers for optional external backend dependencies."""

from __future__ import annotations

import os
import sys
from pathlib import Path


def add_external_rlinf_to_path(project_root: Path | None = None) -> Path:
    """Add the configured external RLinf checkout to ``sys.path``.

    Resolution order:
    1. ``PHYSICALAGENT_RLINF_ROOT``
    2. ``RLINF_REPO_PATH``
    3. sibling checkout named ``rlinf`` next to the PhysicalAgent repo
    """
    if project_root is None:
        project_root = Path(__file__).resolve().parents[1]

    default_path = project_root.parent / "rlinf"
    rlinf_path = Path(
        os.environ.get(
            "PHYSICALAGENT_RLINF_ROOT",
            os.environ.get("RLINF_REPO_PATH", str(default_path)),
        )
    ).expanduser().resolve()

    if str(rlinf_path) not in sys.path:
        sys.path.insert(0, str(rlinf_path))
    return rlinf_path
