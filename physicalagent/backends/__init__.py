"""Helpers for optional external backend dependencies."""

from __future__ import annotations

import os
import sys
from pathlib import Path

from physicalagent.config import get_repo_root, get_rlinf_repo_path


def add_external_rlinf_to_path(project_root: Path | None = None) -> Path:
    """Add the configured external RLinf checkout to ``sys.path``.

    Resolution order:
    1. ``PHYSICALAGENT_RLINF_ROOT`` env var
    2. ``RLINF_REPO_PATH`` env var
    3. sibling checkout named ``rlinf`` next to the PhysicalAgent repo
    """
    if project_root is None:
        project_root = get_repo_root()

    rlinf_path = get_rlinf_repo_path()
    if rlinf_path is None:
        rlinf_path = (project_root.parent / "rlinf").resolve()

    if str(rlinf_path) not in sys.path:
        sys.path.insert(0, str(rlinf_path))
    return rlinf_path