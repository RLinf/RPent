# Copyright 2026 The RPent Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Path resolution and environment-variable configuration."""

from __future__ import annotations

import os
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
    env = os.environ.get("RPENT_RLINF_ROOT") or os.environ.get("RLINF_REPO_PATH")
    if env:
        return Path(env).expanduser().resolve()
    return None
