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
from argparse import ArgumentParser, Namespace
from pathlib import Path


def add_cuda_device_arg(parser: ArgumentParser, *, device_type: type = int) -> None:
    """Retain the shared GPU default for existing RPent commands."""
    parser.add_argument(
        "--cuda-device",
        type=device_type,
        help="Default GPU for local services; per-service CUDA options override it. "
        "Services may share GPUs. Omit to use each service's environment/default.",
    )


def resolve_cuda_device(args: Namespace, component: str) -> str | int | None:
    """Prefer the service-specific device over the legacy shared default."""
    device = getattr(args, f"{component}_cuda_device", None)
    return device if device is not None else getattr(args, "cuda_device", None)


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


def get_memory_dir(robot_name: str) -> Path:
    """Return the persistent, cross-run memory directory for a robot."""
    return get_repo_root() / "memory" / robot_name


def get_pi05_checkpoint_path() -> str:
    return os.environ.get("PI05_CHECKPOINT_PATH", "")


def get_libero_type() -> str:
    return os.environ.get("LIBERO_TYPE", "pro")


def get_rlinf_repo_path() -> Path:
    """Return the RLinf path that RPC servers import."""
    env = os.environ.get("RPENT_RLINF_ROOT") or os.environ.get("RLINF_REPO_PATH")
    if env:
        return Path(env).expanduser().resolve()
    return (get_repo_root().parent / "rlinf").resolve()
