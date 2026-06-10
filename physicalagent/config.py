"""Centralised path resolution and environment-variable configuration.

Every other module should import its paths and defaults from here instead
of computing ``Path(__file__).resolve().parents[N]`` ad-hoc.
"""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path


# ============================================================================
# Repository / package roots
# ============================================================================

def get_repo_root() -> Path:
    """Return the PhysicalAgent repository root directory.

    Resolution: ``PHYSICALAGENT_REPO_ROOT`` env var, then the parent of
    the ``physicalagent/`` package directory.
    """
    env = os.environ.get("PHYSICALAGENT_REPO_ROOT")
    if env:
        return Path(env).expanduser().resolve()
    # config.py lives at <repo>/physicalagent/config.py
    return Path(__file__).resolve().parents[1]


def get_package_root() -> Path:
    """Return the ``physicalagent/`` package directory."""
    return Path(__file__).resolve().parent


# ============================================================================
# Paths derived from the repo root  (callable so tests can override)
# ============================================================================

def get_memory_dir() -> Path:
    return get_repo_root() / "physicalagent" / "context" / "memory"


def get_guides_dir() -> Path:
    return get_repo_root() / "physicalagent" / "context" / "guides"


def get_backends_dir() -> Path:
    return get_repo_root() / "physicalagent" / "backends" / "rlinf"


def get_repl_driver_script() -> Path:
    return get_backends_dir() / "repl_driver.py"


def get_logs_dir() -> Path:
    return get_repo_root() / "logs"


# ============================================================================
# External tool / environment paths  (env var with sensible defaults)
# ============================================================================

def get_python_bin() -> str:
    """Python interpreter that can load RLinf + LIBERO + OpenPI."""
    return os.environ.get("PYTHON_BIN", sys.executable)


def get_pi05_checkpoint_path() -> str:
    return os.environ.get("PI05_CHECKPOINT_PATH", "")


def get_libero_type() -> str:
    return os.environ.get("LIBERO_TYPE", "pro")


def get_cuda_device() -> str:
    return os.environ.get("CUDA_DEVICE", "0")


def get_rlinf_repo_path() -> Path | None:
    """Return the configured RLinf checkout path, or *None*."""
    env = os.environ.get("PHYSICALAGENT_RLINF_ROOT") or os.environ.get("RLINF_REPO_PATH")
    if env:
        return Path(env).expanduser().resolve()
    return None


def get_default_workdir_prefix() -> str:
    """Default prefix for REPL working directories."""
    default = Path(tempfile.gettempdir()) / "physicalagent_repl"
    return os.environ.get("PHYSICALAGENT_WORKDIR_PREFIX", str(default))


# ============================================================================
# Anthropic API  (env-only — no hard-coded fallback for secrets)
# ============================================================================

def get_anthropic_api_key() -> str | None:
    return os.environ.get("ANTHROPIC_API_KEY")


def get_anthropic_base_url() -> str | None:
    return os.environ.get("ANTHROPIC_BASE_URL")


def get_anthropic_model() -> str:
    return os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-5")