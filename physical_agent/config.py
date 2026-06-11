"""Centralised path resolution and environment-variable configuration.

Every other module should import its paths and defaults from here instead
of computing ``Path(__file__).resolve().parents[N]`` ad-hoc.
"""
from __future__ import annotations

import os
import tempfile
from pathlib import Path


# ============================================================================
# Repository / package roots
# ============================================================================

def get_repo_root() -> Path:
    """Return the PhysicalAgent repository root directory.

    Resolution: ``PHYSICALAGENT_REPO_ROOT`` env var, then the parent of
    the ``physical_agent/`` package directory.
    """
    env = os.environ.get("PHYSICALAGENT_REPO_ROOT")
    if env:
        return Path(env).expanduser().resolve()
    # config.py lives at <repo>/physical_agent/config.py
    return Path(__file__).resolve().parents[1]


def get_package_root() -> Path:
    """Return the ``physical_agent/`` package directory."""
    return Path(__file__).resolve().parent


# ============================================================================
# Paths derived from the repo root  (callable so tests can override)
# ============================================================================

def get_memory_dir() -> Path:
    return get_repo_root() / "logs" / "memory"


def get_guides_dir() -> Path:
    return get_repo_root() / "physical_agent" / "context" / "guides"


def get_backends_dir() -> Path:
    return get_repo_root() / "physical_agent" / "backends" / "rlinf"


def get_repl_driver_script() -> Path:
    return get_backends_dir() / "repl_driver.py"


def get_logs_dir() -> Path:
    return get_repo_root() / "logs"


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
    default = Path(tempfile.gettempdir()) / "physical_agent_repl"
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


# ============================================================================
# OpenAI-compatible API  (env-only — no hard-coded fallback for secrets)
# ============================================================================

def get_openai_compat_api_key() -> str | None:
    """Return the API key for OpenAI-compatible chat-completion providers."""
    return os.environ.get("OPENAI_COMPAT_API_KEY") or os.environ.get("OPENAI_API_KEY")


def get_openai_compat_base_url() -> str | None:
    """Return the optional base URL for an OpenAI-compatible provider."""
    return os.environ.get("OPENAI_COMPAT_BASE_URL") or os.environ.get("OPENAI_BASE_URL")


def get_openai_compat_model() -> str:
    """Return the default model id for the OpenAI-compatible backend."""
    return os.environ.get("OPENAI_COMPAT_MODEL", "gpt-4.1")


def get_openai_compat_supports_images() -> bool:
    """Return whether OpenAI-compatible tool results should include images."""
    value = os.environ.get("OPENAI_COMPAT_SUPPORTS_IMAGES", "1")
    return value.lower() not in {"0", "false", "no", "off"}
