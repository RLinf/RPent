"""Env registry: maps env name to its ``get_env_spec`` / ``get_toolkit`` factories.

Env implementations live in the top-level ``robots/`` directory (a sibling of
the ``rpent`` package); an env is resolved by importing ``robots.<name>``. The
``EnvSpec`` / ``PromptBundle`` dataclasses themselves live in :mod:`rpent.envs`
so cerebrums and envs share the same contract types without crossing module
layers.
"""

from __future__ import annotations

import importlib
import sys
from typing import Any

from rpent.envs.env_spec import EnvSpec
from rpent.tools.toolkit import Toolkit
from rpent.utils.config import get_repo_root

# Source checkouts keep env packages under ``<repo>/robots/``. Installed wheels
# package the same namespace, while this path setup preserves checkout execution
# regardless of the process's current working directory.
_REPO_ROOT = str(get_repo_root())
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


def _resolve_env(name: str) -> Any:
    """Import ``robots.<name>`` lazily and return the module."""
    if not name:
        raise ValueError("env name must be non-empty")
    env_name = name.lower()
    try:
        return importlib.import_module(f"robots.{env_name}")
    except ModuleNotFoundError as e:
        raise ValueError(f"unknown env: {env_name!r}") from e


def get_env_spec(name: str) -> EnvSpec:
    """Return the static descriptor exposed by ``robots.<name>``."""
    return _resolve_env(name).get_env_spec()


def get_toolkit(name: str, **kwargs) -> Toolkit:
    """Build the env toolkit (common tools + env-specific tools)."""
    return _resolve_env(name).get_toolkit(**kwargs)


def get_runtime(name: str, **kwargs):
    """Build the lifecycle adapter exposed by ``robots.<name>``."""
    module = _resolve_env(name)
    factory = getattr(module, "get_runtime", None)
    if factory is None:
        raise ValueError(f"environment {name!r} does not expose get_runtime")
    return factory(**kwargs)


def validate_env_args(name: str, args: Any, parser: Any) -> None:
    """Run optional environment-specific CLI validation before side effects."""
    validator = getattr(_resolve_env(name), "validate_args", None)
    if validator is not None:
        validator(args, parser)
