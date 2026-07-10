"""Env registry: maps env name to its ``get_env_spec`` / ``get_toolkit`` factories.

The ``EnvSpec`` / ``PromptBundle`` dataclasses themselves live in
:mod:`rpent.envs` so cerebrums and envs share the same contract
types without crossing module layers.
"""
from __future__ import annotations

import importlib
from typing import Any

from rpent.envs.env_spec import EnvSpec
from rpent.tools.toolkit import Toolkit


def _resolve_env(name: str) -> Any:
    """Import ``rpent.envs.<name>`` lazily and return the module."""
    if not name:
        raise ValueError("env name must be non-empty")
    env_name = name.lower()
    try:
        return importlib.import_module(f"rpent.envs.{env_name}")
    except ModuleNotFoundError as e:
        raise ValueError(f"unknown env: {env_name!r}") from e


def get_env_spec(name: str) -> EnvSpec:
    return _resolve_env(name).get_env_spec()


def get_toolkit(name: str, **kwargs) -> Toolkit:
    """Build the env toolkit (common tools + env-specific tools)."""
    return _resolve_env(name).get_toolkit(**kwargs)
