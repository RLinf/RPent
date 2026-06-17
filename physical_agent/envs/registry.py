"""Environment registry and suite-to-env inference."""
from __future__ import annotations

from importlib import import_module

from physical_agent.envs.base import EnvSpec

_ENV_MODULES = {
    "libero": "physical_agent.envs.libero",
}


def get_env_spec(name: str | None = None) -> EnvSpec:
    env_name = (name or "libero").lower()
    module_name = _ENV_MODULES.get(env_name)
    if module_name is None:
        known = ", ".join(sorted(_ENV_MODULES))
        raise ValueError(f"unknown env: {env_name!r}; known envs: {known}")
    module = import_module(module_name)
    return module.get_env_spec()


def infer_env_from_suite(suite: str | None) -> str:
    if suite and suite.startswith("libero_"):
        return "libero"
    return "libero"
