"""Environment-specific PhysicalAgent extensions."""

from physical_agent.envs.base import EnvSpec, PromptBundle
from physical_agent.envs.registry import get_env_spec, infer_env_from_suite

__all__ = [
    "EnvSpec",
    "PromptBundle",
    "get_env_spec",
    "infer_env_from_suite",
]
