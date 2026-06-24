"""Static env-extension descriptor.

Lives in :mod:`physical_agent.envs` alongside
:class:`~physical_agent.envs.prompt_bundle.PromptBundle` so envs
and cerebrums can both import it without crossing into
:mod:`physical_agent.envs`. Tool schemas, handlers, driver lifecycle,
and the MCP allowlist live on
:class:`physical_agent.tools.toolkit.Toolkit` and its env subclasses —
``EnvSpec`` carries only the env identity and the prompt bundle.
"""
from __future__ import annotations

from dataclasses import dataclass

from physical_agent.envs.prompt_bundle import PromptBundle


@dataclass(frozen=True)
class EnvSpec:
    """Environment-level (non-tool) extension points for PhysicalAgent."""

    name: str
    prompts: PromptBundle
