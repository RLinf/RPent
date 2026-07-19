"""Static env-extension descriptor.

Lives in :mod:`rpent.envs` alongside
:class:`~rpent.envs.prompt_bundle.PromptBundle` so envs
and cerebrums can both import it without pulling in
:mod:`rpent.tools` or the RPC transport layer. Tool schemas,
handlers, server lifecycle, and the MCP allowlist live on
:class:`rpent.tools.toolkit.Toolkit` and its env subclasses —
``EnvSpec`` carries only the env identity and the prompt bundle.
"""
from __future__ import annotations

from dataclasses import dataclass

from rpent.envs.prompt_bundle import PromptBundle


@dataclass(frozen=True)
class EnvSpec:
    """Environment-level (non-tool) extension points for RPent."""

    name: str
    prompts: PromptBundle
