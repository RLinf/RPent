"""Prompt bundle dataclass for env-contributed LLM prompts.

Lives in :mod:`physical_agent.envs` so each env's
``prompt_bundle.py`` (e.g. :mod:`physical_agent.envs.libero.prompt_bundle`)
can import it without depending on the driver-client transport layer.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass

from physical_agent.context.prompt_base import PromptNode, format_prompt

PromptFactory = Callable[..., PromptNode]

@dataclass(frozen=True)
class PromptBundle:
    """Python-defined prompt variants for one environment."""

    api_system: PromptFactory
    api_user: PromptFactory
    cli_system: PromptFactory
    cli_user: PromptFactory

    def render(
        self,
        variant: str,
        *,
        variables: Mapping[str, object] | None = None,
        perception: bool = False,
    ) -> str:
        """Render one prompt variant."""
        prompt = getattr(self, variant)(perception=perception)
        return format_prompt(prompt, variables=variables)

