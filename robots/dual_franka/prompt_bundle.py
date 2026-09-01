"""Prompt bundle assembly for the dual-Franka environment."""

from __future__ import annotations

from collections.abc import Mapping

from robots.dual_franka.prompts import system as system_parts
from robots.dual_franka.prompts import user as user_parts
from rpent.prompt.utils import Numbered, PromptNode


def system_prompt(variables: Mapping[str, object] | None = None) -> PromptNode:
    """Assemble the dual-Franka system prompt."""
    return {
        "ROLE": system_parts.ROLE,
        "RUNTIME": system_parts.RUNTIME,
        "SAFETY RULES": Numbered(system_parts.RULES),
        "WORKFLOW": Numbered(system_parts.WORKFLOW),
    }


def user_prompt(variables: Mapping[str, object] | None = None) -> PromptNode:
    """Assemble the task-specific initial user prompt."""
    return {
        "TASK": user_parts.TASK,
        "TASK CONSTRAINTS": user_parts.CONSTRAINTS,
        "BEGIN": user_parts.BEGIN,
    }
