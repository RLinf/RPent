"""Prompt bundle assembly for the single-Franka environment."""

from __future__ import annotations

from robots.franka.prompts import system as system_parts
from robots.franka.prompts import user as user_parts
from rpent.context.prompt_utils import Numbered, PromptNode


def system_prompt() -> PromptNode:
    """Assemble the Franka system prompt."""
    return {
        "ROLE": system_parts.ROLE,
        "RUNTIME": system_parts.RUNTIME,
        "SAFETY RULES": Numbered(system_parts.RULES),
        "WORKFLOW": Numbered(system_parts.WORKFLOW),
    }


def user_prompt() -> PromptNode:
    """Assemble the task-specific initial user prompt."""
    return {
        "TASK": user_parts.TASK,
        "TASK CONSTRAINTS": user_parts.CONSTRAINTS,
        "BEGIN": user_parts.BEGIN,
    }
