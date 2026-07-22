"""LIBERO prompt bundle assembly."""

from __future__ import annotations

from robots.libero.prompts import system as system_parts
from robots.libero.prompts import user as user_parts
from rpent.context.prompt_utils import Numbered, PromptNode


def system_prompt() -> PromptNode:
    """Assemble the LIBERO system prompt tree."""
    return {
        "Role and Evaluation": system_parts.ROLE_AND_EVALUATION,
        "Proven Levers & Lessons — libero_10_task seed-0 sweep solved 9/10 (Read This)": (
            system_parts.PROVEN_LEVERS
        ),
        "Runtime": system_parts.RUNTIME,
        "Goal": system_parts.GOAL,
        "Rules (Non-negotiable)": system_parts.RULES,
        "Localization — how to get an object's world xyz without GT coords": (
            system_parts.LOCALIZATION
        ),
        "First-step Algorithm — agentview = Identity, wrist = Geometry": (
            system_parts.PERCEPTION_ALGORITHM
        ),
        "Workflow": Numbered(system_parts.WORKFLOW_STEPS),
        "Key Hyperparameters": system_parts.KEY_HYPERPARAMETERS,
        "Output Discipline": system_parts.OUTPUT_DISCIPLINE,
    }


def user_prompt() -> PromptNode:
    """Assemble the LIBERO user prompt tree."""
    return {
        "Cell": user_parts.CELL,
        "Mode": user_parts.MODE,
        "Begin": user_parts.BEGIN,
    }


__all__ = ["system_prompt", "user_prompt"]
