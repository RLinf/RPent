"""LIBERO prompt bundle assembly."""
from __future__ import annotations

from pathlib import Path

from rpent.context.prompt_utils import PromptNode
from rpent.context.prompts import prompt as base_prompt
from robots.libero import prompts as libero_prompt


def system_prompt() -> PromptNode:
    """Return the system prompt text."""
    return (
        Path(__file__).parent / "prompts" / "perception_system_prompt.md"
    ).read_text(encoding="utf-8")

    # Previous sectioned prompt kept for reference while the aligned perception
    # prompt is reviewed:
    # return {
    #     "Intro": libero_prompt.PREAMBLE,
    #     "Goal": libero_prompt.GOAL,
    #     "Rules": libero_prompt.RULES,
    #     "Localization": libero_prompt.LOCALIZATION,
    #     "Workflow": libero_prompt.WORKFLOW,
    #     "Environment": libero_prompt.ENVIRONMENT,
    #     "Output": base_prompt.OUTPUT,
    #     "Next": libero_prompt.NEXT,
    # }


def user_prompt() -> dict[str, PromptNode]:
    """Return the first user message tree."""
    sections = dict(base_prompt.USER)
    sections["Mode"] = libero_prompt.USER_MODE
    return sections
