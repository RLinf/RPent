"""RoboCasa prompt bundle assembly."""
from __future__ import annotations

from rpent.context.prompt_utils import PromptNode
from rpent.context.prompts import prompt as base_prompt
from robots.robocasa import prompts as robocasa_prompt


def system_prompt() -> dict[str, PromptNode]:
    """Return the system prompt tree."""
    return {
        "Intro": robocasa_prompt.PREAMBLE,
        "Goal": robocasa_prompt.GOAL,
        "Rules": robocasa_prompt.RULES,
        "Localization": robocasa_prompt.LOCALIZATION,
        "Navigation": robocasa_prompt.NAVIGATION,
        "Primitives": robocasa_prompt.PRIMITIVES,
        "VLA_Rules": robocasa_prompt.VLA_RULES,
        "Gripper_Rules": robocasa_prompt.GRIPPER_RULES,
        "Workflow": robocasa_prompt.WORKFLOW,
        "Environment": robocasa_prompt.ENVIRONMENT,
        "Output": base_prompt.OUTPUT,
        "Next": robocasa_prompt.NEXT,
    }


def user_prompt() -> dict[str, PromptNode]:
    """Return the first user message tree."""
    return {
        "Task": """
        - task:    {{task_name}} / {{split}}
        - seed:    {{seed}}
        - output_dir: {{output_dir}}
        - output:  {{output_dir}}/
          - audit filename:  {{recipe_tag}}.json
        """,
        "Mode": robocasa_prompt.USER_MODE,
    }