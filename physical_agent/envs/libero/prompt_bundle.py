"""Prompt bundle for the LIBERO environment."""
from __future__ import annotations

from physical_agent.context.libero_prompts import (
    CLAUDE_CODE_PERCEPTION_PROMPT_TEMPLATE,
    CLAUDE_CODE_PROMPT_TEMPLATE,
    INITIAL_USER_TEMPLATE,
    PERCEPTION_PREFIX,
    PERCEPTION_USER_TEMPLATE,
    SYSTEM_PROMPT,
    format_claude_code_prompt,
)
from physical_agent.envs.base import PromptBundle


PROMPTS = PromptBundle(
    system_prompt=SYSTEM_PROMPT,
    initial_user_template=INITIAL_USER_TEMPLATE,
    perception_prefix=PERCEPTION_PREFIX,
    perception_user_template=PERCEPTION_USER_TEMPLATE,
    claude_code_prompt_template=CLAUDE_CODE_PROMPT_TEMPLATE,
    claude_code_perception_prompt_template=CLAUDE_CODE_PERCEPTION_PROMPT_TEMPLATE,
    format_claude_code_prompt=format_claude_code_prompt,
)
