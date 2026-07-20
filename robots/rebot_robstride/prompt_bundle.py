"""Prompt bundle assembly for the reBot DevArm RobStride environment."""

from __future__ import annotations

from robots.rebot_robstride.prompts.system import SYSTEM, USER


def system_prompt() -> str:
    return SYSTEM


def user_prompt() -> dict:
    return dict(USER)
