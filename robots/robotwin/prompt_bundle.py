"""RoboTwin prompt bundle assembly."""

from __future__ import annotations

from robots.robotwin.prompts import system as system_parts
from robots.robotwin.prompts import user as user_parts
from rpent.context.prompt_utils import Numbered


def system_prompt():
    return {
        "PREAMBLE": system_parts.PREAMBLE,
        "GOAL": system_parts.GOAL,
        "RULES": system_parts.RULES,
        "AUTHORITY": system_parts.AUTHORITY,
        "HISTORICAL CONTEXT": system_parts.HISTORICAL_CONTEXT,
        "PERCEPTION": system_parts.PERCEPTION,
        "CAMERA_ROLES": system_parts.CAMERA_ROLES,
        "EMBODIMENT": system_parts.EMBODIMENT,
        "PRIMITIVES": system_parts.PRIMITIVES,
        "VLA_RULES": system_parts.VLA_RULES,
        "GRIPPER_RULES": system_parts.GRIPPER_RULES,
        "PLANNER_RULES": system_parts.PLANNER_RULES,
        "BIMANUAL_RULES": system_parts.BIMANUAL_RULES,
        "RECOVERY": system_parts.RECOVERY,
        "BUDGET": system_parts.BUDGET,
        "WORKFLOW": Numbered(system_parts.WORKFLOW),
        "SUCCESS": system_parts.SUCCESS,
        "ACTION_COMMITMENT": system_parts.ACTION_COMMITMENT,
        "USER_MODE": system_parts.USER_MODE,
    }


def user_prompt():
    return {
        "CELL": user_parts.CELL,
        "BEGIN": user_parts.BEGIN,
    }


__all__ = ["system_prompt", "user_prompt"]
