# Copyright 2026 The RPent Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""BEHAVIOR prompts shared by the existing RPent planners."""

from __future__ import annotations

from collections.abc import Mapping

from robots.behavior.prompts import evaluate
from rpent.prompt.utils import PromptNode


def system_prompt(variables: Mapping[str, object] | None = None) -> PromptNode:
    """Return task and tool instructions for one BEHAVIOR episode."""
    variables = variables or {}
    allowed = set(variables.get("tool_allowlist", ("vla_act",)))
    tool_instructions = evaluate.TOOLS
    if "vla_act" in allowed:
        tool_instructions += evaluate.VLA
        limit = variables.get("max_chunks_per_call")
        if limit is not None:
            tool_instructions += (
                f"\nUse at most {limit} VLA chunks per tool call, then inspect the "
                "returned observation before selecting the next action.\n"
            )
        tool_instructions += (
            evaluate.MAIN_INSTRUCTION
            if variables.get("instruction_mode") == "main"
            else evaluate.OVERRIDE_INSTRUCTION
        )
    if "pixel_to_robot_base" in allowed:
        tool_instructions += evaluate.GEOMETRY
    if "read_robot_state" in allowed:
        tool_instructions += evaluate.ROBOT_STATE
    if "move_eef_delta" in allowed:
        tool_instructions += evaluate.LOCAL_MOTION
    native = allowed & {
        "segment",
        "move_eef_delta",
        "set_gripper",
        "release",
        "move_to",
        "rotate_wrist",
        "rotate_wrist_joint",
        "move_base",
    }
    if native:
        for required, guidance in evaluate.NATIVE_SELECTION:
            if allowed.issuperset(required):
                tool_instructions += guidance
    for name, guidance in (
        ("segment", evaluate.SEGMENT),
        ("set_gripper", evaluate.SET_GRIPPER),
        ("release", evaluate.RELEASE),
        ("move_to", evaluate.NATIVE_MOTION),
        ("rotate_wrist", evaluate.WRIST_ROTATION),
        ("rotate_wrist_joint", evaluate.WRIST_JOINT),
        ("move_base", evaluate.BASE_MOTION),
    ):
        if name in native:
            tool_instructions += guidance
    return {
        "TASK": evaluate.TASK,
        "OBSERVATIONS": evaluate.OBSERVATIONS,
        "TOOLS": tool_instructions,
        "MEMORY": evaluate.MEMORY,
        "FEEDBACK": evaluate.FEEDBACK,
    }


def user_prompt(variables: Mapping[str, object] | None = None) -> PromptNode:
    """Start from the recorded initial observation."""
    return {
        "BEGIN": (
            "Task {{task_id}}: {{task_name}}. Read view_env_state(step=0) first; "
            "its images and measured robot state describe the starting situation."
        )
    }
