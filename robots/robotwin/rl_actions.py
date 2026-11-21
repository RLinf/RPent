# Copyright 2026 The RPent Authors.
# Licensed under the Apache License, Version 2.0.
"""RoboTwin general-frame inputs for the shared RL action executor."""

from typing import Annotated, Literal

from pydantic import Field

from robots.rl.actions import ActionBatch, GripperAction, MoveAction, Vector3
from rpent.tools import ToolResult, tool

Arm = Literal["left", "right"]


class RoboTwinMoveAction(MoveAction):
    """Move the selected finger-pad TCP, optionally with the other arm."""

    arm: Arm
    xyz: Vector3 | None = Field(
        default=None, description="Absolute general-frame xyz in metres."
    )
    orientation_rpy: Vector3 | None = Field(
        default=None,
        description="Absolute [roll,pitch,yaw] in radians, fixed-axis XYZ.",
    )
    other_xyz: Vector3 | None = Field(
        default=None,
        description="Optional synchronous xyz target for the other arm in metres.",
    )
    other_orientation_rpy: Vector3 | None = Field(
        default=None,
        description="Optional synchronous absolute RPY target for the other arm.",
    )


class RoboTwinGripperAction(GripperAction):
    """Set the selected gripper, optionally with the other gripper."""

    arm: Arm
    other_opening: Annotated[float, Field(ge=0, le=1)] | None = None


RoboTwinAction = Annotated[
    RoboTwinMoveAction | RoboTwinGripperAction, Field(discriminator="action")
]


class RoboTwinActionBatch(ActionBatch):
    """Run ordered RoboTwin actions through the existing batch executor."""

    @tool
    def execute_actions(
        self, actions: Annotated[list[RoboTwinAction], Field(min_length=1)]
    ) -> ToolResult:
        """Execute 1..K RoboTwin actions in order. Each action names an anatomical arm and may synchronize the other arm. A move_to specifies xyz in metres, absolute orientation_rpy in radians, or both. A set_gripper specifies opening from 0 closed to 1 open. Stops on native success, truncation, error or partial. Read the measured results and executed_count. Example: {"actions":[{"action":"set_gripper","arm":"left","opening":1}]}."""
        return self._execute_actions(actions)
