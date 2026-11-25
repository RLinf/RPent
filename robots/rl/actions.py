# Copyright 2026 The RPent Authors.
# Licensed under the Apache License, Version 2.0.
"""One exclusive, prevalidated sequential action tool over native primitives."""

from __future__ import annotations

import time
from collections.abc import Sequence
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from rpent.tools import ToolResult, tool
from rpent.tools.toolkit import ToolCancelled
from rpent.utils.logging import get_logger

logger = get_logger(__name__)

Vector3 = Annotated[list[float], Field(min_length=3, max_length=3)]


class MoveAction(BaseModel):
    """One absolute grasp TCP target; omitted parts keep measured values."""

    model_config = ConfigDict(
        extra="forbid",
        allow_inf_nan=False,
        json_schema_extra={
            "anyOf": [
                {"required": ["xyz"], "properties": {"xyz": {"type": "array"}}},
                {
                    "required": ["orientation_rpy"],
                    "properties": {"orientation_rpy": {"type": "array"}},
                },
            ]
        },
    )
    action: Literal["move_to"]
    xyz: Vector3 | None = Field(
        default=None,
        description="Absolute grasp TCP xyz in the selected world frame, metres.",
    )
    orientation_rpy: Vector3 | None = Field(
        default=None,
        description="Absolute grasp TCP fixed-axis XYZ roll,pitch,yaw, radians; Rz@Ry@Rx.",
    )

    @model_validator(mode="after")
    def require_target(self):
        if self.xyz is None and self.orientation_rpy is None:
            raise ValueError("move_to needs xyz, orientation_rpy, or both")
        return self


class GripperAction(BaseModel):
    """One normalized gripper target."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    action: Literal["set_gripper"]
    opening: Annotated[
        float,
        Field(
            ge=0,
            le=1,
            description="0 closed, 1 fully open; contact is not proof of grasp.",
        ),
    ]


Action = Annotated[MoveAction | GripperAction, Field(discriminator="action")]


class ArmMove(MoveAction):
    arm: Literal["left", "right"]


class ArmGripper(GripperAction):
    arm: Literal["left", "right"]


class BaseMove(BaseModel):
    """Drive the chassis at a bounded local velocity, then stop."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    action: Literal["move_base"]
    vx: Annotated[float, Field(ge=-0.35, le=0.35, description="Forward speed in m/s.")]
    vy: Annotated[float, Field(ge=-0.35, le=0.35, description="Leftward speed in m/s.")]
    yaw_rate: Annotated[
        float, Field(ge=-0.5, le=0.5, description="Counterclockwise speed in rad/s.")
    ]
    duration_s: Annotated[float, Field(ge=0.1, le=2)]


class LiftMove(BaseModel):
    """Move only the arm mounting lift."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    action: Literal["move_lift"]
    delta_m: Annotated[
        float, Field(ge=-0.25, le=0.25, description="Relative lift height in metres.")
    ]
    duration_s: Annotated[float, Field(ge=0.1, le=5)] = 3


ArmAction = Annotated[ArmMove | ArmGripper, Field(discriminator="action")]
MobileAction = Annotated[
    MoveAction | GripperAction | BaseMove | LiftMove, Field(discriminator="action")
]


class ActionBatch:
    """Execute under the toolkit's existing exclusive scheduler without recursion."""

    def __init__(self, toolkit):
        self.toolkit = toolkit

    @tool
    def execute_actions(
        self, actions: Annotated[list[Action], Field(min_length=1)]
    ) -> ToolResult:
        """Execute 1..K move_to/set_gripper actions strictly in list order. The entire list is validated before any movement. Each item finishes and is measured before the next begins. Stop immediately on native success, truncation, error or partial; remaining items are not executed. Returns every executed result and pose, executed_count, zero-based stopped_at, stop_reason and final_state. Use view_env to see the scene after motion. Example: {"actions":[{"action":"move_to","xyz":[0.1,0,0.3]},{"action":"set_gripper","opening":0}]}.

        Args:
            actions: Ordered list, at most the configured K; each item is move_to with xyz/orientation_rpy or set_gripper with opening. Omitted move targets keep the state measured at that item's start. The list contains only physical actions.
        """
        return self._execute_actions(actions)

    def _execute_actions(self, actions: Sequence[BaseModel]) -> ToolResult:
        """Run native actions using the toolkit's normal recording path."""
        tk = self.toolkit
        results, reason, stopped_at = [], None, None
        tk.workflow_stats["batch_calls"] = tk.workflow_stats.get("batch_calls", 0) + 1
        for index, item in enumerate(actions):
            tk.raise_if_cancelled()
            if tk._primitives.ended:
                reason = "terminated" if tk.solved() else "truncated"
                break
            command = item.model_dump(exclude_none=True)
            name = command.pop("action")
            begin = time.perf_counter()
            start_version = tk.frame_version()
            try:
                start_version = tk.frame_version(refresh=True)
                native = getattr(tk._primitives, name)(**command)
                result = tk.get_env_state(
                    command={"action": name, **command},
                    result=native.to_dict(),
                    elapsed_s=time.perf_counter() - begin,
                    start_frame_version=start_version,
                )
                tk._publish_step(tk.state.latest_record())
            except ToolCancelled:
                raise
            except Exception as exc:
                logger.exception(
                    "Native action %s did not complete with measured feedback", name
                )
                tk.action_error = {"action": name, "message": str(exc)}
                tk.state.save("infrastructure_error.json", tk.action_error, step=None)
                result = ToolResult(
                    data={
                        "status": "error",
                        "error": "native action outcome is uncertain",
                        "step": tk.state.latest_step,
                        "state_stale": True,
                        "start_frame_version": start_version,
                    }
                )
            tk.workflow_stats["atomic_actions"] = (
                tk.workflow_stats.get("atomic_actions", 0) + 1
            )
            results.append({"index": index, "action": name, **result.data})
            stopped_at = index
            if result.data.get("terminated") or result.data.get("truncated"):
                reason = "terminated" if result.data.get("terminated") else "truncated"
            elif result.data.get("status") in {"partial", "error"}:
                reason = (
                    result.data.get("reason")
                    or result.data.get("error")
                    or result.data["status"]
                )
            if reason is not None:
                break
        if reason is not None:
            tk.workflow_stats["batch_pauses"] = (
                tk.workflow_stats.get("batch_pauses", 0) + 1
            )
        return ToolResult(
            data={
                "status": "completed" if reason is None else "stopped",
                "results": results,
                "executed_count": len(results),
                "stopped_at": stopped_at,
                "stop_reason": reason,
                "final_state": (
                    tk.state.latest_record().state
                    if tk.infrastructure_error is not None
                    else tk._robot_state()
                ),
            }
        )


class DualArmBatch(ActionBatch):
    @tool
    def execute_actions(
        self, actions: Annotated[list[ArmAction], Field(min_length=1)]
    ) -> ToolResult:
        """Execute 1..K actions strictly in order. Each move_to or set_gripper names one left or right arm. The other arm holds its last target. Read the measured results; partial, error, native success or truncation stops the remaining list. Use view_env for RGB."""
        return self._execute_actions(actions)


class MobileBatch(ActionBatch):
    @tool
    def execute_actions(
        self, actions: Annotated[list[MobileAction], Field(min_length=1)]
    ) -> ToolResult:
        """Execute 1..K ordered move_base, move_lift, move_to or set_gripper items. Each item controls one component. An arm target uses the selected coordinate convention at the start of its item. In general coordinates the mount follows the base and lift; supply the target for the frame after each preceding item. Partial, error, success or truncation stops the remaining list. Use view_env for RGB."""
        return self._execute_actions(actions)
