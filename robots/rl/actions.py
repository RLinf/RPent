# Copyright 2026 The RPent Authors.
# Licensed under the Apache License, Version 2.0.
"""One exclusive, prevalidated sequential action tool over native primitives."""

from __future__ import annotations

import time
from collections.abc import Sequence
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from rpent.tools import ToolResult, tool

Vector3 = Annotated[list[float], Field(min_length=3, max_length=3)]


class MoveAction(BaseModel):
    """One absolute LIBERO world-frame target; omitted parts keep measured values."""

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
        default=None, description="Absolute grip-site world xyz, metres."
    )
    orientation_rpy: Vector3 | None = Field(
        default=None,
        description="Absolute grip-site fixed-axis XYZ roll,pitch,yaw, radians; Rz@Ry@Rx.",
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


class ActionBatch:
    """Execute under the toolkit's existing exclusive scheduler without recursion."""

    def __init__(self, toolkit):
        self.toolkit = toolkit

    @tool
    def execute_actions(
        self, actions: Annotated[list[Action], Field(min_length=1)]
    ) -> ToolResult:
        """Execute 1..K move_to/set_gripper actions strictly in list order. The entire list is validated before any movement. Each item finishes and is measured before the next begins. Stop immediately on native success, truncation, error or partial; remaining items are not executed. Returns every executed result and pose, executed_count, zero-based stopped_at, stop_reason and final_state. In episode_image only the final two-camera observation is returned. Example: {"actions":[{"action":"move_to","xyz":[0.1,0,0.3]},{"action":"set_gripper","opening":0}]}.

        Args:
            actions: Ordered list, at most the configured K; each item is move_to with xyz/orientation_rpy or set_gripper with opening. Omitted move targets keep the state measured at that item's start. Never include perception or finish tools.
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
            try:
                native = getattr(tk._primitives, name)(**command)
            except Exception as exc:
                native = ToolResult(data={"status": "error", "error": str(exc)})
            result = tk.get_env_state(
                command={"action": name, **command},
                result=native.to_dict(),
                elapsed_s=time.perf_counter() - begin,
            )
            tk._publish_step(tk.state.latest_record())
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
                "final_state": tk._robot_state(),
            }
        )


class LatestImageTools:
    """Explicit perception bound to the most recent action-returned observation."""

    def __init__(self, primitives):
        self.primitives = primitives

    @tool(readonly=True)
    def segment(
        self,
        prompt: str | None = None,
        point: Annotated[list[int], Field(min_length=2, max_length=2)] | None = None,
        camera: Literal["agentview", "wrist"] = "agentview",
    ) -> ToolResult:
        """Locate one object in the latest returned image. Provide exactly one non-empty prompt or point [row,col], never both. Returns mask_id, half-open row/col box, visible-surface world_xyz median, top_z_m (95th world-z percentile), size_xyz_m (5th..95th extents) and axis_yaw (xy major-axis direction or null). These are visible surfaces, not object centres or grasp poses. Missing query or detection returns an actionable text message. Example: {"prompt":"red cup","camera":"agentview"}.

        Args:
            prompt: Non-empty short object description; mutually exclusive with point.
            point: One pixel [row,col] on the target in the latest image of camera.
            camera: agentview fixed external camera or wrist moving with the gripper; default agentview.
        """
        return self.primitives.segment(
            prompt=prompt,
            point=point,
            camera=camera,
            step=self.primitives._state.latest_step,
        )

    @tool(readonly=True)
    def back_project(
        self,
        pixel: Annotated[list[int], Field(min_length=2, max_length=2)] | None = None,
        box: Annotated[list[int], Field(min_length=4, max_length=4)] | None = None,
        mask_id: str | None = None,
        camera: Literal["agentview", "wrist"] | None = None,
        z_range_m: Annotated[list[float], Field(min_length=2, max_length=2)]
        | None = None,
    ) -> ToolResult:
        """Measure the latest image's visible 3D surface. Provide exactly one pixel, box or current mask_id. Pixel yields one point; box/mask yields a per-axis median, not an object centre. Returns world_xyz in metres. A mask fixes its camera and must belong to the current observation; do not pass camera with mask_id. Example: {"pixel":[500,400],"camera":"agentview"}.

        Args:
            pixel: [row,col] in the latest image; no z filter.
            box: Half-open [row_min,col_min,row_max,col_max]; includes background.
            mask_id: Identifier from segment on the current observation; old masks must be re-segmented.
            camera: agentview or wrist for pixel/box, defaults to agentview; omit for mask.
            z_range_m: Optional inclusive [low,high] world-z filter for box/mask only.
        """
        p = self.primitives
        if mask_id is not None:
            mask = p._masks.get(mask_id)
            if mask is not None and mask[0] != p._state.latest_step:
                return ToolResult(
                    data={"error": "stale_mask"},
                    model_text="This mask belongs to an older observation. Call segment again on the latest image.",
                )
        return p.back_project(
            pixel=pixel,
            box=box,
            mask_id=mask_id,
            camera=camera,
            step=None if mask_id is not None else p._state.latest_step,
            z_range_m=z_range_m,
        )
