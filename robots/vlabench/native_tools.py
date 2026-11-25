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

"""VLABench motion primitives and the tools the planner sees."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import Field

from robots.vlabench.motion import VLABenchMotion
from rpent.tools import ToolResult, tool

Grip = Annotated[float, Field(ge=-1, le=1)] | Literal["hold"]
Vector3 = Annotated[list[float], Field(min_length=3, max_length=3)]
Quaternion = Annotated[list[float], Field(min_length=4, max_length=4)]


class VLABenchPrimitives(VLABenchMotion):
    """Pose and gripper motions over one VLABench environment client."""

    @tool(readonly=True)
    def finish(self, *, status: str, summary: str) -> ToolResult:
        """Stop the run; native task_status is authoritative over status."""
        return super().finish(status=status, summary=summary)

    @tool
    def move_pose(
        self,
        *,
        xyz: Vector3 | None = None,
        quaternion_xyzw: Quaternion | None = None,
        gripper: Grip = "hold",
        max_steps: Annotated[int, Field(gt=0, le=1000)] = 150,
        tol: Annotated[float, Field(gt=0)] = 0.012,
        step_clip: Annotated[float, Field(gt=0)] = 0.02,
        ori_tol: Annotated[float, Field(gt=0)] = 0.05,
    ) -> ToolResult:
        """Move the EEF to an absolute WORLD pose; omit xyz or quaternion to hold."""
        return super().move_pose(
            xyz=xyz,
            quaternion_xyzw=quaternion_xyzw,
            gripper=gripper,
            max_steps=max_steps,
            tol=tol,
            step_clip=step_clip,
            ori_tol=ori_tol,
        )

    @tool
    def move_delta(
        self,
        *,
        dxyz: Vector3 | None = None,
        rotation_vector_world: Vector3 | None = None,
        gripper: Grip = "hold",
        max_steps: Annotated[int, Field(gt=0, le=1000)] = 120,
        tol: Annotated[float, Field(gt=0)] = 0.012,
        step_clip: Annotated[float, Field(gt=0)] = 0.02,
        ori_tol: Annotated[float, Field(gt=0)] = 0.05,
    ) -> ToolResult:
        """Move relative to measured pose (WORLD meters and rotvec radians)."""
        return super().move_delta(
            dxyz=dxyz,
            rotation_vector_world=rotation_vector_world,
            gripper=gripper,
            max_steps=max_steps,
            tol=tol,
            step_clip=step_clip,
            ori_tol=ori_tol,
        )

    @tool
    def set_gripper(
        self,
        *,
        gripper: Annotated[float, Field(ge=-1, le=1)],
        steps: Annotated[int, Field(gt=0, le=1000)] = 5,
    ) -> ToolResult:
        """Drive the fingers: -1 opens, +1 closes, 0 holds measured width."""
        return super().set_gripper(gripper=gripper, steps=steps)
