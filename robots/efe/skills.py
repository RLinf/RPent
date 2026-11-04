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

"""High-level EFE skills built on the public client.

A skill composes the task's own action type. It uses IK to turn a Cartesian
target into joint targets, submits short chunks, and re-reads public
proprioception to close the loop. It never reads simulator state and never
replaces the EFE score with its own.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Annotated, Any

import numpy as np
from pydantic import Field

from robots.efe.env_client import EfeEnvClient
from robots.efe.tools import proprio_map
from rpent.tools import ToolResult, tool

_JOINT_ACTION = "joint_position"
_GRIPPER = "gripper"
SKILL_NAMES = ("move_to_xyz", "set_gripper")


def joint_layout(spec: dict[str, Any]) -> tuple[list[str], list[str]]:
    """Return the joint names and the matching action columns.

    The skill needs ``joint_position`` rows that follow the kinematics joint
    order plus one gripper column. A task with another layout must not be
    driven by this skill.
    """
    kinematics = spec.get("kinematics") or {}
    joint_names = list(kinematics.get("joint_names") or [])
    action = (spec.get("action_types") or {}).get(_JOINT_ACTION)
    if not joint_names or action is None:
        raise ValueError(
            f"move_to_xyz needs kinematics joint names and a {_JOINT_ACTION!r} action"
        )
    layout = list(action["layout"])
    columns = [name.removeprefix("q_") for name in joint_names]
    if layout[: len(columns)] != columns:
        raise ValueError(
            "move_to_xyz needs joint_position columns in kinematics order; "
            f"got {layout} for joints {columns}"
        )
    if layout[len(columns) :] != [_GRIPPER]:
        raise ValueError(
            f"move_to_xyz needs one trailing {_GRIPPER!r} column; got {layout}"
        )
    return joint_names, layout


class EfeSkills:
    """High-level skills over one EFE episode."""

    skill_names = SKILL_NAMES

    def __init__(
        self,
        *,
        client: EfeEnvClient,
        check_cancelled: Callable[[], None],
        tolerance_m: float = 0.01,
        max_step_m: float = 0.02,
        chunk_s: float = 0.2,
    ) -> None:
        if tolerance_m <= 0 or max_step_m <= 0 or chunk_s <= 0:
            raise ValueError("tolerance_m, max_step_m, and chunk_s must be positive")
        self.client = client
        self._check_cancelled = check_cancelled
        self.tolerance_m = tolerance_m
        self.max_step_m = max_step_m
        self.chunk_s = chunk_s
        self._joint_names, _ = joint_layout(client.spec)
        kinematics = client.spec["kinematics"]
        if not kinematics.get("tcp_frames"):
            raise ValueError("The task exposes no tool frame")
        self.tcp = kinematics["tcp_frames"][0]

    @tool
    def move_to_xyz(
        self,
        x: float,
        y: float,
        z: float,
        gripper: float | None = None,
        max_iterations: Annotated[int, Field(ge=1, le=50)] = 20,
    ) -> ToolResult:
        """Move the tool frame to a world position in metres, keeping its orientation.

        The skill closes the loop: it reads proprioception, solves IK for a
        bounded step, submits a short joint chunk, and repeats until the
        position error is within tolerance. It stops early when the episode
        ends. The EFE verifier still decides task success.

        Args:
            x: Target world x in metres.
            y: Target world y in metres.
            z: Target world z in metres.
            gripper: Finger opening in metres. Default: keep the current command.
            max_iterations: Bounded correction steps.
        """
        target = np.asarray([x, y, z], dtype=np.float64)
        if not np.isfinite(target).all():
            raise ValueError(f"Target must be finite, got {target.tolist()}")
        submit_calls = 0
        control_steps = 0
        position = None
        error = None
        for iteration in range(1, int(max_iterations) + 1):
            self._check_cancelled()
            if self.client.done:
                break
            current = self._current_state()
            position = current["position"]
            error = target - position
            distance = float(np.linalg.norm(error))
            if distance <= self.tolerance_m:
                break
            step = error * min(1.0, self.max_step_m / distance)
            pose = [*(position + step), *current["orientation"]]
            solution = self.client.ik({self.tcp: pose}, seed=current["joints"])
            if not solution.get("ok"):
                detail = solution.get("error") or solution
                return ToolResult(
                    data={
                        "error": f"IK did not converge for {pose[:3]}: {detail}",
                        "tcp": self.tcp,
                        "target": target.tolist(),
                        "position": position.tolist(),
                        "position_error_m": distance,
                        "iterations": iteration - 1,
                    }
                )
            joint_error = solution.get("error") or {}
            target_error = joint_error.get(self.tcp) or {}
            position_error = target_error.get("pos_m")
            if position_error is not None and float(position_error) > self.tolerance_m:
                return ToolResult(
                    data={
                        "error": (
                            f"Target is outside the reachable workspace: "
                            f"IK error {float(position_error):.4f} m"
                        ),
                        "tcp": self.tcp,
                        "target": target.tolist(),
                        "ik_error": target_error,
                        "iterations": iteration - 1,
                    }
                )
            joints = np.asarray(solution["q"], dtype=np.float64)
            if (
                joints.shape != (len(self._joint_names),)
                or not np.isfinite(joints).all()
            ):
                raise ValueError(f"IK returned an unusable solution: {solution['q']}")
            opening = current["gripper"] if gripper is None else float(gripper)
            row = [*joints.tolist(), opening]
            result = self.client.submit(_JOINT_ACTION, [row], dt=self.chunk_s)
            submit_calls += 1
            control_steps += int(result.meta.get("control_steps_chunk") or 0)
        if self.client.last_meta.get("proprio"):
            current = self._state_from_meta(self.client.last_meta)
            position = current["position"]
            error = target - position
        if position is None:
            current = self._current_state()
            position = current["position"]
            error = target - position
        data: dict[str, Any] = {
            "ok": bool(np.linalg.norm(error) <= self.tolerance_m),
            "tcp": self.tcp,
            "target": target.tolist(),
            "position": position.tolist(),
            "position_error_m": float(np.linalg.norm(error)),
            "tolerance_m": self.tolerance_m,
            "submit_calls": submit_calls,
            "control_steps": control_steps,
            "done": self.client.done,
            "reason": self.client.done_reason,
        }
        if not data["ok"]:
            data["error"] = (
                f"Move stopped {data['position_error_m']:.4f} m from the target "
                f"after {submit_calls} chunks"
            )
        return ToolResult(data=data)

    @tool
    def set_gripper(self, opening: float) -> ToolResult:
        """Set the gripper finger opening in metres, holding the current arm pose.

        The skill commands one short chunk. Read proprioception afterwards to
        confirm the fingers moved; a blocked finger reports the opening it
        actually reached.

        Args:
            opening: Target finger opening in metres, within the service limits.
        """
        self._check_cancelled()
        opening = float(opening)
        if not math.isfinite(opening):
            raise ValueError(f"Gripper opening must be finite, got {opening}")
        if self.client.done:
            return ToolResult(
                data={
                    "ok": False,
                    "done": True,
                    "reason": self.client.done_reason,
                }
            )
        current = self._current_state()
        row = [*current["joints"], opening]
        result = self.client.submit(_JOINT_ACTION, [row], dt=self.chunk_s)
        meta = result.meta
        measured = None
        if meta.get("proprio"):
            proprio = proprio_map(self.client.spec, meta["proprio"])
            measured = proprio.get("gripper_opening")
        return ToolResult(
            data={
                "ok": measured is None
                or abs(float(measured) - opening) <= self.tolerance_m,
                "command_m": opening,
                "opening_m": measured,
                "submit_calls": 1,
                "control_steps": int(meta.get("control_steps_chunk") or 0),
                "done": self.client.done,
                "reason": self.client.done_reason,
            }
        )

    def _current_state(self) -> dict[str, Any]:
        """Read the tool position, orientation, joints, and gripper command."""
        return self._state_from_meta(self.client.observe(video="none").meta)

    def _state_from_meta(self, meta: dict[str, Any]) -> dict[str, Any]:
        """Convert one public proprioception record into skill inputs."""
        proprio = proprio_map(self.client.spec, meta.get("proprio") or {})
        try:
            position = np.asarray(
                [proprio[f"{self.tcp}_{axis}"] for axis in ("x", "y", "z")],
                dtype=np.float64,
            )
            orientation = [
                proprio[f"{self.tcp}_{axis}"] for axis in ("qw", "qx", "qy", "qz")
            ]
        except KeyError as exc:
            raise ValueError(
                f"Proprioception has no pose for tool frame {self.tcp!r}: {exc}"
            ) from exc
        try:
            joints = [proprio[name] for name in self._joint_names]
        except KeyError as exc:
            raise ValueError(f"Proprioception has no joint {exc}") from exc
        gripper = proprio.get("cmd_gripper", proprio.get("gripper_opening"))
        if gripper is None:
            limits = self.client.spec["action_types"][_JOINT_ACTION]["limits"][_GRIPPER]
            gripper = float(sum(limits) / 2)
        if not math.isfinite(gripper):
            raise ValueError("Proprioception reports a non-finite gripper command")
        return {
            "position": position,
            "orientation": orientation,
            "joints": joints,
            "gripper": float(gripper),
        }
