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
from dataclasses import dataclass
from typing import Annotated, Any

import numpy as np
from pydantic import Field

from robots.efe.env_client import EfeEnvClient
from robots.efe.tools import proprio_map
from rpent.tools import ToolResult, tool

_JOINT_ACTION = "joint_position"
_GRIPPER = "gripper"
SKILL_NAMES = ("move_to_xyz", "set_gripper")
# A skill chunk stays short so the loop can re-read proprioception often. The
# service spec decides the real bound; this is only the preferred duration.
_PREFERRED_CHUNK_S = 0.2
# The protocol also caps the row spacing, independent of the task.
_PROTOCOL_MAX_DT_S = 0.5
# The gripper travel is small, so a Cartesian tolerance would accept almost any
# opening. Ten percent of the declared travel keeps the check meaningful.
_GRIPPER_TOLERANCE_FRACTION = 0.1
_GRIPPER_TOLERANCE_FLOOR_M = 0.002


def chunk_duration(spec: dict[str, Any], chunk_s: float | None = None) -> float:
    """Return a chunk duration the service accepts, or raise ValueError.

    The client requires a whole number of control periods, a duration at most
    ``0.5`` s, and a chunk no longer than ``spec["max_chunk_s"]``. A fixed
    0.2 s default violates the first rule on tasks whose control rate is not a
    multiple of 5 Hz, so the default is derived from the spec instead.

    Args:
        spec: The service capability spec.
        chunk_s: An explicit duration. Checked against the same rules.
    """
    control_hz = float(spec["control_hz"])
    if not math.isfinite(control_hz) or control_hz <= 0:
        raise ValueError(f"control_hz must be positive, got {control_hz}")
    period = 1.0 / control_hz
    max_chunk_s = float(spec["max_chunk_s"])
    if chunk_s is not None:
        value = float(chunk_s)
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"chunk_s must be positive and finite, got {chunk_s}")
        periods = value / period
        if abs(periods - round(periods)) > 1e-6:
            raise ValueError(
                f"chunk_s must be a whole multiple of the {period:.4f} s "
                f"control period, got {value}"
            )
        if value > min(_PROTOCOL_MAX_DT_S, max_chunk_s) + 1e-9:
            raise ValueError(
                f"chunk_s must be at most "
                f"{min(_PROTOCOL_MAX_DT_S, max_chunk_s):.4f} s for this task, "
                f"got {value}"
            )
        return value
    periods = int(math.floor(min(_PREFERRED_CHUNK_S, max_chunk_s) / period + 1e-9))
    if periods < 1 and period <= max_chunk_s + 1e-9:
        # A slow robot's single control step is longer than the preferred
        # chunk; one step is still legal and keeps the skills available.
        periods = 1
    if periods < 1:
        raise ValueError(
            f"A {period:.4f} s control period does not fit in max_chunk_s={max_chunk_s}"
        )
    duration = periods / control_hz
    if duration > _PROTOCOL_MAX_DT_S + 1e-9:
        # A slow robot's single control period can exceed the row-spacing cap
        # that every action type shares. No legal chunk exists in that case.
        raise ValueError(
            f"A single {duration:.4f} s control period exceeds the "
            f"{_PROTOCOL_MAX_DT_S} s row-spacing limit"
        )
    return duration


def _gripper_bounds(spec: dict[str, Any]) -> tuple[float | None, float | None]:
    """Return the declared gripper travel; either bound may be absent."""
    limits = (spec.get("action_types", {}).get(_JOINT_ACTION) or {}).get("limits") or {}
    bounds = limits.get(_GRIPPER)
    if not bounds or len(bounds) != 2:
        return None, None
    low = None if bounds[0] is None else float(bounds[0])
    high = None if bounds[1] is None else float(bounds[1])
    return low, high


def _proprio_joint(proprio: dict[str, float], name: str) -> float:
    """Return one joint from a proprioception record under either spelling."""
    bare = name.removeprefix("q_")
    for candidate in (name, bare, f"q_{bare}"):
        if candidate in proprio:
            return proprio[candidate]
    raise ValueError(f"Proprioception has no joint {name!r}")


def default_gripper_opening(spec: dict[str, Any]) -> float:
    """Return the midpoint of the declared travel, or ``0`` when it declares none."""
    low, high = _gripper_bounds(spec)
    if low is None or high is None:
        return 0.0
    return (low + high) / 2


def gripper_tolerance(spec: dict[str, Any]) -> float:
    """Return the opening tolerance in metres for the declared gripper travel."""
    low, high = _gripper_bounds(spec)
    if low is None or high is None:
        # Without declared limits, keep a value that is small against the
        # gripper travel of every standard task seen so far.
        return _GRIPPER_TOLERANCE_FLOOR_M
    return max(
        _GRIPPER_TOLERANCE_FLOOR_M, _GRIPPER_TOLERANCE_FRACTION * abs(high - low)
    )


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
    # The kinematics and the action layout may spell the same joint with or
    # without the ``q_`` prefix, so compare the bare joint names.
    columns = [name.removeprefix("q_") for name in joint_names]
    leading = [name.removeprefix("q_") for name in layout[: len(columns)]]
    if leading != columns:
        raise ValueError(
            "move_to_xyz needs joint_position columns in kinematics order; "
            f"got {layout} for joints {joint_names}"
        )
    if layout[len(columns) :] != [_GRIPPER]:
        raise ValueError(
            f"move_to_xyz needs one trailing {_GRIPPER!r} column; got {layout}"
        )
    return joint_names, layout


@dataclass(frozen=True)
class SkillParameters:
    """Values every skill run reads from the service spec."""

    chunk_s: float
    gripper_tolerance_m: float
    joint_names: tuple[str, ...]
    tcp: str


def skill_parameters(
    spec: dict[str, Any], chunk_s: float | None = None
) -> SkillParameters:
    """Return the spec values the skills need, or raise ValueError.

    A task whose layout, control period, or kinematics the skills cannot use
    raises here, so the toolkit and the prompt agree on availability.
    """
    joint_names, _ = joint_layout(spec)
    kinematics = spec.get("kinematics") or {}
    tcp_frames = [str(frame) for frame in kinematics.get("tcp_frames") or []]
    if not tcp_frames:
        raise ValueError("The task exposes no tool frame")
    # ``ik`` only accepts frames the service lists as targets, and the skill
    # reads the tool pose back from proprioception, so both must name the frame.
    ik_targets = set(kinematics.get("ik_targets") or [])
    tcp = next((frame for frame in tcp_frames if frame in ik_targets), None)
    if tcp is None:
        raise ValueError(
            "move_to_xyz needs a tool frame the service solves IK for; "
            f"tcp_frames={tcp_frames}, ik_targets={sorted(ik_targets)}"
        )
    _validate_proprio_layout(spec, tcp, joint_names)
    return SkillParameters(
        chunk_s=chunk_duration(spec, chunk_s),
        gripper_tolerance_m=gripper_tolerance(spec),
        joint_names=tuple(joint_names),
        tcp=tcp,
    )


def _validate_proprio_layout(
    spec: dict[str, Any], tcp: str, joint_names: list[str]
) -> None:
    """Reject a task whose proprioception omits the columns the skills read."""
    layout = spec.get("proprio_layout")
    if not layout:
        # A response may carry its own layout, so an absent spec layout does not
        # prove the columns are missing. Leave that check to the first read.
        return
    columns = {str(name) for name in layout}
    missing = [
        f"{tcp}_{axis}"
        for axis in ("x", "y", "z", "qw", "qx", "qy", "qz")
        if f"{tcp}_{axis}" not in columns
    ]
    bare = {name.removeprefix("q_") for name in columns}
    missing += [name for name in joint_names if name.removeprefix("q_") not in bare]
    if missing:
        raise ValueError(
            "move_to_xyz needs proprioception columns the task does not "
            f"declare: {sorted(set(missing))}"
        )


class EfeSkills:
    """High-level skills over one EFE episode."""

    skill_names = SKILL_NAMES

    @staticmethod
    def unavailable_reason(spec: dict[str, Any]) -> str | None:
        """Return why the skills cannot drive this task, or None when they can.

        Availability depends only on the service spec, so a caller can decide
        whether to offer the skills before it builds them.
        """
        try:
            skill_parameters(spec)
        except ValueError as exc:
            return str(exc)
        return None

    def __init__(
        self,
        *,
        client: EfeEnvClient,
        check_cancelled: Callable[[], None],
        tolerance_m: float = 0.01,
        max_step_m: float = 0.02,
        chunk_s: float | None = None,
    ) -> None:
        if tolerance_m <= 0 or max_step_m <= 0:
            raise ValueError("tolerance_m and max_step_m must be positive")
        parameters = skill_parameters(client.spec, chunk_s)
        self.client = client
        self._check_cancelled = check_cancelled
        self.tolerance_m = tolerance_m
        self.max_step_m = max_step_m
        self.chunk_s = parameters.chunk_s
        self.gripper_tolerance_m = parameters.gripper_tolerance_m
        self._joint_names = list(parameters.joint_names)
        self.tcp = parameters.tcp

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
        ends. An explicit ``gripper`` target is applied even when the tool is
        already at the position. The EFE verifier still decides task success.

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
                # The arm holds its pose, but a requested opening still runs:
                # otherwise the same argument would apply or not depending on
                # the starting position error.
                if gripper is not None and not self._gripper_reached(
                    float(gripper), current["gripper"]
                ):
                    row = [*current["joints"], float(gripper)]
                    result = self.client.submit(_JOINT_ACTION, [row], dt=self.chunk_s)
                    submit_calls += 1
                    control_steps += int(result.meta.get("control_steps_chunk") or 0)
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
        # Reaching the pose is not the only way a move succeeds: the chunk that
        # crosses the goal can end the episode before the arm arrives within
        # tolerance. Report that as success instead of an error.
        within_tolerance = bool(np.linalg.norm(error) <= self.tolerance_m)
        ended_in_success = (
            self.client.done and "success" in (self.client.done_reason or "").lower()
        )
        data: dict[str, Any] = {
            "ok": within_tolerance or ended_in_success,
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
        if not within_tolerance:
            data["note"] = (
                f"the move stopped {data['position_error_m']:.4f} m from the target"
                + (" after the episode succeeded" if ended_in_success else "")
            )
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
                    "error": (
                        "The episode ended before the gripper could move "
                        f"({self.client.done_reason or 'no reason reported'})"
                    ),
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
        ok = measured is None or self._gripper_reached(opening, float(measured))
        data: dict[str, Any] = {
            "ok": ok,
            "command_m": opening,
            "opening_m": measured,
            "tolerance_m": self.gripper_tolerance_m,
            "submit_calls": 1,
            "control_steps": int(meta.get("control_steps_chunk") or 0),
            "done": self.client.done,
            "reason": self.client.done_reason,
        }
        if not ok:
            data["error"] = (
                f"Gripper reached {float(measured):.4f} m, {opening:.4f} m "
                f"commanded (tolerance {self.gripper_tolerance_m:.4f} m)"
            )
        return ToolResult(data=data)

    def _gripper_reached(self, target: float, measured: float) -> bool:
        """Whether a measured opening is within tolerance of the target."""
        return abs(float(measured) - float(target)) <= self.gripper_tolerance_m

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
        # The kinematics and the proprioception layout may spell a joint with
        # or without the ``q_`` prefix; ``joint_layout`` already stripped it on
        # one side, so accept either spelling here.
        joints = [_proprio_joint(proprio, name) for name in self._joint_names]
        gripper = proprio.get("cmd_gripper", proprio.get("gripper_opening"))
        if gripper is None:
            # A task that declares no gripper travel has no midpoint to fall
            # back to; keep the current command at the neutral opening.
            gripper = default_gripper_opening(self.client.spec)
        if not math.isfinite(gripper):
            raise ValueError("Proprioception reports a non-finite gripper command")
        return {
            "position": position,
            "orientation": orientation,
            "joints": joints,
            "gripper": float(gripper),
        }
