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

"""BEHAVIOR action tools and local robot helpers using borrowed service clients.

The primitives own action submission and feedback accounting. Base validation
and per-call motion ledgers remain implementation helpers.
CuRobo planning is provided by ``robots.behavior.motion``.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from copy import deepcopy
from numbers import Real
from typing import TYPE_CHECKING, Annotated, Any, Literal

import numpy as np
from pydantic import Field

from robots.behavior.motion import (
    MAX_SINGLE_ARM_REPOSITION_M as MAX_MOVE_TO_REPOSITION_M,
)
from robots.behavior.tools import (
    decode_robot_state,
    extract_public_proprio,
    prepare_behavior_observation,
)
from rpent.tools import ToolResult, tool
from rpent.utils.rpc.rpc_client import RpcError

if TYPE_CHECKING:
    from robots.behavior.env_client import BehaviorEnvClient
    from rpent.robots.components.pi05_vla_client import Pi05VLAClient


CONTROL_DT_S = 1.0 / 30.0
BASE_OUTPUT_SCALE = np.array([0.75, 0.75, 1.0], dtype=np.float64)
MAX_NORMALIZED_LINEAR_NORM = 1.0
MAX_NORMALIZED_TURN = 1.0
# One call may not exceed the official episode budget; the evaluator stops
# the session at its own limit, so this is only a sanity bound.
MAX_MOVE_STEPS = 5000
# Limit ramp plus final holds before constructing a gripper action array.
MAX_GRIPPER_ACTIONS = MAX_MOVE_STEPS
# motion.py owns the single reposition bound, so the tool layer never carries
# a second, stricter number.
MAX_EEF_DELTA_M = MAX_MOVE_TO_REPOSITION_M
# Bounded hold actions spent reaching a planable stationary start. Counted
# against the episode's action budget and reported to the caller.
MAX_AUTO_SETTLE_ACTIONS = 12
# Wrist joints by zero-based index into one arm's seven joints. From the R1Pro
# URDF: joint5 turns about the forearm axis (roll), joint6 about the wrist's
# lateral axis (pitch), joint7 about its remaining axis (yaw).
WRIST_JOINT_OFFSET = {"roll": 4, "pitch": 5, "yaw": 6}
COUNT_NAMES = (
    "issued_actions",
    "confirmed_actions",
    "unconfirmed_actions",
    "executed_actions",
)


def _unissued(**data: Any) -> ToolResult:
    """Report a rejected operation without losing any earlier ledger entries."""
    return ToolResult(
        data={"requested_actions": 0, **dict.fromkeys(COUNT_NAMES, 0), **data}
    )


def _action_counts(
    info: dict[str, Any], limit: int
) -> tuple[dict[str, int | None], list[str]]:
    """Normalize execution counts and retain missing or inconsistent values as unknown."""
    counts: dict[str, int | None] = {}
    errors: list[str] = []
    for name in COUNT_NAMES:
        value = info.get(name)
        if value is not None and (
            isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer))
            or not 0 <= value <= limit
        ):
            errors.append(f"{name} is outside the submitted chunk")
            value = None
        counts[name] = None if value is None else int(value)
    issued, confirmed, unknown, executed = (counts[name] for name in COUNT_NAMES)
    accounted = confirmed if confirmed is not None else executed
    if (
        issued is not None
        and (
            (accounted is not None and accounted > issued)
            or (unknown is not None and unknown > issued)
            or (
                accounted is not None
                and unknown is not None
                and issued != accounted + unknown
            )
        )
    ) or (confirmed is not None and executed is not None and executed != confirmed):
        errors.append("execution counters disagree")
        counts = dict.fromkeys(COUNT_NAMES)
    return counts, errors


def _termination_flag(value: Any) -> bool | None:
    """Reduce supplied chunk feedback without treating missing flags as false."""
    if value is None:
        return None
    array = np.asarray(value)
    if array.dtype.kind != "b" or array.size == 0:
        raise ValueError("BEHAVIOR termination feedback must be boolean or None")
    return bool(array.any())


def _build_joint_ramp_actions(
    native: np.ndarray,
    index: int,
    current: float,
    target: float,
    steps: int,
    settle_steps: int,
    *,
    scale: float = 1.0,
    offset: float = 0.0,
) -> np.ndarray:
    """Interpolate one channel, repeat its target, and hold all other channels.

    Scale and offset map tool units to native commands: radians use (1, 0),
    while a gripper opening in [0, 1] uses (2, -1).
    """
    values = [current + (target - current) * i / steps for i in range(1, steps + 1)]
    values.extend([target] * settle_steps)
    actions = np.tile(native, (len(values), 1))
    actions[:, index] = scale * np.asarray(values) + offset
    return actions


def _validate_single_step_observation(
    observation: dict | list,
    info: dict[str, Any],
    expected: dict[str, int],
    before: np.ndarray,
    *,
    task_id: int,
) -> tuple[dict, np.ndarray, dict[str, int]]:
    """Bind one confirmed action to its exact adjacent public observation pair."""
    if isinstance(observation, list):
        if len(observation) != 1:
            raise ValueError("one action requires one final observation")
        observation = observation[0]
    public = extract_public_proprio(observation, task_id=task_id)
    token = validate_observation_token(info.get("observation_token"))
    if token != {
        "generation": expected["generation"],
        "observation_index": expected["observation_index"] + 1,
    }:
        raise ValueError("confirmation token is stale or from another generation")
    context = info.get("motion_context")
    validate_public_base_context(context, token, public)
    if (
        context.get("pair_confirmed") is not True
        or context.get("previous_token") != expected
        or not np.array_equal(context.get("previous_proprio"), before)
    ):
        raise ValueError("observations are not a confirmed adjacent pair")
    return observation, public, token


def _validated_arm_plan(
    planned: dict[str, Any], *, arm: str, state: dict[str, Any]
) -> np.ndarray:
    """Validate native CuRobo output before packing it into control actions.

    Waypoint row zero is the command at one control period, not the initial
    state at t=0. The backend resamples its path; the primitive never discards
    waypoints or changes their timing.
    """
    names = [f"{arm}_arm_joint{index}" for index in range(1, 8)]
    if planned.get("joint_names") != names:
        raise ValueError("arm plan joint_names do not match R1Pro native order")
    if (
        planned.get("frame") != "robot_base"
        or planned.get("quaternion_order") != "xyzw"
    ):
        raise ValueError("arm plan requires robot_base frame and xyzw quaternions")
    dt = planned.get("dt_s")
    control_dt = planned.get("control_dt_s")
    if any(
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, (int, float, np.integer, np.floating))
        or not np.isfinite(value)
        or value <= 0
        for value in (dt, control_dt)
    ) or not np.isclose(dt, control_dt, rtol=1e-7, atol=1e-10):
        raise ValueError("arm path dt_s must equal the official control_dt_s")
    path = np.asarray(planned.get("position"), dtype=np.float64)
    if (
        path.ndim != 2
        or path.shape[1] != 7
        or not 1 <= len(path) <= 600
        or not np.isfinite(path).all()
    ):
        raise ValueError("arm plan must contain 1 to 600 finite seven-joint waypoints")
    other = "right" if arm == "left" else "left"
    for key, expected in (
        ("start_arm_qpos", state[arm]["arm_qpos"]),
        ("held_trunk_qpos", state["trunk_qpos"]),
        ("held_other_arm_qpos", state[other]["arm_qpos"]),
    ):
        actual = np.asarray(planned.get(key), dtype=np.float64)
        if actual.shape != np.shape(expected) or not np.allclose(
            actual, expected, atol=1e-6, rtol=0
        ):
            raise ValueError(f"arm plan {key} differs from the current observation")
    contract = planned.get("motion_state_contract") or {}
    if contract.get("stationary") is not True:
        raise ValueError("arm plan requires a stationary measured robot state")
    for side in ("left", "right"):
        fingers = np.asarray((planned.get("held_finger_qpos") or {}).get(side))
        if fingers.shape != (2,) or not np.allclose(
            fingers, state[side]["finger_qpos"], atol=1e-6, rtol=0
        ):
            raise ValueError("arm plan finger geometry differs from the observation")
    limits = np.asarray(planned.get("joint_position_limits_rad"), dtype=np.float64)
    velocity = np.asarray(planned.get("joint_velocity_limits_rad_s"), dtype=np.float64)
    if (
        limits.shape != (7, 2)
        or not np.isfinite(limits).all()
        or np.any(limits[:, 0] >= limits[:, 1])
        or velocity.shape != (7,)
        or not np.isfinite(velocity).all()
        or np.any(velocity <= 0)
    ):
        raise ValueError(
            "arm plan requires finite position and positive velocity limits"
        )
    trajectory = np.vstack((state[arm]["arm_qpos"], path))
    if np.any(trajectory < limits[:, 0] - 1e-6) or np.any(
        trajectory > limits[:, 1] + 1e-6
    ):
        raise ValueError("arm plan violates a joint position limit")
    if np.any(np.abs(np.diff(trajectory, axis=0)) > velocity * dt + 1e-6):
        raise ValueError("arm plan exceeds the velocity limit at the control period")
    calibration = planned.get("fk_comparison", {})
    for side in ("left", "right"):
        values = calibration.get(side, {}) if isinstance(calibration, dict) else {}
        errors = np.asarray(
            [values.get("position_error_m"), values.get("orientation_error_rad")],
            dtype=np.float64,
        )
        if (
            not np.isfinite(errors).all()
            or np.any(errors < 0)
            or np.any(errors > [0.02, 0.10])
        ):
            raise ValueError("arm plan FK does not agree with both public EEF poses")
    return path


class BehaviorPrimitives:
    """Compose tools without taking ownership of runtime clients or reset."""

    def __init__(
        self,
        *,
        env: BehaviorEnvClient,
        model: Pi05VLAClient,
        task_id: int,
        task_language: str,
        check_cancelled: Callable[[], None],
        execution_horizon: int,
        max_chunks_per_call: int | None = None,
        instruction_mode: Literal["main", "override"] = "override",
    ) -> None:
        if (
            isinstance(execution_horizon, bool)
            or not isinstance(execution_horizon, int)
            or execution_horizon < 1
        ):
            raise ValueError("execution_horizon must be a positive integer")
        if max_chunks_per_call is not None and (
            type(max_chunks_per_call) is not int or max_chunks_per_call < 1
        ):
            raise ValueError("max_chunks_per_call must be a positive integer")
        if instruction_mode not in {"main", "override"}:
            raise ValueError("instruction_mode must be main or override")
        self.env = env
        self.model = model
        self.task_id = task_id
        self.task_language = task_language
        self.execution_horizon = execution_horizon
        self.max_chunks_per_call = max_chunks_per_call
        self.instruction_mode = instruction_mode
        self.last_observation: dict[str, Any] | None = None
        self._check_cancelled = check_cancelled
        self._terminated: bool | None = None
        self._truncated: bool | None = None
        self._session_ended = False
        self._last_issued_action: np.ndarray | None = None
        self._command_history_unknown = False

    # Shared execution and measured hold state.

    def _ended(self) -> bool:
        if self._session_ended:
            return True
        # Injected environments may omit the optional session-status RPC.
        status_reader = getattr(self.env, "get_session_status", None)
        if callable(status_reader):
            self._session_ended = status_reader()["ended"]
        return self._session_ended

    def _send_actions(
        self, actions: np.ndarray, *, expected_observation_token: dict | None = None
    ) -> tuple:
        """Retain only a known issued command; failed RPCs leave its value unknown."""
        previous = self._last_issued_action
        previously_unknown = self._command_history_unknown
        self._last_issued_action = None
        # A transport error may occur after the simulator has received actions.
        # Never turn that uncertainty into a new measured-position hold target.
        self._command_history_unknown = True
        kwargs = (
            {"expected_observation_token": expected_observation_token}
            if expected_observation_token is not None
            else {}
        )
        ledger = getattr(self, "_move_call_ledger", None)
        entry = ledger.start(actions, self._move_call_phase) if ledger else None
        result = self.env.chunk_step(actions, **kwargs)
        if ledger:
            ledger.returned(entry, result)
        issued = result[4].get("issued_actions")
        if (
            isinstance(issued, (int, np.integer))
            and not isinstance(issued, (bool, np.bool_))
            and 0 <= issued <= len(actions)
        ):
            self._last_issued_action = (
                actions[int(issued) - 1].copy() if issued else previous
            )
            self._command_history_unknown = previously_unknown if issued == 0 else False
        return result

    def _accept_execution(
        self, result: tuple, *, update_flags: bool = True
    ) -> dict[str, Any]:
        """Update cached observations and episode flags from one returned submission."""
        observation, _, terminated, truncated, info = result
        if update_flags:
            self._terminated = _termination_flag(terminated)
            self._truncated = _termination_flag(truncated)
        if isinstance(observation, list):
            if observation:
                self.last_observation = observation[-1]
        else:
            self.last_observation = observation
        self._session_ended = info.get("stop_reason") == "session_ended"
        return info

    def _execution_summary(
        self, result: tuple, *, requested_actions: int, default_reason: str
    ) -> dict[str, Any]:
        """Validate direct joint feedback before treating its counts as execution."""
        previous_observation = self.last_observation
        info = self._accept_execution(result)
        counts, errors = _action_counts(info, requested_actions)
        fresh = info.get("observation_after_last_issued_action")
        if errors:
            counts = dict.fromkeys(COUNT_NAMES)
            self._last_issued_action = None
            self._command_history_unknown = True
            self.last_observation = previous_observation
            fresh = False
        return {
            **counts,
            "observation_fresh": fresh,
            "observation_after_last_issued_action": fresh,
            "session_ended": self._session_ended,
            "stop_reason": "invalid_execution_feedback"
            if errors
            else info.get("stop_reason", default_reason),
            "terminated": self._terminated,
            "truncated": self._truncated,
            **(
                {"error": "; ".join(errors), "code": "invalid_execution_feedback"}
                if errors
                else {}
            ),
        }

    def _robot_state(self, observation: dict[str, Any]) -> dict[str, Any]:
        return decode_robot_state(
            extract_public_proprio(observation, task_id=self.task_id)
        )

    def _current_hold_action(self, state: dict[str, Any]) -> tuple[np.ndarray, str]:
        """Build a stationary native23 target without fabricating command history.

        R1Pro's official eval/r1pro.yaml uses smooth position gripper control.
        MultiFingerGripperController maps [-1,1] to the mean finger joint limits;
        the R1Pro URDF sets both prismatic fingers to [0,0.05] metres. Only before
        any issued command, symmetric stationary measured fingers can establish
        a *new* shared position target. Contact can make measurements differ from
        the commanded target, so an existing gripper command always takes priority.
        """
        if self._command_history_unknown:
            raise ValueError("previous action execution outcome is unknown")
        native = np.zeros(23, dtype=np.float32)
        native[3:7] = state["trunk_qpos"]
        native[7:14] = state["left"]["arm_qpos"]
        native[15:22] = state["right"]["arm_qpos"]
        if self._last_issued_action is not None:
            native[[14, 22]] = np.clip(self._last_issued_action[[14, 22]], -1.0, 1.0)
            source = "last_issued_native_action"
        else:
            for arm, index in (("left", 14), ("right", 22)):
                fingers = np.asarray(state[arm]["finger_qpos"], dtype=np.float64)
                velocity = np.asarray(state[arm]["finger_qvel"], dtype=np.float64)
                if (
                    fingers.shape != (2,)
                    or velocity.shape != (2,)
                    or not np.isfinite(fingers).all()
                    or not np.isfinite(velocity).all()
                    or np.any(fingers < -0.001)
                    or np.any(fingers > 0.051)
                    or abs(fingers[0] - fingers[1]) > 0.003
                    or np.any(np.abs(velocity) > 0.002)
                ):
                    raise ValueError(
                        f"{arm} fingers cannot establish a measured hold: require "
                        "0..0.05 m within 1 mm, symmetry within 3 mm and "
                        "speed at most 0.002 m/s"
                    )
                native[index] = np.clip(2.0 * fingers.mean() / 0.05 - 1.0, -1.0, 1.0)
            source = "new_measured_position_hold"
        if not np.isfinite(native).all():
            raise ValueError("native hold target must be finite")
        return native, source

    def _prepare_joint_action(self) -> tuple[dict, np.ndarray, str] | ToolResult:
        """Read one hold state shared by direct gripper and wrist-joint tools."""
        self._check_cancelled()
        ended = self._ended()
        if ended or self._terminated is True or self._truncated is True:
            return _unissued(
                error="episode has ended",
                session_ended=ended,
                terminated=self._terminated,
                truncated=self._truncated,
                observation_fresh=False,
                stop_reason="session_ended"
                if ended
                else ("terminated" if self._terminated else "truncated"),
            )
        if self._command_history_unknown:
            return _unissued(
                error="previous action execution outcome is unknown",
                code="command_state_unknown",
            )
        self.last_observation = self.env.get_observation()
        try:
            state = self._robot_state(self.last_observation)
            native, source = self._current_hold_action(state)
            return state, native, source
        except (TypeError, ValueError) as error:
            return _unissued(error=str(error), code="invalid_hold_state")

    def _prepare_base_action(self, data: dict[str, Any]) -> tuple | ToolResult:
        """Capture an atomic start frame and fixed non-base commands for one segment."""
        from rpent.tools.toolkit import ToolCancelled

        try:
            self._check_cancelled()
            if self._ended() or self._terminated is True or self._truncated is True:
                data.update(
                    session_ended=self._session_ended,
                    terminated=self._terminated,
                    truncated=self._truncated,
                    stop_reason="session_ended"
                    if self._session_ended
                    else "episode_ended",
                )
                return ToolResult(data=data)
            snapshot_reader = getattr(self.env, "get_motion_snapshot", None)
            if not callable(snapshot_reader):
                raise ValueError("move_base requires the atomic public motion snapshot")
            snapshot = snapshot_reader()
            expected = validate_observation_token(snapshot["observation_token"])
            context = snapshot["motion_context"]
            self.last_observation = snapshot["observation"]
            proprio = extract_public_proprio(
                self.last_observation, task_id=self.task_id
            )
            validate_public_base_context(context, expected, proprio)
            if (
                self._last_issued_action is None
                and context.get("initial_bootstrap") is not True
            ):
                raise ValueError(
                    "a new measured gripper hold requires the initial session snapshot; "
                    "existing actions have no known gripper command in this tool session"
                )
            state = decode_robot_state(proprio)
            hold, hold_source = self._current_hold_action(state)
        except ToolCancelled as error:
            data.update(
                error=str(error), interrupted=True, stop_reason="tool_cancelled"
            )
            return ToolResult(data=data)
        except (
            KeyError,
            OSError,
            RpcError,
            RuntimeError,
            TypeError,
            ValueError,
        ) as error:
            data.update(error=str(error), stop_reason="base_start_unavailable")
            return ToolResult(data=data)

        hold = np.asarray(hold, dtype=np.float32).copy()
        if hold.shape != (23,) or not np.isfinite(hold).all():
            data.update(
                error="invalid native23 hold target",
                stop_reason="base_start_unavailable",
            )
            return ToolResult(data=data)
        # Existing hold owner fixes all 20 non-base channels once for this call.
        # Its original command-history checks also protect partial/unknown grasps.
        hold[:3] = 0.0
        data["observation_fresh"] = True
        return hold, proprio.copy(), expected

    # Planner tools; registration still applies the configured allowlist.

    @tool
    def move_base(
        self,
        *,
        forward: Annotated[float, Field(ge=-1.0, le=1.0, allow_inf_nan=False)] = 0.0,
        lateral: Annotated[float, Field(ge=-1.0, le=1.0, allow_inf_nan=False)] = 0.0,
        turn: Annotated[float, Field(ge=-1.0, le=1.0, allow_inf_nan=False)] = 0.0,
        steps: Annotated[int, Field(ge=1, le=5000, strict=True)] = 10,
    ) -> ToolResult:
        """Apply a short local base velocity segment, then confirm a zero-base command.

        Use only for a nearby adjustment through visibly clear space. This is
        not world-coordinate navigation or obstacle avoidance. Commands are normalized velocities, not metres of
        displacement. The combined forward/lateral norm and the turn magnitude
        must each not exceed 1.0. Steps run at 30 Hz, plus one stopping action.
        Both arms and the trunk keep their initial measured joint targets; both
        grippers retain their known commands. This holds joints relative to the
        robot, not the end-effectors fixed in the world. No VLA call is made.

        Completion means every requested action and the final zero-base command
        were confirmed. It does not require zero measured velocity, and no
        displacement or odometry is reported. Use the new images to judge the
        effect and choose the next action. A failed or unknown action stops
        further submissions and is not replayed.

        Args:
            forward: Local base +X command; 0.1 requests 0.075 m/s.
            lateral: Local base +Y command; 0.1 requests 0.075 m/s.
            turn: Local positive yaw command; 0.1 requests 0.1 rad/s.
            steps: Moving control actions at 30 Hz; default 10.
        """
        from rpent.tools.toolkit import ToolCancelled

        native_velocity = validate_base_request(forward, lateral, turn, steps)
        data: dict[str, Any] = {
            "completed": False,
            "target_base_velocity_m_s_rad_s": (
                native_velocity * BASE_OUTPUT_SCALE
            ).tolist(),
            "frame": "current_robot_base",
            "requested_move_steps": steps,
            "requested_actions": 0,
            **dict.fromkeys(COUNT_NAMES, 0),
            "confirmed_move_steps": 0,
            "stop_command_confirmed": False,
            "observation_fresh": False,
            "session_ended": False,
            "collision_checked": False,
        }
        prepared = self._prepare_base_action(data)
        if isinstance(prepared, ToolResult):
            return prepared
        hold, previous_proprio, expected = prepared
        for index in range(steps + 1):
            stopping = index == steps
            try:
                self._check_cancelled()
            except ToolCancelled as error:
                data.update(
                    error=str(error), interrupted=True, stop_reason="tool_cancelled"
                )
                break
            command = hold.copy()
            if not stopping:
                command[:3] = native_velocity
            prefix = {name: data[name] for name in ("requested_actions", *COUNT_NAMES)}
            data["requested_actions"] += 1
            try:
                result = self._send_actions(
                    command[None, :], expected_observation_token=dict(expected)
                )
                if not isinstance(result, (tuple, list)) or len(result) != 5:
                    raise ValueError("base execution must return a Gym 5-tuple")
                observation, _, terminated, truncated, info = result
                if not isinstance(info, dict):
                    raise ValueError("base execution info must be a dictionary")
                counters = one_action_counts(info)
                self._terminated = _termination_flag(terminated)
                self._truncated = _termination_flag(truncated)
            except (
                AttributeError,
                IndexError,
                OSError,
                RpcError,
                RuntimeError,
                TypeError,
                ValueError,
            ) as error:
                self._command_history_unknown = True
                self._last_issued_action = None
                data.update(dict.fromkeys(COUNT_NAMES))
                data.update(
                    error=str(error),
                    stop_reason="execution_outcome_unknown",
                    known_execution_before_failure=prefix,
                    observation_fresh=False,
                )
                break

            for name, count in counters.items():
                data[name] += count
            fresh = info.get("observation_after_last_issued_action") is True
            self._session_ended = info.get("stop_reason") == "session_ended"
            data.update(
                session_ended=self._session_ended,
                terminated=self._terminated,
                truncated=self._truncated,
                observation_fresh=fresh,
            )
            try:
                token = validate_observation_token(info.get("observation_token"))
                if (
                    info.get("motion_feedback_error")
                    or token["generation"] != expected["generation"]
                    or token["observation_index"]
                    != expected["observation_index"] + counters["confirmed_actions"]
                    or not isinstance(info.get("motion_context"), dict)
                    or info["motion_context"].get("current_token") != token
                ):
                    raise ValueError(
                        "base feedback is not bound to the confirmed frame"
                    )
            except (TypeError, ValueError) as error:
                self._command_history_unknown = True
                self._last_issued_action = None
                data.update(
                    error=str(error),
                    stop_reason="invalid_execution_feedback",
                    observation_fresh=False,
                )
                break
            confirmed = counters["confirmed_actions"] == 1 and fresh
            if not confirmed:
                if counters["issued_actions"] or fresh:
                    self._command_history_unknown = True
                    self._last_issued_action = None
                data.update(
                    error="base action lacks its confirmed fresh observation",
                    stop_reason="session_ended"
                    if self._session_ended
                    else "partial_execution",
                    observation_fresh=False,
                )
                break

            try:
                observation, after, token = _validate_single_step_observation(
                    observation, info, expected, previous_proprio, task_id=self.task_id
                )
            except (KeyError, TypeError, ValueError) as error:
                self._command_history_unknown = True
                self._last_issued_action = None
                data.update(
                    error=f"invalid public base feedback: {error}",
                    stop_reason="invalid_public_observation",
                    observation_fresh=False,
                )
                break

            self.last_observation = observation
            previous_proprio = after.copy()
            expected = token
            if stopping:
                data.update(
                    completed=True,
                    stop_command_confirmed=True,
                    stop_reason="velocity_segment_completed",
                )
            else:
                data["confirmed_move_steps"] += 1
            if (
                self._session_ended
                or self._terminated is True
                or self._truncated is True
            ):
                if not stopping:
                    data.update(
                        error="episode ended before the stop command",
                        stop_reason="session_ended",
                    )
                break
        return ToolResult(data=deepcopy(data))

    @tool
    def vla_act(
        self,
        *,
        chunks: Annotated[int, Field(ge=1, json_schema_extra={"default": 1})] = 1,
        prompt: str | None = None,
    ) -> ToolResult:
        """Predict and execute bounded VLA chunks, refreshing observations between chunks.

        Failures retain the confirmed execution prefix. An uncertain submission
        remains unknown and is never replayed.

        Args:
            chunks: Number of prediction/execution cycles within this run's call limit.
            prompt: Policy instruction; main mode permits only the configured task text.
        """
        from rpent.tools.toolkit import ToolCancelled

        if type(chunks) is not int or chunks < 1:
            raise ValueError("chunks must be a positive integer")
        if self.max_chunks_per_call is not None and chunks > self.max_chunks_per_call:
            raise ValueError(
                f"at most {self.max_chunks_per_call} VLA chunks are allowed per call"
            )
        instruction = self.task_language if prompt is None else prompt
        if not instruction.strip():
            raise ValueError("VLA instruction must not be empty")
        if self.instruction_mode == "main" and instruction != self.task_language:
            raise ValueError(
                "instruction_mode=main requires the full configured task instruction; "
                "omit prompt or pass that exact text"
            )
        submitted = 0
        counts: dict[str, int | None] = dict.fromkeys(COUNT_NAMES, 0)
        failure: dict[str, Any] = {}
        observation_fresh: bool | None = None
        stop_reason = "requested_chunks_returned"
        for _ in range(chunks):
            prefix = {"requested_actions": submitted, **counts}
            previous_observation = self.last_observation
            pending = False
            phase = "preparation"
            try:
                self._check_cancelled()
                if self._ended():
                    stop_reason = "session_ended"
                    observation_fresh = False
                    break
                if self._terminated is True or self._truncated is True:
                    stop_reason = "terminated" if self._terminated else "truncated"
                    break
                phase = "observation"
                self.last_observation = self.env.get_observation()
                observation = prepare_behavior_observation(
                    self.last_observation,
                    task_id=self.task_id,
                    task_language=self.task_language,
                )
                phase = "prediction"
                actions = self.model.predict(
                    {
                        **observation["policy_observation"],
                        "task_descriptions": instruction,
                    }
                )[: self.execution_horizon]
                self._check_cancelled()
                phase = "execution"
                submitted += len(actions)
                pending = True
                info = self._accept_execution(self._send_actions(actions))
                segment, errors = _action_counts(info, len(actions))
                if errors:
                    raise ValueError("; ".join(errors))
                for key, total in counts.items():
                    value = segment[key]
                    counts[key] = (
                        total + value
                        if total is not None and value is not None
                        else None
                    )
                pending = False
                observation_fresh = info.get("observation_after_last_issued_action")
                if self._session_ended:
                    stop_reason = "session_ended"
                    break
                if self._terminated is True or self._truncated is True:
                    stop_reason = "terminated" if self._terminated else "truncated"
                    break
                count = segment["executed_actions"]
                if count is not None and count < len(actions):
                    stop_reason = "partial_execution"
                    break
            except Exception as error:
                if submitted == 0:
                    raise
                if pending:
                    counts = dict.fromkeys(COUNT_NAMES)
                    self._command_history_unknown = True
                    self._last_issued_action = None
                    failure["known_execution_before_failure"] = prefix
                if previous_observation is not None:
                    self.last_observation = previous_observation
                observation_fresh = False
                cancelled = isinstance(error, ToolCancelled)
                stop_reason = (
                    "tool_cancelled"
                    if cancelled
                    else "execution_outcome_unknown"
                    if pending
                    else f"{phase}_failed"
                )
                failure.update(
                    error=str(error),
                    error_type=type(error).__name__,
                    code=stop_reason,
                    interrupted=cancelled,
                )
                break
        return ToolResult(
            data={
                "requested_chunks": chunks,
                "requested_actions": submitted,
                **counts,
                "observation_fresh": observation_fresh,
                "observation_after_last_issued_action": observation_fresh,
                "session_ended": self._session_ended,
                "stop_reason": stop_reason,
                "terminated": self._terminated,
                "truncated": self._truncated,
                "prompt": instruction,
                **failure,
            }
        )

    @tool
    def move_to(
        self,
        *,
        arm: Literal["left", "right"],
        xyz: Annotated[list[float], Field(min_length=3, max_length=3)],
        quat: Annotated[list[float], Field(min_length=4, max_length=4)] | None = None,
        settle_steps: Annotated[
            int, Field(ge=0, le=180, json_schema_extra={"default": 0})
        ] = 0,
    ) -> ToolResult:
        """Reposition one hand to a nearby pose through visible free space.

        Use for a clear local approach correction or retreat, within the
        documented single-call repositioning bound of the current EEF. Supply a free-space EEF target, not a raw object
        surface point. CuRobo solves IK for one seven-joint arm while the trunk and
        other arm hold position, the base stops, and both grippers hold their
        settled commands. Planning requires a stationary start. At the strict initial frame,
        one explicitly counted native hold action obtains an adjacent public
        observation before planning; unknown prior execution cannot be
        recovered this way. This does not grasp, press, carry or navigate.

        Planning solves IK for the requested pose only and interpolates in
        joint space; it does not check the path against the scene, so allow
        visible clearance. If IK fails, no path actions are sent and the
        result reports the nearest reachable xyz; an
        already confirmed initial sampling hold remains in the action counts.
        After motion,
        inspect the returned images and measured EEF error. target_reached
        confirms only the requested pose, not task success.
        Completion requires every submitted action to be confirmed and the
        latest measured EEF pose to reach the target. Residual joint velocities
        are diagnostic feedback, not a completion gate. By default return after
        the path without extra holds. If settle_steps is positive and the pose
        has not reached the target, repeat the same final command up to that
        budget, returning as soon as the pose reaches it. Partial, unknown or
        stale feedback stops execution without replay. The next action tool
        checks its own planning prerequisites against the latest observation.

        Args:
            arm: The single seven-joint arm to move.
            xyz: Absolute [x,y,z] metres in the current robot-base frame.
            quat: Unit [x,y,z,w]; omit to keep the current EEF orientation.
            settle_steps: Optional maximum final-target actions if the measured
                pose has not reached the target, 0 to 180, default 0.
        """
        return self._run_arm_motion(
            arm=arm, xyz=xyz, quat=quat, settle_steps=settle_steps
        )

    @tool
    def rotate_wrist(
        self,
        *,
        arm: Literal["left", "right"],
        delta_yaw_deg: Annotated[
            float, Field(ge=-180, le=180, allow_inf_nan=False, strict=True)
        ],
        settle_steps: Annotated[
            int, Field(ge=0, le=180, json_schema_extra={"default": 0})
        ] = 0,
    ) -> ToolResult:
        """Adjust one hand's yaw about robot-base Z; reachability is decided by IK.

        Use for a small visible approach-orientation correction in free space.
        Positive angles follow the right-hand rule about base +Z, not the wrist's
        local axis. The target keeps the current EEF xyz; intermediate positions
        are determined by CuRobo and are not constrained to stay fixed. The whole
        seven-joint arm can move, and fingertips sweep an arc. Allow clearance.

        This uses the same confirmed-snapshot planning and execution as move_to;
        planning starts stationary. The base stops, the trunk and other arm hold,
        and both gripper commands remain unchanged. This does not turn a grasped
        knob, press a button, carry an object or rotate the base.

        Read target_reached, position_error_m, orientation_error_rad and the
        confirmed/unconfirmed action counts. Completion requires the target pose
        and confirmed execution, not zero terminal velocity or task success.

        Args:
            arm: The single seven-joint arm to reorient.
            delta_yaw_deg: Relative robot-base Z rotation in degrees, -180 to 180.
            settle_steps: Optional final-target action budget if the measured pose
                is not reached, 0 to 180, default 0; not a delay or speed gate.
        """
        if (
            isinstance(delta_yaw_deg, (bool, np.bool_))
            or not isinstance(delta_yaw_deg, Real)
            or not math.isfinite(delta_yaw_deg)
            or not -180 <= delta_yaw_deg <= 180
        ):
            raise ValueError("delta_yaw_deg must be finite and between -180 and 180")
        angle = float(delta_yaw_deg)
        result = self._run_arm_motion(
            arm=arm, xyz=None, settle_steps=settle_steps, delta_yaw_deg=angle
        )
        result.data.update(
            requested_delta_yaw_deg=angle,
            rotation_axis="robot_base_z",
        )
        return result

    @tool
    def rotate_wrist_joint(
        self,
        *,
        arm: Literal["left", "right"],
        joint: Literal["roll", "pitch", "yaw"],
        delta_deg: Annotated[
            float, Field(ge=-90, le=90, allow_inf_nan=False, strict=True)
        ],
        steps: Annotated[
            int, Field(ge=1, le=120, json_schema_extra={"default": 20})
        ] = 20,
        settle_steps: Annotated[
            int, Field(ge=0, le=60, json_schema_extra={"default": 10})
        ] = 10,
    ) -> ToolResult:
        """Turn one wrist joint by a relative angle without IK planning.

        Use when the hand is already at the target and only its orientation
        needs a small change: roll turns the gripper opening about the forearm
        axis (align the fingers with a handle), pitch tips the palm toward or
        away from the forearm (face a control from above), yaw bends the hand
        sideways. The hand position moves along a short arc about the wrist;
        allow clearance. Unlike rotate_wrist this always executes: it sends a
        linear joint ramp, then holds the final target for settle_steps. The
        other joints, the trunk, the base and both gripper commands are held.

        Read measured_delta_deg and the wrist image afterward. A joint-limit
        clip is reported, not silently applied.

        Args:
            arm: The arm whose wrist turns.
            joint: roll, pitch or yaw of the wrist, in the robot's wrist order.
            delta_deg: Relative joint change in degrees, -90 to 90.
            steps: Interpolated native actions for the ramp, default 20.
            settle_steps: Additional final-target actions, 0 to 60, default 10.
        """
        if arm not in ("left", "right"):
            raise ValueError("arm must be left or right")
        if joint not in WRIST_JOINT_OFFSET:
            raise ValueError("joint must be roll, pitch or yaw")
        if (
            isinstance(delta_deg, (bool, np.bool_))
            or not isinstance(delta_deg, Real)
            or not math.isfinite(delta_deg)
            or not -90 <= delta_deg <= 90
        ):
            raise ValueError("delta_deg must be finite and between -90 and 90")
        if type(steps) is not int or not 1 <= steps <= 120:
            raise ValueError("steps must be an integer from 1 to 120")
        if type(settle_steps) is not int or not 0 <= settle_steps <= 60:
            raise ValueError("settle_steps must be an integer from 0 to 60")
        prepared = self._prepare_joint_action()
        if isinstance(prepared, ToolResult):
            return prepared
        state, native, hold_source = prepared
        arm_slice = slice(7, 14) if arm == "left" else slice(15, 22)
        joint_index = arm_slice.start + WRIST_JOINT_OFFSET[joint]
        limits = self.env.arm_joint_limits[arm][WRIST_JOINT_OFFSET[joint]]
        current = float(native[joint_index])
        requested = current + math.radians(float(delta_deg))
        target = float(np.clip(requested, limits[0], limits[1]))
        clipped = abs(target - requested) > 1e-9
        actions = _build_joint_ramp_actions(
            native, joint_index, current, target, steps, settle_steps
        )
        self._check_cancelled()
        feedback = self._execution_summary(
            self._send_actions(actions),
            requested_actions=len(actions),
            default_reason="requested_joint_ramp_returned",
        )
        fresh = feedback["observation_fresh"]
        measured_after = None
        measured_delta_deg = None
        measurement_error = None
        if fresh is True:
            try:
                after_arm = self._robot_state(self.last_observation)[arm]
                measured_after = float(after_arm["arm_qpos"][WRIST_JOINT_OFFSET[joint]])
                measured_delta_deg = math.degrees(
                    measured_after
                    - float(state[arm]["arm_qpos"][WRIST_JOINT_OFFSET[joint]])
                )
            except (TypeError, ValueError, IndexError) as error:
                measurement_error = f"post-motion joint could not be evaluated: {error}"
        return ToolResult(
            data={
                "arm": arm,
                "joint": joint,
                "joint_name": f"{arm}_arm_joint{WRIST_JOINT_OFFSET[joint] + 1}",
                "requested_delta_deg": float(delta_deg),
                "commanded_delta_deg": math.degrees(target - current),
                "clipped_to_joint_limit": clipped,
                "joint_limit_rad": limits.tolist(),
                "measured_delta_deg": measured_delta_deg,
                "measured_joint_rad_after": measured_after,
                "requested_actions": len(actions),
                "ramp_steps": steps,
                "settle_steps": settle_steps,
                **feedback,
                "gripper_hold_source": hold_source,
                **(
                    {"error": measurement_error}
                    if measurement_error is not None
                    else {}
                ),
            }
        )

    @tool
    def set_gripper(
        self,
        *,
        arm: Literal["left", "right"],
        val: Annotated[float, Field(ge=0, le=1)],
        steps: Annotated[
            int, Field(ge=1, le=MAX_GRIPPER_ACTIONS, json_schema_extra={"default": 10})
        ] = 10,
        settle_steps: Annotated[
            int, Field(ge=0, le=60, json_schema_extra={"default": 10})
        ] = 10,
    ) -> ToolResult:
        """Interpolate one gripper command to val: 0 closed, 1 open.

        Sends a linear command sequence to R1Pro's native [-1,1] smooth
        position controller (0..0.05 m per finger). Before any action,
        symmetric stationary fingers may establish a new measured hold target.
        Otherwise the last issued gripper commands are preserved, including the
        other hand's command under contact. Unknown execution feedback prevents
        further gripper commands. Arms and trunk hold measured joint positions;
        base velocity is zero. After interpolation, repeat the final target for
        settle_steps so the controller can approach it. Returned positions and
        velocities determine settling; they do not prove a grasp or object release.

        Args:
            arm: Gripper to command.
            val: Normalized target command, 0 closed and 1 open.
            steps: Interpolated actions, 1 to 5000; ramp plus holds must not
                exceed 5000 actions. Default 10.
            settle_steps: Additional final-target actions, 0 to 60, default 10.
        """
        if arm not in ("left", "right"):
            raise ValueError("arm must be left or right")
        if (
            isinstance(val, (bool, np.bool_))
            or not isinstance(val, (int, float, np.integer, np.floating))
            or not np.isfinite(val)
            or not 0 <= val <= 1
        ):
            raise ValueError("val must be finite and between 0 and 1")
        if type(steps) is not int or not 1 <= steps <= MAX_GRIPPER_ACTIONS:
            raise ValueError(
                f"steps must be an integer from 1 to {MAX_GRIPPER_ACTIONS}"
            )
        if type(settle_steps) is not int or not 0 <= settle_steps <= 60:
            raise ValueError("settle_steps must be an integer from 0 to 60")
        if steps + settle_steps > MAX_GRIPPER_ACTIONS:
            raise ValueError(
                f"gripper action budget is at most {MAX_GRIPPER_ACTIONS} including holds"
            )
        prepared = self._prepare_joint_action()
        if isinstance(prepared, ToolResult):
            return prepared
        state, native, hold_source = prepared
        gripper_index = 14 if arm == "left" else 22
        current = float((native[gripper_index] + 1.0) / 2.0)
        target = float(val)
        actions = _build_joint_ramp_actions(
            native,
            gripper_index,
            current,
            target,
            steps,
            settle_steps,
            scale=2.0,
            offset=-1.0,
        )
        self._check_cancelled()
        feedback = self._execution_summary(
            self._send_actions(actions),
            requested_actions=len(actions),
            default_reason="requested_gripper_returned",
        )
        fresh = feedback["observation_fresh"]
        measured_finger_qpos = None
        measured_finger_qvel = None
        target_error = None
        position_target_reached = None
        settled = None
        measurement_error = None
        if fresh is True:
            try:
                after_state = self._robot_state(self.last_observation)[arm]
                measured_finger_qpos = after_state["finger_qpos"]
                measured_finger_qvel = after_state["finger_qvel"]
                error = np.asarray(measured_finger_qpos) - 0.05 * target
                target_error = error.tolist()
                position_target_reached = bool(np.all(np.abs(error) <= 0.001))
                settled = position_target_reached and bool(
                    np.all(np.abs(measured_finger_qvel) <= 0.002)
                )
            except (TypeError, ValueError) as error:
                measurement_error = (
                    f"post-gripper state could not be evaluated: {error}"
                )
        return ToolResult(
            data={
                "arm": arm,
                "val": target,
                "start_val": current,
                "start_source": hold_source,
                "native_target_command": 2.0 * target - 1.0,
                "target_finger_qpos_m": [0.05 * target, 0.05 * target],
                "requested_actions": len(actions),
                "ramp_steps": steps,
                "settle_steps": settle_steps,
                **feedback,
                "finger_qpos_before_m": state[arm]["finger_qpos"],
                "finger_qpos_after_m": measured_finger_qpos,
                "finger_qvel_before_m_s": state[arm]["finger_qvel"],
                "finger_qvel_after_m_s": measured_finger_qvel,
                "finger_target_position_error_m": target_error,
                "finger_target_position_reached": position_target_reached,
                "finger_target_position_tolerance_m": 0.001,
                "settled": settled,
                "finger_settled_velocity_tolerance_m_s": 0.002,
                "other_gripper_hold_source": hold_source,
                "other_gripper_native_hold_command": float(
                    native[22 if arm == "left" else 14]
                ),
                "grasp_success": None,
                **(
                    {"error": measurement_error}
                    if measurement_error is not None
                    else {}
                ),
            }
        )

    @tool
    def release(
        self,
        *,
        arm: Literal["left", "right"],
        val: Annotated[
            float, Field(ge=1, le=1, json_schema_extra={"default": 1.0})
        ] = 1.0,
        steps: Annotated[
            int, Field(ge=1, le=MAX_GRIPPER_ACTIONS, json_schema_extra={"default": 10})
        ] = 10,
        settle_steps: Annotated[
            int, Field(ge=0, le=60, json_schema_extra={"default": 10})
        ] = 10,
    ) -> ToolResult:
        """Compatibility alias for set_gripper(val=1); opens the selected gripper.

        Execution counts and measured finger positions report feedback, not
        proof that an object was released.

        Args:
            arm: Gripper to open.
            val: Opening target; this alias accepts only 1.
            steps: Actions used to open; ramp plus holds must not exceed 5000.
            settle_steps: Additional actions holding the open target.
        """
        if (
            isinstance(val, (bool, np.bool_))
            or not isinstance(val, (int, float, np.integer, np.floating))
            or val != 1
        ):
            raise ValueError("release only opens the gripper; val must be 1")
        return self.set_gripper(
            arm=arm, val=1.0, steps=steps, settle_steps=settle_steps
        )

    @tool
    def move_eef_delta(
        self,
        *,
        arm: Literal["left", "right"],
        delta_xyz: Annotated[list[float], Field(min_length=3, max_length=3)],
    ) -> ToolResult:
        """Move one EEF by a small relative step (single-call bound) in robot-base xyz, keeping its orientation.

        Resolves the target from the current confirmed snapshot (current EEF
        position plus delta_xyz) and then uses the same CuRobo IK planning
        and execution as move_to, so the same preconditions apply:
        stationary robot, base stopped.

        Args:
            arm: The single arm to move.
            delta_xyz: Relative [dx,dy,dz] in metres in the robot-base frame.
        """
        delta = np.asarray(delta_xyz, dtype=np.float64)
        if (
            delta.shape != (3,)
            or not np.isfinite(delta).all()
            or not 0 < np.linalg.norm(delta) <= MAX_EEF_DELTA_M
        ):
            raise ValueError(
                f"delta_xyz must be finite, nonzero and at most {MAX_EEF_DELTA_M:g} m"
            )
        result = self._run_arm_motion(arm=arm, xyz=None, delta_xyz=delta)
        result.data.update(requested_delta_xyz=delta.tolist(), frame="robot_base")
        return result

    # Arm planning, initial sampling, settling and execution.

    def _run_arm_motion(
        self, *, arm, xyz, quat=None, settle_steps=0, delta_yaw_deg=None, delta_xyz=None
    ) -> ToolResult:
        """Share the motion ledger for absolute poses and same-snapshot rotations."""
        from rpent.tools.toolkit import ToolCancelled

        if getattr(self, "_move_call_ledger", None) is not None:
            raise RuntimeError("an arm motion invocation is already active")
        ledger = MotionCallLedger()
        self._auto_settle_report = None
        self._move_call_ledger = ledger
        self._move_call_phase = "path"
        try:
            try:
                result = self._plan_and_execute_arm_motion(
                    arm=arm,
                    xyz=xyz,
                    quat=quat,
                    settle_steps=settle_steps,
                    _delta_yaw_deg=delta_yaw_deg,
                    _delta_xyz=delta_xyz,
                )
            except Exception as error:
                if not ledger.submissions:
                    raise
                if (
                    ledger.submissions[-1]["phase"] == "preflight"
                    and ledger.preflight["status"] != "confirmed"
                ):
                    ledger.preflight["status"] = "execution_outcome_unknown"
                    ledger.submissions[-1]["observation_fresh"] = False
                    self._command_history_unknown = True
                    self._last_issued_action = None
                # The native Toolkit catches these exceptions but otherwise
                # drops action counters. Return the real spent/unknown prefix
                # and keep its cancellation flag; never submit or retry again.
                result = ToolResult(
                    data={
                        "error": str(error),
                        "error_type": type(error).__name__,
                        "code": "tool_cancelled"
                        if isinstance(error, ToolCancelled)
                        else "execution_outcome_unknown"
                        if not ledger.submissions[-1]["counts_valid"]
                        else "motion_call_failed",
                        "interrupted": isinstance(error, ToolCancelled),
                        "completed": False,
                        "target_reached": None,
                    }
                )
            result.data = ledger.merge(result.data)
            report = getattr(self, "_auto_settle_report", None)
            if report is not None:
                result.data.setdefault("auto_settle", report)
            return result
        finally:
            self._move_call_ledger = None
            self._move_call_phase = None

    def _initial_motion_probe(self, snapshot, native, planning_state):
        """One counted initial hold, never an implicit observation-side step."""

        ledger = self._move_call_ledger
        if self._last_issued_action is not None or ledger.submissions:
            return _unissued(
                error="initial sampling requires no prior command or probe",
                code="initial_command_history_mismatch",
            )
        unsettled = set(planning_state["unsettled_groups"])
        if unsettled & {"base", "left_fingers", "right_fingers"}:
            return _unissued(
                error="initial hold sampling requires the original base/finger preconditions",
                code="base_not_stationary"
                if "base" in unsettled
                else "gripper_state_unsettled",
                unsettled_groups=sorted(unsettled),
            )
        expected = dict(snapshot["observation_token"])
        before = extract_public_proprio(
            snapshot["observation"], task_id=self.task_id
        ).copy()
        validate_public_base_context(snapshot["motion_context"], expected, before)
        action = np.asarray(native, dtype=np.float32).reshape(1, 23).copy()
        if not np.isfinite(action).all() or np.any(action[:, :3] != 0):
            return _unissued(
                error="invalid initial native23 hold", code="invalid_hold_state"
            )
        ledger.preflight["status"] = "requested"
        self._check_cancelled()
        self._move_call_phase = "preflight"
        returned, _, terminated, truncated, info = self._send_actions(
            action, expected_observation_token=expected
        )
        self._terminated, self._truncated = (
            _termination_flag(terminated),
            _termination_flag(truncated),
        )
        self._session_ended = info.get("stop_reason") == "session_ended"
        try:
            counts = one_action_counts(info)
            if (
                counts
                != {
                    "issued_actions": 1,
                    "confirmed_actions": 1,
                    "unconfirmed_actions": 0,
                    "executed_actions": 1,
                }
                or info.get("observation_after_last_issued_action") is not True
                or info.get("motion_feedback_error")
            ):
                raise ValueError(
                    "initial sampling action lacks one confirmed fresh observation"
                )
            returned, _, token = _validate_single_step_observation(
                returned, info, expected, before, task_id=self.task_id
            )
            context = info["motion_context"]
            if context.get("initial_bootstrap") is not False:
                raise ValueError("initial sampling did not end the bootstrap state")
        except (KeyError, TypeError, ValueError) as error:
            ledger.preflight["status"] = "feedback_invalid"
            ledger.submissions[-1]["observation_fresh"] = False
            self._command_history_unknown = True
            self._last_issued_action = None
            return _unissued(
                error=str(error),
                code="initial_sampling_unconfirmed",
                completed=False,
                observation_fresh=False,
                session_ended=self._session_ended,
            )
        self.last_observation = returned
        ledger.preflight["status"] = "confirmed"
        self._move_call_phase = "path"
        return {
            "observation": returned,
            "observation_token": token,
            "motion_context": context,
        }

    def _arm_start(self, snapshot: dict[str, Any]) -> tuple[dict, dict]:
        """Read robot state and planning prerequisites from one bound public frame."""
        from robots.behavior.motion import R1ProMotionModel

        self.last_observation = snapshot["observation"]
        proprio = extract_public_proprio(self.last_observation, task_id=self.task_id)
        context = snapshot["motion_context"]
        if context.get("current_token") != snapshot["observation_token"]:
            raise ValueError("motion snapshot token differs from its public context")
        if self._last_issued_action is None and not context.get("initial_bootstrap"):
            raise ValueError(
                "a measured gripper hold requires the initial session snapshot; "
                "existing actions have no known gripper command in this tool session"
            )
        return decode_robot_state(proprio), R1ProMotionModel.planning_state_contract(
            proprio, context
        )

    def _settled_arm_start(self, snapshot, state, contract):
        """Attempt bounded holds, then return the final planning snapshot and state."""
        report = None
        if contract.get("valid") is True and not contract["stationary"]:
            report = self._auto_settle_before_planning(state)
            self._auto_settle_report = report
            if report.get("status") == "stationary":
                snapshot = self.env.get_motion_snapshot()
                state, contract = self._arm_start(snapshot)
        if contract.get("valid") is not True or not contract["stationary"]:
            code = contract.get("code") or (
                "base_not_stationary"
                if "base" in contract["unsettled_groups"]
                else "motion_state_unsettled"
            )
            return _unissued(
                error=contract.get(
                    "error",
                    "measured motion exceeds the static arm-planning tolerances",
                ),
                code=code,
                auto_settle=report,
                rejection={
                    "code": code,
                    "reason": "planning requires a settled robot; the measured motion is still above the static tolerance",
                    "unsettled_groups": contract.get("unsettled_groups"),
                    "auto_settle": report,
                    "suggested_next": "wait for motion to settle and retry, or take a different approach; retrying the identical request immediately will be rejected again",
                },
            )
        return snapshot, state

    @staticmethod
    def _arm_target(state, arm, position, orientation, delta_xyz, delta_yaw_deg):
        """Resolve absolute or relative intent once against the selected start frame."""
        measured = state[arm]
        if delta_xyz is not None:
            position = np.asarray(measured["eef_position"]) + delta_xyz
            orientation = np.asarray(measured["eef_quaternion_xyzw"])
        elif delta_yaw_deg is not None:
            from scipy.spatial.transform import Rotation

            position = np.asarray(measured["eef_position"])
            orientation = (
                Rotation.from_rotvec([0, 0, math.radians(delta_yaw_deg)])
                * Rotation.from_quat(measured["eef_quaternion_xyzw"])
            ).as_quat()
        elif orientation is None:
            orientation = np.asarray(measured["eef_quaternion_xyzw"])
            orientation = orientation / np.linalg.norm(orientation)
        target = np.concatenate((position, orientation))
        distance = float(
            np.linalg.norm(position - np.asarray(measured["eef_position"]))
        )
        if distance > MAX_MOVE_TO_REPOSITION_M:
            return _unissued(
                error=f"this motion mode permits at most {MAX_MOVE_TO_REPOSITION_M:g} m EEF repositioning",
                code="target_outside_motion_scope",
                rejection={
                    "code": "target_outside_motion_scope",
                    "reason": "requested target is beyond one call's local repositioning bound",
                    "measured_reposition_m": distance,
                    "allowed_reposition_m": MAX_MOVE_TO_REPOSITION_M,
                    "suggested_next": "split the move into successive targets inside the bound, or use move_base for the coarse part first",
                },
            )
        return target

    def _auto_settle_before_planning(self, state: dict[str, Any]) -> dict[str, Any]:
        """Spend bounded hold actions to reach a planable stationary start.

        A rejected plan costs one whole call and yields zero actions, so the
        caller is better served by settling first. Holds repeat the current
        measured joint targets and the last known gripper commands, so they
        never invent command history; the base is commanded to zero.
        """
        report: dict[str, Any] = {"attempted": 0, "confirmed": 0, "status": "unsettled"}
        if self._command_history_unknown:
            report.update(status="command_state_unknown")
            return report
        try:
            native, hold_source = self._current_hold_action(state)
        except (TypeError, ValueError) as error:
            report.update(status="hold_unavailable", error=str(error))
            return report
        native = np.asarray(native, dtype=np.float32).copy()
        native[:3] = 0.0
        if native.shape != (23,) or not np.isfinite(native).all():
            report.update(status="invalid_hold")
            return report
        action = native.reshape(1, 23)
        from robots.behavior.motion import R1ProMotionModel

        for _ in range(MAX_AUTO_SETTLE_ACTIONS):
            self._check_cancelled()
            if self._ended() or self._terminated is True or self._truncated is True:
                report["status"] = "episode_ended"
                return report
            snapshot = self.env.get_motion_snapshot()
            self.last_observation = snapshot["observation"]
            before = extract_public_proprio(self.last_observation, task_id=self.task_id)
            contract = R1ProMotionModel.planning_state_contract(
                before, snapshot["motion_context"]
            )
            if contract.get("stationary") is True:
                report["status"] = "stationary"
                return report
            self._move_call_phase = "auto_settle"
            returned, _, terminated, truncated, info = self._send_actions(
                action, expected_observation_token=snapshot["observation_token"]
            )
            report["attempted"] += 1
            self._terminated, self._truncated = (
                _termination_flag(terminated),
                _termination_flag(truncated),
            )
            confirmed = info.get("confirmed_actions")
            if type(confirmed) is int and confirmed == 1:
                report["confirmed"] += 1
                self.last_observation = (
                    returned[0] if isinstance(returned, list) else returned
                )
            else:
                self._command_history_unknown = True
                self._last_issued_action = None
                report["status"] = "hold_unconfirmed"
                return report
            if info.get("stop_reason") == "session_ended":
                self._session_ended = True
                report["status"] = "episode_ended"
                return report
        report["status"] = "still_unsettled"
        return report

    def _plan_and_execute_arm_motion(
        self,
        *,
        arm,
        xyz,
        quat=None,
        settle_steps=0,
        _delta_yaw_deg=None,
        _delta_xyz=None,
    ) -> ToolResult:
        """Prepare one start, resolve the request, plan, and execute without recursion."""
        if arm not in ("left", "right"):
            raise ValueError("arm must be left or right")
        if type(settle_steps) is not int or not 0 <= settle_steps <= 180:
            raise ValueError("settle_steps must be an integer from 0 to 180")
        position = (
            None
            if _delta_yaw_deg is not None or _delta_xyz is not None
            else np.asarray(xyz, dtype=np.float64)
        )
        if position is not None and (
            position.shape != (3,) or not np.isfinite(position).all()
        ):
            raise ValueError("xyz must contain three finite robot-base coordinates")
        orientation = None if quat is None else np.asarray(quat, dtype=np.float64)
        if orientation is not None and (
            orientation.shape != (4,)
            or not np.isfinite(orientation).all()
            or not np.isclose(np.linalg.norm(orientation), 1.0, atol=1e-4, rtol=0)
        ):
            raise ValueError("quat must be a unit xyzw quaternion")
        self._check_cancelled()
        ended = self._ended()
        if ended or self._terminated is True or self._truncated is True:
            return _unissued(
                error="episode has ended",
                code="session_ended",
                session_ended=ended,
                terminated=self._terminated,
                truncated=self._truncated,
                observation_fresh=False,
                stop_reason="session_ended"
                if ended
                else ("terminated" if self._terminated else "truncated"),
            )
        planner = getattr(self.env, "plan_arm_path", None)
        snapshot_reader = getattr(self.env, "get_motion_snapshot", None)
        if not callable(planner) or not callable(snapshot_reader):
            return _unissued(
                error="the Env bridge requires a native arm planner and atomic motion snapshots",
                code="motion_planner_unavailable",
            )
        if self._command_history_unknown:
            return _unissued(
                error="previous action execution is unknown; inspect the transport before sending another motion",
                code="command_state_unknown",
            )
        snapshot = snapshot_reader()
        target = None
        relax = _delta_yaw_deg is None and (quat is None or _delta_xyz is not None)
        try:
            state, contract = self._arm_start(snapshot)
            if (
                contract.get("valid") is True
                and contract.get("source") == "initial_public_velocity_bootstrap"
            ):
                # Freeze the requested target before the counted sampling action.
                target = self._arm_target(
                    state, arm, position, orientation, _delta_xyz, _delta_yaw_deg
                )
                if isinstance(target, ToolResult):
                    return target
                native, _ = self._current_hold_action(state)
                snapshot = self._initial_motion_probe(snapshot, native, contract)
                if isinstance(snapshot, ToolResult):
                    return snapshot
                if self._ended() or self._terminated is True or self._truncated is True:
                    return _unissued(
                        error="episode ended during initial sampling",
                        code="session_ended",
                        session_ended=self._session_ended,
                        observation_fresh=True,
                        terminated=self._terminated,
                        truncated=self._truncated,
                    )
                state, contract = self._arm_start(snapshot)
            prepared = self._settled_arm_start(snapshot, state, contract)
            if isinstance(prepared, ToolResult):
                return prepared
            snapshot, state = prepared
            # Rebuild all nonmoving targets from the final planning state.
            native, _ = self._current_hold_action(state)
        except (TypeError, ValueError) as error:
            return _unissued(error=str(error), code="invalid_hold_state")
        if target is None:
            target = self._arm_target(
                state, arm, position, orientation, _delta_xyz, _delta_yaw_deg
            )
            if isinstance(target, ToolResult):
                return target
        target = self._arm_target(state, arm, target[:3], target[3:], None, None)
        if isinstance(target, ToolResult):
            return target
        self._move_call_ledger.target_pose = target.tolist()
        token = snapshot["observation_token"]
        planned = planner(
            arm=arm,
            target_pose=target,
            expected_observation_token=token,
            allow_orientation_relax=relax,
        )
        self._check_cancelled()
        if not isinstance(planned, dict):
            raise TypeError("arm planner must return a result dictionary")
        if planned.get("status") != "Success":
            return _unissued(
                error=planned.get("error", "CuRobo could not plan this target"),
                code=planned.get("code", "plan_failed"),
                plan_status=planned.get("status"),
                stop_reason="plan_failed",
                requested_target_pose=target.tolist(),
                **{
                    name: planned[name]
                    for name in (
                        "solver_status",
                        "nearest_reachable_xyz",
                        "suggested_retreat_xyz",
                        "orientation_relax_attempted",
                        "path_points",
                    )
                    if name in planned
                },
            )
        if not planned.get("collision_scope"):
            return _unissued(
                error="planner did not describe its collision scope",
                code="collision_scope_missing",
            )
        try:
            if planned.get("observation_token") != token:
                raise ValueError(
                    "arm plan does not belong to the requested observation"
                )
            path = _validated_arm_plan(planned, arm=arm, state=state)
        except (TypeError, ValueError) as error:
            return _unissued(error=str(error), code="invalid_arm_plan")
        if self._ended():
            return _unissued(
                error="episode ended during arm planning",
                code="session_ended",
                stop_reason="session_ended",
                session_ended=True,
                observation_fresh=False,
            )
        current = snapshot_reader()
        self.last_observation = current["observation"]
        if current["observation_token"] != token:
            self._move_call_ledger.observation_invalidated_by_newer_snapshot = True
            return _unissued(
                error="public robot state changed while the path was planned",
                code="stale_arm_plan",
            )
        return self._execute_arm_path(
            arm=arm,
            native=native,
            path=path,
            target=target,
            planned=planned,
            observation_token=token,
            settle_steps=settle_steps,
        )

    def _execute_arm_path(
        self, *, arm, native, path, target, planned, observation_token, settle_steps
    ) -> ToolResult:
        """Submit one path and optional target holds, then measure the achieved pose."""
        position, orientation = target[:3], target[3:]
        actions = np.tile(native, (len(path), 1))
        actions[:, slice(7, 14) if arm == "left" else slice(15, 22)] = path
        final_target = actions[-1:].copy()
        expected_token = observation_token
        ledger = self._move_call_ledger
        stop_reason = (
            "target_not_reached"
            if settle_steps == 0
            else "target_hold_budget_exhausted"
        )
        execution_error = None
        fresh = None
        actual = None
        position_error = orientation_error = None
        at_target = None
        # Send the native path once. Each later submission is exactly its final
        # native23 command, including every locked joint and both grippers.
        # Reading an observation alone would not advance the official evaluator.
        for hold_index in range(settle_steps + 1):
            self._check_cancelled()  # Never convert toolkit cancellation to a retry.
            submitted = actions if hold_index == 0 else final_target
            actual = None
            position_error = orientation_error = None
            at_target = None
            try:
                self._move_call_phase = "path" if hold_index == 0 else "target_hold"
                result = self._send_actions(
                    submitted, expected_observation_token=expected_token
                )
                _, _, terminated, truncated, info = result
            except (OSError, RpcError, TypeError, ValueError) as error:
                # Preserve the existing first-submission exception behavior.
                # A later failed RPC or malformed reply retains the confirmed path
                # and holds; none of the latest submission's counts are knowable.
                if hold_index == 0:
                    raise
                fresh = False
                stop_reason = "execution_outcome_unknown"
                execution_error = f"post-motion hold outcome is unknown: {error}"
                break
            motion_feedback_error = info.get("motion_feedback_error")
            entry = ledger.submissions[-1]
            segment_counts = {name: entry[name] for name in COUNT_NAMES}
            feedback_errors = list(entry["count_errors"])
            try:
                self._terminated = _termination_flag(terminated)
                self._truncated = _termination_flag(truncated)
            except (TypeError, ValueError) as error:
                self._terminated = self._truncated = None
                feedback_errors.append(str(error))
            invalid_execution_feedback = bool(feedback_errors)
            if motion_feedback_error:
                feedback_errors.append(f"motion feedback: {motion_feedback_error}")
            self._accept_execution(result, update_flags=False)
            fresh = (
                False
                if motion_feedback_error
                else info.get("observation_after_last_issued_action")
            )
            if fresh is True and not motion_feedback_error:
                # EnvClient has bound this token to the last confirmed public
                # frame; preserve it even if the episode has just terminated.
                expected_token = info["observation_token"]
            complete = (
                not feedback_errors
                and segment_counts["issued_actions"] == len(submitted)
                and segment_counts["confirmed_actions"] == len(submitted)
                and segment_counts["executed_actions"] == len(submitted)
                and segment_counts["unconfirmed_actions"] == 0
            )
            missing_counts = any(value is None for value in segment_counts.values())
            measurement_error = None
            if fresh is True and not feedback_errors and not missing_counts:
                try:
                    actual = self._robot_state(self.last_observation)[arm]
                    position_error = float(
                        np.linalg.norm(np.asarray(actual["eef_position"]) - position)
                    )
                    actual_quat = np.asarray(actual["eef_quaternion_xyzw"])
                    dot = abs(float(np.dot(actual_quat, orientation)))
                    dot /= np.linalg.norm(actual_quat) * np.linalg.norm(orientation)
                    orientation_error = float(2 * np.arccos(np.clip(dot, 0, 1)))
                    # A relaxed plan was solved for position only; judge it by
                    # position. The achieved orientation is still reported.
                    at_target = position_error <= 0.005 and (
                        planned.get("orientation_relaxed") or orientation_error <= 0.05
                    )
                except (TypeError, ValueError) as error:
                    actual = None
                    position_error = orientation_error = None
                    measurement_error = (
                        f"post-motion pose could not be evaluated: {error}"
                    )
            if (
                self._session_ended
                or self._terminated is True
                or self._truncated is True
            ):
                stop_reason = (
                    "session_ended"
                    if self._session_ended
                    else "terminated"
                    if self._terminated
                    else "truncated"
                )
                break
            if feedback_errors or missing_counts:
                self._command_history_unknown = True
                stop_reason = (
                    "invalid_execution_feedback"
                    if invalid_execution_feedback
                    else "execution_feedback_unknown"
                    if missing_counts
                    else "motion_feedback_mismatch"
                )
                execution_error = (
                    "; ".join(feedback_errors)
                    or "motion execution counters are incomplete"
                )
                break
            if not complete:
                if segment_counts["unconfirmed_actions"]:
                    self._command_history_unknown = True
                stop_reason = "partial_execution"
                execution_error = "motion execution returned before all requested actions were confirmed"
                break
            if fresh is not True:
                self._command_history_unknown = True
                stop_reason = "stale_observation"
                execution_error = (
                    "motion execution did not return a fresh post-action observation"
                )
                break
            if measurement_error:
                stop_reason = "measurement_error"
                execution_error = measurement_error
                break
            if at_target is True:
                stop_reason = "target_reached"
                break
        if stop_reason in {"target_not_reached", "target_hold_budget_exhausted"}:
            execution_error = "confirmed motion did not reach the requested EEF pose within the final-target action budget"
        return ToolResult(
            data={
                "completed": stop_reason == "target_reached" and not execution_error,
                "arm": arm,
                "frame": "robot_base",
                "quaternion_order": "xyzw",
                "target_pose": target.tolist(),
                "planned_motion_actions": len(path),
                "settle_steps": settle_steps,
                "actual_settle_steps": ledger.total(
                    [
                        entry
                        for entry in ledger.submissions
                        if entry["phase"] == "target_hold"
                    ]
                )["confirmed_actions"],
                "observation_fresh": fresh,
                "session_ended": self._session_ended,
                "stop_reason": stop_reason,
                "terminated": self._terminated,
                "truncated": self._truncated,
                "final_eef_position": None
                if actual is None
                else actual["eef_position"],
                "position_error_m": position_error,
                "orientation_error_rad": orientation_error,
                "target_reached": at_target,
                "collision_checked": planned.get("collision_checked"),
                "orientation_relaxed": planned.get("orientation_relaxed", False),
                "planned_orientation_error_rad": planned.get(
                    "ik_orientation_error_rad"
                ),
                **({"error": execution_error} if execution_error else {}),
            }
        )


def validate_base_request(
    forward: float, lateral: float, turn: float, steps: int
) -> np.ndarray:
    """Validate a short velocity segment without silently clipping the request."""
    values = (forward, lateral, turn)
    if any(
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, Real)
        or not np.isfinite(value)
        for value in values
    ):
        raise ValueError("base velocities must be finite real numbers, not booleans")
    native = np.asarray(values, dtype=np.float64)
    if np.linalg.norm(native[:2]) > MAX_NORMALIZED_LINEAR_NORM + 1e-12:
        raise ValueError("normalized planar velocity norm must be at most 1.0")
    if abs(native[2]) > MAX_NORMALIZED_TURN:
        raise ValueError("normalized turn magnitude must be at most 1.0")
    if type(steps) is not int or not 1 <= steps <= MAX_MOVE_STEPS:
        raise ValueError("steps must be a positive integer within the episode budget")
    return native


def validate_observation_token(value: Any) -> dict[str, int]:
    keys = {"generation", "observation_index"}
    if (
        not isinstance(value, dict)
        or set(value) != keys
        or any(type(value[key]) is not int or value[key] < 0 for key in keys)
    ):
        raise ValueError("motion token must identify one public observation")
    return dict(value)


def one_action_counts(info: dict[str, Any]) -> dict[str, int]:
    counts, errors = _action_counts(info, 1)
    if errors or any(value is None for value in counts.values()):
        raise ValueError("one-action execution requires consistent known 0/1 counters")
    return counts


def validate_public_base_context(
    context: Any, token: dict[str, int], proprio: np.ndarray
) -> None:
    """Check public state/frame binding without imposing a stationary-state gate."""
    if not isinstance(context, dict):
        raise ValueError("base feedback requires a public motion context")
    dt = context.get("control_dt_s")
    if (
        context.get("source") != "confirmed_public_control_observations"
        or isinstance(dt, bool)
        or not isinstance(dt, Real)
        or not np.isfinite(dt)
        or abs(dt - CONTROL_DT_S) > 1e-12
        or context.get("current_token") != token
        or context.get("actions_pending") is not False
        or type(context.get("unconfirmed_actions")) is not int
        or context["unconfirmed_actions"] != 0
    ):
        raise ValueError(
            "the public base frame has pending/unknown execution or timing"
        )
    bound = np.asarray(context.get("current_proprio"))
    if bound.shape != (61,) or not np.array_equal(bound, proprio):
        raise ValueError("public base observation differs from its bound frame")


class MotionCallLedger:
    """Count one arm-motion call's native submissions, including failed ones."""

    def __init__(self):
        self.submissions: list[dict[str, Any]] = []
        self.preflight: dict[str, Any] = {"status": "not_required"}
        self.target_pose: list[float] | None = None
        self.observation_invalidated_by_newer_snapshot = False

    def start(self, actions, phase):
        item = {
            "phase": phase,
            "requested_actions": len(actions),
            **dict.fromkeys(COUNT_NAMES),
            "count_errors": [],
            "observation_fresh": False,
            "counts_valid": False,
        }
        self.submissions.append(item)
        return item

    def returned(self, item, result):
        if (
            not isinstance(result, (tuple, list))
            or len(result) != 5
            or not isinstance(result[4], dict)
        ):
            return
        info = result[4]
        counts, errors = _action_counts(info, item["requested_actions"])
        item.update(counts)
        item["counts_valid"] = not errors and all(
            value is not None for value in counts.values()
        )
        item["count_errors"] = errors
        item["observation_fresh"] = info.get(
            "observation_after_last_issued_action"
        ) is True and not info.get("motion_feedback_error")
        item["stop_reason"] = info.get("stop_reason")

    @staticmethod
    def total(items):
        return {
            "requested_actions": sum(item["requested_actions"] for item in items),
            **{
                name: sum(item[name] for item in items)
                if all(item[name] is not None for item in items)
                else None
                for name in COUNT_NAMES
            },
        }

    def merge(self, data):
        data = dict(data)
        if self.target_pose is not None:
            data.setdefault("target_pose", self.target_pose)
        if not self.submissions:
            return data
        # Toolkit must persist this call's captured frame. A second live read
        # could observe another consumer's action while retaining our token.
        data["observation_capture_source"] = "motion_call_cache"
        data.update(self.total(self.submissions))
        last = self.submissions[-1]
        # Unknown/malformed submissions cannot revive an earlier fresh flag.
        data["observation_fresh"] = (
            last["counts_valid"]
            and last["observation_fresh"]
            and not self.observation_invalidated_by_newer_snapshot
        )
        if last.get("stop_reason") == "session_ended":
            data["session_ended"] = True
        return data
