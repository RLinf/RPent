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

"""Dual-Franka planner tools, primitives, and canonical state capture."""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Annotated, Any, Literal

import numpy as np
from pydantic import BeforeValidator, Field

from robots.franka.tools import FrankaPrimitives, coerce_vec3
from rpent.session import EnvState, StepRecord
from rpent.tools import ToolResult, tool


def coerce_arm(value: Any) -> str:
    """Return exactly ``'left'`` or ``'right'`` or raise a useful error."""
    arm = str(value).strip().lower()
    if arm not in {"left", "right"}:
        raise ValueError("arm must be exactly 'left' or 'right'")
    return arm


def _camera_alias_from_key(raw_key: Any) -> str | None:
    key = str(raw_key or "")
    if key.endswith("_rgb"):
        key = key[: -len("_rgb")]
    if key.endswith("_0"):
        key = key[: -len("_0")]
    return key or None


def _agent_observation_policy(meta: dict[str, Any] | None) -> dict[str, list[str]]:
    raw = (meta or {}).get("agent_observation")
    if not isinstance(raw, dict):
        raw = {}
    inline = raw.get("inline_cameras", ["d455"])
    auxiliary = raw.get("auxiliary_cameras", ["left_wrist", "base", "right_wrist"])
    return {
        "inline_cameras": [str(item) for item in inline if isinstance(item, str)],
        "auxiliary_cameras": [str(item) for item in auxiliary if isinstance(item, str)],
    }


class DualFrankaPrimitives(FrankaPrimitives):
    """Safe agent-facing operations over a remote dual-Franka environment."""

    def __init__(
        self,
        *,
        env: Any,
        model: Any | None,
        task_description: str,
        check_cancelled: Callable[[], None],
        sam3_client: Any | None = None,
        vla_instruction: str | None = None,
    ) -> None:
        super().__init__(
            env=env,
            model=model,
            task_description=task_description,
            check_cancelled=check_cancelled,
        )
        self._sam3_client = sam3_client
        self._vla_instruction = vla_instruction or task_description

    @tool
    def move_delta(
        self,
        arm: Annotated[Literal["left", "right"], BeforeValidator(coerce_arm)],
        delta_xyz: Annotated[list[float], Field(min_length=3, max_length=3)],
    ) -> ToolResult:
        """Move one Franka TCP by a bounded world-frame xyz delta in meters.

        Args:
            arm: Which arm to command; the other arm is left uncommanded.
        """
        self._check_cancelled()
        data = self.env.move_delta(
            coerce_arm(arm), coerce_vec3(delta_xyz, name="delta_xyz")
        )
        return ToolResult(data=data, error=data.pop("error", None))

    @tool
    def rotate_delta(
        self,
        arm: Annotated[Literal["left", "right"], BeforeValidator(coerce_arm)],
        delta_rpy: Annotated[list[float], Field(min_length=3, max_length=3)],
    ) -> ToolResult:
        """Rotate one Franka TCP by a bounded world-frame rpy delta in radians.

        Args:
            arm: Which arm to command; the other arm is left uncommanded.
        """
        self._check_cancelled()
        data = self.env.rotate_delta(
            coerce_arm(arm), coerce_vec3(delta_rpy, name="delta_rpy")
        )
        return ToolResult(data=data, error=data.pop("error", None))

    @tool
    def open_gripper(
        self, arm: Annotated[Literal["left", "right"], BeforeValidator(coerce_arm)]
    ) -> ToolResult:
        """Open one Franka gripper and wait for the command to settle.

        Args:
            arm: Which arm to command; the other arm is left uncommanded.
        """
        self._check_cancelled()
        data = self.env.set_gripper(coerce_arm(arm), open=True)
        return ToolResult(data=data, error=data.pop("error", None))

    @tool
    def close_gripper(
        self, arm: Annotated[Literal["left", "right"], BeforeValidator(coerce_arm)]
    ) -> ToolResult:
        """Close one Franka gripper and wait for the command to settle.

        Args:
            arm: Which arm to command; the other arm is left uncommanded.
        """
        self._check_cancelled()
        data = self.env.set_gripper(coerce_arm(arm), open=False)
        return ToolResult(data=data, error=data.pop("error", None))

    @tool
    def recover_joint_posture(
        self,
        reason: Annotated[str, Field(json_schema_extra={"default": ""})] = "",
        return_to_start: Annotated[
            bool, Field(json_schema_extra={"default": True})
        ] = True,
    ) -> ToolResult:
        """Reset both arms to their healthy configured joint posture while preserving each gripper's open/closed state. Closed grippers are re-commanded before/after the joint reset so held objects stay clamped, then both TCPs return near their prior poses."""
        self._check_cancelled()
        data = self.env.recover_joint_posture(
            reason=str(reason), return_to_start=bool(return_to_start)
        )
        return ToolResult(data=data, error=data.pop("error", None))

    @tool(readonly=True)
    def describe_dual_franka_setup(self) -> ToolResult:
        """Read the dual-Franka runtime conventions, camera aliases, VLA policy conditioning text, semantic stop rules, and available primitive names before acting. This is read-only."""
        from robots.dual_franka.toolkit import DualFrankaToolkit

        meta = self.env.meta
        observation_policy = _agent_observation_policy(meta)
        return ToolResult(
            data={
                "ok": True,
                "phase": "strict",
                "reset_policy": "The runner reset the robot at startup. Do not call reset during a task unless reset is explicitly exposed and there is a clear robot-side reason.",
                "coordinate_frame": "right_base",
                "camera_aliases": meta.get("observation_camera_map", {}),
                "projection_views": meta.get("projection_views", {}),
                "agent_observation": observation_policy,
                "vla": {
                    "policy_instruction": self._vla_instruction,
                    "num_action_chunks": 20,
                    "action_dim": 20,
                    "num_images_in_input": 3,
                    "external_localization_views_in_policy_input": False,
                    "agent_visible_images": observation_policy["inline_cameras"],
                    "auxiliary_artifact_images": observation_policy[
                        "auxiliary_cameras"
                    ],
                    "skill_stop_rules": {
                        "enabled": True,
                        "grasp_lift_m": 0.15,
                        "place_lift_m": 0.1,
                        "handoff_release_delay_s": 1.5,
                        "min_steps_after_gripper_event": 2,
                    },
                },
                "sam3": {
                    "tool": "segment",
                    "status": "optional; returns an error and falls back to manual camera projection when no SAM3 client is configured",
                    "usage": "Use text prompt or one [row, col] positive camera point, then inspect the returned mask overlay before trusting point_xyz. SAM3 text grounding is phrase-sensitive; if a prompt returns a very low score, retry a shorter/rephrased prompt or point prompt rather than lowering min_score blindly.",
                },
                "available_primitives": [
                    tool.name for tool in DualFrankaToolkit.declared_tools()
                ],
                "operator_guidance": "Named VLA semantic boundaries are segment boundaries, not proof of physical success. Verify images, gripper widths/open flags, joint_health, and projection evidence after every action. recover_joint_posture re-commands and preserves each gripper's open/closed state; inspect its gripper_preserved result before continuing.",
            }
        )

    def _run_named_vla_skill(
        self,
        *,
        skill_name: str,
        boundary: str,
        prompt: str,
        max_chunks: int = 20,
    ) -> dict[str, Any]:
        if self.model is None:
            raise RuntimeError(f"{skill_name} requires --vla-endpoint")
        requested_prompt = str(prompt).strip()
        if not requested_prompt:
            raise ValueError("prompt must be non-empty")
        # Task configuration owns policy conditioning; retain planner intent in logs.
        effective_prompt = self._vla_instruction
        if not 1 <= int(max_chunks) <= 20:
            raise ValueError("max_chunks must be between 1 and 20")

        state = self.env.get_robot_state()
        start_state = state
        previous_left_open = bool(state["left_arm"]["gripper_open"])
        previous_right_open = bool(state["right_arm"]["gripper_open"])
        event_z: float | None = None
        event_time: float | None = None
        steps_after_event = 0
        chunks_executed = 0
        steps_executed = 0
        event_step: int | None = None
        boundary_reached = False
        terminated = False
        truncated = False
        action_count = 0
        action_min: np.ndarray | None = None
        action_max: np.ndarray | None = None
        action_sum: np.ndarray | None = None
        first_action: list[float] | None = None
        first_policy_state: list[float] | None = None
        first_policy_state_shape: list[int] | None = None
        started_at = time.perf_counter()

        for _ in range(int(max_chunks)):
            self._check_cancelled()
            observation = dict(self.env.get_observation())
            if first_policy_state is None:
                obs_states = np.asarray(observation.get("states"), dtype=np.float32)
                first_policy_state_shape = list(obs_states.shape)
                first_policy_state = (
                    np.round(obs_states.reshape(-1)[:20], 5).astype(float).tolist()
                )
            observation["task_descriptions"] = effective_prompt
            actions = np.asarray(
                self.model.predict(observation, options={"mode": "eval"}),
                dtype=np.float32,
            )
            if actions.ndim != 2 or actions.shape[1] != 20:
                raise RuntimeError(
                    f"{skill_name} expected [chunk, 20] actions, got {actions.shape}"
                )
            if not np.isfinite(actions).all():
                raise RuntimeError(f"{skill_name} received non-finite VLA actions")
            chunks_executed += 1
            if first_action is None:
                first_action = np.round(actions[0], 5).astype(float).tolist()
            if action_count == 0:
                action_min = actions.min(axis=0)
                action_max = actions.max(axis=0)
                action_sum = actions.sum(axis=0)
            else:
                action_min = np.minimum(action_min, actions.min(axis=0))
                action_max = np.maximum(action_max, actions.max(axis=0))
                action_sum = action_sum + actions.sum(axis=0)
            action_count += int(actions.shape[0])

            for action in actions:
                self._check_cancelled()
                result = self.env.chunk_step(action[None, :])
                steps_executed += 1
                terminated = terminated or bool(result.get("terminated"))
                truncated = truncated or bool(result.get("truncated"))
                state = self.env.get_robot_state()
                left = state["left_arm"]
                right = state["right_arm"]
                left_open = bool(left["gripper_open"])
                right_open = bool(right["gripper_open"])

                if boundary == "grasp":
                    if event_z is None and previous_right_open and not right_open:
                        event_z = float(right["tcp_pose"][2])
                        event_step = steps_executed
                        steps_after_event = 0
                    elif event_z is not None:
                        steps_after_event += 1
                        boundary_reached = (
                            not right_open
                            and steps_after_event >= 2
                            and float(right["tcp_pose"][2]) - event_z >= 0.15
                        )
                elif boundary == "handoff":
                    if event_time is None and not previous_right_open and right_open:
                        event_time = time.monotonic()
                        event_step = steps_executed
                        steps_after_event = 0
                    elif event_time is not None:
                        steps_after_event += 1
                        boundary_reached = (
                            right_open
                            and steps_after_event >= 2
                            and time.monotonic() - event_time >= 1.5
                        )
                elif boundary == "place":
                    if event_z is None and not previous_left_open and left_open:
                        event_z = float(left["tcp_pose"][2])
                        event_step = steps_executed
                        steps_after_event = 0
                    elif event_z is not None:
                        steps_after_event += 1
                        boundary_reached = (
                            left_open
                            and steps_after_event >= 2
                            and float(left["tcp_pose"][2]) - event_z >= 0.10
                        )
                else:  # pragma: no cover - internal programming guard
                    raise ValueError(f"unknown VLA skill boundary: {boundary}")

                previous_left_open = left_open
                previous_right_open = right_open
                if boundary_reached or terminated or truncated:
                    break
            if boundary_reached or terminated or truncated:
                break

        stop_rule: dict[str, Any] = {
            "phase": boundary,
            "skill_name": skill_name,
            "step_count": steps_executed,
            "event_step": event_step,
            "success_claim": False,
        }
        if boundary == "grasp":
            stop_rule.update(
                {
                    "condition": "right_gripper_closed_then_lifted",
                    "right_close_step": event_step,
                    "right_close_z": event_z,
                    "right_current_z": float(state["right_arm"]["tcp_pose"][2]),
                    "lift_m": (
                        float(state["right_arm"]["tcp_pose"][2]) - event_z
                        if event_z is not None
                        else None
                    ),
                    "threshold_m": 0.15,
                }
            )
        elif boundary == "handoff":
            stop_rule.update(
                {
                    "condition": "right_gripper_opened_then_delay",
                    "right_open_step": event_step,
                    "elapsed_after_open_s": (
                        time.monotonic() - event_time
                        if event_time is not None
                        else None
                    ),
                    "delay_s": 1.5,
                    "right_current_open": bool(state["right_arm"]["gripper_open"]),
                }
            )
        elif boundary == "place":
            stop_rule.update(
                {
                    "condition": "left_gripper_opened_then_lifted",
                    "left_open_step": event_step,
                    "left_open_z": event_z,
                    "left_current_z": float(state["left_arm"]["tcp_pose"][2]),
                    "lift_m": (
                        float(state["left_arm"]["tcp_pose"][2]) - event_z
                        if event_z is not None
                        else None
                    ),
                    "threshold_m": 0.10,
                }
            )

        action_summary: dict[str, Any] = {
            "count": action_count,
            "shape": [action_count, 20],
            "finite": True,
        }
        if action_count and action_min is not None and action_max is not None:
            action_summary.update(
                {
                    "min": np.round(action_min, 5).astype(float).tolist(),
                    "max": np.round(action_max, 5).astype(float).tolist(),
                    "mean": np.round(action_sum / action_count, 5)
                    .astype(float)
                    .tolist(),
                    "first_action": first_action,
                    "first_policy_state_shape": first_policy_state_shape,
                    "first_policy_state": first_policy_state,
                }
            )

        return {
            "ok": boundary_reached and not (terminated or truncated),
            "skill_name": skill_name,
            "boundary": boundary,
            "requested_prompt": requested_prompt,
            "effective_policy_prompt": effective_prompt,
            "prompt_overridden": requested_prompt != effective_prompt,
            "boundary_reached": boundary_reached,
            "chunks_executed": chunks_executed,
            "steps_executed": steps_executed,
            "terminated": terminated,
            "truncated": truncated,
            "stop_rule": stop_rule,
            "action_summary": action_summary,
            "elapsed_s": time.perf_counter() - started_at,
            "vla_start_robot_state": start_state,
            "robot_state": state,
        }

    @tool
    def vla_right_grasp(
        self,
        prompt: str,
        max_chunks: Annotated[
            int, Field(ge=1, le=20, json_schema_extra={"default": 20})
        ] = 20,
    ) -> ToolResult:
        """Run the learned right-grasp VLA segment. The active task prompt decides which object is currently allowed; this tool only defines the capability boundary: right gripper closes and the right TCP lifts.

        Args:
            prompt: Planner-facing segment intent. This is recorded in the tool result; the current live clean-desk checkpoint still receives its fixed training instruction during policy inference.
        """
        data = self._run_named_vla_skill(
            skill_name="vla_right_grasp",
            boundary="grasp",
            prompt=prompt,
            max_chunks=max_chunks,
        )
        return ToolResult(data=data, error=data.pop("error", None))

    @tool
    def vla_handoff(
        self,
        prompt: str,
        max_chunks: Annotated[
            int, Field(ge=1, le=20, json_schema_extra={"default": 20})
        ] = 20,
    ) -> ToolResult:
        """Run the learned bimanual handoff VLA segment. The capability boundary is right-gripper release followed by the configured settle delay; do not rule-base pre-position either arm for it.

        Args:
            prompt: Planner-facing segment intent. This is recorded in the tool result; the current live clean-desk checkpoint still receives its fixed training instruction during policy inference.
        """
        data = self._run_named_vla_skill(
            skill_name="vla_handoff",
            boundary="handoff",
            prompt=prompt,
            max_chunks=max_chunks,
        )
        return ToolResult(data=data, error=data.pop("error", None))

    @tool
    def vla_left_place(
        self,
        prompt: str,
        max_chunks: Annotated[
            int, Field(ge=1, le=20, json_schema_extra={"default": 20})
        ] = 20,
    ) -> ToolResult:
        """Run the learned left-placement VLA segment. The active task decides the destination; this tool only defines the capability boundary: left gripper opens and the left TCP lifts.

        Args:
            prompt: Planner-facing segment intent. This is recorded in the tool result; the current live clean-desk checkpoint still receives its fixed training instruction during policy inference.
        """
        data = self._run_named_vla_skill(
            skill_name="vla_left_place",
            boundary="place",
            prompt=prompt,
            max_chunks=max_chunks,
        )
        return ToolResult(data=data, error=data.pop("error", None))


def dump_state(
    primitives: DualFrankaPrimitives,
    state: EnvState,
    *,
    command: dict[str, Any] | None,
    result: dict[str, Any] | None,
    elapsed_s: float | None,
) -> StepRecord:
    """Capture per-arm robot state and synchronized camera images."""
    observation = primitives.env.get_observation()
    robot_state = primitives.env.get_robot_state()
    metadata = primitives.env.get_camera_meta()
    with state.record_step(
        state=robot_state,
        command=command,
        result=result,
        elapsed_s=elapsed_s,
    ) as step:
        raw_frames = observation.get("raw_camera_frames") or {}
        raw_depths = observation.get("raw_camera_depths") or {}
        observation_map = (
            metadata.get("observation_camera_map", {})
            if isinstance(metadata, dict)
            else {}
        )
        projection_views = (
            metadata.get("projection_views", {}) if isinstance(metadata, dict) else {}
        )
        main_image = observation.get("main_images")
        if main_image is not None:
            main_alias = (
                _camera_alias_from_key(observation_map.get("main")) or "left_wrist"
            )
            state.save(f"{main_alias}.png", np.asarray(main_image), step=step)
        extra_images = observation.get("extra_view_images")
        if extra_images is not None:
            extra_array = np.asarray(extra_images)
            if extra_array.ndim == 5:
                extra_array = extra_array[0]
            if extra_array.ndim == 4:
                for index, image in enumerate(extra_array):
                    raw_key = observation_map.get(f"extra_{index}")
                    alias = _camera_alias_from_key(raw_key) or f"extra_{index}"
                    state.save(f"{alias}.png", np.asarray(image), step=step)
        for raw_key, image in raw_frames.items():
            alias = _camera_alias_from_key(raw_key)
            if alias:
                state.save(f"{alias}.png", np.asarray(image), step=step)
        main_depth = observation.get("main_depths")
        if main_depth is not None:
            main_alias = (
                _camera_alias_from_key(observation_map.get("main")) or "left_wrist"
            )
            state.save(f"{main_alias}_depth.npy", np.asarray(main_depth), step=step)
        extra_depths = observation.get("extra_view_depths")
        if extra_depths is not None:
            depth_array = np.asarray(extra_depths)
            if depth_array.ndim == 4 and depth_array.shape[0] == 1:
                depth_array = depth_array[0]
            if depth_array.ndim == 3:
                for index, depth in enumerate(depth_array):
                    raw_key = observation_map.get(f"extra_{index}")
                    alias = _camera_alias_from_key(raw_key) or f"extra_{index}"
                    state.save(f"{alias}_depth.npy", np.asarray(depth), step=step)
        for raw_key, depth in raw_depths.items():
            alias = _camera_alias_from_key(raw_key)
            if alias:
                state.save(f"{alias}_depth.npy", np.asarray(depth), step=step)
        for alias in projection_views:
            image = observation.get(f"{alias}_images")
            if image is not None:
                state.save(f"{alias}.png", np.asarray(image), step=step)
            depth = observation.get(f"{alias}_depths")
            if depth is not None:
                state.save(f"{alias}_depth.npy", np.asarray(depth), step=step)
        for key, value in observation.items():
            if isinstance(key, str) and key.endswith("_images"):
                alias = key[: -len("_images")]
                if alias and alias not in {"main", "extra_view", "raw_camera"}:
                    state.save(f"{alias}.png", np.asarray(value), step=step)
            if isinstance(key, str) and key.endswith("_depths"):
                alias = key[: -len("_depths")]
                if alias and alias not in {"main", "extra_view", "raw_camera"}:
                    state.save(f"{alias}_depth.npy", np.asarray(value), step=step)
        if metadata is not None:
            state.save("camera_meta.json", metadata, step=step)
    return state.get(step)


@tool(readonly=True, exclude=("state",))
def view_env_state(
    step: Annotated[int, Field(json_schema_extra={"default": -1})] = -1,
    *,
    state: EnvState,
) -> ToolResult:
    """Read a dual-Franka state snapshot. Configured inline camera views are returned directly; other available views are returned as artifact paths; use read_image to inspect these artifacts."""
    images: list[bytes] = []
    record = state.get(step)
    output = record.to_blob()
    output["images"] = []
    output["artifact_images"] = []
    meta = (
        state.load("camera_meta.json", step=record.step_idx)
        if state.exists("camera_meta.json", step=record.step_idx)
        else {}
    )
    policy = _agent_observation_policy(meta if isinstance(meta, dict) else {})
    available_views = sorted(
        artifact.removesuffix(".png")
        for artifact in record.artifacts
        if artifact.endswith(".png")
        and "_segment_overlay_" not in artifact
        and "_back_project_" not in artifact
    )
    output["available_camera_views"] = available_views
    output["agent_observation"] = policy
    for view in available_views:
        artifact = f"{view}.png"
        output[f"image_{view}_path"] = str(
            state.artifact_path(artifact, step=record.step_idx)
        )
        output["artifact_images"].append(view)
    for index, view in enumerate(policy["inline_cameras"]):
        artifact = f"{view}.png"
        if index >= 4 or not state.exists(artifact, step=record.step_idx):
            continue
        images.append(state.load_bytes(artifact, step=record.step_idx))
        output["images"].append(view)
    output["image_block_order"] = list(output["images"])
    return ToolResult(data=output, error=output.pop("error", None), images=images)
