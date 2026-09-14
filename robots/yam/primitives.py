# Copyright 2026 The RPent Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0

"""YAM primitives built on the env and Pi0.5 RPC contracts."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

from robots.yam.contracts import MODEL_SPEC, validate_actions
from robots.yam.env_client import YamEnvClient
from robots.yam.servo import JointServoConfig, run_joint_servo
from robots.yam.tasks import classify_episode
from rpent.robots.components.vla_client_base import BaseVLAClient


def _quat_angle_rad(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    left = left / np.linalg.norm(left)
    right = right / np.linalg.norm(right)
    dot = float(abs(np.dot(left, right)))
    return float(2.0 * np.arccos(np.clip(dot, -1.0, 1.0)))


class YamPrimitives:
    def __init__(
        self,
        *,
        env: YamEnvClient,
        model: BaseVLAClient | None = None,
        check_cancelled: Callable[[], None],
    ) -> None:
        self.env = env
        self.model = model
        self._check_cancelled = check_cancelled
        self._frames: list[np.ndarray] = []

    def _record_frame(self, rgb: Any) -> None:
        self._frames.append(np.ascontiguousarray(np.asarray(rgb)))

    def stop_recording(self) -> list[np.ndarray]:
        frames = self._frames
        self._frames = []
        return frames

    @staticmethod
    def _completion(
        *, requested: int, executed: int, status: dict[str, Any]
    ) -> dict[str, Any]:
        budget_exhausted = int(status.get("take_action_cnt", 0)) >= int(
            status.get("step_lim", 0)
        )
        completed = executed == requested
        if status.get("eval_success") is True:
            stop_reason = "operator_or_env_success"
        elif budget_exhausted:
            stop_reason = "budget_exhausted"
        elif completed:
            stop_reason = "completed"
        else:
            stop_reason = "runtime_failure"
        return {
            "completed": completed,
            "requested_steps": requested,
            "executed_steps": executed,
            "stop_reason": stop_reason,
        }

    def status(self) -> dict[str, Any]:
        """Read authoritative episode state and classify permitted continuation."""
        _, info = self.env.read_control_state()
        return classify_episode(info["episode_status"])

    def reset(self) -> dict[str, Any]:
        _, info = self.env.reset()
        return {**info, "success": True}

    def _build_policy_observation(self, *, prompt: str | None = None) -> dict[str, Any]:
        self.env.observe()
        frames = self.env.last_obs["frames"]
        instruction = self.env.get_task_language()
        policy_instruction = (
            prompt if prompt is not None and prompt.strip() else instruction
        )
        return {
            "main_images": np.asarray(frames["top"])[None],
            "wrist_images": None,
            "extra_view_images": np.stack([frames["left"], frames["right"]])[None],
            "states": np.asarray(
                self.env.last_obs["state"]["joint_position"], dtype=np.float32
            )[None],
            "task_descriptions": [policy_instruction],
        }

    def predict_actions(
        self, *, prompt: str | None = None
    ) -> tuple[dict, np.ndarray, dict]:
        """Shared policy transforms for execution and no-motion diagnostics."""
        if self.model is None:
            raise RuntimeError("YAM VLA is not connected")
        observation = self._build_policy_observation(prompt=prompt)
        status = dict(self.env.last_info["episode_status"])
        prediction = np.asarray(self.model.predict(observation))
        if prediction.shape != (1, MODEL_SPEC.action_horizon, 14):
            raise ValueError(f"expected VLA shape (1,30,14), got {prediction.shape}")
        return observation, validate_actions(prediction[0]), status

    def _record_chunk_payload(self, payload: Any) -> None:
        observations: list[Any]
        if isinstance(payload, list):
            observations = payload
        elif isinstance(payload, dict):
            observations = [payload]
        else:
            observations = []
        for obs in observations:
            if isinstance(obs, dict):
                frame = obs.get("frames", {}).get("top")
                if frame is not None:
                    self._record_frame(frame)

    def pi05_act(
        self,
        *,
        chunks: int = 1,
        use_length: int = MODEL_SPEC.use_length,
        prompt: str | None = None,
    ) -> dict[str, Any]:
        if self.model is None:
            raise RuntimeError(
                "YAM VLA is not connected; pi05_act requires a trained --vla-endpoint"
            )
        if int(chunks) < 1:
            raise ValueError("chunks must be at least 1")
        if (
            isinstance(use_length, bool)
            or int(use_length) != use_length
            or not (1 <= int(use_length) <= MODEL_SPEC.action_horizon)
        ):
            raise ValueError(
                f"use_length must be an integer in [1,{MODEL_SPEC.action_horizon}]"
            )
        use_length = int(use_length)
        executed = 0
        requested = int(chunks) * use_length
        native_prompt = None
        for _ in range(int(chunks)):
            self._check_cancelled()
            status = self.env.last_info["episode_status"]
            if status.get("eval_success") is True or int(
                status["take_action_cnt"]
            ) >= int(status["step_lim"]):
                break
            observation, actions, prediction_status = self.predict_actions(
                prompt=prompt
            )
            native_prompt = observation["task_descriptions"][0]
            episode_id = prediction_status["episode_id"]
            if len(actions) < use_length:
                raise ValueError(
                    f"VLA returned {len(actions)} actions; requested {use_length}"
                )
            actions = actions[:use_length]
            payload, _, _, _, info = self.env.chunk_step(
                actions,
                action_type="qpos",
                expected_episode_id=episode_id,
                return_all_frames=self.env.execution_capabilities.get(
                    "chunk_step_all_frames"
                )
                is True,
            )
            self._record_chunk_payload(payload)
            count = int(info.get("executed_actions", 0))
            executed += count
        status = self.env.last_info["episode_status"]
        return {
            **self._completion(requested=requested, executed=executed, status=status),
            "success": executed == requested,
            "prompt": native_prompt,
            "episode_status": status,
        }

    @staticmethod
    def _validate_qpos_updates_request(updates: Any) -> list[dict[str, Any]]:
        if not isinstance(updates, list) or not updates:
            raise ValueError("qpos updates must contain at least one update")
        normalized = []
        for update in updates:
            if not isinstance(update, dict):
                raise TypeError("qpos update must be a mapping")
            arm = update.get("arm")
            if arm not in ("left", "right"):
                raise ValueError("arm must be 'left' or 'right'")
            if update.get("arm_qpos") is None and update.get("gripper") is None:
                raise ValueError("qpos update must set arm_qpos and/or gripper")
            item: dict[str, Any] = {"arm": arm}
            if update.get("arm_qpos") is not None:
                arm_qpos = np.asarray(update["arm_qpos"], dtype=np.float64)
                if arm_qpos.shape != (6,) or not np.isfinite(arm_qpos).all():
                    raise ValueError("arm_qpos must be finite and have shape (6,)")
                item["arm_qpos"] = arm_qpos
            if update.get("gripper") is not None:
                gripper = float(update["gripper"])
                if not np.isfinite(gripper) or not 0.0 <= gripper <= 1.0:
                    raise ValueError("gripper must be finite and within [0,1]")
                item["gripper"] = gripper
            normalized.append(item)
        return normalized

    def apply_qpos_updates(
        self,
        updates: list[dict[str, Any]],
        *,
        expected_episode_id: str | None = None,
        compact_control: bool = False,
    ) -> dict[str, Any]:
        updates = self._validate_qpos_updates_request(updates)
        self._check_cancelled()
        if compact_control and len(updates) != 1:
            raise ValueError("compact control requires exactly one update")
        observation = (
            self.env.last_control_obs if compact_control else self.env.last_obs
        )
        current_info = (
            self.env.last_control_info if compact_control else self.env.last_info
        )
        state = np.asarray(observation["state"]["joint_position"], dtype=np.float64)
        action = np.asarray(
            current_info.get("commanded_qpos", state), dtype=np.float64
        ).copy()
        actions = []
        for update in updates:
            offset = 0 if update["arm"] == "left" else 7
            if "arm_qpos" in update:
                action[offset : offset + 6] = update["arm_qpos"]
            if "gripper" in update:
                action[offset + 6] = update["gripper"]
            actions.append(action.copy())
        actions_array = validate_actions(actions)
        if compact_control:
            _, _, _, _, info = self.env.control_step(
                actions_array[0],
                expected_episode_id=expected_episode_id
                or current_info["episode_status"]["episode_id"],
            )
        else:
            payload, _, _, _, info = self.env.chunk_step(
                actions_array,
                action_type="qpos",
                expected_episode_id=expected_episode_id
                or current_info["episode_status"]["episode_id"],
                return_all_frames=self.env.execution_capabilities.get(
                    "chunk_step_all_frames"
                )
                is True,
            )
            self._record_chunk_payload(payload)
        episode_status = info["episode_status"]
        executed = int(info.get("executed_actions", 0))
        return {
            "action_type": "qpos",
            "requested_actions": len(updates),
            "executed_actions": executed,
            "episode_status": episode_status,
        }

    def move_to(
        self,
        *,
        arm: str,
        xyz: list[float],
        quat: list[float] | None = None,
        gripper: float | None = None,
        substeps: int = 25,
        xyz_bounds: list[list[float]] | None = None,
    ) -> dict[str, Any]:
        if isinstance(substeps, bool) or int(substeps) != substeps or substeps < 0:
            raise ValueError("substeps must be a non-negative integer")
        if arm not in ("left", "right"):
            raise ValueError("arm must be 'left' or 'right'")
        servo_config = JointServoConfig.from_config(
            self.env.execution_capabilities.get("joint_servo")
        )
        robot_state = self.env.last_info["robot_state"]
        if quat is None:
            key = "left_eef_pose" if arm == "left" else "right_eef_pose"
            quat = np.asarray(robot_state[key], dtype=np.float64)[3:].tolist()
        target = np.asarray([*xyz, *quat], dtype=np.float64)
        if target.shape != (7,) or not np.isfinite(target).all():
            raise ValueError("target pose must be finite xyz + wxyz")
        episode_id = self.env.last_info["episode_status"]["episode_id"]
        candidates = [target.copy()]
        if xyz_bounds is not None:
            bounds = np.asarray(xyz_bounds, dtype=np.float64)
            if (
                bounds.shape != (2, 3)
                or not np.isfinite(bounds).all()
                or np.any(bounds[0] > bounds[1])
                or np.any(target[:3] < bounds[0])
                or np.any(target[:3] > bounds[1])
            ):
                raise ValueError(
                    "xyz_bounds must be finite [lower_xyz, upper_xyz] containing xyz"
                )
            # Search only the caller's task-valid region. Never execute several
            # candidates blindly: after motion the Agent must observe again.
            axis = int(np.argmax(bounds[1] - bounds[0]))
            for fraction in (0.25, 0.75):
                candidate = target.copy()
                candidate[axis] = bounds[0, axis] + fraction * (
                    bounds[1, axis] - bounds[0, axis]
                )
                if not any(
                    np.array_equal(candidate, previous) for previous in candidates
                ):
                    candidates.append(candidate)
        planning_attempts = []
        for candidate in candidates:
            planned = self.env.plan_arm_path(arm, candidate)
            planning_attempts.append(
                {
                    "xyz": candidate[:3].tolist(),
                    "status": planned.get("status"),
                    "reason": planned.get("reason"),
                }
            )
            if (
                planned.get("status") == "Success"
                and planned.get("position") is not None
            ):
                target = candidate
                break
        if planned.get("status") != "Success" or planned.get("position") is None:
            return {
                "completed": False,
                "requested_steps": 0,
                "executed_steps": 0,
                "stop_reason": "plan_failed",
                "success": False,
                "planning_attempts": planning_attempts,
                "plan_status": planned.get("status"),
                "hint": planned.get("reason", "target may be unreachable or unsafe"),
            }
        path = np.asarray(planned["position"], dtype=np.float64)
        if path.ndim != 2 or path.shape[1] != 6:
            raise ValueError(
                f"YAM plan_arm_path returned invalid path shape {path.shape}"
            )
        # Keep all safety waypoints; only insert samples along planned segments.
        if len(path) > 1 and substeps > len(path):
            counts = np.ones(len(path) - 1, dtype=int)
            extra = int(substeps) - len(path)
            counts += extra // len(counts)
            counts[: extra % len(counts)] += 1
            path = np.concatenate(
                [path[:1]]
                + [
                    np.linspace(start, end, count + 1)[1:]
                    for start, end, count in zip(path[:-1], path[1:], counts)
                ]
            )
        updates = [
            {"arm": arm, "arm_qpos": waypoint, "gripper": gripper} for waypoint in path
        ]
        try:
            execution = self.apply_qpos_updates(updates, expected_episode_id=episode_id)
        except Exception:
            if servo_config.enabled:
                self.env.request_stop()
            raise
        executed = int(execution.get("executed_actions", 0))
        status = execution["episode_status"]
        if servo_config.enabled:
            if executed == len(updates):
                servo = run_joint_servo(
                    env=self.env,
                    apply_updates=self.apply_qpos_updates,
                    check_cancelled=self._check_cancelled,
                    arm=arm,
                    nominal=path[-1],
                    target_pose=target,
                    episode_id=episode_id,
                    config=servo_config,
                )
            else:
                self.env.request_stop()
                servo = {
                    "enabled": True,
                    "success": False,
                    "stop_reason": "path_incomplete",
                    "executed_steps": 0,
                    "trace": [],
                }
            servo_steps = int(servo["executed_steps"])
            servo_requested = int(servo.get("requested_steps", servo_steps))
            return {
                **execution,
                "planning_attempts": planning_attempts,
                "xyz_bounds": xyz_bounds,
                "completed": servo["success"],
                "success": servo["success"],
                "recoverable": servo.get("recoverable", False),
                "requested_steps": len(updates) + servo_requested,
                "requested_actions": len(updates) + servo_requested,
                "executed_steps": executed + servo_steps,
                "executed_actions": executed + servo_steps,
                "path_executed_steps": executed,
                "servo_executed_steps": servo_steps,
                "stop_reason": servo["stop_reason"],
                "servo": servo,
                "plan_status": planned["status"],
                "waypoints": len(path),
                "target_pose": target.tolist(),
                "measured_pose": servo.get("measured_pose"),
                "position_error_m": servo.get("position_error_m"),
                "rotation_error_rad": servo.get("rotation_error_rad"),
                "position_tolerance_m": servo_config.position_tolerance_m,
                "rotation_tolerance_rad": servo_config.rotation_tolerance_rad,
                "episode_status": servo.get(
                    "episode_status", self.env.last_info["episode_status"]
                ),
            }
        key = "left_eef_pose" if arm == "left" else "right_eef_pose"
        measured_pose = np.asarray(
            self.env.last_info["robot_state"][key], dtype=np.float64
        )
        position_error_m = float(np.linalg.norm(measured_pose[:3] - target[:3]))
        rotation_error_rad = _quat_angle_rad(measured_pose[3:], target[3:])
        position_tolerance_m = 0.025
        rotation_tolerance_rad = 0.25
        reached_target = (
            executed == len(updates)
            and position_error_m <= position_tolerance_m
            and rotation_error_rad <= rotation_tolerance_rad
        )
        completion = self._completion(
            requested=len(updates), executed=executed, status=status
        )
        if not reached_target and completion["stop_reason"] == "completed":
            completion["stop_reason"] = "target_not_reached"
        return {
            **execution,
            "planning_attempts": planning_attempts,
            "xyz_bounds": xyz_bounds,
            **completion,
            "success": reached_target,
            "plan_status": planned["status"],
            "waypoints": len(path),
            "measured_pose": measured_pose.tolist(),
            "target_pose": target.tolist(),
            "position_error_m": position_error_m,
            "rotation_error_rad": rotation_error_rad,
            "position_tolerance_m": position_tolerance_m,
            "rotation_tolerance_rad": rotation_tolerance_rad,
        }

    def rotate_wrist(
        self,
        *,
        arm: str,
        delta_yaw_deg: float,
        gripper: float | None = None,
        substeps: int = 25,
    ) -> dict[str, Any]:
        state = self.env.last_info["robot_state"]
        key = "left_eef_pose" if arm == "left" else "right_eef_pose"
        pose = np.asarray(state[key], dtype=np.float64)
        yaw = np.deg2rad(float(delta_yaw_deg))
        c, s = np.cos(yaw / 2), np.sin(yaw / 2)
        w, x, y, z = pose[3:]
        result = self.move_to(
            arm=arm,
            xyz=pose[:3].tolist(),
            quat=[c * w - s * z, c * x - s * y, c * y + s * x, c * z + s * w],
            gripper=gripper,
            substeps=substeps,
        )
        result["requested_delta_yaw_deg"] = float(delta_yaw_deg)
        return result

    def set_gripper(
        self,
        *,
        arm: str,
        val: float,
        steps: int = 10,
    ) -> dict[str, Any]:
        if arm not in ("left", "right"):
            raise ValueError("arm must be 'left' or 'right'")
        if int(steps) < 1:
            raise ValueError("steps must be at least 1")
        state = np.asarray(
            self.env.last_obs["state"]["joint_position"], dtype=np.float64
        )
        current = float(state[6 if arm == "left" else 13])
        target = float(val)
        if not np.isfinite(target) or not 0.0 <= target <= 1.0:
            raise ValueError("val must be finite and within [0,1]")
        values = np.linspace(current, target, int(steps) + 1)[1:].tolist()
        execution = self.apply_qpos_updates(
            [{"arm": arm, "gripper": value} for value in values]
        )
        executed = int(execution.get("executed_actions", 0))
        now = np.asarray(self.env.last_obs["state"]["joint_position"], dtype=np.float64)
        return {
            **execution,
            **self._completion(
                requested=len(values),
                executed=executed,
                status=execution["episode_status"],
            ),
            "success": executed == len(values),
            "gripper_val": float(now[6 if arm == "left" else 13]),
        }

    def release(self, *, arm: str, val: float = 1.0, steps: int = 10) -> dict[str, Any]:
        return self.set_gripper(arm=arm, val=val, steps=steps)
