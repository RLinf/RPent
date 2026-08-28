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

"""RoboTwin primitives built on RLinf environment APIs."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

from robots.robotwin.env_client import RoboTwinEnvClient
from robots.robotwin.robot_spec import MODEL_SPEC, ROBOTWIN_CAMERA_NAMES
from robots.robotwin.vla_client import LingBotVLAClient


def _qmult(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    w1, x1, y1, z1 = left
    w2, x2, y2, z2 = right
    return np.asarray(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        dtype=np.float64,
    )


class RoboTwinPrimitives:
    """Compose RoboTwin operations from the RLinf environment API."""

    def __init__(
        self,
        *,
        env: RoboTwinEnvClient,
        model: LingBotVLAClient,
        seed: int,
        check_cancelled: Callable[[], None],
        seed_mode: str = "exact",
    ):
        if seed_mode != "exact":
            raise ValueError("standard RoboTwin integration requires seed_mode='exact'")
        self.env = env
        self.model = model
        self.seed = int(seed)
        self._check_cancelled = check_cancelled
        self.policy_actions = 0
        self.native_actions = 0
        self._recording = False
        self._frames: list[np.ndarray] = []

    def start_recording(self) -> None:
        self._recording = True
        self._frames = []

    def record_frame(self, rgb: Any) -> None:
        self._frames.append(np.ascontiguousarray(np.asarray(rgb)))

    def recorded_frame_count(self) -> int:
        return len(self._frames)

    def stop_recording(self) -> list[np.ndarray]:
        frames = list(self._frames)
        self._recording = False
        self._frames = []
        return frames

    def frame_slice(self, start: int) -> list[np.ndarray]:
        return list(self._frames[int(start) :])

    def reset(
        self,
        *,
        instruction: str | None = None,
        feasibility_precheck: bool = True,
    ) -> dict[str, Any]:
        """Reset the RoboTwin episode and return the native info plus success."""
        del instruction, feasibility_precheck
        _, info = self.env.reset()
        return {**info, "success": True}

    @staticmethod
    def _completion(
        *, requested: int, executed: int, status: dict[str, Any]
    ) -> dict[str, Any]:
        step_lim = status.get("step_lim")
        budget_exhausted = step_lim is not None and int(
            status.get("take_action_cnt", 0)
        ) >= int(step_lim)
        completed = executed == requested
        if status.get("eval_success") is True:
            stop_reason = "native_success"
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

    def _build_lingbot_observation(self) -> dict[str, Any]:
        """Assemble the rgb-only observation the LingBot policy infers on."""
        views = {}
        for camera_name in ROBOTWIN_CAMERA_NAMES:
            views[camera_name] = {
                "rgb": np.asarray(self.env.render_camera(camera_name))
            }
        return {
            "views": views,
            "robot_state": self.env.last_info["robot_state"],
            "task_language": self.env.get_task_language(),
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
                if not np.isfinite(gripper):
                    raise ValueError("gripper must be finite")
                item["gripper"] = gripper
            normalized.append(item)
        return normalized

    def apply_qpos_updates(
        self,
        updates: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Compose qpos waypoint actions from the latest robot state."""
        if self.env.terminated or self.env.truncated:
            raise RuntimeError("RoboTwin common episode is terminal; reset is required")
        updates = self._validate_qpos_updates_request(updates)
        executed = 0
        episode_status: dict[str, Any] | None = None
        for update in updates:
            state = self.env.last_info["robot_state"]
            qpos_target14 = np.asarray(state.get("qpos_target14"), dtype=np.float64)
            if qpos_target14.shape != (14,) or not np.isfinite(qpos_target14).all():
                raise ValueError(
                    "RoboTwin robot_state.qpos_target14 must be finite and have "
                    "shape (14,)"
                )
            action = qpos_target14.copy()
            offset = 0 if update["arm"] == "left" else 7
            if "arm_qpos" in update:
                action[offset : offset + 6] = update["arm_qpos"]
            if "gripper" in update:
                action[offset + 6] = update["gripper"]
            obs, _, _, _, info = self.env.step(action, action_type="qpos")
            if self._recording and isinstance(obs, dict) and "main_images" in obs:
                self.record_frame(obs["main_images"])
            executed += int(info.get("executed_actions", 0))
            episode_status = info["episode_status"]
            if self.env.terminated or self.env.truncated:
                break
        if episode_status is None:
            raise RuntimeError("RoboTwin qpos composition executed no updates")
        return {
            "action_type": "qpos",
            "requested_actions": len(updates),
            "executed_actions": executed,
            "episode_status": episode_status,
        }

    def lingbot_act(
        self, *, chunks: int = 4, use_length: int = 50, prompt: str | None = None
    ) -> dict[str, Any]:
        """Infer and execute up to the requested number of LingBot EEF action chunks."""
        if int(chunks) < 1:
            raise ValueError("chunks must be at least 1")
        if int(use_length) != MODEL_SPEC.use_length:
            raise ValueError(
                f"RoboTwin LingBot requires use_length={MODEL_SPEC.use_length}"
            )
        executed = 0
        requested = int(chunks) * MODEL_SPEC.use_length
        native_prompt = None
        for _ in range(int(chunks)):
            self._check_cancelled()
            status = self.env.last_info["episode_status"]
            step_lim = status.get("step_lim")
            budget_exhausted = step_lim is not None and int(
                status.get("take_action_cnt", 0)
            ) >= int(step_lim)
            if status.get("eval_success") is True or budget_exhausted:
                break
            observation = self._build_lingbot_observation()
            native_prompt = observation["task_language"]
            actions = self.model.infer(observation)[: MODEL_SPEC.use_length]
            self._check_cancelled()
            payload, _, _, _, info = self.env.chunk_step(
                actions,
                action_type="ee",
                return_all_frames=self._recording
                and self.env.execution_capabilities.get("chunk_step_all_frames")
                is True,
            )
            if self._recording and isinstance(payload, dict) and "frames" in payload:
                for frame in payload["frames"]:
                    self.record_frame(frame)
            count = int(info.get("executed_actions", 0))
            executed += count
            self.policy_actions += count
            self.native_actions += count
            self._check_cancelled()
        status = self.env.last_info["episode_status"]
        return {
            **self._completion(
                requested=requested,
                executed=executed,
                status=status,
            ),
            "success": True,
            "prompt": native_prompt,
            "agent_prompt_ignored": prompt is not None,
            "ignored_agent_prompt": prompt,
            "episode_status": status,
        }

    def move_to(
        self,
        *,
        arm: str,
        xyz: list[float],
        quat: list[float] | None = None,
        gripper: float | None = None,
        substeps: int = 25,
        _primitive_name: str = "move_to",
    ) -> dict[str, Any]:
        """Plan and execute a qpos path to a target end-effector pose."""
        del _primitive_name
        if int(substeps) < 0:
            raise ValueError("substeps must be non-negative")
        self._check_cancelled()
        state = self.env.last_info["robot_state"]
        if quat is None:
            key = "left_eef_pose" if arm == "left" else "right_eef_pose"
            quat = np.asarray(state[key], dtype=np.float64)[3:].tolist()
        target = np.asarray([*xyz, *quat], dtype=np.float64)
        planned = self.env.plan_arm_path(arm, target)
        self._check_cancelled()
        if planned["status"] != "Success" or planned.get("position") is None:
            return {
                "completed": False,
                "requested_steps": 0,
                "executed_steps": 0,
                "stop_reason": "plan_failed",
                "success": False,
                "plan_status": planned["status"],
                "hint": "target may be unreachable or in collision",
            }
        path = np.asarray(planned["position"], dtype=np.float64)
        if substeps == 1:
            path = path[-1:]
        elif substeps >= 2 and len(path) > substeps:
            indices = np.linspace(0, len(path) - 1, substeps).astype(int)
            path = path[indices]
        updates = [
            {"arm": arm, "arm_qpos": waypoint, "gripper": gripper} for waypoint in path
        ]
        execution = self.apply_qpos_updates(updates)
        executed = int(execution.get("executed_actions", 0))
        self.native_actions += executed
        self._check_cancelled()
        status = execution["episode_status"]
        final = self.env.last_info["robot_state"]
        key = "left_eef_pose" if arm == "left" else "right_eef_pose"
        final_pose = np.asarray(final[key], dtype=np.float64)
        return {
            **execution,
            **self._completion(
                requested=len(updates),
                executed=executed,
                status=status,
            ),
            "success": True,
            "plan_status": planned["status"],
            "waypoints": len(path),
            "final_eef_xyz": final_pose[:3].tolist(),
            "final_dist_m": float(
                np.linalg.norm(final_pose[:3] - np.asarray(xyz, dtype=np.float64))
            ),
        }

    def rotate_wrist(
        self,
        *,
        arm: str,
        delta_yaw_deg: float,
        gripper: float | None = None,
        substeps: int = 25,
    ) -> dict[str, Any]:
        """Rotate an arm about world z by the requested yaw delta."""
        state = self.env.last_info["robot_state"]
        key = "left_eef_pose" if arm == "left" else "right_eef_pose"
        pose = np.asarray(state[key], dtype=np.float64)
        yaw = np.deg2rad(float(delta_yaw_deg))
        world_z = np.asarray([np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)])
        result = self.move_to(
            arm=arm,
            xyz=pose[:3].tolist(),
            quat=_qmult(world_z, pose[3:]).tolist(),
            gripper=gripper,
            substeps=substeps,
            _primitive_name="rotate_wrist",
        )
        result["requested_delta_yaw_deg"] = float(delta_yaw_deg)
        return result

    def set_gripper(
        self,
        *,
        arm: str,
        val: float,
        steps: int = 10,
        _primitive_name: str = "set_gripper",
    ) -> dict[str, Any]:
        """Interpolate the gripper command to a target value over multiple steps."""
        del _primitive_name
        if int(steps) < 1:
            raise ValueError("steps must be at least 1")
        self._check_cancelled()
        state = self.env.last_info["robot_state"]
        current = float(state[f"{arm}_gripper"])
        step_count = int(steps)
        target = float(val)
        values = [
            current + (target - current) * index / step_count
            for index in range(1, step_count + 1)
        ]
        updates = [{"arm": arm, "gripper": float(value)} for value in values]
        execution = self.apply_qpos_updates(updates)
        executed = int(execution.get("executed_actions", 0))
        self.native_actions += executed
        self._check_cancelled()
        status = execution["episode_status"]
        now = self.env.last_info["robot_state"]
        return {
            **execution,
            **self._completion(
                requested=len(updates),
                executed=executed,
                status=status,
            ),
            "success": True,
            "gripper_val": float(now[f"{arm}_gripper"]),
        }

    def release(self, *, arm: str, val: float = 1.0, steps: int = 10) -> dict[str, Any]:
        """Open the gripper to the requested release value."""
        return self.set_gripper(
            arm=arm,
            val=val,
            steps=steps,
            _primitive_name="release",
        )

    def status(self) -> dict[str, Any]:
        """Return the native episode status plus action counters."""
        return {
            **self.env.last_info["episode_status"],
            "policy_actions": self.policy_actions,
            "native_actions": self.native_actions,
        }

    def finish(self, *, status: str, summary: str) -> dict[str, Any]:
        """Finish the Planner run and verify success against native episode status."""
        requested_success = status.lower() == "success"
        try:
            native = self.status()
        except Exception as error:  # The terminal tool must still stop the Planner.
            return {
                "_finish": True,
                "status": "error",
                "summary": summary,
                "requested_status": status,
                "requested_success": requested_success,
                "runtime_error": f"{type(error).__name__}: {error}",
            }
        verified_success = native.get("eval_success") is True
        reported_status = (
            "success"
            if verified_success
            else ("failure" if requested_success else status)
        )
        return {
            "_finish": True,
            "status": reported_status,
            "summary": summary,
            "requested_success": requested_success,
            "success": verified_success,
            "episode_status": native,
        }
