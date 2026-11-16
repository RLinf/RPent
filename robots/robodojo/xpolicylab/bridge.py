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

"""Bridge synchronous RPent primitives to externally stepped XPolicyLab actions."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from queue import Queue
from threading import Condition, Event, Lock
from typing import Any, Literal

import numpy as np
import yaml

from robots.robodojo.primitives import RoboDojoPrimitives
from rpent.tools import ToolResult
from rpent.tools.toolkit import ToolCancelled


class Bridge:
    """Serialize tool actions until the official environment acknowledges them."""

    def __init__(self, planner: Any = None) -> None:
        self.condition = Condition()
        self.cancelled = Event()
        self.actions = Queue()
        self.obs = None
        self.status = None
        self.ended = False
        self.pending = False
        self.held = None
        self.planner = planner
        self.capture_lock = Lock()
        self.revision = 0
        self.attached = False

    def check(self) -> None:
        if self.cancelled.is_set():
            raise ToolCancelled("Official episode closed")

    def stop(self) -> None:
        self.cancelled.set()
        with self.condition:
            self.condition.notify_all()

    def observe(self, obs: dict, step: int, step_limit: int, ended: bool) -> None:
        with self.condition:
            expected = (
                0 if self.status is None else self.status["step"] + int(self.pending)
            )
            if step != expected:
                self.stop()
                raise RuntimeError(f"Observation step mismatch: {step} != {expected}")
            self.obs = obs
            self.status = {"step": step, "step_limit": step_limit}
            if ended:
                self.status["success"] = obs["episode_status"]["success"]
            self.revision += 1
            self.ended = ended
            self.pending = False
            self.condition.notify_all()

    def targets(self) -> dict[str, list[float]]:
        return {
            f"{arm}_{field}": np.asarray(self.obs["state"][f"{arm}_{field}"]).tolist()
            for arm in ("left", "right")
            for field in ("arm_joint_state", "ee_joint_state")
        }

    def send(self, action: dict, expected_step: int) -> dict:
        with self.condition:
            self.check()
            if self.pending or expected_step != self.status["step"] or self.ended:
                raise RuntimeError("Action is out of sequence or episode ended")
            before = self.status["step"]
            self.pending = True
            self.actions.put(action)
            if not self.condition.wait_for(
                lambda: not self.pending or self.cancelled.is_set(), timeout=120
            ):
                self.stop()
                raise TimeoutError("Official action feedback did not arrive")
            self.check()
            return {
                "step_before": before,
                "step_after": self.status["step"],
                "executed_actions": 1,
                "episode_status": dict(self.status),
                "observation": self.obs,
                "terminated": self.ended,
                "truncated": self.status["step"] >= self.status["step_limit"],
            }

    def plan(self, arm: str, target_pose: list[float]) -> dict:
        from env.global_configs import ROOT_DIR
        from env.planner_manager.curobo_planner import CuroboPlanner

        root = Path(ROOT_DIR)
        config = yaml.safe_load((root / "env_cfg/robot/dual_x5.yml").read_text())
        robot = config["robots"][("left", "right").index(arm)]
        robot_dir = root / "Assets/Robots/x5"
        spec = yaml.safe_load((robot_dir / "robot_config.yml").read_text())
        if self.planner is None:
            sim = yaml.safe_load((root / "env_cfg/sim/sim_config.yml").read_text())
            self.planner = CuroboPlanner(
                robot_origin_pose=[0, 0, 0, 1, 0, 0, 0],
                active_joints_name=spec["arm_joints_name"],
                all_joints=spec["arm_joints_name"] + spec["gripper_joints_name"],
                dt=sim["dt"],
                yml_path=str(robot_dir / "curobo.yml"),
                table_height=0.74 - robot["default_root_pos"][2],
            )
        target = np.asarray(target_pose, dtype=float)
        if (
            target.shape != (7,)
            or not np.isfinite(target).all()
            or np.linalg.norm(target[3:]) < 1e-8
        ):
            raise ValueError("Expected finite xyz + nonzero wxyz quaternion")
        target[3:] /= np.linalg.norm(target[3:])
        return self.planner.plan_path(
            curr_joint_pos=self.obs["state"][f"{arm}_arm_joint_state"],
            target_ee_pose=target.tolist(),
            real_robot_pose=robot["default_root_pos"] + robot["default_root_rot"],
        )

    def call(
        self, method: str, *, kwargs: dict | None = None, timeout_s: float | None = None
    ) -> Any:
        self.check()
        kw = kwargs or {}
        if method == "env.get_env_meta":
            return {"mode": "xpolicylab"}
        if method == "env.reset":
            # Bind once to an initial observation; never reset the official simulator.
            if self.attached or self.status["step"] != 0:
                raise RuntimeError("Only the official client may reset an episode")
            self.attached = True
            return self.obs
        if method == "env.get_task_language":
            return self.obs["instruction"]
        if method == "env.get_obs":
            with self.capture_lock, self.condition:
                self.check()
                revision = self.revision
                self.actions.put({"_rpent_capture": True})
                if not self.condition.wait_for(
                    lambda: self.revision > revision or self.cancelled.is_set(),
                    timeout=120,
                ):
                    self.stop()
                    raise TimeoutError(
                        "Official observation refresh was not acknowledged"
                    )
                self.check()
                return self.obs
        if method == "env.get_status":
            return dict(self.status)
        if method == "env.get_motion_state":
            return {
                "targets": deepcopy(self.held or self.targets()),
                **{
                    f"{arm}_eef_pose": self.obs["state"][f"{arm}_ee_pose"]
                    for arm in ("left", "right")
                },
            }
        if method == "env.plan_arm_path":
            return self.plan(kw["arm"], kw["target_pose"])
        if method == "env.step_joint_update":
            action = deepcopy(self.held if self.held is not None else self.targets())
            arm = kw["arm"]
            if kw.get("arm_qpos") is not None:
                action[f"{arm}_arm_joint_state"] = kw["arm_qpos"]
            if kw.get("gripper") is not None:
                action[f"{arm}_ee_joint_state"] = [kw["gripper"]]
            result = self.send(action, kw["expected_step"])
            self.held = deepcopy(action)
            return result
        if method == "env.step":
            action = kw["flat_action"]
            result = self.send(action, kw["expected_step"])
            # EE commands hide the IK joint target. Start the next joint sequence
            # from measured joints, rather than inventing the unavailable target.
            self.held = None
            return (
                self.obs,
                None,
                self.ended,
                {**result, "status": result["episode_status"]},
            )
        if method == "env.close":
            return None
        raise NotImplementedError(method)


class PublicPrimitives(RoboDojoPrimitives):
    def set_gripper(
        self, *, arm: Literal["left", "right"], val: float, steps: int = 10
    ) -> ToolResult:
        """Interpolate commanded opening; no measured opening is available."""
        if arm not in ("left", "right") or not np.isfinite(val) or not 0 <= val <= 1:
            raise ValueError("Invalid arm or gripper target")
        if type(steps) is not int or steps < 1:
            raise ValueError("steps must be a positive integer")
        current = self.env.get_motion_state()["targets"][f"{arm}_ee_joint_state"][0]
        values = np.linspace(current, val, steps + 1)[1:]
        result = self._execute([{"arm": arm, "gripper": float(v)} for v in values])
        try:
            result["gripper_target"] = self.env.get_motion_state()["targets"][
                f"{arm}_ee_joint_state"
            ][0]
        except Exception as exc:
            result["feedback_error"] = str(exc)
        return ToolResult(data=result)
