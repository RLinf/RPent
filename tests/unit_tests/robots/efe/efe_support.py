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

"""Shared offline EFE service double for unit and integration tests."""

from __future__ import annotations

import copy
from typing import Any

import numpy as np

JOINTS = ("waist", "shoulder", "elbow", "forearm_roll", "wrist_angle", "wrist_rotate")
TCP = "tcp"
PROPRIO_LAYOUT = [
    *(f"q_{name}" for name in JOINTS),
    "gripper_opening",
    *(f"{TCP}_{axis}" for axis in ("x", "y", "z", "qw", "qx", "qy", "qz")),
    "cmd_gripper",
]

SPEC: dict[str, Any] = {
    "protocol": "eai-standard/2.2",
    "robot": "ALOHA arm (6 joints, parallel gripper)",
    "control_hz": 500.0,
    "max_chunk_s": 10.0,
    "max_control_steps": 600000,
    "max_frames_per_call": 3000,
    "cameras": {
        "table_cam_front": {},
        "table_cam_left": {},
        "wrist_cam": {},
    },
    "default_cameras": ["table_cam_front", "table_cam_left", "wrist_cam"],
    "action_types": {
        "joint_position": {
            "dim": 7,
            "layout": [*JOINTS, "gripper"],
            "limits": {
                **{name: [-3.0, 3.0] for name in JOINTS},
                "gripper": [0.002, 0.037],
            },
            "units": dict.fromkeys(JOINTS, "rad") | {"gripper": "m"},
        },
        "done": {"dim": 0, "layout": []},
        "not_yet": [
            "ee_pose",
            "joint_delta",
            "joint_velocity",
            "ee_delta",
            "joint_torque",
        ],
    },
    "proprio_layout": PROPRIO_LAYOUT,
    "proprio_groups": {
        "joint_pos": [0, 6, ""],
        "tcp_pose": [7, 7, ""],
        "command": [14, 7, ""],
    },
    "kinematics": {
        "joint_names": [f"q_{name}" for name in JOINTS],
        "tcp_frames": [TCP],
        "ik_targets": [TCP],
    },
    "budget": {"reset": 0, "steps": None, "wall_clock_s": 3600},
}


def proprio(
    *,
    position: tuple[float, float, float] = (0.0, 0.0, 0.3),
    quaternion: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0),
    joints: list[float] | None = None,
    gripper: float = 0.037,
) -> dict[str, Any]:
    """Return one public proprioception record for the fake service."""
    values = {
        **{
            f"q_{name}": value
            for name, value in zip(JOINTS, joints or [0.0] * len(JOINTS), strict=False)
        },
        "gripper_opening": gripper,
        **{
            f"{TCP}_{axis}": value
            for axis, value in zip(
                ("x", "y", "z", "qw", "qx", "qy", "qz"),
                (*position, *quaternion),
                strict=False,
            )
        },
        "cmd_gripper": gripper,
    }
    return {"layout": PROPRIO_LAYOUT, "now": [values[name] for name in PROPRIO_LAYOUT]}


class FakeResult:
    """The subset of ``rcb_client.Result`` the tools read."""

    def __init__(
        self,
        meta: dict[str, Any],
        *,
        last: dict[str, np.ndarray] | None = None,
        last_depth: dict[str, np.ndarray] | None = None,
    ) -> None:
        self.meta = meta
        self.last = last or {}
        self.last_depth = last_depth or {}


class FakeEfeClient:
    """Offline stand-in for :class:`robots.efe.env_client.EfeEnvClient`."""

    def __init__(self, spec: dict[str, Any] | None = None) -> None:
        self.spec = copy.deepcopy(spec or SPEC)
        self.done = False
        self.done_reason: str | None = None
        self.last_meta: dict[str, Any] = {
            "proprio": proprio(),
            "step": 0,
            "control_steps": 0,
        }
        self.submit_calls = 0
        self.observe_calls = 0
        self.control_steps = 0
        self.submitted: list[tuple[str, np.ndarray, float | None]] = []
        self.observe_requests: list[dict[str, Any]] = []
        self.reset_calls = 0
        self.finish_calls = 0

    @property
    def reset_budget(self) -> int:
        return int((self.spec.get("budget") or {}).get("reset") or 0)

    @property
    def success_hidden(self) -> bool:
        return self.spec.get("success_signal") == "hidden"

    def observe(
        self,
        cameras: list[str] | None = None,
        *,
        depth: bool = False,
        video: str = "frames",
    ):
        self.observe_calls += 1
        self.observe_requests.append(
            {"cameras": cameras, "depth": depth, "video": video}
        )
        names = self.spec["default_cameras"] if cameras is None else cameras
        last = {name: np.zeros((4, 4, 3), dtype=np.uint8) for name in names}
        last_depth = (
            {name: np.zeros((4, 4), dtype=np.float32) for name in names}
            if depth
            else {}
        )
        return FakeResult(
            {
                "proprio": proprio(),
                "step": 0,
                "control_steps": self.control_steps,
                "sim_time_s": 0.0,
                "time_left_s": 3600.0,
                "done": self.done,
                "reason": self.done_reason,
                "cameras": {name: {} for name in names},
            },
            last=last,
            last_depth=last_depth,
        )

    def submit(
        self,
        action_type: str,
        actions: Any,
        *,
        dt: float | None = None,
        cameras: list[str] | None = None,
        video: str = "none",
    ):
        array = np.asarray(actions, dtype=np.float64)
        rows = 1 if array.ndim == 1 else array.shape[0]
        self.submit_calls += 1
        self.control_steps += rows
        self.submitted.append((action_type, array, dt))
        self.last_meta = {
            "proprio": proprio(),
            "step": self.control_steps,
            "control_steps": self.control_steps,
            "done": self.done,
            "reason": self.done_reason,
            "control_steps_chunk": rows,
            "executed_rows": rows,
            "time_left_s": 3600.0,
        }
        return FakeResult(dict(self.last_meta))

    def status(self) -> dict[str, Any]:
        # GET /status carries no camera, proprioception, or clock-relative fields.
        return {
            "step": self.control_steps,
            "calls": self.submit_calls + self.observe_calls,
            "control_steps": self.control_steps,
            "done": self.done,
            "reason": self.done_reason,
            "elapsed_s": 0.0,
        }

    def wait_ready(self, timeout_s: float = 600.0) -> dict[str, Any]:
        return self.status()

    def reset(self, *, cameras: list[str] | None = None) -> dict[str, Any]:
        self.reset_calls += 1
        self.done = False
        self.done_reason = None
        left = max(self.reset_budget - self.reset_calls, 0)
        return {
            "episode": self.reset_calls,
            "resets_left": left,
            "step": 0,
            "control_steps": 0,
            "done": False,
            "reason": None,
        }

    def finish(self) -> dict[str, Any]:
        self.finish_calls += 1
        self.done = True
        self.done_reason = "finished"
        return {
            "ok": True,
            "step": self.control_steps,
            "control_steps": self.control_steps,
            "done": True,
            "reason": "finished",
        }

    def ik(self, targets: dict[str, list[float]], *, seed: list[float] | None = None):
        return {
            "ok": True,
            "q": list(seed or [0.0] * len(JOINTS)),
            "error": {TCP: {"pos_m": 0.0, "rot_rad": 0.0}},
        }
