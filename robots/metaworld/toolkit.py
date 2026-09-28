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

"""Visual observations and bounded Cartesian primitives for MetaWorld."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from rpent.dashboard.events import DashboardEventSink
from rpent.session import EnvState
from rpent.tools.toolkit import Toolkit, readonly

if TYPE_CHECKING:
    from rpent.memory import MemoryManager


def back_project_pixel(obs: dict, row: int, col: int) -> list[float]:
    """Project one valid metric-depth pixel using the documented camera axes."""
    depth = np.asarray(obs["depth"])
    if (
        isinstance(row, bool)
        or isinstance(col, bool)
        or not isinstance(row, int)
        or not isinstance(col, int)
    ):
        raise ValueError("row and col must be integer pixels")
    if not (0 <= row < depth.shape[0] and 0 <= col < depth.shape[1]):
        raise ValueError("pixel lies outside the rendered image")
    z = float(depth[row, col])
    if not np.isfinite(z) or z <= 0:
        raise ValueError("pixel has no finite positive depth")
    meta = obs["camera"]
    if "clip_range" in meta and z >= meta["clip_range"][1] * (1 - 1e-5):
        raise ValueError("pixel is at the far clipping plane, not a visible surface")
    k = np.asarray(meta["intrinsics"])
    camera_point = np.array(
        [(col - k[0, 2]) * z / k[0, 0], -(row - k[1, 2]) * z / k[1, 1], -z]
    )
    return (
        np.asarray(meta["position_world"])
        + np.asarray(meta["rotation_world_from_camera"]) @ camera_point
    ).tolist()


class MetaWorldToolkit(Toolkit):
    """Drive a single fixed episode without exposing native expert state."""

    def __init__(
        self,
        *,
        runtime_kwargs: dict[str, Any],
        dashboard_events: DashboardEventSink,
        memory: MemoryManager,
        state_output_dir: Path | str,
    ) -> None:
        self.env = runtime_kwargs["env_client"]
        self.obs = self.env.last_obs
        super().__init__(
            dashboard_events=dashboard_events,
            state=EnvState(state_output_dir),
            memory=memory,
        )
        number = {"type": "number"}
        for name, description, properties, required, handler in [
            (
                "view_env_state",
                "Read the current RGB image and arm proprioception.",
                {},
                [],
                self.view_env_state,
            ),
            (
                "back_project",
                "Map an image pixel (row, col) to a visible surface in world meters.",
                {"row": {"type": "integer"}, "col": {"type": "integer"}},
                ["row", "col"],
                self.back_project,
            ),
            (
                "move_to",
                "Move toward a world XYZ target with fixed wrist orientation; gripper +1 closes, -1 opens. Check reached.",
                {
                    "target_xyz": {
                        "type": "array",
                        "items": number,
                        "minItems": 3,
                        "maxItems": 3,
                    },
                    "gripper": number,
                    "max_steps": {"type": "integer", "minimum": 1, "maximum": 100},
                },
                ["target_xyz", "gripper"],
                self.move_to,
            ),
            (
                "set_gripper",
                "Open (-1) or close (+1) while holding the current end-effector position.",
                {
                    "gripper": number,
                    "steps": {"type": "integer", "minimum": 1, "maximum": 20},
                },
                ["gripper"],
                self.set_gripper,
            ),
        ]:
            self.add_tool(
                name,
                {
                    "name": name,
                    "description": description,
                    "input_schema": {
                        "type": "object",
                        "properties": properties,
                        "required": required,
                        "additionalProperties": False,
                    },
                },
                handler,
            )
        self.get_env_state(command={}, result={}, elapsed_s=0)
        self._publish_step(self._state.latest_record())

    @readonly
    def view_env_state(self) -> dict:
        """Read the latest camera snapshot without advancing simulation."""
        return self._view()

    def _view(self) -> dict:
        record = self._state.latest_record()
        return {
            **record.state,
            "episode_ended": bool(record.terminated or record.truncated),
            "terminated": record.terminated,
            "truncated": record.truncated,
            "_image_bytes": self._state.load_bytes("camera.png", step=-1),
        }

    @readonly
    def back_project(self, row: int, col: int) -> dict:
        """Locate a visible surface from the latest snapshot's depth."""
        return {
            "world_xyz": back_project_pixel(
                {
                    "depth": self._state.load("depth.npy", step=-1),
                    "camera": self._state.latest_record().state["camera"],
                },
                row,
                col,
            ),
            "pixel": [row, col],
        }

    @staticmethod
    def _gripper(value: float) -> float:
        value = float(value)
        if not np.isfinite(value) or not -1 <= value <= 1:
            raise ValueError("gripper must be finite and in [-1, 1]")
        return value

    def _step(self, action: list[float]) -> None:
        self.raise_if_cancelled()
        if self.obs["terminated"] or self.obs["truncated"]:
            raise RuntimeError("episode ended; call finish")
        self.obs = self.env.step(action)[0]

    def move_to(
        self, target_xyz: list[float], gripper: float, max_steps: int = 50
    ) -> dict:
        """Apply bounded Cartesian corrections with the requested gripper action."""
        target = np.asarray(target_xyz, dtype=float)
        if target.shape != (3,) or not np.isfinite(target).all():
            raise ValueError("target_xyz must be three finite world coordinates")
        if (
            not isinstance(max_steps, int)
            or isinstance(max_steps, bool)
            or not 1 <= max_steps <= 100
        ):
            raise ValueError("max_steps must be an integer in [1, 100]")
        gripper = self._gripper(gripper)
        count = 0
        while count < max_steps:
            delta = target - np.asarray(self.obs["eef_position"])
            if (
                np.linalg.norm(delta) < 0.005
                or self.obs["terminated"]
                or self.obs["truncated"]
            ):
                break
            # Native MetaWorld scales XYZ actions by 0.01 meters.
            self._step([*np.clip(delta / 0.01, -1, 1).tolist(), gripper])
            count += 1
        distance = float(np.linalg.norm(target - np.asarray(self.obs["eef_position"])))
        return {"reached": distance < 0.005, "distance_m": distance, "steps": count}

    def set_gripper(self, gripper: float, steps: int = 10) -> dict:
        """Apply a gripper command without requesting Cartesian displacement."""
        gripper = self._gripper(gripper)
        if (
            not isinstance(steps, int)
            or isinstance(steps, bool)
            or not 1 <= steps <= 20
        ):
            raise ValueError("steps must be an integer in [1, 20]")
        count = 0
        while count < steps and not (self.obs["terminated"] or self.obs["truncated"]):
            self._step([0, 0, 0, gripper])
            count += 1
        return {"steps": count, "gripper_opening": self.obs["gripper_opening"]}

    def get_env_state(self, *, command: dict, result: dict, elapsed_s: float) -> dict:
        """Record the observation and attach the motion result to the tool response."""
        self.obs = self.env.get_obs()
        public = {
            k: self.obs[k]
            for k in [
                "eef_position",
                "gripper_opening",
                "instruction",
                "steps",
                "camera",
            ]
        }
        with self._state.record_step(
            state=public,
            terminated=self.obs["terminated"],
            truncated=self.obs["truncated"],
            command=command,
            result=result,
            elapsed_s=elapsed_s,
        ):
            self._state.save("camera.png", self.obs["rgb"])
            self._state.save("depth.npy", self.obs["depth"])
        return {**self._view(), "motion_result": result}

    def solved(self) -> bool:
        """Read the native success accumulator only for run finalization."""
        return self.env.is_success()
