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

"""R2 planner tools, primitives, and state capture."""

from __future__ import annotations

from collections.abc import Callable
from typing import Annotated, Any, Literal

import numpy as np
from pydantic import Field

from rpent.session import EnvState, StepRecord
from rpent.tools import ToolResult, tool

Arm = Literal["left", "right"]
Camera = Literal["head", "wrist_left", "wrist_right"]
Vec3 = Annotated[list[float], Field(min_length=3, max_length=3)]

#: Image order in ``view_env_state``; the head camera sees the workspace.
CAMERA_ORDER = ("head", "wrist_left", "wrist_right")


class R2Primitives:
    """Bounded motions on a remote R2; the env server enforces the site limits."""

    def __init__(self, *, env: Any, check_cancelled: Callable[[], None]) -> None:
        self.env = env
        self._check_cancelled = check_cancelled

    @tool
    def move_to(self, arm: Arm, xyz: Vec3, rpy: Vec3 | None = None) -> ToolResult:
        """Move one gripper's TCP to an absolute base_link position in metres.

        Args:
            arm: The arm to move.
            xyz: Target TCP position.
            rpy: Target roll, pitch, yaw in radians (extrinsic xyz); omit to keep the current orientation.
        """
        self._check_cancelled()
        return ToolResult(data=self.env.move_to(arm, xyz, rpy))

    @tool
    def move_delta(self, arm: Arm, delta_xyz: Vec3) -> ToolResult:
        """Translate one TCP by a base_link offset in metres, keeping its orientation."""
        self._check_cancelled()
        return ToolResult(data=self.env.move_delta(arm, delta_xyz))

    @tool
    def rotate_delta(self, arm: Arm, delta_rpy: Vec3) -> ToolResult:
        """Rotate one TCP about the base_link x, y, z axes by radians, keeping its position."""
        self._check_cancelled()
        return ToolResult(data=self.env.rotate_delta(arm, delta_rpy))

    @tool
    def set_gripper(
        self, arm: Arm, opening: Annotated[float, Field(ge=0, le=1)]
    ) -> ToolResult:
        """Move one gripper to an opening between 0 (closed) and 1 (fully open)."""
        self._check_cancelled()
        return ToolResult(data=self.env.set_gripper(arm, opening))

    @tool
    def look(self, yaw: float, pitch: float) -> ToolResult:
        """Turn the head camera to absolute neck angles in radians; negative pitch looks down, positive yaw looks left."""
        self._check_cancelled()
        return ToolResult(data=self.env.look(yaw, pitch))

    @tool
    def go_home(self) -> ToolResult:
        """Return both arms to the rest pose, hanging beside the body."""
        self._check_cancelled()
        return ToolResult(data=self.env.go_home())

    @tool
    def move_base(
        self,
        vx: float,
        vy: float,
        wz: float,
        duration: Annotated[float, Field(gt=0)],
    ) -> ToolResult:
        """Drive the chassis at a body-frame velocity (m/s, m/s, rad/s) for a duration in seconds, then stop."""
        self._check_cancelled()
        return ToolResult(data=self.env.move_base(vx, vy, wz, duration))


def dump_state(
    primitives: R2Primitives,
    state: EnvState,
    *,
    command: dict[str, Any] | None,
    result: dict[str, Any] | None,
    elapsed_s: float | None,
    extra: dict[str, Any] | None = None,
) -> StepRecord:
    """Record the robot state, camera images, depth, and camera poses."""
    observation = primitives.env.observe()
    views = observation["views"]
    snapshot = dict(observation["robot_state"])
    errors = {name: view["error"] for name, view in views.items() if "error" in view}
    if errors:
        snapshot["camera_errors"] = errors
    snapshot.update(extra or {})
    with state.record_step(
        state=snapshot, command=command, result=result, elapsed_s=elapsed_s
    ) as step:
        for name, view in views.items():
            if "rgb" in view:
                state.save(f"{name}.png", view["rgb"], step=step)
            if "depth_mm" in view:
                state.save(f"{name}_depth_mm.npy", view["depth_mm"], step=step)
        state.save("camera_meta.json", observation["camera_meta"], step=step)
    return state.get(step)


@tool(readonly=True, exclude=("state",))
def view_env_state(
    step: Annotated[int, Field(json_schema_extra={"default": -1})] = -1,
    *,
    state: EnvState,
) -> ToolResult:
    """Read one recorded R2 state with its head and wrist camera images."""
    record = state.get(step)
    output = record.to_blob()
    names = [n for n in CAMERA_ORDER if state.exists(f"{n}.png", step=record.step_idx)]
    output["images"] = names
    images = [state.load_bytes(f"{n}.png", step=record.step_idx) for n in names]
    return ToolResult(data=output, images=images)


@tool(readonly=True, exclude=("state",))
def view_camera_meta(
    step: Annotated[int, Field(json_schema_extra={"default": -1})] = -1,
    *,
    state: EnvState,
) -> ToolResult:
    """Read the recorded camera intrinsics and base_link camera poses."""
    record = state.get(step)
    return ToolResult(
        data={
            "step": record.step_idx,
            "camera_meta": state.load("camera_meta.json", step=record.step_idx),
        }
    )


def _undistort(
    x_distorted: float, y_distorted: float, coefficients: list[float] | None
) -> tuple[float, float]:
    """Invert OpenCV's ``k1, k2, p1, p2, k3`` model on normalized coordinates."""
    if not coefficients:
        return x_distorted, y_distorted
    k1, k2, p1, p2, k3 = coefficients
    x, y = x_distorted, y_distorted
    for _ in range(20):
        r2 = x * x + y * y
        radial = 1.0 + r2 * (k1 + r2 * (k2 + r2 * k3))
        x = (x_distorted - (2 * p1 * x * y + p2 * (r2 + 2 * x * x))) / radial
        y = (y_distorted - (p1 * (r2 + 2 * y * y) + 2 * p2 * x * y)) / radial
    return x, y


def _point(
    depth_mm: np.ndarray,
    meta: dict[str, Any],
    row: int,
    col: int,
    radius: int,
) -> dict[str, Any]:
    height, width = depth_mm.shape
    if not (0 <= row < height and 0 <= col < width):
        return {
            "pixel": [row, col],
            "error": f"pixel is outside the {height}x{width} image",
        }
    window = depth_mm[
        max(0, row - radius) : row + radius + 1,
        max(0, col - radius) : col + radius + 1,
    ]
    valid = window[window > 0]
    if valid.size == 0:
        return {"pixel": [row, col], "error": "no valid depth around this pixel"}
    z = float(np.median(valid)) / 1000.0
    k = meta["intrinsics"]
    x, y = _undistort(
        (col - k["cx"]) / k["fx"], (row - k["cy"]) / k["fy"], meta.get("distortion")
    )
    camera_point = np.array([x * z, y * z, z, 1.0])
    xyz = np.asarray(meta["T_base_camera"], dtype=np.float64) @ camera_point
    return {
        "pixel": [row, col],
        "xyz": [round(float(v), 4) for v in xyz[:3]],
        "depth_m": round(z, 4),
        "valid_points": int(valid.size),
    }


@tool(readonly=True, exclude=("state",))
def back_project(
    pixels: Annotated[
        list[Annotated[list[int], Field(min_length=2, max_length=2)]],
        Field(min_length=1, max_length=32),
    ],
    camera: Camera = "head",
    step: Annotated[int, Field(json_schema_extra={"default": -1})] = -1,
    neighborhood: Annotated[
        int, Field(ge=0, le=10, json_schema_extra={"default": 2})
    ] = 2,
    *,
    state: EnvState,
) -> ToolResult:
    """Convert [row, col] pixels of a recorded camera image into base_link points.

    Args:
        pixels: Pixels as [row, col] in that camera's image.
        camera: The camera whose image the pixels come from.
        step: The recorded state the image belongs to.
        neighborhood: Half-width of the window whose median depth is used.
    """
    record = state.get(step)
    meta = state.load("camera_meta.json", step=record.step_idx)["cameras"].get(camera)
    depth_name = f"{camera}_depth_mm.npy"
    missing = []
    if meta is None:
        missing.append(f"camera {camera!r} is not configured")
    else:
        if meta.get("intrinsics") is None:
            missing.append("intrinsics (perception.cameras in the site config)")
        if meta.get("T_base_camera") is None:
            missing.append("a camera pose (perception.urdf in the site config)")
    if not state.exists(depth_name, step=record.step_idx):
        missing.append("a depth image (robot.enable_depth in the site config)")
    if missing:
        return ToolResult(
            data={
                "error": f"cannot back-project {camera!r}: missing "
                + "; ".join(missing),
                "step": record.step_idx,
            }
        )
    depth_mm = state.load(depth_name, step=record.step_idx)
    if list(depth_mm.shape) != meta["image_shape"]:
        return ToolResult(
            data={
                "error": (
                    f"{camera!r} depth {list(depth_mm.shape)} is not aligned with "
                    f"its {meta['image_shape']} color image"
                ),
                "step": record.step_idx,
            }
        )
    points = [
        _point(depth_mm, meta, int(r), int(c), int(neighborhood)) for r, c in pixels
    ]
    return ToolResult(
        data={
            "step": record.step_idx,
            "camera": camera,
            "frame": "base_link",
            "unit": "metre",
            "points": points,
        }
    )
