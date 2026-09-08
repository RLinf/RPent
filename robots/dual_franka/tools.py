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

from collections.abc import Sequence
from typing import Any

import numpy as np

from robots.franka.tools import FrankaPrimitives, coerce_vec3
from rpent.session import EnvState, StepRecord
from rpent.tools.toolkit import readonly

_ARM_PROPERTY = {
    "type": "string",
    "enum": ["left", "right"],
    "description": "Which arm to command; the other arm is left uncommanded.",
}

TOOLS_SPEC = [
    {
        "name": "view_env_state",
        "description": "Read a dual-Franka state snapshot and its synchronized RGB images.",
        "input_schema": {
            "type": "object",
            "properties": {"step": {"type": "integer", "default": -1}},
        },
    },
    {
        "name": "view_camera_meta",
        "description": "Read camera intrinsics, serials, and projection metadata for the dual-Franka rig.",
        "input_schema": {
            "type": "object",
            "properties": {"step": {"type": "integer", "default": -1}},
        },
    },
    {
        "name": "back_project_base_pixel",
        "description": "Back-project one base-camera pixel into shared right-base coordinates.",
        "input_schema": {
            "type": "object",
            "properties": {
                "row": {"type": "integer", "minimum": 0},
                "col": {"type": "integer", "minimum": 0},
                "target_name": {"type": "string", "default": "target"},
                "step": {"type": "integer"},
                "window_radius": {"type": "integer", "minimum": 0, "default": 2},
            },
            "required": ["row", "col"],
        },
    },
    {
        "name": "back_project_d455_pixel",
        "description": "Back-project one D455 pixel into shared right-base coordinates.",
        "input_schema": {
            "type": "object",
            "properties": {
                "row": {"type": "integer", "minimum": 0},
                "col": {"type": "integer", "minimum": 0},
                "target_name": {"type": "string", "default": "target"},
                "step": {"type": "integer"},
                "window_radius": {"type": "integer", "minimum": 0, "default": 2},
            },
            "required": ["row", "col"],
        },
    },
    {
        "name": "move_delta",
        "description": "Move one Franka TCP by a bounded world-frame xyz delta in meters.",
        "input_schema": {
            "type": "object",
            "properties": {
                "arm": _ARM_PROPERTY,
                "delta_xyz": {
                    "type": "array",
                    "items": {"type": "number"},
                    "minItems": 3,
                    "maxItems": 3,
                },
            },
            "required": ["arm", "delta_xyz"],
        },
    },
    {
        "name": "rotate_delta",
        "description": "Rotate one Franka TCP by a bounded world-frame rpy delta in radians.",
        "input_schema": {
            "type": "object",
            "properties": {
                "arm": _ARM_PROPERTY,
                "delta_rpy": {
                    "type": "array",
                    "items": {"type": "number"},
                    "minItems": 3,
                    "maxItems": 3,
                },
            },
            "required": ["arm", "delta_rpy"],
        },
    },
    {
        "name": "open_gripper",
        "description": "Open one Franka gripper and wait for the command to settle.",
        "input_schema": {
            "type": "object",
            "properties": {"arm": _ARM_PROPERTY},
            "required": ["arm"],
        },
    },
    {
        "name": "close_gripper",
        "description": "Close one Franka gripper and wait for the command to settle.",
        "input_schema": {
            "type": "object",
            "properties": {"arm": _ARM_PROPERTY},
            "required": ["arm"],
        },
    },
    {
        "name": "vla_grasp",
        "description": "Run bounded real-world dual-arm VLA action chunks for a grasp attempt.",
        "input_schema": {
            "type": "object",
            "properties": {
                "prompt": {"type": "string"},
                "max_chunks": {"type": "integer", "minimum": 1, "maximum": 20},
            },
            "required": ["prompt"],
        },
    },
]


def coerce_arm(value: Any) -> str:
    """Return exactly ``'left'`` or ``'right'`` or raise a useful error."""
    arm = str(value).strip().lower()
    if arm not in {"left", "right"}:
        raise ValueError("arm must be exactly 'left' or 'right'")
    return arm


class DualFrankaPrimitives(FrankaPrimitives):
    """Safe agent-facing operations over a remote dual-Franka environment."""

    def move_delta(self, arm: str, delta_xyz: Sequence[float]) -> dict[str, Any]:
        self._check_cancelled()
        return self.env.move_delta(
            coerce_arm(arm), coerce_vec3(delta_xyz, name="delta_xyz")
        )

    def rotate_delta(self, arm: str, delta_rpy: Sequence[float]) -> dict[str, Any]:
        self._check_cancelled()
        return self.env.rotate_delta(
            coerce_arm(arm), coerce_vec3(delta_rpy, name="delta_rpy")
        )

    def open_gripper(self, arm: str) -> dict[str, Any]:
        self._check_cancelled()
        return self.env.set_gripper(coerce_arm(arm), open=True)

    def close_gripper(self, arm: str) -> dict[str, Any]:
        self._check_cancelled()
        return self.env.set_gripper(coerce_arm(arm), open=False)


# VLA policy-obs slots mapped to their EnvState stem and raw-frame key. The
# left wrist is ``main_images``; ``extra_view_images`` stacks [base, right wrist].
_VLA_CAMERAS = (
    ("left_wrist", "left_wrist_0_rgb"),
    ("base", "base_0_rgb"),
    ("right_wrist", "right_wrist_0_rgb"),
)


def _policy_views(main: Any, extra: Any, *, channels: bool) -> list[Any]:
    """Return policy views ordered ``[left_wrist, base, right_wrist]``.

    ``extra`` stacks the non-main views as ``[N,H,W,3]`` (RGB) or ``[N,H,W]``
    (depth), optionally behind a leading singleton batch dim stripped here.
    """
    views: list[Any] = [None if main is None else np.asarray(main)]
    if extra is None:
        return views
    array = np.asarray(extra)
    view_ndim = 4 if channels else 3
    if array.ndim == view_ndim + 1 and array.shape[0] == 1:
        array = array[0]
    if array.ndim == view_ndim:
        views.extend(array[index] for index in range(array.shape[0]))
    return views


def dump_state(
    primitives: DualFrankaPrimitives,
    state: EnvState,
    *,
    command: dict[str, Any] | None,
    result: dict[str, Any] | None,
    elapsed_s: float | None,
) -> StepRecord:
    """Capture per-arm robot state and one canonical frame per camera.

    Each VLA camera stores its uncropped raw RGB-D as the single version; the
    policy-resolution view is derived on demand from it with RLinf's crop
    helpers (crop parameters live in ``camera_meta.json``). The policy view is
    persisted only as a fallback when a camera's raw frame is unavailable.
    """
    observation = primitives.env.get_observation()
    robot_state = primitives.env.get_robot_state()
    metadata = primitives.env.get_camera_meta()
    raw_frames = observation.get("raw_camera_frames") or {}
    raw_depths = observation.get("raw_camera_depths") or {}
    policy_frames = _policy_views(
        observation.get("main_images"),
        observation.get("extra_view_images"),
        channels=True,
    )
    policy_depths = _policy_views(
        observation.get("main_depths"),
        observation.get("extra_view_depths"),
        channels=False,
    )
    with state.record_step(
        state=robot_state,
        command=command,
        result=result,
        elapsed_s=elapsed_s,
    ) as step:
        for index, (stem, raw_key) in enumerate(_VLA_CAMERAS):
            frame = raw_frames.get(raw_key)
            if frame is None and index < len(policy_frames):
                frame = policy_frames[index]
            if frame is not None:
                state.save(f"{stem}.png", np.asarray(frame), step=step)
            depth = raw_depths.get(raw_key)
            if depth is None and index < len(policy_depths):
                depth = policy_depths[index]
            if depth is not None:
                state.save(f"{stem}_depth.npy", np.asarray(depth), step=step)
        d455_image = observation.get("d455_images")
        if d455_image is not None:
            state.save("d455.png", np.asarray(d455_image), step=step)
        d455_depth = observation.get("d455_depths")
        if d455_depth is not None:
            state.save("d455_depth.npy", np.asarray(d455_depth), step=step)
        if metadata is not None:
            state.save("camera_meta.json", metadata, step=step)
    return state.get(step)


@readonly
def view_env_state(step: int = -1, *, state: EnvState) -> dict[str, Any]:
    """Return one recorded dual-Franka state with image blocks for the planner."""
    record = state.get(step)
    output = record.to_blob()
    output["images"] = []
    for artifact, path_key, bytes_key in (
        ("left_wrist.png", "image_left_wrist_path", "_image_bytes"),
        ("base.png", "image_base_path", "_image_cam_bytes"),
        ("right_wrist.png", "image_right_wrist_path", "_image_wrist_bytes"),
    ):
        if state.exists(artifact, step=record.step_idx):
            output[path_key] = str(state.artifact_path(artifact, step=record.step_idx))
            output[bytes_key] = state.load_bytes(artifact, step=record.step_idx)
            output["images"].append(artifact.removesuffix(".png"))
    return output
