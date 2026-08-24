"""Dual-Franka planner tools, primitives, and canonical state capture."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from robots.franka.tools import FrankaPrimitives, coerce_vec3
from rpent.tools.state import EnvState, StepRecord
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

def dump_state(
    primitives: DualFrankaPrimitives,
    state: EnvState,
    *,
    command: dict[str, Any] | None,
    result: dict[str, Any] | None,
    elapsed_s: float | None,
) -> StepRecord:
    """Capture per-arm robot state and the three synchronized camera images."""
    observation = primitives.env.get_observation()
    robot_state = primitives.env.get_robot_state()
    metadata = primitives.env.get_camera_metadata()
    with state.record_step(
        state=robot_state,
        command=command,
        result=result,
        elapsed_s=elapsed_s,
    ) as step:
        raw_frames = observation.get("raw_camera_frames") or {}
        raw_depths = observation.get("raw_camera_depths") or {}
        main_image = observation.get("main_images")
        if main_image is not None:
            state.save("left_wrist.png", np.asarray(main_image), step=step)
        extra_images = observation.get("extra_view_images")
        if extra_images is not None:
            extra_array = np.asarray(extra_images)
            if extra_array.ndim == 5:
                extra_array = extra_array[0]
            if extra_array.ndim == 4:
                if extra_array.shape[0] >= 1:
                    state.save(
                        "base.png",
                        np.asarray(raw_frames.get("base_0_rgb", extra_array[0])),
                        step=step,
                    )
                if extra_array.shape[0] >= 2:
                    state.save("right_wrist.png", extra_array[1], step=step)
        main_depth = observation.get("main_depths")
        if main_depth is not None:
            state.save("left_wrist_depth.npy", np.asarray(main_depth), step=step)
        extra_depths = observation.get("extra_view_depths")
        base_raw_depth = raw_depths.get("base_0_rgb")
        if base_raw_depth is not None:
            state.save("base_depth.npy", np.asarray(base_raw_depth), step=step)
        if extra_depths is not None:
            depth_array = np.asarray(extra_depths)
            if depth_array.ndim == 4 and depth_array.shape[0] == 1:
                depth_array = depth_array[0]
            if depth_array.ndim == 3:
                if depth_array.shape[0] >= 1 and base_raw_depth is None:
                    state.save("base_depth.npy", depth_array[0], step=step)
                if depth_array.shape[0] >= 2:
                    state.save("right_wrist_depth.npy", depth_array[1], step=step)
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
    for artifact, path_key, bytes_key in (
        ("left_wrist.png", "image_left_wrist_path", "_image_left_wrist_bytes"),
        ("base.png", "image_base_path", "_image_base_bytes"),
        ("right_wrist.png", "image_right_wrist_path", "_image_right_wrist_bytes"),
    ):
        if state.exists(artifact, step=record.step_idx):
            output[path_key] = str(state.artifact_path(artifact, step=record.step_idx))
            output[bytes_key] = state.load_bytes(artifact, step=record.step_idx)
    return output
