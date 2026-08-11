"""Dual-Franka planner tools, primitives, and canonical state capture."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import numpy as np

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
        "description": "Read camera names, serials, and types for the dual-Franka rig.",
        "input_schema": {
            "type": "object",
            "properties": {"step": {"type": "integer", "default": -1}},
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


def coerce_vec3(value: Sequence[float], *, name: str) -> np.ndarray:
    """Return a finite float32 three-vector or raise a useful error."""
    array = np.asarray(value, dtype=np.float32)
    if array.shape != (3,):
        raise ValueError(f"{name} must contain exactly 3 values, got {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite values")
    return array


class DualFrankaPrimitives:
    """Safe agent-facing operations over a remote dual-Franka environment."""

    def __init__(
        self,
        *,
        env: Any,
        model: Any | None,
        task_description: str,
        check_cancelled: Callable[[], None],
    ) -> None:
        self.env = env
        self.model = model
        self.task_description = task_description
        self._check_cancelled = check_cancelled

    def reset(self) -> dict[str, Any]:
        return self.env.reset()

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

    def vla_grasp(self, prompt: str, max_chunks: int = 4) -> dict[str, Any]:
        if self.model is None:
            raise RuntimeError("vla_grasp requires --vla-endpoint")
        if not prompt.strip():
            raise ValueError("prompt must be non-empty")
        if not 1 <= int(max_chunks) <= 20:
            raise ValueError("max_chunks must be between 1 and 20")

        chunk_results: list[dict[str, Any]] = []
        for _ in range(int(max_chunks)):
            self._check_cancelled()
            observation = dict(self.env.get_observation())
            observation["task_descriptions"] = prompt or self.task_description
            actions, _ = self.model.predict_action_batch(observation, mode="eval")
            result = self.env.step_chunk(actions)
            chunk_results.append(result)
            if result.get("terminated") or result.get("truncated"):
                break

        return {
            "ok": True,
            "chunks_executed": len(chunk_results),
            "last_chunk": chunk_results[-1] if chunk_results else None,
            "robot_state": self.env.get_robot_state(),
        }


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
                    state.save("base.png", extra_array[0], step=step)
                if extra_array.shape[0] >= 2:
                    state.save("right_wrist.png", extra_array[1], step=step)
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


@readonly
def view_camera_meta(step: int = -1, *, state: EnvState) -> dict[str, Any]:
    """Return camera metadata captured for one state step."""
    if not state.exists("camera_meta.json", step=step):
        return {"error": "camera metadata is unavailable", "step": step}
    return {
        "step": state.get(step).step_idx,
        "camera_meta": state.load("camera_meta.json", step=step),
    }
