"""Franka tool implementation for the agent-side toolkit."""
from __future__ import annotations

import time
from typing import Any

import numpy as np

from robots.franka.env_client import FrankaEnvClient
from rpent.tools.common import robust_surface_centroid
from rpent.tools.state import EnvState, StepRecord
from rpent.utils.logging import get_logger

logger = get_logger("franka")

_MAX_DELTA_M = 0.05
_MAX_YAW_DELTA_DEG = 30.0
_MAX_OBSERVE_DELAY_S = 5.0
_BACKPROJECT_RADIUS = 6


def _to_list(value) -> list:
    """Coerce numpy arrays / scalars into a compact JSON-friendly list."""
    if value is None:
        return []
    arr = np.asarray(value, dtype=np.float64).reshape(-1)
    return [round(float(v), 4) for v in arr]


def _to_scalar(value) -> Any:
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    return value


# State-trace I/O, image/depth layout, and the robust back-projection math now
# live in :class:`rpent.tools.state.EnvState` (one owner per run). The thin
# wrappers below delegate to it.


class FrankaPrimitives:
    """Primitive driver owned by :class:`FrankaToolkit`."""

    def __init__(self, env: FrankaEnvClient):
        self.env = env
        self._last_obs: dict | None = None
        self._spec: dict | None = None
        self._num_steps = 0

    def reset(self) -> tuple[dict, Any]:
        """Reset the arm and cache the initial observation."""
        self._spec = self.env.get_spec()
        obs, info = self.env.reset()
        self._last_obs = obs
        self._num_steps = 0
        return obs, info

    def observe(self, delay_s: float = 0.0) -> dict:
        """Refresh the cached observation without moving."""
        delay = float(np.clip(delay_s, 0.0, _MAX_OBSERVE_DELAY_S))
        if delay > 0:
            time.sleep(delay)
        self._last_obs = self.env.get_obs()
        return {"delay_s": delay}

    def get_robot_spec(self) -> dict:
        """Return the driver self-description."""
        if self._spec is None:
            self._spec = self.env.get_spec()
        return self._spec

    def get_ee_pose(self) -> dict:
        """Return the live TCP pose in the Franka base frame."""
        return self.env.get_ee_pose()

    def get_camera_meta(self) -> dict:
        """Return live camera intrinsics/extrinsics metadata."""
        return self.env.get_camera_meta()

    def move_to(
        self,
        xyz,
        *,
        yaw_deg: float | None = None,
        gripper: str | None = None,
    ) -> dict:
        """Move to an absolute base-frame Cartesian target."""
        result = self.env.move_to(xyz, yaw_deg=yaw_deg, gripper=gripper)
        self._refresh()
        return result

    def move_delta(
        self,
        dxyz,
        *,
        gripper: str | None = None,
    ) -> dict:
        """Nudge the TCP by a bounded relative translation."""
        requested = np.asarray(dxyz, dtype=np.float64).reshape(-1)[:3]
        clipped = np.clip(requested, -_MAX_DELTA_M, _MAX_DELTA_M)
        result = self.env.move_delta(dxyz=clipped, gripper=gripper)
        if np.any(clipped != requested):
            result = dict(result)
            result["requested_dxyz"] = _to_list(requested)
            result["clipped_dxyz"] = _to_list(clipped)
        self._refresh()
        return result

    def rotate_wrist_yaw(self, delta_deg: float) -> dict:
        """Rotate the wrist yaw relatively, capped for safety."""
        requested = float(delta_deg)
        clipped = float(np.clip(requested, -_MAX_YAW_DELTA_DEG, _MAX_YAW_DELTA_DEG))
        result = self.env.move_delta(yaw_delta_deg=clipped)
        if clipped != requested:
            result = dict(result)
            result["requested_delta_deg"] = round(requested, 3)
            result["clipped_delta_deg"] = round(clipped, 3)
        self._refresh()
        return result

    def rotate_gripper(self, delta_deg: float) -> dict:
        """Rotate the gripper jaw heading relatively, capped for safety."""
        return self.rotate_wrist_yaw(delta_deg)

    def open_gripper(self) -> dict:
        result = self.env.open_gripper()
        self._refresh()
        return result

    def close_gripper(self) -> dict:
        result = self.env.close_gripper()
        self._refresh()
        return result

    def get_state(self) -> dict:
        """Return compact proprioception from the latest observation."""
        obs = self._last_obs or {}
        state = obs.get("state", {}) if isinstance(obs, dict) else {}
        out = {
            "tcp_xyz": _to_list(state.get("tcp_xyz")),
            "tcp_quat": _to_list(state.get("tcp_quat")),
            "tcp_euler": _to_list(state.get("tcp_euler")),
            "gripper_width": round(float(_to_scalar(state.get("gripper_width", 0.0))), 4),
            "gripper_open": bool(_to_scalar(state.get("gripper_open", False))),
            "num_steps": self._num_steps,
        }
        if self._spec is not None:
            out["workspace_min"] = self._spec.get("workspace_min")
            out["workspace_max"] = self._spec.get("workspace_max")
            out["frame"] = self._spec.get("world_frame")
        return out

    def latest_frames(self) -> dict:
        """Return the camera frames from the latest observation."""
        if self._last_obs is None:
            return {}
        return dict(self._last_obs.get("frames", {}))

    def latest_depths(self) -> dict:
        """Return metric depth maps from the latest observation."""
        if self._last_obs is None:
            return {}
        return dict(self._last_obs.get("depth", {}))

    def latest_camera_meta(self) -> dict:
        """Return camera metadata from the latest observation."""
        if self._last_obs is None:
            return {}
        return dict(self._last_obs.get("camera_meta", {}))

    def _refresh(self) -> None:
        try:
            self._last_obs = self.env.get_obs()
            self._num_steps += 1
        except Exception as exc:
            logger.warning("obs refresh failed: %s", exc)


def dump_state(
    driver: FrankaPrimitives,
    state: EnvState,
    step_idx: int,
    log: dict | None = None,
) -> StepRecord:
    """Dump camera frames + proprioceptive state via the EnvState owner.

    Maps the Franka cameras onto the LIBERO stream layout: scene -> ``image``/
    ``depth`` (the primary view) and wrist -> ``image_wrist``/``depth_wrist``.
    """
    image_streams = {"scene": "image", "wrist": "image_wrist"}
    depth_streams = {"scene": "depth", "wrist": "depth_wrist"}

    saved_frames: list[str] = []
    for camera, frame in driver.latest_frames().items():
        stream = image_streams.get(camera, camera)
        if state.save_image(step_idx, stream, frame):
            saved_frames.append(stream)

    saved_depths: list[str] = []
    for camera, depth in driver.latest_depths().items():
        stream = depth_streams.get(camera, camera)
        if state.save_depth(step_idx, stream, depth):
            saved_depths.append(stream)

    record = StepRecord(
        step_idx=step_idx,
        state=driver.get_state(),
        frames=sorted(saved_frames),
        depth=sorted(saved_depths),
        camera_meta=driver.latest_camera_meta(),
        command=log.get("command") if log else None,
        result=log.get("result") if log else None,
        elapsed_s=log.get("elapsed_s") if log else None,
    )
    return state.append(record)


# view_driver_state now lives on EnvState.view (bound by the toolkit with the
# scene/wrist image slots); nothing module-level is needed here.


def back_project(
    row: int,
    col: int,
    step: int | None = None,
    camera: str = "wrist",
    radius: int | None = _BACKPROJECT_RADIUS,
    *,
    state: EnvState,
) -> dict:
    """Back-project a saved RGB-D pixel into camera and robot-base coordinates.

    Uses :func:`rpent.tools.common.robust_surface_centroid` (median of the
    dominant surface in a window around the pixel) so oblique-view depth noise
    does not become ~cm lateral error. Returns robot-base ``xyz`` in
    ``panda_link0`` when the camera has ``T_base_cam`` for the step; otherwise
    ``xyz_cam`` plus a warning.
    """
    nn = state.latest_step if step is None else int(step)
    if nn is None:
        return {"error": "no steps available"}
    try:
        rec = state.get(nn)
    except Exception as exc:
        return {"error": f"step {nn} not present in driver state trace: {exc}"}

    camera = str(camera or "wrist")
    meta = (rec.camera_meta or {}).get(camera)
    if not meta:
        return {
            "error": f"camera {camera!r} has no metadata at step {nn}",
            "available_cameras": sorted((rec.camera_meta or {}).keys()),
        }
    depth_stream = "depth" if camera == "scene" else "depth_wrist"
    try:
        depth = state.load_depth(nn, depth_stream)
    except Exception as exc:
        return {"error": f"depth for camera {camera!r} step {nn} not found: {exc}"}

    res = robust_surface_centroid(
        depth,
        meta["K"],
        meta.get("T_base_cam"),
        row,
        col,
        radius=_BACKPROJECT_RADIUS if radius is None else radius,
    )
    if "error" in res:
        return res

    res["step"] = nn
    res["camera"] = camera
    res["camera_frame"] = meta.get("frame", f"{camera}_camera")
    res["calibrated"] = bool(meta.get("calibrated"))
    res["calibration_kind"] = meta.get("calibration_kind")
    if meta.get("T_base_cam") is not None:
        res["frame"] = "panda_link0"
    else:
        res["frame"] = meta.get("frame", f"{camera}_camera")
        res["note"] = (
            f"camera {camera!r} is not calibrated to panda_link0 for this step; "
            "do not use xyz_cam as a robot target. Add T_base_cam for a fixed "
            "camera or T_tcp_cam for a wrist camera."
        )
    return res


TOOLS_SPEC: list[dict[str, Any]] = [
    {
        "name": "view_driver_state",
        "description": (
            "Read step NN from states.json plus matching camera PNGs. If step "
            "is null, returns the latest entry. Embeds scene and wrist images."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "step": {
                    "type": ["integer", "null"],
                    "description": "Step number; 0 = initial. Null = latest.",
                }
            },
        },
    },
    {
        "name": "back_project",
        "description": (
            "Backproject a pixel from a saved RGB-D camera image to a 3D point. "
            "Defaults to camera='wrist', the preferred camera for close-range "
            "Franka manipulation after wrist calibration. Returns robot-base "
            "`xyz` in panda_link0 only when that camera has calibration for "
            "the selected step; otherwise returns "
            "xyz_cam plus a warning. Pick row/col on the image returned by "
            "view_driver_state."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "row": {"type": "integer", "description": "Pixel row (y)."},
                "col": {"type": "integer", "description": "Pixel column (x)."},
                "step": {
                    "type": ["integer", "null"],
                    "description": "Step whose saved depth to use; null = latest.",
                },
                "camera": {
                    "type": "string",
                    "enum": ["scene", "wrist"],
                    "description": "RGB-D camera to use. Default and preferred after calibration: wrist.",
                },
                "radius": {
                    "type": ["integer", "null"],
                    "description": "Half-size of depth window in pixels; null = default 6.",
                },
            },
            "required": ["row", "col"],
        },
    },
    {
        "name": "observe",
        "description": (
            "Refresh the live observation without moving the arm, dump a new "
            "step, and return the updated state/images. Use this when the scene "
            "may have changed or after waiting for settling."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "delay_s": {
                    "type": ["number", "null"],
                    "description": "Optional wait before observing; clipped to 0..5 seconds.",
                }
            },
        },
    },
    {
        "name": "get_ee_pose",
        "description": (
            "Live TCP pose in the Franka base frame panda_link0. Returns xyz "
            "meters, quat_xyzw, euler_xyz radians, and frame."
        ),
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "get_robot_spec",
        "description": (
            "Static robot/environment description: workspace bounds, frame, "
            "camera names, control mode, gripper mode, and reset pose."
        ),
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "get_camera_meta",
        "description": (
            "Live per-camera intrinsics, depth scale, and calibration status. "
            "For base-frame back_project, wrist needs T_tcp_cam composed with "
            "the live TCP pose; scene needs T_base_cam."
        ),
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "move_to",
        "description": (
            "Move the TCP to absolute [x, y, z] in panda_link0, meters. The "
            "driver clips targets to the safe workspace and returns reached, "
            "pos_error_m, final_xyz, and clipping info. yaw_deg optionally sets "
            "a down-facing grasp yaw; null keeps current orientation. gripper "
            "may be null, 'open', or 'close'."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "xyz": {
                    "type": "array",
                    "description": "Absolute target [x, y, z] in meters, panda_link0 frame.",
                    "items": {"type": "number"},
                    "minItems": 3,
                    "maxItems": 3,
                },
                "yaw_deg": {
                    "type": ["number", "null"],
                    "description": "Optional down-facing wrist yaw in degrees; null keeps current orientation.",
                },
                "gripper": {
                    "type": ["string", "null"],
                    "enum": ["open", "close", None],
                    "description": "Optional gripper action to execute before the move.",
                },
            },
            "required": ["xyz"],
        },
    },
    {
        "name": "move_delta",
        "description": (
            "Nudge the TCP by relative [dx, dy, dz] meters in panda_link0. Each "
            "axis is clipped to +/-0.05 m per call. Use for visual servoing and "
            "small approach/lift motions. gripper may be null, 'open', or 'close'."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "dxyz": {
                    "type": "array",
                    "description": "Relative TCP translation [dx, dy, dz] in meters.",
                    "items": {"type": "number", "minimum": -0.05, "maximum": 0.05},
                    "minItems": 3,
                    "maxItems": 3,
                },
                "gripper": {
                    "type": ["string", "null"],
                    "enum": ["open", "close", None],
                    "description": "Optional gripper action to execute before the nudge.",
                },
            },
            "required": ["dxyz"],
        },
    },
    {
        "name": "rotate_wrist_yaw",
        "description": (
            "Rotate the wrist yaw relatively without translating the TCP. The "
            "requested delta is clipped to +/-30 degrees per call. Use only for "
            "jaw alignment after the arm is already near the target."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "delta_deg": {
                    "type": "number",
                    "minimum": -30,
                    "maximum": 30,
                    "description": "Relative yaw change in degrees.",
                }
            },
            "required": ["delta_deg"],
        },
    },
    {
        "name": "rotate_gripper",
        "description": (
            "Rotate the gripper jaw heading relatively without translating the "
            "TCP. The requested delta is clipped to +/-30 degrees per call. "
            "Use when the fingers need a different angle before grasping."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "delta_deg": {
                    "type": "number",
                    "minimum": -30,
                    "maximum": 30,
                    "description": "Relative gripper yaw change in degrees.",
                }
            },
            "required": ["delta_deg"],
        },
    },
    {
        "name": "open_gripper",
        "description": "Open the Franka Hand, refresh observation, and return updated state/images.",
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "close_gripper",
        "description": (
            "Close/grasp with the Franka Hand, refresh observation, and return "
            "updated state/images."
        ),
        "input_schema": {"type": "object", "properties": {}},
    },
]
