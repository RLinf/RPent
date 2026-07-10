"""Franka tool implementation for the agent-side toolkit."""
from __future__ import annotations

import json
import os
import time
from typing import Any

import imageio.v2 as imageio
import numpy as np

from robots.franka.env_client import FrankaEnvClient
from rpent.utils.logging import get_logger, get_output_dir

logger = get_logger("franka")

_MAX_DELTA_M = 0.05
_MAX_YAW_DELTA_DEG = 30.0
_MAX_OBSERVE_DELAY_S = 5.0
_BACKPROJECT_RADIUS = 6
_DEPTH_BAND_M = 0.02


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


def _states_path(output_dir: str) -> str:
    return os.path.join(output_dir, "states.json")


def _append_state(output_dir: str, blob: dict) -> None:
    path = _states_path(output_dir)
    states: list = []
    if os.path.exists(path):
        with open(path) as f:
            states = json.load(f)
    states.append(blob)
    with open(path, "w") as f:
        json.dump(states, f, indent=2, default=str)


def _load_states() -> list:
    path = _states_path(str(get_output_dir()))
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return json.load(f)


def _latest_step() -> int | None:
    states = _load_states()
    if not states:
        return None
    return int(states[-1]["step_idx"])


def _load_step(step_idx: int) -> dict:
    for state in _load_states():
        if int(state["step_idx"]) == step_idx:
            return state
    raise KeyError(f"step {step_idx} not present in states.json")


def _load_image(step_idx: int, camera: str) -> bytes | None:
    path = os.path.join(str(get_output_dir()), "images", f"{camera}_{step_idx:02d}.png")
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        return f.read()


def _load_depth(step_idx: int, camera: str) -> np.ndarray:
    path = os.path.join(str(get_output_dir()), "depths", f"{camera}_{step_idx:02d}.npy")
    return np.load(path)


def _backproject_points(K, rows, cols, depths) -> np.ndarray:
    K = np.asarray(K, dtype=np.float64)
    rows = np.asarray(rows, dtype=np.float64)
    cols = np.asarray(cols, dtype=np.float64)
    depths = np.asarray(depths, dtype=np.float64)
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    return np.stack(
        [(cols - cx) * depths / fx, (rows - cy) * depths / fy, depths], axis=1
    )


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
    output_dir: str,
    step_idx: int,
    log: dict | None = None,
) -> dict:
    """Dump current camera frames and proprioceptive state to the run dir."""
    images_dir = os.path.join(output_dir, "images")
    depths_dir = os.path.join(output_dir, "depths")
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(depths_dir, exist_ok=True)

    saved: dict[str, str] = {}
    for camera, frame in driver.latest_frames().items():
        arr = np.asarray(frame)
        if arr.dtype != np.uint8:
            arr = arr.astype(np.uint8)
        out_path = os.path.join(images_dir, f"{camera}_{step_idx:02d}.png")
        try:
            imageio.imwrite(out_path, arr)
            saved[camera] = out_path
        except Exception as exc:
            logger.warning("frame dump failed for camera %s: %s", camera, exc)

    saved_depths: dict[str, str] = {}
    for camera, depth in driver.latest_depths().items():
        out_path = os.path.join(depths_dir, f"{camera}_{step_idx:02d}.npy")
        try:
            np.save(out_path, np.asarray(depth, dtype=np.float32))
            saved_depths[camera] = out_path
        except Exception as exc:
            logger.warning("depth dump failed for camera %s: %s", camera, exc)

    blob: dict[str, Any] = {
        "step_idx": step_idx,
        "state": driver.get_state(),
        "frames": sorted(saved),
        "depth": sorted(saved_depths),
        "camera_meta": driver.latest_camera_meta(),
    }
    if log is not None:
        blob["command"] = log.get("command")
        blob["result"] = log.get("result")
        blob["elapsed_s"] = log.get("elapsed_s")
    _append_state(output_dir, blob)
    return blob


def view_driver_state(step: int | None = None) -> dict:
    """Read a dumped step and embed the scene/wrist camera images."""
    latest = _latest_step()
    if latest is None:
        return {"error": "no driver state entries; driver not ready"}
    step_idx = latest if step is None else int(step)
    try:
        data = _load_step(step_idx)
    except Exception as exc:
        return {"error": f"step {step_idx} not present in driver state trace: {exc}"}

    out: dict[str, Any] = {
        "step": step_idx,
        "state": data.get("state", {}),
        "frames": data.get("frames", []),
        "depth": data.get("depth", []),
        "camera_meta": {
            name: {
                key: value
                for key, value in meta.items()
                if key not in {"K", "T_base_cam"}
            }
            for name, meta in (data.get("camera_meta") or {}).items()
        },
        "log": {
            "command": data.get("command"),
            "result": data.get("result"),
            "elapsed_s": data.get("elapsed_s"),
        },
    }
    scene = _load_image(step_idx, "scene")
    wrist = _load_image(step_idx, "wrist")
    if scene:
        out["_image_bytes"] = scene
    if wrist:
        out["_image_cam_bytes"] = wrist
    return out


def back_project(
    row: int,
    col: int,
    step: int | None = None,
    camera: str = "wrist",
    radius: int = _BACKPROJECT_RADIUS,
) -> dict:
    """Backproject a saved RGB-D pixel into camera and robot-base coordinates."""
    step_idx = _latest_step() if step is None else int(step)
    if step_idx is None:
        return {"error": "no steps available"}
    try:
        data = _load_step(step_idx)
    except Exception as exc:
        return {"error": f"step {step_idx} not present in driver state trace: {exc}"}

    camera = str(camera or "wrist")
    meta = (data.get("camera_meta") or {}).get(camera)
    if not meta:
        return {
            "error": f"camera {camera!r} has no metadata at step {step_idx}",
            "available_cameras": sorted((data.get("camera_meta") or {}).keys()),
        }
    try:
        depth = _load_depth(step_idx, camera)
    except Exception as exc:
        return {"error": f"depth for camera {camera!r} step {step_idx} not found: {exc}"}

    row, col = int(row), int(col)
    radius = max(0, int(_BACKPROJECT_RADIUS if radius is None else radius))
    h, w = depth.shape[:2]
    if not (0 <= row < h and 0 <= col < w):
        return {"error": f"pixel ({row},{col}) out of bounds for {camera} image {h}x{w}"}

    r0, r1 = max(0, row - radius), min(h, row + radius + 1)
    c0, c1 = max(0, col - radius), min(w, col + radius + 1)
    rr, cc = np.mgrid[r0:r1, c0:c1]
    zz = depth[r0:r1, c0:c1].reshape(-1).astype(np.float64)
    rr = rr.reshape(-1).astype(np.float64)
    cc = cc.reshape(-1).astype(np.float64)
    valid = np.isfinite(zz) & (zz > 0)
    if not np.any(valid):
        return {"error": f"no valid depth near ({row},{col}) in {camera}; pick another pixel"}
    zz, rr, cc = zz[valid], rr[valid], cc[valid]
    z_med = float(np.median(zz))
    surface = np.abs(zz - z_med) <= _DEPTH_BAND_M
    zz, rr, cc = zz[surface], rr[surface], cc[surface]
    if zz.size == 0:
        return {"error": f"no dominant surface depth near ({row},{col}) in {camera}"}

    pts_cam = _backproject_points(meta["K"], rr, cc, zz)
    p_cam = np.median(pts_cam, axis=0)
    out: dict[str, Any] = {
        "step": step_idx,
        "camera": camera,
        "pixel": [row, col],
        "radius": radius,
        "n_points": int(pts_cam.shape[0]),
        "depth_m": round(z_med, 4),
        "xyz_cam": [round(float(v), 4) for v in p_cam],
        "camera_frame": meta.get("frame", f"{camera}_camera"),
        "calibrated": bool(meta.get("calibrated")),
        "calibration_kind": meta.get("calibration_kind"),
    }

    T_base_cam = meta.get("T_base_cam")
    if T_base_cam is not None:
        T = np.asarray(T_base_cam, dtype=np.float64)
        pts_base = pts_cam @ T[:3, :3].T + T[:3, 3]
        p_base = np.median(pts_base, axis=0)
        out["xyz"] = [round(float(v), 4) for v in p_base]
        out["frame"] = "panda_link0"
        out["xy_spread_m"] = round(
            float(np.hypot(*pts_base[:, :2].std(axis=0))), 4
        )
    else:
        out["frame"] = meta.get("frame", f"{camera}_camera")
        out["xy_spread_m"] = round(float(np.hypot(*pts_cam[:, :2].std(axis=0))), 4)
        out["note"] = (
            f"camera {camera!r} is not calibrated to panda_link0 for this step; "
            "do not use xyz_cam as a robot target. Add T_base_cam for a fixed "
            "camera or T_tcp_cam for a wrist camera."
        )
    return out


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
