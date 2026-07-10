"""LeRobot SO101 tool implementation.

Structure mirrors :mod:`robots.libero.tools`:

* :class:`LerobotPrimitives` — the primitive driver the toolkit owns. Holds
  the env client (and an optional policy/VLA model) plus per-run state, and
  exposes one method per primitive tool.
* per-step state dump (:func:`dump_state`) + stateless reader tools
  (:func:`view_driver_state`).
* :data:`TOOLS_SPEC` — Anthropic-shaped tool schemas.

NOTE: this is a scaffold. The concrete robot primitives (move / grasp /
release / ...) and their schemas are intentionally left as TODOs — only the
loop infrastructure (state dump + view) is implemented so the env
loads and the pattern is in place.
"""
from __future__ import annotations

import json
import os
from typing import Any

import imageio.v2 as imageio
import numpy as np

from robots.lerobot.env_client import LerobotEnvClient
from rpent.utils.logging import get_logger, get_output_dir

logger = get_logger("lerobot")

# back_project robust-centroid params: it back-projects every valid pixel in a
# (2*radius+1) square window and keeps those whose depth is within
# _DEPTH_BAND_M of the window median (the dominant / object surface), then
# takes the world-median. This tames the scene camera's oblique-view depth
# noise, which otherwise turns per-pixel depth error into ~cm lateral error.
_BACKPROJECT_RADIUS = 6
_DEPTH_BAND_M = 0.02


def _to_list(x) -> list:
    """Coerce a numpy array / sequence / scalar into a plain list[float]."""
    if x is None:
        return []
    arr = np.asarray(x, dtype=np.float32).reshape(-1)
    return [round(float(v), 4) for v in arr]


# ---------------------------------------------------------------------------
# Primitive driver
# ---------------------------------------------------------------------------


class LerobotPrimitives:
    """Wraps a single SO101 env client (+ optional policy) with primitive-
    level methods.

    The toolkit constructs this from the env RPC client and calls
    :meth:`reset` once at start-up; :func:`dump_state` reads back the latest
    observation via :meth:`get_state` / :meth:`latest_frames` after each tool.

    TODO: add the concrete robot primitives (e.g. ``move_to``, ``grasp``,
    ``release``, ``home``) on top of the low-level :meth:`step` passthrough.
    Each should return a ``dict`` log and leave ``self._last_obs`` current.
    """

    def __init__(self, env: LerobotEnvClient, model: Any | None = None):
        self.env = env
        self.model = model  # optional policy/VLA client; None for scripted prims
        self._last_obs: dict | None = None
        self._spec: dict | None = None
        self._scene_meta: dict | None = None
        self._num_steps = 0

    # -- lifecycle ---------------------------------------------------------

    def reset(self) -> tuple[dict, Any]:
        """Reset the env (arm → rest) and cache the first observation."""
        self._spec = self.env.get_spec()
        obs, info = self.env.reset()
        self._last_obs = obs
        self._num_steps = 0
        return obs, info

    def step(self, action) -> dict:
        """Low-level passthrough: one ``env.step``. Higher-level primitives
        are layered on top of this (TODO).
        """
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._last_obs = obs
        self._num_steps += 1
        return {
            "reward": float(reward),
            "terminated": bool(terminated),
            "truncated": bool(truncated),
            "num_steps": self._num_steps,
        }

    # -- state accessors used by dump_state --------------------------------

    def get_state(self) -> dict:
        """Return the robot proprioceptive state from the last observation."""
        if self._last_obs is None:
            return {}
        st = self._last_obs.get("state", {})
        out = {
            "joint_position": _to_list(st.get("joint_position")),
            "gripper_position": _to_list(st.get("gripper_position")),
            "num_steps": self._num_steps,
        }
        if st.get("ee_pose_base") is not None:
            out["ee_pose_base"] = _to_list(st.get("ee_pose_base"))
        if st.get("ee_quat_base") is not None:
            out["ee_quat_base"] = _to_list(st.get("ee_quat_base"))
        return out

    def latest_frames(self) -> dict:
        """Return the camera frames dict from the last observation."""
        if self._last_obs is None:
            return {}
        return dict(self._last_obs.get("frames", {}))

    def latest_depth(self) -> np.ndarray | None:
        """Return the scene depth map (meters) from the last observation."""
        if self._last_obs is None:
            return None
        depth = self._last_obs.get("depth", {})
        scene = depth.get("scene") if isinstance(depth, dict) else None
        return None if scene is None else np.asarray(scene, dtype=np.float32)

    # -- localization (base/world frame) -----------------------------------

    def get_ee_pose(self) -> dict:
        """Live FK: gripper pose in the base (world) frame."""
        return self.env.get_ee_pose()

    def get_scene_camera_meta(self) -> dict:
        """Scene-camera intrinsics + depth scale + base extrinsic (cached)."""
        if self._scene_meta is None:
            self._scene_meta = self.env.get_scene_camera_meta()
        return self._scene_meta

    # -- primitives (move the robot, then refresh the cached observation) ---

    def move_to(
        self,
        xyz,
        gripper: float | None = None,
        approach: str = "free",
        yaw_deg: float | None = None,
    ) -> dict:
        """Move the gripper to a world-frame (base_link) XYZ via driver IK.

        ``approach="down"`` keeps the gripper pointing straight down (for
        grasping); ``yaw_deg`` sets the jaw heading. See the driver for details.
        """
        result = self.env.move_to(
            xyz, gripper=gripper, approach=approach, yaw_deg=yaw_deg
        )
        self._refresh()
        return result

    def move_joints_delta(
        self,
        delta_deg,
        gripper_delta: float | None = None,
    ) -> dict:
        """Nudge each arm joint relatively (degrees) for fine alignment."""
        result = self.env.move_joints_delta(delta_deg, gripper_delta=gripper_delta)
        self._refresh()
        return result

    def _refresh(self) -> None:
        """Refresh the cached observation after a motion primitive."""
        try:
            self._last_obs = self.env.get_obs()
        except Exception as e:
            logger.warning("obs refresh failed: %s", e)


# ---------------------------------------------------------------------------
# states.json + image helpers
# ---------------------------------------------------------------------------


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


def _load_step(nn: int) -> dict:
    for s in _load_states():
        if int(s["step_idx"]) == nn:
            return s
    raise KeyError(f"step {nn} not present in states.json")


def _load_image(nn: int, cam: str) -> bytes | None:
    path = os.path.join(str(get_output_dir()), "images", f"{cam}_{nn:02d}.png")
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        return f.read()


def _load_depth(nn: int) -> np.ndarray:
    path = os.path.join(str(get_output_dir()), "depths", f"scene_{nn:02d}.npy")
    return np.load(path)


def _load_camera_meta() -> dict:
    path = os.path.join(str(get_output_dir()), "camera_meta.json")
    with open(path) as f:
        return json.load(f)


def dump_state(
    driver: LerobotPrimitives,
    output_dir: str,
    step_idx: int,
    log: dict | None = None,
) -> dict:
    """Dump the camera frames, scene depth, and proprioceptive state.

    Writes per step:
      - ``<output_dir>/images/<cam>_NN.png``   (arm + scene color)
      - ``<output_dir>/depths/scene_NN.npy``   (metric depth, aligned to color)
    and once:
      - ``<output_dir>/camera_meta.json``      (scene K, depth scale, T_base_cam)
    then appends the step blob (state incl. ``ee_pose_base`` + optional command
    log) to ``<output_dir>/states.json``.
    """
    images_dir = os.path.join(output_dir, "images")
    depths_dir = os.path.join(output_dir, "depths")
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(depths_dir, exist_ok=True)

    saved: dict[str, str] = {}
    for cam, frame in driver.latest_frames().items():
        arr = np.asarray(frame)
        if arr.dtype != np.uint8:
            arr = arr.astype(np.uint8)
        out_path = os.path.join(images_dir, f"{cam}_{step_idx:02d}.png")
        try:
            imageio.imwrite(out_path, arr)
            saved[cam] = out_path
        except Exception as e:
            logger.warning("frame dump failed for cam %s: %s", cam, e)

    depth = driver.latest_depth()
    if depth is not None:
        try:
            np.save(os.path.join(depths_dir, f"scene_{step_idx:02d}.npy"),
                    depth.astype(np.float32))
        except Exception as e:
            logger.warning("depth dump failed: %s", e)

    # Scene camera calibration is static — fetch + dump once.
    meta_path = os.path.join(output_dir, "camera_meta.json")
    if not os.path.exists(meta_path):
        meta = driver.get_scene_camera_meta()
        if isinstance(meta, dict) and "error" not in meta:
            with open(meta_path, "w") as f:
                json.dump(meta, f, indent=2, default=str)

    blob: dict = {
        "step_idx": step_idx,
        "state": driver.get_state(),
        "frames": sorted(saved),
    }
    if log is not None:
        blob["command"] = log.get("command")
        blob["result"] = log.get("result")
        blob["elapsed_s"] = log.get("elapsed_s")
    _append_state(output_dir, blob)
    return blob


# ---------------------------------------------------------------------------
# Stateless reader tools
# ---------------------------------------------------------------------------


def view_driver_state(step: int | None = None) -> dict:
    """Read step NN from ``states.json`` + the matching camera PNGs.

    Returns the proprioceptive state and embeds the scene/arm camera frames
    as multimodal image content blocks (via the ``_image_bytes`` /
    ``_image_cam_bytes`` conventions consumed by ``ToolResult``).
    """
    latest = _latest_step()
    if latest is None:
        return {"error": "no driver state entries; driver not ready"}
    nn = latest if step is None else int(step)
    try:
        data = _load_step(nn)
    except Exception as e:
        return {"error": f"step {nn} not present in driver state trace: {e}"}

    out: dict = {
        "step": nn,
        "state": data.get("state", {}),
        "log": {
            "command": data.get("command"),
            "result": data.get("result"),
            "elapsed_s": data.get("elapsed_s"),
        },
    }
    # Map the two cameras onto the two image slots ToolResult understands.
    scene = _load_image(nn, "scene")
    arm = _load_image(nn, "arm")
    if scene:
        out["_image_bytes"] = scene
    if arm:
        out["_image_cam_bytes"] = arm
    return out


def back_project(
    row: int,
    col: int,
    step: int | None = None,
    radius: int = _BACKPROJECT_RADIUS,
) -> dict:
    """Backproject a scene-camera pixel neighborhood to a robust world point.

    The scene camera views the table at a steep oblique angle, so a single
    pixel's depth error becomes a large lateral error. Instead of trusting one
    pixel, this back-projects EVERY valid pixel in a ``(2*radius+1)`` square
    window around ``(row, col)``, keeps those on the dominant surface (depth
    within ``_DEPTH_BAND_M`` of the window median -- rejecting background /
    table / dropouts), and returns the MEDIAN world ``xyz`` of that surface: a
    stable object centroid rather than one face pixel. Use ``radius=0`` for the
    old single-pixel behavior.

    Pick ``(row, col)`` on the scene color image ``images/scene_NN.png``; depth
    is aligned to it (``depths/scene_NN.npy``). Uses ``camera_meta.json`` (K +
    ``T_base_cam``). Returns base/world ``xyz`` when calibrated, else the
    camera-frame ``xyz_cam`` with a note. Also reports ``n_points`` (surface
    pixels used) and ``xy_spread_m`` (their world-xy stdev) as a quality gauge.
    """
    try:
        meta = _load_camera_meta()
    except Exception as e:
        return {"error": f"camera_meta.json not found: {e}"}
    nn = _latest_step() if step is None else int(step)
    if nn is None:
        return {"error": "no steps available"}
    try:
        depth = _load_depth(nn)
    except Exception as e:
        return {"error": f"depth for step {nn} not found: {e}"}

    row, col = int(row), int(col)
    radius = max(0, int(radius))
    h, w = depth.shape[:2]
    if not (0 <= row < h and 0 <= col < w):
        return {"error": f"pixel ({row},{col}) out of bounds; image is {h}x{w}"}

    # Gather the window, keep valid depths, then restrict to the dominant
    # surface (depths within a band of the window median) so background / table
    # pixels and dropouts don't drag the centroid.
    r0, r1 = max(0, row - radius), min(h, row + radius + 1)
    c0, c1 = max(0, col - radius), min(w, col + radius + 1)
    rr, cc = np.mgrid[r0:r1, c0:c1]
    zz = depth[r0:r1, c0:c1].reshape(-1).astype(np.float64)
    rr = rr.reshape(-1).astype(np.float64)
    cc = cc.reshape(-1).astype(np.float64)
    valid = np.isfinite(zz) & (zz > 0)
    if not np.any(valid):
        return {"error": f"no valid depth near ({row},{col}); pick another pixel"}
    zz, rr, cc = zz[valid], rr[valid], cc[valid]
    z_med = float(np.median(zz))
    surf = np.abs(zz - z_med) <= _DEPTH_BAND_M
    zz, rr, cc = zz[surf], rr[surf], cc[surf]

    K = np.asarray(meta["K"], dtype=np.float64)
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    pts_cam = np.stack(
        [(cc - cx) * zz / fx, (rr - cy) * zz / fy, zz], axis=1
    )  # (N, 3)
    p_cam = np.median(pts_cam, axis=0)

    out: dict = {
        "pixel": [row, col],
        "radius": radius,
        "n_points": int(pts_cam.shape[0]),
        "depth_m": round(z_med, 4),
        "xyz_cam": [round(float(v), 4) for v in p_cam],
        "frame": "scene_cam",
    }
    T = meta.get("T_base_cam")
    if T is not None:
        T = np.asarray(T, dtype=np.float64)
        pts_base = pts_cam @ T[:3, :3].T + T[:3, 3]
        p_base = np.median(pts_base, axis=0)
        out["xyz"] = [round(float(v), 4) for v in p_base]
        out["xy_spread_m"] = round(float(np.hypot(*pts_base[:, :2].std(axis=0))), 4)
        out["frame"] = "base_link"
    else:
        out["xy_spread_m"] = round(float(np.hypot(*pts_cam[:, :2].std(axis=0))), 4)
        out["note"] = (
            "scene camera not calibrated (no T_base_cam); returning camera-frame "
            "xyz only. Run deployment/lerobot/calibrate_scene_cam.py."
        )
    return out


# ---------------------------------------------------------------------------
# Tool schema declarations (Anthropic-shaped)
# ---------------------------------------------------------------------------

TOOLS_SPEC: list[dict[str, Any]] = [
    {
        "name": "view_driver_state",
        "description": (
            "Read step NN from `states.json` + the matching camera PNGs in "
            "{output_dir}/images. If step is null, returns the latest entry. "
            "Embeds the scene and arm camera frames as image content blocks."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "step": {
                    "type": ["integer", "null"],
                    "description": "Step number; 0 = initial. Null = latest.",
                },
            },
        },
    },
    {
        "name": "get_ee_pose",
        "description": (
            "Live forward kinematics: the gripper tip pose in the WORLD frame "
            "(arm base_link). Returns xyz (meters), quat_wxyz, and joints_deg. "
            "Use this to know where the gripper currently is in world coords."
        ),
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "get_scene_camera_meta",
        "description": (
            "Scene-camera calibration: intrinsics K, depth scale, and whether "
            "the camera->base extrinsic (T_base_cam) is calibrated. If "
            "calibrated is false, back_project returns camera-frame coords only."
        ),
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "back_project",
        "description": (
            "Backproject a SCENE-camera pixel to a 3D point in the WORLD frame "
            "(arm base_link), using the saved aligned depth. Pick (row, col) on "
            "the scene color image from view_driver_state, near the CENTER of "
            "the target. It samples a small window around the pixel and returns "
            "the robust MEDIAN world `xyz` of the object surface (not one noisy "
            "pixel), plus `n_points` and `xy_spread_m` (a small spread means a "
            "confident estimate). Returns world `xyz` when calibrated (else "
            "camera-frame `xyz_cam`). This is the primary tool for locating "
            "objects in the robot's coordinate system."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "row": {"type": "integer", "description": "Pixel row (y) in the scene image, near the target center."},
                "col": {"type": "integer", "description": "Pixel column (x) in the scene image, near the target center."},
                "step": {
                    "type": ["integer", "null"],
                    "description": "Step whose depth to use; null = latest.",
                },
                "radius": {
                    "type": ["integer", "null"],
                    "description": "Half-size (px) of the sampling window; null = default (6). Use a smaller value for tiny/cluttered targets, 0 for a single pixel.",
                },
            },
            "required": ["row", "col"],
        },
    },
    {
        "name": "move_to",
        "description": (
            "Move the gripper to a target [x, y, z] in the WORLD frame (arm "
            "base_link), meters. The target is clipped to a safe workspace box "
            "and approached in small capped steps. `approach` controls the "
            "wrist orientation: 'free' (default) lets IK pick any orientation "
            "(maximal reach, but the fingertips' exact location is "
            "unpredictable -- not for grasping); 'down' keeps the gripper "
            "pointing STRAIGHT DOWN so the fingers descend vertically (use this "
            "to grasp). With approach='down', `yaw_deg` sets the jaw-line "
            "heading about vertical (0=+x/forward, 90=+y/left); leave null to "
            "auto-pick a reachable heading. Optionally set the gripper opening. "
            "Returns `reached`, `pos_error_m`, and `approach_tilt_deg` (0 = "
            "perfectly vertical). Use get_ee_pose / back_project to choose "
            "targets in the same frame."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "xyz": {
                    "type": "array",
                    "description": "World-frame target [x, y, z] in meters (base_link).",
                    "items": {"type": "number"},
                    "minItems": 3,
                    "maxItems": 3,
                },
                "gripper": {
                    "type": ["number", "null"],
                    "description": "Gripper opening degrees (~90 open .. ~15 grasp); null keeps current. Never 0.",
                },
                "approach": {
                    "type": "string",
                    "enum": ["free", "down"],
                    "description": "'down' = gripper points straight down (for grasping); 'free' = any orientation. Default 'free'.",
                },
                "yaw_deg": {
                    "type": ["number", "null"],
                    "description": "With approach='down', jaw-line heading about vertical in degrees (0=forward, 90=left). Null = auto-pick a reachable heading.",
                },
            },
            "required": ["xyz"],
        },
    },
    {
        "name": "move_joints_delta",
        "description": (
            "Fine-adjust the arm by nudging each joint RELATIVELY (degrees). "
            "`delta_deg` is 5 values added to the current joints: [shoulder_pan, "
            "shoulder_lift, elbow_flex, wrist_flex, wrist_roll]. Each is capped "
            "to +/-15 deg/call and clamped to joint limits. Positive wrist_roll "
            "rotates the jaw line; wrist_flex tilts the gripper up/down. Use "
            "this when move_to gets you close but the grasp needs a small tweak "
            "(align the jaws across the object, or descend a few mm). Optionally "
            "nudge the gripper with `gripper_delta`. Returns the new joints and "
            "EE xyz. Prefer move_to for big moves; this is for fine alignment."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "delta_deg": {
                    "type": "array",
                    "description": "Relative joint deltas in degrees [pan, lift, elbow, wrist_flex, wrist_roll].",
                    "items": {"type": "number"},
                    "minItems": 5,
                    "maxItems": 5,
                },
                "gripper_delta": {
                    "type": ["number", "null"],
                    "description": "Relative gripper opening change in degrees; null keeps current.",
                },
            },
            "required": ["delta_deg"],
        },
    },
]
