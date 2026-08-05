"""LeRobot SO101 tool implementation.

Structure mirrors :mod:`robots.libero.tools`:

* :class:`LerobotPrimitives` — the primitive driver the toolkit owns. Holds
  the env client (and an optional policy/VLA model) plus per-run state, and
  exposes one method per primitive tool.
* per-step state dump (:func:`dump_state`) plus reader tools bound to the
    run's :class:`rpent.tools.state.EnvState`.
* :data:`TOOLS_SPEC` — Anthropic-shaped tool schemas.

NOTE: this is a scaffold. The concrete robot primitives (move / grasp /
release / ...) and their schemas are intentionally left as TODOs — only the
loop infrastructure (state dump + view) is implemented so the env
loads and the pattern is in place.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from robots.lerobot.env_client import LerobotEnvClient
from rpent.tools.common import robust_surface_centroid
from rpent.tools.state import EnvState, StepRecord
from rpent.tools.toolkit import updatestate
from rpent.utils.logging import get_logger

logger = get_logger("lerobot")

_BACKPROJECT_RADIUS = 6


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

    # -- primitives (move the robot; toolkit capture refreshes observation) --

    @updatestate
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
        return self.env.move_to(
            xyz, gripper=gripper, approach=approach, yaw_deg=yaw_deg
        )

    @updatestate
    def move_joints_delta(
        self,
        delta_deg,
        gripper_delta: float | None = None,
    ) -> dict:
        """Nudge each arm joint relatively (degrees) for fine alignment."""
        return self.env.move_joints_delta(
            delta_deg,
            gripper_delta=gripper_delta,
        )

    def _refresh(self) -> None:
        """Refresh the cached observation for toolkit state capture."""
        try:
            self._last_obs = self.env.get_obs()
        except Exception as e:
            logger.warning("obs refresh failed: %s", e)


def dump_state(
    driver: LerobotPrimitives,
    state: EnvState,
    log: dict | None = None,
) -> StepRecord:
    """Dump camera artifacts and proprioceptive state through ``EnvState``."""
    log = log or {}
    with state.record_step(
        state=driver.get_state(),
        command=log.get("command"),
        result=log.get("result"),
        elapsed_s=log.get("elapsed_s"),
    ) as step_idx:
        for camera, frame in driver.latest_frames().items():
            state.save(
                f"{camera}.png",
                frame,
                step=step_idx,
            )

        depth = driver.latest_depth()
        if depth is not None:
            state.save(
                "scene_depth.npy",
                depth,
                step=step_idx,
            )

        scene_meta = driver.get_scene_camera_meta()
        if isinstance(scene_meta, dict) and "error" not in scene_meta:
            state.save(
                "scene_metadata.json",
                scene_meta,
                step=step_idx,
            )

    return state.get(step_idx)


def back_project(
    row: int,
    col: int,
    step: int = -1,
    radius: int | None = _BACKPROJECT_RADIUS,
    *,
    state: EnvState,
) -> dict:
    """Backproject a scene-camera pixel neighborhood to a robust world point.

    The scene camera views the table at a steep oblique angle, so a single
    pixel's depth error becomes a large lateral error. Instead of trusting one
    pixel, this back-projects EVERY valid pixel in a ``(2*radius+1)`` square
    window around ``(row, col)``, keeps those on the dominant surface (depth
    within a narrow band of the window median, rejecting background, table,
    and dropouts), and returns the MEDIAN world ``xyz`` of that surface: a
    stable object centroid rather than one face pixel. Use ``radius=0`` for the
    old single-pixel behavior.

    Pick ``(row, col)`` on the saved ``scene.png`` observation; depth is
    aligned in ``scene_depth.npy``. Returns base/world ``xyz`` when calibrated, else
    camera-frame ``xyz_cam`` with a note.
    """
    try:
        record = state.get(step)
    except Exception as exc:
        return {"error": f"state step not available: {exc}"}
    nn = record.step_idx
    metadata_name = "scene_metadata.json"
    depth_name = "scene_depth.npy"
    if metadata_name not in record.artifacts:
        return {"error": f"scene camera metadata not recorded for step {nn}"}
    try:
        meta = state.load(metadata_name, step=nn)
        if depth_name not in record.artifacts:
            raise FileNotFoundError(depth_name)
        depth = state.load(depth_name, step=nn)
    except Exception as exc:
        return {"error": f"depth for step {nn} not found: {exc}"}

    out = robust_surface_centroid(
        depth,
        meta["K"],
        meta.get("T_base_cam"),
        row,
        col,
        radius=_BACKPROJECT_RADIUS if radius is None else radius,
    )
    if "error" in out:
        return out

    out["step"] = nn
    out["camera"] = "scene"
    out["camera_frame"] = meta.get("frame", "scene_cam")
    if meta.get("T_base_cam") is not None:
        out["frame"] = "base_link"
    else:
        out["frame"] = meta.get("frame", "scene_cam")
        out["note"] = (
            "scene camera not calibrated (no T_base_cam); returning camera-frame "
            "xyz only. Run robots/lerobot/calibrate_scene_cam.py."
        )
    return out


# ---------------------------------------------------------------------------
# Tool schema declarations (Anthropic-shaped)
# ---------------------------------------------------------------------------

TOOLS_SPEC: list[dict[str, Any]] = [
    {
        "name": "view_driver_state",
        "description": (
            "Read one recorded state and its observation artifacts. Step -1 "
            "selects the latest entry. Embeds scene and arm camera frames."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "step": {
                    "type": "integer",
                    "default": -1,
                    "description": "Step number; 0 = initial, -1 = latest.",
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
                    "type": "integer",
                    "default": -1,
                    "description": "Step whose depth to use; -1 = latest.",
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
