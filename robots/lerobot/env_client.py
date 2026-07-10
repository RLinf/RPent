"""LeRobot SO101 env client that forwards calls over a driver RPC client.

Mirrors the RPC surface exposed by ``deployment/lerobot/env_server.py``
(:class:`SO101LeRobotEnv`): a minimal gym-style ``reset`` / ``step`` plus a
``get_spec`` self-description. Each method turns one agent-side call into one
RPC against the driver process via :class:`RpcClient`.

The agent process does not import torch; the driver returns plain numpy /
floats, so values cross the wire unchanged.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from rpent.rpc_driver.base import RpcClient


# Per-method RPC timeouts (seconds). ``reset`` moves the arm to its rest pose
# (server sleeps ~1s); ``step`` advances one control tick.
_TIMEOUT_S = {
    "default": 30.0,
    "env.reset": 120.0,
    "env.step": 60.0,
    "env.get_spec": 30.0,
    "env.get_ee_pose": 30.0,
    "env.get_scene_camera_meta": 30.0,
    "env.move_to": 120.0,
    "env.move_joints_delta": 60.0,
    "env.get_obs": 60.0,
}


class LerobotEnvClient:
    """Remote stub for the SO101 LeRobot env protocol."""

    def __init__(self, client: RpcClient):
        self._client = client
        self._spec: dict | None = None

    def reset(self) -> tuple[dict, Any]:
        """Drive the arm to its rest pose and return ``(obs, info)``.

        ``obs`` is ``{"state": {"joint_position": (5,) float32,
        "gripper_position": (1,) float32}, "frames": {<cam>: (H, W, 3)
        uint8, ...}}``.
        """
        return self._client.call("env.reset", timeout_s=_TIMEOUT_S["env.reset"])

    def step(self, action) -> tuple[dict, float, bool, bool, dict]:
        """Send one absolute joint-target command ``[q1..q5, gripper]``.

        Returns ``(obs, reward, terminated, truncated, info)``. Targets are
        clipped to the arm's joint limits server-side.
        """
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        return self._client.call(
            "env.step", args=(action,), timeout_s=_TIMEOUT_S["env.step"]
        )

    def get_spec(self) -> dict:
        """Return (and cache) the env's static self-description.

        Keys: ``action_dim``, ``arm_joints``, ``action_low``,
        ``action_high``, ``camera_names``, ``max_episode_steps``.
        """
        if self._spec is None:
            self._spec = self._client.call(
                "env.get_spec", timeout_s=_TIMEOUT_S["env.get_spec"]
            )
        return self._spec

    def get_ee_pose(self) -> dict:
        """Live FK: gripper pose in the base (world) frame.

        Returns ``{xyz, quat_wxyz, joints_deg, T_base_gripper, frame}`` (or
        ``{error}`` if FK is unavailable on the driver).
        """
        return self._client.call(
            "env.get_ee_pose", timeout_s=_TIMEOUT_S["env.get_ee_pose"]
        )

    def get_scene_camera_meta(self) -> dict:
        """Scene-camera intrinsics + depth scale + base extrinsic.

        Returns ``{serial, K, width, height, depth_scale, frame, calibrated,
        T_base_cam}`` (``T_base_cam`` is ``None`` until calibrated).
        """
        return self._client.call(
            "env.get_scene_camera_meta",
            timeout_s=_TIMEOUT_S["env.get_scene_camera_meta"],
        )

    def move_to(
        self,
        xyz,
        *,
        gripper: float | None = None,
        approach: str = "free",
        yaw_deg: float | None = None,
    ) -> dict:
        """Move the gripper to a world-frame (base_link) XYZ via IK.

        ``approach="free"`` uses position-only IK (wrist orientation free);
        ``approach="down"`` drives the gripper to point straight down for
        grasping, with ``yaw_deg`` setting the jaw-line heading (auto-searched
        when ``None``). The target is clipped to the workspace box and
        approached in capped waypoints server-side; holds the current gripper
        unless ``gripper`` is given. Returns the move log (``reached``,
        ``pos_error_m``, ``approach_tilt_deg``, ...).
        """
        return self._client.call(
            "env.move_to", args=(xyz,),
            kwargs={"gripper": gripper, "approach": approach, "yaw_deg": yaw_deg},
            timeout_s=_TIMEOUT_S["env.move_to"],
        )

    def move_joints_delta(
        self,
        delta_deg,
        *,
        gripper_delta: float | None = None,
    ) -> dict:
        """Nudge each arm joint by a relative amount (degrees).

        ``delta_deg`` is ``[d_pan, d_lift, d_elbow, d_wrist_flex, d_wrist_roll]``
        added to the current joints (each capped + clamped to limits
        server-side). Optionally nudge the gripper by ``gripper_delta``. Returns
        the achieved joints + EE pose. Use for fine alignment move_to cannot
        express.
        """
        return self._client.call(
            "env.move_joints_delta", args=(delta_deg,),
            kwargs={"gripper_delta": gripper_delta},
            timeout_s=_TIMEOUT_S["env.move_joints_delta"],
        )

    def get_obs(self) -> dict:
        """Fetch the current observation without moving the arm."""
        return self._client.call("env.get_obs", timeout_s=_TIMEOUT_S["env.get_obs"])
