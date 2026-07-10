"""Franka env client that forwards calls over the driver RPC boundary."""
from __future__ import annotations

from typing import Any

import numpy as np

from rpent.rpc_driver.base import RpcClient


_TIMEOUT_S = {
    "default": 30.0,
    "env.reset": 180.0,
    "env.get_spec": 30.0,
    "env.get_obs": 60.0,
    "env.get_ee_pose": 30.0,
    "env.get_camera_meta": 30.0,
    "env.move_to": 120.0,
    "env.move_delta": 90.0,
    "env.open_gripper": 30.0,
    "env.close_gripper": 30.0,
}


class FrankaEnvClient:
    """Remote stub for the standalone Franka env server protocol."""

    def __init__(self, client: RpcClient):
        self._client = client
        self._spec: dict | None = None

    def reset(self) -> tuple[dict, Any]:
        """Clear errors, drive the arm to home, and return ``(obs, info)``."""
        return self._client.call("env.reset", timeout_s=_TIMEOUT_S["env.reset"])

    def get_spec(self) -> dict:
        """Return and cache the env's static self-description."""
        if self._spec is None:
            self._spec = self._client.call(
                "env.get_spec", timeout_s=_TIMEOUT_S["env.get_spec"]
            )
        return self._spec

    def get_obs(self) -> dict:
        """Fetch the current observation without moving the arm."""
        return self._client.call("env.get_obs", timeout_s=_TIMEOUT_S["env.get_obs"])

    def get_ee_pose(self) -> dict:
        """Return the current TCP pose in the Franka base frame."""
        return self._client.call(
            "env.get_ee_pose", timeout_s=_TIMEOUT_S["env.get_ee_pose"]
        )

    def get_camera_meta(self) -> dict:
        """Return live RGB-D camera intrinsics and base-frame extrinsics."""
        return self._client.call(
            "env.get_camera_meta", timeout_s=_TIMEOUT_S["env.get_camera_meta"]
        )

    def move_to(
        self,
        xyz,
        *,
        yaw_deg: float | None = None,
        gripper: str | None = None,
    ) -> dict:
        """Move to a base-frame Cartesian target.

        The agent-facing API intentionally hides arbitrary quaternions. When
        ``yaw_deg`` is provided, the driver receives a down-facing euler target
        with the requested yaw; otherwise it preserves the current orientation.
        """
        xyz = np.asarray(xyz, dtype=float).reshape(-1)[:3].tolist()
        kwargs: dict[str, Any] = {"gripper": gripper}
        if yaw_deg is not None:
            kwargs["euler_xyz"] = [float(np.pi), 0.0, float(np.radians(yaw_deg))]
        return self._client.call(
            "env.move_to",
            args=(xyz,),
            kwargs=kwargs,
            timeout_s=_TIMEOUT_S["env.move_to"],
        )

    def move_delta(
        self,
        *,
        dxyz=None,
        yaw_delta_deg: float | None = None,
        gripper: str | None = None,
    ) -> dict:
        """Nudge the TCP by relative translation and optional yaw."""
        kwargs: dict[str, Any] = {"gripper": gripper}
        if dxyz is not None:
            kwargs["dxyz"] = np.asarray(dxyz, dtype=float).reshape(-1)[:3].tolist()
        if yaw_delta_deg is not None:
            kwargs["drpy_deg"] = [0.0, 0.0, float(yaw_delta_deg)]
        return self._client.call(
            "env.move_delta",
            kwargs=kwargs,
            timeout_s=_TIMEOUT_S["env.move_delta"],
        )

    def open_gripper(self) -> dict:
        """Open the Franka Hand."""
        return self._client.call(
            "env.open_gripper", timeout_s=_TIMEOUT_S["env.open_gripper"]
        )

    def close_gripper(self) -> dict:
        """Close/grasp with the Franka Hand."""
        return self._client.call(
            "env.close_gripper", timeout_s=_TIMEOUT_S["env.close_gripper"]
        )
