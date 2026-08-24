"""Dual-Franka env client forwarding explicit methods over an RPC transport."""

from __future__ import annotations

from typing import Any

import numpy as np

from robots.franka.env_client import FrankaEnvClient

_MOTION_TIMEOUT_S = 120.0


class DualFrankaEnvClient(FrankaEnvClient):
    """Remote client for one RLinf-backed dual-Franka environment."""

    def move_delta(
        self, arm: str, delta_xyz: np.ndarray | list[float]
    ) -> dict[str, Any]:
        return self._client.call(
            "env.move_delta",
            kwargs={
                "arm": str(arm),
                "delta_xyz": np.asarray(delta_xyz, dtype=np.float32),
            },
            timeout_s=_MOTION_TIMEOUT_S,
        )

    def rotate_delta(
        self, arm: str, delta_rpy: np.ndarray | list[float]
    ) -> dict[str, Any]:
        return self._client.call(
            "env.rotate_delta",
            kwargs={
                "arm": str(arm),
                "delta_rpy": np.asarray(delta_rpy, dtype=np.float32),
            },
            timeout_s=_MOTION_TIMEOUT_S,
        )

    def set_gripper(self, arm: str, *, open: bool) -> dict[str, Any]:
        return self._client.call(
            "env.set_gripper",
            kwargs={"arm": str(arm), "open": bool(open)},
            timeout_s=_MOTION_TIMEOUT_S,
        )
