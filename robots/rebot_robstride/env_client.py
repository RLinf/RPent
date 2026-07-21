"""Agent-side RPC client for the reBot DevArm RobStride server."""

from __future__ import annotations

from typing import Any

from robots.rebot_robstride.config import MOTION_RPC_TIMEOUT_S
from rpent.utils.rpc import RpcClient

_DEFAULT_TIMEOUT_S = 10.0
_ENABLE_TIMEOUT_S = 30.0


class RebotRobstrideEnvClient:
    """Typed facade over the stable ``robot.*`` RPC method names."""

    def __init__(self, client: RpcClient) -> None:
        self._client = client

    def state(self) -> dict[str, Any]:
        return self._client.call("robot.state", timeout_s=_DEFAULT_TIMEOUT_S)

    def enable(self) -> dict[str, Any]:
        return self._client.call("robot.enable", timeout_s=_ENABLE_TIMEOUT_S)

    def move_joints(
        self, positions: list[float], *, duration_s: float = 2.0
    ) -> dict[str, Any]:
        return self._client.call(
            "robot.move_joints",
            kwargs={"positions": positions, "duration_s": duration_s},
            timeout_s=MOTION_RPC_TIMEOUT_S,
        )

    def set_gripper(
        self, position: float, *, duration_s: float = 1.0
    ) -> dict[str, Any]:
        return self._client.call(
            "robot.set_gripper",
            kwargs={"position": position, "duration_s": duration_s},
            timeout_s=MOTION_RPC_TIMEOUT_S,
        )

    def stop_motion(self) -> dict[str, Any]:
        return self._client.call("robot.stop_motion", timeout_s=_DEFAULT_TIMEOUT_S)

    def reset_stop(self) -> dict[str, Any]:
        return self._client.call("robot.reset_stop", timeout_s=_DEFAULT_TIMEOUT_S)

    def emergency_stop(self) -> dict[str, Any]:
        return self._client.call("robot.emergency_stop", timeout_s=_DEFAULT_TIMEOUT_S)

    def heartbeat(self) -> dict[str, Any]:
        return self._client.call("robot.heartbeat", timeout_s=1.0)
