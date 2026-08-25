"""LingBot-VLA websocket client using the checkpoint's official runtime."""

from __future__ import annotations

import os
from typing import Any

import numpy as np

from rpent.tools.vla_client_base import BaseVLAClient


class LingBotWebSocketTransport:
    """Adapt LingBot's official WebSocket client to the RPent RPC interface."""

    def __init__(
        self,
        host: str,
        port: int,
    ):
        self._host = host
        self._port = int(port)
        self._policy = None

    def _get_policy(self):
        if self._policy is None:
            if self._host in {"127.0.0.1", "localhost", "::1"}:
                current = os.environ.get("NO_PROXY", os.environ.get("no_proxy", ""))
                hosts = {item.strip() for item in current.split(",") if item.strip()}
                hosts.update({self._host, "127.0.0.1", "localhost"})
                os.environ["NO_PROXY"] = ",".join(sorted(hosts))
            from deploy.websocket_client_policy import WebsocketClientPolicy

            self._policy = WebsocketClientPolicy(host=self._host, port=self._port)
        return self._policy

    def call(
        self,
        method: str,
        args: tuple = (),
        kwargs: dict | None = None,
        *,
        timeout_s: float | None = None,
    ) -> Any:
        """Map ``vla.predict`` to LingBot's native ``infer`` call.

        The native LingBot WebSocket client doesn't expose a per-call receive
        timeout, so ``timeout_s`` is accepted to satisfy the common transport
        interface but isn't enforced by this transport.
        """
        if method != "vla.predict":
            raise ValueError(f"unsupported LingBot VLA method: {method!r}")
        if kwargs:
            raise ValueError("LingBot vla.predict does not accept keyword arguments")
        if len(args) != 2 or args[1] is not None:
            raise ValueError("LingBot vla.predict requires one payload and no options")
        return self._get_policy().infer(args[0])

    def get_server_metadata(self) -> dict[str, Any]:
        return self._get_policy().get_server_metadata()


class LingBotVLAClient(BaseVLAClient):
    """Wrap ``deploy.websocket_client_policy.WebsocketClientPolicy``."""

    def __init__(
        self,
        host: str,
        port: int,
    ):
        self._transport = LingBotWebSocketTransport(host, port)
        super().__init__(self._transport)

    def validate_contract(self, expected_meta: dict[str, Any]) -> None:
        """Connect and require the expected LingBot runtime identity."""
        actual_meta = self._transport.get_server_metadata()
        if actual_meta != expected_meta:
            raise RuntimeError(
                "LingBot VLA metadata mismatch: "
                f"expected={expected_meta!r} actual={actual_meta!r}. "
                "Connect to the RoboTwin EEF16 LingBot server."
            )

    def infer(self, observation: dict[str, Any]) -> np.ndarray:
        """Infer one eef16 action chunk."""
        views = observation["views"]
        state = observation["robot_state"]
        left_pose = np.asarray(state["left_eef_pose"], dtype=np.float32)
        right_pose = np.asarray(state["right_eef_pose"], dtype=np.float32)
        if left_pose.shape != (7,) or right_pose.shape != (7,):
            raise RuntimeError("RoboTwin end-effector poses must have shape (7,)")
        policy_state = np.concatenate(
            [
                left_pose,
                np.asarray([state["left_gripper"]], dtype=np.float32),
                right_pose,
                np.asarray([state["right_gripper"]], dtype=np.float32),
            ]
        )
        payload = {
            "observation.images.cam_high": views["head"]["rgb"],
            "observation.images.cam_left_wrist": views["left_wrist"]["rgb"],
            "observation.images.cam_right_wrist": views["right_wrist"]["rgb"],
            "observation.state": policy_state,
            "task": observation["task_language"],
        }
        actions = np.asarray(super().predict(payload)["action"], dtype=np.float64)
        if actions.ndim != 2 or actions.shape[1] != 16:
            raise RuntimeError(
                f"LingBot returned {actions.shape}; expected [chunk, 16]"
            )
        return actions
