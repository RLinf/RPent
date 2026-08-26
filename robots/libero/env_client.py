"""LIBERO env client that forwards calls over an RPC transport.

Lives in :mod:`robots.libero` because the methods exposed
here (``raw_obs`` / ``render_camera`` / ``get_camera_meta`` / …)
reference LIBERO-specific obs dict keys and camera names. The generic
transport layer lives in :mod:`rpent.utils.rpc.socket_rpc`.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from rpent.robots.components.env_client_base import BaseEnvClient
from rpent.utils.rpc import RpcClient


class LiberoEnvClient(BaseEnvClient):
    """Remote implementation of the LIBERO env protocol."""

    _TIMEOUT_S = {
        **BaseEnvClient._TIMEOUT_S,
        "env.render_camera": 120.0,
    }

    def __init__(
        self,
        client: RpcClient,
        *,
        expected_meta: dict,
        return_all_frames: bool = False,
    ):
        self.return_all_frames = return_all_frames
        self.terminated = False
        self.truncated = False
        super().__init__(client, expected_meta=expected_meta)

    def check_done(self, term, trunc) -> None:
        self.terminated |= bool(np.asarray(term).any())
        self.truncated |= bool(np.asarray(trunc).any())

    def reset(self) -> tuple[dict, Any]:
        ret = super().reset()
        self.last_obs = ret[0]
        self.terminated = False
        self.truncated = False
        return ret

    def step(self, action) -> tuple[dict, Any, np.ndarray, Any, Any]:
        assert not (self.terminated or self.truncated), (
            "env.step called after the episode signaled term/trunc"
        )
        ret = super().step(action)
        _, _, term, trunc, _ = ret
        self.check_done(term, trunc)
        return ret

    def chunk_step(
        self, actions, *, return_all_frames: bool | None = None
    ) -> tuple[Any, Any, Any, Any, Any]:
        """Run an action chunk in one RPC. Returns the 5-positional tuple
        ``(obs_or_list, reward, terminated, truncated, info)``.

        ``obs`` is ``list[Obs]`` when ``return_all_frames`` is True
        (one entry per chunk step), otherwise the final ``Obs`` dict.
        Terminated / truncated have shape ``[chunk_size]`` after the
        server strips the env dim.
        """
        assert not (self.terminated or self.truncated), (
            "env.chunk_step called after the episode signaled term/trunc"
        )
        if return_all_frames is None:
            return_all_frames = self.return_all_frames
        ret = super().chunk_step(
            actions,
            return_all_frames=return_all_frames,
        )
        _, _, term, trunc, _ = ret
        self.check_done(term, trunc)
        return ret

    def raw_obs(self) -> dict:
        return self._client.call("env.raw_obs", timeout_s=self._TIMEOUT_S["default"])

    def render_camera(
        self,
        camera_name: str = "agentview",
        height: int = 1024,
        width: int = 1024,
        depth: bool = False,
    ):
        return self._client.call(
            "env.render_camera",
            kwargs={
                "camera_name": camera_name,
                "height": height,
                "width": width,
                "depth": depth,
            },
            timeout_s=self._TIMEOUT_S["env.render_camera"],
        )

    def get_camera_meta(
        self,
        camera_name: str = "agentview",
        height: int = 256,
        width: int = 256,
    ) -> dict | None:
        return self._client.call(
            "env.get_camera_meta",
            kwargs={"camera_name": camera_name, "height": height, "width": width},
            timeout_s=self._TIMEOUT_S["default"],
        )

    def get_task_language(self) -> str | None:
        return self._client.call(
            "env.get_task_language", timeout_s=self._TIMEOUT_S["default"]
        )
