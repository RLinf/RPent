"""LIBERO env client that forwards calls over a driver client.

Lives in :mod:`physical_agent.envs.libero` because the methods exposed
here (``raw_obs`` / ``render_agentview`` / ``cached_image`` / …)
reference LIBERO-specific obs dict keys and camera names. The generic
gym-style base lives in :mod:`physical_agent.rpc_driver.env_client`.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from physical_agent.rpc_driver.base import RpcClient


_TIMEOUT_S = {
    "default": 30.0,
    "env.reset": 120.0,
    "env.step": 60.0,
    "env.chunk_step": 120.0,
}


class LiberoEnvClient:
    """Remote implementation of the LIBERO env protocol."""

    def __init__(
        self,
        client: RpcClient,
        *,
        expected_meta: dict,
        return_all_frames: bool = False,
    ):
        self._client = client
        self.return_all_frames = return_all_frames
        self.episode_done = False
        server_meta = self._client.call(
            "env.get_env_meta", timeout_s=_TIMEOUT_S["default"]
        )
        assert server_meta == expected_meta, (
            f"env_meta mismatch: expected={expected_meta!r} "
            f"actual={server_meta!r}. The env_server was launched with "
            "different args than this client expects — kill the stale "
            "env_server and relaunch."
        )
        self.reset()

    def check_done(self, term, trunc) -> None:
        if np.asarray(term).any() or np.asarray(trunc).any():
            self.episode_done = True

    def reset(self) -> tuple[dict, Any]:
        ret = self._client.call("env.reset", timeout_s=_TIMEOUT_S["env.reset"])
        self.episode_done = False
        return ret

    def step(self, action) -> tuple[dict, Any, np.ndarray, Any, Any]:
        assert not self.episode_done, (
            "env.step called after the episode signaled term/trunc"
        )
        ret = self._client.call(
            "env.step", args=(action,), timeout_s=_TIMEOUT_S["env.step"]
        )
        _, _, term, trunc, _ = ret
        self.check_done(term, trunc)
        return ret

    def chunk_step(self, actions) -> tuple[Any, Any, Any, Any, Any]:
        """Run an action chunk in one RPC. Returns the 5-positional tuple
        ``(obs_or_list, reward, terminated, truncated, info)``.

        ``obs`` is ``list[Obs]`` when ``self.return_all_frames`` is True
        (one entry per chunk step), otherwise the final ``Obs`` dict.
        Terminated / truncated have shape ``[chunk_size]`` after the
        server strips the env dim.
        """
        assert not self.episode_done, (
            "env.chunk_step called after the episode signaled term/trunc"
        )
        ret = self._client.call(
            "env.chunk_step",
            args=(actions,),
            kwargs={"return_all_frames": self.return_all_frames},
            timeout_s=_TIMEOUT_S["env.chunk_step"],
        )
        _, _, term, trunc, _ = ret
        self.check_done(term, trunc)
        return ret

    def raw_obs(self) -> dict:
        return self._client.call("env.raw_obs", timeout_s=_TIMEOUT_S["default"])

    def render_agentview(self) -> np.ndarray:
        return self._client.call(
            "env.render_agentview", timeout_s=_TIMEOUT_S["default"]
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
            timeout_s=_TIMEOUT_S["default"],
        )

    def get_task_language(self) -> str | None:
        return self._client.call(
            "env.get_task_language", timeout_s=_TIMEOUT_S["default"]
        )

    def cached_image(self) -> np.ndarray | None:
        return self._client.call(
            "env.cached_image", timeout_s=_TIMEOUT_S["default"]
        )
