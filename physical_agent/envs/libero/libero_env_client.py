"""LIBERO env client that forwards calls over a driver client.

Lives in :mod:`physical_agent.envs.libero` because the methods exposed
here (``current_raw_obs`` / ``render_agentview`` / ``cached_image`` /
…) reference LIBERO-specific obs dict keys and camera names. The
generic gym-style base lives in
:mod:`physical_agent.rpc_driver.env_client`.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from physical_agent.rpc_driver.base import RpcClient


_TIMEOUT_S = {
    "env.reset": 120.0,
    "env.step": 60.0,
    "env.chunk_step": 120.0,
    "env.raw_obs": 30.0,
    "env.current_raw_obs": 30.0,
    "env.render_agentview": 30.0,
    "env.get_camera_meta": 30.0,
    "env.set_image_render_enabled": 30.0,
    "env.cached_image": 30.0,
}


class LiberoEnvClient:
    """Remote implementation of the LIBERO env protocol."""

    def __init__(
        self,
        client: RpcClient,
        *,
        return_all_frames: bool = False,
    ):
        self._client = client
        self.return_all_frames = return_all_frames

    def reset(self) -> tuple[dict, Any]:
        return self._client.call("env.reset", timeout_s=_TIMEOUT_S["env.reset"])

    def step(self, action) -> tuple[dict, Any, np.ndarray, Any, Any]:
        return self._client.call(
            "env.step", args=(action,), timeout_s=_TIMEOUT_S["env.step"]
        )

    def chunk_step(self, actions) -> tuple[Any, Any, Any, Any, Any]:
        """Run an action chunk in one RPC. Returns the 5-positional tuple
        ``(obs_or_list, reward, terminated, truncated, info)``.

        ``obs`` is ``list[Obs]`` when ``self.return_all_frames`` is True
        (one entry per chunk step), otherwise the final ``Obs`` dict.
        Terminated / truncated retain their ``[num_envs, chunk_size]``
        shape — callers reduce per-env-idx themselves.
        """
        return self._client.call(
            "env.chunk_step",
            args=(actions,),
            kwargs={"return_all_frames": self.return_all_frames},
            timeout_s=_TIMEOUT_S["env.chunk_step"],
        )

    def raw_obs(self, env_idx: int = 0) -> dict:
        return self._client.call(
            "env.raw_obs", args=(env_idx,), timeout_s=_TIMEOUT_S["env.raw_obs"]
        )

    def render_agentview(self, env_idx: int = 0) -> np.ndarray:
        return self._client.call(
            "env.render_agentview",
            args=(env_idx,),
            timeout_s=_TIMEOUT_S["env.render_agentview"],
        )

    def get_camera_meta(self) -> dict | None:
        return self._client.call(
            "env.get_camera_meta", timeout_s=_TIMEOUT_S["env.get_camera_meta"]
        )

    def set_image_render_enabled(self, enabled: bool) -> None:
        self._client.call(
            "env.set_image_render_enabled",
            args=(bool(enabled),),
            timeout_s=_TIMEOUT_S["env.set_image_render_enabled"],
        )

    def cached_image(self) -> np.ndarray | None:
        return self._client.call(
            "env.cached_image", timeout_s=_TIMEOUT_S["env.cached_image"]
        )
