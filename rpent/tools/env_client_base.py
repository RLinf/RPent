"""Unified env client base class. Design reference for adding a new env
backend: ``docs/source-zh/rst_source/development/add_env.rst``.
"""

from __future__ import annotations

from typing import Any


class BaseEnvClient:
    """Unified env client base class."""

    _TIMEOUT_S: dict[str, float] = {
        "default": 30.0,
        "env.reset": 120.0,
        "env.step": 60.0,
        "env.chunk_step": 120.0,
    }

    def __init__(
        self,
        client,
        *,
        expected_meta: dict,
    ):
        self._client = client
        server_meta = self._client.call(
            "env.get_env_meta", timeout_s=self._TIMEOUT_S["default"]
        )
        if server_meta != expected_meta:
            raise RuntimeError(
                f"env_meta mismatch: expected={expected_meta!r} "
                f"actual={server_meta!r}. The env_server was launched with "
                "different args than this client expects; kill the stale "
                "env_server and relaunch."
            )
        self.server_meta = dict(server_meta)
        execution = self.server_meta.get("execution", {})
        self.execution_capabilities = (
            dict(execution) if isinstance(execution, dict) else {}
        )
        self.last_obs: Any = None
        self.last_reset_info: dict[str, Any] = {}
        self.last_info: dict[str, Any] = {}
        self.reset()

    @staticmethod
    def _require_result_tuple(result: Any, size: int, method: str) -> tuple:
        if not isinstance(result, (list, tuple)) or len(result) != size:
            raise TypeError(f"{method} must return a {size}-item tuple, got {result!r}")
        return tuple(result)

    def reset(self):
        """Reset the env, cache it, and return ``(obs, info)``."""
        result = self._client.call("env.reset", timeout_s=self._TIMEOUT_S["env.reset"])
        obs, info = self._require_result_tuple(result, 2, "env.reset")
        if not isinstance(info, dict):
            raise TypeError(f"env.reset info must be a mapping, got {info!r}")
        self.last_obs = obs
        self.last_reset_info = dict(info)
        self.last_info = info
        return obs, info

    def step(self, flat_action):
        """Execute one env action. Returns the gym 5-tuple
        ``(obs, rew, term, trunc, info)``.

        Also updates the ``self.last_obs`` cache with the first element (obs)
        of the returned tuple.
        """
        result = self._client.call(
            "env.step", args=(flat_action,), timeout_s=self._TIMEOUT_S["env.step"]
        )
        result = self._require_result_tuple(result, 5, "env.step")
        self.last_obs = result[0]
        self.last_info = result[4]
        return result

    def chunk_step(self, flat_actions, *, return_all_frames: bool = False):
        """Execute N actions in one batch. Returns the 5-tuple
        ``(obs, rew, term, trunc, info)``.

        - ``obs_or_list``: ``list[Obs]`` when ``return_all_frames=True`` (one
          per step, carrying the per-step render); the final obs dict when
          ``False``.
        - Updates the ``self.last_obs`` cache: if the returned obs is a list,
          take the last element; otherwise assign directly.

        ``return_all_frames`` is an optional capability. Backends that declare
        ``execution.chunk_step_all_frames = false`` reject it before the RPC.
        """
        supports_all_frames = self.execution_capabilities.get(
            "chunk_step_all_frames", True
        )
        if return_all_frames and supports_all_frames is not True:
            raise ValueError("env.chunk_step does not support return_all_frames=True")
        result = self._client.call(
            "env.chunk_step",
            args=(flat_actions,),
            kwargs={"return_all_frames": return_all_frames},
            timeout_s=self._TIMEOUT_S["env.chunk_step"],
        )
        result = self._require_result_tuple(result, 5, "env.chunk_step")
        obs_field = result[0]
        if isinstance(obs_field, list):
            self.last_obs = obs_field[-1]
        else:
            self.last_obs = obs_field
        self.last_info = result[4]
        return result
