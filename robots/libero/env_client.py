# Copyright 2026 The RPent Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""LIBERO env client that forwards calls over an RPC transport.

Lives in :mod:`robots.libero` because the methods exposed
here (``raw_obs`` / ``render_camera`` / ``get_camera_meta`` / …)
reference LIBERO-specific obs dict keys and camera names. The generic
transport layer lives in :mod:`rpent.utils.rpc.socket_rpc`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from rpent.robots.components.env_client_base import BaseEnvClient
from rpent.utils.logging import get_logger
from rpent.utils.rpc import RpcClient, RpcError

if TYPE_CHECKING:
    from robots.libero.evaluation import LiberoTrial


logger = get_logger(__name__)


class LiberoEnvClient(BaseEnvClient):
    """Remote implementation of the LIBERO env protocol."""

    active_trial: LiberoTrial | None = None

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
        if self.active_trial is not None:
            if not self.active_trial.finished:
                raise RuntimeError("Cannot reset an active evaluation episode")
            self.active_trial = None
        self.last_obs, info = super().reset()
        self.last_reset_info = info
        self.terminated = False
        self.truncated = False
        return self.last_obs, info

    def begin_trial(self) -> bool | None:
        """Associate this exact reset with a native verdict and an action fence."""
        episode_id = self.last_reset_info["episode_id"]
        try:
            status = self._client.call(
                "env.begin_trial",
                args=(episode_id,),
                timeout_s=self._TIMEOUT_S["default"],
            )
        except (RpcError, OSError):
            # A lost acknowledgment may still have reserved the server. Seal the
            # exact reset even if the delayed begin request arrives after this one.
            try:
                self._client.call(
                    "env.seal_episode",
                    args=(episode_id,),
                    timeout_s=self._TIMEOUT_S["default"],
                )
            except (RpcError, OSError) as exc:
                logger.warning("Could not release LIBERO evaluation admission: %s", exc)
            raise
        self.terminated = status["success"] is True
        return status["success"]

    def seal_episode(self) -> bool:
        episode_id = self.last_reset_info["episode_id"]
        status = self._client.call(
            "env.seal_episode",
            args=(episode_id,),
            timeout_s=self._TIMEOUT_S["env.chunk_step"],
        )
        if status.get("episode_id") != episode_id or status.get("sealed") is not True:
            return False
        self.active_trial.observe(status)
        return True

    def _trial_actions(self, actions, *, return_all_frames):
        remaining = self.active_trial.admit_action()
        obs, rew, term, trunc, info, status = self._client.call(
            "env.trial_actions",
            args=(self.last_reset_info["episode_id"], actions),
            kwargs={"return_all_frames": return_all_frames, "remaining_s": remaining},
            timeout_s=self._TIMEOUT_S["env.chunk_step"],
        )
        self.last_obs = obs[-1] if return_all_frames else obs
        self.check_done(term, trunc)
        self.active_trial.observe(status)
        return obs, rew, term, trunc, info

    def step(self, action) -> tuple[dict, Any, np.ndarray, Any, Any]:
        assert not (self.terminated or self.truncated), (
            "env.step called after the episode signaled term/trunc"
        )
        if self.active_trial is not None:
            obs, rew, term, trunc, info = self._trial_actions(
                [action], return_all_frames=False
            )
            return obs, rew[0], term[0], trunc[0], info[0]
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
        if self.active_trial is not None:
            return self._trial_actions(actions, return_all_frames=return_all_frames)
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
        return super().render_camera(
            camera_name, height=height, width=width, depth=depth
        )

    def get_camera_meta(
        self,
        camera_name: str = "agentview",
        height: int = 256,
        width: int = 256,
    ) -> dict | None:
        return super().get_camera_meta(camera_name, height=height, width=width)
