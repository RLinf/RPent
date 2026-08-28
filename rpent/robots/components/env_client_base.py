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

"""Unified env client base class. Design reference for adding a new env
backend: ``docs/source-zh/rst_source/development/add_env.rst``.
"""

from __future__ import annotations


class BaseEnvClient:
    """Unified env client base class."""

    _TIMEOUT_S: dict[str, float] = {
        "default": 30.0,
        "env.reset": 120.0,
        "env.step": 60.0,
        "env.chunk_step": 120.0,
        "env.render_camera": 120.0,
    }

    def __init__(self, client, *, expected_meta: dict):
        self._client = client
        server_meta = self._client.call(
            "env.get_env_meta",
            timeout_s=self._TIMEOUT_S["default"],
        )
        assert server_meta == expected_meta, (
            f"env_meta mismatch: expected={expected_meta!r} actual={server_meta!r}. "
            "The env_server was launched with different args than this client "
            "expects — kill the stale env_server and relaunch."
        )
        self.reset()

    def reset(self):
        """Reset the env and return the initial obs. Also updates the
        ``self.last_obs`` cache."""
        self.last_obs = self._client.call(
            "env.reset",
            timeout_s=self._TIMEOUT_S["env.reset"],
        )
        return self.last_obs

    def step(self, flat_action):
        """Execute one env action. Returns the gym 5-tuple
        ``(obs, rew, term, trunc, info)``.

        Also updates the ``self.last_obs`` cache with the first element (obs)
        of the returned tuple.
        """
        result = self._client.call(
            "env.step",
            args=(flat_action,),
            timeout_s=self._TIMEOUT_S["env.step"],
        )
        self.last_obs = result[0]
        return result

    def chunk_step(self, flat_actions, *, return_all_frames: bool = False):
        """Execute N actions in one batch. Returns the 5-tuple
        ``(obs, rew, term, trunc, info)``.

        - ``obs_or_list``: ``list[Obs]`` when ``return_all_frames=True`` (one
          per step, carrying the per-step render); the final obs dict when
          ``False``.
        - Updates the ``self.last_obs`` cache: if the returned obs is a list,
          take the last element; otherwise assign directly.

        ``return_all_frames`` is the standard perf-vs-video switch. ``True``
        renders agentview per step (high-density video); ``False`` renders only
        the final obs (fast).
        """
        result = self._client.call(
            "env.chunk_step",
            args=(flat_actions,),
            kwargs={"return_all_frames": return_all_frames},
            timeout_s=self._TIMEOUT_S["env.chunk_step"],
        )
        obs_field = result[0]
        if isinstance(obs_field, list):
            self.last_obs = obs_field[-1]
        else:
            self.last_obs = obs_field
        return result

    def get_camera_meta(self, camera_name, **kwargs):
        return self._client.call(
            "env.get_camera_meta",
            kwargs={"camera_name": camera_name, **kwargs},
            timeout_s=self._TIMEOUT_S["default"],
        )

    def render_camera(self, camera_name, **kwargs):
        return self._client.call(
            "env.render_camera",
            kwargs={"camera_name": camera_name, **kwargs},
            timeout_s=self._TIMEOUT_S["env.render_camera"],
        )

    def get_task_language(self) -> str | None:
        """Return the language description of the current task."""
        return self._client.call(
            "env.get_task_language", timeout_s=self._TIMEOUT_S["default"]
        )
