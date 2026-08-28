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

"""Unified env backend base class. Design reference for adding a new env
backend: ``docs/source-zh/rst_source/development/add_env.rst``.
"""

from __future__ import annotations

from rpent.utils.rpc import RpcFacade


class BaseEnvFacade(RpcFacade):
    """Unified env backend base class.

    State-caching principle:
        The server side does **not** cache any observation results. Cache
        variables such as ``_last_obs`` / ``_terminated`` live only on the
        client side (see ``BaseEnvClient`` in ``rpent.robots.components.env_client_base.py``).
        The server is stateless (apart from the env's own physical state) and
        re-reads the env on every request.

        Why:
        1. Supports multiple concurrent clients without cross-contamination.
        2. Clear responsibilities: the server only executes and returns; the
           client owns the caching policy.

    RPC routing:
        ``_dispatch`` uses a registration dict (``self._rpc``) instead of
        dynamic ``getattr`` routing. Subclasses register their own methods in
        ``_register_rpc``.

    EGL single-thread:
        Subclasses that must keep EGL single-threaded must override ``serve``
        and dispatch everything to a dedicated render thread, so the MuJoCo EGL
        context stays on the same thread. See robocasa's override for reference.
    """

    def __init__(self):
        # server side does not cache obs / terminated — only the client caches
        super().__init__()
        self._register_rpc()

    # ---- framework ----
    def get_env_meta(self) -> dict:
        """Returns a snapshot dict of the launch args, used by the client to
        verify config consistency after startup."""
        raise NotImplementedError

    def _register_rpc(self):
        """Can be overridden to register more RPC methods."""
        self._rpc["env.get_env_meta"] = self.get_env_meta

        self._rpc["env.reset"] = self.reset
        self._rpc["env.step"] = self.step
        self._rpc["env.chunk_step"] = self.chunk_step

        self._rpc["env.get_task_language"] = self.get_task_language
        self._rpc["env.get_camera_meta"] = self.get_camera_meta
        self._rpc["env.render_camera"] = self.render_camera

        # Read-only methods that can run parallel
        self._readonly_methods.update(
            [
                "env.get_env_meta",
                "env.get_task_language",
                "env.get_camera_meta",
                "env.render_camera",
            ]
        )

    # ---- abstract methods (subclasses must override) ----
    def reset(self, *args, **kwargs):
        """Reset the env and return ``(initial_obs, info)``."""
        raise NotImplementedError

    def step(self, *args, **kwargs):
        """Execute one env action. Returns the gym tuple result."""
        raise NotImplementedError

    def chunk_step(self, *args, **kwargs):
        """Execute N actions in one batch. Returns the gym tuple result."""
        raise NotImplementedError

    def get_camera_meta(self, *args, **kwargs):
        raise NotImplementedError

    def render_camera(self, *args, **kwargs):
        raise NotImplementedError

    def get_task_language(self, *args, **kwargs):
        raise NotImplementedError
