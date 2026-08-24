"""Unified env backend base class. Design reference for adding a new env
backend: ``docs/source-zh/rst_source/development/add_env.rst``.
"""

from __future__ import annotations

from typing import Any, Callable

from rpent.utils.rpc import RpcFacade
from rpent.utils.rwlock import RWLock


class BaseEnvFacade(RpcFacade):
    """Unified env backend base class.

    State-caching principle:
        The server side does **not** cache any observation results. Cache
        variables such as ``_last_obs`` / ``_terminated`` live only on the
        client side (see ``BaseEnvClient`` in ``rpent/tools/env_client_base.py``).
        The server is stateless (apart from the env's own physical state) and
        re-reads the env on every request.

        Why:
        1. Supports multiple concurrent clients without cross-contamination.
        2. Clear responsibilities: the server only executes and returns; the
           client owns the caching policy.

        TODO: the current robocasa ``robots/robocasa/env_server.py`` already
        partially violates this principle (the server-side ``_last_obs`` is
        used for chunk_step's embedded rendering). After unification, chunk_step's
        embedded rendering should use a temporary local variable within the
        current RPC call, not ``self``.

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
        self._dispatch_lock = RWLock()
        self._rpc: dict[str, Callable] = {}
        self._readonly_methods: set[str] = set()
        self._register_rpc()

    # ---- framework ----
    def close(self):
        pass

    def _register_rpc(self):
        """Can be overridden to register more RPC methods."""
        self._rpc["env.get_env_meta"] = self.get_env_meta

        self._rpc["env.reset"] = self.reset
        self._rpc["env.step"] = self.step
        self._rpc["env.chunk_step"] = self.chunk_step

        self._rpc["env.get_camera_meta"] = self.get_camera_meta
        self._rpc["env.render_camera"] = self.render_camera
        self._rpc["env.get_task_language"] = self.get_task_language

        # Read-only methods that can run parallel
        self._readonly_methods.update([
            "env.get_env_meta",
            "env.get_camera_meta",
            "env.get_camera_transform",
            "env.get_task_language",
        ])

    # ---- lifecycle ----
    def get_env_meta(self) -> dict:
        """Returns a snapshot dict of the launch args, used by the client to
        verify config consistency after startup."""
        raise NotImplementedError

    # ---- functionality (subclasses must override) ----
    def reset(self):
        """Reset the env and return ``(initial_obs, info)``."""
        raise NotImplementedError

    def step(self):
        """Execute one env action. Returns the gym tuple result."""
        raise NotImplementedError

    def chunk_step(self):
        """Execute N actions in one batch. Returns the gym tuple result."""
        raise NotImplementedError

    def get_camera_meta(self):
        raise NotImplementedError

    def render_camera(self):
        raise NotImplementedError

    def get_task_language(self):
        raise NotImplementedError
