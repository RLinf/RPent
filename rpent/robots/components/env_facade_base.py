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
        super().__init__()
        self._dispatch_lock = RWLock()
        # server side does not cache obs / terminated — only the client caches
        self._rpc: dict[str, Callable] = {}
        self._readonly_methods: set[str] = set()
        self._register_rpc()

    # ---- framework ----
    def get_env_meta(self) -> dict:
        """Returns a snapshot dict of the launch args, used by the client to
        verify config consistency after startup."""
        raise NotImplementedError

    def close(self):
        raise NotImplementedError

    def _register_rpc(self):
        """Can be overridden to register more RPC methods."""
        self._rpc["env.reset"] = self.reset
        self._rpc["env.get_env_meta"] = self.get_env_meta
        self._rpc["env.close"] = self.close
        self._rpc["env.step"] = self.step
        self._rpc["env.chunk_step"] = self.chunk_step
        self._readonly_methods.add("env.get_env_meta")

    def _dispatch(self, method: str, args: tuple, kwargs: dict) -> Any:
        handler = self._rpc.get(method)
        if handler is None:
            raise ValueError(f"unknown RPC method: {method!r}")
        if method in self._readonly_methods:
            with self._dispatch_lock.read():
                return handler(*args, **kwargs)
        with self._dispatch_lock.write():
            return handler(*args, **kwargs)

    # ---- abstract methods (subclasses must override) ----
    def reset(self):
        """Reset the env and return ``(initial_obs, info)``."""
        raise NotImplementedError

    def step(self, flat_action):
        """Execute one env action. Returns the gym 5-tuple result."""
        raise NotImplementedError

    def chunk_step(self, flat_actions, *, return_all_frames: bool = False):
        """Execute N actions in one batch. Returns the 5-tuple result.

        - ``obs_or_list``: ``list[Obs]`` when ``return_all_frames=True`` (one
          per step, carrying the per-step render); the final obs dict when
          ``False``.
        - ``return_all_frames``: optional capability for backends that can
          return one observation per action. Unsupported backends must reject
          it before executing any action.
        """
        raise NotImplementedError
