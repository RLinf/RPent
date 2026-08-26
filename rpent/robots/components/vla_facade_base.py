"""Unified VLA backend base class.
Design reference: ``docs/source-zh/rst_source/development/add_vla.rst``.
"""
from __future__ import annotations

from typing import Any

from rpent.utils.rpc import RpcFacade
from rpent.utils.rwlock import RWLock


class BaseVLAFacade(RpcFacade):
    """Unified VLA backend base class.

    Methods subclasses must implement:
        ``predict`` — the subclass performs the actual inference.
        ``__init__`` —  the subclass loads the model itself.

    RPC routing:
        ``_dispatch`` uses a registration dict (``self._rpc``) instead of an
        ``if method == "predict"`` chain. Subclasses register their own methods
        in ``_register_rpc``.

    Session-isolation model (backend-specific, not in the base class):
        For session-aware VLA models, the subclass may implement a session
        isolation model. See the robocasa RLDX VLA implementation for reference.
        Implement ``_on_session_drop`` and ``reset_session``; optionally
        customize ``session_timeout_s`` and ``session_sweep_s`` to periodically
        evict expired sessions.
    """

    def __init__(self):
        super().__init__()
        # Serializes ``predict`` execution: the model is not thread-safe and
        # concurrent RPC requests must not interleave. Kept at the facade level
        # so it still applies when a subclass overrides ``serve``.
        self._dispatch_lock = RWLock()
        self._rpc: dict[str, Any] = {}
        self._readonly_methods: set[str] = set()
        self._register_rpc()

    # ---- framework ----
    def _register_rpc(self):
        self._rpc["vla.predict"] = self.predict

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
    def predict(self, obs, options):
        raise NotImplementedError
