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

"""Unified VLA backend base class."""

from __future__ import annotations

from rpent.utils.rpc import RpcFacade
from rpent.utils.rpc.rpc_facade import DEFAULT_SESSION_TIMEOUT_S


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

    def __init__(
        self,
        *,
        enable_sessions: bool = False,
        session_timeout_s: float = DEFAULT_SESSION_TIMEOUT_S,
    ):
        super().__init__(
            enable_sessions=enable_sessions, session_timeout_s=session_timeout_s
        )
        self._register_rpc()

    # ---- framework ----
    def _register_rpc(self):
        self._rpc["vla.predict"] = self.predict

    # ---- abstract methods (subclasses must override) ----
    def predict(self, *args, **kwargs):
        raise NotImplementedError
