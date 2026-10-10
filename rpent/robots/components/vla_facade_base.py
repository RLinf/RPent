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

"""Server base for the legacy VLA action-chunk RPC contract."""

from rpent.utils.rpc import RpcFacade
from rpent.utils.rpc.rpc_facade import DEFAULT_SESSION_TIMEOUT_S


class BaseVLAFacade(RpcFacade):
    """Register VLA prediction while reusing RPC lifecycle and sessions.

    Subclasses implement ``predict`` and may extend ``_register_rpc`` for
    additional model methods. Session-aware services can implement
    ``_on_session_drop`` and ``reset_session``; see RoboCasa's VLA service.
    """

    PREDICT_METHOD = "vla.predict"

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

    def _register_rpc(self):
        self._rpc[self.PREDICT_METHOD] = self.predict

    def predict(self, *args, **kwargs):
        """Run model inference. Subclasses must implement this method."""
        raise NotImplementedError
