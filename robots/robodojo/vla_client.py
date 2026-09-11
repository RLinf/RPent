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

"""RoboDojo Pi_05 VLA client — shared ``BaseVLAClient`` RPC surface."""

from __future__ import annotations

from typing import Any

from rpent.robots.components.vla_client_base import BaseVLAClient


class RoboDojoVLAClient(BaseVLAClient):
    """Client for the RoboDojo Pi_05 policy server (XPolicyLab ws backend).

    Uses the shared :class:`BaseVLAClient` wire protocol (``vla.predict``);
    the RoboDojo-specific server lives in ``robots/robodojo/vla_server.py``
    and adapts the XPolicyLab WebSocket Pi_05 to that protocol.
    """

    def healthz(self) -> dict[str, Any]:
        return self._client.call("healthz")

    def reset(self) -> None:
        self._client.call("reset")
