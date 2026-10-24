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

"""XPolicyLab VLA client — shared ``BaseVLAClient`` RPC surface."""

from __future__ import annotations

from typing import Any

from rpent.robots.components.vla_client_base import BaseVLAClient


class XPolicyLabVLAClient(BaseVLAClient):
    """Client for the shared XPolicyLab WebSocket policy adapter.

    Legacy callers use ``predict`` and ``reset``. A stateful bridge requires
    a session-aware transport registered through ``wait_for_ready``; use
    ``reset_session``, ``update_obs`` and ``get_action`` on that connection.
    """

    def healthz(self) -> dict[str, Any]:
        return self._client.call("healthz")

    def reset(self) -> None:
        self._client.call("reset")

    def reset_session(self) -> None:
        """Acquire the stateful policy and clear its episode history."""
        self._client.call("vla.reset_session", timeout_s=self._TIMEOUT_S["predict"])

    def update_obs(self, obs: dict[str, Any], step: int) -> None:
        """Send one observation at the next contiguous episode control step."""
        self._client.call(
            "vla.update_obs",
            args=(obs, step),
            timeout_s=self._TIMEOUT_S["predict"],
        )

    def get_action(self, *, include_info: bool = False) -> Any:
        """Infer once, optionally retaining the backend's result envelope."""
        return self._client.call(
            "vla.get_action",
            **(
                {"kwargs": {"include_info": include_info}}
                if include_info is not False
                else {}
            ),
            timeout_s=self._TIMEOUT_S["predict"],
        )

    def close(self) -> None:
        """Release this client's RPC session."""
        self._client.close()

    def end_episode(self) -> None:
        """Release policy history while keeping this transport reusable."""
        self._client.call("vla.end_episode")
