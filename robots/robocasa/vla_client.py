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

"""RoboCasa VLA client — thin RPC layer over the VLA server."""

from __future__ import annotations

from rpent.robots.components.vla_client_base import BaseVLAClient


class RoboCasaVLAClient(BaseVLAClient):
    _TIMEOUT_S = BaseVLAClient._TIMEOUT_S

    def __init__(self, client):
        super().__init__(client)

    def get_modality_config(self) -> dict:
        return self._client.call(
            "vla.get_modality_config",
            timeout_s=self._TIMEOUT_S["default"],
        )

    def predict(self, obs_dict: dict, options: dict) -> dict:
        """Run inference; returns raw actions dict.

        Actions are numpy arrays.
        """
        return self._client.call(
            "vla.predict",
            args=(obs_dict, options),
            timeout_s=self._TIMEOUT_S["predict"],
        )

    def reset_session(self, session_id: str) -> dict:
        return self._client.call(
            "vla.reset_session",
            args=(session_id,),
            timeout_s=self._TIMEOUT_S["default"],
        )
