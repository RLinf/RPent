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

"""Thin client wrapping the APXInf VLA RPC server.

The server lifecycle is the caller's responsibility: bring up
``rpent.robots.components.apxinf_vla_server`` before constructing this client.

Obs encoding mirrors ``Pi05VLAClient``: the same ``_ENCODE_OBS`` registry is
reused so the wire format sent over RPC is identical to the openpi path.
The APXInf server decodes it internally into the APXInf observation dict.
"""

from __future__ import annotations

import numpy as np

from rpent.robots.components.pi05_vla_client import _ENCODE_OBS
from rpent.robots.components.vla_client_base import BaseVLAClient


class ApxInfVLAClient(BaseVLAClient):
    """Client wrapping the APXInf VLA over any :class:`RpcClient` transport.

    Construction requires an ``embodiment`` name (e.g. ``"libero"``) that
    selects the obs encoder from the shared ``_ENCODE_OBS`` registry. The
    wire format is the same openpi batched format used by ``Pi05VLAClient``;
    the server performs the APXInf-specific decode.
    """

    def __init__(self, client, *, embodiment: str):
        super().__init__(client)
        if embodiment != "libero":
            raise ValueError(
                f"unknown apxinf client embodiment: {embodiment!r}; "
                "registered=['libero']"
            )
        self._embodiment = embodiment

    def encode_obs(self, env_obs: dict) -> dict:
        """Dispatch to the embodiment's encoder from the shared registry."""
        return _ENCODE_OBS[self._embodiment](env_obs)

    def predict(self, env_obs: dict, options: dict | None = None) -> np.ndarray:
        """Encode obs, request ``vla.predict``, strip batch dim, return ``[chunk, action_dim]``."""
        wire_obs = self.encode_obs(env_obs)
        actions = np.asarray(super().predict(wire_obs, options))
        return actions[0]
