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

"""Unified VLA client base class.
Design reference: ``docs/source-zh/rst_source/development/add_vla.rst``.
"""

from __future__ import annotations


class BaseVLAClient:
    """Unified VLA client base class."""

    _TIMEOUT_S: dict[str, float] = {"default": 30.0, "predict": 120.0}

    def __init__(self, client):
        self._client = client

    def predict(self, obs, options=None):
        """Request a single VLA action chunk.

        Args:
            obs: observation data.
            options: optional dict.

        Returns:
            actions.
        """
        return self._client.call(
            "vla.predict", args=(obs, options), timeout_s=self._TIMEOUT_S["predict"]
        )
