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

"""Shared base for perception helpers used by robot tools."""

from __future__ import annotations


class PerceptionToolBase:
    """Base for perception helpers that robot tools call on recorded images.

    A perception tool owns one model client and the robot-independent work
    around it. Robot tools keep their own ``@tool`` declarations and pass in
    the run state and the artifacts to read.
    """

    def __init__(self, client: object) -> None:
        """Keep the model client; the caller owns its service lifecycle."""
        if client is None:
            raise ValueError(f"{type(self).__name__} requires a model client")
        self._client = client
