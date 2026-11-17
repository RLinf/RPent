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

"""Expose official UniWAM actions in the common WAM result."""

import numpy as np

from rpent.robots.components.am_contracts import WAMPrediction


def decode_robotwin(actions: np.ndarray) -> WAMPrediction:
    """Return native qpos14 actions; the shared client validates the contract."""
    return {"actions": np.asarray(actions, dtype=np.float64)}
