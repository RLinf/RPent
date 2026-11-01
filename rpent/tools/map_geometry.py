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

"""Geometry recorded in ``map.json`` that several map tools consume.

The mapping server records gravity, the floor plane, and the scale provenance
in the model frame. These helpers return that geometry in the form the
consumers need: an up direction, a rotation that puts up at ``+z``, and the
optional metres-per-unit anchor.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np


def rotation_to_z(up: np.ndarray) -> np.ndarray:
    """Return the shortest rotation that maps ``up`` onto ``+z``.

    Points transform as ``points @ rotation.T``.
    """
    axis = np.asarray(up, dtype=np.float64)
    norm = float(np.linalg.norm(axis))
    if not math.isfinite(norm) or norm <= 0.0:
        raise ValueError(f"up direction must be a finite non-zero vector: {up!r}")
    axis = axis / norm
    target = np.asarray([0.0, 0.0, 1.0])
    cross = np.cross(axis, target)
    cosine = float(axis @ target)
    if float(np.linalg.norm(cross)) < 1e-9:
        return np.eye(3) if cosine > 0.0 else np.diag([1.0, -1.0, -1.0])
    skew = np.asarray(
        [
            [0.0, -cross[2], cross[1]],
            [cross[2], 0.0, -cross[0]],
            [-cross[1], cross[0], 0.0],
        ]
    )
    return np.eye(3) + skew + skew @ skew * (1.0 / (1.0 + cosine))


def gravity_up(metadata: dict[str, Any]) -> np.ndarray | None:
    """Return the estimated gravity-up direction in the model frame."""
    block = metadata.get("gravity")
    if not isinstance(block, dict) or block.get("status") != "estimated":
        return None
    values = np.asarray(block.get("up", []), dtype=np.float64)
    if values.shape != (3,) or not np.isfinite(values).all():
        return None
    norm = float(np.linalg.norm(values))
    if norm <= 1e-9:
        return None
    return values / norm


def meters_per_unit(metadata: dict[str, Any]) -> float | None:
    """Return how many metres one model unit is, when the map records it."""
    scale = metadata.get("scale")
    if not isinstance(scale, dict):
        return None
    value = scale.get("meters_per_unit")
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        return None
    return result
