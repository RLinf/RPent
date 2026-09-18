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

"""Environment-owned specifications for flat native action vectors."""

from __future__ import annotations

import json
from typing import Any

import numpy as np


def box_action_spec(low: Any, high: Any, description: str) -> dict[str, Any]:
    """Serialize native per-coordinate bounds, using None for unbounded values."""
    low, high = np.asarray(low, dtype=float), np.asarray(high, dtype=float)
    if low.ndim != 1 or not low.size or high.shape != low.shape:
        raise ValueError("action bounds must be nonempty vectors of equal shape")
    if np.isnan(low).any() or np.isnan(high).any() or (low > high).any():
        raise ValueError("invalid environment action bounds")
    return {
        "description": description,
        "low": [float(x) if np.isfinite(x) else None for x in low],
        "high": [float(x) if np.isfinite(x) else None for x in high],
    }


def validate_action(values: Any, spec: dict[str, Any]) -> np.ndarray:
    """Validate one native action against the active environment's bounds."""
    action = np.asarray(values, dtype=float)
    low = np.asarray([x if x is not None else -np.inf for x in spec["low"]])
    high = np.asarray([x if x is not None else np.inf for x in spec["high"]])
    if action.shape != low.shape or not np.isfinite(action).all():
        raise ValueError(f"values must contain {len(low)} finite numbers")
    if (action < low).any() or (action > high).any():
        raise ValueError("action values are outside the environment's bounds")
    return action


def direct_action_tool_spec(specs: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Build the planner tool from the connected environment's native layouts.

    The ``default`` layout has no action_type argument. Typed environments
    advertise their supported modes as keys, with the first mode the default.
    """
    sizes = [len(spec["low"]) for spec in specs.values()]
    properties = {
        "values": {
            "type": "array",
            "items": {"type": "number"},
            "minItems": min(sizes),
            "maxItems": max(sizes),
        },
    }
    if list(specs) != ["default"]:
        properties["action_type"] = {
            "type": "string",
            "enum": list(specs),
            "default": next(iter(specs)),
        }
    return {
        "name": "execute_action",
        "description": (
            "Execute one native environment action alongside VLA and scripted primitives. "
            "Use the connected environment's layouts and per-coordinate bounds below "
            "(null means unbounded). " + json.dumps(specs, allow_nan=False)
        ),
        "input_schema": {
            "type": "object",
            "properties": properties,
            "required": ["values"],
        },
    }
