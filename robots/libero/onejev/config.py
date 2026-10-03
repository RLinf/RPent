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
"""Explicit control and RGB-D heuristics for bounded LIBERO tasks."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, fields
from pathlib import Path


@dataclass(frozen=True)
class OneJevConfig:
    """Public-observation settings; lengths are meters and ROIs are normalized."""

    target_roi: tuple[float, float, float, float] | None = None
    workspace: tuple[float, float, float, float, float, float] = (
        -0.45,
        0.45,
        -0.5,
        0.5,
        0.0,
        1.2,
    )
    pick_max_chunks: int = 24
    contact_max_chunks: int = 8
    max_pick_attempts: int = 3
    max_placement_attempts: int = 2
    max_contact_attempts: int = 5
    max_regions: int = 6
    carry_clearance: float = 0.14
    eef_object_offset: float = 0.09
    placement_clearance: float = 0.01
    max_xy_step: float = 0.15
    move_max_steps: int = 80
    move_tolerance: float = 0.012
    min_gripper_opening: float = 0.004
    max_gripper_opening: float = 0.065
    region_min_width: float = 0.05
    region_max_width: float = 0.40
    max_no_progress: int = 3

    @classmethod
    def load(cls, path: str | Path | None) -> OneJevConfig:
        """Load optional JSON settings and validate their physical boundaries."""
        values = {} if path is None else json.loads(Path(path).expanduser().read_text())
        if not isinstance(values, dict):
            raise ValueError("OneJev config must be a JSON object")
        names = {item.name for item in fields(cls)}
        unknown = set(values).difference(names)
        if unknown:
            raise ValueError(f"Unknown OneJev config fields: {sorted(unknown)}")
        for name in ("target_roi", "workspace"):
            if name not in values:
                continue
            if name == "target_roi" and values[name] is None:
                continue
            if not isinstance(values[name], list):
                raise ValueError(f"OneJev {name} must be a JSON array")
            values[name] = tuple(values[name])
        config = cls(**values)
        for name in (
            "pick_max_chunks",
            "contact_max_chunks",
            "max_pick_attempts",
            "max_placement_attempts",
            "max_contact_attempts",
            "max_regions",
            "move_max_steps",
            "max_no_progress",
        ):
            value = getattr(config, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"OneJev {name} must be a positive integer")
        if config.max_regions > 12:
            raise ValueError("OneJev max_regions must not exceed 12")
        for name in names.difference(
            {
                "target_roi",
                "workspace",
                "pick_max_chunks",
                "contact_max_chunks",
                "max_pick_attempts",
                "max_placement_attempts",
                "max_contact_attempts",
                "max_regions",
                "move_max_steps",
                "max_no_progress",
            }
        ):
            value = getattr(config, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value <= 0
            ):
                raise ValueError(f"OneJev {name} must be finite and positive")
        bounds = config.workspace
        if len(bounds) != 6 or not all(
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(value)
            for value in bounds
        ):
            raise ValueError("OneJev workspace must contain six finite numbers")
        if any(bounds[index] >= bounds[index + 1] for index in (0, 2, 4)):
            raise ValueError("OneJev workspace requires min < max on every axis")
        roi = config.target_roi
        if roi is not None and (
            len(roi) != 4
            or not all(
                isinstance(v, (int, float))
                and not isinstance(v, bool)
                and math.isfinite(v)
                and 0 <= v <= 1
                for v in roi
            )
            or roi[0] >= roi[2]
            or roi[1] >= roi[3]
        ):
            raise ValueError(
                "OneJev target_roi must be [row_min, col_min, row_max, col_max] in [0, 1]"
            )
        if not 0 < config.min_gripper_opening < config.max_gripper_opening < 0.08:
            raise ValueError(
                "OneJev gripper thresholds must be ordered within (0, 0.08)"
            )
        if config.max_xy_step > 0.25:
            raise ValueError("OneJev max_xy_step must not exceed 0.25 m")
        if config.region_min_width >= config.region_max_width:
            raise ValueError("OneJev region width bounds must be ordered")
        return config
