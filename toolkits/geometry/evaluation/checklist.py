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

"""Fixed initial-frame LIBERO geometry checklist: 80 cases and 160 requests."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from toolkits.geometry.part_geometry import PART_DEFINITIONS

REFERENCES = {
    "plate": (
        "盘面中心",
        "the white ceramic plate with concentric red rings around its rim",
        "plate_center",
    ),
    "ramekin": (
        "小烤碗碗口中心",
        "the small light-colored round ramekin with vertically fluted sides",
        "inner_rim_center",
    ),
    "cheese": (
        "奶酪盒顶面中心",
        "the blue-and-white rectangular cream cheese box",
        "rectangle_center",
    ),
    "soup": (
        "汤罐罐顶中心",
        "the cylindrical soup can with a blue-and-yellow label",
        "lid_center",
    ),
    "black_bowl": (
        "黑碗碗口中心",
        "the black-and-white floral-patterned bowl with a thin yellow rim",
        "inner_rim_center",
    ),
    "wine_bottle": (
        "酒瓶瓶身主段中心",
        "the tall dark-green wine bottle with a narrow neck and a gold-colored cap",
        "body_segment_center",
    ),
    "gray_white_mug": (
        "灰白纹理杯杯口中心",
        "the gray-and-white ceramic mug with raised textured sides",
        "inner_rim_center",
    ),
    "yellow_white_mug": (
        "黄白杯杯口中心",
        "the smooth two-tone yellow-and-white mug with a yellow handle",
        "inner_rim_center",
    ),
    "microwave": (
        "微波炉机身顶面中心",
        "the black microwave oven housing, excluding its hinged door",
        "rectangle_center",
    ),
}
TASK_ROWS = (
    ("libero_spatial_task", 0, "plate", "ramekin"),
    ("libero_spatial_task", 4, "plate", "ramekin"),
    ("libero_spatial_swap", 0, "plate", "ramekin"),
    ("libero_spatial_swap", 4, "plate", "ramekin"),
    ("libero_object_task", 0, "cheese", "soup"),
    ("libero_object_task", 4, "cheese", "soup"),
    ("libero_object_swap", 0, "cheese", "soup"),
    ("libero_object_swap", 4, "cheese", "soup"),
    ("libero_goal_task", 0, "plate", "black_bowl"),
    ("libero_goal_task", 7, "plate", "black_bowl"),
    ("libero_goal_swap", 0, "plate", "black_bowl"),
    ("libero_goal_swap", 7, "plate", "black_bowl"),
    ("libero_10_task", 3, "black_bowl", "wine_bottle"),
    ("libero_10_task", 9, "gray_white_mug", "yellow_white_mug"),
    ("libero_10_swap", 3, "black_bowl", "wine_bottle"),
    ("libero_10_swap", 9, "yellow_white_mug", "microwave"),
)


def build_manifest() -> dict[str, Any]:
    """Expand the fixed 16 task rows into 80 cases and 160 reference requests."""
    cases = []
    for suite, task, a, b in TASK_ROWS:
        for seed in range(1, 6):
            case_id = f"{suite}_t{task}_s{seed}"
            cases.append(
                {
                    "case_id": case_id,
                    "suite": suite,
                    "task": task,
                    "seed": seed,
                    "requests": [
                        {
                            "request_id": f"{case_id}/{slot}",
                            "slot": slot,
                            "reference": ref,
                        }
                        for slot, ref in (("A", a), ("B", b))
                    ],
                }
            )
    return {
        "schema_version": 1,
        "protocol": {
            "step": 0,
            "camera": "agentview",
            "units": "m",
            "actions": False,
            "additional_views": False,
        },
        "references": {
            key: {
                "label": label,
                "query": query,
                "quantity": quantity,
                "definition": PART_DEFINITIONS[quantity],
            }
            for key, (label, query, quantity) in REFERENCES.items()
        },
        "counts": {"task_rows": 16, "cases": 80, "requests": 160},
        "cases": cases,
    }


def manifest_digest(manifest: dict[str, Any]) -> str:
    """Hash canonical JSON to bind observations, predictions and annotations."""
    return hashlib.sha256(
        json.dumps(manifest, sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()


def load_manifest(path: Path) -> dict[str, Any]:
    """Reject changed targets, seeds or semantics in the fixed checklist protocol."""
    manifest = json.loads(path.read_text())
    if manifest != build_manifest():
        raise ValueError(
            "manifest does not match the fixed 16-task / seed 1..5 checklist; regenerate with write-manifest"
        )
    return manifest


def select_cases(
    manifest: dict[str, Any],
    case_ids: list[str] | None = None,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    """Select a reproducible pilot subset without changing any reference targets."""
    known = {case["case_id"] for case in manifest["cases"]}
    if case_ids and (len(case_ids) != len(set(case_ids)) or set(case_ids) - known):
        raise ValueError("--case must contain distinct known case IDs")
    if limit is not None and limit <= 0:
        raise ValueError("--limit must be positive")
    selected = [
        case
        for case in manifest["cases"]
        if not case_ids or case["case_id"] in case_ids
    ]
    return selected if limit is None else selected[:limit]


def frame_id(case_id: str) -> str:
    """Name the independent world frame of one fixed initial observation."""
    return f"{case_id}/step0/libero_world"
