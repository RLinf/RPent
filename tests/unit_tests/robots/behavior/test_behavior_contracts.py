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

"""Offline contracts for the BEHAVIOR toolkit."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from robots.behavior.schemas import BEHAVIOR_TOOL_NAMES, MOVE_TO_SPEC
from robots.behavior.toolkit import BehaviorToolkit
from robots.behavior.tools import BehaviorPrimitives
from rpent.dashboard.events import NullDashboardEventSink
from rpent.memory import MemoryManager

EXPECTED_TOOLS = (
    "pi0_nav_pick",
    "observe",
    "pixel_to_world",
    "navigate_to",
    "move_to",
    "rotate_wrist",
    "close",
    "open",
    "press",
)


class _FakeEnv:
    total_env_steps = 0
    official_success_latched = False
    official_success_receipt = None

    def __init__(self) -> None:
        self.last_move: dict[str, Any] | None = None

    def move_to(self, **kwargs: Any) -> dict[str, Any]:
        self.last_move = kwargs
        return {"status": "failed", "stop_reason": "motion_unavailable"}


def _both_hand_request() -> dict[str, Any]:
    return {
        "hand": "both",
        "targets": {
            "left": {"delta_xyz": [0.01, 0.0, 0.0], "frame": "world"},
            "right": {"delta_xyz": [-0.01, 0.0, 0.0], "frame": "eef"},
        },
        "visual_hand_checks": {
            "left": {
                "camera": "left_wrist",
                "frame_id": "left-frame",
                "selected_hand": "left",
                "assessment": "selected_hand_visually_confirmed",
            },
            "right": {
                "camera": "right_wrist",
                "frame_id": "right-frame",
                "selected_hand": "right",
                "assessment": "selected_hand_visually_confirmed",
            },
        },
    }


def test_public_behavior_surface_is_exactly_nine_tools() -> None:
    assert BEHAVIOR_TOOL_NAMES == EXPECTED_TOOLS


def test_move_to_contract_separates_single_and_dual_hand_branches() -> None:
    schema = MOVE_TO_SPEC["input_schema"]
    single, dual = schema["oneOf"]
    assert schema["properties"]["hand"]["enum"] == ["left", "right", "both"]
    assert single["properties"]["hand"] == {"enum": ["left", "right"]}
    assert single["required"] == ["hand", "target"]
    assert dual["properties"]["hand"] == {"const": "both"}
    assert dual["required"] == ["hand", "targets", "visual_hand_checks"]

    env = _FakeEnv()
    primitives = BehaviorPrimitives(env=env, task_name="turning_on_radio")
    for hand in ("left", "right"):
        request = {"hand": hand, "target": {"delta_xyz": [0.0, 0.0, 0.01]}}
        primitives.move_to(**request)
        assert env.last_move == request

    both = _both_hand_request()
    primitives.move_to(**both)
    assert env.last_move == both
    invalid = _both_hand_request()
    invalid["targets"] = {"left": invalid["targets"]["left"]}
    with pytest.raises(ValueError, match="exactly left and right"):
        primitives.move_to(**invalid)


def test_finish_writes_terminal_receipt(tmp_path: Path) -> None:
    output_dir = tmp_path / "run"
    toolkit = BehaviorToolkit(
        primitives_kwargs={
            "task_name": "turning_on_radio",
            "output_dir": output_dir,
        },
        dashboard_events=NullDashboardEventSink(),
        memory=MemoryManager(tmp_path / "memory"),
    )

    result = toolkit.execute_tool(
        "finish", {"status": "incomplete", "summary": "bounded test"}
    )
    receipt = json.loads((output_dir / "terminal_receipt.json").read_text())

    assert result.is_finish is True
    assert receipt["_finish"] is True
    assert receipt["kind"] == "behavior_finish_terminal_receipt"
    assert receipt["planner_status"] == "incomplete"
    assert receipt["summary"] == "bounded test"
