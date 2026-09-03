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

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pytest

from robots.behavior.rlinf_env import OfficialBehaviorBackend
from robots.behavior.robot_spec import get_robot_spec
from robots.behavior.schemas import (
    BEHAVIOR_TOOL_NAMES,
    CURRENT_PUBLIC_TOOL_CONTRACT_VERSION,
    MOVE_TO_SPEC,
    PUBLIC_PRIMITIVE_ENTRYPOINTS,
    behavior_tool_specs_for_task,
)
from robots.behavior.toolkit import BehaviorToolkit
from robots.behavior.tools import BehaviorPrimitives
from rpent.dashboard.events import NullDashboardEventSink
from rpent.memory import MemoryManager
from rpent.tools.toolkit import ToolResult

EXPECTED_PRIMITIVES = (
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

    def observe(self, **_kwargs: Any) -> dict[str, Any]:
        return {"status": "ok"}

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


def test_public_behavior_surface_is_exactly_nine_primitives() -> None:
    assert CURRENT_PUBLIC_TOOL_CONTRACT_VERSION == 5
    assert BEHAVIOR_TOOL_NAMES == EXPECTED_PRIMITIVES
    assert tuple(PUBLIC_PRIMITIVE_ENTRYPOINTS) == EXPECTED_PRIMITIVES
    assert (
        tuple(spec["name"] for spec in behavior_tool_specs_for_task("turning_on_radio"))
        == EXPECTED_PRIMITIVES
    )


def test_move_to_schema_has_distinct_single_and_dual_hand_branches() -> None:
    schema = MOVE_TO_SPEC["input_schema"]
    assert schema["properties"]["hand"]["enum"] == ["left", "right", "both"]
    assert "plan_only" not in schema["properties"]
    assert "prepared_plan_id" not in schema["properties"]
    assert len(schema["oneOf"]) == 2
    assert schema["oneOf"][1]["properties"]["hand"] == {"const": "both"}
    assert schema["oneOf"][1]["required"] == [
        "hand",
        "targets",
        "visual_hand_checks",
    ]


def test_move_to_both_validates_and_uses_the_single_env_entrypoint() -> None:
    env = _FakeEnv()
    primitives = BehaviorPrimitives(env=env, task_name="turning_on_radio")

    result = primitives.move_to(**_both_hand_request())

    assert result["name"] == "move_to"
    assert env.last_move == _both_hand_request()
    invalid = _both_hand_request()
    invalid["targets"] = {"left": invalid["targets"]["left"]}
    with pytest.raises(ValueError, match="exactly left and right"):
        primitives.move_to(**invalid)


def test_rlinf_move_to_dispatches_dual_hand_requests() -> None:
    backend = object.__new__(OfficialBehaviorBackend)
    routed: list[tuple[str, dict[str, Any]]] = []
    backend._move_single_hand_to = lambda request: routed.append(("single", request))
    backend._move_both_hands_to = lambda request: routed.append(("both", request))

    backend.move_to(hand="left", target={})
    backend.move_to(hand="both", targets={})

    assert [name for name, _request in routed] == ["single", "both"]


def test_behavior_toolkit_returns_the_shared_tool_result(tmp_path: Path) -> None:
    output_dir = tmp_path / "run"
    toolkit = BehaviorToolkit(
        primitives_kwargs={
            "env": _FakeEnv(),
            "task_name": "turning_on_radio",
            "output_dir": output_dir,
        },
        dashboard_events=NullDashboardEventSink(),
        memory=MemoryManager(tmp_path / "memory"),
    )

    result = toolkit.execute_tool("observe", {"camera": "head"})

    assert type(result) is ToolResult


def test_finish_still_writes_the_terminal_receipt(tmp_path: Path) -> None:
    output_dir = tmp_path / "run"
    toolkit = BehaviorToolkit(
        primitives_kwargs={
            "env": _FakeEnv(),
            "task_name": "turning_on_radio",
            "output_dir": output_dir,
        },
        dashboard_events=NullDashboardEventSink(),
        memory=MemoryManager(tmp_path / "memory"),
    )

    result = toolkit.execute_tool(
        "finish", {"status": "incomplete", "summary": "bounded test"}
    )

    assert type(result) is ToolResult
    assert result.is_finish is True
    assert (output_dir / "terminal_receipt.json").is_file()


@pytest.mark.parametrize(
    "field",
    [
        "_image_bytes",
        "_depth_image_bytes",
        "_image_left_wrist_bytes",
        "_depth_left_wrist_bytes",
        "_image_right_wrist_bytes",
        "_depth_right_wrist_bytes",
    ],
)
def test_tool_result_preserves_non_bytes_error_semantics(field: str) -> None:
    with pytest.raises(TypeError):
        ToolResult("observe", {field: "not-bytes"})


@pytest.mark.parametrize(
    ("mode", "task_name", "public_seed"),
    [("eval", "turning_on_radio", 1), ("explore", "picking_up_trash", 0)],
)
def test_behavior_prompt_has_only_the_nine_peer_primitives(
    tmp_path: Path,
    mode: str,
    task_name: str,
    public_seed: int,
) -> None:
    spec = get_robot_spec()
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir")
    spec.add_cli_args(parser, use_dashboard=False)
    args = parser.parse_args(
        [
            "--task-name",
            task_name,
            "--public-seed",
            str(public_seed),
            "--behavior-mode",
            mode,
            "--output-dir",
            str(tmp_path / mode),
        ]
    )
    config = spec.parse_config(args)
    rendered = "\n".join(
        (
            spec.prompts.render("system", variables=config.prompt_vars),
            spec.prompts.render("user", variables=config.prompt_vars),
        )
    )
    lowered = rendered.lower()

    assert str(list(EXPECTED_PRIMITIVES)) in rendered
    assert "unordered peer tools" in rendered
    assert "hand=both" in rendered
    assert lowered.count("`finish`") == 1
    forbidden = (
        "save_" + "robot_state_checkpoint",
        "move_" + "both_to",
        "get_" + "prepared_motion_status",
        "restore_" + "robot_state_checkpoint",
        "reset",
        "inspect_",
        "post_" + "pick_",
        "post_" + "success_",
        "held_" + "wrist",
        "press_" + "wrist",
        "first",
        "exactly once",
        "stage",
        "pre/post",
    )
    assert not [item for item in forbidden if item in lowered]
