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

"""Offline contracts for the RoboCasa toolkit."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from robots.robocasa import robot_spec, toolkit
from rpent.dashboard.events import NullDashboardEventSink
from rpent.memory import MemoryManager
from rpent.robots import RunConfig
from rpent.tools.toolkit import Toolkit, _is_readonly
from rpent.utils import templates

COMMON_TOOLS = {"read_text_file", "write_text_file", "list_dir", "finish"}

EXPECTED_TOOLS = COMMON_TOOLS | {
    "move_to",
    "move_delta",
    "rotate_pitch",
    "set_gripper",
    "release",
    "scripted_grasp",
    "rldx_skill",
    "rldx_arm",
    "navigate_to",
    "move_base",
    "reset",
    "view_env_state",
    "view_camera_meta",
    "back_project",
    "back_project_batch",
    "query_world_map",
}


def _record(step_idx: int = 0) -> SimpleNamespace:
    return SimpleNamespace(step_idx=step_idx, terminated=False)


def _tool_names(robot_toolkit: Toolkit) -> set[str]:
    return {spec["name"] for spec in robot_toolkit.get_tools_spec()}


def _readonly_names(robot_toolkit: Toolkit) -> set[str]:
    return {
        name
        for name, (_, handler) in robot_toolkit._tools.items()
        if _is_readonly(handler)
    }


def test_toolkit_falls_back_to_resource_memory_root(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    resources_dir = tmp_path / "robocasa"
    monkeypatch.setattr(robot_spec, "get_resources_dir", lambda _: resources_dir)
    monkeypatch.setattr(
        toolkit,
        "RoboCasaToolkit",
        lambda **kwargs: SimpleNamespace(**kwargs),
    )
    config = RunConfig(
        recipe_tag="cell-s0",
        output_dir=tmp_path / "run",
        prompt_vars={},
        task_desc={},
    )

    robot_toolkit = robot_spec.get_toolkit(
        primitives_kwargs={},
        dashboard_events=NullDashboardEventSink(),
        config=config,
    )

    assert robot_toolkit.memory.root == (resources_dir / "memory").resolve()


def test_toolkit_constructs_and_classifies_tools_with_a_fake(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    fake_single_arm_primitives: type[Any],
) -> None:
    monkeypatch.delenv("RLDX_MAX_CHUNKS", raising=False)
    monkeypatch.delenv("RLDX_SETTLE_PATIENCE", raising=False)
    import robots.robocasa.primitives as primitives_module

    dumped: list[Any] = []
    monkeypatch.setattr(
        templates, "default_variables", lambda: {"output_dir": "/offline/output"}
    )
    monkeypatch.setattr(
        primitives_module,
        "RoboCasaPrimitives",
        fake_single_arm_primitives,
    )
    monkeypatch.setattr(toolkit, "get_output_dir", lambda: tmp_path)
    monkeypatch.setattr(
        toolkit.robocasa_tools,
        "dump_state",
        lambda primitives, state, log: dumped.append(primitives) or _record(),
    )

    robot_toolkit = toolkit.RoboCasaToolkit(
        primitives_kwargs={"env_client": object(), "vla_client": object()},
        dashboard_events=NullDashboardEventSink(),
        memory=MemoryManager(tmp_path / "memory"),
    )

    assert _tool_names(robot_toolkit) == EXPECTED_TOOLS
    assert _readonly_names(robot_toolkit) == COMMON_TOOLS | {
        "view_env_state",
        "view_camera_meta",
        "back_project",
        "back_project_batch",
        "query_world_map",
    }
    assert dumped == fake_single_arm_primitives.instances
    primitive = fake_single_arm_primitives.instances[0]
    assert primitive.reset_calls == 1
    assert primitive.recording_started is True
    assert callable(primitive.kwargs["check_cancelled"])
    assert (tmp_path / "success_criteria.md").read_text() == "offline success criteria"
