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
    "back_project_batch",
    "query_world_map",
}


def _record(step_idx: int = 0) -> SimpleNamespace:
    return SimpleNamespace(step_idx=step_idx, terminated=False, extras={})


def _tool_names(robot_toolkit: Toolkit) -> set[str]:
    return {spec["name"] for spec in robot_toolkit.get_tools_spec()}


def _readonly_names(robot_toolkit: Toolkit) -> set[str]:
    return {
        name
        for name, (_, handler) in robot_toolkit._tools.items()
        if _is_readonly(handler)
    }


def test_toolkit_falls_back_to_memory_root(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    make_corpus,
) -> None:
    memory_dir = tmp_path / "robocasa"
    make_corpus(memory_dir)
    (tmp_path / "run").mkdir()
    monkeypatch.setattr(robot_spec, "get_memory_dir", lambda _: memory_dir)
    monkeypatch.setattr(
        toolkit,
        "RoboCasaToolkit",
        lambda **kwargs: SimpleNamespace(**kwargs),
    )
    config = RunConfig(
        recipe_tag="cell-s0",
        output_dir=tmp_path / "run",
        prompt_vars={},
        task_desc={"task_name": "OpenDrawer"},
    )

    robot_toolkit = robot_spec.get_toolkit(
        runtime_kwargs={},
        dashboard_events=NullDashboardEventSink(),
        config=config,
    )

    assert robot_toolkit.memory.root == memory_dir.resolve()


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
        runtime_kwargs={"env_client": object(), "vla_client": object()},
        dashboard_events=NullDashboardEventSink(),
        memory=MemoryManager(tmp_path / "memory"),
    )

    assert _tool_names(robot_toolkit) == EXPECTED_TOOLS
    assert _readonly_names(robot_toolkit) == COMMON_TOOLS | {
        "view_env_state",
        "back_project_batch",
        "query_world_map",
    }
    assert dumped == fake_single_arm_primitives.instances
    primitive = fake_single_arm_primitives.instances[0]
    assert primitive.reset_calls == 1
    assert primitive.recording_started is True
    assert callable(primitive.kwargs["check_cancelled"])
    assert (tmp_path / "success_criteria.md").read_text() == "offline success criteria"


def test_solved_reads_only_the_final_environment_record() -> None:
    records = [SimpleNamespace(extras={"success": True})]
    robot_toolkit = toolkit.RoboCasaToolkit.__new__(toolkit.RoboCasaToolkit)
    robot_toolkit._state = SimpleNamespace(latest_record=lambda: records[-1])

    assert robot_toolkit.solved() is True

    records.append(SimpleNamespace(extras={"success": False}))
    assert robot_toolkit.solved() is False


@pytest.mark.parametrize("policy", ["task-global", "task-only"])
@pytest.mark.parametrize("allow_reset", [False, True])
def test_reset_requires_complete_memory_and_preserves_no_reset(
    tmp_path, monkeypatch, make_corpus, policy, allow_reset
):
    from robots.robocasa.memory import RoboCasaMemoryManager, TaskMemory
    from robots.robocasa.primitives import RoboCasaPrimitives
    from robots.robocasa.tools import TOOLS_SPEC
    from rpent.session import EnvState

    root = make_corpus(tmp_path / "robocasa")
    memory = RoboCasaMemoryManager(TaskMemory.load(root, "OpenDrawer", policy=policy))
    robot_toolkit = toolkit.RoboCasaToolkit.__new__(toolkit.RoboCasaToolkit)
    Toolkit.__init__(
        robot_toolkit,
        dashboard_events=NullDashboardEventSink(),
        memory=memory,
        state=EnvState(tmp_path / "run"),
    )
    resets = []
    primitive = RoboCasaPrimitives.__new__(RoboCasaPrimitives)
    primitive._allow_reset = allow_reset
    primitive.env = SimpleNamespace(
        reset=lambda: resets.append("env"),
        eef_pos=SimpleNamespace(tolist=lambda: [0, 0, 0]),
    )
    primitive._rldx = SimpleNamespace(reset_session=lambda: resets.append("vla"))
    robot_toolkit.add_tool(
        "reset",
        next(spec for spec in TOOLS_SPEC if spec["name"] == "reset"),
        primitive.reset,
    )
    monkeypatch.setattr(
        robot_toolkit, "get_env_state", lambda **kwargs: kwargs["result"]
    )

    assert (
        "Read the selected memory"
        in robot_toolkit.execute_tool("reset", {}).result["error"]
    )
    read = memory.get_common_tool_bindings()["read_text_file"][1]
    read(path=str(root / memory.selection.selected[0]), max_chars=1)
    assert (
        "Read the selected memory"
        in robot_toolkit.execute_tool("reset", {}).result["error"]
    )
    assert resets == []

    for name in memory.selection.selected:
        read(path=str(root / name))
    result = robot_toolkit.execute_tool("reset", {}).result
    if allow_reset:
        assert result["reset"] is True
        assert resets == ["env", "vla"]
    else:
        assert "reset is DISABLED" in result["error"]
        assert resets == []
