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

"""Offline contracts for the LIBERO toolkit."""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from robots.libero import robot_spec, toolkit
from rpent.dashboard.events import NullDashboardEventSink
from rpent.memory import MemoryManager
from rpent.robots import RunConfig
from rpent.robots.components.sam3_client import Sam3Result
from rpent.session import EnvState
from rpent.tools import Toolkit, ToolResult
from rpent.tools.common import CommonTools
from rpent.utils import templates

COMMON_TOOLS = {"read_text_file", "write_text_file", "list_dir", "finish"}

EVALUATION_TOOLS = COMMON_TOOLS | {
    "view_env_state",
    "move_to",
    "pi0_pick",
    "pi0_doubled",
    "release",
    "set_gripper",
    "rotate_wrist",
    "rotate_pitch",
    "move_pose",
    "view_camera_meta",
    "segment",
    "back_project",
}


def _record(step_idx: int = 0) -> SimpleNamespace:
    return SimpleNamespace(step_idx=step_idx, terminated=False)


def _tool_names(robot_toolkit: Toolkit) -> set[str]:
    return {definition.name for definition in robot_toolkit.list_tools()}


def _readonly_names(robot_toolkit: Toolkit) -> set[str]:
    return {
        definition.name
        for definition in robot_toolkit.list_tools()
        if definition.readonly
    }


def _run_config(memory_dir: Path, *, recipe_tag: str = "cell-s0") -> RunConfig:
    return RunConfig(
        recipe_tag=recipe_tag,
        output_dir=memory_dir.parent / "run",
        prompt_vars={"memory_dir": str(memory_dir)},
        task_desc={},
    )


def test_toolkit_factory_configures_memory_access_by_mode(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    captured: list[dict[str, Any]] = []

    def fake_toolkit(**kwargs: Any) -> SimpleNamespace:
        captured.append(kwargs)
        return SimpleNamespace(**kwargs)

    monkeypatch.setattr(toolkit, "LiberoToolkit", fake_toolkit)
    memory_dir = tmp_path / "libero-memory"
    config = _run_config(memory_dir)

    evaluation = robot_spec.get_toolkit(
        runtime_kwargs={"env": "evaluation"},
        dashboard_events=NullDashboardEventSink(),
        config=config,
    )
    exploration = robot_spec.get_toolkit(
        runtime_kwargs={"env": "exploration"},
        dashboard_events=NullDashboardEventSink(),
        config=config,
        mode="exploration",
        attempts_per_session=2,
        state_output_dir=tmp_path / "state",
    )

    assert evaluation.memory.root == memory_dir.resolve()
    assert exploration.memory.root == memory_dir.resolve()
    evaluation_write = CommonTools(memory=evaluation.memory).write_text_file
    exploration_write = CommonTools(memory=exploration.memory).write_text_file
    own_draft = memory_dir / "_internal" / "inbox" / config.recipe_tag / "draft.md"
    with pytest.raises(PermissionError, match="writing to memory is denied"):
        evaluation_write(str(own_draft), "draft")
    assert exploration_write(str(own_draft), "draft").data["bytes_written"] == 5
    assert captured[0]["mode"] == "evaluation"
    assert captured[1]["mode"] == "exploration"
    assert captured[1]["attempts_per_session"] == 2


def test_toolkit_modes_construct_with_fake_primitives(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    fake_single_arm_primitives: type[Any],
) -> None:
    fake_single_arm_primitives = fake_single_arm_primitives.for_robot(
        toolkit.libero_tools.LiberoPrimitives
    )
    dumped: list[Any] = []
    monkeypatch.setattr(
        templates, "default_variables", lambda: {"output_dir": "/offline/output"}
    )
    monkeypatch.setattr(
        toolkit.libero_tools,
        "LiberoPrimitives",
        fake_single_arm_primitives,
    )
    monkeypatch.setattr(
        toolkit.libero_tools,
        "dump_state",
        lambda primitives, state, log: dumped.append(primitives) or _record(),
    )

    evaluation = toolkit.LiberoToolkit(
        runtime_kwargs={"env_client": object()},
        dashboard_events=NullDashboardEventSink(),
        memory=MemoryManager(tmp_path / "evaluation-memory"),
        mode="evaluation",
        state_output_dir=tmp_path / "evaluation",
    )
    exploration = toolkit.LiberoToolkit(
        runtime_kwargs={"env_client": object()},
        dashboard_events=NullDashboardEventSink(),
        memory=MemoryManager(
            tmp_path / "exploration-memory",
            memory_access="inbox_write",
            inbox_cell_tag="offline-cell",
        ),
        mode="exploration",
        attempts_per_session=3,
        state_output_dir=tmp_path / "exploration",
    )

    assert _tool_names(evaluation) == EVALUATION_TOOLS
    assert _tool_names(exploration) == EVALUATION_TOOLS | {"reset"}
    assert _readonly_names(evaluation) == COMMON_TOOLS | {
        "view_env_state",
        "view_camera_meta",
        "back_project",
        "segment",
    }
    assert _readonly_names(exploration) == _readonly_names(evaluation)
    assert len(dumped) == 2
    assert all(
        instance.reset_calls == 1 for instance in fake_single_arm_primitives.instances
    )
    assert all(
        instance.recording_started for instance in fake_single_arm_primitives.instances
    )
    assert all(
        callable(instance.kwargs["check_cancelled"])
        for instance in fake_single_arm_primitives.instances
    )

    refused = exploration.execute_tool(
        "finish", {"status": "failure", "summary": "first attempt"}
    )
    assert refused.data["error"] == "finish refused"
    assert refused.data.get("_finish", False) is False

    exploration.get_env_state = lambda *, command, result, elapsed_s: ToolResult(
        data=dict(result)
    )
    assert (
        exploration.execute_tool("reset", {"reason": "new approach"}).data["attempt"]
        == 2
    )
    assert (
        exploration.execute_tool("reset", {"reason": "third approach"}).data["attempt"]
        == 3
    )
    allowed = exploration.execute_tool(
        "finish", {"status": "failure", "summary": "budget spent"}
    )
    assert allowed.data.get("_finish", False) is True


def test_parallel_segments_preserve_artifacts_without_capturing_state(
    tmp_path, monkeypatch
):
    state = EnvState(tmp_path / "state")
    with state.record_step(state={}):
        state.save("agentview.png", np.zeros((4, 4, 3), dtype=np.uint8))
        state.save("agentview_world.npz", np.ones((4, 4, 3)))
    calls = 4
    inference_barrier = threading.Barrier(calls, timeout=5)
    projection_barrier = threading.Barrier(calls, timeout=5)
    project = toolkit.libero_tools._mask_to_world

    def project_together(mask, world_map):
        projection_barrier.wait()
        return project(mask, world_map)

    monkeypatch.setattr(toolkit.libero_tools, "_mask_to_world", project_together)

    def segment(image, **kwargs):
        # Every inference must enter before any returns: segment stays readonly.
        inference_barrier.wait()
        return Sam3Result(found=True, mask=np.ones((4, 4), dtype=bool), score=1.0)

    primitives = toolkit.libero_tools.LiberoPrimitives(
        env=object(),
        model=object(),
        sam3_client=SimpleNamespace(segment=segment),
        check_cancelled=lambda: None,
    )
    robot_toolkit = Toolkit(
        dashboard_events=NullDashboardEventSink(),
        state=state,
        memory=MemoryManager(tmp_path / "memory"),
    )
    definition = primitives.segment
    robot_toolkit.add_tool(definition.with_handler(partial(definition, state=state)))
    captured = []

    def capture(**kwargs):
        captured.append(kwargs)
        return ToolResult(data={})

    robot_toolkit.get_env_state = capture
    prompts = [f"object {index}" for index in range(calls)]
    with ThreadPoolExecutor(max_workers=calls) as pool:
        results = list(
            pool.map(
                lambda prompt: robot_toolkit.execute_tool(
                    "segment", {"prompt": prompt}
                ),
                prompts,
            )
        )

    assert captured == []
    assert len(state.records()) == 1
    assert all(not result.is_error for result in results)
    assert {result.data["segment_artifact"] for result in results} == {
        f"segment_{index:02d}.json" for index in range(calls)
    }
    for prompt, result in zip(prompts, results):
        artifact = state.load(result.data["segment_artifact"], step=0)
        assert artifact["prompt"] == prompt
        assert artifact["source_step"] == 0
        assert result.images == [
            state.load_bytes(result.data["overlay_artifact"], step=0)
        ]

    # A failed JSON write must not release an index for another call to reuse.
    monkeypatch.setattr(toolkit.libero_tools, "_mask_to_world", project)
    primitives._sam3_client.segment = lambda *args, **kwargs: Sam3Result(
        found=True, mask=np.ones((4, 4), dtype=bool), score=1.0
    )
    save = state.save

    def fail_segment_save(name, value, **kwargs):
        if name == f"segment_{calls:02d}.json":
            return None
        return save(name, value, **kwargs)

    monkeypatch.setattr(state, "save", fail_segment_save)
    failed = robot_toolkit.execute_tool("segment", {"prompt": "failed"})
    assert failed.data["code"] == "segment_artifact_save_failed"
    result = robot_toolkit.execute_tool("segment", {"prompt": "next"})
    assert result.data["segment_artifact"] == f"segment_{calls + 1:02d}.json"
    assert state.load(result.data["segment_artifact"], step=0)["prompt"] == "next"
