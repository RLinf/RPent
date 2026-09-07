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

import json
import threading
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

from rpent.memory import MemoryManager
from rpent.session import EnvState
from rpent.tools import Toolkit, ToolResult, parallel, readonly, tool


def export(toolkit):
    path = toolkit._task_output_dir / toolkit.write_recipe("cell")
    return [json.loads(line) for line in path.read_text().splitlines()]


@tool
@readonly
@parallel
def sense(label: str, *, ctx) -> ToolResult:
    """Read a sensor."""
    if label == "slow":
        ctx.robot.started.set()
        assert ctx.robot.release.wait(5)
    return ToolResult(data={"label": label})


@tool
@readonly
def finish(status: str, summary: str, *, ctx) -> ToolResult:
    """Finish this session."""
    return ToolResult(data={"status": status, "summary": summary})


def test_generic_parallel_perception_is_recorded_without_new_state_steps(tmp_path):
    runtime = SimpleNamespace(started=threading.Event(), release=threading.Event())
    state = EnvState(tmp_path / "observations")
    toolkit = Toolkit(
        state=state,
        memory=MemoryManager(tmp_path / "memory"),
        robot=runtime,
        output_dir=tmp_path / "recipe",
        tools=(sense, finish),
    )
    try:
        with ThreadPoolExecutor(max_workers=2) as executor:
            slow = executor.submit(toolkit.execute_tool, "sense", {"label": "slow"})
            try:
                assert runtime.started.wait(5)
                fast = executor.submit(toolkit.execute_tool, "sense", {"label": "fast"})
                assert not fast.result(timeout=5).is_error
            finally:
                runtime.release.set()
            assert not slow.result(timeout=5).is_error
        assert state.records() == []
        assert export(toolkit) == [
            {"action": "sense", "label": "fast"},
            {"action": "sense", "label": "slow"},
        ]
    finally:
        toolkit.close()


def test_relative_output_is_bound_at_construction_and_empty_export_is_valid(
    tmp_path, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    toolkit = Toolkit(
        state=EnvState("observations"),
        memory=MemoryManager(tmp_path / "memory"),
        robot=None,
        output_dir="recipes",
        tools=(sense, finish),
    )
    try:
        other = tmp_path / "other"
        other.mkdir()
        monkeypatch.chdir(other)
        assert export(toolkit) == []
        assert (tmp_path / "recipes" / "cell_recipe.jsonl").is_file()
    finally:
        toolkit.close()
