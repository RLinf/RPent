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

"""Dashboard primitive traces remain valid inputs to LIBERO Flash replay."""

from __future__ import annotations

import argparse
import json
from types import SimpleNamespace

from robots.libero import robot_spec
from robots.libero.flash.generate import generate_flash_plan
from robots.libero.flash.replay import run_flash
from robots.libero.tools import LiberoPrimitives, write_recipe_from_states
from rpent.dashboard.events import RunStartedEvent
from rpent.dashboard.state import DashboardState
from rpent.memory import MemoryManager
from rpent.session import EnvState
from rpent.tools import Toolkit, ToolResult


def test_manual_recipe_replays_through_flash_without_provenance_arguments(
    tmp_path, monkeypatch
):
    monkeypatch.setattr("rpent.utils.logging._output_dir", tmp_path)

    class RecordingToolkit(Toolkit):
        def __init__(self, path, sink, memory):
            self.calls = []
            self.primitives = SimpleNamespace(molmo_client=object())
            super().__init__(dashboard_events=sink, state=EnvState(path), memory=memory)
            self.add_tool(LiberoPrimitives.release.with_handler(self.release))
            with self.state.record_step(state={}):
                pass

        def release(self, **arguments):
            self.calls.append(arguments)
            return ToolResult(data={"released": True})

        def get_env_state(self, *, command, result, elapsed_s):
            with self.state.record_step(
                state={},
                command=command,
                result=result,
                elapsed_s=elapsed_s,
                terminated=result.get("released", False),
            ):
                pass
            return ToolResult(data={"terminated": self.solved()})

        def solved(self):
            return self.state.latest_record().terminated

    spec = robot_spec.get_robot_spec().resolve_dashboard(argparse.Namespace())
    dashboard = DashboardState(output_dir=tmp_path / "dashboard", dashboard_spec=spec)
    dashboard.shared_services_ready()
    dashboard.submit_input("/rpent-task libero_goal_task 0 1")
    assert dashboard.wait_for_task(timeout=0) is not None
    dashboard.emit(RunStartedEvent())
    memory = MemoryManager(tmp_path / "memory")
    recording = RecordingToolkit(tmp_path / "recording", dashboard, memory)
    dashboard.bind_toolkit(recording)
    dashboard.set_planner_activity("idle", accepting_input=True)
    try:
        result = dashboard.execute_primitive("release", {"max_steps": 2})
        assert not result.is_error
        assert recording.state.latest_record().command == {
            "action": "release",
            "max_steps": 2,
        }
        tag = "goal_task_t0_s1"
        recipe = tmp_path / write_recipe_from_states(
            recording.state, tag, output_dir=tmp_path
        )
        audit = tmp_path / f"{tag}.json"
        audit.write_text(
            json.dumps(
                {"libero_terminated": True, "task_language": "open the top drawer"}
            )
        )
        generate_flash_plan(audit, recipe, memory.root / "task_card")
    finally:
        dashboard.unbind_toolkit(recording)
        recording.close()

    replay = RecordingToolkit(tmp_path / "replay", dashboard, memory)
    try:
        result = run_flash(replay, tag)
        assert result["done"] and result["plan"] == 1 and result["anchors"] == 0
        assert replay.calls == [{"max_steps": 2}]
    finally:
        replay.close()
