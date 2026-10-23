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

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from robots.behavior.toolkit import BehaviorToolkit
from rpent.dashboard.events import NullDashboardEventSink
from rpent.memory import MemoryManager
from rpent.robots import RunConfig


def test_finish_writes_terminal_receipt(tmp_path: Path) -> None:
    output_dir = tmp_path / "run"
    toolkit = BehaviorToolkit(
        runtime_kwargs={
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

    assert result.data["_finish"] is True
    assert receipt["_finish"] is True
    assert receipt["kind"] == "behavior_finish_terminal_receipt"
    assert receipt["planner_status"] == "incomplete"
    assert receipt["summary"] == "bounded test"


def test_receipt_is_session_artifact_and_recipe_is_run_artifact(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    session_dir = run_dir / "sessions" / "session_001"
    toolkit = BehaviorToolkit(
        runtime_kwargs={
            "task_name": "turning_on_radio",
            "output_dir": run_dir,
        },
        dashboard_events=NullDashboardEventSink(),
        memory=MemoryManager(tmp_path / "memory"),
        config=RunConfig(
            recipe_tag="turning_on_radio_s0",
            output_dir=run_dir,
            prompt_vars={
                "task_name": "turning_on_radio",
                "public_seed": 0,
                "memory_dir": str(tmp_path / "memory"),
            },
            task_desc={},
        ),
        state_output_dir=session_dir,
    )
    run_audit = run_dir / "turning_on_radio_s0.json"
    run_audit.parent.mkdir(parents=True, exist_ok=True)
    run_audit.write_text('{"audit": true}\n')

    toolkit.execute_tool("finish", {"status": "incomplete", "summary": "done"})
    assert (session_dir / "terminal_receipt.json").is_file()
    assert json.loads((session_dir / "states.json").read_text())["run_artifacts"] == [
        "terminal_receipt.json"
    ]
    assert run_audit.read_text() == '{"audit": true}\n'

    toolkit.primitives._official_success_latched = True
    with toolkit.state.record_step(
        state={"task_success": True},
        terminated=True,
        command={"action": "future_stateful_command", "arg": 1},
        result={"ok": True},
        elapsed_s=0.1,
    ):
        pass
    with toolkit.state.record_step(
        state={"task_success": True},
        terminated=True,
        command={"action": "bad_command"},
        result={"error": "failed"},
        elapsed_s=0.1,
    ):
        pass

    recipe_path = Path(toolkit.write_recipe("turning_on_radio_s0") or "")
    assert recipe_path == run_dir / "turning_on_radio_s0_recipe.jsonl"
    assert not (session_dir / "turning_on_radio_s0_recipe.jsonl").exists()
    commands = [
        json.loads(line)
        for line in recipe_path.read_text().splitlines()
        if line.strip()
    ]
    assert commands == [{"action": "future_stateful_command", "arg": 1}]
    assert "command" not in commands[0]


def test_eval_recipe_preserves_state_manifest(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    toolkit = BehaviorToolkit(
        runtime_kwargs={
            "task_name": "turning_on_radio",
            "output_dir": run_dir,
            "initial_info": {"done": {"success": True}},
        },
        memory=MemoryManager(tmp_path / "memory"),
    )
    with toolkit.state.record_step(
        state={"task_success": True},
        command={"action": "pi0_nav_pick", "instruction": "turn on radio"},
        result={"ok": True},
    ):
        pass
    toolkit.execute_tool("finish", {"status": "complete", "summary": "done"})

    assert toolkit.write_recipe("turning_on_radio_s0") is not None
    manifest = json.loads((run_dir / "states.json").read_text())
    assert len(manifest["steps"]) == 1
    assert manifest["steps"][0]["command"]["action"] == "pi0_nav_pick"
    assert manifest["run_artifacts"] == [
        "terminal_receipt.json",
        "turning_on_radio_s0_recipe.jsonl",
    ]


def test_solved_behavior_session_without_terminal_receipt_refuses_recipe(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    toolkit = BehaviorToolkit(
        runtime_kwargs={"task_name": "turning_on_radio", "output_dir": run_dir},
        memory=MemoryManager(tmp_path / "memory"),
        state_output_dir=run_dir / "sessions" / "session_001",
    )
    toolkit.primitives._official_success_latched = True
    with toolkit.state.record_step(
        state={"task_success": True},
        command={"action": "pi0_nav_pick", "instruction": "turn on radio", "chunks": 1},
        result={"ok": True},
    ):
        pass
    with pytest.raises(RuntimeError, match="missing terminal_receipt.json"):
        toolkit.write_recipe("turning_on_radio_s0")
    assert not (run_dir / "turning_on_radio_s0_recipe.jsonl").exists()


def test_unsolved_behavior_session_does_not_write_recipe(tmp_path: Path) -> None:
    toolkit = BehaviorToolkit(
        runtime_kwargs={
            "task_name": "turning_on_radio",
            "output_dir": tmp_path / "run",
        },
        dashboard_events=NullDashboardEventSink(),
        memory=MemoryManager(tmp_path / "memory"),
    )
    with toolkit.state.record_step(
        state={"task_success": False},
        command={"action": "pi0_nav_pick", "instruction": "turn on the radio"},
        result={"ok": True},
        elapsed_s=0.1,
    ):
        pass

    assert toolkit.write_recipe("turning_on_radio_s0") is None
    assert not (tmp_path / "run" / "turning_on_radio_s0_recipe.jsonl").exists()


@pytest.mark.parametrize("name", ["observe", "open"])
def test_behavior_images_stay_out_of_step_json(tmp_path: Path, name: str) -> None:
    from robots.behavior.tools import _PUBLIC_IMAGE_BYTE_FIELDS

    env = SimpleNamespace(
        total_env_steps=0, official_success_latched=False, official_success_receipt=None
    )
    toolkit = BehaviorToolkit(
        runtime_kwargs={
            "env": env,
            "task_name": "turning_on_radio",
            "output_dir": tmp_path / "run",
            "initial_observation": {
                "main_images": np.zeros((16, 16, 3), dtype=np.uint8),
            },
        },
        memory=MemoryManager(tmp_path / "memory"),
    )
    image = toolkit.state.load_bytes("head_rgb.png", step=0)
    payload = {"status": "ok", **dict.fromkeys(_PUBLIC_IMAGE_BYTE_FIELDS, image)}
    env.observe = lambda **kwargs: payload
    env.open_gripper = lambda **kwargs: payload
    try:
        arguments = (
            {"camera": "head"}
            if name == "observe"
            else {
                "hand": "left",
                "visual_hand_check": {
                    "camera": "left_wrist",
                    "frame_id": "left-frame",
                    "selected_hand": "left",
                    "assessment": "selected_hand_visually_confirmed",
                },
            }
        )
        result = toolkit.execute_tool(name, arguments)
        assert "error" not in result.data
        assert "state_capture_error" not in result.data
        assert len(result.images) == 6
        assert not any(field in result.to_text() for field in _PUBLIC_IMAGE_BYTE_FIELDS)
        saved = json.loads((tmp_path / "run" / "states.json").read_text())
        record = saved["steps"][-1]
        assert not any(field in record["result"] for field in _PUBLIC_IMAGE_BYTE_FIELDS)
        assert record.get("command") == {"action": name, **arguments}
    finally:
        toolkit.close()


@pytest.mark.parametrize(
    ("payload", "expected_truncated", "expected_terminated"),
    [
        ({"truncated": True}, True, False),
        ({"stop_reason": "truncated"}, True, False),
        ({"terminated": True, "truncated": False}, False, True),
        ({"stop_reason": "terminated"}, False, True),
    ],
)
def test_behavior_truncation_reaches_state_and_dashboard(
    tmp_path: Path,
    payload: dict,
    expected_truncated: bool,
    expected_terminated: bool,
) -> None:
    from robots.behavior.robot_spec import BEHAVIOR_DASHBOARD_SPEC
    from rpent.dashboard.state import DashboardState

    toolkit = BehaviorToolkit(
        runtime_kwargs={
            "task_name": "turning_on_radio",
            "output_dir": tmp_path / "run",
        },
        memory=MemoryManager(tmp_path / "memory"),
    )
    try:
        toolkit.get_env_state(
            command={
                "action": "pi0_nav_pick",
                "instruction": "turn on the radio",
                "chunks": 1,
            },
            result=payload,
            elapsed_s=0.1,
        )
        record = toolkit.state.latest_record()
        assert record.truncated is expected_truncated
        assert record.terminated is expected_terminated
        assert toolkit.solved() is False
        stored = json.loads((tmp_path / "run" / "states.json").read_text())["steps"][-1]
        assert stored["truncated"] is expected_truncated
        assert stored["terminated"] is expected_terminated
        dashboard = DashboardState(
            output_dir=tmp_path / "dashboard", dashboard_spec=BEHAVIOR_DASHBOARD_SPEC
        )
        dashboard.on_step(record)
        snapshot = dashboard.session_detail()
        assert snapshot["truncated"] is expected_truncated
        assert snapshot["terminated"] is expected_terminated
        assert snapshot["timeline"][-1]["result"]["official_success"] is False
        assert snapshot["timeline"][-1]["truncated"] is expected_truncated
        assert snapshot["timeline"][-1]["terminated"] is expected_terminated
    finally:
        toolkit.close()
