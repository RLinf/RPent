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
import sys
from pathlib import Path
from typing import Any

import pytest

from robots.behavior.dino_v2.client import BehaviorDinoClient
from robots.behavior.dino_v2.encoder import DINOV2_DIMENSION
from robots.behavior.dino_v2.server import BehaviorDinoFacade
from robots.behavior.env_client import BehaviorEnvClient
from robots.behavior.env_server import BehaviorEnvFacade
from robots.behavior.robot_spec import get_toolkit
from robots.behavior.schemas import BEHAVIOR_TOOL_NAMES, MOVE_TO_SPEC
from robots.behavior.toolkit import BehaviorToolkit
from robots.behavior.tools import BehaviorPrimitives
from rpent.dashboard.events import NullDashboardEventSink
from rpent.memory import MemoryManager
from rpent.robots import RunConfig

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
        self.gripper_calls: list[tuple[str, dict[str, Any]]] = []

    def move_to(self, **kwargs: Any) -> dict[str, Any]:
        self.last_move = kwargs
        return {"status": "failed", "stop_reason": "motion_unavailable"}

    def close_gripper(self, **kwargs: Any) -> dict[str, Any]:
        self.gripper_calls.append(("close_gripper", kwargs))
        return {"status": "success"}

    def open_gripper(self, **kwargs: Any) -> dict[str, Any]:
        self.gripper_calls.append(("open_gripper", kwargs))
        return {"status": "success"}


class _FakeRpcClient:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def call(
        self,
        method: str,
        *,
        args: tuple = (),
        kwargs: dict[str, Any] | None = None,
        timeout_s: float | None = None,
    ) -> dict[str, Any]:
        self.calls.append((method, kwargs or {}))
        if method == "env.get_env_meta":
            return {}
        if method == "dino.get_meta":
            return {
                "runtime": "behavior_dino",
                "dimension": DINOV2_DIMENSION,
            }
        return {"status": "success"}


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


@pytest.mark.parametrize(
    ("extra_args", "message"),
    [
        (
            [
                "--behavior-mode",
                "explore",
                "--explore-attempts-per-session",
                "1",
            ],
            "BEHAVIOR explore runs one attempt per session; use --explore-sessions",
        ),
        ([], "BEHAVIOR --explore requires --behavior-mode explore"),
        (
            ["--behavior-mode", "explore", "--dashboard"],
            "BEHAVIOR --explore is CLI-only",
        ),
        (
            ["--behavior-mode", "explore", "--env-endpoint", "127.0.0.1:1"],
            "BEHAVIOR explore requires an owned env sidecar; omit --env-endpoint",
        ),
    ],
)
def test_behavior_explore_rejects_incompatible_options(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    extra_args: list[str],
    message: str,
) -> None:
    from rpent.cli import main as cli

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "rpent",
            "--robot",
            "behavior",
            "--task-name",
            "turning_on_radio",
            "--public-seed",
            "0",
            "--explore",
            *extra_args,
        ],
    )

    with pytest.raises(SystemExit) as exc_info:
        cli.main()

    assert exc_info.value.code == 2
    assert message in capsys.readouterr().err


def test_behavior_toolkit_factory_maps_shared_modes(tmp_path: Path) -> None:
    config = RunConfig(
        recipe_tag="turning_on_radio_s0",
        output_dir=tmp_path / "run",
        prompt_vars={
            "behavior_mode": "eval",
            "memory_dir": str(tmp_path / "memory"),
            "task_name": "turning_on_radio",
            "public_seed": 0,
            "max_episode_steps": 64,
        },
        task_desc={},
    )
    dashboard_events = NullDashboardEventSink()

    evaluation = get_toolkit(
        primitives_kwargs={},
        dashboard_events=dashboard_events,
        config=config,
        mode="evaluation",
    )
    exploration = get_toolkit(
        primitives_kwargs={"output_dir": tmp_path / "session"},
        dashboard_events=dashboard_events,
        config=config,
        mode="exploration",
    )
    config.prompt_vars["behavior_mode"] = "explore"
    inherited = get_toolkit(
        primitives_kwargs={},
        dashboard_events=dashboard_events,
        config=config,
    )

    assert evaluation.primitives.behavior_phase == "eval"
    assert exploration.primitives.behavior_phase == "explore"
    assert inherited.primitives.behavior_phase == "explore"
    assert evaluation.memory._memory_access == "read_only"
    assert exploration.memory._memory_access == "inbox_write"
    assert exploration.primitives.output_dir == tmp_path / "session"
    with pytest.raises(ValueError, match="one attempt per session"):
        get_toolkit(
            primitives_kwargs={},
            dashboard_events=dashboard_events,
            config=config,
            mode="exploration",
            attempts_per_session=1,
        )
    with pytest.raises(ValueError, match="unsupported BEHAVIOR toolkit mode"):
        get_toolkit(
            primitives_kwargs={},
            dashboard_events=dashboard_events,
            config=config,
            mode="unsupported",
        )


def test_behavior_facades_use_default_healthz_and_registered_metadata() -> None:
    facade = BehaviorEnvFacade(backend=object(), meta={"task_language": "test"})
    dino = BehaviorDinoFacade(
        encoder=object(),
        meta={"runtime": "behavior_dino", "dimension": DINOV2_DIMENSION},
    )

    assert facade._dispatch("healthz", (), {}) == {"status": "ok"}
    assert facade._dispatch("env.get_env_meta", (), {}) == {"task_language": "test"}
    assert "env.close_gripper" in facade._rpc
    assert "env.open_gripper" in facade._rpc
    assert "env.close" not in facade._rpc
    assert "env.open" not in facade._rpc
    assert dino._dispatch("healthz", (), {}) == {"status": "ok"}
    dino_meta = dino._dispatch("dino.get_meta", (), {})
    assert dino_meta["runtime"] == "behavior_dino"
    assert dino_meta["dimension"] == DINOV2_DIMENSION
    assert isinstance(dino_meta["pid"], int)


def test_behavior_clients_and_tools_use_explicit_component_rpc_names() -> None:
    rpc = _FakeRpcClient()
    client = BehaviorEnvClient(rpc, expected_meta={})
    dino_client = BehaviorDinoClient(rpc, expected_meta={"runtime": "behavior_dino"})
    env = _FakeEnv()
    primitives = BehaviorPrimitives(env=env, task_name="turning_on_radio")

    client.close_gripper(hand="right")
    client.open_gripper(hand="right")
    primitives.close(hand="right")
    primitives.open(hand="right")

    assert rpc.calls == [
        ("env.get_env_meta", {}),
        ("dino.get_meta", {}),
        ("env.close_gripper", {"hand": "right"}),
        ("env.open_gripper", {"hand": "right"}),
    ]
    assert dino_client.server_meta["dimension"] == DINOV2_DIMENSION
    assert env.gripper_calls == [
        ("close_gripper", {"hand": "right"}),
        ("open_gripper", {"hand": "right"}),
    ]
    assert "env.close_gripper" in BehaviorEnvClient._TIMEOUT_S
    assert "env.open_gripper" in BehaviorEnvClient._TIMEOUT_S
    assert "env.close" not in BehaviorEnvClient._TIMEOUT_S
    assert "env.open" not in BehaviorEnvClient._TIMEOUT_S


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


def test_receipt_is_session_artifact_and_recipe_is_run_artifact(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    session_dir = run_dir / "sessions" / "session_001"
    toolkit = BehaviorToolkit(
        primitives_kwargs={
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
    assert recipe_path == run_dir / "recipe_turning_on_radio_s0.jsonl"
    assert not (session_dir / "recipe_turning_on_radio_s0.jsonl").exists()
    commands = [
        json.loads(line)
        for line in recipe_path.read_text().splitlines()
        if line.strip()
    ]
    assert commands == [{"action": "future_stateful_command", "arg": 1}]
    assert "command" not in commands[0]


def test_unsolved_behavior_session_does_not_write_recipe(tmp_path: Path) -> None:
    toolkit = BehaviorToolkit(
        primitives_kwargs={
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
    assert not (tmp_path / "run" / "recipe_turning_on_radio_s0.jsonl").exists()
