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

"""Offline contracts for EFE tool-driven sessions."""

from __future__ import annotations

import json
import os
import subprocess
import sys

import pytest

from robots.efe.runner import run_efe
from robots.efe.toolkit import EfeToolkit
from rpent.dashboard.events import NullDashboardEventSink
from rpent.memory.manager import MemoryManager
from rpent.planner.base import PlannerResult
from rpent.utils.logging import init_output_dir

READ_ONLY_EFE_TOOLS = [
    "efe_spec",
    "efe_observe",
    "efe_status",
    "efe_recover",
]

RAW_EFE_TOOLS = [*READ_ONLY_EFE_TOOLS, "efe_submit"]

SKILL_TOOLS = ["move_to_xyz", "set_gripper"]


@pytest.fixture
def planner_calls(monkeypatch):
    import rpent.planner.codex as codex

    calls = {}

    class FakePlanner:
        def __init__(self, **options):
            calls["options"] = options

        def solve(self, **options):
            calls["solve"] = options
            calls.setdefault("codex_homes", []).append(os.environ.get("CODEX_HOME"))
            return PlannerResult(
                finish_result={"status": "completed", "summary": "Program executed."},
                stats={
                    "total_input_tokens": 17,
                    "total_output_tokens": 9,
                    "tool_calls": 2,
                },
            )

    monkeypatch.setattr(codex, "CodexPlanner", FakePlanner)
    return calls


def test_runner_preserves_task_budget_and_external_scoring(
    tmp_path, planner_calls, efe_client
):
    instruction = "Use the public robot client to move the object.\n"
    output_dir = tmp_path / "agent-logs"

    report = run_efe(
        instruction,
        workdir=tmp_path,
        output_dir=output_dir,
        model="test-model",
        timeout_s=42,
        max_turns=7,
        reasoning_effort="high",
        client=efe_client,
    )

    assert planner_calls["options"]["repo_root"] == tmp_path
    assert planner_calls["options"]["timeout_s"] == 42
    assert planner_calls["options"]["model"] == "test-model"
    assert planner_calls["options"]["reasoning_effort"] == "high"
    assert planner_calls["options"]["sandbox"] == "read-only"
    assert report["codex_sandbox"] == "read-only"
    assert instruction.strip() in planner_calls["solve"]["user_message"]
    assert planner_calls["solve"]["max_turns"] == 7
    assert "verifier" in planner_calls["solve"]["system_prompt"]
    assert "no network access" in planner_calls["solve"]["system_prompt"]
    assert report["agent_finish"]["status"] == "completed"
    assert report["stats"]["total_input_tokens"] == 17
    assert "reward" not in report
    assert json.loads((output_dir / "session.json").read_text()) == report


def test_full_access_sandbox_leaves_shell_available(
    tmp_path, planner_calls, efe_client
):
    report = run_efe(
        "Task",
        workdir=tmp_path,
        output_dir=tmp_path / "logs",
        model="test",
        sandbox="full-access",
        client=efe_client,
    )

    assert planner_calls["options"]["sandbox"] == "full-access"
    assert report["codex_sandbox"] == "full-access"
    assert "no network access" not in planner_calls["solve"]["system_prompt"]


def test_runner_builds_prompt_variables_from_the_shared_helper(
    tmp_path, planner_calls, efe_client, monkeypatch
):
    import robots.efe.prompt_bundle as prompt_bundle
    from robots.efe import session_vars

    captured: dict[str, object] = {}
    build = prompt_bundle.system_prompt

    def capture(variables=None, **kwargs):
        captured.update(variables or {})
        return build(variables, **kwargs)

    monkeypatch.setattr(prompt_bundle, "system_prompt", capture)
    output_dir = tmp_path / "logs"

    run_efe(
        "Task",
        workdir=tmp_path,
        output_dir=output_dir,
        model="test",
        client=efe_client,
    )

    workdir = tmp_path.resolve()
    output = output_dir.resolve()
    memory = output / "memory"
    assert captured == {
        **session_vars.prompt_vars(
            workdir=workdir,
            output_dir=output,
            memory_dir=memory,
            inbox_tag=session_vars.DEFAULT_INBOX_TAG,
            session_max=1,
            skills=True,
        ),
        "session_number": 1,
    }


def test_finish_does_not_count_as_a_service_success(
    tmp_path, planner_calls, efe_client
):
    report = run_efe(
        "Task",
        workdir=tmp_path,
        output_dir=tmp_path / "logs",
        model="test",
        client=efe_client,
    )

    assert report["agent_finish"]["status"] == "completed"
    assert report["sessions"][0]["solved"] is False
    assert report["efe"]["solved"] is False


def test_solved_follows_the_service_end_reason(tmp_path, monkeypatch, efe_client):
    import rpent.planner.codex as codex

    class EndingPlanner:
        def __init__(self, **options):
            pass

        def solve(self, **options):
            efe_client.done = True
            efe_client.done_reason = "success"
            return PlannerResult(stats={"usage_observed": True})

    monkeypatch.setattr(codex, "CodexPlanner", EndingPlanner)

    report = run_efe(
        "Task",
        workdir=tmp_path,
        output_dir=tmp_path / "logs",
        model="test",
        client=efe_client,
    )

    assert report["sessions"][0]["solved"] is True
    assert report["efe"]["solved"] is True


def test_runner_records_startup_failure(tmp_path, monkeypatch, efe_client):
    import rpent.planner.codex as codex

    def failed_planner(**options):
        raise RuntimeError("Codex executable is unavailable.")

    monkeypatch.setattr(codex, "CodexPlanner", failed_planner)

    report = run_efe(
        "Task",
        workdir=tmp_path,
        output_dir=tmp_path / "logs",
        model="test",
        client=efe_client,
    )

    assert report["error"] == "RuntimeError: Codex executable is unavailable."
    assert report["agent_finish"] is None
    assert (
        json.loads((tmp_path / "logs/session.json").read_text())["error"]
        == report["error"]
    )


@pytest.mark.parametrize(
    "options",
    [
        {"timeout_s": 0},
        {"max_turns": -1},
        {"model": " "},
        {"reasoning_effort": "unsupported"},
        {"sandbox": "none"},
        {"connect_timeout_s": 0},
    ],
)
def test_invalid_configuration_does_not_start_planner(tmp_path, planner_calls, options):
    settings = {
        "workdir": tmp_path,
        "output_dir": tmp_path / "logs",
        "model": "test",
        **options,
    }

    with pytest.raises(ValueError):
        run_efe("Task", **settings)

    assert planner_calls == {}
    assert not (tmp_path / "logs").exists()


def test_existing_session_cannot_be_overwritten(tmp_path, planner_calls):
    output_dir = tmp_path / "logs"
    output_dir.mkdir()
    original = '{"error": "previous run"}'
    (output_dir / "session.json").write_text(original)

    with pytest.raises(FileExistsError):
        run_efe("Task", workdir=tmp_path, output_dir=output_dir, model="test")

    assert planner_calls == {}
    assert (output_dir / "session.json").read_text() == original


def _toolkit(tmp_path, client, *, skills=True):
    init_output_dir(tmp_path)
    memory = MemoryManager(tmp_path / "memory")
    return EfeToolkit(
        client=client,
        dashboard_events=NullDashboardEventSink(),
        memory=memory,
        state_output_dir=tmp_path,
        enable_skills=skills,
    )


def test_toolkit_registers_skills_and_shared_file_tools(tmp_path, efe_client):
    toolkit = _toolkit(tmp_path, efe_client)

    result = toolkit.execute_tool(
        "finish", {"status": "completed", "summary": "Session ended."}
    )

    assert toolkit.memory.root == (tmp_path / "memory").resolve()
    assert [declaration.name for declaration in toolkit.list_tools()] == [
        "read_text_file",
        "write_text_file",
        "list_dir",
        "finish",
        *READ_ONLY_EFE_TOOLS,
        "view_env_state",
        *SKILL_TOOLS,
    ]
    assert result.data == {
        "_finish": True,
        "status": "completed",
        "summary": "Session ended.",
    }


def test_raw_only_mode_hides_skills(tmp_path, efe_client):
    toolkit = _toolkit(tmp_path, efe_client, skills=False)

    assert toolkit.skills_enabled is False
    assert [declaration.name for declaration in toolkit.list_tools()] == [
        "read_text_file",
        "write_text_file",
        "list_dir",
        "finish",
        *RAW_EFE_TOOLS,
        "view_env_state",
    ]


def test_incompatible_layout_skips_skills_and_records_the_reason(tmp_path, efe_client):
    efe_client.spec["kinematics"] = {}
    toolkit = _toolkit(tmp_path, efe_client)

    names = [declaration.name for declaration in toolkit.list_tools()]

    assert toolkit.skills_enabled is False
    assert "efe_submit" in names
    assert "move_to_xyz" not in names
    assert "set_gripper" not in names
    assert "move_to_xyz" in toolkit.metrics()["skills_error"]


def test_tool_descriptions_do_not_describe_coding_artifacts(tmp_path, efe_client):
    toolkit = _toolkit(tmp_path, efe_client)

    descriptions = {
        declaration.name: declaration.description or ""
        for declaration in toolkit.list_tools()
    }

    for name in ("read_text_file", "write_text_file", "list_dir"):
        lowered = descriptions[name].lower()
        assert "recipe" not in lowered
        assert "audit" not in lowered


def test_state_records_keep_the_frames_of_their_episode(tmp_path, efe_client):
    efe_client.spec["budget"] = {"reset": 2}
    toolkit = _toolkit(tmp_path, efe_client, skills=False)

    toolkit.execute_tool("efe_observe", {"cameras": ["wrist_cam"]})
    toolkit.execute_tool(
        "efe_submit", {"action_type": "joint_position", "actions": [[0.0] * 7]}
    )
    before = toolkit.execute_tool("view_env_state", {"step": -1})

    toolkit.execute_tool("efe_reset", {})
    after = toolkit.execute_tool("view_env_state", {"step": -1})

    assert len(before.images) == 1
    assert before.data["camera_images"]["wrist_cam"].endswith(
        "camera_wrist_cam_ep000_s000000.png"
    )
    assert after.images == []
    assert after.data["state"]["episode"] == 1


def test_solved_reads_the_service_verdict_not_finish(tmp_path, efe_client):
    toolkit = _toolkit(tmp_path, efe_client)

    assert toolkit.solved() is False
    efe_client.done = True
    efe_client.done_reason = "success"
    assert toolkit.solved() is True


def test_metrics_count_session_calls_and_service_progress(tmp_path, efe_client):
    toolkit = _toolkit(tmp_path, efe_client)

    toolkit.execute_tool("set_gripper", {"opening": 0.02})
    toolkit.execute_tool("move_to_xyz", {"x": 0.0, "y": 0.0, "z": 0.3})

    metrics = toolkit.metrics()

    assert metrics["tool_calls"] == 2
    assert metrics["skill_calls"] == 2
    assert metrics["efe_submit_calls"] == 1
    assert metrics["efe_control_steps"] == 1
    assert metrics["skills_enabled"] is True
    assert metrics["skills_error"] is None


def test_finish_contract_rejects_unsupported_status(tmp_path, efe_client):
    toolkit = _toolkit(tmp_path, efe_client)
    finish = next(
        declaration
        for declaration in toolkit.list_tools()
        if declaration.name == "finish"
    )

    assert "verifier" in finish.description.lower()
    assert toolkit.execute_tool(
        "finish", {"status": "success", "summary": "x"}
    ).is_error


class _RecordingPlanner:
    """Record each planner process the runner builds."""

    instances: list["_RecordingPlanner"] = []
    probe: dict[str, str] | None = None
    session_errors: list[str | None] = []

    def __init__(self, **options):
        self.options = options
        self.solve_options = {}
        _RecordingPlanner.instances.append(self)

    def solve(self, **options):
        self.solve_options = options
        self.env_repo_root = os.environ.get("RPENT_REPO_ROOT")
        index = len(_RecordingPlanner.instances) - 1
        error = (
            _RecordingPlanner.session_errors[index]
            if index < len(_RecordingPlanner.session_errors)
            else None
        )
        if _RecordingPlanner.probe is not None:
            self.probe_result = options["toolkit"].execute_tool(
                "write_text_file", _RecordingPlanner.probe
            )
        return PlannerResult(
            finish_result={"status": "completed", "summary": "Program executed."},
            stats={
                "backend": "codex_sdk",
                "usage_observed": True,
                "turns_used": 1,
                "tool_calls": 2,
                "total_input_tokens": 5,
                "total_cached_input_tokens": 1,
                "total_output_tokens": 2,
                "total_reasoning_output_tokens": 0,
            },
            error=error,
        )


@pytest.fixture
def recording_planner(monkeypatch):
    import rpent.planner.codex as codex

    _RecordingPlanner.instances = []
    _RecordingPlanner.probe = None
    _RecordingPlanner.session_errors = []
    monkeypatch.setattr(codex, "CodexPlanner", _RecordingPlanner)
    return _RecordingPlanner


def test_rpent_arm_loads_memory_and_shared_prompt(
    tmp_path, recording_planner, efe_client
):
    memory_dir = tmp_path / "corpus"
    (memory_dir / "global").mkdir(parents=True)
    (memory_dir / "global" / "pickup.md").write_text("# Pickup lesson\n")
    output_dir = tmp_path / "logs"

    report = run_efe(
        "Use the public robot client.",
        workdir=tmp_path,
        output_dir=output_dir,
        model="test-model",
        memory_dir=memory_dir,
        client=efe_client,
    )

    assert report["error"] is None
    assert report["memory_mode"] == "loaded"
    assert report["memory_root"] == str(memory_dir.resolve())
    assert report["stats"]["total_input_tokens"] == 5
    planner = recording_planner.instances[0]
    system_prompt = planner.solve_options["system_prompt"]
    assert str(memory_dir.resolve()) in system_prompt
    assert "memory" in system_prompt.lower()
    assert "Use the public robot client." in planner.solve_options["user_message"]
    tools = [
        declaration.name
        for declaration in planner.solve_options["toolkit"].list_tools()
    ]
    assert tools == [
        "read_text_file",
        "write_text_file",
        "list_dir",
        "finish",
        *READ_ONLY_EFE_TOOLS,
        "view_env_state",
        *SKILL_TOOLS,
    ]


def test_exploration_handoff_uses_fresh_planners_and_aggregates_stats(
    tmp_path, recording_planner, efe_client
):
    output_dir = tmp_path / "logs"

    report = run_efe(
        "Use the public robot client.",
        workdir=tmp_path,
        output_dir=output_dir,
        model="test-model",
        explore_sessions=2,
        client=efe_client,
    )

    assert report["error"] is None
    assert [session["session"] for session in report["sessions"]] == [1, 2]
    assert report["stats"]["turns_used"] == 2
    assert report["stats"]["tool_calls"] == 4
    assert report["stats"]["total_input_tokens"] == 10
    assert report["stats"]["usage_observed"] is True
    first, second = recording_planner.instances
    assert first.options["output_path"] == (
        output_dir / "sessions" / "session_001" / "codex.txt"
    )
    assert (
        second.options["output_path"]
        == output_dir / "sessions" / "session_002" / "codex_02.txt"
    )
    assert "Earlier sessions" not in first.solve_options["system_prompt"]
    assert "session 1 of up to 2" not in first.solve_options["system_prompt"]
    assert "notes-session-1.md" in first.solve_options["system_prompt"]
    assert "Earlier sessions archived notes" in second.solve_options["system_prompt"]
    assert "session 2 of up to 2" in second.solve_options["system_prompt"]
    assert "notes-session-2.md" in second.solve_options["system_prompt"]


def test_exploration_inbox_is_the_only_writable_memory_path(
    tmp_path, recording_planner, efe_client
):
    output_dir = tmp_path / "logs"
    denied_dir = tmp_path / "logs-denied"
    inbox = output_dir / "memory" / "_internal" / "inbox" / "efe"
    allowed = {"path": str(inbox / "wip" / "notes.md"), "content": "observed\n"}
    denied = {
        "path": str(denied_dir / "memory" / "global" / "bad.md"),
        "content": "x",
    }

    recording_planner.probe = allowed
    run_efe(
        "Task",
        workdir=tmp_path,
        output_dir=output_dir,
        model="test-model",
        explore_sessions=2,
        client=efe_client,
    )
    assert not recording_planner.instances[0].probe_result.is_error

    recording_planner.instances = []
    recording_planner.probe = denied
    run_efe(
        "Task",
        workdir=tmp_path,
        output_dir=denied_dir,
        model="test-model",
        explore_sessions=2,
        client=efe_client,
    )
    assert recording_planner.instances[0].probe_result.is_error


def test_recovered_session_error_is_not_reported_as_run_failure(
    tmp_path, recording_planner, efe_client
):
    recording_planner.session_errors = ["Codex SDK timed out after 5s", None]

    report = run_efe(
        "Task",
        workdir=tmp_path,
        output_dir=tmp_path / "logs",
        model="test-model",
        explore_sessions=2,
        client=efe_client,
    )

    assert report["error"] is None
    assert [session["error"] for session in report["sessions"]] == [
        "Codex SDK timed out after 5s",
        None,
    ]


def test_terminal_session_error_is_reported(tmp_path, recording_planner, efe_client):
    recording_planner.session_errors = ["RuntimeError: boom"]

    report = run_efe(
        "Task",
        workdir=tmp_path,
        output_dir=tmp_path / "logs",
        model="test-model",
        explore_sessions=2,
        client=efe_client,
    )

    assert report["error"] == "RuntimeError: boom"
    assert len(report["sessions"]) == 1


def test_rpent_arm_resolves_relative_paths_from_the_workspace(
    tmp_path, recording_planner, monkeypatch, efe_client
):
    monkeypatch.delenv("RPENT_REPO_ROOT", raising=False)

    report = run_efe(
        "Task",
        workdir=tmp_path,
        output_dir=tmp_path / "logs",
        model="test-model",
        client=efe_client,
    )

    assert report["error"] is None
    assert recording_planner.instances[0].env_repo_root == str(tmp_path.resolve())
    assert "RPENT_REPO_ROOT" not in os.environ


def test_each_trial_uses_separate_empty_memory_and_codex_home(
    tmp_path, planner_calls, monkeypatch, efe_client
):
    host_home = tmp_path / "host-codex"
    first_output = tmp_path / "first"
    second_output = tmp_path / "second"
    monkeypatch.setenv("CODEX_HOME", str(host_home))
    monkeypatch.setenv("CODEX_API_KEY", "test-key")

    first = run_efe(
        "First",
        workdir=tmp_path,
        output_dir=first_output,
        model="test",
        client=efe_client,
    )
    second = run_efe(
        "Second",
        workdir=tmp_path,
        output_dir=second_output,
        model="test",
        client=efe_client,
    )

    assert first["memory_mode"] == "empty"
    assert first["memory_root"] != second["memory_root"]
    assert first["codex_home"] != second["codex_home"]
    assert not any((first_output / "memory").rglob("*"))
    assert not any((second_output / "memory").rglob("*"))
    assert (first_output / "codex-home").is_dir()
    assert (second_output / "codex-home").is_dir()
    assert planner_calls["codex_homes"] == [
        str(first_output.resolve() / "codex-home"),
        str(second_output.resolve() / "codex-home"),
    ]
    assert os.environ["CODEX_HOME"] == str(host_home)
    assert os.environ["CODEX_API_KEY"] == "test-key"


def test_cli_help_runs_without_loading_robot_or_planner_modules():
    script = """import sys
from robots.efe.cli import main
try:
    main(['--help'])
except SystemExit as exc:
    assert exc.code == 0
assert 'rpent.planner.codex' not in sys.modules
assert 'rpent.robots' not in sys.modules
assert 'torch' not in sys.modules
"""

    completed = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True
    )

    assert completed.returncode == 0, completed.stderr
    assert "--instruction-file" in completed.stdout
