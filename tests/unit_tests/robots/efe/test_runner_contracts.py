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

SKILL_TOOLS = ["move_to_pose", "set_gripper"]


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
    assert efe_client.deadline is not None
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

    def capture(variables=None):
        captured.update(variables or {})
        return build(variables)

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
    shared = session_vars.prompt_vars(
        workdir=workdir,
        output_dir=output,
        memory_dir=memory,
        skills=True,
    )
    assert {key: captured[key] for key in shared} == shared
    assert captured["skills"] is True
    assert captured["reset"] is False
    assert captured["end_episode"] is False
    assert captured["restricted_shell"] is True


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
    assert "move_to_pose" not in names
    assert "set_gripper" not in names
    assert "move_to_pose" in toolkit.metrics()["skills_error"]


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
    toolkit.execute_tool("move_to_pose", {"x": 0.0, "y": 0.0, "z": 0.3})

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
    session_stop_reasons: list[str | None] = []

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
        stop_reason = (
            _RecordingPlanner.session_stop_reasons[index]
            if index < len(_RecordingPlanner.session_stop_reasons)
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
            stop_reason=stop_reason,
        )


@pytest.fixture
def recording_planner(monkeypatch):
    import rpent.planner.codex as codex

    _RecordingPlanner.instances = []
    _RecordingPlanner.probe = None
    _RecordingPlanner.session_errors = []
    _RecordingPlanner.session_stop_reasons = []
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


def test_terminal_session_error_is_reported(tmp_path, recording_planner, efe_client):
    recording_planner.session_errors = ["RuntimeError: boom"]

    report = run_efe(
        "Task",
        workdir=tmp_path,
        output_dir=tmp_path / "logs",
        model="test-model",
        client=efe_client,
    )

    assert report["error"] == "RuntimeError: boom"


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


def test_a_spent_planner_budget_is_a_clean_stop(
    tmp_path, recording_planner, efe_client
):
    """A budget stop must not look like a crash: Harbor reads exit 1 as a failure."""
    recording_planner.session_errors = ["Codex SDK timed out after 5s"]

    report = run_efe(
        "Task",
        workdir=tmp_path,
        output_dir=tmp_path / "logs",
        model="test-model",
        client=efe_client,
    )

    assert report["error"] is None
    assert report["budget_stop"] == "Codex SDK timed out after 5s"


def test_a_structured_timeout_is_a_clean_stop_whatever_the_error_says(
    tmp_path, recording_planner, efe_client
):
    """Do not make Harbor's exit code depend on the planner's error wording."""
    recording_planner.session_errors = ["the model returned no final message"]
    recording_planner.session_stop_reasons = ["timeout"]

    report = run_efe(
        "Task",
        workdir=tmp_path,
        output_dir=tmp_path / "logs",
        model="test-model",
        client=efe_client,
    )

    assert report["error"] is None
    assert report["budget_stop"] == "the model returned no final message"


def test_budget_expired_before_the_session_is_a_clean_stop(
    tmp_path, efe_client, monkeypatch
):
    """Reaching the deadline is a budget stop even with no session result."""
    import robots.efe.runner as runner

    clock = [0.0, 100.0]
    monkeypatch.setattr(
        runner.time, "monotonic", lambda: clock.pop(0) if clock else 100.0
    )

    report = run_efe(
        "Task",
        workdir=tmp_path,
        output_dir=tmp_path / "logs",
        model="test-model",
        timeout_s=1,
        client=efe_client,
    )

    assert report["error"] is None
    assert report["budget_stop"] is not None


def test_a_missing_codex_wheel_does_not_abort_the_report(
    tmp_path, recording_planner, efe_client, monkeypatch
):
    """``CODEX_BIN`` may supply a binary no wheel provides; the report still lands."""
    import robots.efe.runner as runner

    def missing(name):
        raise runner.PackageNotFoundError(name)

    monkeypatch.setattr(runner, "version", missing)

    report = run_efe(
        "Task",
        workdir=tmp_path,
        output_dir=tmp_path / "logs",
        model="test-model",
        client=efe_client,
    )

    assert report["versions"] == {
        "rpent": None,
        "openai-codex": None,
        "openai-codex-cli-bin": None,
    }
    assert (tmp_path / "logs" / "session.json").is_file()


def test_a_transient_network_timeout_is_not_reported_as_a_budget_stop(
    tmp_path, recording_planner, efe_client
):
    recording_planner.session_errors = ["ConnectionError: Request timed out after 30s"]

    report = run_efe(
        "Task",
        workdir=tmp_path,
        output_dir=tmp_path / "logs",
        model="test-model",
        client=efe_client,
    )

    assert report["budget_stop"] is None
    assert report["error"] == "ConnectionError: Request timed out after 30s"


def test_the_connect_wait_cannot_outlast_the_planner_budget(
    tmp_path, recording_planner, efe_client
):
    report = run_efe(
        "Task",
        workdir=tmp_path,
        output_dir=tmp_path / "logs",
        model="test-model",
        timeout_s=600,
        connect_timeout_s=3600,
        client=efe_client,
    )

    assert report["connect_timeout_s"] == 600.0
