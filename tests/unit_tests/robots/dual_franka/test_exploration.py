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

"""Offline attended exploration: real toolkit, logger and memory, fake hardware."""

import argparse
import json
from pathlib import Path

import numpy as np
import pytest

from robots.dual_franka import robot_spec
from robots.dual_franka.toolkit import DualFrankaToolkit
from rpent.dashboard.events import NullDashboardEventSink
from rpent.memory import MemoryManager
from rpent.planner.base import PlannerResult
from rpent.tools import ToolResult


class FakeEnv:
    def __init__(self):
        self.meta = {}
        self.resets = 0
        self.moves = []
        self.reset_result = {"ok": True}
        self.fail_observation = False

    def reset(self):
        self.resets += 1
        return self.reset_result

    def get_observation(self):
        if self.fail_observation:
            raise RuntimeError("camera offline")
        return {
            "main_images": np.full((8, 8, 3), self.resets, dtype=np.uint8),
            "extra_view_images": np.zeros((2, 8, 8, 3), dtype=np.uint8),
            "main_depths": np.ones((8, 8), dtype=np.float32),
            "extra_view_depths": np.ones((2, 8, 8), dtype=np.float32),
            "d455_images": np.zeros((8, 8, 3), dtype=np.uint8),
            "d455_depths": np.ones((8, 8), dtype=np.float32),
        }

    def get_robot_state(self):
        return {
            "left_arm": {"tcp_pose": [0.5, 0, 0.5, 0, 0, 0, 1]},
            "right_arm": {"tcp_pose": [0.5, 0, 0.5, 0, 0, 0, 1]},
        }

    def get_camera_meta(self):
        return {"cameras": {}}

    def move_delta(self, arm, delta):
        self.moves.append((arm, list(delta)))
        return {"ok": True}


@pytest.fixture
def setup(tmp_path, monkeypatch, dual_franka_robot_config):
    from rpent.utils import logging

    monkeypatch.setattr(logging, "_output_dir", tmp_path)
    monkeypatch.setattr("robots.dual_franka.toolkit.get_output_dir", lambda: tmp_path)
    from robots.franka.runtime_config import set_robot_config_path

    set_robot_config_path(dual_franka_robot_config)
    env = FakeEnv()
    replies = []
    toolkit = DualFrankaToolkit(
        runtime_kwargs={"env": env, "model": None, "task_description": "test"},
        dashboard_events=NullDashboardEventSink(),
        memory=MemoryManager(
            tmp_path / "memory",
            memory_access="inbox_write",
            inbox_cell_tag="dual_franka_t0",
        ),
        mode="exploration",
        attempts_per_session=2,
        state_output_dir=tmp_path / "sessions" / "session_001",
        operator_input=lambda prompt, cancelled: replies.pop(0),
    )
    return toolkit, env, replies


def call(t, name, **kwargs):
    return t.execute_tool(name, kwargs).data


def reset(t, replies):
    replies.append("done")
    return call(t, "request_scene_reset", reason="fresh attempt")


def verdict(t, replies, answer):
    replies.append(answer)
    return call(t, "request_operator_verdict")


def test_failed_attempt_reset_success_preserves_logs_and_exports_only_winner(
    setup, tmp_path
):
    t, env, replies = setup
    assert env.resets == 0
    assert "error" in call(t, "move_delta", arm="left", delta_xyz=[0.01, 0, 0])
    assert not env.moves
    assert reset(t, replies)["exploration"]["attempt"] == 1
    call(t, "move_delta", arm="left", delta_xyz=[0.01, 0, 0])
    assert verdict(t, replies, "failure slipped")["status"] == "failure"
    assert "error" in call(t, "finish", status="success", summary="agent claim")
    first_steps = len(t.state.records())
    first_image = t.state.load("left_wrist.png", step=2).copy()
    assert reset(t, replies)["exploration"]["attempt"] == 2
    call(t, "move_delta", arm="right", delta_xyz=[0.02, 0, 0])
    assert verdict(t, replies, "success object lifted")["status"] == "success"
    assert t.solved()
    finish = call(t, "finish", status="failure", summary="done")
    assert finish["_finish"] and finish["status"] == "success"
    assert len(t.state.records()) > first_steps
    np.testing.assert_array_equal(t.state.load("left_wrist.png", step=2), first_image)
    assert t.state.exists("d455_depth.npy")
    assert t.state.exists("camera_meta.json")
    recipe = Path(t.write_recipe("dual_franka_t0"))
    commands = [json.loads(line) for line in recipe.read_text().splitlines()]
    assert len(commands) == 1 and commands[0]["arm"] == "right"
    audit = json.loads((tmp_path / "dual_franka_t0.json").read_text())
    assert audit["success_source"] == "operator"
    assert audit["command_sequence"] == commands
    assert len(t.state.load("operator_events.json", step=None)) >= 7


def test_old_verdict_clears_on_continue_and_motion(setup):
    t, _, replies = setup
    reset(t, replies)
    verdict(t, replies, "success")
    assert t.solved()
    verdict(t, replies, "continue")
    assert not t.solved()
    assert "error" in call(t, "finish", status="success", summary="done")
    verdict(t, replies, "success")
    call(t, "move_delta", arm="right", delta_xyz=[0, 0, 0.01])
    assert not t.solved()
    assert t.write_recipe("dual_franka_t0") is None


@pytest.mark.parametrize("failure", ["return", "exception", "camera"])
def test_reset_failure_never_advances_attempt_or_allows_motion(setup, failure):
    t, env, replies = setup
    if failure == "return":
        env.reset_result = {"ok": False}
    elif failure == "exception":

        def fail():
            raise RuntimeError("reset failed")

        env.reset = fail
    else:
        env.fail_observation = True
    assert "error" in reset(t, replies)
    assert t._attempt == 0 and not t._scene_ready and not t.solved()
    call(t, "move_delta", arm="left", delta_xyz=[0.01, 0, 0])
    assert not env.moves


@pytest.mark.parametrize(
    "response",
    [
        "abort",
        "abort calibration board still attached",
        "  ABORT operator stop  ",
        None,
    ],
)
def test_operator_abort_allows_finish_even_with_budget_and_never_succeeds(
    setup, response
):
    t, env, replies = setup
    replies.append(response)
    reset_result = call(t, "request_scene_reset", reason="initial")
    assert reset_result["operator_aborted"]
    result = call(t, "finish", status="success", summary="stop")
    assert result["_finish"] and result["operator_aborted"]
    assert result["status"] == "failure" and not t.solved() and env.resets == 0


def test_budget_and_old_perception_boundary(setup):
    t, env, replies = setup
    reset(t, replies)
    old = t.state.latest_step
    reset(t, replies)
    assert "error" in call(t, "back_project", camera="d455", row=1, col=1, step=old)
    assert "error" in call(t, "request_scene_reset", reason="third")
    assert env.resets == 2


def test_explore_prompt_and_factory_use_local_layered_memory(
    tmp_path, dual_franka_robot_config
):
    parser = argparse.ArgumentParser()
    parser.add_argument("--explore", action="store_true")
    parser.add_argument("--memory-dir")
    parser.add_argument("--memory-profile", default=None)
    parser.add_argument("--output-dir")
    robot_spec.get_robot_spec().add_cli_args(parser, False)
    args = parser.parse_args(
        [
            "--explore",
            "--memory-dir",
            str(tmp_path / "memory"),
            "--output-dir",
            str(tmp_path),
            "--robot-config",
            str(dual_franka_robot_config),
        ]
    )
    config = robot_spec.get_robot_spec().parse_config(args)
    prompt = robot_spec.get_robot_spec().prompts.render(
        "system", variables={**config.prompt_vars, "output_dir": tmp_path}
    )
    assert "request_scene_reset" in prompt and "request_operator_verdict" in prompt
    assert (
        "scope: global" in prompt
        and "scope: task-family" in prompt
        and "task-specific" in prompt
    )
    assert "{{" not in prompt and "libero_terminated" not in prompt
    t = robot_spec.get_toolkit(
        runtime_kwargs={"env": FakeEnv(), "model": None, "task_description": "test"},
        dashboard_events=NullDashboardEventSink(),
        config=config,
        mode="exploration",
        state_output_dir=tmp_path / "session",
    )
    assert t.memory.root == tmp_path / "memory"
    rejected = call(
        t, "write_text_file", path=str(tmp_path / "memory/global/no.md"), content="no"
    )
    assert "error" in rejected
    accepted = call(
        t,
        "write_text_file",
        path=str(tmp_path / "memory/_internal/inbox/dual_franka_t0/wip/notes.md"),
        content="evidence",
    )
    assert "error" not in accepted


def test_successful_memory_pair_uses_existing_merge_and_index(setup, tmp_path):
    t, _, replies = setup
    reset(t, replies)
    call(t, "move_delta", arm="left", delta_xyz=[0.01, 0, 0])
    verdict(t, replies, "success")
    t.write_recipe("dual_franka_t0")
    inbox = t.memory.root / "_internal/inbox/dual_franka_t0"
    inbox.mkdir(parents=True)
    (inbox / "new_global_strategy_small.md").write_text("""---
id: small
scope: global
kind: strategy
title: Small moves
applies_when: Staging
confidence: single-shot
evidence:
  cells: [dual_franka_t0]
---
Observed once; see attempt 1.
""")
    (inbox / "task-family_dual_franka_t0_draft.md").write_text("""---
id: task-family_dual_franka_real_t0
scope: task-family
suite: dual_franka
regime: real
task_id: 0
task_language: Test
confidence: single-shot
evidence:
  cells: [dual_franka_t0]
---
Winning technique and failure evidence.
""")
    result = t.memory.merge_memory(
        cell_tag="dual_franka_t0", run_state_dir=tmp_path, solved=t.solved()
    )
    assert result["global"] == result["task-family"] == result["task"] == 1
    assert not t.memory.validate()
    assert (t.memory.root / "MEMORY.md").exists()
    assert (t.memory.root / "task-specific/dual_franka_t0_recipe.jsonl").exists()


@pytest.mark.parametrize("interactive", [False, True])
def test_cli_two_sessions_operator_feedback_and_memory_pipeline(
    tmp_path, monkeypatch, interactive, dual_franka_robot_config
):
    import sys
    from dataclasses import replace
    from types import SimpleNamespace

    from rpent.cli import main as cli
    from rpent.tools.human_in_the_loop import HumanInTheLoopInput

    env = FakeEnv()
    runtimes = []
    planners = []
    replies = iter(["done", "failure dropped", "done", "success lifted"])

    class Operator(HumanInTheLoopInput):
        def request(self, prompt, cancelled, *, kind=None):
            cancelled()
            return next(replies)

        def __call__(self, prompt, cancelled):
            return self.request(prompt, cancelled)

    class Planner:
        def __init__(self, *args, **kwargs):
            self.number = len(planners) + 1
            planners.append(self)

        def solve(self, *, toolkit, system_prompt, user_message, **kwargs):
            assert "scope: global" in system_prompt
            if self.number == 2:
                assert "not automatically reset" in user_message
                assert "task_name:" in user_message
                assert "Original operator task instruction:" in user_message
            assert not toolkit._scene_ready
            assert "error" not in call(
                toolkit, "request_scene_reset", reason="new attempt"
            )
            call(
                toolkit, "move_delta", arm="left", delta_xyz=[self.number * 0.01, 0, 0]
            )
            call(toolkit, "request_operator_verdict")
            if self.number == 2:
                inbox = toolkit.memory.root / "_internal/inbox/dual_franka_t0"
                call(
                    toolkit,
                    "write_text_file",
                    path=str(inbox / "new_global_strategy_cli.md"),
                    content="""---
id: cli
scope: global
kind: strategy
title: CLI lesson
applies_when: Same setup
confidence: single-shot
evidence:
  cells: [dual_franka_t0]
---
Observed success in session 2.
""",
                )
            finish = call(
                toolkit, "finish", status="success", summary="attempt completed"
            )
            assert finish["_finish"]
            return SimpleNamespace(
                finish_result=finish, messages=[], stats={}, error=None
            )

    def init_runtime(*args):
        runtimes.append(args)
        return [], {
            "env": env,
            "model": None,
            "task_description": "test",
        }

    spec = replace(robot_spec.get_robot_spec(), init_runtime=init_runtime)
    monkeypatch.setattr(cli, "get_robot_spec", lambda name: spec)
    monkeypatch.setattr(cli, "build_planner", Planner)
    monkeypatch.setattr(
        cli,
        "start_interactive_reader",
        lambda inputs, **kwargs: inputs.put(kwargs["first_prompt_default"]),
    )
    monkeypatch.setattr("rpent.tools.human_in_the_loop.HumanInTheLoopInput", Operator)
    monkeypatch.setattr(sys, "stdin", SimpleNamespace(isatty=lambda: True))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "rpent",
            "--robot",
            "dual_franka",
            "--planner",
            "codex",
            "--explore",
            "--explore-sessions",
            "2",
            "--explore-attempts-per-session",
            "1",
            "--output-dir",
            str(tmp_path / "run"),
            "--memory-dir",
            str(tmp_path / "memory"),
            "--robot-config",
            str(dual_franka_robot_config),
            "--auto-merge-memory",
            *(["--interactive"] if interactive else []),
        ],
    )
    assert cli.main() == 0
    assert len(planners) == 2 and len(runtimes) == 1 and env.resets == 2
    traces = [
        json.loads((tmp_path / f"run/sessions/session_{i:03d}/states.json").read_text())
        for i in (1, 2)
    ]
    for trace in traces:
        assert any(
            r.get("command", {}).get("action") == "move_delta" for r in trace["steps"]
        )
    recipe = (tmp_path / "run/dual_franka_t0_recipe.jsonl").read_text()
    assert "0.02" in recipe and "0.01" not in recipe
    audit = json.loads(
        (tmp_path / "memory/task-specific/dual_franka_t0.json").read_text()
    )
    assert (
        "session_002" in audit["state_trace"] and audit["success_source"] == "operator"
    )
    assert (tmp_path / "memory/global/cli.md").exists()
    assert (tmp_path / "memory/MEMORY.md").exists()


def test_env_client_explore_attachment_does_not_reset():
    from robots.dual_franka.env_client import DualFrankaEnvClient

    class Rpc:
        calls = []

        def call(self, name, **kwargs):
            self.calls.append(name)
            return {"ok": True, "explicit_reset_only": True}

    rpc = Rpc()
    DualFrankaEnvClient(rpc, reset_on_connect=False)
    assert rpc.calls == ["env.get_env_meta"]


@pytest.mark.parametrize("dashboard", [False, True])
def test_invalid_operator_transport_rejected_before_runtime(
    monkeypatch, capsys, dashboard
):
    import sys
    from types import SimpleNamespace

    from rpent.cli import main as cli

    monkeypatch.setattr(sys, "stdin", SimpleNamespace(isatty=lambda: dashboard))
    argv = ["rpent", "--robot", "dual_franka", "--explore"]
    if dashboard:
        argv.append("--dashboard")
    monkeypatch.setattr(sys, "argv", argv)
    monkeypatch.setattr(
        cli, "build_planner", lambda *a, **k: pytest.fail("planner should not start")
    )
    with pytest.raises(SystemExit) as exc:
        cli.main()
    assert exc.value.code == 2
    assert ("Dashboard" if dashboard else "TTY") in capsys.readouterr().err


def test_explore_rejects_old_external_env_without_reset_contract():
    from robots.dual_franka.env_client import DualFrankaEnvClient

    class OldRpc:
        calls = []

        def call(self, name, **kwargs):
            self.calls.append(name)
            return {"ok": True}

    rpc = OldRpc()
    with pytest.raises(RuntimeError, match="explicit_reset_only"):
        DualFrankaEnvClient(rpc, reset_on_connect=False)
    assert rpc.calls == ["env.get_env_meta"]


@pytest.mark.parametrize(
    "tool,arguments",
    [
        ("vla_right_grasp", {"prompt": "grasp"}),
        ("vla_handoff", {"prompt": "transfer"}),
        ("vla_left_place", {"prompt": "place"}),
        ("recover_joint_posture", {"reason": "joint warning"}),
    ],
)
def test_pr176_added_motion_tools_share_explore_guards(setup, tool, arguments):
    t, env, replies = setup
    executed = []
    t._primitives._run_named_vla_skill = lambda **kwargs: (
        executed.append(kwargs) or {"ok": True}
    )
    env.recover_joint_posture = lambda **kwargs: executed.append(kwargs) or {"ok": True}
    assert tool in t._tools
    assert "error" in call(t, tool, **arguments)
    assert not executed
    reset(t, replies)
    verdict(t, replies, "success")
    assert t.solved()
    assert "error" not in call(t, tool, **arguments)
    assert executed and not t.solved()


def test_setup_describes_operator_reset_in_exploration(setup):
    t, _, _ = setup
    result = call(t, "describe_dual_franka_setup")
    assert result["phase"] == "exploration"
    assert "No automatic reset" in result["reset_policy"]
    assert "recover_joint_posture" in result["available_primitives"]
    assert "vla_handoff" in result["available_primitives"]


def test_direct_success_stops_active_tool_and_records_memory(setup, tmp_path):
    import threading

    from rpent.tools.toolkit import ToolCancelled

    t, env, replies = setup
    assert not t.request_direct_verdict("success")
    reset(t, replies)
    entered, release = threading.Event(), threading.Event()
    results = []

    def active_motion(**kwargs):
        env.moves.append(("right", [0.01, 0, 0]))
        entered.set()
        assert release.wait(2)
        t.raise_if_cancelled()
        pytest.fail("motion must not continue after success")

    t.add_tool(t._tools["move_delta"].with_handler(active_motion), replace=True)
    worker = threading.Thread(
        target=lambda: results.append(
            call(t, "move_delta", arm="right", delta_xyz=[0.01, 0, 0])
        )
    )
    worker.start()
    assert entered.wait(2)
    assert t.request_direct_verdict("success")
    assert not t.request_direct_verdict("success")  # no duplicate planner EOF
    assert call(t, "open_gripper", arm="left")["motion_refused"]
    release.set()
    worker.join(2)
    assert not worker.is_alive()
    assert results[0]["code"] == "tool_cancelled"
    with pytest.raises(ToolCancelled):
        t.raise_if_cancelled()
    result = t.finalize_direct_verdict()
    assert result["status"] == "success" and t.solved()
    assert env.resets == 1 and len(env.moves) == 1
    assert t.state.latest_record().command["action"] == "observe_for_verdict"
    t.write_recipe("dual_franka_t0")
    merged = t.memory.merge_memory(
        cell_tag="dual_franka_t0", run_state_dir=tmp_path, solved=True
    )
    assert merged["task"] == 1
    audit = json.loads(
        (tmp_path / "memory/task-specific/dual_franka_t0.json").read_text()
    )
    assert audit["success_source"] == "operator"
    assert len(audit["command_sequence"]) == 1


def test_direct_success_with_failed_observation_does_not_publish(setup):
    t, env, replies = setup
    reset(t, replies)
    assert t.request_direct_verdict("success")
    env.fail_observation = True
    with pytest.raises(RuntimeError, match="camera offline"):
        t.finalize_direct_verdict()
    assert not t.solved()
    assert t.write_recipe("dual_franka_t0") is None


@pytest.mark.parametrize(
    "verdict,auto_merge_memory,planner_error,session_count,consume_cancel",
    [
        ("success", True, None, 2, True),
        ("success", False, None, 1, False),
        ("success", True, "planner transport failed", 1, True),
        ("failure", True, None, 2, False),
        ("failure", True, "planner transport failed", 2, True),
        ("abort", True, None, 1, False),
        ("abort", True, "planner transport failed", 1, True),
    ],
)
def test_cli_direct_verdict_finalizes_and_merges_only_without_errors(
    tmp_path,
    run_interactive_cli,
    verdict,
    planner_error,
    session_count,
    consume_cancel,
    auto_merge_memory,
):
    env = FakeEnv()
    handlers = {}
    solve_calls = []

    def reader(input_queue, **kwargs):
        handlers["line"] = kwargs["line_handler"]
        handlers["inputs"] = input_queue
        input_queue.put("test task")

    class Planner:
        def solve(self, *, toolkit, input_queue, **kwargs):
            import queue

            with pytest.raises(queue.Empty):
                input_queue.get(block=False)
            if solve_calls:
                assert "write a grounded outcome summary" in kwargs["user_message"]
                assert "Focus on the later task stages" in kwargs["user_message"]
                assert not toolkit._scene_ready
                if verdict == "success" and auto_merge_memory:
                    assert (
                        tmp_path / "memory/task-specific/dual_franka_t0_recipe.jsonl"
                    ).is_file()
            solve_calls.append(True)
            assert not handlers["line"]("Focus on the later task stages")
            toolkit._operator_input = lambda *args: "done"
            call(toolkit, "request_scene_reset", reason="test")
            call(toolkit, "move_delta", arm="right", delta_xyz=[0.01, 0, 0])
            assert handlers["line"]("/" + verdict + " 建议采用混合控制")
            assert handlers["line"]("/" + verdict + " 重复提交")
            if consume_cancel:
                assert input_queue.get(timeout=1) is None
            assert handlers["inputs"].empty()
            return PlannerResult(error=planner_error)

    result = run_interactive_cli(
        Planner(),
        env,
        reader,
        sessions=session_count,
        auto_merge_memory=auto_merge_memory,
    )
    assert result == (1 if planner_error and verdict != "failure" else 0)
    if verdict == "failure":
        failures = list((tmp_path / "memory/_internal/inbox").rglob("*-failure.json"))
        assert len(failures) == session_count
        assert json.loads(failures[0].read_text())["verdict"]["status"] == "failure"
        assert (
            json.loads(failures[0].read_text())["verdict"]["operator_notes"]
            == "建议采用混合控制"
        )
    assert len(solve_calls) == (
        session_count
        if verdict == "failure" or (verdict == "success" and not planner_error)
        else 1
    )
    if verdict == "success" and not planner_error:
        assert (
            len(list((tmp_path / "run/sessions").glob("*/dual_franka_t0_recipe.jsonl")))
            == session_count
        )
    assert (tmp_path / "memory/task-specific/dual_franka_t0.json").is_file() == (
        auto_merge_memory and verdict == "success" and planner_error is None
    )
    assert (
        tmp_path / "memory/task-specific/dual_franka_t0_recipe.jsonl"
    ).is_file() == (
        auto_merge_memory and verdict == "success" and planner_error is None
    )
    events = json.loads(
        (tmp_path / "run/sessions/session_001/operator_events.json").read_text()
    )
    assert events[-1]["status"] == ("failure" if verdict == "abort" else verdict)
    assert events[-1]["operator_finished"] is (verdict == "abort")


def test_direct_failure_ends_without_success_memory_even_before_reset(setup):
    t, env, _ = setup
    assert t.request_direct_verdict("failure")
    assert not t.request_direct_verdict("failure")
    assert not t.request_direct_verdict("success")
    assert call(t, "request_scene_reset", reason="late reset")["motion_refused"]
    result = t.finalize_direct_verdict()
    assert result["status"] == "failure" and not result["operator_finished"]
    assert not t.solved() and env.resets == 0
    assert t.write_recipe("dual_franka_t0") is None


def test_direct_abort_exits_even_when_camera_is_unavailable(setup):
    t, env, _ = setup
    env.fail_observation = True
    assert t.request_direct_verdict("abort")
    result = t.finalize_direct_verdict()
    assert result["operator_aborted"] and result["operator_finished"]
    assert result["operator_verdict"] == "abort"
    assert not t.solved() and env.resets == 0
    assert t.write_recipe("dual_franka_t0") is None


def test_explore_task_is_only_in_user_prompt():
    from robots.dual_franka import prompt_bundle

    variables = {"mode": "explore"}
    system = prompt_bundle.system_prompt(variables)
    user = prompt_bundle.user_prompt(variables)
    for section in ("TASK", "TASK CONSTRAINTS"):
        assert section not in system
        assert section in user


def test_current_view_refreshes_but_historical_view_does_not(setup, monkeypatch):
    t, env, replies = setup
    reset(t, replies)
    old_step = t.state.latest_step
    reads = []
    published = []
    monkeypatch.setattr(t, "_publish_step", published.append)
    original = env.get_observation

    def observe():
        reads.append(True)
        obs = original()
        obs["d455_images"][:] = 77
        return obs

    env.get_observation = observe
    observed = t.execute_tool("view_env_state", {})
    assert len(reads) == 1 and t.state.latest_step == old_step + 1
    assert observed.images == [t.state.load_bytes("d455.png")]
    assert len(published) == 1
    assert published[0].command == {"action": "observe_current"}
    assert np.all(t.state.load("d455.png") == 77)
    call(t, "view_env_state", step=old_step)
    assert len(reads) == 1
    assert (
        "error" in t._current_perception(lambda **kw: ToolResult(), step=old_step).data
    )
    t._current_perception(lambda **kw: ToolResult(), step=t.state.latest_step)
    assert len(reads) == 1  # Localization must not silently acquire another frame.
    assert len(published) == 1
    assert env.resets == 1 and not env.moves
    call(t, "move_delta", arm="left", delta_xyz=[0.01, 0, 0])
    assert len(reads) == 2 and t.state.latest_step == old_step + 2
    assert len(published) == 2
    assert published[-1].command["action"] == "move_delta"
    assert len(env.moves) == 1 and env.moves[0][0] == "left"
    np.testing.assert_allclose(env.moves[0][1], [0.01, 0, 0])


@pytest.fixture
def run_interactive_cli(tmp_path, monkeypatch, dual_franka_robot_config):
    import sys
    from dataclasses import replace
    from types import SimpleNamespace

    from rpent.cli import main as cli

    monkeypatch.setattr(sys, "stdin", SimpleNamespace(isatty=lambda: True))

    def run(
        planner,
        env,
        reader,
        *,
        sessions,
        auto_merge_memory=True,
    ):
        spec = replace(
            robot_spec.get_robot_spec(),
            init_runtime=lambda *a: (
                [],
                {"env": env, "model": None, "task_description": "test"},
            ),
        )
        monkeypatch.setattr(cli, "get_robot_spec", lambda name: spec)
        monkeypatch.setattr(cli, "start_interactive_reader", reader)
        monkeypatch.setattr(cli, "build_planner", lambda *a, **kw: planner)
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "rpent",
                "--robot",
                "dual_franka",
                "--planner",
                "codex",
                "--explore",
                "--interactive",
                "--explore-sessions",
                str(sessions),
                "--output-dir",
                str(tmp_path / "run"),
                "--memory-dir",
                str(tmp_path / "memory"),
                "--robot-config",
                str(dual_franka_robot_config),
                "--auto-merge-memory"
                if auto_merge_memory
                else "--no-auto-merge-memory",
            ],
        )
        return cli.main()

    return run


@pytest.mark.parametrize("last_verdict", ["failure", "abort", "unfinished"])
def test_success_memory_survives_a_later_unsuccessful_session(
    tmp_path, monkeypatch, last_verdict, run_interactive_cli
):
    env = FakeEnv()
    handlers = {}
    calls = []
    waiting_tools = []

    def reader(input_queue, **kwargs):
        handlers["line"] = kwargs["line_handler"]
        input_queue.put("test task")

    class Planner:
        def solve(self, *, toolkit, input_queue, **kwargs):
            assert not toolkit._scene_ready
            if calls:
                assert (tmp_path / "memory/task-specific/dual_franka_t0.json").is_file()
            verdict = "success" if not calls else last_verdict
            calls.append(verdict)
            if verdict == "unfinished":
                import threading

                ready = threading.Event()
                monkeypatch.setattr(
                    "builtins.print", lambda *args, **kwargs: ready.set()
                )
                worker = threading.Thread(
                    target=lambda: call(
                        toolkit, "request_scene_reset", reason="pending"
                    )
                )
                waiting_tools.append(worker)
                worker.start()
                assert ready.wait(2)
                assert handlers["line"]("/abort")
                return PlannerResult()
            toolkit._operator_input = lambda *args: "done"
            call(toolkit, "request_scene_reset", reason="test")
            call(toolkit, "move_delta", arm="right", delta_xyz=[0.01, 0, 0])
            assert handlers["line"]("/" + verdict)
            assert input_queue.get(timeout=1) is None
            return PlannerResult()

    sessions = 50 if last_verdict == "unfinished" else 2
    assert run_interactive_cli(Planner(), env, reader, sessions=sessions) == 0
    assert calls == ["success", last_verdict]
    published = json.loads(
        (tmp_path / "memory/task-specific/dual_franka_t0.json").read_text()
    )
    assert "session_001" in published["state_trace"]
    assert not (
        tmp_path / "run/sessions/session_002/dual_franka_t0_recipe.jsonl"
    ).exists()
    assert env.resets == (1 if last_verdict == "unfinished" else 2)
    for worker in waiting_tools:
        worker.join(2)
        assert not worker.is_alive()


@pytest.mark.parametrize("kind", ["reset", "verdict"])
def test_close_retires_waiting_request_without_robot_motion(setup, monkeypatch, kind):
    import threading

    from rpent.tools.human_in_the_loop import HumanInTheLoopInput

    t, env, replies = setup
    if kind == "verdict":
        reset(t, replies)
    resets_before = env.resets
    broker = HumanInTheLoopInput(interactive=True)
    t._operator_input = broker
    ready = threading.Event()
    monkeypatch.setattr("builtins.print", lambda *args, **kwargs: ready.set())
    result = []
    tool = "request_scene_reset" if kind == "reset" else "request_operator_verdict"
    kwargs = {"reason": "test cancellation"} if kind == "reset" else {}
    worker = threading.Thread(target=lambda: result.append(call(t, tool, **kwargs)))
    worker.start()
    assert ready.wait(2)
    old_id = broker._pending[0]
    t.close()
    worker.join(2)
    assert not worker.is_alive()
    assert broker.pending_kind is None
    assert env.resets == resets_before and not env.moves
    assert call(t, "request_scene_reset", reason="late call")["motion_refused"]
    assert broker.route_line(f"/operator {old_id} done")
    assert broker.pending_kind is None


@pytest.mark.parametrize("pending_reset", [False, True])
def test_attended_reply_resumes_same_toolkit_without_automatic_reset(
    setup, monkeypatch, tmp_path, pending_reset
):
    import queue
    import threading

    from rpent.cli.attended import _solve_attended
    from rpent.tools.human_in_the_loop import HumanInTheLoopInput

    t, env, replies = setup
    if not pending_reset:
        reset(t, replies)
    broker = HumanInTheLoopInput(interactive=True)
    t._operator_input = broker
    assert not broker.route_line("Keep the completed task stages")
    inputs = queue.Queue()
    first_reply = threading.Event()
    second_reply = threading.Event()
    prompt_ready = threading.Event()
    monkeypatch.setattr("builtins.print", lambda *a, **kw: prompt_ready.set())
    workers = []
    calls = []
    servers = []

    async def reset_over_http(url):
        import httpx
        from mcp import ClientSession
        from mcp.client.streamable_http import streamable_http_client

        async with httpx.AsyncClient(trust_env=False) as client:
            async with streamable_http_client(url, http_client=client) as (
                read,
                write,
                _,
            ):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    result = await session.call_tool(
                        "request_scene_reset", {"reason": "test"}
                    )
                    assert not result.isError

    class Planner:
        def solve(self, *, toolkit, user_message, **kwargs):
            assert toolkit is t
            server = kwargs["mcp_server"]
            servers.append(server)
            assert server._thread.is_alive() and not server._server.should_exit
            calls.append(user_message)
            if len(calls) == 1:
                if pending_reset:
                    import asyncio

                    worker = threading.Thread(
                        target=lambda: asyncio.run(reset_over_http(server.url))
                    )
                    worker.start()
                    workers.append(worker)
                    assert prompt_ready.wait(2)
                first_reply.set()
                return PlannerResult(
                    messages=[{"role": "assistant", "content": "Please confirm"}]
                )
            assert "SAME attempt" in user_message
            assert user_message.count("Operator feedback, chronological") == 1
            assert "Keep the completed task stages" in user_message
            assert t._scene_ready
            assert env.resets == 1
            t.request_direct_verdict("success", "test complete")
            second_reply.set()
            return PlannerResult()

    results = []
    runner = threading.Thread(
        target=lambda: results.append(
            _solve_attended(
                Planner(),
                operator_input=broker,
                state_output_dir=tmp_path / "session",
                keep_mcp_alive=True,
                toolkit=t,
                input_queue=inputs,
                system_prompt="test",
                user_message="original task" + broker.feedback_prompt(),
                max_turns=10,
            )
        )
    )
    runner.start()
    assert first_reply.wait(2)
    assert not second_reply.wait(0.15)
    if pending_reset:
        assert broker.route_line("/done 保留前半段成果")
    else:
        inputs.put("场景已调整，请继续后半段")
    runner.join(3)
    for worker in workers:
        worker.join(2)
    assert not runner.is_alive() and len(results) == 1
    assert len(calls) == 2 and env.resets == 1
    assert len(list((tmp_path / "session").glob("dialogue_*.json"))) == 2
    assert not env.moves
    assert servers[0] is servers[1]
    assert servers[0]._thread is None


@pytest.mark.parametrize("input_closed", [False, True])
def test_attended_wait_eof_aborts_without_another_planner_call(
    setup, tmp_path, monkeypatch, input_closed
):
    import queue

    from rpent.cli.attended import _solve_attended
    from rpent.tools.human_in_the_loop import HumanInTheLoopInput

    t, env, _ = setup
    inputs = queue.Queue()
    inputs.put(None)
    calls = []
    broker = HumanInTheLoopInput(interactive=True)
    if input_closed:
        broker.close()
        monkeypatch.setattr(
            t,
            "wait_active",
            lambda: pytest.fail("closed input must abort before waiting"),
        )

    class Planner:
        def solve(self, **kwargs):
            calls.append(True)
            return PlannerResult()

    _solve_attended(
        Planner(),
        operator_input=broker,
        state_output_dir=tmp_path / "session",
        toolkit=t,
        input_queue=inputs,
        user_message="task",
        system_prompt="",
        max_turns=10,
    )
    assert calls == [True]
    assert t.finalize_direct_verdict()["operator_aborted"]
    assert not env.resets and not env.moves


@pytest.mark.parametrize("reply_style", ["shortcut", "request_id"])
def test_attended_consumes_continue_once_without_duplicate_wait(
    setup, tmp_path, monkeypatch, reply_style
):
    import queue
    import threading

    from rpent.cli.attended import _solve_attended
    from rpent.tools.human_in_the_loop import HumanInTheLoopInput

    t, env, _ = setup
    broker = HumanInTheLoopInput(interactive=True)
    ready = threading.Event()
    monkeypatch.setattr("builtins.print", lambda *a, **kw: ready.set())
    calls = []
    inputs = queue.Queue()
    inputs.put(None)

    class Planner:
        def solve(self, **kwargs):
            calls.append(kwargs["user_message"])
            if len(calls) == 1:
                replies = []
                thread = threading.Thread(
                    target=lambda: replies.append(
                        broker.request("continue?", lambda: None, kind="verdict")
                    )
                )
                thread.start()
                assert ready.wait(2)
                command = (
                    "/continue"
                    if reply_style == "shortcut"
                    else f"/operator {broker._pending[0]} continue"
                )
                assert broker.route_line(command)
                thread.join(2)
                assert replies == ["continue"]
            return PlannerResult()

    _solve_attended(
        Planner(),
        operator_input=broker,
        state_output_dir=tmp_path / "session",
        toolkit=t,
        input_queue=inputs,
        user_message="task",
        system_prompt="",
        max_turns=10,
    )
    assert len(calls) == 2
    assert "already answered continue" in calls[1]
    assert broker.continue_revision == 1
    assert t.finalize_direct_verdict()["operator_aborted"]
    assert not env.resets and not env.moves


@pytest.mark.parametrize("phase", ["active", "reply_returned", "operator_request"])
def test_attended_verdict_preserves_next_attempt_input(
    setup, tmp_path, monkeypatch, phase
):
    import queue
    import threading

    from rpent.cli import attended
    from rpent.planner.utils.http_mcp_server import HttpMcpServer
    from rpent.session.input import SessionInputQueue
    from rpent.tools.human_in_the_loop import HumanInTheLoopInput
    from tests.unit_tests.rpent.planner.test_codex_contracts import (
        FakeCodex,
        FakeTurn,
        RecordingSink,
        install_fake_backend,
        make_planner,
    )

    install_fake_backend(monkeypatch)
    monkeypatch.setattr("rpent.planner.codex.HttpMcpServer", HttpMcpServer)
    toolkit, env, replies = setup
    reset(toolkit, replies)
    inputs = queue.Queue()
    scope = SessionInputQueue(inputs)
    broker = HumanInTheLoopInput(interactive=True)
    ready, interrupted = threading.Event(), threading.Event()

    def verdict(outcome, notes):
        assert toolkit.request_direct_verdict(outcome, notes)
        scope.cancel()
        broker.cancel_pending()
        return True

    broker.bind_verdict(verdict)
    request = None
    if phase == "operator_request":
        request_ready = threading.Event()
        monkeypatch.setattr("builtins.print", lambda *a, **kw: request_ready.set())
        request = threading.Thread(
            target=lambda: broker.request("Evaluate?", lambda: None, kind="verdict")
        )
        request.start()
        assert request_ready.wait(3)

    def info(message, *args):
        if message.startswith("Waiting for"):
            ready.set()

    monkeypatch.setattr(attended.logger, "info", info)

    def interrupt(turn):
        turn.interrupt_calls += 1
        interrupted.set()

    def stream(turn):
        if phase == "active":
            ready.set()
            assert interrupted.wait(3)
        yield from turn.events

    monkeypatch.setattr(FakeTurn, "interrupt", interrupt)
    monkeypatch.setattr(FakeTurn, "stream", stream)
    planner = make_planner(tmp_path, RecordingSink(), timeout_s=5)
    results = []
    runner = threading.Thread(
        target=lambda: results.append(
            attended._solve_attended(
                planner,
                operator_input=broker,
                state_output_dir=tmp_path / "attempt1",
                keep_mcp_alive=True,
                toolkit=toolkit,
                input_queue=scope,
                system_prompt="test",
                user_message="task",
                max_turns=10,
            )
        )
    )
    runner.start()
    try:
        assert ready.wait(3)
        assert broker.route_line("/success")
        inputs.put("Focus on the later stages")
        runner.join(5)
        assert not runner.is_alive() and results[0].error is None
        assert FakeCodex.instances[0].thread.fake_turn.interrupt_calls == int(
            phase == "active"
        )
    finally:
        scope.cancel()
        broker.close()
        interrupted.set()
        runner.join(5)
        if request is not None:
            request.join(3)
            assert not request.is_alive()

    delivered = threading.Event()

    def steer(turn, text):
        turn.steered.append(text)
        delivered.set()

    def next_stream(turn):
        assert delivered.wait(3)
        yield from turn.events

    monkeypatch.setattr(FakeTurn, "steer", steer)
    monkeypatch.setattr(FakeTurn, "stream", next_stream)
    next_toolkit = DualFrankaToolkit(
        runtime_kwargs={"env": env, "model": None, "task_description": "test"},
        dashboard_events=NullDashboardEventSink(),
        memory=toolkit.memory,
        mode="exploration",
        state_output_dir=tmp_path / "attempt2",
        operator_input=broker,
    )
    result = planner.solve(
        system_prompt="test",
        user_message="next attempt",
        toolkit=next_toolkit,
        max_turns=10,
        input_queue=SessionInputQueue(inputs),
    )
    assert result.error is None
    turn = FakeCodex.instances[-1].thread.fake_turn
    assert turn.steered == ["Focus on the later stages"] and turn.interrupt_calls == 0
    assert inputs.empty()


def test_continue_captures_scene_after_operator_adjustment(setup, monkeypatch):
    t, env, replies = setup
    reset(t, replies)
    original = env.get_observation
    scene = [10]
    reads = []
    published = []
    monkeypatch.setattr(t, "_publish_step", published.append)

    def observe():
        reads.append(scene[0])
        obs = original()
        obs["d455_images"][:] = scene[0]
        return obs

    env.get_observation = observe

    def operator(*args):
        scene[0] = 90
        return "continue moved object"

    t._operator_input = operator
    observed = t.execute_tool("request_operator_verdict", {})
    result = observed.data
    assert result["status"] == "continue"
    assert np.all(t.state.load("d455.png", step=result["evidence_step"]) == 10)
    assert np.all(t.state.load("d455.png", step=result["observation_step"]) == 90)
    assert observed.images == [
        t.state.load_bytes("d455.png", step=result["observation_step"])
    ]
    assert result["observation_step"] == result["evidence_step"] + 1
    assert reads == [10, 90]
    assert [record.step_idx for record in published] == [
        result["evidence_step"],
        result["observation_step"],
    ]
    assert [record.command for record in published] == [
        {"action": "observe_for_verdict"},
        {"action": "observe_current"},
    ]
    assert env.resets == 1 and not env.moves
