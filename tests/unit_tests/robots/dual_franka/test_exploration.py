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
def setup(tmp_path, monkeypatch):
    from rpent.utils import logging

    monkeypatch.setattr(logging, "_output_dir", tmp_path)
    monkeypatch.setattr("robots.dual_franka.toolkit.get_output_dir", lambda: tmp_path)
    env = FakeEnv()
    replies = []
    toolkit = DualFrankaToolkit(
        primitives_kwargs={"env": env, "model": None, "task_description": "test"},
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
    return t.execute_tool(name, kwargs).result


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


def test_operator_abort_allows_finish_even_with_budget_and_never_succeeds(setup):
    t, env, replies = setup
    replies.append("abort")
    call(t, "request_scene_reset", reason="initial")
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


def test_explore_prompt_and_factory_use_local_layered_memory(tmp_path):
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
        ]
    )
    config = robot_spec.get_robot_spec().parse_config(args)
    prompt = robot_spec.get_robot_spec().prompts.render(
        "system", variables={**config.prompt_vars, "output_dir": tmp_path}
    )
    assert "request_scene_reset" in prompt and "request_operator_verdict" in prompt
    assert (
        "scope: global" in prompt and "scope: suite" in prompt and "task_only" in prompt
    )
    assert "{{" not in prompt and "libero_terminated" not in prompt
    t = robot_spec.get_toolkit(
        primitives_kwargs={"env": FakeEnv(), "model": None, "task_description": "test"},
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
    (inbox / "suite_dual_franka_t0_draft.md").write_text("""---
id: suite_dual_franka_real_t0
scope: suite
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
    assert result["global"] == result["suite"] == result["task"] == 1
    assert not t.memory.validate()
    assert (t.memory.root / "MEMORY.md").exists()
    assert (t.memory.root / "task_only/dual_franka_t0_recipe.jsonl").exists()


def test_cli_two_sessions_operator_feedback_and_memory_pipeline(tmp_path, monkeypatch):
    import sys
    from dataclasses import replace
    from types import SimpleNamespace

    from rpent.cli import main as cli

    env = FakeEnv()
    runtimes = []
    planners = []
    replies = iter(["done", "failure dropped", "done", "success lifted"])

    class Operator:
        def __init__(self, **kwargs):
            pass

        def __call__(self, prompt, cancelled):
            cancelled()
            return next(replies)

        def close(self):
            pass

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
            "calibration_path": "/tmp/unused-calibration.json",
        }

    spec = replace(robot_spec.get_robot_spec(), init_runtime=init_runtime)
    monkeypatch.setattr(cli, "get_robot_spec", lambda name: spec)
    monkeypatch.setattr(cli, "build_planner", Planner)
    monkeypatch.setattr("rpent.tools.human_in_the_loop.HumanInTheLoopInput", Operator)
    monkeypatch.setattr(sys, "stdin", SimpleNamespace(isatty=lambda: True))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "rpent",
            "--robot",
            "dual_franka",
            "--explore",
            "--explore-sessions",
            "2",
            "--explore-attempts-per-session",
            "1",
            "--output-dir",
            str(tmp_path / "run"),
            "--memory-dir",
            str(tmp_path / "memory"),
            "--auto-merge-memory",
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
    audit = json.loads((tmp_path / "memory/task_only/dual_franka_t0.json").read_text())
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

    t.add_tool("move_delta", t._tools["move_delta"][0], active_motion)
    worker = threading.Thread(target=lambda: results.append(call(t, "move_delta")))
    worker.start()
    assert entered.wait(2)
    assert t.request_direct_verdict("success")
    assert t.request_direct_verdict("success")  # idempotent, never another attempt
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
    audit = json.loads((tmp_path / "memory/task_only/dual_franka_t0.json").read_text())
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


@pytest.mark.parametrize("robot_name", ["dual_franka", "libero"])
@pytest.mark.parametrize("verdict", ["success", "failure", "abort"])
@pytest.mark.parametrize("planner_error", [None, "planner transport failed"])
def test_cli_direct_verdict_finalizes_and_merges_only_without_errors(
    tmp_path, monkeypatch, verdict, planner_error, robot_name
):
    import sys
    from dataclasses import replace
    from types import SimpleNamespace

    from rpent.cli import main as cli

    env = FakeEnv()
    handlers = {}

    def reader(input_queue, **kwargs):
        handlers["line"] = kwargs["line_handler"]
        input_queue.put("test task")

    monkeypatch.setattr(cli, "start_interactive_reader", reader)

    class Planner:
        def solve(self, *, toolkit, input_queue, **kwargs):
            toolkit._operator_input = lambda *args: "done"
            call(toolkit, "request_scene_reset", reason="test")
            call(toolkit, "move_delta", arm="right", delta_xyz=[0.01, 0, 0])
            assert handlers["line"]("/" + verdict)
            assert input_queue.get(timeout=1) is None
            return SimpleNamespace(
                finish_result=None, messages=[], stats={}, error=planner_error
            )

    spec = replace(
        robot_spec.get_robot_spec(),
        init_runtime=lambda *args: (
            [],
            {
                "env": env,
                "model": None,
                "task_description": "test",
            },
        ),
    )
    monkeypatch.setattr(cli, "get_robot_spec", lambda name: spec)
    original_toolkit_factory = cli.get_toolkit
    monkeypatch.setattr(
        cli,
        "get_toolkit",
        lambda name, **kwargs: original_toolkit_factory("dual_franka", **kwargs),
    )
    monkeypatch.setattr(cli, "build_planner", lambda *args, **kwargs: Planner())
    monkeypatch.setattr(sys, "stdin", SimpleNamespace(isatty=lambda: True))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "rpent",
            "--robot",
            robot_name,
            "--explore",
            "--interactive",
            "--output-dir",
            str(tmp_path / "run"),
            "--memory-dir",
            str(tmp_path / "memory"),
        ],
    )
    assert cli.main() == (1 if planner_error else 0)
    assert (tmp_path / "memory/task_only/dual_franka_t0.json").is_file() == (
        verdict == "success" and planner_error is None
    )
    assert (tmp_path / "memory/task_only/dual_franka_t0_recipe.jsonl").is_file() == (
        verdict == "success" and planner_error is None
    )
    events = json.loads(
        (tmp_path / "run/sessions/session_001/operator_events.json").read_text()
    )
    assert events[-1]["status"] == ("failure" if verdict == "abort" else verdict)
    assert events[-1]["operator_finished"] is True


def test_direct_failure_ends_without_success_memory_even_before_reset(setup):
    t, env, _ = setup
    assert t.request_direct_verdict("failure")
    assert t.request_direct_verdict("failure")
    assert not t.request_direct_verdict("success")
    assert call(t, "request_scene_reset", reason="late reset")["motion_refused"]
    result = t.finalize_direct_verdict()
    assert result["status"] == "failure" and result["operator_finished"]
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
