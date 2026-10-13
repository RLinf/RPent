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
def setup(tmp_path, monkeypatch, dual_franka_robot_config, request):
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
        mode=getattr(request, "param", "exploration"),
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

    env = FakeEnv()
    runtimes = []
    planners = []
    replies = iter(["done", "failure dropped", "done", "success lifted"])

    class Operator:
        closed = False

        def __init__(self, **kwargs):
            assert kwargs["interactive"] is False

        def __call__(self, prompt, cancelled):
            cancelled()
            return next(replies)

        def close(self):
            self.closed = True

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
        argv.extend(["--dashboard", "--interactive"])
    monkeypatch.setattr(sys, "argv", argv)
    monkeypatch.setattr(
        cli, "build_planner", lambda *a, **k: pytest.fail("planner should not start")
    )
    with pytest.raises(SystemExit) as exc:
        cli.main()
    assert exc.value.code == 2
    assert (
        "cannot be used together" if dashboard else "TTY"
    ) in capsys.readouterr().err


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
    assert not t.request_direct_verdict("success")  # idempotent, never another attempt
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


@pytest.mark.parametrize("robot_name", ["dual_franka", "libero"])
@pytest.mark.parametrize(
    "verdict",
    [
        "success",
        "failure",
        "abort",
        "eof",
        "success-failure",
        "success-error",
        "success-eof",
    ],
)
@pytest.mark.parametrize(
    "planner_error,auto_merge",
    [
        (None, True),
        (None, False),
        ("planner transport failed", True),
        ("planner timed out", True),
    ],
)
def test_cli_direct_verdict_finalizes_and_merges_only_without_errors(
    tmp_path,
    monkeypatch,
    dual_franka_robot_config,
    verdict,
    planner_error,
    auto_merge,
    robot_name,
):
    import sys
    from dataclasses import replace
    from types import SimpleNamespace

    from rpent.cli import main as cli

    env = FakeEnv()
    handlers = {}
    sessions = []

    def reader(input_queue, **kwargs):
        handlers["line"] = kwargs["line_handler"]
        handlers["close"] = kwargs["on_close"]
        input_queue.put("test task")

    monkeypatch.setattr(cli, "start_interactive_reader", reader)

    class Planner:
        def solve(self, *, toolkit, input_queue, **kwargs):
            sessions.append(toolkit)
            current = (
                verdict.split("-")[0] if len(sessions) == 1 else verdict.split("-")[-1]
            )
            error = planner_error or ("backend failed" if current == "error" else None)
            audit_path = tmp_path / "run/dual_franka_t0.json"
            assert not audit_path.exists()
            if (
                current == "success"
                and error is None
                and (len(sessions) == 1 or not auto_merge)
            ):
                audit_path.write_text(json.dumps({"agent_session": len(sessions)}))
            toolkit._operator_input = lambda *args: "done"
            call(toolkit, "request_scene_reset", reason="test")
            call(toolkit, "move_delta", arm="right", delta_xyz=[0.01, 0, 0])
            inbox = toolkit.memory.root / "_internal/inbox/dual_franka_t0"
            inbox.mkdir(parents=True, exist_ok=True)
            (inbox / "new_global_strategy_lesson.md").write_text(
                "---\nid: lesson\nscope: global\nkind: strategy\ntitle: Lesson\n"
                "applies_when: Same setup\nconfidence: single-shot\n"
                "evidence:\n  cells: [dual_franka_t0]\n---\nObserved attempt.\n"
            )
            if current == "eof":
                handlers["close"]()
                input_queue.put(None)
                assert input_queue.get() is None
            else:
                assert handlers["line"](
                    "/" + ("failure" if current == "error" else current)
                )
            from rpent.session.input import InputCancelled

            if current != "eof":
                with pytest.raises(InputCancelled):
                    input_queue.get()
            return SimpleNamespace(
                finish_result=None, messages=[], stats={}, error=error
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
            "--planner",
            "codex",
            "--explore",
            "--interactive",
            "--explore-sessions",
            "2",
            *([] if auto_merge else ["--no-auto-merge-memory"]),
            "--output-dir",
            str(tmp_path / "run"),
            "--memory-dir",
            str(tmp_path / "memory"),
            "--robot-config",
            str(dual_franka_robot_config),
        ],
    )
    assert cli.main() == int(bool(planner_error or verdict == "success-error"))
    assert len(sessions) == (
        2 if verdict not in {"abort", "eof"} and not planner_error else 1
    )
    for i in range(1, len(sessions) + 1):
        assert (
            tmp_path / f"run/sessions/session_{i:03d}/transcript_dual_franka_t0.json"
        ).is_file()
        if (
            verdict.startswith("success")
            and (i == 1 or verdict == "success")
            and planner_error is None
        ):
            audit = json.loads(
                (
                    tmp_path / f"run/sessions/session_{i:03d}/dual_franka_t0.json"
                ).read_text()
            )
            expected_agent = {"agent_session": i} if i == 1 or not auto_merge else None
            assert audit["agent_audit"] == expected_agent
            assert f"session_{i:03d}" in audit["state_trace"]
    session_dir = tmp_path / "run/sessions/session_001"
    assert (session_dir / "dual_franka_t0.json").is_file() == verdict.startswith(
        "success"
    )
    assert (session_dir / "dual_franka_t0_recipe.jsonl").is_file() == (
        verdict.startswith("success") and planner_error is None
    )
    assert (
        json.loads((session_dir / "transcript_dual_franka_t0.json").read_text())[
            "error"
        ]
        == planner_error
    )
    published = (
        auto_merge
        and not planner_error
        and verdict in {"success", "success-failure", "failure"}
    )
    assert (tmp_path / "memory/global/lesson.md").is_file() == published
    assert (tmp_path / "memory/task-specific/dual_franka_t0.json").is_file() == (
        published and verdict != "failure"
    )
    assert (
        tmp_path / "memory/task-specific/dual_franka_t0_recipe.jsonl"
    ).is_file() == (published and verdict != "failure")
    if published and verdict != "failure":
        assert json.loads(
            (tmp_path / "memory/task-specific/dual_franka_t0.json").read_text()
        )["agent_audit"] == {"agent_session": 1}
    events = json.loads(
        (tmp_path / "run/sessions/session_001/operator_events.json").read_text()
    )
    assert events[-1]["status"] == (
        "failure" if verdict in {"abort", "eof"} else verdict.split("-")[0]
    )
    assert events[-1]["operator_finished"] is (verdict in {"abort", "eof"})


def test_direct_failure_seals_session_without_success_memory_even_before_reset(setup):
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


@pytest.mark.parametrize("setup", ["exploration", "evaluation"], indirect=True)
def test_fresh_observation_and_pixel_provenance(setup):
    from rpent.tools import ToolResult

    t, env, replies = setup
    if t._mode == "exploration":
        reset(t, replies)
    old = t.state.latest_step
    env.resets += 1  # Out-of-band scene change, visible only on capture.
    assert not call(t, "view_env_state", step=old).get("error")
    assert t.state.latest_step == old
    fresh = call(t, "view_env_state")
    assert t.state.latest_step == old + 1
    assert not fresh.get("error")
    assert t.state.load("left_wrist.png")[0, 0, 0] == env.resets
    latest = t.state.latest_step
    for tool, coords in (
        ("back_project", {"row": 1, "col": 1}),
        ("segment", {"point": [1, 1]}),
    ):
        seen = []
        for step in (None, -1, old, latest + 1):
            assert call(t, tool, **coords, step=step).get("error")
        result = t._current_perception(
            lambda **kw: seen.append(kw) or ToolResult(data={}), **coords, step=latest
        )
        assert not result.is_error and seen[0]["step"] == latest
    env.fail_observation = True
    assert call(t, "view_env_state").get("error")


def test_fresh_capture_waits_for_active_motion(setup, monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Event

    t, env, replies = setup
    reset(t, replies)
    moving, release = Event(), Event()
    observe = env.get_observation

    def move(*args):
        moving.set()
        assert release.wait(3)
        moving.clear()
        return {"ok": True}

    def capture():
        assert not moving.is_set()
        return observe()

    monkeypatch.setattr(env, "move_delta", move)
    monkeypatch.setattr(env, "get_observation", capture)
    with ThreadPoolExecutor() as pool:
        motion = pool.submit(call, t, "move_delta", arm="right", delta_xyz=[0, 0, 0.01])
        assert moving.wait(2)
        fresh = pool.submit(call, t, "view_env_state")
        try:
            with t._scheduler.condition:
                assert t._scheduler.condition.wait_for(
                    lambda: bool(t._scheduler.pending), timeout=2
                )
            assert not fresh.done()
        finally:
            release.set()
        assert not motion.result(3).get("error")
        assert not fresh.result(3).get("error")
