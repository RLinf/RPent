# Copyright 2026 The RPent Authors.
# Licensed under the Apache License, Version 2.0 (the "License").

"""Dashboard ownership and real-outcome contracts; never start hardware."""

import json
from types import SimpleNamespace

import pytest

from robots.yam import robot_spec as yam
from robots.yam.vla_test import VLATest
from rpent.cli import dashboard, main
from rpent.dashboard.state import DashboardState, InteractionUnavailableError
from rpent.planner.base import PlannerResult


def args_for(tmp_path, *extra):
    parser = main._build_argparser()
    yam.get_robot_spec().add_cli_args(parser, use_dashboard=True)
    return parser.parse_args(
        [
            "--robot",
            "yam",
            "--dashboard",
            "--task-name",
            "tabletop_cleanup_a",
            "--output-dir",
            str(tmp_path),
            "--env-endpoint",
            "socket://127.0.0.1:8110",
            *extra,
        ]
    )


def test_defaults_and_explicit_overrides(tmp_path):
    assert yam.get_robot_spec().is_real_robot
    args = args_for(tmp_path)
    common = main._build_argparser().parse_args([])
    assert (args.planner, args.model, args.reasoning_effort) == (
        common.planner,
        common.model,
        common.reasoning_effort,
    )
    assert not args.explore and args.auto_merge_memory
    assert args.memory_profile == "local"
    assert args.explore_attempts_per_session == 5
    assert args.task_name == "tabletop_cleanup_a"  # explicitly supplied by fixture
    assert (args.dashboard_host, args.dashboard_port) == (
        common.dashboard_host,
        common.dashboard_port,
    )
    override = args_for(
        tmp_path,
        "--memory-profile",
        "hf",
        "--model",
        "other",
        "--reasoning-effort",
        "medium",
        "--explore-attempts-per-session",
        "3",
    )
    assert override.memory_profile == "hf"
    assert (
        override.model,
        override.reasoning_effort,
        override.explore_attempts_per_session,
    ) == ("other", "medium", 3)


@pytest.mark.parametrize("endpoint", [False, True])
def test_external_dashboard_checks_endpoint_without_operator_tty(
    tmp_path, monkeypatch, capsys, endpoint
):
    import sys

    argv = [
        "rpent",
        "--robot",
        "yam",
        "--dashboard",
        "--planner",
        "codex",
        "--output-dir",
        str(tmp_path),
    ]
    if endpoint:
        argv += ["--env-endpoint", "socket://localhost:8110"]
    monkeypatch.setattr(sys, "argv", argv)
    monkeypatch.setattr(sys, "stdin", SimpleNamespace(isatty=lambda: False))

    def reached_startup(*args, **kwargs):
        raise RuntimeError("dashboard validation passed")

    monkeypatch.setattr(dashboard, "init_output_dir", reached_startup)
    if endpoint:
        with pytest.raises(RuntimeError, match="dashboard validation passed"):
            main.main()
    else:
        with pytest.raises(SystemExit):
            main.main()
        assert "--env-endpoint is required" in capsys.readouterr().err


def test_dashboard_tasks_keep_language_separate(tmp_path):
    state = DashboardState(output_dir=tmp_path, dashboard_spec=yam.YAM_DASHBOARD_SPEC)
    state.shared_services_ready()
    for task, seed in (("tabletop_cleanup_a", 0), ("tabletop_cleanup_b", 1)):
        state.submit_input(f"/rpent-task {task} {seed}")
        claimed = state.wait_for_task(timeout=0)
        args = args_for(tmp_path)
        for key, value in claimed.request.items():
            setattr(args, key, value)
        config = yam.get_robot_spec().parse_config(args)
        assert config.prompt_vars["instruction"] == yam.TASK_INSTRUCTIONS[task]
        state.complete_task(state="failed", error=None)
    assert (
        yam.TASK_INSTRUCTIONS["tabletop_cleanup_a"]
        != yam.TASK_INSTRUCTIONS["tabletop_cleanup_b"]
    )
    args.task_language = "stale A instruction"
    with pytest.raises(ValueError, match="omit --task-language"):
        yam.get_robot_spec().parse_config(args)


@pytest.mark.parametrize("started,language", [(False, None), (True, "wrong task")])
def test_preflight_rejects_before_observation(tmp_path, started, language):
    calls = []

    class RPC:
        def call(self, method, **kwargs):
            calls.append(method)
            if method == "env.is_started":
                return started
            if method == "env.get_task_language":
                return language
            pytest.fail(f"unexpected RPC with possible hardware side effects: {method}")

    with pytest.raises((ValueError, RuntimeError)):
        yam._build_env_runtime_kwargs(args_for(tmp_path), RPC())
    assert "env.observe" not in calls


def test_passive_started_check_does_not_initialize():
    import threading

    from robots.yam.env_server import YamEnvFacade
    from robots.yam.rlinf_env import YamAgentEnv

    env = object.__new__(YamAgentEnv)
    env._lock = threading.RLock()
    env._started = False
    env._closed = False
    env._startup_failed = False
    env._ensure_started = lambda: pytest.fail("must not initialize")
    facade = YamEnvFacade(env, metadata={"test": True})
    assert facade._dispatch("env.is_started", [], {}) is False
    env._started = True
    assert facade._dispatch("env.is_started", [], {}) is True
    env._closed = True
    assert facade._dispatch("env.is_started", [], {}) is False


def test_external_runtime_does_not_own_daemons(tmp_path, monkeypatch):
    from rpent.dashboard.events import NullDashboardEventSink

    monkeypatch.setattr(yam, "make_rpc_client", lambda endpoint: object())
    monkeypatch.setattr(yam, "try_wait_server", lambda *a, **k: k["post_fn"]())
    monkeypatch.setattr(
        yam, "_build_env_runtime_kwargs", lambda *a: {"env": "external-env"}
    )
    monkeypatch.setattr(
        yam, "_build_vla_runtime_kwargs", lambda *a: {"model": "external-vla"}
    )
    args = args_for(tmp_path, "--vla-endpoint", "socket://gpu:8220")
    owned, components = yam._init_runtime(
        args, tmp_path, NullDashboardEventSink(), {"env", "vla"}
    )
    assert owned == []
    assert components == {"env": "external-env", "model": "external-vla"}


@pytest.mark.parametrize(
    "solved,planner_error,replaced",
    [
        (False, None, False),
        (True, None, False),
        (False, "stopped", False),
        (False, None, True),
        (False, "keyboard-interrupt", False),
    ],
)
def test_task_result_memory_and_cleanup(
    tmp_path, monkeypatch, solved, planner_error, replaced
):
    events = []
    seen = {}
    state = DashboardState(output_dir=tmp_path, dashboard_spec=yam.YAM_DASHBOARD_SPEC)
    state.shared_services_ready()
    state.request_task({"task_name": "tabletop_cleanup_a", "seed": 0})
    claimed = state.wait_for_task(timeout=0)

    class Memory:
        def merge_memory(self, **kwargs):
            events.append("merge")
            seen["merge"] = kwargs
            return {"global": 1}

    class Toolkit:
        memory = Memory()

        def cancel_active_and_wait(self):
            events.append("cancel")

        def solved(self):
            return solved

        def write_recipe(self, tag):
            events.append("recipe")
            return str(tmp_path / "recipe.jsonl")

        def close(self):
            with pytest.raises(InteractionUnavailableError):
                state.execute_primitive("move_to", {})
            events.append("hold")

    toolkit = Toolkit()

    def get_toolkit(*args, **kwargs):
        seen["toolkit"] = kwargs
        return toolkit

    def solve(**kwargs):
        state.set_planner_activity("idle", accepting_input=True)
        if planner_error == "keyboard-interrupt":
            raise KeyboardInterrupt()
        if replaced:
            state.request_task({"task_name": "tabletop_cleanup_b", "seed": 1})
        return PlannerResult(
            finish_result={"status": "success"},
            messages=[],
            stats={},
            error=planner_error,
        )

    monkeypatch.setattr(dashboard, "get_toolkit", get_toolkit)
    monkeypatch.setattr(
        dashboard, "build_planner", lambda *a, **k: SimpleNamespace(solve=solve)
    )
    spec = yam.get_robot_spec()
    from dataclasses import replace

    spec = replace(spec, init_runtime=lambda *a: ([], {}))
    args = args_for(
        tmp_path,
        "--explore",
        "--explore-attempts-per-session",
        "50",
        "--memory-dir",
        str(tmp_path / "memory"),
    )
    error = dashboard._run_dashboard_task(
        args=args,
        robot_spec=spec,
        state=state,
        claimed=claimed,
        shared_primitives_kwargs={},
        unique_components={"env"},
        session_root=tmp_path,
    )
    assert seen["toolkit"]["mode"] == "exploration"
    assert seen["toolkit"]["attempts_per_session"] == 50
    assert events.index("cancel") < events.index("hold") < events.index("merge")
    assert state._toolkit is None
    assert ("recipe" in events) == solved
    assert seen["merge"]["solved"] == solved
    result = json.loads((claimed.output_dir / "result.json").read_text())
    assert result["environment_success"] is solved
    assert result["status"] == ("success" if solved else "not_successful")
    assert bool(error) is (not solved)


@pytest.mark.parametrize("task_id", [103, 104])
def test_diagnostic_task_routed_before_planner(monkeypatch, task_id):
    import sys

    from robots.yam import manual
    from rpent.cli import main

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "rpent",
            "--robot",
            "yam",
            "--task-id",
            str(task_id),
            "--task-name",
            "tabletop_cleanup_a",
            "--env-endpoint",
            "socket://localhost:8110",
        ],
    )
    seen = []
    monkeypatch.setattr(
        manual, "run_session", lambda args: seen.append(args.task_id) or 0
    )
    assert main.main() == 0
    assert seen == [task_id]


def test_infer_is_no_motion_and_execution_single_use(primitives, env, tmp_path):
    test = VLATest(primitives, tmp_path)
    assert test.infer()["inference_only"] and not env._runtime.commands
    assert test.execute(20)["executed_actions"] == 20
    with pytest.raises(RuntimeError):
        test.execute()
    assert len(env._runtime.commands) == 20
    assert len(list(tmp_path.glob("*.npz"))) == 1


@pytest.mark.parametrize("fault", ["episode", "position", "count", "stop", "expired"])
def test_stale_prediction_never_executes(primitives, env, tmp_path, clock, fault):
    test = VLATest(primitives, tmp_path)
    test.infer()
    if fault == "episode":
        env._episode_id = "next"
    elif fault == "position":
        env._runtime.qpos[0] += 0.03
    elif fault == "count":
        env._take_action_cnt += 1
    elif fault == "stop":
        env.request_stop()
    else:
        clock.now += 31
    with pytest.raises(RuntimeError):
        test.execute()
    assert not env._runtime.commands


def test_uncertain_execution_cannot_replay(primitives, env, tmp_path, monkeypatch):
    test = VLATest(primitives, tmp_path)
    test.infer()
    calls = []

    def uncertain(*args, **kwargs):
        calls.append(1)
        raise TimeoutError("unknown execution outcome")

    monkeypatch.setattr(primitives.env, "chunk_step", uncertain)
    with pytest.raises(TimeoutError):
        test.execute()
    with pytest.raises(RuntimeError):
        test.infer()
    with pytest.raises(RuntimeError):
        test.execute()
    assert calls == [1]


@pytest.mark.parametrize(
    "task_id,commands",
    [(103, ['{"tool":"status"}', "quit"]), (104, ['{"tool":"infer"}', "quit"])],
)
def test_diagnostic_console_does_not_implicitly_move(
    primitives, env, tmp_path, monkeypatch, task_id, commands
):
    from robots.yam.manual import run_session

    args = args_for(
        tmp_path, "--task-id", str(task_id), "--vla-endpoint", "socket://gpu:8220"
    )
    monkeypatch.setattr(
        "rpent.utils.rpc.make_rpc_client",
        lambda _: SimpleNamespace(call=lambda *a, **k: True),
    )
    monkeypatch.setattr(
        yam,
        "_init_runtime",
        lambda *a: ([], {"env": primitives.env, "model": primitives.model}),
    )
    inputs = iter(commands)
    monkeypatch.setattr("builtins.input", lambda _: next(inputs))
    assert run_session(args) == 0
    assert not env._runtime.commands and env._stop_requested.is_set()
    assert "close" not in env._runtime.events
