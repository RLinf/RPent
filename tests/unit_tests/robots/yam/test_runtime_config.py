# Copyright 2026 The RPent Authors.
# Licensed under the Apache License, Version 2.0.

import sys
from types import SimpleNamespace

import pytest
import yaml

from robots.yam import runtime_config
from robots.yam.env_server import YamEnvFacade


def test_yaml_combines_hardware_with_cli_task(tmp_path):
    raw = runtime_config.load_mapping()
    assert not raw["reset"]["enabled"] and not raw["park_on_close"]["enabled"]
    assert [camera["name"] for camera in raw["cameras"]] == ["top", "left", "right"]
    raw["park_on_close"].update(
        enabled=True, left_qpos=[0.0] * 6 + [1.0], right_qpos=[0.0] * 6 + [1.0]
    )
    path = tmp_path / "robot.yaml"
    path.write_text(yaml.safe_dump(raw))
    config = runtime_config.load_config(
        path,
        task_id=1,
        task_language="Place the cube",
        seed=7,
        max_episode_steps=42,
    )
    assert config["task_language"] == "Place the cube"
    assert (config["seed"], config["max_episode_steps"]) == (7, 42)
    assert config["robot"] == raw["robot"]
    assert "control" not in config
    assert (
        config["max_joint_delta_per_step"] == raw["control"]["max_joint_delta_per_step"]
    )
    runtime_config.validate_site_poses(config)


@pytest.mark.parametrize(
    "update,error",
    [
        ({"task_id": 2}, "task fields must come from CLI"),
        ({"feedback_timeout_s": 99}, "duplicate top-level"),
    ],
)
def test_yaml_rejects_ambiguous_fields(tmp_path, update, error):
    path = tmp_path / "robot.yaml"
    path.write_text(yaml.safe_dump({**runtime_config.load_mapping(), **update}))
    with pytest.raises(ValueError, match=error):
        runtime_config.load_config(path, task_id=1)
    with pytest.raises(ValueError, match="must be a YAML file"):
        runtime_config.load_mapping(tmp_path / "robot.json")


@pytest.mark.parametrize("task", [0, 1, 2])
@pytest.mark.parametrize("mode", ["primitives", "vla"])
def test_env_and_agent_share_default_instruction(task, mode):
    from robots.yam.manual import _build_parser
    from robots.yam.robot_spec import _parse_config
    from robots.yam.tasks import TASK_INSTRUCTIONS

    args = _build_parser().parse_args(
        [
            "--mode",
            mode,
            "--task-id",
            str(task),
            "--env-endpoint",
            "socket://localhost:8110",
        ]
    )
    assert (
        runtime_config.load_config(task_id=task)["task_language"]
        == _parse_config(args).prompt_vars["instruction"]
        == TASK_INSTRUCTIONS[task]
    )


@pytest.mark.parametrize("task_id", [0, 1, 2])
def test_numeric_dashboard_task_reaches_env_and_agent(tmp_path, task_id):
    import argparse

    from robots.yam.robot_spec import YAM_DASHBOARD_SPEC, _add_cli_args, _parse_config
    from robots.yam.tasks import TASK_INSTRUCTIONS
    from rpent.dashboard.state import DashboardState

    state = DashboardState(output_dir=tmp_path, dashboard_spec=YAM_DASHBOARD_SPEC)
    state.shared_services_ready()
    state.submit_input(f"/rpent-task {task_id} 7")
    claimed = state.wait_for_task(timeout=0)
    assert claimed is not None and claimed.request == {"task_id": task_id, "seed": 7}
    parser = argparse.ArgumentParser()
    _add_cli_args(parser, use_dashboard=True)
    args = parser.parse_args(["--without-vla"])
    vars(args).update(claimed.request)
    config = _parse_config(args)
    env_config = runtime_config.load_config(task_id=task_id, seed=7)
    facade = YamEnvFacade(SimpleNamespace(task_id=env_config["task_id"]))
    assert facade.get_env_meta()["task_id"] == config.task_desc["task_id"] == task_id
    assert (
        config.prompt_vars["instruction"]
        == env_config["task_language"]
        == TASK_INSTRUCTIONS[task_id]
    )
    assert config.recipe_tag.startswith(f"yam_t{task_id}_s7_")
    with pytest.raises(ValueError):
        state.submit_input("/rpent-task 3 7")


def test_task_zero_does_not_connect_or_expose_vla(env, receipt, tmp_path, monkeypatch):
    import argparse

    from robots.yam import robot_spec
    from robots.yam.contracts import env_runtime_contract
    from robots.yam.env_client import YamEnvClient
    from robots.yam.tasks import TASK_INSTRUCTIONS
    from rpent.dashboard.events import NullDashboardEventSink

    parser = argparse.ArgumentParser()
    robot_spec._add_cli_args(parser, use_dashboard=False)
    args = parser.parse_args(
        [
            "--task-id",
            "0",
            "--env-endpoint",
            "socket://unused:8110",
            "--vla-endpoint",
            "socket://unused:8220",
        ]
    )
    args.memory_dir = str(tmp_path / "memory")
    args.output_dir = str(tmp_path / "output")
    monkeypatch.setattr("rpent.utils.logging._output_dir", tmp_path / "output")
    config = robot_spec._parse_config(args)
    assert config.prompt_vars["vla_enabled"] is False
    monkeypatch.setattr(
        robot_spec,
        "make_rpc_client",
        lambda _: pytest.fail("task 0 must not connect VLA"),
    )
    assert robot_spec._init_runtime(
        args, tmp_path, NullDashboardEventSink(), {"vla"}
    ) == ([], {})

    env.task_id = 0
    env.task_language = TASK_INSTRUCTIONS[0]
    metadata = env_runtime_contract(task_id=0, seed=12, max_episode_steps=100)
    facade = YamEnvFacade(env, metadata=metadata)

    class RPC:
        def call(self, method, args=(), kwargs=None, *, timeout_s=None):
            return facade._dispatch(method, args, kwargs or {})

    client = YamEnvClient(RPC(), expected_meta=metadata)
    receipt("ready")
    client.reset()
    toolkit = robot_spec.get_toolkit(
        runtime_kwargs={"env": client, "model": object()},
        dashboard_events=NullDashboardEventSink(),
        config=config,
        state_output_dir=tmp_path / "state",
    )
    try:
        names = {tool.name for tool in toolkit.list_tools()}
        assert "move_to" in names and "pi05_act" not in names
        assert not env._runtime.commands
    finally:
        toolkit.close()


@pytest.mark.parametrize("joint", [None, float("nan"), 100.0])
def test_invalid_park_pose_is_rejected_before_hardware(tmp_path, monkeypatch, joint):
    from robots.yam import env_server, rlinf_env

    raw = runtime_config.load_mapping()
    raw["park_on_close"].update(
        enabled=True,
        left_qpos=None if joint is None else [joint] + [0.0] * 5 + [1.0],
        right_qpos=[0.0] * 6 + [1.0],
    )
    path = tmp_path / "robot.yaml"
    path.write_text(yaml.safe_dump(raw))
    monkeypatch.setattr(
        rlinf_env, "YamAgentEnv", lambda _: pytest.fail("must not initialize hardware")
    )
    monkeypatch.setattr(
        sys,
        "argv",
        ["env_server", "--robot-config", str(path), "--task-id", "1"],
    )
    with pytest.raises(SystemExit) as error:
        env_server.main()
    assert error.value.code == 2


def test_env_rpc_session_stop_and_failed_shutdown():
    events = []

    def close():
        events.append("close")
        if events.count("close") == 1:
            raise RuntimeError("park failed")

    facade = YamEnvFacade(
        SimpleNamespace(
            task_id=1,
            get_task_language=lambda: "probe",
            close=close,
            request_stop=lambda: events.append("stop"),
        )
    )
    assert facade._dispatch("healthz", (), {}, session_id="probe") == {"status": "ok"}
    assert facade._dispatch("env.request_stop", (), {}, session_id="probe") == {
        "stop_requested": True,
        "hold_confirmed": False,
    }
    assert events == ["stop"]
    with pytest.raises(RuntimeError, match="park failed"):
        facade._dispatch("shutdown", (), {}, session_id="probe")
    assert not facade._shutdown_event.is_set()
    assert facade._dispatch("healthz", (), {}, session_id="probe") == {"status": "ok"}
    assert facade._dispatch("shutdown", (), {}, session_id="probe") == {"ok": True}
    assert facade._shutdown_event.is_set()
