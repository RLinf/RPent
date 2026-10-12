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
        task_name="place_cube",
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
        ({"task_name": "stale"}, "task fields must come from CLI"),
        ({"feedback_timeout_s": 99}, "duplicate top-level"),
    ],
)
def test_yaml_rejects_ambiguous_fields(tmp_path, update, error):
    path = tmp_path / "robot.yaml"
    path.write_text(yaml.safe_dump({**runtime_config.load_mapping(), **update}))
    with pytest.raises(ValueError, match=error):
        runtime_config.load_config(path, task_name="place_cube")
    with pytest.raises(ValueError, match="must be a YAML file"):
        runtime_config.load_mapping(tmp_path / "robot.json")


def test_env_and_agent_share_default_instruction():
    from robots.yam.manual import _build_parser
    from robots.yam.robot_spec import _parse_config
    from robots.yam.tasks import TASK_INSTRUCTIONS

    task = next(iter(TASK_INSTRUCTIONS))
    args = _build_parser().parse_args(
        [
            "--task-id",
            "103",
            "--task-name",
            task,
            "--env-endpoint",
            "socket://localhost:8110",
        ]
    )
    assert (
        runtime_config.load_config(task_name=task)["task_language"]
        == _parse_config(args).prompt_vars["instruction"]
        == TASK_INSTRUCTIONS[task]
    )


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
        ["env_server", "--robot-config", str(path), "--task-name", "place_cube"],
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
