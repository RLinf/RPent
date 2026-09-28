# Copyright 2026 The RPent Authors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy at https://www.apache.org/licenses/LICENSE-2.0

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import yaml

from robots.yam import runtime_config


def _site_yaml(tmp_path: Path) -> Path:
    raw = runtime_config.load_mapping()
    raw["park_on_close"].update(
        enabled=True,
        left_qpos=[0.0] * 6 + [1.0],
        right_qpos=[0.0] * 6 + [1.0],
    )
    path = tmp_path / "site.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    return path


def test_packaged_yaml_has_machine_identity_controls_and_safe_pose_defaults() -> None:
    raw = runtime_config.load_mapping()
    assert set(raw["robot"]) == {"left_follower", "right_follower"}
    assert [camera["name"] for camera in raw["cameras"]] == [
        "top",
        "left",
        "right",
    ]
    assert raw["operator_receipt_path"]
    assert raw["park_on_close"]["enabled"] is False
    assert raw["reset"]["enabled"] is False
    assert raw["joint_servo"]["enabled"] is False
    assert "max_joint_delta_per_step" in raw["control"]
    assert "task_name" not in raw
    assert "seed" not in raw


def test_loader_flattens_control_and_preserves_site_safety_fields(
    tmp_path: Path,
) -> None:
    path = _site_yaml(tmp_path)
    config = runtime_config.load_config(
        path,
        task_name="tabletop_cleanup_a",
        task_language="Clean the table",
        seed=7,
        max_episode_steps=42,
    )
    assert config["task_name"] == "tabletop_cleanup_a"
    assert config["task_language"] == "Clean the table"
    assert config["seed"] == 7
    assert config["max_episode_steps"] == 42
    assert config["max_joint_delta_per_step"] == 0.05
    assert "control" not in config
    assert config["robot"]["left_follower"]["channel"] == "can0"
    assert config["park_on_close"]["enabled"] is True
    assert config["collision_guard"]["enabled"] is False
    runtime_config.validate_site_poses(config)


def test_loader_rejects_task_fields_or_json_robot_config(tmp_path: Path) -> None:
    path = _site_yaml(tmp_path)
    raw = yaml.safe_load(path.read_text())
    raw["task_name"] = "stale"
    path.write_text(yaml.safe_dump(raw))
    with pytest.raises(ValueError, match="task fields must come from CLI"):
        runtime_config.load_config(path, task_name="fresh")
    with pytest.raises(ValueError, match="must be a YAML file"):
        runtime_config.load_mapping(tmp_path / "site.json")


def test_loader_rejects_conflicting_control_fields(tmp_path: Path) -> None:
    path = _site_yaml(tmp_path)
    raw = yaml.safe_load(path.read_text())
    raw["feedback_timeout_s"] = 99
    path.write_text(yaml.safe_dump(raw))
    with pytest.raises(ValueError, match="duplicate top-level"):
        runtime_config.load_config(path, task_name="fresh")


def test_site_pose_must_be_enabled_finite_and_within_joint_limits(
    tmp_path: Path,
) -> None:
    path = _site_yaml(tmp_path)
    config = runtime_config.load_config(path, task_name="test")
    runtime_config.validate_site_poses(config)
    config["park_on_close"]["enabled"] = False
    with pytest.raises(ValueError, match="must enable park_on_close"):
        runtime_config.validate_site_poses(config)
    config["park_on_close"]["enabled"] = True
    config["park_on_close"]["left_qpos"][0] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        runtime_config.validate_site_poses(config)
    config["park_on_close"]["left_qpos"][0] = 100.0
    with pytest.raises(ValueError, match="exceeds joint limits"):
        runtime_config.validate_site_poses(config)


def test_env_server_uses_yaml_and_cli_task_before_hardware(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import robots.yam.env_server as env_server
    import robots.yam.rlinf_env as rlinf_env

    captured: dict[str, Any] = {}
    path = _site_yaml(tmp_path)

    class FakeEnv:
        def __init__(self, config: dict[str, Any]) -> None:
            captured["config"] = config
            self.lower = np.zeros((2, 6))
            self.upper = np.ones((2, 6))

    class FakeFacade:
        def __init__(self, env: Any, *, metadata: dict[str, Any]) -> None:
            captured["metadata"] = metadata

        def serve(self, **kwargs: Any) -> None:
            captured["serve"] = kwargs

        def close(self) -> None:
            captured["closed"] = True

    monkeypatch.setattr(rlinf_env, "YamAgentEnv", FakeEnv)
    monkeypatch.setattr(env_server, "YamEnvFacade", FakeFacade)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "env_server",
            "--robot-config",
            str(path),
            "--task-name",
            "tabletop_cleanup_a",
            "--task-language",
            "Clean the table",
            "--seed",
            "4",
            "--max-episode-steps",
            "77",
        ],
    )

    env_server.main()

    assert captured["config"]["task_language"] == "Clean the table"
    assert captured["config"]["seed"] == 4
    assert captured["metadata"]["task_name"] == "tabletop_cleanup_a"
    assert captured["metadata"]["execution"]["step_limit"] == 77
    assert captured["closed"] is True


def test_env_server_rejects_unparkable_site_before_hardware(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import robots.yam.env_server as env_server
    import robots.yam.rlinf_env as rlinf_env

    path = _site_yaml(tmp_path)
    raw = yaml.safe_load(path.read_text())
    raw["park_on_close"]["left_qpos"] = None
    path.write_text(yaml.safe_dump(raw))
    monkeypatch.setattr(
        rlinf_env,
        "YamAgentEnv",
        lambda config: pytest.fail("hardware constructor must not run"),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        ["env_server", "--robot-config", str(path), "--task-name", "test"],
    )
    with pytest.raises(SystemExit) as error:
        env_server.main()
    assert error.value.code == 2


def test_operator_control_reads_receipt_path_from_yaml(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import robots.yam.operator_control as operator_control
    import rpent.utils.rpc as rpc

    path = _site_yaml(tmp_path)
    raw = yaml.safe_load(path.read_text())
    receipt_path = tmp_path / "receipt.json"
    raw["operator_receipt_path"] = str(receipt_path)
    path.write_text(yaml.safe_dump(raw))
    monkeypatch.setattr(rpc, "make_rpc_client", lambda endpoint: object())
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "operator_control",
            "--robot-config",
            str(path),
            "--event",
            "ready",
            "--episode-id",
            "episode-1",
            "--note",
            "Operator confirmed scene",
        ],
    )

    operator_control.main()

    receipt = operator_control.read_receipt(receipt_path)
    assert receipt["episode_id"] == "episode-1"
    assert receipt["event"] == "ready"
