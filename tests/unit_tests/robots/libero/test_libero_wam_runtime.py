# Copyright 2026 The RPent Authors.
# Licensed under the Apache License, Version 2.0 (the "License");

from __future__ import annotations

from types import SimpleNamespace

import pytest

from robots.libero import robot_spec
from robots.libero.prompt_bundle import system_prompt
from robots.libero.tools import LIBERO_ACTION_SCHEMA
from rpent.robots.components.action_model_protocol import (
    ActionModelCapabilities,
    ActionModelCompatibilityError,
)


def test_prompt_mentions_wam_only_when_enabled() -> None:
    assert "OPTIONAL ACTION MODEL" not in system_prompt({"wam_enabled": False})
    assert "wam_act" in system_prompt({"wam_enabled": True})["OPTIONAL ACTION MODEL"]


def test_runtime_connects_cosmos_under_distinct_wam_key(monkeypatch, tmp_path) -> None:
    rpc = SimpleNamespace(call=lambda *args, **kwargs: None, close=lambda: None)
    capabilities = ActionModelCapabilities(
        backend="cosmos_policy",
        checkpoint="predict2-2b-libero",
        supported_embodiments=("libero_7d",),
        action_dim=7,
        camera_roles=("primary", "wrist"),
        action_schema=LIBERO_ACTION_SCHEMA,
    )
    monkeypatch.setattr(robot_spec, "_connect_wam_server", lambda args: (None, rpc))
    monkeypatch.setattr(
        "rpent.robots.components.cosmos_policy_client.CosmosPolicyClient.get_capabilities",
        lambda self: capabilities,
    )
    monkeypatch.setattr(
        robot_spec,
        "try_spawn_server",
        lambda owned, events, component, starter: starter(),
    )
    monkeypatch.setattr(
        robot_spec,
        "try_wait_server",
        lambda owned, events, component, rpc, daemon, timeout, post_fn: post_fn(),
    )
    args = SimpleNamespace(
        wam_backend="cosmos",
        wam_endpoint="http://127.0.0.1:8120",
        planner="api",
        collect_flywheel_data=False,
    )
    daemons, runtime = robot_spec._init_runtime(args, tmp_path, None, {"wam"})
    assert daemons == []
    assert set(runtime) == {"wam_model"}


def test_runtime_starts_owned_cosmos_daemon_when_checkpoint_is_given(
    monkeypatch, tmp_path
) -> None:
    rpc = SimpleNamespace(call=lambda *args, **kwargs: None, close=lambda: None)
    daemon = SimpleNamespace(name="wam")
    capabilities = ActionModelCapabilities(
        backend="cosmos_policy",
        checkpoint="predict2-2b-libero",
        supported_embodiments=("libero_7d",),
        action_dim=7,
        camera_roles=("primary", "wrist"),
        action_schema=LIBERO_ACTION_SCHEMA,
    )
    monkeypatch.setattr(
        robot_spec,
        "_spawn_wam_server",
        lambda args, output_dir: (daemon, rpc),
    )
    monkeypatch.setattr(
        "rpent.robots.components.cosmos_policy_client.CosmosPolicyClient.get_capabilities",
        lambda self: capabilities,
    )
    monkeypatch.setattr(
        robot_spec,
        "try_wait_server",
        lambda owned, events, component, rpc, daemon, timeout, post_fn: post_fn(),
    )
    monkeypatch.setattr(
        robot_spec,
        "try_spawn_server",
        lambda owned, events, component, starter: (
            owned.__setitem__(component, daemon) or starter()
        ),
    )
    args = SimpleNamespace(
        wam_backend="cosmos",
        wam_endpoint=None,
        wam_checkpoint=str(tmp_path / "cosmos-policy"),
        planner="api",
        collect_flywheel_data=False,
    )

    daemons, runtime = robot_spec._init_runtime(args, tmp_path, None, {"wam"})

    assert daemons == [daemon]
    assert set(runtime) == {"wam_model"}


def test_owned_cosmos_starter_uses_isolated_environment(monkeypatch, tmp_path) -> None:
    created = {}

    class FakeDaemon:
        def __init__(self, name, cmd, **kwargs):
            created.update(name=name, cmd=cmd, kwargs=kwargs)

        def start(self):
            created["started"] = True

    cosmos_root = tmp_path / "cosmos-policy"
    python_path = cosmos_root / ".venv/bin/python"
    python_path.parent.mkdir(parents=True)
    python_target = tmp_path / "python3.10"
    python_target.touch()
    python_path.symlink_to(python_target)
    monkeypatch.setenv("COSMOS_POLICY_ROOT", str(cosmos_root))
    monkeypatch.setenv("COSMOS_POLICY_PYTHON", str(python_path))
    monkeypatch.setattr(robot_spec, "ProcessDaemon", FakeDaemon)
    monkeypatch.setattr(robot_spec, "pick_free_port", lambda host: 8123)
    monkeypatch.setattr(
        robot_spec,
        "HttpRpcClient",
        lambda endpoint: created.update(endpoint=endpoint) or endpoint,
    )
    args = SimpleNamespace(
        wam_backend="cosmos",
        wam_endpoint=None,
        wam_checkpoint=str(tmp_path / "checkpoint"),
    )

    daemon, client = robot_spec._spawn_wam_server(args, tmp_path)

    assert daemon is not None
    assert client == "http://127.0.0.1:8123"
    assert created["name"] == "wam"
    assert created["cmd"][:2] == [
        str(python_path),
        str(robot_spec.get_repo_root() / "scripts/wam/cosmos_policy_rpc_bridge.py"),
    ]
    assert created["kwargs"]["cwd"] == str(cosmos_root)
    env = created["kwargs"]["env_overrides"]
    assert env["COSMOS_INTERNAL"] == "1"
    assert env["HF_HOME"]
    assert env["HF_HUB_CACHE"]
    assert env["CUDA_HOME"].endswith("nvidia/cuda_nvrtc")
    assert created["started"] is True


def test_runtime_rejects_dreamzero_droid_for_libero(monkeypatch, tmp_path) -> None:
    rpc = SimpleNamespace(call=lambda *args, **kwargs: None, close=lambda: None)
    capabilities = ActionModelCapabilities(
        backend="dreamzero",
        checkpoint="DreamZero-DROID",
        supported_embodiments=("droid_joint_position_8d",),
        action_dim=8,
        camera_roles=("primary", "extra_0", "wrist"),
    )
    monkeypatch.setattr(robot_spec, "_connect_wam_server", lambda args: (None, rpc))
    monkeypatch.setattr(
        "rpent.robots.components.dreamzero_client.DreamZeroClient.get_capabilities",
        lambda self: capabilities,
    )
    monkeypatch.setattr(
        robot_spec,
        "try_spawn_server",
        lambda owned, events, component, starter: starter(),
    )
    monkeypatch.setattr(
        robot_spec,
        "try_wait_server",
        lambda owned, events, component, rpc, daemon, timeout, post_fn: post_fn(),
    )
    args = SimpleNamespace(
        wam_backend="dreamzero",
        wam_endpoint="http://127.0.0.1:8121",
        planner="api",
        collect_flywheel_data=False,
    )
    with pytest.raises(ActionModelCompatibilityError, match="libero_7d"):
        robot_spec._init_runtime(args, tmp_path, None, {"wam"})
