# Copyright 2026 The RPent Authors.
# Licensed under the Apache License, Version 2.0 (the "License");

"""WAM boundary and deployment behavior without model dependencies."""

import argparse
from unittest.mock import Mock

import numpy as np
import pytest

from rpent.robots.components import wam_runtime
from rpent.robots.components.wam_client_base import BaseWAMClient
from rpent.robots.components.wam_facade_base import BaseWAMFacade


@pytest.fixture
def worker():
    control = {
        "embodiment": "arm",
        "action_space": "joint",
        "action_schema": ["x", "y"],
        "action_type": None,
    }
    facade = BaseWAMFacade(
        {
            "backend": "test",
            "checkpoint": "test",
            "control": control,
            "camera_roles": ("primary",),
            "state_schema": {"joints": 2},
            "chunk_size": 3,
            "uses_sessions": False,
        }
    )
    facade.predict_native = Mock(
        return_value={
            "actions": np.ones((3, 2)),
            "future_observation": {"predicted": True},
            "value": 0.5,
        }
    )
    rpc = Mock(
        call=Mock(
            side_effect=lambda method, args=(), **kw: facade._dispatch(method, args, {})
        )
    )
    request = {
        "embodiment": "arm",
        "action_space": "joint",
        "instruction": "move",
        "images": {"primary": np.zeros((2, 2, 3), np.uint8)},
        "state": {"joints": np.zeros(2)},
    }
    yield facade, BaseWAMClient(rpc, expected_backend="test"), rpc, request
    facade.close()


def test_prediction_keeps_optional_outputs_and_caches_control(worker):
    facade, client, rpc, request = worker
    client.validate_contract(facade.get_capabilities()["control"])
    for _ in range(2):
        result = client.predict(request)
        assert result["actions"].shape == (3, 2)
        assert result["value"] == 0.5
        assert result["future_observation"] == {"predicted": True}
    assert [call.args[0] for call in rpc.call.call_args_list].count(
        "wam.capabilities"
    ) == 1
    client.reset()
    client.close()
    rpc.close.assert_called_once()


@pytest.mark.parametrize(
    "change",
    [
        {"instruction": " "},
        {"embodiment": "other"},
        {"images": {}},
        {"state": {"joints": np.array([0, np.nan])}},
    ],
)
def test_bad_observation_stops_before_inference(worker, change):
    facade, client, _, request = worker
    with pytest.raises(ValueError):
        client.predict({**request, **change})
    facade.predict_native.assert_not_called()


@pytest.mark.parametrize(
    "actions",
    [
        np.zeros((3, 3)),
        np.zeros((2, 2)),
        np.full((3, 2), np.nan),
    ],
)
def test_bad_actions_stop_at_client_boundary(worker, actions):
    facade, client, _, request = worker
    facade.predict_native.return_value = {"actions": actions}
    with pytest.raises(ValueError):
        client.predict(request)


@pytest.mark.parametrize("mismatch", ["backend", "control"])
def test_wrong_worker_is_rejected_before_prediction(worker, mismatch):
    facade, client, _, _ = worker
    expected = {**facade.get_capabilities()["control"]}
    if mismatch == "backend":
        facade.get_capabilities()["backend"] = "other"
    else:
        expected["action_schema"] = ["y", "x"]
    with pytest.raises(ValueError):
        client.validate_contract(expected)
    facade.predict_native.assert_not_called()


@pytest.mark.parametrize(
    "backend,platform",
    [
        ("cosmos-policy", "libero"),
        ("fast-wam", "libero"),
        ("fast-wam", "robotwin"),
    ],
)
def test_worker_launch_and_external_options(tmp_path, monkeypatch, backend, platform):
    for name in ("python", "checkpoint", "config", "stats"):
        (tmp_path / name).touch()
    parser = argparse.ArgumentParser()
    wam_runtime.add_wam_args(parser)
    extra = (
        ["--wam-predict-future"]
        if backend == "cosmos-policy"
        else [
            "--wam-config",
            str(tmp_path / "config"),
            "--wam-dataset-stats",
            str(tmp_path / "stats"),
        ]
    )
    args = parser.parse_args(
        [
            "--wam-backend",
            backend,
            "--wam-checkpoint",
            str(tmp_path / "checkpoint"),
            "--wam-python",
            str(tmp_path / "python"),
            "--wam-root",
            str(tmp_path),
            *extra,
        ]
    )
    daemon, rpc = Mock(), Mock()
    factory = Mock(return_value=daemon)
    monkeypatch.setattr(wam_runtime, "ProcessDaemon", factory)
    monkeypatch.setattr(wam_runtime, "pick_free_port", lambda: 8116)
    monkeypatch.setattr(wam_runtime, "make_rpc_client", lambda *a, **kw: rpc)
    selected = wam_runtime.select_wam(args, platform)
    assert selected.start_service(args, tmp_path) == (daemon, rpc)
    command = factory.call_args.kwargs["cmd"]
    assert command[:3] == [
        str(tmp_path / "python"),
        "-m",
        f"rpent.robots.components.{selected.rpc_backend}.server",
    ]
    assert command.count("--platform") == command.count("--checkpoint") == 1
    assert command[command.index("--platform") + 1] == platform
    assert (
        "--predict-future" if backend == "cosmos-policy" else "--dataset-stats"
    ) in command
    daemon.start.assert_called_once()
    daemon.start.side_effect = RuntimeError("launch failed")
    with pytest.raises(RuntimeError, match="launch failed"):
        selected.start_service(args, tmp_path)
    rpc.close.assert_called_once()
    daemon.stop.assert_called_once()
    args.wam_checkpoint = None
    args.wam_endpoint = "http://localhost:8116"
    with pytest.raises(ValueError, match="require --wam-checkpoint"):
        wam_runtime.select_wam(args, platform)
