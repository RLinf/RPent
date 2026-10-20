# Copyright 2026 The RPent Authors.
# Licensed under the Apache License, Version 2.0 (the "License");

"""Shared action-model startup and VLA/WAM boundaries without model dependencies."""

import argparse
from unittest.mock import Mock

import numpy as np
import pytest

from rpent.robots.components import am_runtime
from rpent.robots.components.am_client_base import BaseVLAClient, BaseWAMClient
from rpent.robots.components.am_facade_base import BaseVLAFacade, BaseWAMFacade


def test_vla_dispatch_preserves_options_and_model_extensions():
    class VLA(BaseVLAFacade):
        def _register_rpc(self):
            super()._register_rpc()
            self._rpc["vla.metadata"] = lambda: {"model": "test"}

        def predict(self, obs, options):
            return {"actions": obs["state"], "options": options}

    facade = VLA()
    rpc = Mock(call=lambda method, args, **kw: facade._dispatch(method, args, {}))
    options = {"mode": "eval"}
    try:
        result = BaseVLAClient(rpc).predict({"state": [1, 2]}, options)
        assert result == {"actions": [1, 2], "options": options}
        assert facade._dispatch("vla.metadata", (), {}) == {"model": "test"}
    finally:
        facade.close()


@pytest.mark.parametrize("owned", [False, True])
@pytest.mark.parametrize("sessions", [False, True])
def test_shared_service_preserves_ownership_and_sessions(monkeypatch, owned, sessions):
    rpc, daemon = Mock(), Mock() if owned else None
    factory = Mock(return_value=rpc)
    monkeypatch.setattr(am_runtime, "make_rpc_client", factory)
    result = am_runtime.start_am_service(
        daemon, "http://localhost:8116", enable_sessions=sessions
    )
    assert result == (daemon, rpc)
    factory.assert_called_once_with("http://localhost:8116", enable_sessions=sessions)
    if owned:
        daemon.start.assert_called_once()
        daemon.stop.assert_not_called()
    rpc.close.assert_not_called()


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


@pytest.mark.parametrize("timeout_s", [300.0, 12.0])
def test_prediction_and_reset_preserve_rpc_arguments(worker, timeout_s):
    _, _, rpc, request = worker
    client = BaseWAMClient(rpc, timeout_s=timeout_s)
    client.predict(request)
    rpc.call.assert_called_with("wam.predict", args=(request,), timeout_s=timeout_s)
    client.reset()
    rpc.call.assert_called_with("wam.reset", timeout_s=timeout_s)


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
@pytest.mark.parametrize("legacy_options", [False, True])
def test_worker_launch_and_external_options(
    tmp_path, monkeypatch, backend, platform, legacy_options
):
    for name in ("python", "checkpoint", "config", "stats"):
        (tmp_path / name).touch()
    parser = argparse.ArgumentParser()
    am_runtime.add_wam_args(parser)
    extra = (
        [
            "--cosmos-predict-future",
            "--cosmos-cached-instructions-only",
            "--cosmos-text-embeddings",
            str(tmp_path / "embeddings.pkl"),
        ]
        if backend == "cosmos-policy"
        else [
            "--fast-wam-config",
            str(tmp_path / "config"),
            "--fast-wam-dataset-stats",
            str(tmp_path / "stats"),
        ]
    )
    if legacy_options:
        extra = [
            arg.replace("--cosmos-", "--wam-").replace("--fast-wam-", "--wam-")
            for arg in extra
        ]
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
    monkeypatch.setattr(am_runtime, "ProcessDaemon", factory)
    monkeypatch.setattr(am_runtime, "pick_free_port", lambda: 8116)
    monkeypatch.setattr(am_runtime, "make_rpc_client", lambda *a, **kw: rpc)
    selected = am_runtime.select_wam(args, platform)
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
    if backend == "cosmos-policy":
        assert "--cached-instructions-only" in command
        assert command[command.index("--text-embeddings") + 1] == str(
            tmp_path / "embeddings.pkl"
        )
    else:
        assert command[command.index("--config") + 1] == str(tmp_path / "config")
        assert command[command.index("--dataset-stats") + 1] == str(tmp_path / "stats")
    daemon.start.assert_called_once()
    daemon.start.side_effect = RuntimeError("launch failed")
    with pytest.raises(RuntimeError, match="launch failed"):
        selected.start_service(args, tmp_path)
    rpc.close.assert_called_once()
    daemon.stop.assert_called_once()
    args.wam_checkpoint = None
    args.wam_endpoint = "http://localhost:8116"
    with pytest.raises(ValueError, match="require --wam-checkpoint"):
        am_runtime.select_wam(args, platform)
    external_args = parser.parse_args(
        ["--wam-backend", backend, "--wam-endpoint", "http://localhost:8116"]
    )
    factory.reset_mock()
    assert am_runtime.select_wam(external_args, platform).start_service(
        external_args, tmp_path
    ) == (None, rpc)
    factory.assert_not_called()
    other_backend = "fast-wam" if backend == "cosmos-policy" else "cosmos-policy"
    external_args = parser.parse_args(
        [
            "--wam-backend",
            other_backend,
            "--wam-endpoint",
            "http://localhost:8116",
            *extra,
        ]
    )
    with pytest.raises(ValueError, match=f"require --wam-backend {backend}"):
        am_runtime.select_wam(external_args, platform)
