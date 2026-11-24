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

"""Shared action-model startup and VLA/WAM boundaries without model dependencies."""

import argparse
import json
from unittest.mock import Mock

import numpy as np
import pytest

from rpent.robots.components import am_runtime
from rpent.robots.components.am_client_base import BaseVLAClient, BaseWAMClient
from rpent.robots.components.am_facade_base import BaseVLAFacade, BaseWAMFacade


@pytest.mark.parametrize("option", ["--am-backend", "--vla-backend", "--wam-backend"])
def test_backend_aliases_preserve_supported_models(option):
    for platform, value, expected in (
        ("robodojo", "vpp2", "vpp2"),
        ("robodojo", "awomo05", "awomo05"),
        ("libero", "rlinf", "pi05"),
    ):
        parser = argparse.ArgumentParser()
        am_runtime.add_am_args(parser, platform=platform)
        assert parser.parse_args([option, value]).am_backend == expected


def test_robodojo_rejects_removed_policy_option(capsys):
    parser = argparse.ArgumentParser()
    am_runtime.add_am_args(parser, platform="robodojo")
    with pytest.raises(SystemExit):
        parser.parse_args(["--policy-backend", "rlinf"])
    assert "unrecognized arguments: --policy-backend rlinf" in capsys.readouterr().err


@pytest.mark.parametrize("model_meta", [None, {"backend": "test", "action_dim": 2}])
def test_vla_dispatch_preserves_options_and_model_extensions(model_meta):
    class VLA(BaseVLAFacade):
        def _register_rpc(self):
            super()._register_rpc()
            self._rpc["vla.metadata"] = lambda: {"model": "test"}

        def predict(self, obs, options):
            return {"actions": obs["state"], "options": options}

    facade = VLA(model_meta=model_meta)
    rpc = Mock(call=lambda method, args, **kw: facade._dispatch(method, args, {}))
    options = {"mode": "eval"}
    try:
        result = BaseVLAClient(rpc).predict({"state": [1, 2]}, options)
        assert result == {"actions": [1, 2], "options": options}
        assert facade._dispatch("vla.metadata", (), {}) == {"model": "test"}
        if model_meta is not None:
            assert facade._dispatch("vla.get_model_meta", (), {}) == model_meta
            assert "vla.get_model_meta" in facade._readonly_methods
        else:
            assert "vla.get_model_meta" not in facade._rpc
    finally:
        facade.close()


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
        },
        adapter=(lambda request: request, lambda result: result),
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
    yield facade, BaseWAMClient(rpc), rpc, request
    facade.close()


def test_prediction_keeps_optional_outputs_and_caches_control(worker):
    facade, client, rpc, request = worker
    rpc.call.assert_called_once_with("wam.get_model_meta", timeout_s=30.0)
    client.validate_contract(facade.get_model_meta()["control"])
    for _ in range(2):
        result = client.predict(request)
        assert result["actions"].shape == (3, 2)
        assert result["value"] == 0.5
        assert result["future_observation"] == {"predicted": True}
        assert rpc.call.call_args.kwargs["timeout_s"] == 300.0
    assert [call.args[0] for call in rpc.call.call_args_list].count(
        "wam.get_model_meta"
    ) == 1
    client.reset()
    client.close()
    rpc.close.assert_called_once()


@pytest.mark.parametrize("timeout_s", [300.0, 12.0])
def test_prediction_and_reset_preserve_rpc_arguments(worker, timeout_s):
    facade, _, rpc, request = worker
    facade.get_model_meta()["prediction_timeout_s"] = 630.0
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
        {"images": {"primary": np.zeros((2, 2, 3), np.float32)}},
        {"state": {"joints": np.zeros(3)}},
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
        np.zeros((1, 3, 2)),
        np.zeros((0, 2)),
        np.zeros((3, 3)),
        np.zeros((2, 2)),
        np.full((3, 2), np.nan),
        np.full((3, 2), np.inf),
    ],
)
def test_bad_actions_stop_at_client_boundary(worker, actions):
    facade, client, _, request = worker
    facade.predict_native.return_value = {"actions": actions}
    with pytest.raises(ValueError):
        client.predict(request)


def test_wrong_control_is_rejected_before_prediction(worker):
    facade, client, _, _ = worker
    expected = {**facade.get_model_meta()["control"]}
    expected["action_schema"] = ["y", "x"]
    with pytest.raises(ValueError):
        client.validate_contract(expected)
    facade.predict_native.assert_not_called()


@pytest.mark.parametrize(
    "history",
    [
        None,
        [],
        [1, 2],
        [[1]],
        [["bad", "data"]],
        np.zeros((5, 2)),
        [[np.nan, 0]],
        [[0, np.inf]],
    ],
)
def test_required_history_is_checked_before_encoding(worker, history):
    facade, client, _, request = worker
    facade.get_model_meta()["joint_target_history_shape"] = (4, 2)
    encode = Mock()
    facade._adapter = (encode, lambda result: result)
    if history is not None:
        request["joint_target_history"] = history
    with pytest.raises(ValueError, match="joint_target_history"):
        client.predict(request)
    encode.assert_not_called()
    facade.predict_native.assert_not_called()


@pytest.mark.parametrize("length", [1, 4])
def test_history_length_is_independent_of_action_chunk(worker, length):
    facade, client, _, request = worker
    facade.get_model_meta()["joint_target_history_shape"] = (4, 2)
    request["joint_target_history"] = np.zeros((length, 2))
    assert client.predict(request)["actions"].shape == (3, 2)


def uniwam_bundle(tmp_path):
    root = tmp_path / "bundle"
    root.mkdir()
    for name in ("policy", "checkpoint", "wan", "vlm"):
        resource = tmp_path / "resources" / name
        resource.mkdir(parents=True)
        (root / name).symlink_to(resource, target_is_directory=True)
    (root / "checkpoint/pytorch_model").mkdir()
    (root / "checkpoint/config.json").write_text(
        json.dumps(
            {
                "common": {"action_chunk_size": 16},
                "flow_source": {"mode": "history", "history_length": 16},
            }
        )
    )
    (root / "policy/deploy_policy.py").touch()
    return root


@pytest.mark.parametrize(
    "section,key,value",
    [
        ("common", "action_chunk_size", 8),
        ("flow_source", "mode", "gaussian"),
        ("flow_source", "history_length", 8),
        ("flow_source", "history_length", None),
    ],
)
def test_incompatible_uniwam_checkpoint_is_rejected_before_launch(
    tmp_path, section, key, value
):
    root = uniwam_bundle(tmp_path)
    path = root / "checkpoint/config.json"
    config = json.loads(path.read_text())
    config[section][key] = value
    path.write_text(json.dumps(config))
    parser = argparse.ArgumentParser()
    am_runtime.add_am_args(parser, platform="robotwin")
    with pytest.raises(ValueError, match="UniWAM checkpoint requires"):
        am_runtime.select_am(
            parser.parse_args(["--am-backend", "uniwam", "--am-checkpoint", str(root)]),
            "robotwin",
        )


@pytest.mark.parametrize(
    "backend,platform",
    [
        ("cosmos-policy", "libero"),
        ("cosmos3", "libero"),
        ("fast-wam", "libero"),
        ("fast-wam", "robotwin"),
        ("uniwam", "robotwin"),
    ],
)
@pytest.mark.parametrize("legacy_options,model_device", [(False, None), (True, "0")])
def test_worker_launch_and_external_options(
    tmp_path, monkeypatch, backend, platform, legacy_options, model_device
):
    for name in ("python", "checkpoint", "config", "stats"):
        (tmp_path / name).touch()
    (tmp_path / "python").chmod(0o755)
    parser = argparse.ArgumentParser()
    am_runtime.add_am_args(parser, platform=platform)
    extra = (
        [
            "--predict-future",
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
    if backend == "cosmos3":
        extra = [
            "--cosmos3-config",
            str(tmp_path / "config"),
            "--cosmos3-sglang-endpoint",
            "http://localhost:30000",
        ]
    if backend == "uniwam":
        extra = []
    checkpoint = (
        uniwam_bundle(tmp_path) if backend == "uniwam" else tmp_path / "checkpoint"
    )
    if legacy_options:
        extra = [
            arg.replace("--cosmos-", "--wam-").replace("--fast-wam-", "--wam-")
            for arg in extra
        ]
    args = parser.parse_args(
        [
            "--vla-backend" if legacy_options else "--am-backend",
            backend,
            "--vla-model-path" if legacy_options else "--am-checkpoint",
            str(checkpoint),
            "--am-python",
            str(tmp_path / "python"),
            "--am-root",
            str(tmp_path),
            *extra,
        ]
    )
    args.am_cuda_device = model_device
    if backend == "cosmos3":
        args.am_checkpoint = "nvidia/cosmos3-nano-libero10-sdg-export-S2_s43"
    args.cuda_device = 1
    args.env_cuda_device = args.sam3_cuda_device = 0
    daemon, rpc = Mock(), Mock()
    factory = Mock(return_value=daemon)
    monkeypatch.setattr(am_runtime, "ProcessDaemon", factory)
    monkeypatch.setattr(am_runtime, "pick_free_port", lambda: 8116)
    rpc_factory = Mock(return_value=rpc)
    monkeypatch.setattr(am_runtime, "make_rpc_client", rpc_factory)
    selected = am_runtime.select_am(args, platform)
    assert am_runtime.start_wam_service(selected, tmp_path) == (daemon, rpc)
    rpc_factory.assert_called_once_with("http://127.0.0.1:8116", enable_sessions=False)
    command = factory.call_args.kwargs["cmd"]
    assert factory.call_args.kwargs["env_overrides"]["CUDA_VISIBLE_DEVICES"] == (
        model_device or "1"
    )
    assert command[:3] == [
        str(tmp_path / "python"),
        "-m",
        f"rpent.robots.components.{am_runtime.WAM_BACKENDS[backend]}.server",
    ]
    assert command.count("--platform") == command.count("--checkpoint") == 1
    assert command[command.index("--checkpoint") + 1] == args.am_checkpoint
    assert command[command.index("--platform") + 1] == platform
    if backend == "cosmos-policy":
        assert "--predict-future" in command
        assert "--cached-instructions-only" in command
        assert command[command.index("--text-embeddings") + 1] == str(
            tmp_path / "embeddings.pkl"
        )
    elif backend == "cosmos3":
        assert command[command.index("--config") + 1] == str(tmp_path / "config")
        assert (
            command[command.index("--sglang-endpoint") + 1] == "http://localhost:30000"
        )
    elif backend == "fast-wam":
        assert command[command.index("--config") + 1] == str(tmp_path / "config")
        assert command[command.index("--dataset-stats") + 1] == str(tmp_path / "stats")
    daemon.start.assert_called_once()
    rpc_factory.reset_mock()
    daemon.start.side_effect = RuntimeError("launch failed")
    with pytest.raises(RuntimeError, match="launch failed"):
        am_runtime.start_wam_service(selected, tmp_path)
    rpc_factory.assert_not_called()
    if backend != "cosmos3":
        args.am_checkpoint = str(tmp_path / "missing-checkpoint")
        message = (
            "UniWAM resource directory" if backend == "uniwam" else "--am-checkpoint"
        )
        with pytest.raises(ValueError, match=message):
            am_runtime.select_am(args, platform)
    args.am_endpoint = "http://localhost:8116"
    with pytest.raises(ValueError, match="configures local deployment"):
        am_runtime.select_am(args, platform)
    args.am_checkpoint = args.am_python = args.am_root = args.am_cuda_device = None
    if backend != "uniwam":
        with pytest.raises(ValueError, match="configures local deployment"):
            am_runtime.select_am(args, platform)
    external_args = parser.parse_args(
        ["--am-backend", backend, "--am-endpoint", "http://localhost:8116"]
    )
    factory.reset_mock()
    assert am_runtime.start_wam_service(
        am_runtime.select_am(external_args, platform), tmp_path
    ) == (None, rpc)
    factory.assert_not_called()
    rpc_factory.assert_called_once_with("http://localhost:8116", enable_sessions=False)
