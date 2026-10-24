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

"""Awomo endpoint validation and native observation/action contracts without weights."""

import argparse
from types import SimpleNamespace

import numpy as np
import pytest

from robots.robodojo import robot_spec
from robots.robodojo.awomo05 import Awomo05Client
from robots.robodojo.awomo05_contract import (
    CAMERAS,
    STATE_FIELDS,
    validate_model_meta,
)
from robots.robodojo.awomo05_policy_server import ModelWithMetadata
from robots.robodojo.vpp2_client import VPP2Client
from tests.unit_tests.robots.robodojo.fakes import args, metadata


def test_awomo_requires_exactly_one_endpoint_and_native_joint_robot():
    for flags in [
        [],
        ["--action-type", "ee", "--vla-endpoint", "http://bridge"],
        ["--env-cfg-type", "other", "--vla-endpoint", "http://bridge"],
        ["--policy-server-url", "http://wrong"],
    ]:
        with pytest.raises(ValueError):
            robot_spec._validate_args(args("--vla-backend", "awomo05", *flags))
    with pytest.raises(SystemExit):
        args("--vla-endpoint", "http://bridge", "--policy-server-url", "ws://model")
    with pytest.raises(ValueError, match="requires --vla-backend"):
        robot_spec._validate_args(args("--policy-server-url", "ws://model"))
    for flag, url in [
        ("--vla-endpoint", "http://bridge"),
        ("--policy-server-url", "ws://model"),
    ]:
        config = args("--vla-backend", "awomo05", flag, url)
        robot_spec._validate_args(config)
        assert config.vla_backend == "awomo05"
    assert args("--policy-python", "/custom/python").pi05_python == "/custom/python"


@pytest.mark.parametrize(
    "key,value",
    [
        ("policy_name", "Pi_05"),
        ("stateful", False),
        ("rpc_methods", []),
        ("methods", ["reset"]),
        ("action_dim", 12),
        ("camera_order", list(reversed(CAMERAS))),
        ("input_color_order", "rgb"),
        ("history_enabled", False),
        ("control_hz", 30),
        ("replan_steps", 0),
        ("replan_steps", 17),
        ("replan_steps", True),
        ("action_layout", "qpos14"),
        ("action_info", "true"),
    ],
)
def test_incompatible_endpoint_is_rejected_without_reset(key, value):
    meta = {**metadata(), key: value}
    calls = []

    def call(method):
        calls.append(method)
        return meta

    with pytest.raises(ValueError):
        Awomo05Client(SimpleNamespace(call=call))
    assert calls == ["vla.get_model_meta"]


def test_missing_metadata_is_actionable_and_does_not_probe_reset():
    def call(method):
        assert method == "vla.get_model_meta"
        raise ValueError("unknown method")

    with pytest.raises(RuntimeError, match="Awomo05 bridge"):
        Awomo05Client(SimpleNamespace(call=call))


def test_native_payload_keeps_camera_colors_and_named_joint_order():
    calls = []
    state = {
        key: np.full(size, index, np.float32)
        for index, (key, size) in enumerate(STATE_FIELDS.items())
    }
    image = np.array([[[255, 17, 3]]], dtype=np.uint8)
    obs = {
        "state": state,
        "vision": {
            camera: {
                "color": image.copy(),
                "depth": np.ones((1, 1)),
                "intrinsic_matrix": np.eye(3),
                "extrinsic_matrix": np.eye(4),
            }
            for camera in CAMERAS
        },
        "instruction": "stack blocks",
    }
    actions = [state]

    def call(method, **kwargs):
        calls.append((method, kwargs))
        return metadata() if method == "vla.get_model_meta" else actions

    client = Awomo05Client(SimpleNamespace(call=call))
    client.update_obs(obs, 0)
    sent = calls[-1][1]["args"][0]
    assert sent["instruction"] == obs["instruction"]
    assert all(sent["state"][key] is state[key] for key in STATE_FIELDS)
    assert all(
        sent["vision"][name]["color"] is obs["vision"][name]["color"]
        for name in CAMERAS
    )
    assert all(set(camera) == {"color"} for camera in sent["vision"].values())
    assert all("depth" in camera for camera in obs["vision"].values())
    np.testing.assert_array_equal(obs["vision"][CAMERAS[0]]["color"], image)
    assert client.get_action() is actions
    obs["vision"].pop(CAMERAS[-1])
    with pytest.raises(ValueError, match="cam_right_wrist"):
        client.update_obs(obs, 1)


@pytest.mark.parametrize(
    "actions",
    [
        [],
        [[0.0] * 14],
        [{"left_arm_joint_state": [0] * 6}],
        [{key: [float("nan")] * size for key, size in STATE_FIELDS.items()}],
    ],
)
def test_invalid_native_actions_are_rejected(actions):
    def call(method, **kwargs):
        return metadata() if method == "vla.get_model_meta" else actions

    client = Awomo05Client(SimpleNamespace(call=call))
    with pytest.raises(ValueError):
        client.get_action()


def test_standalone_wrapper_delegates_without_action_or_history_cache():
    events = []
    native = SimpleNamespace(
        env_cfg_type="arx_x5",
        action_type="joint",
        expected_action_dim=14,
        cameras=CAMERAS,
        input_color_order="bgr",
        history_grid=object(),
        action_horizon=16,
        replan_steps=8,
        reset=lambda: events.append("reset"),
        update_obs=lambda obs: events.append(obs),
        get_action=lambda: events.append("get_action"),
    )
    wrapped = ModelWithMetadata(native, history_fps=25)
    validate_model_meta(wrapped.get_model_meta())
    assert not events
    wrapped.reset()
    wrapped.update_obs({"frame": 0})
    wrapped.get_action()
    assert events == ["reset", {"frame": 0}, "get_action"]
    with pytest.raises(ValueError, match="control_hz"):
        ModelWithMetadata(native, history_fps=10)


@pytest.mark.parametrize("interrupted", [False, True])
def test_owned_bridge_uses_selected_interpreter_and_preserves_borrowed_service(
    monkeypatch, tmp_path, interrupted
):
    from robots.robodojo import awomo05_bridge

    root = tmp_path / "XPolicyLab"
    module = root / "client_server/ws/model_client.py"
    module.parent.mkdir(parents=True)
    module.touch()
    interpreter = tmp_path / "python"
    interpreter.touch()
    recorded = {}

    class Daemon:
        def __init__(self, **kwargs):
            recorded.update(kwargs)

        def start(self):
            recorded["started"] = True
            if interrupted:
                raise KeyboardInterrupt

        def stop(self):
            recorded["stopped"] = True

    monkeypatch.setattr(awomo05_bridge, "ProcessDaemon", Daemon)
    config = args(
        "--vla-backend",
        "awomo05",
        "--policy-server-url",
        "ws://model:9000",
        "--xpolicylab-root",
        str(root),
        "--policy-python",
        str(interpreter),
    )
    if interrupted:
        with pytest.raises(KeyboardInterrupt):
            robot_spec._spawn_awomo_server(config, tmp_path)
        assert recorded["stopped"]
        return
    daemon, rpc = robot_spec._spawn_awomo_server(config, tmp_path)
    try:
        assert recorded["started"]
        assert recorded["cmd"][0] == str(interpreter)
        assert "robots.robodojo.awomo05_bridge" in recorded["cmd"]
        assert "ws://model:9000" in recorded["cmd"]
        assert "CUDA_VISIBLE_DEVICES" not in recorded["env_overrides"]
        assert "--ckpt" not in recorded["cmd"]
    finally:
        rpc.close()


@pytest.mark.parametrize("compatible", [True, False])
def test_bridge_metadata_check_releases_transport_without_mutation(
    monkeypatch, compatible
):
    from robots.robodojo.awomo05_bridge import Awomo05Facade
    from rpent.robots.components.xpolicylab_vla_server import add_connection_args

    calls, closed = [], []

    def call(func_name):
        calls.append(func_name)
        return {**metadata(), "policy_name": "Awomo05" if compatible else "Pi_05"}

    model = SimpleNamespace(call=call, close=lambda: closed.append(True))
    monkeypatch.setattr(Awomo05Facade, "_connect_policy", lambda *args: model)
    parser = argparse.ArgumentParser()
    add_connection_args(parser, required=True)
    config = parser.parse_args(["--task", "stack", "--policy-server-url", "ws://model"])
    if compatible:
        facade = Awomo05Facade(config)
        try:
            validate_model_meta(facade.get_model_meta(), bridge=True)
        finally:
            facade.close()
            facade.close()
    else:
        with pytest.raises(ValueError, match="policy_name"):
            Awomo05Facade(config)
    assert calls == ["get_model_meta"]
    assert closed == [True]


@pytest.mark.parametrize(
    "failure", [None, RuntimeError("serve failed"), SystemExit(143)]
)
def test_bridge_cli_closes_connection_and_restores_signal_handler(monkeypatch, failure):
    import signal
    import sys

    from robots.robodojo import awomo05_bridge

    closed = []
    model = SimpleNamespace(
        call=lambda **kwargs: metadata(), close=lambda: closed.append(True)
    )
    monkeypatch.setattr(
        awomo05_bridge.Awomo05Facade, "_connect_policy", lambda *args: model
    )

    def serve(self, **kwargs):
        assert kwargs == {
            "transport": "http",
            "host": "127.0.0.1",
            "port": 9205,
            "parent_watch": True,
            "session_sweep_s": 60.0,
        }
        if failure is not None:
            raise failure

    monkeypatch.setattr(awomo05_bridge.Awomo05Facade, "serve", serve)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "bridge",
            "--task",
            "pick",
            "--policy-server-url",
            "ws://model",
            "--port",
            "9205",
            "--parent-watch",
        ],
    )
    previous = signal.getsignal(signal.SIGTERM)
    if failure is None:
        awomo05_bridge.main()
    else:
        with pytest.raises(type(failure)) as caught:
            awomo05_bridge.main()
        assert caught.value is failure
    assert signal.getsignal(signal.SIGTERM) is previous
    assert closed == [True]


@pytest.fixture
def runtime_services(monkeypatch):
    from robots.robodojo import env_client, vpp2_server

    events = []
    failure = {}

    def fail(phase):
        if failure.get("phase") == phase:
            raise failure["error"]

    def spawn(name, config):
        events.append(("spawn", name))
        if name == "vla":
            fail("spawn")

        def call(method, **kwargs):
            events.append((method, name))
            if method == "vla.get_model_meta":
                return metadata()
            assert method == "healthz"
            return {"status": "ok"}

        rpc = SimpleNamespace(call=call, close=lambda: events.append(("close", name)))
        # The environment is owned, while the model service is borrowed.
        daemon = (
            SimpleNamespace(
                name=name, poll=lambda: None, stop=lambda: events.append(("stop", name))
            )
            if name == "env"
            else None
        )
        return daemon, rpc

    for component, suffix in (
        ("env", "env"),
        ("sam3", "sam3"),
        ("vla", "awomo"),
        ("vla", "vpp2"),
    ):
        monkeypatch.setattr(
            robot_spec,
            f"_spawn_{suffix}_server",
            lambda config, output, name=component: spawn(name, config),
        )

    def environment(rpc, **kwargs):
        events.append(("connect", "env"))
        return rpc

    monkeypatch.setattr(env_client, "RoboDojoEnvClient", environment)
    original = robot_spec._connect_awomo

    def connect(rpc):
        fail("connect")
        return original(rpc)

    monkeypatch.setattr(robot_spec, "_connect_awomo", connect)
    from robots.robodojo import vpp2_client

    original_vpp2 = vpp2_client.VPP2Client

    def connect_vpp2(rpc):
        fail("connect")
        return original_vpp2(rpc)

    monkeypatch.setattr(vpp2_client, "VPP2Client", connect_vpp2)
    monkeypatch.setattr(
        vpp2_server,
        "model_paths",
        lambda *args: pytest.fail("unexpected VPP2 weight lookup"),
    )
    return events, failure


@pytest.mark.parametrize(
    "backend,flags,components,expected",
    [
        ("vpp2", [], None, {"env", "vla", "sam3"}),
        ("awomo05", [], None, {"env", "vla"}),
        (
            "awomo05",
            ["--vla-agent", "--agent-tools", "policy", "sam3"],
            None,
            {"env", "vla", "sam3"},
        ),
        ("awomo05", [], {"sam3"}, {"sam3"}),
        ("awomo05", [], {"vla"}, {"vla"}),
    ],
)
def test_runtime_selects_services_and_backend(
    runtime_services, tmp_path, backend, flags, components, expected
):
    from rpent.dashboard.events import NullDashboardEventSink

    events, _ = runtime_services
    config = args(
        "--vla-backend",
        backend,
        "--vla-endpoint",
        "http://bridge",
        "--vla-model-path",
        "/missing/vpp2",
        *flags,
    )
    owned, kwargs = robot_spec._init_runtime(
        config, tmp_path, NullDashboardEventSink(), components
    )
    assert {name for phase, name in events if phase == "spawn"} == expected
    assert len(owned) == int("env" in expected)
    if "vla" in expected:
        assert isinstance(
            kwargs["vla_client"], Awomo05Client if backend == "awomo05" else VPP2Client
        )
    if backend == "awomo05" and {"env", "vla"} <= expected:
        assert events.index(("vla.get_model_meta", "vla")) < events.index(
            ("connect", "env")
        )


@pytest.mark.parametrize("backend", ["vpp2", "awomo05"])
@pytest.mark.parametrize("phase", ["spawn", "connect"])
@pytest.mark.parametrize("error_type", [ValueError, KeyboardInterrupt, SystemExit])
def test_runtime_failure_cleans_owned_processes_and_pending_connections(
    runtime_services, tmp_path, backend, phase, error_type
):
    from rpent.dashboard.events import NullDashboardEventSink

    events, failure = runtime_services
    error = error_type("interrupted startup")
    failure.update(phase=phase, error=error)
    config = args("--vla-backend", backend, "--vla-endpoint", "http://bridge")
    with pytest.raises(
        RuntimeError if error_type is ValueError else error_type
    ) as caught:
        robot_spec._init_runtime(
            config, tmp_path, NullDashboardEventSink(), {"env", "vla"}
        )
    if error_type is not ValueError:
        assert caught.value is error
    assert [(op, name) for op, name in events if op == "stop"] == [("stop", "env")]
    assert {name for op, name in events if op == "close"} == (
        {"env"} if phase == "spawn" else {"env", "vla"}
    )
    if backend == "awomo05" and phase == "connect":
        assert ("connect", "env") not in events


def test_borrowed_awomo_rpc_uses_shared_session_transport(monkeypatch, tmp_path):
    called = []
    monkeypatch.setattr(
        robot_spec,
        "make_rpc_client",
        lambda endpoint, **kw: called.append((endpoint, kw)),
    )
    config = args("--vla-backend", "awomo05", "--vla-endpoint", "socket://bridge:1234")
    daemon, _ = robot_spec._spawn_awomo_server(config, tmp_path)
    assert daemon is None
    assert called == [("socket://bridge:1234", {"enable_sessions": True})]


def test_optional_action_info_preserves_public_action_list_without_second_rpc():
    actions = [
        {
            key: np.zeros(width)
            for key, width in {
                "left_arm_joint_state": 6,
                "right_arm_joint_state": 6,
                "left_ee_joint_state": 1,
                "right_ee_joint_state": 1,
            }.items()
        }
    ]
    calls = []
    info = {"upstream": {"latency_ms": 12.5}, "model": {"custom": {"x": 1}}}

    def call(method, **kwargs):
        calls.append((method, kwargs))
        if method == "vla.get_model_meta":
            return {**metadata(), "action_info": True}
        assert kwargs["kwargs"] == {"include_info": True}
        return {"actions": actions, "info": info}

    client = Awomo05Client(SimpleNamespace(call=call))
    assert client.get_action() is actions
    assert client.last_action_info is info
    assert len(calls) == 2


def test_bridge_envelope_is_optional_and_never_repeats_inference(monkeypatch):

    from robots.robodojo.awomo05_bridge import Awomo05Facade
    from rpent.robots.components import xpolicylab_vla_server as shared

    calls = []
    actions = [{"joint": [1, 2]}]
    upstream = {"ok": True, "latency_ms": 987.25, "future_metric": 17}
    result = {"actions": actions, "model_stats": {"decode_s": 0.5}}

    def connect(self, url, args):
        self._observed_client = SimpleNamespace(call_info=upstream)

        def call(func_name, obs=None):
            calls.append(func_name)
            if func_name == "get_model_meta":
                return metadata()
            if func_name == "get_action":
                return result

        return SimpleNamespace(call=call, close=lambda: None)

    monkeypatch.setattr(Awomo05Facade, "_connect_policy", connect)
    parser = argparse.ArgumentParser()
    shared.add_backend_args(parser)
    facade = Awomo05Facade(
        parser.parse_args(["--task", "stack", "--policy-server-url", "ws://test"])
    )
    try:
        facade.reset_session(session_id="owner")
        facade.update_obs({}, 0, session_id="owner")
        with pytest.raises(ValueError):
            facade.get_action(session_id="owner", include_info="yes")
        envelope = facade.get_action(session_id="owner", include_info=True)
        assert envelope["actions"] is actions
        assert envelope["info"]["upstream"] == upstream
        assert envelope["info"]["model"] == {"model_stats": {"decode_s": 0.5}}
        assert envelope["info"]["bridge_rpc_s"] >= 0
        with pytest.raises(RuntimeError, match="new observation"):
            facade.get_action(session_id="owner", include_info=True)
        assert calls.count("get_action") == 1
        facade.update_obs({}, 1, session_id="owner")
        assert facade.get_action(session_id="owner") is actions
        assert calls.count("get_action") == 2
    finally:
        facade.close()
