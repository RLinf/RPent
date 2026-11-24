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

"""Offline contracts for APXInf policy selection and RPent RPC."""

import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

from rpent.robots.components.apxinf_vla_server import ApxInfVLAFacade
from rpent.robots.components.pi05_vla_client import Pi05VLAClient
from rpent.utils.daemon import pick_free_port
from rpent.utils.rpc import RpcError, make_rpc_client, wait_for_ready


@pytest.fixture
def loaded_facade(monkeypatch, tmp_path):
    (tmp_path / "norm_stats.json").write_text("{}")
    calls = []
    policy = SimpleNamespace(
        image_keys=("base", "wrist"),
        prompt_key="prompt",
        state_key=None,
        action_dim=7,
        action_horizon=10,
        infer=lambda obs: calls.append(obs) or {"actions": np.ones((10, 7))},
        close=lambda: calls.append("closed"),
    )
    loader = ModuleType("apxinf_robo.presets")

    def load(robot, path, **kwargs):
        assert robot == "franka_libero"
        calls.append((path, kwargs))
        return policy

    loader.build_robot_policy = load
    monkeypatch.setitem(sys.modules, loader.__name__, loader)
    facade = ApxInfVLAFacade(model_path=str(tmp_path), embodiment="libero")
    yield facade, policy, calls
    facade.close()


def test_client_server_returns_five_actions_with_explicit_stats(loaded_facade):
    facade, _, calls = loaded_facade
    path, kwargs = calls[0]
    assert kwargs["norm_stats"] == path + "/norm_stats.json"
    assert kwargs["action_horizon"] == 10
    assert kwargs["num_flow_steps"] == 5
    assert kwargs["num_views"] == 2
    assert kwargs["discrete_state"] is False

    class Rpc:
        def call(self, method, args=(), timeout_s=None):
            return facade._dispatch(method, args, {})

    image = np.zeros((16, 16, 3), dtype=np.uint8)
    client = Pi05VLAClient(Rpc(), embodiment="libero")
    actions = client.predict(
        {
            "main_images": image,
            "wrist_images": image.copy(),
            "states": np.zeros(8),
            "task_descriptions": "put the soup in the basket",
        }
    )
    assert actions.shape == (5, 7)
    assert actions.dtype == np.float32
    assert calls[-1]["prompt"] == "put the soup in the basket"
    np.testing.assert_array_equal(calls[-1]["base"], image)
    assert facade._dispatch("vla.status", (), {}) == {"action_chunk": 5}


@pytest.mark.parametrize("shape", [(7,), (10, 6), (4, 7), (11, 7), (1, 10, 7)])
def test_invalid_action_shape_is_rejected(loaded_facade, shape):
    facade, policy, _ = loaded_facade
    policy.infer = lambda obs: {"actions": np.zeros(shape)}
    with pytest.raises(ValueError, match="invalid APXInf actions"):
        facade.predict({})


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_nonfinite_actions_are_rejected(loaded_facade, value):
    facade, policy, _ = loaded_facade
    policy.infer = lambda obs: {"actions": np.full((10, 7), value)}
    with pytest.raises(ValueError, match="invalid APXInf actions"):
        facade.predict({})


def test_missing_stats_fail_before_loading_engine(tmp_path):
    with pytest.raises(FileNotFoundError, match="normalization statistics"):
        ApxInfVLAFacade(model_path=str(tmp_path), embodiment="libero")


def test_training_mode_is_rejected(loaded_facade):
    facade, _, _ = loaded_facade
    with pytest.raises(ValueError, match="eval mode only"):
        facade.predict({}, {"mode": "train"})


def test_close_is_idempotent(loaded_facade):
    facade, _, calls = loaded_facade
    facade.close()
    facade.close()
    assert calls.count("closed") == 1


@pytest.mark.parametrize("transport", ["http", "socket"])
def test_rpc_predict_stays_on_loading_thread_and_closes(loaded_facade, transport):
    facade, policy, calls = loaded_facade
    owner = threading.get_ident()
    expected = np.arange(70, dtype=np.float32).reshape(10, 7)
    main = np.arange(48, dtype=np.uint8).reshape(4, 4, 3)
    wrist = main + 50

    def infer(obs):
        assert threading.get_ident() == owner
        np.testing.assert_array_equal(obs["base"], main)
        np.testing.assert_array_equal(obs["wrist"], wrist)
        assert obs["prompt"] == "pick the can"
        return {"actions": expected}

    def close():
        assert threading.get_ident() == owner
        calls.append("closed")

    policy.infer = infer
    policy.close = close
    port = pick_free_port()

    def request():
        rpc = make_rpc_client(f"{transport}://127.0.0.1:{port}")
        try:
            wait_for_ready(rpc, timeout_s=5)
            client = Pi05VLAClient(rpc, embodiment="libero")
            assert client.status() == {"action_chunk": 5}
            obs = {
                "main_images": main,
                "wrist_images": wrist,
                "states": np.zeros(8),
                "task_descriptions": "pick the can",
            }
            with pytest.raises(RpcError, match="eval mode only"):
                client.predict(obs, {"mode": "train"})
            actions = client.predict(obs)
            np.testing.assert_array_equal(actions, expected[:5])
            assert actions.dtype == np.float32
        finally:
            try:
                rpc.call("shutdown", timeout_s=2)
            except Exception:
                # Also release serve if the transport failed before readiness.
                facade._shutdown_event.set()
                raise
            finally:
                rpc.close()

    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(request)
        facade.serve(transport=transport, host="127.0.0.1", port=port)
        future.result(timeout=10)
    assert calls.count("closed") == 1


def test_help_and_discovery_do_not_import_apxinf():
    code = """
import importlib.abc
import runpy
import sys

class RejectApxInf(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in {"apxinf", "apxinf_py", "apxinf_robo"}:
            raise AssertionError("unexpected optional import: " + fullname)

sys.meta_path.insert(0, RejectApxInf())
import rpent
from rpent.cli.main import _build_argparser
_build_argparser()
sys.argv = ["apxinf_vla_server", "--help"]
runpy.run_module("rpent.robots.components.apxinf_vla_server", run_name="__main__")
"""
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
    assert "--embodiment" in result.stdout


@pytest.mark.parametrize(
    "embodiment,preset,width,horizon,view_order",
    [
        ("dual_franka", "dual_franka_tcp_rot6d", 20, 20, (20, 10, 30)),
        ("yam", "yam_joint", 14, 30, (10, 20, 30)),
    ],
)
def test_real_robot_policy_selection_and_wire_contract(
    monkeypatch, tmp_path, embodiment, preset, width, horizon, view_order
):
    (tmp_path / "norm_stats.json").write_text("{}")
    received = []
    closed = []
    policy = SimpleNamespace(
        image_keys=("base", "left", "right"),
        prompt_key="prompt",
        state_key="state",
        action_dim=width,
        action_horizon=horizon,
        infer=lambda obs: (
            received.append(obs) or {"actions": np.zeros((horizon, width))}
        ),
        close=lambda: closed.append(True),
    )
    loader = ModuleType("apxinf_robo.presets")

    def load(robot, path, **kwargs):
        assert robot == preset
        assert kwargs["num_views"] == 3
        assert kwargs["action_horizon"] == horizon
        return policy

    loader.build_robot_policy = load
    monkeypatch.setitem(sys.modules, loader.__name__, loader)
    facade = ApxInfVLAFacade(model_path=str(tmp_path), embodiment=embodiment)

    class Rpc:
        def call(self, method, args=(), timeout_s=None):
            return facade._dispatch(method, args, {})

    try:
        client = Pi05VLAClient(Rpc(), embodiment=embodiment)
        image = np.full((4, 4, 3), 10, dtype=np.uint8)
        state = np.arange(width, dtype=np.float32)
        result = client.predict(
            {
                "main_images": image,
                "extra_view_images": np.stack([image * 2, image * 3]),
                "states": state,
                "task_descriptions": "pick the cup",
            }
        )
        assert result.shape == (horizon, width)
        assert facade.get_status() == {"action_chunk": horizon}
        for key, value in zip(policy.image_keys, view_order):
            np.testing.assert_array_equal(received[-1][key], np.full_like(image, value))
        np.testing.assert_array_equal(received[-1]["state"], state)
        if embodiment == "yam":
            from robots.yam.contracts import vla_runtime_contract

            assert Rpc().call("vla.get_model_meta") == vla_runtime_contract()
    finally:
        facade.close()
    assert closed == [True]


@pytest.mark.parametrize(
    "field,value",
    [("action_dim", 7), ("action_horizon", 10), ("image_keys", ("base",))],
)
def test_wrong_yam_policy_contract_closes_model(monkeypatch, tmp_path, field, value):
    (tmp_path / "norm_stats.json").write_text("{}")
    closed = []
    policy = SimpleNamespace(
        action_dim=14,
        action_horizon=30,
        image_keys=("top", "left", "right"),
        close=lambda: closed.append(True),
    )
    setattr(policy, field, value)
    loader = ModuleType("apxinf_robo.presets")
    loader.build_robot_policy = lambda *args, **kwargs: policy
    monkeypatch.setitem(sys.modules, loader.__name__, loader)
    with pytest.raises(ValueError, match="runtime contract"):
        ApxInfVLAFacade(model_path=str(tmp_path), embodiment="yam")
    assert closed == [True]
