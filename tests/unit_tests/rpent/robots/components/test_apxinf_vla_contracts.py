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

"""Offline contracts for the experimental LIBERO APXInf service."""

import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

from rpent.robots.components.apxinf_vla_client import ApxInfVLAClient
from rpent.robots.components.apxinf_vla_server import ApxInfVLAFacade


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
    loader = ModuleType("apxinf.policies.auto")

    def load(path, **kwargs):
        calls.append((path, kwargs))
        return policy

    loader.AutoPolicy = SimpleNamespace(from_pretrained=load)
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
    client = ApxInfVLAClient(Rpc(), embodiment="libero")
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


@pytest.mark.parametrize("shape", [(7,), (10, 6)])
def test_invalid_action_shape_is_rejected(loaded_facade, shape):
    facade, policy, _ = loaded_facade
    policy.infer = lambda obs: {"actions": np.zeros(shape)}
    with pytest.raises(ValueError, match="invalid APXInf actions"):
        facade.predict({})


def test_nonfinite_actions_are_rejected(loaded_facade):
    facade, policy, _ = loaded_facade
    policy.infer = lambda obs: {"actions": np.full((10, 7), np.nan)}
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
