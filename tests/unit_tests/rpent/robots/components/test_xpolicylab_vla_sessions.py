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

"""Session ownership and ordered observations across both shared RPC transports."""

import argparse

import numpy as np
import pytest

from rpent.robots.components import xpolicylab_vla_server as bridge
from rpent.utils.rpc import RpcError, make_rpc_client
from tests.unit_tests.rpent.robots.components.xpolicylab_helpers import serve_policy


class FakeModel:
    def __init__(self):
        self.history = []
        self.inferences = []
        self.resets = 0
        self.failure = None
        self.closed = False

    def call(self, func_name, obs=None):
        if func_name == "reset":
            self.resets += 1
            self.history.clear()
        elif func_name == "update_obs":
            # Check the actual image payload survived the RPC boundary.
            frame = obs["state"]["frame"]
            image = obs["vision"]["cam_head"]["color"]
            np.testing.assert_array_equal(image, np.full((2, 2, 3), frame, np.uint8))
            assert obs["instruction"] == "stack the blocks"
            self.history.append(frame)
        elif func_name == "get_action":
            self.inferences.append(list(self.history))
        else:
            raise AssertionError(func_name)
        if self.failure == func_name:
            raise RuntimeError("lost response after model mutation")
        if func_name == "get_action":
            return {
                "actions": [
                    {"expected_step": self.history[-1] + i} for i in range(1, 4)
                ]
            }

    def close(self):
        self.closed = True


def observation(frame=0):
    return {
        "state": {"frame": frame},
        "vision": {"cam_head": {"color": np.full((2, 2, 3), frame, np.uint8)}},
        "instruction": "stack the blocks",
    }


@pytest.fixture(params=["http", "socket"])
def live_bridge(request, monkeypatch):
    model = FakeModel()
    monkeypatch.setattr(bridge, "_connect_policy", lambda *args: model)
    parser = argparse.ArgumentParser()
    bridge.add_backend_args(parser)
    facade = bridge.XPolicyLabVLAFacade(
        parser.parse_args(
            ["--task", "stack", "--policy-server-url", "ws://fake", "--stateful"]
        )
    )
    with serve_policy(facade, request.param) as service:
        service.model = model
        yield service
    assert model.closed


def test_missing_duplicate_and_out_of_order_observations(live_bridge):
    policy = live_bridge.connect()
    policy.reset_session()
    with pytest.raises(RpcError, match="new observation"):
        policy.get_action()
    obs = observation()
    for step in [1, -1, True, 0.0]:
        with pytest.raises(RpcError, match="Expected observation step 0"):
            policy.update_obs(obs, step)
    policy.update_obs(obs, 0)
    with pytest.raises(RpcError, match="Expected observation step 1"):
        policy.update_obs(obs, 0)
    policy.get_action()
    with pytest.raises(RpcError, match="new observation"):
        policy.get_action()
    assert live_bridge.model.history == [0]
    assert live_bridge.model.inferences == [[0]]


@pytest.mark.parametrize("method", ["reset", "vla.predict"])
def test_legacy_mutations_are_rejected_in_stateful_mode(live_bridge, method):
    policy = live_bridge.connect()
    policy.reset_session()
    args = ({},) if method == "vla.predict" else ()
    with pytest.raises(RpcError, match="stateful mode"):
        policy._client.call(method, args=args)
    assert live_bridge.model.resets == 1


def test_unbound_clients_cannot_mutate_policy(live_bridge):
    rpc = make_rpc_client(live_bridge.endpoint)
    with pytest.raises(RpcError, match="bound session"):
        rpc.call("reset")
    assert live_bridge.model.resets == 0


def test_session_expiry_requires_reset_by_a_new_owner(live_bridge):
    policy = live_bridge.connect()
    policy.reset_session()
    policy.update_obs(observation(), 0)
    policy.get_action()
    facade = live_bridge.facade
    facade._session_timeout_s = -1
    for sid in list(facade._sessions):
        facade._expire_session(sid)
    with pytest.raises(RpcError, match="session not found"):
        policy.get_action()
    other = live_bridge.connect()
    other.reset_session()
    assert live_bridge.model.history == []


def test_session_ownership_release_and_connection_reuse(live_bridge):
    owner, other = live_bridge.connect(), live_bridge.connect()
    owner.reset_session()
    owner.update_obs(observation(), 0)
    with pytest.raises(RpcError, match="another session"):
        other.reset_session()
    for method, args in [(other.update_obs, ({}, 1)), (other.get_action, ())]:
        with pytest.raises(RpcError, match="acquire"):
            method(*args)
    other.end_episode()
    owner.get_action()
    assert live_bridge.model.history == [0]
    owner.end_episode()
    owner.end_episode()
    with pytest.raises(RpcError, match="acquire"):
        owner.get_action()
    other.reset_session()
    other.end_episode()
    owner.reset_session()
    assert live_bridge.model.resets == 3
    assert live_bridge.model.history == []


@pytest.mark.parametrize("failure", ["reset", "update_obs", "get_action"])
def test_ambiguous_mutation_latches_session_until_explicit_reset(live_bridge, failure):
    policy = live_bridge.connect()
    if failure != "reset":
        policy.reset_session()
        policy.update_obs(observation(), 0)
    live_bridge.model.failure = failure
    with pytest.raises(RpcError, match="lost response"):
        {
            "reset": policy.reset_session,
            "update_obs": lambda: policy.update_obs(observation(1), 1),
            "get_action": policy.get_action,
        }[failure]()
    with pytest.raises(RpcError, match="uncertain"):
        policy.get_action()
    live_bridge.model.failure = None
    policy.reset_session()
    policy.update_obs(observation(), 0)
    assert policy.get_action()
    assert live_bridge.model.history == [0]
