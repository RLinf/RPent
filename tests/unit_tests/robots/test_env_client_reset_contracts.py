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

"""Atomic reset-result contracts for the three exploration adapters."""

from __future__ import annotations

import copy
from typing import Any

import pytest

from robots.libero.env_client import LiberoEnvClient
from robots.robocasa.env_client import (
    RoboCasaEnvClient,
    RoboCasaResetContractError,
)
from robots.robotwin.env_client import RoboTwinEnvClient


class ResetRpc:
    def __init__(self, meta: dict[str, Any], resets: list[Any]) -> None:
        self.meta = meta
        self.resets = list(resets)
        self.calls: list[str] = []

    def call(self, method: str, **kwargs: Any) -> Any:
        del kwargs
        self.calls.append(method)
        if method == "env.get_env_meta":
            return self.meta
        result = self.resets.pop(0)
        if isinstance(result, BaseException):
            raise result
        return result


def _robotwin_info(
    *,
    requested_seed: Any = 7,
    actual_seed: Any = 7,
    instruction: Any = "pick up the block",
    omit_status: tuple[str, ...] = (),
) -> dict[str, Any]:
    status = {
        "eval_success": False,
        "take_action_cnt": 0,
        "step_lim": 100,
        "actual_seed": actual_seed,
    }
    for key in omit_status:
        status.pop(key)
    return {
        "requested_seed": requested_seed,
        "instruction": instruction,
        "episode_status": status,
        "robot_state": {"joints": [0.0]},
    }


def _robocasa_result(**updates: Any) -> dict[str, Any]:
    result = {
        "observation": {"fresh": True},
        "seed": 7,
        "reset_contract": "configured_seed_reinitialization",
        "notice": ("按配置 seed 重新初始化，完整物理布局确定性仍需真实仿真验证"),
    }
    result.update(updates)
    return result


def _assert_robotwin_cache_unchanged(
    client: RoboTwinEnvClient,
    *,
    observation: dict[str, Any],
    last_info: dict[str, Any],
    last_reset_info: dict[str, Any],
) -> None:
    assert client.last_obs is observation
    assert client.last_info is last_info
    assert client.last_reset_info is last_reset_info
    assert client.terminated is True
    assert client.truncated is True


def test_libero_reset_commits_only_after_full_decode_and_preserves_shape() -> None:
    meta = {"suite": "spatial", "task": 1, "seed": 7}
    first = ({"frame": "initial"}, {"episode": 1})
    second = ({"frame": "reset"}, {"episode": 2, "nested": {"value": 1}})
    rpc = ResetRpc(meta, [first, second])
    client = LiberoEnvClient(rpc, expected_meta=meta)
    client.terminated = True
    client.truncated = True

    result = client.reset()

    assert isinstance(result, tuple) and len(result) == 2
    assert result == second
    assert client.last_obs is second[0]
    assert client.terminated is False
    assert client.truncated is False
    result[1]["nested"]["value"] = 99
    assert second[1]["nested"]["value"] == 1
    assert rpc.calls == ["env.get_env_meta", "env.reset", "env.reset"]


@pytest.mark.parametrize(
    "failure, error",
    [
        ({"not": "a pair"}, TypeError),
        (({"new": True},), TypeError),
        ((["not an observation mapping"], {}), TypeError),
        (({"new": True}, ["not info"]), TypeError),
        (RuntimeError("reset RPC failed"), RuntimeError),
    ],
)
def test_libero_reset_failure_preserves_observation_and_flags(
    failure: Any, error: type[Exception]
) -> None:
    meta = {"suite": "spatial", "task": 1, "seed": 7}
    rpc = ResetRpc(meta, [({"frame": "initial"}, {"episode": 1}), failure])
    client = LiberoEnvClient(rpc, expected_meta=meta)
    previous_observation = client.last_obs
    client.terminated = True
    client.truncated = True

    with pytest.raises(error):
        client.reset()

    assert client.last_obs is previous_observation
    assert client.terminated is True
    assert client.truncated is True


def test_robocasa_reset_preserves_notice_shape_and_raw_response() -> None:
    meta = {"seed": 7, "camera_h": 256, "camera_w": 256}
    raw = _robocasa_result(extra={"nested": True})
    original = copy.deepcopy(raw)
    rpc = ResetRpc(meta, [raw])
    client = RoboCasaEnvClient(rpc, expected_meta=meta, defer_reset=True)

    result = client.reset_exploration()

    assert isinstance(result, dict)
    assert result == {
        key: value for key, value in original.items() if key != "observation"
    }
    assert client.last_obs is raw["observation"]
    assert raw == original
    result["extra"]["nested"] = False
    assert raw["extra"]["nested"] is True
    assert "物理布局" in result["notice"]
    assert rpc.calls == ["env.get_env_meta", "env.reset_exploration"]


@pytest.mark.parametrize(
    "payload, message",
    [
        (None, "response must be a mapping"),
        ({"seed": 7}, "missing fields"),
        (_robocasa_result(observation=[]), "observation must be a mapping"),
        (_robocasa_result(seed="7"), "seed must be an integer"),
        (_robocasa_result(notice=None), "notice must be a string"),
    ],
)
def test_robocasa_malformed_reset_is_clear_and_preserves_observation(
    payload: Any, message: str
) -> None:
    meta = {"seed": 7, "camera_h": 256, "camera_w": 256}
    client = RoboCasaEnvClient(
        ResetRpc(meta, [payload]), expected_meta=meta, defer_reset=True
    )
    previous_observation = {"old": True}
    client.last_obs = previous_observation

    with pytest.raises(RoboCasaResetContractError, match=message):
        client.reset_exploration()

    assert client.last_obs is previous_observation


def test_robocasa_reset_contract_mismatch_preserves_observation() -> None:
    meta = {"seed": 7, "camera_h": 256, "camera_w": 256}
    client = RoboCasaEnvClient(
        ResetRpc(meta, [_robocasa_result(reset_contract="different")]),
        expected_meta=meta,
        defer_reset=True,
    )
    previous_observation = {"old": True}
    client.last_obs = previous_observation

    with pytest.raises(RoboCasaResetContractError, match="contract mismatch"):
        client.reset_exploration()

    assert client.last_obs is previous_observation


def test_robocasa_configured_seed_mismatch_preserves_observation() -> None:
    meta = {"seed": 7, "camera_h": 256, "camera_w": 256}
    client = RoboCasaEnvClient(
        ResetRpc(meta, [_robocasa_result(seed=8)]),
        expected_meta=meta,
        defer_reset=True,
    )
    previous_observation = {"old": True}
    client.last_obs = previous_observation

    with pytest.raises(RoboCasaResetContractError, match="configured-seed mismatch"):
        client.reset_exploration()

    assert client.last_obs is previous_observation


def test_robocasa_rpc_failure_preserves_observation() -> None:
    meta = {"seed": 7, "camera_h": 256, "camera_w": 256}
    client = RoboCasaEnvClient(
        ResetRpc(meta, [RuntimeError("reset RPC failed")]),
        expected_meta=meta,
        defer_reset=True,
    )
    previous_observation = {"old": True}
    client.last_obs = previous_observation

    with pytest.raises(RuntimeError, match="reset RPC failed"):
        client.reset_exploration()

    assert client.last_obs is previous_observation


def test_robotwin_reset_commits_all_cache_state_and_preserves_shape() -> None:
    meta = {"seed": 7}
    initial_info = _robotwin_info()
    reset_info = _robotwin_info()
    reset_info["robot_state"]["joints"] = [1.0]
    rpc = ResetRpc(
        meta,
        [({"frame": "initial"}, initial_info), ({"frame": "reset"}, reset_info)],
    )
    client = RoboTwinEnvClient(rpc, expected_meta=meta)
    client.terminated = True
    client.truncated = True

    result = client.reset()

    assert isinstance(result, tuple) and len(result) == 2
    assert result == ({"frame": "reset"}, reset_info)
    assert client.last_obs is result[0]
    assert client.last_info == reset_info
    assert client.last_reset_info == reset_info
    assert client.last_info is not client.last_reset_info
    assert client.terminated is False
    assert client.truncated is False
    assert result[1] is not client.last_info
    result[1]["robot_state"]["joints"][0] = 99.0
    assert client.last_info["robot_state"]["joints"] == [1.0]
    assert client.last_reset_info["robot_state"]["joints"] == [1.0]
    assert reset_info["robot_state"]["joints"] == [1.0]
    assert rpc.calls == ["env.get_env_meta", "env.reset", "env.reset"]


@pytest.mark.parametrize("missing", [("actual_seed",), ("eval_success",)])
def test_robotwin_missing_status_keys_preserves_all_cache_state(
    missing: tuple[str, ...],
) -> None:
    meta = {"seed": 7}
    rpc = ResetRpc(
        meta,
        [
            ({"frame": "initial"}, _robotwin_info()),
            ({"frame": "bad"}, _robotwin_info(omit_status=missing)),
        ],
    )
    client = RoboTwinEnvClient(rpc, expected_meta=meta)
    previous_observation = client.last_obs
    previous_info = client.last_info
    previous_reset_info = client.last_reset_info
    client.terminated = True
    client.truncated = True

    with pytest.raises(ValueError, match="episode_status is missing"):
        client.reset()

    _assert_robotwin_cache_unchanged(
        client,
        observation=previous_observation,
        last_info=previous_info,
        last_reset_info=previous_reset_info,
    )


@pytest.mark.parametrize(
    "info, message",
    [
        (_robotwin_info(requested_seed=8), "requested seed mismatch"),
        (_robotwin_info(actual_seed=8), "requested seed"),
        (_robotwin_info(instruction="a different task"), "instruction mismatch"),
    ],
)
def test_robotwin_reset_mismatch_preserves_all_cache_state(
    info: dict[str, Any], message: str
) -> None:
    meta = {"seed": 7}
    rpc = ResetRpc(
        meta,
        [({"frame": "initial"}, _robotwin_info()), ({"frame": "bad"}, info)],
    )
    client = RoboTwinEnvClient(rpc, expected_meta=meta)
    previous_observation = client.last_obs
    previous_info = client.last_info
    previous_reset_info = client.last_reset_info
    client.terminated = True
    client.truncated = True

    with pytest.raises(ValueError, match=message):
        client.reset()

    _assert_robotwin_cache_unchanged(
        client,
        observation=previous_observation,
        last_info=previous_info,
        last_reset_info=previous_reset_info,
    )


@pytest.mark.parametrize(
    "failure, error",
    [
        ((["not an observation mapping"], _robotwin_info()), TypeError),
        (({"new": True}, {"instruction": "task"}), TypeError),
        (({"new": True}, _robotwin_info(requested_seed=None)), TypeError),
        (({"new": True}, _robotwin_info(instruction=None)), TypeError),
        (RuntimeError("reset RPC failed"), RuntimeError),
    ],
)
def test_robotwin_malformed_or_rpc_failure_preserves_all_cache_state(
    failure: Any, error: type[Exception]
) -> None:
    meta = {"seed": 7}
    rpc = ResetRpc(
        meta,
        [({"frame": "initial"}, _robotwin_info()), failure],
    )
    client = RoboTwinEnvClient(rpc, expected_meta=meta)
    previous_observation = client.last_obs
    previous_info = client.last_info
    previous_reset_info = client.last_reset_info
    client.terminated = True
    client.truncated = True

    with pytest.raises(error):
        client.reset()

    _assert_robotwin_cache_unchanged(
        client,
        observation=previous_observation,
        last_info=previous_info,
        last_reset_info=previous_reset_info,
    )
