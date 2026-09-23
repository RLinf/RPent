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

"""Offline Cosmos Policy input, execution, and runtime contracts."""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from robots.libero import robot_spec, toolkit
from robots.libero.cosmos_policy_client import CosmosPolicyClient
from robots.libero.cosmos_policy_server import CosmosPolicyFacade, prepare_observation
from robots.libero.tools import LiberoPrimitives
from rpent.dashboard.events import NullDashboardEventSink
from rpent.memory import MemoryManager
from rpent.robots.components.vla_facade_base import BaseVLAFacade
from rpent.utils import templates


def _raw_obs() -> dict:
    image = np.zeros((256, 256, 3), dtype=np.uint8)
    image[0] = [1, 2, 3]
    image[-1] = [4, 5, 6]
    return {
        "agentview_image": image,
        "robot0_eye_in_hand_image": image + 10,
        "robot0_gripper_qpos": np.array([0.04, -0.04]),
        "robot0_eef_pos": np.array([0.1, 0.2, 0.3]),
        "robot0_eef_quat": np.array([0, 0, 0, -1.0]),
        "task_descriptions": "put the bowl on the plate",
    }


def _args(*extra: str) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--memory-profile", default=None)
    parser.add_argument("--memory-dir", default=None)
    parser.add_argument("--planner", default="api")
    parser.add_argument("--explore", action="store_true")
    robot_spec._add_cli_args(parser, False)
    return parser.parse_args(
        [
            "--suite",
            "libero_spatial",
            "--task",
            "0",
            "--vla-backend",
            "cosmos-policy",
            "--vla-endpoint",
            "http://127.0.0.1:8116",
            *extra,
        ]
    )


def test_raw_observation_matches_cosmos_training_convention() -> None:
    raw = _raw_obs()
    prepared = prepare_observation(raw)
    np.testing.assert_array_equal(prepared["primary_image"][0, 0], [4, 5, 6])
    np.testing.assert_array_equal(prepared["wrist_image"][0, 0], [14, 15, 16])
    np.testing.assert_allclose(
        prepared["proprio"], [0.04, -0.04, 0.1, 0.2, 0.3, 0, 0, 0, -1]
    )
    np.testing.assert_array_equal(raw["agentview_image"][0, 0], [1, 2, 3])


@pytest.mark.parametrize(
    "key,value",
    [
        ("agentview_image", np.zeros((128, 128, 3), np.uint8)),
        ("robot0_eye_in_hand_image", np.zeros((256, 256, 3), np.float32)),
        ("robot0_eef_quat", np.zeros(3)),
        ("robot0_gripper_qpos", np.array([np.nan, 0])),
    ],
)
def test_invalid_policy_inputs_fail_before_inference(key, value) -> None:
    raw = {**_raw_obs(), key: value}
    with pytest.raises(ValueError, match=key):
        prepare_observation(raw)


def _facade() -> CosmosPolicyFacade:
    facade = CosmosPolicyFacade.__new__(CosmosPolicyFacade)
    BaseVLAFacade.__init__(facade)
    facade._cfg = SimpleNamespace(seed=7, num_denoising_steps_action=5)
    facade._model = object()
    facade._dataset_stats = object()
    facade._get_action = Mock(return_value={"actions": np.full((16, 7), 0.25)})
    return facade


def test_facade_uses_official_action_api_without_predicting_future_video() -> None:
    facade = _facade()
    actions = facade.predict(_raw_obs(), {"mode": "eval"})
    np.testing.assert_array_equal(actions, np.full((16, 7), 0.25, np.float32))
    args, kwargs = facade._get_action.call_args
    assert args[4] == "put the bowl on the plate"
    assert args[3]["proprio"].shape == (9,)
    assert kwargs == {
        "seed": 7,
        "num_denoising_steps_action": 5,
        "generate_future_state_and_value_in_parallel": False,
    }
    with pytest.raises(ValueError, match="instruction"):
        facade.predict({**_raw_obs(), "task_descriptions": ""})
    with pytest.raises(ValueError, match="evaluation"):
        facade.predict(_raw_obs(), {"mode": "train"})


@pytest.mark.parametrize(
    "actions", [np.zeros((1, 16, 7)), np.zeros((0, 7)), np.full((16, 7), np.inf)]
)
def test_client_rejects_invalid_chunks(actions) -> None:
    client = CosmosPolicyClient(Mock(call=Mock(return_value=actions)))
    with pytest.raises(ValueError, match="finite actions"):
        client.predict(_raw_obs())


def test_client_sends_only_required_raw_fields() -> None:
    rpc = Mock(call=Mock(return_value=np.zeros((16, 7))))
    client = CosmosPolicyClient(rpc)
    client.predict({**_raw_obs(), "segmentation": object()}, {"mode": "eval"})
    args, kwargs = rpc.call.call_args
    assert args == ("vla.predict",)
    assert "segmentation" not in kwargs["args"][0]
    assert kwargs["timeout_s"] == 300.0


def test_cosmos_config_isolates_memory_and_renders_available_tools() -> None:
    args = _args()
    spec = robot_spec.get_robot_spec()
    config = spec.parse_config(args)
    assert args.libero_type == "standard"
    assert args.memory_profile == "local"
    assert config.prompt_vars["memory_dir"].endswith("memory/libero_cosmos")
    assert config.task_desc["vla_backend"] == "cosmos-policy"
    for variant in ("system", "user"):
        rendered = spec.prompts.render(
            variant,
            variables={**config.prompt_vars, "output_dir": str(config.output_dir)},
        )
        assert "cosmos_act" in rendered
        assert "pi0_pick" not in rendered


@pytest.mark.parametrize("local_memory", [False, True])
def test_public_cli_cosmos_requires_explicit_local_memory(
    monkeypatch, capsys, tmp_path, local_memory
) -> None:
    from rpent.cli import main as cli

    spec = robot_spec.get_robot_spec()

    class ConfigCaptured(Exception):
        pass

    def capture_config(args):
        config = spec.parse_config(args)
        assert args.memory_profile == "local"
        assert config.task_desc["vla_backend"] == "cosmos-policy"
        raise ConfigCaptured

    monkeypatch.setattr(
        cli, "get_robot_spec", lambda name: replace(spec, parse_config=capture_config)
    )
    argv = [
        "rpent",
        "--robot",
        "libero",
        "--suite",
        "libero_spatial",
        "--task",
        "0",
        "--vla-backend",
        "cosmos-policy",
        "--vla-endpoint",
        "http://127.0.0.1:8116",
        "--output-dir",
        str(tmp_path),
    ]
    if local_memory:
        argv.extend(["--memory-profile", "local"])
    monkeypatch.setattr(sys, "argv", argv)
    if local_memory:
        with pytest.raises(ConfigCaptured):
            cli.main()
    else:
        with pytest.raises(SystemExit) as exc:
            cli.main()
        assert exc.value.code == 2
        assert "requires --memory-profile local" in capsys.readouterr().err


@pytest.mark.parametrize(
    "extra,message",
    [
        (["--explore"], "evaluation"),
        (["--planner", "flash"], "evaluation"),
        (["--libero-type", "pro"], "standard LIBERO"),
        (["--suite", "libero_object_swap"], "standard LIBERO"),
        (["--memory-profile", "hf"], "local memory"),
    ],
)
def test_unsupported_cosmos_modes_fail_early(extra, message) -> None:
    with pytest.raises(ValueError, match=message):
        robot_spec._parse_config(_args(*extra))


def test_cosmos_requires_external_endpoint() -> None:
    args = _args()
    args.vla_endpoint = None
    with pytest.raises(ValueError, match="--vla-endpoint"):
        robot_spec._parse_config(args)


def test_runtime_borrows_cosmos_service_without_spawning_pi05(
    tmp_path, monkeypatch
) -> None:
    rpc = Mock()
    monkeypatch.setattr(robot_spec, "make_rpc_client", lambda endpoint: rpc)
    monkeypatch.setattr("rpent.robots.runtime.wait_for_ready", lambda *a, **k: None)
    daemons, runtime = robot_spec._init_runtime(
        _args(), tmp_path, NullDashboardEventSink(), {"vla"}
    )
    assert daemons == []
    assert isinstance(runtime["model"], CosmosPolicyClient)
    assert runtime["model"]._client is rpc


def test_cosmos_tool_uses_fresh_raw_obs_and_stops_after_termination() -> None:
    observation = {"states": np.zeros(8), "task_descriptions": "native task"}
    env = Mock(terminated=False, truncated=False, return_all_frames=False)
    env.get_task_language.return_value = "native task"
    env.raw_obs.return_value = _raw_obs()
    model = Mock(predict=Mock(return_value=np.zeros((16, 7))))
    primitive = LiberoPrimitives(env, model, Mock(), lambda: None)
    primitive.set_obs(observation)

    def step(actions):
        env.terminated = True
        return observation.copy(), 1, True, False, {}

    env.chunk_step.side_effect = step
    result = primitive.cosmos_act(max_chunks=3)
    assert result["chunks"] == 1
    assert result["success"] is True
    assert model.predict.call_args.args[0]["task_descriptions"] == "native task"
    assert "robot0_eef_quat" in model.predict.call_args.args[0]
    assert primitive._last_obs["task_descriptions"] == "native task"
    env.raw_obs.assert_called_once()
    assert primitive.cosmos_act()["chunks"] == 0
    with pytest.raises(ValueError, match="max_chunks"):
        primitive.cosmos_act(max_chunks=0)


def test_cosmos_toolkit_exposes_cosmos_instead_of_pi05(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(
        templates, "default_variables", lambda: {"output_dir": str(tmp_path)}
    )

    def init_primitives(self, *, runtime_kwargs):
        self._primitives = LiberoPrimitives(Mock(), Mock(), Mock(), lambda: None)

    monkeypatch.setattr(toolkit.LiberoToolkit, "init_primitives", init_primitives)
    instance = toolkit.LiberoToolkit(
        runtime_kwargs={},
        dashboard_events=NullDashboardEventSink(),
        memory=MemoryManager(tmp_path / "memory"),
        state_output_dir=tmp_path / "output",
        vla_backend="cosmos-policy",
    )
    names = {spec["name"] for spec in instance.get_tools_spec()}
    assert "cosmos_act" in names
    assert {"pi0_pick", "pi0_doubled", "reset"}.isdisjoint(names)
    assert {"move_to", "view_env_state", "finish"} <= names
