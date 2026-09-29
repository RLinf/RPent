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

from robots.libero import policy, robot_spec, toolkit
from robots.libero.suites import LIBERO_SUITE_NAMES, suite_variant
from robots.libero.tools import LiberoPrimitives
from rpent.dashboard.events import NullDashboardEventSink
from rpent.memory import MemoryManager
from rpent.robots.components.cosmos_policy_client import CosmosPolicyClient
from rpent.robots.components.cosmos_policy_server import (
    CosmosPolicyFacade,
    prepare_observation,
)
from rpent.robots.components.policy_facade_base import BasePolicyFacade
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
            "--wam-backend",
            "cosmos-policy",
            "--wam-endpoint",
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
    BasePolicyFacade.__init__(facade)
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
    assert args == ("wam.predict",)
    assert "segmentation" not in kwargs["args"][0]
    assert kwargs["timeout_s"] == 300.0


def test_cosmos_config_renders_observation_only_prompts() -> None:
    args = _args()
    spec = robot_spec.get_robot_spec()
    config = spec.parse_config(args)
    assert args.libero_type == "standard"
    assert args.memory_profile == "local"
    assert "memory_dir" not in config.prompt_vars
    assert "memory_inbox" not in config.prompt_vars
    assert config.task_desc["policy_backend"] == "cosmos-policy"
    for variant in ("system", "user"):
        rendered = spec.prompts.render(
            variant,
            variables={**config.prompt_vars, "output_dir": str(config.output_dir)},
        )
        assert "cosmos_act" in rendered
        assert "pi0_pick" not in rendered
        assert "memory" not in rendered.lower()
        assert "read_text_file" not in rendered
        assert "{{" not in rendered


@pytest.mark.parametrize("skip_hf_sync", [False, True])
def test_public_cli_cosmos_requires_skipping_hf_sync(
    monkeypatch, capsys, tmp_path, skip_hf_sync
) -> None:
    from rpent.cli import main as cli

    spec = robot_spec.get_robot_spec()

    class ConfigCaptured(Exception):
        pass

    def capture_config(args):
        config = spec.parse_config(args)
        assert args.memory_profile == "local"
        assert config.task_desc["policy_backend"] == "cosmos-policy"
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
        "--wam-backend",
        "cosmos-policy",
        "--wam-endpoint",
        "http://127.0.0.1:8116",
        "--output-dir",
        str(tmp_path),
    ]
    if skip_hf_sync:
        argv.extend(["--memory-profile", "local"])
    monkeypatch.setattr(sys, "argv", argv)
    if skip_hf_sync:
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
        (["--libero-type", "plus"], "standard and pro"),
        (
            ["--suite", "libero_object_swap", "--libero-type", "standard"],
            "requires --libero-type pro",
        ),
        (["--memory-profile", "hf"], "memory is not supported"),
        (["--memory-dir", "/some/corpus"], "does not support --memory-dir"),
    ],
)
def test_unsupported_cosmos_modes_fail_early(extra, message) -> None:
    with pytest.raises(ValueError, match=message):
        robot_spec._parse_config(_args(*extra))


def test_cosmos_requires_external_endpoint() -> None:
    args = _args()
    args.wam_endpoint = None
    with pytest.raises(ValueError, match="--wam-endpoint"):
        robot_spec._parse_config(args)


def test_cosmos_toolkit_factory_does_not_construct_memory(
    tmp_path, monkeypatch
) -> None:
    factory = Mock()
    monkeypatch.setattr(toolkit, "LiberoToolkit", factory)
    memory_factory = Mock(side_effect=AssertionError("unexpected memory creation"))
    monkeypatch.setattr(robot_spec, "MemoryManager", memory_factory)
    config = robot_spec._parse_config(_args())

    robot_spec.get_toolkit(
        runtime_kwargs={},
        dashboard_events=NullDashboardEventSink(),
        config=config,
        state_output_dir=tmp_path,
    )

    assert factory.call_args.kwargs["memory"] is None
    memory_factory.assert_not_called()


@pytest.mark.parametrize("suite", LIBERO_SUITE_NAMES)
def test_cosmos_uses_shared_suite_routing(suite) -> None:
    args = _args("--suite", suite)
    config = robot_spec._parse_config(args)
    assert args.libero_type == suite_variant(suite)
    assert config.task_desc["suite"] == suite


def test_runtime_borrows_cosmos_service_without_spawning_pi05(
    tmp_path, monkeypatch
) -> None:
    rpc = Mock()
    monkeypatch.setattr(policy, "make_rpc_client", lambda endpoint: rpc)
    monkeypatch.setattr("rpent.robots.runtime.wait_for_ready", lambda *a, **k: None)
    daemons, runtime = robot_spec._init_runtime(
        _args(), tmp_path, NullDashboardEventSink(), {"wam"}
    )
    assert daemons == []
    assert isinstance(runtime["model"], CosmosPolicyClient)
    assert runtime["model"]._client is rpc


@pytest.mark.parametrize("backend,kind", [("pi05", "vla"), ("cosmos-policy", "wam")])
def test_dashboard_category_matches_borrowed_runtime(
    tmp_path, monkeypatch, backend, kind
):
    args = _args()
    if kind == "vla":
        args.wam_backend = args.wam_endpoint = None
        args.vla_backend = backend
        args.vla_endpoint = "http://127.0.0.1:8115"
    spec = robot_spec.get_robot_spec()
    dashboard = spec.resolve_dashboard(args)
    components = {item["name"]: item for item in dashboard["runtime_components"]}
    assert components[kind]["label"] == kind.upper()
    assert ({"vla", "wam"} - {kind}).isdisjoint(components)
    rpc = Mock()
    monkeypatch.setattr(policy, "make_rpc_client", lambda endpoint: rpc)
    monkeypatch.setattr("rpent.robots.runtime.wait_for_ready", lambda *a, **k: None)
    events = Mock()
    owned, runtime = spec.init_runtime(args, tmp_path, events, {kind})
    assert owned == []
    assert runtime["model"].PREDICT_METHOD == f"{kind}.predict"
    assert {call.args[0].component for call in events.emit.call_args_list} == {kind}


@pytest.mark.parametrize(
    "extra,message",
    [
        (["--vla-endpoint", "http://localhost:8115"], "cannot be combined"),
        (["--suite", "unknown_suite"], "Unknown LIBERO suite"),
    ],
)
def test_invalid_policy_or_suite_configuration(extra, message):
    with pytest.raises(ValueError, match=message):
        robot_spec._parse_config(_args(*extra))


def test_backends_are_mutually_exclusive():
    with pytest.raises(SystemExit):
        _args("--vla-backend", "pi05")


def test_wam_endpoint_needs_explicit_backend():
    args = _args()
    args.wam_backend = None
    with pytest.raises(ValueError, match="--wam-endpoint requires --wam-backend"):
        policy.select_policy(args)


@pytest.mark.parametrize("variant", ["pro", "plus"])
def test_pi05_can_explicitly_select_variant_for_base_suite(variant):
    args = _args("--libero-type", variant)
    args.wam_backend = args.wam_endpoint = None
    config = robot_spec._parse_config(args)
    assert args.libero_type == variant
    assert config.prompt_vars["policy_backend"] == "pi05"
    assert config.prompt_vars["memory_enabled"] is True


def test_libero_variant_preserves_environment_with_cli_override(monkeypatch):
    monkeypatch.setenv("LIBERO_TYPE", "pro")
    args = _args()
    robot_spec._parse_config(args)
    assert args.libero_type == "pro"
    args = _args("--libero-type", "standard")
    robot_spec._parse_config(args)
    assert args.libero_type == "standard"
    monkeypatch.setenv("LIBERO_TYPE", "standard")
    with pytest.raises(ValueError, match="requires --libero-type pro"):
        robot_spec._parse_config(_args("--suite", "libero_spatial_lan"))


@pytest.mark.parametrize("stop_after", [1, 2])
@pytest.mark.parametrize("prompt", [None, "pick up the black bowl"])
def test_cosmos_tool_uses_fresh_raw_obs_and_stops_after_termination(
    stop_after, prompt
) -> None:
    observation = {"states": np.zeros(8), "task_descriptions": "native task"}
    env = Mock(terminated=False, truncated=False, return_all_frames=False)
    env.raw_obs.side_effect = [
        {**_raw_obs(), "robot0_eef_pos": np.full(3, step)} for step in range(stop_after)
    ]
    model = Mock(predict=Mock(return_value=np.zeros((16, 7))))
    primitive = LiberoPrimitives(env, model, Mock(), lambda: None)
    primitive.set_obs(observation)

    def step(actions):
        env.terminated = env.chunk_step.call_count == stop_after
        return observation.copy(), 1, env.terminated, False, {}

    env.chunk_step.side_effect = step
    result = primitive.cosmos_act(prompt=prompt, max_chunks=3)
    assert result["chunks"] == stop_after
    assert result["success"] is True
    assert result["terminated"] is True
    assert result["truncated"] is False
    assert observation["task_descriptions"] == "native task"
    assert "robot0_eef_quat" in model.predict.call_args.args[0]
    assert primitive._last_obs["task_descriptions"] == "native task"
    assert env.raw_obs.call_count == stop_after
    for step, call in enumerate(model.predict.call_args_list):
        assert call.args[0]["task_descriptions"] == (prompt or "native task")
        np.testing.assert_array_equal(call.args[0]["robot0_eef_pos"], np.full(3, step))
    env.get_task_language.assert_not_called()
    assert primitive.cosmos_act()["chunks"] == 0
    with pytest.raises(ValueError, match="max_chunks"):
        primitive.cosmos_act(max_chunks=0)


def test_cosmos_subtask_is_request_local_and_preserves_native_verdict() -> None:
    observation = {"states": np.zeros(8), "task_descriptions": "native task"}
    raw = _raw_obs()
    env = Mock(terminated=False, truncated=False, return_all_frames=False)
    env.raw_obs.return_value = raw
    env.chunk_step.return_value = (observation.copy(), 0, False, False, {})
    model = Mock(predict=Mock(return_value=np.zeros((16, 7))))
    primitive = LiberoPrimitives(env, model, Mock(), lambda: None)
    primitive.set_obs(observation)

    assert primitive.cosmos_act("pick up the black bowl", max_chunks=2) == {
        "model": "cosmos-policy",
        "chunks": 2,
        "success": False,
        "terminated": False,
        "truncated": False,
    }
    primitive.cosmos_act()
    assert [
        call.args[0]["task_descriptions"] for call in model.predict.call_args_list
    ] == [
        "pick up the black bowl",
        "pick up the black bowl",
        "native task",
    ]
    assert raw["task_descriptions"] == "put the bowl on the plate"
    assert primitive._last_obs["task_descriptions"] == "native task"


def test_cosmos_truncation_is_not_task_success() -> None:
    observation = {"states": np.zeros(8), "task_descriptions": "native task"}
    env = Mock(terminated=False, truncated=False, return_all_frames=False)
    env.raw_obs.return_value = _raw_obs()
    model = Mock(predict=Mock(return_value=np.zeros((16, 7))))
    primitive = LiberoPrimitives(env, model, Mock(), lambda: None)
    primitive.set_obs(observation)

    def step(actions):
        env.truncated = True
        return observation.copy(), 0, False, True, {}

    env.chunk_step.side_effect = step
    assert primitive.cosmos_act(max_chunks=4) == {
        "model": "cosmos-policy",
        "chunks": 1,
        "success": False,
        "terminated": False,
        "truncated": True,
    }
    model.predict.assert_called_once()
    assert primitive.cosmos_act()["chunks"] == 0
    model.predict.assert_called_once()


@pytest.mark.parametrize("max_chunks", [0, 5, 20, 1.5, True])
def test_cosmos_rejects_long_or_invalid_calls_before_inference(max_chunks) -> None:
    env, model = Mock(), Mock()
    primitive = LiberoPrimitives(env, model, Mock(), lambda: None)
    with pytest.raises(ValueError, match="max_chunks"):
        primitive.cosmos_act(max_chunks=max_chunks)
    model.predict.assert_not_called()
    env.chunk_step.assert_not_called()


@pytest.mark.parametrize("terminated,truncated", [(True, False), (False, True)])
def test_cosmos_toolkit_rejects_motion_after_episode_end(terminated, truncated) -> None:
    instance = toolkit.LiberoToolkit.__new__(toolkit.LiberoToolkit)
    instance._policy_backend = "cosmos-policy"
    instance._primitives = Mock(
        env=SimpleNamespace(terminated=terminated, truncated=truncated)
    )
    handler = Mock()
    with pytest.raises(ValueError, match="Episode already ended"):
        instance._execute_primitive("move_to", handler, xyz=[0, 0, 1])
    handler.assert_not_called()
    instance._primitives.begin_primitive.assert_not_called()


@pytest.mark.parametrize("prompt", ["", "  ", 12, False, [], {}])
def test_cosmos_rejects_invalid_prompt_before_any_action(prompt) -> None:
    env, model = Mock(), Mock()
    primitive = LiberoPrimitives(env, model, Mock(), lambda: None)
    with pytest.raises(ValueError, match="prompt"):
        primitive.cosmos_act(prompt)
    env.raw_obs.assert_not_called()
    model.predict.assert_not_called()
    env.chunk_step.assert_not_called()


@pytest.mark.parametrize(
    "backend,mode",
    [("pi05", "evaluation"), ("pi05", "exploration"), ("cosmos-policy", "evaluation")],
)
def test_toolkit_exposes_selected_backend_tools(
    tmp_path, monkeypatch, backend, mode
) -> None:
    monkeypatch.setattr(
        templates, "default_variables", lambda: {"output_dir": str(tmp_path)}
    )

    def init_primitives(self, *, runtime_kwargs):
        self._primitives = LiberoPrimitives(Mock(), Mock(), Mock(), lambda: None)

    monkeypatch.setattr(toolkit.LiberoToolkit, "init_primitives", init_primitives)
    instance = toolkit.LiberoToolkit(
        runtime_kwargs={},
        dashboard_events=NullDashboardEventSink(),
        memory=None
        if backend == "cosmos-policy"
        else MemoryManager(tmp_path / "memory"),
        state_output_dir=tmp_path / "output",
        policy_backend=backend,
        mode=mode,
    )
    names = {spec["name"] for spec in instance.get_tools_spec()}
    if backend == "cosmos-policy":
        assert "cosmos_act" in names
        assert {"pi0_pick", "pi0_doubled"}.isdisjoint(names)
        assert {"read_text_file", "write_text_file", "list_dir"}.isdisjoint(names)
        result = instance.execute_tool("read_text_file", {"path": "MEMORY.md"})
        assert "error" in result.result
    else:
        assert "cosmos_act" not in names
        assert {"pi0_pick", "pi0_doubled"} <= names
        assert {"read_text_file", "write_text_file", "list_dir"} <= names
    assert ("reset" in names) == (mode == "exploration")
    assert {"move_to", "view_env_state", "finish"} <= names


@pytest.mark.parametrize("truncated", [False, True])
def test_cosmos_finish_rejects_false_success_without_ending_loop(
    tmp_path, monkeypatch, truncated
) -> None:
    monkeypatch.setattr(
        templates, "default_variables", lambda: {"output_dir": str(tmp_path)}
    )

    def init_primitives(self, *, runtime_kwargs):
        self._primitives = LiberoPrimitives(
            SimpleNamespace(terminated=False, truncated=truncated),
            Mock(),
            Mock(),
            lambda: None,
        )

    monkeypatch.setattr(toolkit.LiberoToolkit, "init_primitives", init_primitives)
    instance = toolkit.LiberoToolkit(
        runtime_kwargs={},
        dashboard_events=NullDashboardEventSink(),
        memory=None,
        state_output_dir=tmp_path / "output",
        policy_backend="cosmos-policy",
    )
    refused = instance.execute_tool("finish", {"status": "success", "summary": "done"})
    assert not refused.is_finish
    assert "native terminated=true" in refused.result["error"]
    assert refused.result["truncated"] == truncated
    failure = instance.execute_tool(
        "finish", {"status": "failure", "summary": "failed"}
    )
    assert failure.is_finish
    instance._solved = True
    instance.primitives.env.terminated = True
    success = instance.execute_tool("finish", {"status": "success", "summary": "done"})
    assert success.is_finish
