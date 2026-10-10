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

"""Offline contracts for Cosmos configuration and LIBERO tool execution."""

import argparse
from unittest.mock import Mock

import numpy as np
import pytest

from robots.libero import robot_spec, toolkit
from robots.libero.tools import LiberoPrimitives
from robots.libero.wam_client import LiberoWAMClient
from rpent.dashboard.events import NullDashboardEventSink
from rpent.robots.components import wam_runtime
from rpent.robots.components.pi05_vla_client import Pi05VLAClient
from rpent.utils import templates


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


@pytest.mark.parametrize(
    "suite,variant",
    [
        ("libero_spatial", "standard"),
        ("libero_object_task", "pro"),
        ("libero_goal_swap", "pro"),
    ],
)
@pytest.mark.parametrize("backend", ["cosmos-policy", "fast-wam"])
def test_cosmos_config_routes_suite_and_renders_without_memory(suite, variant, backend):
    args = _args("--suite", suite, "--wam-backend", backend)
    spec = robot_spec.get_robot_spec()
    config = spec.parse_config(args)
    assert args.libero_type == variant
    assert args.memory_profile == "local"
    assert config.task_desc["suite"] == suite
    assert config.task_desc["policy_backend"] == backend
    for name in ("system", "user"):
        rendered = spec.prompts.render(
            name, variables={**config.prompt_vars, "output_dir": str(config.output_dir)}
        )
        assert "wam_act" in rendered
        assert "16 actions" not in rendered
        assert all(
            token not in rendered.lower()
            for token in ("pi0_pick", "memory", "read_text_file", "{{")
        )


@pytest.mark.parametrize(
    "extra,message",
    [
        (["--explore"], "evaluation"),
        (["--planner", "flash"], "evaluation"),
        (["--libero-type", "plus"], "standard and pro"),
        (["--memory-profile", "hf"], "memory is not supported"),
        (["--memory-dir", "/some/corpus"], "does not support --memory-dir"),
        (["--wam-endpoint", ""], "requires --wam-endpoint"),
        (["--vla-endpoint", "http://localhost:8115"], "cannot be combined"),
    ],
)
def test_unsupported_cosmos_configuration_fails_early(extra, message):
    with pytest.raises(ValueError, match=message):
        robot_spec._parse_config(_args(*extra))


@pytest.mark.parametrize(
    "kind,client_type", [("vla", Pi05VLAClient), ("wam", LiberoWAMClient)]
)
def test_dashboard_category_matches_borrowed_runtime(
    tmp_path, monkeypatch, kind, client_type
):
    args = _args()
    if kind == "vla":
        args.wam_backend = args.wam_endpoint = None
        args.vla_endpoint = "http://127.0.0.1:8115"
    spec = robot_spec.get_robot_spec()
    components = {
        item["name"]: item
        for item in spec.resolve_dashboard(args)["runtime_components"]
    }
    assert components[kind]["label"] == kind.upper()
    assert ({"vla", "wam"} - {kind}).isdisjoint(components)
    rpc = Mock()
    monkeypatch.setattr(LiberoWAMClient, "validate_libero", lambda self: None)
    monkeypatch.setattr(robot_spec, "make_rpc_client", lambda endpoint: rpc)
    monkeypatch.setattr(wam_runtime, "make_rpc_client", lambda endpoint, **kwargs: rpc)
    monkeypatch.setattr("rpent.robots.runtime.wait_for_ready", lambda *a, **k: None)
    events = Mock()
    owned, runtime = spec.init_runtime(args, tmp_path, events, {kind})
    assert owned == []
    assert isinstance(runtime["model"], client_type)
    model_client = runtime["model"].wam if kind == "wam" else runtime["model"]
    assert model_client._client is rpc
    assert model_client.PREDICT_METHOD == (
        "vla.predict" if kind == "vla" else "wam.predict"
    )
    assert {call.args[0].component for call in events.emit.call_args_list} == {kind}


def test_libero_variant_preserves_environment_with_cli_override(monkeypatch):
    monkeypatch.setenv("LIBERO_TYPE", "pro")
    args = _args()
    robot_spec._parse_config(args)
    assert args.libero_type == "pro"
    args = _args("--libero-type", "standard")
    robot_spec._parse_config(args)
    assert args.libero_type == "standard"
    with pytest.raises(ValueError, match="requires --libero-type pro"):
        robot_spec._parse_config(
            _args("--suite", "libero_spatial_lan", "--libero-type", "standard")
        )


def test_owned_cosmos_uses_isolated_environment_and_component_entrypoint(
    tmp_path, monkeypatch
):
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    python = tmp_path / "python"
    interpreter = tmp_path / "base-python"
    interpreter.touch()
    python.symlink_to(interpreter)
    args = _args(
        "--wam-endpoint",
        "",
        "--wam-checkpoint",
        str(checkpoint),
        "--wam-python",
        str(python),
        "--wam-root",
        str(tmp_path),
        "--wam-predict-future",
    )
    daemon = Mock()
    factory = Mock(return_value=daemon)
    monkeypatch.setattr(wam_runtime, "ProcessDaemon", factory)
    monkeypatch.setattr(wam_runtime, "pick_free_port", lambda: 8117)
    owned, rpc = wam_runtime.select_wam(args, "libero").start_service(args, tmp_path)
    assert owned is daemon
    daemon.start.assert_called_once()
    command = factory.call_args.kwargs["cmd"]
    assert command[:3] == [
        str(python),
        "-m",
        "rpent.robots.components.cosmos_policy.server",
    ]
    assert "--parent-watch" in command and "--predict-future" in command
    assert factory.call_args.kwargs["cwd"] == str(tmp_path)
    args.wam_endpoint = "http://localhost:8116"
    with pytest.raises(ValueError, match="mutually exclusive"):
        wam_runtime.select_wam(args, "libero")


@pytest.fixture
def primitives():
    observation = {"states": np.zeros(8), "task_descriptions": "native task"}
    env = Mock(terminated=False, truncated=False, return_all_frames=False)
    env.raw_obs.return_value = {"robot0_eef_pos": np.zeros(3)}
    env.chunk_step.return_value = (observation, 0, False, False, {})
    model = Mock(predict=Mock(return_value=np.zeros((16, 7))))
    model.wam.get_capabilities.return_value = {"backend": "cosmos_policy"}
    instance = LiberoPrimitives(env, model, Mock(), lambda: None, policy_kind="wam")
    instance.set_obs(observation)
    return instance


def test_cosmos_subtask_is_request_local_and_uses_fresh_observations(primitives):
    raw = [{"robot0_eef_pos": np.full(3, step)} for step in range(3)]
    primitives.env.raw_obs.side_effect = raw
    result = primitives.wam_act("pick up the black bowl", max_chunks=2).data
    assert result["chunks"] == 2
    assert result["success"] is False
    assert primitives.wam_act().data["chunks"] == 1
    calls = primitives.model.predict.call_args_list
    assert [call.args[0]["task_descriptions"] for call in calls] == [
        "pick up the black bowl",
        "pick up the black bowl",
        "native task",
    ]
    for step, call in enumerate(calls):
        np.testing.assert_array_equal(call.args[0]["robot0_eef_pos"], np.full(3, step))
    primitives.env.reset.return_value = (primitives._last_obs, {})
    primitives.model.reset.assert_not_called()
    for episode in range(2):
        primitives.reset()
        assert primitives.model.reset.call_count == episode + 1
    assert all("task_descriptions" not in obs for obs in raw)
    assert primitives._last_obs["task_descriptions"] == "native task"


@pytest.mark.parametrize("terminated,truncated", [(True, False), (False, True)])
def test_cosmos_stops_at_episode_end_with_native_verdict(
    primitives, terminated, truncated
):
    def step(actions):
        assert actions.shape == (16, 7)
        primitives.env.terminated = terminated
        primitives.env.truncated = truncated
        return primitives._last_obs.copy(), 0, terminated, truncated, {}

    primitives.env.chunk_step.side_effect = step
    result = primitives.wam_act(max_chunks=4).data
    assert result["chunks"] == 1
    assert result["success"] == result["terminated"] == terminated
    assert result["truncated"] == truncated
    assert primitives.wam_act().data["chunks"] == 0
    primitives.model.predict.assert_called_once()
    primitives.env.chunk_step.assert_called_once()


@pytest.mark.parametrize(
    "kwargs",
    [{"max_chunks": 0}, {"max_chunks": 5}, {"max_chunks": True}, {"prompt": ""}],
)
def test_cosmos_invalid_call_fails_before_inference(primitives, kwargs):
    with pytest.raises(ValueError):
        primitives.wam_act(**kwargs)
    primitives.model.predict.assert_not_called()
    primitives.env.chunk_step.assert_not_called()


@pytest.mark.parametrize("truncated", [False, True])
def test_cosmos_toolkit_without_memory_requires_native_success(
    tmp_path, monkeypatch, primitives, truncated
):
    primitives.env.truncated = truncated
    monkeypatch.setattr(
        templates, "default_variables", lambda: {"output_dir": str(tmp_path)}
    )
    monkeypatch.setattr(
        toolkit.LiberoToolkit,
        "init_primitives",
        lambda self, **kwargs: setattr(self, "_primitives", primitives),
    )
    memory_factory = Mock(side_effect=AssertionError("unexpected memory creation"))
    monkeypatch.setattr(robot_spec, "MemoryManager", memory_factory)
    instance = robot_spec.get_toolkit(
        runtime_kwargs={},
        dashboard_events=NullDashboardEventSink(),
        config=robot_spec._parse_config(_args()),
        state_output_dir=tmp_path / "output",
    )
    memory_factory.assert_not_called()
    names = {definition.name for definition in instance.list_tools()}
    assert {"wam_act", "move_to", "view_env_state", "finish"} <= names
    assert {
        "cosmos_act",
        "pi0_pick",
        "pi0_doubled",
        "read_text_file",
        "write_text_file",
        "list_dir",
        "reset",
    }.isdisjoint(names)
    refused = instance.execute_tool("finish", {"status": "success", "summary": "done"})
    assert not refused.data.get("_finish")
    assert "native terminated=true" in refused.data["error"]
    assert refused.data["truncated"] == truncated
    assert instance.execute_tool(
        "finish", {"status": "failure", "summary": "failed"}
    ).data["_finish"]
    instance._solved = True
    primitives.env.terminated = True
    assert instance.execute_tool(
        "finish", {"status": "success", "summary": "done"}
    ).data["_finish"]
    handler = Mock()
    with pytest.raises(ValueError, match="Episode already ended"):
        instance._execute_primitive("move_to", handler, xyz=[0, 0, 1])
    handler.assert_not_called()
