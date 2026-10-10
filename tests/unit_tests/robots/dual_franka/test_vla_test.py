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

import json
from types import SimpleNamespace

import numpy as np
import pytest

from tests.e2e_tests.dual_franka.dual_franka_vla import DeploymentTest, run_console


def make_session(tmp_path, *, fail=False, terminate=False):
    calls = []

    def observation():
        calls.append("observe")
        return {
            "states": np.zeros(20, np.float32),
            "main_images": np.zeros((4, 4, 3), np.uint8),
            "extra_view_images": np.zeros((2, 4, 4, 3), np.uint8),
        }

    actions = np.tile([0.5, 0, 0.5, 1, 0, 0, 0, 1, 0, 0] * 2, (20, 1)).astype(
        np.float32
    )

    def predict(obs, options):
        calls.append("predict")
        assert obs["task_descriptions"] == "pick"
        return actions.copy()

    def execute(a):
        calls.append("execute")
        if fail:
            raise RuntimeError("RPC timeout")
        return {"terminated": terminate, "observation": observation()}

    env = SimpleNamespace(
        get_observation=observation,
        chunk_step=execute,
        get_robot_state=lambda: {"state": "after"},
    )
    workspace = {
        "ee_pose_limit_min": [[0, -1, 0]] * 2,
        "ee_pose_limit_max": [[1, 1, 1]] * 2,
    }
    s = DeploymentTest(
        env,
        SimpleNamespace(predict=predict),
        tmp_path,
        workspace,
        "pick",
        20,
    )
    return s, calls, actions


def test_infer_records_without_execution(tmp_path):
    s, calls, _ = make_session(tmp_path)
    result = s.chunk()
    assert calls == ["observe", "predict"]
    assert not result["executed"]
    record = json.loads((tmp_path / "vla_000001.json").read_text())
    assert record["prompt"] == "pick"
    with np.load(tmp_path / "vla_000001.npz", allow_pickle=False) as a:
        assert a["root/actions"].shape == (20, 20)
        assert a["root/input/main_images"].dtype == np.uint8


def test_step_reinfers_instead_of_reusing_preview(tmp_path):
    s, calls, _ = make_session(tmp_path)
    s.chunk()
    s.chunk(True)
    assert calls.count("predict") == 2 and calls.count("execute") == 1
    assert "robot_state_after" in json.loads((tmp_path / "vla_000002.json").read_text())


@pytest.mark.parametrize("kind", ["nan", "shape", "workspace", "rotation"])
def test_bad_actions_never_execute(tmp_path, kind):
    s, calls, a = make_session(tmp_path)
    if kind == "nan":
        a[0, 0] = np.nan
    if kind == "shape":
        a = a[:1]
    if kind == "workspace":
        a[0, 10] = 2
    if kind == "rotation":
        a[0, 3:9] = 0
    s.model.predict = lambda *args, **kw: a
    with pytest.raises(ValueError):
        s.chunk(True)
    assert "execute" not in calls
    assert not json.loads((tmp_path / "vla_000001.json").read_text())["ok"]


def test_failure_aborts_run_and_blocks_further_motion(tmp_path):
    s, calls, _ = make_session(tmp_path, fail=True)
    lines = iter(["run 3", "step", "quit"])
    run_console(s, read=lambda _: next(lines), emit=lambda _: None)
    assert calls.count("execute") == 1
    assert s.execution_uncertain


def test_termination_stops_run(tmp_path):
    s, calls, _ = make_session(tmp_path, terminate=True)
    lines = iter(["run 3", "quit"])
    run_console(s, read=lambda _: next(lines), emit=lambda _: None)
    assert calls.count("execute") == 1


@pytest.mark.parametrize(
    "line", ["run 0", "run 21", "run abc", "step 3", "infer 2", "prompt "]
)
def test_invalid_commands_do_nothing(tmp_path, line):
    s, calls, _ = make_session(tmp_path)
    lines = iter([line, "quit"])
    run_console(s, read=lambda _: next(lines), emit=lambda _: None)
    assert calls == []


def test_diagnostics_are_not_registered_as_planner_tasks():
    from robots.dual_franka.tasks import get_dual_franka_task

    for task_id in (103, 104):
        with pytest.raises(ValueError):
            get_dual_franka_task(task_id)


def test_terminated_episode_requires_reset(tmp_path):
    s, calls, _ = make_session(tmp_path, terminate=True)
    s.chunk(True)
    with pytest.raises(RuntimeError, match="Episode ended"):
        s.chunk(True)
    assert calls.count("execute") == 1


def test_dual_client_uses_live_state_instead_of_cached_reset():
    from robots.dual_franka.env_client import DualFrankaEnvClient

    client = object.__new__(DualFrankaEnvClient)
    client._last_states = np.zeros(20)
    client._client = SimpleNamespace(call=lambda *a, **k: {"states": np.ones(20)})
    np.testing.assert_array_equal(client.get_observation()["states"], np.ones(20))


@pytest.mark.parametrize("instruction", [None, "diagnostic prompt"])
@pytest.mark.parametrize("expected_steps", [None, 20])
def test_session_accepts_standard_config_without_local_deployment(
    tmp_path, monkeypatch, instruction, expected_steps
):
    import argparse

    import yaml

    from robots.dual_franka.tasks import get_dual_franka_task
    from tests.e2e_tests.dual_franka import dual_franka_vla as vla_test

    config_path = tmp_path / "robot.yaml"
    config_path.write_text(yaml.safe_dump({"workspace": {}}))
    closed = []
    observed = []
    args = argparse.Namespace(
        expected_action_steps=expected_steps,
        robot_config=str(config_path),
        task_id=1,
        instruction=instruction,
        vla_model_path="checkpoint",
        vla_repo_id="dataset",
    )
    metadata = {"model_path": "checkpoint", "config": {"openpi": {"action_chunk": 7}}}
    status_calls = []

    def status(**kwargs):
        assert expected_steps is None, "explicit chunk length must support old servers"
        status_calls.append(kwargs)
        return metadata

    model = SimpleNamespace(status=status)

    def init_runtime(args, output, events, components):
        assert components == {"env", "vla"}
        return (
            [SimpleNamespace(stop=lambda: closed.append(True))],
            {"model": model, "env": SimpleNamespace(get_camera_meta=lambda: {})},
        )

    spec = SimpleNamespace(
        parse_config=lambda args: SimpleNamespace(output_dir=tmp_path / "run"),
        init_runtime=init_runtime,
    )
    monkeypatch.setattr(vla_test, "get_robot_spec", lambda: spec)
    monkeypatch.setattr(
        vla_test, "run_console", lambda session: observed.append(session)
    )
    assert vla_test.run_session(args) == 0
    assert observed[0].prompt == (
        instruction or get_dual_franka_task(1).vla_instruction
    )
    assert observed[0].expected_steps == (
        7 if expected_steps is None else expected_steps
    )
    deployment = json.loads((tmp_path / "run" / "deployment.json").read_text())
    assert deployment["expected_action_steps"] == observed[0].expected_steps
    assert deployment["server"] == (metadata if expected_steps is None else None)
    assert status_calls == ([{"timeout_s": 10}] if expected_steps is None else [])
    assert closed == [True]


def test_primitive_profile_is_rejected_before_runtime(tmp_path, monkeypatch):
    import argparse

    from tests.e2e_tests.dual_franka import dual_franka_vla as vla_test

    config = tmp_path / "config.yaml"
    config.write_text("workspace: {}\n")
    args = argparse.Namespace(
        expected_action_steps=20,
        task_id=0,
        instruction="explicit prompt",
        robot_config=str(config),
        vla_endpoint=None,
    )
    monkeypatch.setattr(
        vla_test,
        "get_robot_spec",
        lambda: SimpleNamespace(
            parse_config=lambda args: pytest.fail("must reject before runtime setup")
        ),
    )
    with pytest.raises(ValueError, match="VLA task profile"):
        vla_test.run_session(args)


def test_diagnostic_parser_provides_shared_config_defaults(
    tmp_path, dual_franka_robot_config
):
    from robots.dual_franka.robot_spec import get_robot_spec
    from tests.e2e_tests.dual_franka.dual_franka_vla import build_parser

    args = build_parser().parse_args(
        [
            "--output-dir",
            str(tmp_path),
            "--robot-config",
            str(dual_franka_robot_config),
        ]
    )
    assert args.explore is False
    assert args.memory_dir is None and args.memory_profile is None
    config = get_robot_spec().parse_config(args)
    assert config.prompt_vars["mode"] == "eval"
    assert config.prompt_vars["session_max"] == 1


def test_diagnostic_runtime_ignores_configured_sam3(monkeypatch, tmp_path):
    from robots.dual_franka import robot_spec
    from tests.e2e_tests.dual_franka.dual_franka_vla import build_parser

    monkeypatch.setenv("SAM3_CHECKPOINT_PATH", "/unused/checkpoint")
    args = build_parser().parse_args(["--sam3-endpoint", "http://unused:9999"])
    started, waited = [], []
    monkeypatch.setattr(
        robot_spec,
        "try_spawn_server",
        lambda owned, events, name, fn: started.append(name) or (None, object()),
    )
    monkeypatch.setattr(
        robot_spec,
        "try_wait_server",
        lambda owned, events, name, *args, **kwargs: waited.append(name) or {},
    )
    robot_spec._init_runtime(
        args, tmp_path, SimpleNamespace(emit=lambda event: None), {"env", "vla"}
    )
    assert started == ["env", "vla"] and waited == ["env", "vla"]


@pytest.mark.parametrize("steps", [0, -1])
def test_invalid_chunk_length_rejected_before_runtime(monkeypatch, steps):
    from tests.e2e_tests.dual_franka import dual_franka_vla

    monkeypatch.setattr(
        dual_franka_vla,
        "get_robot_spec",
        lambda: pytest.fail("must reject before initializing runtime"),
    )
    with pytest.raises(ValueError, match="must be positive"):
        dual_franka_vla.run_session(SimpleNamespace(expected_action_steps=steps))


def test_configured_chunk_length_is_checked_before_execution(tmp_path):
    session, calls, actions = make_session(tmp_path)
    session.expected_steps = 5
    with pytest.raises(ValueError, match="Expected finite actions"):
        session.chunk(execute=True)
    assert "execute" not in calls
    session.model.predict = lambda *args, **kwargs: actions[:5]
    assert session.chunk(execute=True)["actions"] == 5
    assert calls.count("execute") == 1


def test_dual_franka_spawns_shared_vla_server():
    from robots.dual_franka.robot_spec import _vla_server_command
    from tests.e2e_tests.dual_franka.dual_franka_vla import build_parser

    args = build_parser().parse_args(
        [
            "--vla-model-path",
            "/checkpoint",
            "--vla-repo-id",
            "test/data",
            "--cuda-device",
            "2",
        ]
    )
    command = _vla_server_command(args, host="127.0.0.1", port=6000)
    assert command[1:5] == [
        "-m",
        "rpent.robots.components.pi05_vla_server",
        "--embodiment",
        "dual_franka",
    ]
    assert command[command.index("--repo-id") + 1] == "test/data"
    assert command[command.index("--model-path") + 1] == "/checkpoint"
    assert command[command.index("--cuda-device") + 1] == "2"


def test_vla_status_queries_metadata_without_inference():
    from unittest.mock import Mock

    from rpent.robots.components.pi05_vla_client import Pi05VLAClient

    metadata = {"config": {"openpi": {"action_chunk": 20}}}
    rpc = Mock()
    rpc.call.return_value = metadata
    client = Pi05VLAClient(rpc, embodiment="dual_franka")
    assert client.status(timeout_s=5) == metadata
    rpc.call.assert_called_once_with("vla.status", timeout_s=5)


def test_status_command_keeps_explicit_validation_length(tmp_path):
    session, calls, _ = make_session(tmp_path)
    metadata = {"config": {"openpi": {"action_chunk": 7}}, "device": "cuda:0"}
    session.model.status = lambda **kwargs: metadata
    lines = iter(["status", "quit"])
    output = []
    run_console(session, read=lambda _: next(lines), emit=output.append)
    result = json.loads(output[-1])
    assert result["device"] == "cuda:0"
    assert result["expected_action_steps"] == 20
    assert result["prompt"] == "pick"
    assert session.status == metadata
    assert calls == []


@pytest.mark.parametrize("reported_steps", [0, -1, None])
def test_invalid_or_unavailable_status_does_not_fall_back(
    tmp_path, monkeypatch, reported_steps
):
    from tests.e2e_tests.dual_franka import dual_franka_vla

    config_path = tmp_path / "robot.yaml"
    config_path.write_text("workspace: {}\n")
    args = SimpleNamespace(
        expected_action_steps=None,
        robot_config=str(config_path),
        task_id=1,
        instruction="pick",
    )
    closed = []

    def status(**kwargs):
        if reported_steps is None:
            raise RuntimeError("vla.status unavailable")
        return {"config": {"openpi": {"action_chunk": reported_steps}}}

    spec = SimpleNamespace(
        parse_config=lambda args: SimpleNamespace(output_dir=tmp_path / "run"),
        init_runtime=lambda *args: (
            [SimpleNamespace(stop=lambda: closed.append(True))],
            {"model": SimpleNamespace(status=status)},
        ),
    )
    monkeypatch.setattr(dual_franka_vla, "get_robot_spec", lambda: spec)
    monkeypatch.setattr(
        dual_franka_vla,
        "run_console",
        lambda session: pytest.fail("invalid metadata must not reach the console"),
    )
    error = RuntimeError if reported_steps is None else ValueError
    with pytest.raises(error, match="unavailable|must be positive"):
        dual_franka_vla.run_session(args)
    assert closed == [True]


def test_dual_vla_status_and_prediction_follow_component_rpc_contract(monkeypatch):
    import sys
    from types import ModuleType

    import torch

    from rpent.robots.components.pi05_vla_server import Pi05VLAFacade

    calls = []

    class Model:
        def cuda(self):
            return self

        def eval(self):
            return self

        def parameters(self):
            return iter([torch.zeros(1)])

        def predict_action_batch(self, obs, mode):
            calls.append(mode)
            return torch.ones((20, 20)), None

    def get_model(cfg, torch_dtype):
        assert cfg.action_dim == 20 and cfg.openpi.num_images_in_input == 3
        assert cfg.openpi_data.repo_id == "test/dataset"
        return Model()

    loader = ModuleType("rlinf.models.embodiment.openpi")
    loader.get_model = get_model
    monkeypatch.setitem(sys.modules, loader.__name__, loader)
    facade = Pi05VLAFacade(
        model_path="/unused/checkpoint",
        embodiment="dual_franka",
        repo_id="test/dataset",
    )
    try:
        status = facade._dispatch("vla.status", (), {})
        assert status["config"]["openpi"]["action_chunk"] == 20
        assert calls == []
        actions = facade._dispatch("vla.predict", ({},), {"options": {"mode": "eval"}})
        assert actions.shape == (20, 20) and actions.dtype == np.float32
        assert calls == ["eval"]
    finally:
        facade.close()
