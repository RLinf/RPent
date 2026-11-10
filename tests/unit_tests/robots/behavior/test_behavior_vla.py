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

"""VLA service, policy adaptation and the offline agent loop."""

from __future__ import annotations

import contextlib
import json
import subprocess
import sys
from io import BytesIO
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import numpy as np
import pytest
from PIL import Image
from pydantic_ai import BinaryContent, ModelRequest, ToolReturnPart
from pydantic_ai.models.function import DeltaToolCall, FunctionModel

from robots.behavior import vla_server
from robots.behavior.robot_spec import get_robot_spec
from robots.behavior.toolkit import BehaviorToolkit
from robots.behavior.vla_client import BehaviorVLAClient
from robots.behavior.vla_policy import RlinfBehaviorPolicy, adapt_observation
from robots.behavior.vla_server import BehaviorVLAFacade, TracedPolicy
from rpent.dashboard.events import NullDashboardEventSink
from rpent.memory import MemoryManager
from rpent.planner.api_loop import ApiAgentLoop


class FakePolicy:
    def __init__(self):
        self.inputs = []
        self.reset_count = 0
        self.close_count = 0
        self.actions = np.arange(32 * 23, dtype=np.float32).reshape(32, 23)

    def infer_chunk(self, observation, prompt=None):
        self.inputs.append((observation, prompt))
        return self.actions

    def reset(self):
        self.reset_count += 1

    def close(self):
        self.close_count += 1


class LocalRpc:
    def __init__(self, facade):
        self.facade = facade
        self.calls = []

    def call(self, method, args=(), kwargs=None, timeout_s=None):
        self.calls.append(method)
        return self.facade._dispatch(method, args, kwargs or {})


def cli_args(output_dir, *extra):
    return [
        "--checkpoint",
        "/unused/checkpoint",
        "--task-prompt",
        "make popcorn",
        "--output-dir",
        str(output_dir),
        *extra,
    ]


def test_main_task_inference_and_reset_keep_unmodified_actions(tmp_path):
    policy = FakePolicy()
    backend = TracedPolicy(policy, task_prompt="make popcorn", output_dir=tmp_path)
    rpc = LocalRpc(BehaviorVLAFacade(backend))
    client = BehaviorVLAClient(rpc)
    observation = {"camera": np.zeros((2, 2, 3), dtype=np.uint8)}

    client.reset()
    actions = client.infer(observation)
    client.reset()
    assert client.infer(observation, "make popcorn") is actions

    assert rpc.calls == ["vla.reset", "vla.predict", "vla.reset", "vla.predict"]
    assert policy.reset_count == 2
    assert policy.inputs == [(observation, "make popcorn")] * 2
    assert actions is policy.actions
    assert backend.calls == 2  # Reset must not reuse old artifact numbers.
    rows = [
        json.loads(line)
        for line in (tmp_path / "inference.jsonl").read_text().splitlines()
    ]
    assert [row["call"] for row in rows] == [1, 2]
    for row in rows:
        assert row["ok"] is True
        assert row["prompt"] == "make popcorn"
        assert row["duration_s"] >= 0
        assert row["finished_at"] >= row["started_at"]
        np.testing.assert_array_equal(np.load(tmp_path / row["actions_file"]), actions)


def test_main_task_service_rejects_subtask_before_running_model(tmp_path):
    policy = FakePolicy()
    backend = TracedPolicy(policy, task_prompt="make popcorn", output_dir=tmp_path)
    client = BehaviorVLAClient(LocalRpc(BehaviorVLAFacade(backend)))
    with pytest.raises(ValueError, match="fixed main-task prompt"):
        client.infer({}, prompt="open the microwave door")
    assert policy.inputs == []
    assert (tmp_path / "inference.jsonl").read_text() == ""


@pytest.mark.parametrize("actions", [np.zeros((2, 23)), np.full((32, 23), np.nan)])
def test_invalid_model_output_is_logged_but_never_returned_or_saved(tmp_path, actions):
    policy = FakePolicy()
    policy.actions = actions
    backend = TracedPolicy(policy, task_prompt="make popcorn", output_dir=tmp_path)
    with pytest.raises(ValueError):
        backend.infer_chunk({})
    row = json.loads((tmp_path / "inference.jsonl").read_text())
    assert row["ok"] is False
    assert row["error"]
    assert not list(tmp_path.glob("vla_actions_*.npy"))


def test_trace_directory_and_existing_action_files_are_not_overwritten(tmp_path):
    policy = FakePolicy()
    backend = TracedPolicy(policy, task_prompt="make popcorn", output_dir=tmp_path)
    previous = tmp_path / "vla_actions_001.npy"
    previous.write_bytes(b"historical action bytes")
    with pytest.raises(FileExistsError):
        backend.infer_chunk({})
    assert previous.read_bytes() == b"historical action bytes"
    saved_log = (tmp_path / "inference.jsonl").read_bytes()
    with pytest.raises(FileExistsError):
        TracedPolicy(FakePolicy(), task_prompt="make popcorn", output_dir=tmp_path)
    assert (tmp_path / "inference.jsonl").read_bytes() == saved_log


@pytest.mark.parametrize("owns_backend", [False, True])
def test_facade_cleanup_respects_backend_ownership(tmp_path, owns_backend):
    policy = FakePolicy()
    backend = TracedPolicy(policy, task_prompt="make popcorn", output_dir=tmp_path)
    facade = BehaviorVLAFacade(backend, owns_backend=owns_backend)
    facade.close()
    facade.close()
    assert policy.close_count == int(owns_backend)
    if owns_backend:
        with pytest.raises(RuntimeError, match="closed"):
            backend.infer_chunk({})
    else:
        backend.infer_chunk({})


def test_reset_reports_unsupported_backend_and_bad_acknowledgment():
    rpc = LocalRpc(BehaviorVLAFacade(SimpleNamespace(infer_chunk=lambda obs: None)))
    with pytest.raises(RuntimeError, match="does not support policy reset"):
        BehaviorVLAClient(rpc).reset()
    with pytest.raises(RuntimeError, match="not acknowledged"):
        BehaviorVLAClient(SimpleNamespace(call=lambda *a, **k: {"ok": False})).reset()


@pytest.mark.parametrize(
    "extra",
    [
        ["--task-prompt", " "],
        ["--port=-1"],
        ["--port=65536"],
        ["--seed=-1"],
        ["--seed=4294967296"],
        ["--device", ""],
    ],
)
def test_bad_cli_config_fails_without_output_or_model_load(tmp_path, extra):
    with pytest.raises(SystemExit) as error:
        vla_server.parse_args(cli_args(tmp_path / "out", *extra))
    assert error.value.code == 2
    assert not (tmp_path / "out").exists()


def test_cli_help_does_not_import_optional_model_or_simulator_packages():
    code = """
import importlib.abc
import sys
class NoModelImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch', 'behavior_eval', 'rlinf', 'omnigibson'}:
            raise AssertionError('optional import during --help: ' + fullname)
sys.meta_path.insert(0, NoModelImports())
from robots.behavior.vla_server import main
main(['--help'])
"""
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
    assert "--checkpoint" in result.stdout
    assert "--task-prompt" in result.stdout


def test_real_loader_uses_existing_transforms_and_validated_config(monkeypatch):
    events = []
    model = SimpleNamespace(
        to=lambda **kwargs: events.append(("to", kwargs)),
        eval=lambda: events.append(("eval",)),
    )
    policy = object()

    def policy_factory(backend, **kwargs):
        events.append(("policy", backend, kwargs))
        return policy

    modules = {
        "torch": {"manual_seed": lambda seed: events.append(("seed", seed))},
        "rlinf.models.embodiment.openpi_pytorch": {
            "get_model": lambda cfg: events.append(("config", cfg)) or model
        },
    }
    for name, attrs in modules.items():
        module = ModuleType(name)
        module.__dict__.update(attrs)
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setattr(
        np.random, "seed", lambda seed: events.append(("np_seed", seed))
    )
    monkeypatch.setattr(vla_server, "RlinfBehaviorPolicy", policy_factory)

    assert vla_server.load_policy("/checkpoint", task_prompt="make popcorn") is policy
    config = next(event for event in events if event[0] == "config")
    assert config[1].model_path == "/checkpoint"
    assert config[1].precision == "bf16"
    assert config[1].openpi.task == "eval"
    assert config[1].openpi.mode == "vla"
    assert config[1].openpi.state_token == "abs_joint"
    assert config[1].openpi.config_name == "pi05_behavior"
    assert config[1].num_action_chunks == 32
    assert config[1].action_dim == 23
    assert config[1].openpi.model_action_dim == 32
    assert config[1].num_steps == 5
    assert ("to", {"device": "cuda:0"}) in events
    assert ("eval",) in events
    assert events[-1][2] == {
        "task_prompt": "make popcorn",
    }


def test_service_main_passes_transport_and_closes_owned_policy(tmp_path, monkeypatch):
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "model.safetensors").touch()
    output = tmp_path / "service"
    policy = FakePolicy()
    monkeypatch.setattr(vla_server, "load_policy", lambda *a, **k: policy)
    monkeypatch.setattr(vla_server, "init_output_dir", lambda path: path.mkdir())
    served = []

    def fake_serve(self, **kwargs):
        served.append(kwargs)
        client = BehaviorVLAClient(LocalRpc(self))
        client.reset()
        client.infer({})
        assert self._dispatch("shutdown", (), {}) == {"ok": True}
        assert self._shutdown_event.is_set()

    monkeypatch.setattr(BehaviorVLAFacade, "serve", fake_serve)
    vla_server.main(
        cli_args(output, "--checkpoint", str(checkpoint), "--port", "25000")
    )
    assert served == [
        {
            "transport": "socket",
            "host": "127.0.0.1",
            "port": 25000,
            "parent_watch": False,
        }
    ]
    assert policy.close_count == 1
    manifest = json.loads((output / "vla-service.json").read_text())
    assert manifest["prompt_mode"] == "main"
    assert manifest["environment_execution"] is False
    assert (output / "vla_actions_001.npy").is_file()
    with pytest.raises(FileExistsError, match="must be empty"):
        vla_server.main(cli_args(output, "--checkpoint", str(checkpoint)))


def observation():
    return {
        "robot::proprio": np.arange(61, dtype=np.float64),
        "robot::zed_link:Camera:0::rgb": np.full((240, 240, 4), 17, np.uint8),
        "robot::left_realsense_link:Camera:0::rgb": np.full(
            (240, 240, 3), 31, np.uint8
        ),
        "robot::right_realsense_link:Camera:0::rgb": np.full(
            (240, 240, 3), 49, np.uint8
        ),
    }


def test_official_r1pro_fields_are_embedded_without_reordering_or_normalizing():
    raw = observation()
    adapted = adapt_observation(raw, "full task")
    state = adapted["states"][0]
    assert state.dtype == np.float32
    # Public R1Pro layout and RLinf BehaviorInputs' legacy source positions.
    expected = np.zeros(256, dtype=np.float32)
    expected[158:165] = np.arange(3, 10)  # Left arm joints.
    expected[193:195] = [24, 25]  # Both measured left fingers remain separate.
    expected[197:204] = np.arange(28, 35)  # Right arm joints.
    expected[232:234] = [49, 50]
    expected[236:240] = [53, 54, 55, 56]  # Trunk.
    expected[253:256] = [0, 1, 2]  # Base velocity.
    np.testing.assert_array_equal(state, expected)
    assert adapted["main_images"].shape == (1, 240, 240, 3)
    assert adapted["wrist_images"].shape == (1, 2, 240, 240, 3)
    np.testing.assert_array_equal(adapted["main_images"], 17)
    np.testing.assert_array_equal(adapted["wrist_images"][0, 0], 31)
    np.testing.assert_array_equal(adapted["wrist_images"][0, 1], 49)
    assert adapted["task_descriptions"].tolist() == ["full task"]
    assert raw["robot::zed_link:Camera:0::rgb"].shape[-1] == 4


def test_legacy_256_state_and_chw_float_camera_keep_existing_conversion():
    raw = {key.encode(): value for key, value in observation().items()}
    raw[b"robot::proprio"] = np.linspace(-2, 2, 300)
    rgb = np.stack(
        [np.full((240, 240), value, np.float32) for value in (0.0, 0.5, 1.0)]
    )
    raw[b"robot::zed_link:Camera:0::rgb"] = rgb
    adapted = adapt_observation(raw, "main")
    np.testing.assert_array_equal(
        adapted["states"][0], raw[b"robot::proprio"][:256].astype(np.float32)
    )
    assert adapted["main_images"][0, 0, 0].tolist() == [0, 127, 255]


@pytest.mark.parametrize("kind", ["array", "batched", "dictionary"])
def test_eval_backend_retains_native_actions_and_resets_without_action_queue(
    monkeypatch, kind
):
    monkeypatch.setitem(
        sys.modules, "torch", SimpleNamespace(inference_mode=contextlib.nullcontext)
    )
    actions = np.linspace(-7, 7, 32 * 32, dtype=np.float64).reshape(32, 32)
    output = actions
    if kind == "batched":
        output = actions[None]
    elif kind == "dictionary":
        output = {"actions": actions[None]}
    calls, resets = [], []

    def predict(env_obs, **kwargs):
        calls.append((env_obs, kwargs))
        return output, {}

    backend = SimpleNamespace(
        predict_action_batch=predict, reset=lambda: resets.append(True)
    )
    policy = RlinfBehaviorPolicy(backend, task_prompt="original task")
    returned = policy.infer_chunk(observation(), prompt="request task")
    np.testing.assert_array_equal(returned, actions[:, :23].astype(np.float32))
    assert returned.flags.c_contiguous
    assert returned.min() < -1 and returned.max() > 1  # No clipping/normalizing.
    assert calls[0][1] == {"mode": "eval"}
    assert calls[0][0]["task_descriptions"].tolist() == ["request task"]
    returned[:] = 0
    policy.reset()
    again = policy.infer_chunk(observation())
    assert resets == [True]
    assert len(calls) == 2
    assert calls[1][0]["task_descriptions"].tolist() == ["original task"]
    np.testing.assert_array_equal(again, actions[:, :23].astype(np.float32))


def test_missing_or_ambiguous_sensor_never_reaches_the_model(monkeypatch):
    monkeypatch.setitem(
        sys.modules, "torch", SimpleNamespace(inference_mode=contextlib.nullcontext)
    )
    calls = []
    policy = RlinfBehaviorPolicy(
        SimpleNamespace(predict_action_batch=lambda *a, **k: calls.append(1)),
        task_prompt="task",
    )
    raw = observation()
    del raw["robot::left_realsense_link:Camera:0::rgb"]
    with pytest.raises(KeyError, match="left_realsense"):
        policy.infer_chunk(raw)
    raw = observation()
    raw["other::proprio"] = raw["robot::proprio"]
    with pytest.raises(KeyError, match="proprio"):
        policy.infer_chunk(raw)
    assert calls == []


_CAMERAS = (
    "robot_r1::robot_r1:zed_link:Camera:0::rgb",
    "robot_r1::robot_r1:left_realsense_link:Camera:0::rgb",
    "robot_r1::robot_r1:right_realsense_link:Camera:0::rgb",
)


_TASK = "Put the popcorn bag in the microwave and cook the popcorn."


class _Environment:
    def __init__(self) -> None:
        self.action_chunks: list[np.ndarray] = []
        self.actions_executed = 0

    def get_observation(self) -> dict[str, Any]:
        observation = {
            key: np.full(
                (1, 6, 8, 3),
                10 * (index + 1) + self.actions_executed,
                dtype=np.uint8,
            )
            for index, key in enumerate(_CAMERAS)
        }
        proprio = np.zeros(61, dtype=np.float32)
        proprio[[23, 48]] = 1
        proprio[17:20] = [0.4 + self.actions_executed / 1000, 0.2, 0.8]
        proprio[42:45] = [0.4, -0.2, 0.8]
        proprio[24:26] = 0.05
        proprio[49:51] = 0.03
        observation["robot_r1::proprio"] = proprio[None]
        observation["task_id"] = np.array([40])
        return observation

    def get_session_status(self) -> dict[str, Any]:
        return {"ended": False, "reason": None, "outcome": "unknown", "generation": 0}

    def chunk_step(self, actions: np.ndarray) -> tuple:
        self.action_chunks.append(actions.copy())
        count = len(actions)
        self.actions_executed += count
        return (
            self.get_observation(),
            None,
            None,
            None,
            {
                "executed_actions": count,
                "issued_actions": count,
                "confirmed_actions": count,
                "unconfirmed_actions": 0,
                "observation_after_last_issued_action": True,
                "stop_reason": "chunk_completed",
            },
        )


class _VLA:
    def __init__(self) -> None:
        self.inputs: list[tuple[dict[str, Any], str]] = []
        self.prediction = np.arange(64 * 23, dtype=np.float32).reshape(64, 23)

    def infer(self, observation: dict[str, Any], prompt: str) -> np.ndarray:
        self.inputs.append((observation, prompt))
        return self.prediction.copy()


def _model_observation(part: ToolReturnPart) -> tuple[dict[str, Any], list[int]]:
    """Read the JSON and three PNG images actually delivered through the SDK."""
    assert isinstance(part.content, list)
    text = [item for item in part.content if isinstance(item, str)]
    images = [item for item in part.content if isinstance(item, BinaryContent)]
    assert len(text) == 1 and len(images) == 3
    pixels = []
    for image in images:
        assert image.media_type == "image/png"
        with Image.open(BytesIO(image.data)) as decoded:
            pixels.append(decoded.getpixel((0, 0))[0])
    return json.loads(text[0]), pixels


def test_api_planner_observes_then_executes_one_chunk_and_sees_new_views(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("rpent.utils.templates.get_output_dir", lambda: tmp_path)
    monkeypatch.setenv("PYDANTIC_AI_NO_BANNER", "1")
    env, vla = _Environment(), _VLA()
    events = NullDashboardEventSink()
    toolkit = BehaviorToolkit(
        runtime_kwargs={
            "env": env,
            "model": vla,
            "task_id": 40,
            "task_language": _TASK,
            "execution_horizon": 32,
            "max_chunks_per_call": 1,
            "instruction_mode": "main",
            "tool_allowlist": ["view_env_state", "vla_act", "finish"],
        },
        dashboard_events=events,
        memory=MemoryManager(tmp_path / "memory", memory_access="read_only"),
        state_output_dir=tmp_path / "state",
    )
    received_views: list[list[int]] = []
    received_feedback: list[dict[str, Any]] = []

    async def respond(messages, info):
        returned = [
            part
            for message in messages
            if isinstance(message, ModelRequest)
            for part in message.parts
            if isinstance(part, ToolReturnPart)
        ]
        if not returned:
            assert not env.action_chunks and not vla.inputs
            yield {0: DeltaToolCall(name="view_env_state", json_args='{"step": 0}')}
            return

        latest = returned[-1]
        observation, pixels = _model_observation(latest)
        received_views.append(pixels)
        assert observation["state"]["task_language"] == _TASK
        robot_state = observation["state"]["robot_state"]
        assert robot_state["available"] is True
        assert robot_state["source_step"] == observation["step"]
        assert robot_state["frame"] == "robot_base"
        assert robot_state["units"]["finger_qpos"] == "metre per finger"
        assert robot_state["left"]["eef_quaternion_xyzw"] == [0, 0, 0, 1]
        if latest.tool_name == "view_env_state":
            assert pixels == [10, 20, 30]
            assert not env.action_chunks and not vla.inputs
            assert observation["step"] == 0
            assert robot_state["left"]["eef_position"] == [0.4, 0.2, 0.8]
            yield {0: DeltaToolCall(name="vla_act", json_args='{"chunks": 1}')}
            return

        assert latest.tool_name == "vla_act"
        assert pixels == [42, 52, 62]
        assert observation["step"] == 1
        assert robot_state["left"]["eef_position"] == [0.432, 0.2, 0.8]
        feedback = observation["log"]["result"]
        received_feedback.append(feedback)
        assert feedback["requested_actions"] == feedback["confirmed_actions"] == 32
        assert feedback["observation_fresh"] is True
        yield {
            0: DeltaToolCall(
                name="finish",
                json_args=json.dumps(
                    {
                        "status": "success",
                        "summary": "Observed the next three views.",
                    }
                ),
            )
        }

    prompts = get_robot_spec().prompts
    variables = {
        "task_id": 40,
        "task_name": "make_microwave_popcorn",
        "task_language": _TASK,
        "execution_horizon": 32,
        "max_chunks_per_call": 1,
        "instruction_mode": "main",
        "tool_allowlist": ["view_env_state", "vla_act", "finish"],
        "memory_dir": str(tmp_path / "memory"),
    }
    planner = ApiAgentLoop(
        model=FunctionModel(stream_function=respond),
        dashboard_events=events,
        timeout_s=30,
    )
    try:
        result = planner.solve(
            system_prompt=prompts.render("system", variables=variables),
            user_message=prompts.render("user", variables=variables),
            toolkit=toolkit,
            max_turns=3,
        )
        assert result.error is None
        assert result.stats["turns_used"] == 3
        assert result.stats["tool_calls"] == 3
        assert received_views == [[10, 20, 30], [42, 52, 62]]
        assert len(received_feedback) == len(vla.inputs) == len(env.action_chunks) == 1
        assert env.actions_executed == 32
        np.testing.assert_array_equal(env.action_chunks[0], vla.prediction[:32])
        assert vla.inputs[0][1] == _TASK
        assert vla.inputs[0][0][_CAMERAS[0]].shape == (6, 8, 3)
        assert toolkit.state.latest_step == 1
        assert result.finish_result["status"] == "unverified"
        assert [
            entry["name"] for entry in result.messages if entry["role"] == "tool"
        ] == [
            "view_env_state",
            "vla_act",
            "finish",
        ]
    finally:
        toolkit.close()
