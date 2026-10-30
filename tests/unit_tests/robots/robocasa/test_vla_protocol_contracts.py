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

"""Offline contracts for RoboCasa live-task VLA execution."""

from __future__ import annotations

import json
from contextlib import nullcontext
from copy import deepcopy
from functools import partial
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from robots.robocasa.env_client import RoboCasaEnvClient
from robots.robocasa.env_server import DEFAULT_CAMS, RoboCasaEnvFacade
from robots.robocasa.primitives import RoboCasaPrimitives
from robots.robocasa.prompt_bundle import system_prompt
from robots.robocasa.toolkit import _capture_robocasa_call
from robots.robocasa.vla_server import _normalize_legacy_processor_geometry
from rpent.dashboard.events import NullDashboardEventSink
from rpent.flywheel.reader import read_array
from rpent.memory import MemoryManager
from rpent.prompt.utils import format_prompt
from rpent.robots.components.action_spec import box_action_spec
from rpent.robots.components.env_facade_base import BaseEnvFacade
from rpent.session import EnvState, Payload
from rpent.session.env_capture import CapturingEnvClient, ToolCallCapture
from rpent.tools import Toolkit, ToolResult, iter_tools
from rpent.tools.direct_action import direct_action_tool
from rpent.utils.rpc import RpcError


class _RecordingRldx:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def run(
        self,
        prompt: str,
        max_chunks: int,
        n_action_steps: int,
        **kwargs: Any,
    ) -> dict[str, Any]:
        self.calls.append(
            {
                "prompt": prompt,
                "max_chunks": max_chunks,
                "n_action_steps": n_action_steps,
                **kwargs,
            }
        )
        return {"ok": True, "prompt": prompt, "status": "cap"}


def _fake_primitives(task_language: str) -> tuple[RoboCasaPrimitives, _RecordingRldx]:
    rldx = _RecordingRldx()
    primitives = RoboCasaPrimitives.__new__(RoboCasaPrimitives)
    primitives.env = SimpleNamespace(
        current_raw_obs={"language": task_language},
        get_task_language=lambda: task_language,
    )
    primitives._rldx = rldx
    primitives._vla_desync = True
    primitives._recording = False
    primitives.record_frame = lambda: None
    return primitives, rldx


def test_prompt_requires_live_task_language_and_fresh_geometry(tmp_path: Path) -> None:
    rendered = format_prompt(
        system_prompt(),
        variables={
            "task_name": "OpenDrawer",
            "reference_tag": "OpenDrawer_s0",
            "memory_profile": "hf",
            "memory_dir": str(tmp_path / "results"),
        },
    )

    assert "complete live task_language" in rendered
    assert "Never shorten, paraphrase" in rendered
    assert "Only after two" in rendered
    assert (
        "consecutive calls with no contact and no visible semantic progress" in rendered
    )
    assert "Historical entries may name vla_act" in rendered
    assert "Never replay stored xyz, xy, pixels, base poses" in rendered
    assert "{{" not in rendered


def test_vla_tool_schema_hides_historical_prompt_override() -> None:
    vla_specs = {
        spec.name: spec
        for spec in iter_tools(RoboCasaPrimitives)
        if spec.name.startswith("rldx_")
    }

    assert set(vla_specs) == {"rldx_skill", "rldx_arm"}
    for spec in vla_specs.values():
        schema = spec.input_schema
        assert schema["required"] == ["prompt"]
        assert "use_prompt" not in schema["properties"]
        assert "complete live task_language" in spec.description


def test_vla_always_uses_live_task_language_and_preserves_continuity() -> None:
    task_language = "Pick the squash up and place it in the microwave."
    primitives, rldx = _fake_primitives(task_language)

    first = primitives.rldx_skill(
        prompt="Pick the squash up.",
        use_prompt=True,
        max_chunks=3,
    ).data
    second = primitives.rldx_arm(
        prompt="Place it in the microwave.",
        use_prompt=True,
        max_chunks=4,
    ).data

    assert [call["prompt"] for call in rldx.calls] == [task_language, task_language]
    assert rldx.calls[0]["force_reset"] is True
    assert rldx.calls[1]["force_reset"] is False
    assert primitives._vla_desync is False
    assert first["effective_prompt"] == task_language
    assert first["effective_max_chunks"] == 3
    assert first["prompt_overridden"] is True
    assert first["requested_prompt"] == "Pick the squash up."
    assert second["effective_prompt"] == task_language
    assert second["effective_max_chunks"] == 4
    assert second["prompt_overridden"] is True


def test_environment_max_chunks_locks_the_formal_protocol(monkeypatch) -> None:
    task_language = "Open the left drawer."
    primitives, rldx = _fake_primitives(task_language)
    monkeypatch.setenv("RLDX_MAX_CHUNKS", "40")

    result = primitives.rldx_skill(prompt=task_language, max_chunks=70).data

    assert rldx.calls[0]["max_chunks"] == 40
    assert result["effective_max_chunks"] == 40


def test_ordinary_robocasa_keeps_default_max_chunks_at_70(monkeypatch) -> None:
    task_language = "Open the left drawer."
    primitives, rldx = _fake_primitives(task_language)
    monkeypatch.delenv("RLDX_MAX_CHUNKS", raising=False)

    result = primitives.rldx_skill(prompt=task_language).data

    assert rldx.calls[0]["max_chunks"] == 70
    assert result["effective_max_chunks"] == 70


def test_environment_locks_all_formal_rldx_runtime_values(monkeypatch) -> None:
    task_language = "Open the left drawer."
    primitives, rldx = _fake_primitives(task_language)
    monkeypatch.setenv("RLDX_MAX_CHUNKS", "40")
    monkeypatch.setenv("RLDX_ACTION_STEPS_PER_CHUNK", "8")
    monkeypatch.setenv("RLDX_SETTLE_PATIENCE", "999")

    result = primitives.rldx_skill(
        prompt=task_language,
        max_chunks=2,
        n_action_steps=1,
        settle_patience=2,
    ).data

    assert rldx.calls[0]["max_chunks"] == 40
    assert rldx.calls[0]["n_action_steps"] == 8
    assert rldx.calls[0]["settle_patience"] == 999
    assert result["effective_max_chunks"] == 40
    assert result["effective_n_action_steps"] == 8
    assert result["effective_settle_patience"] == 999


def test_invalid_vla_budgets_return_errors_without_executing_rldx(
    monkeypatch,
) -> None:
    monkeypatch.delenv("RLDX_MAX_CHUNKS", raising=False)
    monkeypatch.delenv("RLDX_ACTION_STEPS_PER_CHUNK", raising=False)
    monkeypatch.delenv("RLDX_SETTLE_PATIENCE", raising=False)

    for arguments, parameter in (
        ({"max_chunks": 0}, "max_chunks"),
        ({"n_action_steps": 0}, "n_action_steps"),
        ({"settle_patience": 0}, "settle_patience"),
    ):
        primitives, rldx = _fake_primitives("Open the left drawer.")

        result = primitives.rldx_skill(
            prompt="Open the left drawer.",
            **arguments,
        ).data

        assert result == {
            "error": f"{parameter} must be positive; VLA was not executed"
        }
        assert rldx.calls == []


def test_matching_vla_prompt_is_reported_without_override() -> None:
    task_language = "Open the left drawer."
    primitives, rldx = _fake_primitives(task_language)

    result = primitives.rldx_skill(prompt=task_language, use_prompt=False).data

    assert rldx.calls[0]["prompt"] == task_language
    assert result["effective_prompt"] == task_language
    assert result["prompt_overridden"] is False
    assert "requested_prompt" not in result


def test_vla_does_not_run_without_environment_task_language() -> None:
    primitives, rldx = _fake_primitives("")

    result = primitives.rldx_skill(prompt="atomic fallback", use_prompt=True).data

    assert "task language is unavailable" in result["error"]
    assert rldx.calls == []
    assert primitives._vla_desync is True


def test_legacy_rldx_processor_null_geometry_uses_release_defaults(monkeypatch) -> None:
    processor = SimpleNamespace(
        image_max_area=None,
        image_resize_m=None,
        random_crop_fraction=None,
        random_rotation_angle=None,
        color_jitter_params=None,
    )
    calls = []

    def fake_build(candidate):
        calls.append((candidate.image_max_area, candidate.image_resize_m))
        return "train-transform", "eval-transform"

    monkeypatch.setattr(
        "robots.robocasa.vla_server._build_processor_image_transforms",
        fake_build,
    )

    assert _normalize_legacy_processor_geometry(processor) is True
    assert processor.image_max_area == 65536
    assert processor.image_resize_m == 32
    assert processor.train_image_transform == "train-transform"
    assert processor.eval_image_transform == "eval-transform"
    assert calls == [(65536, 32)]


def test_current_rldx_processor_geometry_is_not_rebuilt(monkeypatch) -> None:
    processor = SimpleNamespace(image_max_area=131072, image_resize_m=64)

    def unexpected_build(candidate):
        raise AssertionError(f"unexpected transform rebuild for {candidate!r}")

    monkeypatch.setattr(
        "robots.robocasa.vla_server._build_processor_image_transforms",
        unexpected_build,
    )

    assert _normalize_legacy_processor_geometry(processor) is False
    assert processor.image_max_area == 131072
    assert processor.image_resize_m == 64


class _Backend:
    action_dim = 12

    def __init__(self):
        self.executed = []
        self.reset_count = 0
        self.observation_reads = 0
        self.fail_at = None
        self.obs = {
            key: np.zeros(size)
            for key, size in (
                ("robot0_eef_pos", 3),
                ("robot0_eef_quat", 4),
                ("robot0_gripper_qpos", 2),
                ("robot0_base_pos", 3),
                ("robot0_base_quat", 4),
                ("robot0_base_to_eef_pos", 3),
                ("robot0_base_to_eef_quat", 4),
            )
        }

    def reset(self):
        self.reset_count += 1
        self.executed.clear()
        for value in self.obs.values():
            value.fill(0)
        return self.obs

    def _get_observations(self):
        self.observation_reads += 1
        return self.obs

    def step(self, action):
        self.executed.append(action.copy())
        count = len(self.executed)
        for value in self.obs.values():
            value.fill(count)
        if count == self.fail_at:
            raise RuntimeError("native step failed")
        return self.obs, float(count), False, {"count": count}

    def _check_success(self):
        return False

    def get_ep_meta(self):
        return {"lang": "Open the left drawer."}

    def render_camera(self, camera_name, height, width, depth=False):
        assert not depth
        rows = np.arange(height, dtype=np.uint8) + 10 * len(self.executed)
        return np.broadcast_to(rows[:, None, None], (height, width, 3)).copy()


class _VLA:
    def __init__(self):
        self.inputs = []

    def get_modality_config(self):
        return {"video_delta_indices": [-2, 0], "hist_maxlen": 3}

    def predict(self, obs, options):
        self.inputs.append((deepcopy(obs), deepcopy(options)))
        return {
            "action.end_effector_position": np.full((1, 4, 3), 0.25),
            "action.end_effector_rotation": np.full((1, 4, 3), 0.1),
            "action.gripper_close": np.array([[[0.2], [0.8], [0.4], [0.6]]]),
            "action.base_motion": np.tile([0.8, -0.8, 0.2, -0.2], (1, 4, 1)),
            "action.control_mode": np.full((1, 4, 1), 0.2),
        }


@pytest.fixture
def facade(monkeypatch):
    for name in (
        "RLDX_RESET_SEED",
        "RLDX_VIDEO_DIR",
        "RLDX_MAX_CHUNKS",
        "RLDX_ACTION_STEPS_PER_CHUNK",
        "RLDX_SETTLE_PATIENCE",
    ):
        monkeypatch.delenv(name, raising=False)
    facade = RoboCasaEnvFacade.__new__(RoboCasaEnvFacade)
    facade.env = _Backend()
    facade.cameras = DEFAULT_CAMS
    facade.camera_h = facade.camera_w = 4
    facade._meta = {"camera_h": 4, "camera_w": 4}
    facade.render_camera = facade.env.render_camera
    facade.grasp_contact = lambda: (False, None)
    facade.reassemble_env_action = lambda action: action
    facade.get_action_spec = lambda: {
        "default": box_action_spec(-np.ones(12), np.ones(12), "test action")
    }
    BaseEnvFacade.__init__(facade)
    return facade


def test_capture_robocasa_call_snapshots_native_step(
    facade, transport, make_server_and_client
):
    with make_server_and_client(facade, transport) as rpc:
        client = RoboCasaEnvClient(rpc, expected_meta=facade._meta)
        initial_obs = client.last_obs
        action = np.arange(12, dtype=np.float32).reshape(3, 4) / 10
        response = _capture_robocasa_call(client, "step", action)
        obs, reward, done, info = response["result"]
        assert obs is client.last_obs
        assert (reward, done, info) == (1.0, False, {"count": 1})
        (step,) = response["steps"]
        assert step["action"]["data"].dtype == np.float64
        np.testing.assert_array_equal(step["action"]["data"], facade.env.executed[0])
        assert step["result"]["data"] == {
            "reward": reward,
            "done": done,
            "info": info,
        }

        client.step(np.zeros(12))
        for payload, count in (
            (response["before_observation"], 0),
            (step["after_observation"], 1),
        ):
            np.testing.assert_array_equal(
                payload["data"]["robot0_eef_pos"], [count] * 3
            )
            assert "/robot0_eef_pos" in payload["inline_array_paths"]
            assert set(payload["data"]["images"]) == set(DEFAULT_CAMS)
            for image in payload["data"]["images"].values():
                assert image.shape == (4, 4, 3)
                np.testing.assert_array_equal(
                    image[:, 0, 0], np.arange(4)[::-1] + count * 10
                )
        assert all(
            "images" not in value for value in (initial_obs, obs, client.last_obs)
        )


def _primitives(env, tmp_path):
    primitives = RoboCasaPrimitives(env, tmp_path, 4, _VLA(), allow_reset=False)
    skill = primitives._rldx
    skill.VLA_OBS_RES = 4
    # Stand in for the optional RoboCasa converter; keep the skill's conversion call.
    skill._unmap = lambda fields: np.concatenate(
        [
            np.where(value >= 0.5, 1.0, -1.0)
            if key in ("action.gripper_close", "action.control_mode")
            else value
            for key, value in fields.items()
        ]
    )
    return primitives


@pytest.mark.parametrize(
    ("transport", "mode"),
    [
        ("socket", "capture"),
        ("http", "capture"),
        ("socket", "disabled"),
        ("http", "unwrapped"),
    ],
)
def test_robocasa_tools_capture_executed_actions_and_rldx_history(
    facade, transport, make_server_and_client, tmp_path, monkeypatch, mode
):
    state = EnvState(tmp_path)
    recording = mode == "capture"
    planner = None
    if recording:
        state.start_task(
            instruction="Open the left drawer.", config=Payload("test", {})
        )
        planner = state.register_planner(backend="api")
    monkeypatch.setattr("rpent.utils.templates.get_output_dir", lambda: tmp_path)
    toolkit = Toolkit(
        state=state,
        dashboard_events=NullDashboardEventSink(),
        memory=MemoryManager(tmp_path / "memory"),
    )
    monkeypatch.setattr(
        toolkit, "get_env_state", lambda **kw: ToolResult(data=kw["result"])
    )

    def invoke(name, arguments):
        capture = None
        if recording:
            tool_id = state.register_tool_call(
                planner_id=planner, name=name, arguments=Payload("json", arguments)
            )
            capture = ToolCallCapture(state, tool_id)
        with capture.bind() if capture else nullcontext():
            result = toolkit.execute_tool(name, arguments).data
        assert "error" not in result
        return result

    with make_server_and_client(facade, transport) as rpc:
        client = RoboCasaEnvClient(rpc, expected_meta=facade._meta)
        env = (
            client
            if mode == "unwrapped"
            else CapturingEnvClient(
                client, state, capture_call=partial(_capture_robocasa_call, client)
            )
        )
        primitives = _primitives(env, tmp_path)
        assert primitives._rldx.env is env
        toolkit.add_tools(iter_tools(primitives))
        toolkit.add_tool(
            direct_action_tool(env.action_specs, primitives.execute_action)
        )
        invoke("set_gripper", {"steps": 1})
        result = invoke(
            "rldx_arm",
            {
                "prompt": "Open it",
                "max_chunks": 2,
                "n_action_steps": 2,
                "base_clip": 0.1,
                "settle_patience": 99,
            },
        )
        assert result["steps_applied"] == 4
        assert result["effective_prompt"] == "Open the left drawer."
        invoke("release", {"steps": 1})
        invoke("execute_action", {"values": [0.2] * 12})
        np.testing.assert_array_equal(client.last_obs["robot0_eef_pos"], [7] * 3)
        assert "images" not in client.last_obs
    assert facade.env.reset_count == 2
    assert len(facade.env.executed) == 7
    assert facade.env.observation_reads == 0
    np.testing.assert_allclose(facade.env.executed[1][7:11], [0.1, -0.1, 0.1, -0.1])
    assert [a[6] for a in facade.env.executed[1:5]] == [-1, 1, -1, 1]
    inputs = primitives._rldx._vla_client.inputs
    assert len(inputs) == 2
    for index, (observation, options) in enumerate(inputs):
        assert options == {"reset_memory": [index == 0]}
        assert observation["annotation.human.task_description"] == [
            "Open the left drawer."
        ]
        for camera in DEFAULT_CAMS:
            history = observation[f"video.{camera}"]
            assert history.shape == (1, 2, 4, 4, 3)
            for frame, count in enumerate((1, 1 + index * 2)):
                np.testing.assert_array_equal(
                    history[0, frame, :, 0, 0], np.arange(4)[::-1] + count * 10
                )
        for key, value in observation.items():
            if key.startswith("state."):
                np.testing.assert_array_equal(value, np.full_like(value, 1 + index * 2))
    if not recording:
        assert state.action_records() == state.prediction_records() == []
        assert not (tmp_path / "trajectories").exists()
        return

    state.end_task(status="completed", stop_reason="finish")
    root = state.save_planner_trajectory()
    assert json.loads((root / "task.json").read_text())["schema_version"] == 5
    actions = [
        json.loads(line) for line in (root / "actions.jsonl").read_text().splitlines()
    ]
    assert [a["source"] for a in actions] == [
        "tool_computed",
        "vla",
        "vla",
        "vla",
        "vla",
        "tool_computed",
        "direct",
    ]
    for index, action in enumerate(actions):
        assert action["before_observation_id"] == f"O{index + 1}"
        assert action["after_observation_id"] == f"O{index + 2}"
        assert action["action"]["format"] == "robocasa.action"
        native = read_array(root, action["action"]["data"])
        assert native.dtype == np.float64
        np.testing.assert_array_equal(native, facade.env.executed[index])
        assert action["result"]["data"] == {
            "reward": float(index + 1),
            "done": False,
            "info": {"count": index + 1},
        }
        for side, count in (("before", index), ("after", index + 1)):
            payload = action[f"{side}_observation"]["payload"]
            assert payload["format"] == "robocasa.observation"
            np.testing.assert_array_equal(
                read_array(root, payload["data"]["robot0_eef_pos"]), [count] * 3
            )
            for camera in DEFAULT_CAMS:
                image = read_array(root, payload["data"]["images"][camera])
                np.testing.assert_array_equal(
                    image[:, 0, 0], np.arange(4)[::-1] + count * 10
                )
    predictions = [
        json.loads(line)
        for line in (root / "predictions.jsonl").read_text().splitlines()
    ]
    assert [p["action_ids"] for p in predictions] == [["A2", "A3"], ["A4", "A5"]]
    for prediction in predictions:
        assert prediction["instruction"] == "Open the left drawer."
        assert "input_observation" not in prediction
        assert "input_observation_id" not in prediction
        assert "actions" not in prediction


def test_robocasa_failed_step_keeps_prior_steps_and_ends_prediction_scope(
    facade, transport, make_server_and_client, tmp_path
):
    state = EnvState(tmp_path)
    state.start_task(instruction="Open drawer", config=Payload("test", {}))
    tool_id = state.register_tool_call(
        planner_id=state.register_planner(backend="api"),
        name="rldx_skill",
        arguments=Payload("json", {}),
    )
    capture = ToolCallCapture(state, tool_id)
    with make_server_and_client(facade, transport) as rpc:
        client = RoboCasaEnvClient(rpc, expected_meta=facade._meta)
        env = CapturingEnvClient(
            client, state, capture_call=partial(_capture_robocasa_call, client)
        )
        primitives = _primitives(env, tmp_path)
        facade.env.fail_at = 2
        with capture.bind():
            with pytest.raises(RpcError, match="native step failed"):
                primitives._rldx.run("Open drawer", max_chunks=1, n_action_steps=3)
            assert capture.prediction_id is None
            assert len(state.action_records()) == 1
            assert state.prediction_records()[0].action_ids == ["A1"]
            env.step(np.zeros(12))
    first, last = state.action_records()
    assert first.source == "vla" and last.source == "tool_computed"
    assert last.before_observation_id != first.after_observation_id
