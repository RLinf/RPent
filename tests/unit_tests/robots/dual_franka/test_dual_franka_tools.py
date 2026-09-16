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

"""Offline dual-Franka primitive and state-capture tests."""

from __future__ import annotations

import json
from pathlib import Path
from threading import Event
from types import SimpleNamespace

import numpy as np
import pytest

from robots.dual_franka import tools
from robots.dual_franka.perception import (
    back_project_base_pixel,
    load_calibration_bundle,
)
from robots.dual_franka.runtime_config import DUAL_FRANKA_CONFIG
from robots.dual_franka.toolkit import DualFrankaToolkit
from robots.dual_franka.tools import (
    dump_state,
    view_env_state,
)
from robots.franka import runtime_config
from robots.franka import toolkit as franka_toolkit
from robots.franka.runtime_config import (
    set_calibration_path,
    set_robot_config_path,
)
from robots.franka.toolkit import FrankaRuntime
from robots.franka.tools import view_camera_meta
from rpent.dashboard.events import NullDashboardEventSink, StepRecordEvent
from rpent.memory import MemoryManager
from rpent.session import EnvState
from rpent.tools import ToolContext
from tests.unit_tests.robots.dual_franka._fakes import FakeEnv
from tests.unit_tests.robots.franka._fakes import FakeModel


def _context(env: FakeEnv, *, model=None, state=None):
    return ToolContext(
        robot=FrankaRuntime(env=env, model=model, task_description="default task"),
        state=state,
        memory=None,
        output_dir=Path("."),
        record_frame=lambda frame: None,
        _cancel_event=Event(),
    )


def test_arm_and_vec3_validation_and_motion_forwarding():
    env = FakeEnv()
    ctx = _context(env)

    tools.move_delta.handler("left", [0.01, 0.0, -0.02], ctx=ctx)
    tools.rotate_delta.handler("right", [0.0, 0.0, 0.1], ctx=ctx)
    tools.open_gripper.handler("left", ctx=ctx)
    tools.close_gripper.handler("right", ctx=ctx)

    assert env.moves[0][0] == "left"
    np.testing.assert_allclose(env.moves[0][1], [0.01, 0.0, -0.02])
    assert env.rotations[0][0] == "right"
    assert env.grippers == [("left", True), ("right", False)]


def test_dump_state_saves_three_camera_artifacts(tmp_path: Path):
    env = FakeEnv()
    ctx = _context(env)
    state = EnvState(tmp_path)

    record = dump_state(
        ctx.robot,
        state,
        command={"action": "move_delta"},
        result={"ok": True},
        elapsed_s=0.2,
    )

    assert record.artifacts == {
        "left_wrist.png",
        "left_wrist_depth.npy",
        "base.png",
        "base_depth.npy",
        "right_wrist.png",
        "right_wrist_depth.npy",
        "d455.png",
        "d455_depth.npy",
        "camera_meta.json",
    }
    output = view_env_state.handler(ctx=_context(env, state=state))
    assert output.images == [
        state.load_bytes(f"{name}.png")
        for name in ("left_wrist", "base", "right_wrist")
    ]
    # Every VLA camera persists its raw frame as the single canonical version.
    np.testing.assert_array_equal(state.load("left_wrist.png"), 5)
    np.testing.assert_array_equal(state.load("base.png"), 7)
    np.testing.assert_array_equal(state.load("right_wrist.png"), 6)
    np.testing.assert_array_equal(state.load("left_wrist_depth.npy"), 8)
    np.testing.assert_array_equal(state.load("base_depth.npy"), 9)
    np.testing.assert_array_equal(state.load("right_wrist_depth.npy"), 11)
    camera_meta = view_camera_meta.handler(ctx=_context(env, state=state)).data[
        "camera_meta"
    ]
    assert camera_meta["observation_camera_map"]["main"] == "left_wrist_0_rgb"


def test_dump_state_falls_back_to_policy_view_when_raw_missing(tmp_path: Path):
    env = FakeEnv()
    full_obs = env.get_observation()

    def base_raw_only() -> dict:
        obs = dict(full_obs)
        obs["raw_camera_frames"] = {
            "base_0_rgb": full_obs["raw_camera_frames"]["base_0_rgb"]
        }
        obs["raw_camera_depths"] = {
            "base_0_rgb": full_obs["raw_camera_depths"]["base_0_rgb"]
        }
        return obs

    env.get_observation = base_raw_only
    state = EnvState(tmp_path)
    dump_state(_context(env).robot, state, command=None, result=None, elapsed_s=None)

    # base keeps its raw frame; the wrists fall back to the policy views.
    np.testing.assert_array_equal(state.load("base.png"), 7)
    np.testing.assert_array_equal(state.load("left_wrist.png"), 0)  # main_images
    np.testing.assert_array_equal(
        state.load("right_wrist.png"), 1
    )  # extra_view_images[1]


def test_view_env_state_emits_multimodal_image_blocks(tmp_path: Path):
    env = FakeEnv()
    ctx = _context(env)
    state = EnvState(tmp_path)

    dump_state(ctx.robot, state, command=None, result=None, elapsed_s=None)
    output = view_env_state.handler(ctx=_context(env, state=state))
    # The text must name the views in the same order the image blocks are emitted.
    assert output.data["images"] == ["left_wrist", "base", "right_wrist"]

    assert len(output.images) == 3
    assert "_image_" not in output.to_text()


def test_back_project_base_pixel_reads_rpent_state_artifacts(tmp_path: Path):
    state = EnvState(tmp_path)
    with state.record_step(
        state={
            "raw": {
                "left": {"tcp_pose": [0, 0, 0, 0, 0, 0, 1]},
                "right": {"tcp_pose": [0, 0, 0, 0, 0, 0, 1]},
            }
        }
    ) as step:
        state.save("base_depth.npy", np.full((4, 4), 0.5), step=step)
        state.save(
            "camera_meta.json",
            {
                "base_0_rgb": {
                    "color_intrinsics": {
                        "fx": 100,
                        "fy": 100,
                        "ppx": 2,
                        "ppy": 2,
                    }
                }
            },
            step=step,
        )

    set_calibration_path(
        Path(__file__).parent / "fixtures" / "hand_eye_calibration.json"
    )
    set_robot_config_path(DUAL_FRANKA_CONFIG)
    result = back_project_base_pixel(row=2, col=2, state=state)

    assert result["coordinate_frame"] == "right_base"
    assert result["depth_m"] == 0.5
    assert len(result["point_xyz"]) == 3


def test_load_calibration_bundle_follows_robot_config_override(tmp_path: Path):
    config = tmp_path / "robot_config.yaml"
    config.write_text(
        "perception:\n"
        "  localization_validity:\n"
        "    base_camera:\n"
        "      depth_m: [0.2, 0.9]\n"
        "  base_frames:\n"
        "    T_right_base_left_base:\n"
        "      matrix:\n"
        "        - [1.0, 0.0, 0.0, 0.02]\n"
        "        - [0.0, 1.0, 0.0, 0.7]\n"
        "        - [0.0, 0.0, 1.0, 0.0]\n"
        "        - [0.0, 0.0, 0.0, 1.0]\n"
    )
    set_calibration_path(
        Path(__file__).parent / "fixtures" / "hand_eye_calibration.json"
    )
    set_robot_config_path(config)
    try:
        bundle = load_calibration_bundle()
    finally:
        set_robot_config_path(None)

    assert bundle["base_camera"]["localization_validity"] == {"depth_m": [0.2, 0.9]}
    assert bundle["base_frames"]["T_right_base_left_base"]["matrix"][0][3] == 0.02
    # Hand-eye transforms from the calibration bundle survive the merge.
    assert "transformation" in bundle["d455_camera"]


def test_toolkit_factory_validation_capture_and_cancellation(tmp_path, monkeypatch):
    monkeypatch.setattr(franka_toolkit, "get_output_dir", lambda: tmp_path)
    env = FakeEnv()
    # RPC state contains NumPy arrays; native result text must still be JSON.
    get_robot_state = env.get_robot_state
    env.get_robot_state = lambda: {**get_robot_state(), "states": np.zeros(8)}
    events = []
    kwargs = {"env": env, "model": None, "task_description": "test task"}
    toolkit = DualFrankaToolkit(
        runtime_kwargs=kwargs,
        dashboard_events=SimpleNamespace(enabled=True, emit=events.append),
        memory=MemoryManager(root=tmp_path / "memory"),
    )
    assert kwargs == {"env": env, "model": None, "task_description": "test task"}
    assert env.observation_calls == 1
    assert len(events) == 1 and isinstance(events[0], StepRecordEvent)
    assert all(
        "ctx" not in tool.input_schema["properties"] for tool in toolkit.list_tools()
    )
    for delta in ([1, 2], [0, 0, float("inf")]):
        result = toolkit.execute_tool("move_delta", {"arm": "left", "delta_xyz": delta})
        assert result.is_error
    assert env.moves == []
    assert env.observation_calls == 1
    assert len(events) == 1
    result = toolkit.execute_tool(
        "move_delta", {"arm": "left", "delta_xyz": [0.01, 0, 0]}
    )
    assert not result.is_error
    assert result.data["images"] == ["left_wrist", "base", "right_wrist"]
    assert len(result.images) == 3
    assert "states" in result.to_text()
    assert len(events) == 2
    assert events[-1].record.command["action"] == "move_delta"
    read = toolkit.execute_tool("view_env_state", {})
    np.testing.assert_equal(
        read.data,
        {key: value for key, value in result.data.items() if key != "agent_elapsed_s"},
    )
    assert read.images == result.images
    assert len(events) == 2
    toolkit.cancel_active_and_wait()
    assert not toolkit.execute_tool("view_env_state", {}).is_error
    assert "finish" in {tool.name for tool in toolkit.list_tools()}
    assert toolkit.finish_result is None
    assert len(events) == 2
    toolkit.close()


# Historical PR #172 schemas and return fields; do not regenerate from native tools.
_CONTRACT = json.loads(
    (Path(__file__).parent / "fixtures/pre_native_tool_contracts.json").read_text()
)


def _schema_contract(value):
    """Compare main parameters, allowing descriptions, defaults and nullability to differ."""
    if isinstance(value, dict):
        result = {
            key: _schema_contract(item)
            for key, item in value.items()
            if key not in {"default", "description"}
            and not (key == "additionalProperties" and item is True)
        }
        if isinstance(result.get("type"), list):
            types = [kind for kind in result["type"] if kind != "null"]
            result["type"] = types[0] if len(types) == 1 else types
        return result
    if isinstance(value, list):
        return [_schema_contract(item) for item in value]
    return value


def test_main_parameter_contract():
    actual = {item.name: item.input_schema for item in tools.DUAL_FRANKA_TOOLS}
    assert len(actual) == len(tools.DUAL_FRANKA_TOOLS)
    assert _schema_contract(actual) == _schema_contract(_CONTRACT["schemas"])


def _prepare_contract_perception(state):
    metadata = {
        name: {"color_intrinsics": {"fx": 100, "fy": 100, "ppx": 2, "ppy": 2}}
        for name in ("base_0_rgb", "d455_rgb")
    }
    step = state.get().step_idx
    state.save("camera_meta.json", metadata, step=step)
    for name in ("base", "d455"):
        state.save(
            f"{name}_depth.npy", np.full((4, 4), 0.5, dtype=np.float32), step=step
        )


@pytest.mark.parametrize("case", _CONTRACT["cases"], ids=lambda case: case["name"])
def test_normal_return_contract(case, tmp_path, monkeypatch):
    monkeypatch.setattr(franka_toolkit, "get_output_dir", lambda: tmp_path)
    toolkit = DualFrankaToolkit(
        runtime_kwargs={
            "env": FakeEnv(),
            "model": FakeModel(),
            "task_description": "default task",
        },
        dashboard_events=NullDashboardEventSink(),
        memory=MemoryManager(tmp_path / "memory"),
    )
    name = case["name"]
    if name.startswith("back_project") or name == "view_perception_setup":
        _prepare_contract_perception(toolkit.state)
        monkeypatch.setattr(
            runtime_config,
            "_calibration_path",
            Path(__file__).parent / "fixtures/hand_eye_calibration.json",
        )
        monkeypatch.setattr(runtime_config, "_robot_config_path", DUAL_FRANKA_CONFIG)
    result = toolkit.execute_tool(name, case["arguments"])
    assert not result.is_error
    data = result.to_dict()
    assert sorted(data) == case["fields"]
    assert len(result.images) == case["image_count"]
    if "result_fields" in case:
        assert sorted(data["result"]) == case["result_fields"]
    if "last_chunk_fields" in case:
        assert sorted(data["result"]["last_chunk"]) == case["last_chunk_fields"]
    if name == "finish":
        assert data == {"_finish": True, **case["arguments"]}
        assert toolkit.finish_result == case["arguments"]
