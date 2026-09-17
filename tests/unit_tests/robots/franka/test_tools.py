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

"""Offline Franka primitive and state-capture tests."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from threading import Event
from types import SimpleNamespace

import numpy as np
import pytest

from robots.franka import runtime_config, tools
from robots.franka import toolkit as franka_toolkit
from robots.franka.perception import back_project
from robots.franka.runtime_config import set_calibration_path
from robots.franka.toolkit import FrankaRuntime, FrankaToolkit
from robots.franka.tools import (
    dump_state,
    view_camera_meta,
    view_env_state,
)
from rpent.dashboard.events import NullDashboardEventSink, StepRecordEvent
from rpent.memory import MemoryManager
from rpent.session import EnvState
from rpent.tools import ToolContext
from tests.unit_tests.robots.franka._fakes import FakeEnv, FakeModel


def _context(env: FakeEnv, *, model=None, state=None):
    return ToolContext(
        robot=FrankaRuntime(env=env, model=model, task_description="default task"),
        state=state,
        memory=None,
        output_dir=Path("."),
        record_frame=lambda frame: None,
        _cancel_event=Event(),
    )


def test_vec3_validation_and_motion_forwarding():
    env = FakeEnv()
    ctx = _context(env)

    tools.move_delta.handler([0.01, 0.0, -0.02], ctx=ctx)
    tools.rotate_delta.handler([0.0, 0.0, 0.1], ctx=ctx)

    np.testing.assert_allclose(env.moves[0], [0.01, 0.0, -0.02])
    np.testing.assert_allclose(env.rotations[0], [0.0, 0.0, 0.1])


def test_dump_state_saves_canonical_rgbd_artifacts(tmp_path: Path):
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
        "camera.png",
        "camera_depth.npy",
        "camera_meta.json",
        "wrist.png",
        "wrist_depth.npy",
    }
    output = view_env_state.handler(ctx=_context(env, state=state))
    assert output.images == [
        state.load_bytes("camera.png"),
        state.load_bytes("wrist.png"),
    ]
    assert "images" not in output.data
    assert output.data["image_cam_path"] == str(state.artifact_path("camera.png"))
    assert output.data["image_wrist_path"] == str(state.artifact_path("wrist.png"))
    assert (
        view_camera_meta.handler(ctx=_context(env, state=state)).data["camera_meta"][
            "depth_unit"
        ]
        == "m"
    )


def test_vla_grasp_runs_bounded_chunks():
    env = FakeEnv()
    ctx = _context(env, model=FakeModel())

    result = tools.vla_grasp.handler("pick up the cube", max_chunks=3, ctx=ctx)

    assert result.data["chunks_executed"] == 3
    assert len(env.chunks) == 3
    # Obs is fetched once, then threaded from each chunk_step result.
    assert env.observation_calls == 1


def test_back_project_reads_rpent_state_artifacts(tmp_path: Path):
    state = EnvState(tmp_path)
    with state.record_step(
        state={"raw_base_state": {"tcp_pose": [0, 0, 0, 0, 0, 0, 1]}}
    ) as step:
        state.save("wrist_depth.npy", np.full((4, 4), 0.5), step=step)
        state.save(
            "camera_meta.json",
            {
                "observation_camera_map": {"main": "wrist_cam"},
                "cameras": {
                    "wrist_cam": {"intrinsic_K": [[100, 0, 2], [0, 100, 2], [0, 0, 1]]}
                },
            },
            step=step,
        )

    set_calibration_path(
        Path(__file__).parent / "fixtures" / "hand_eye_calibration.json"
    )
    result = back_project(row=2, col=2, camera="wrist", state=state)

    assert result["coordinate_frame"] == "franka_base"
    assert result["depth_m"] == 0.5
    assert len(result["point_base"]) == 3


def test_toolkit_factory_validation_and_capture(tmp_path, monkeypatch):
    monkeypatch.setattr(franka_toolkit, "get_output_dir", lambda: tmp_path)
    env = FakeEnv()
    # RPC state contains NumPy arrays; native result text must still be JSON.
    get_robot_state = env.get_robot_state
    env.get_robot_state = lambda: {**get_robot_state(), "states": np.zeros(8)}
    events = []
    kwargs = {"env": env, "model": None, "task_description": "test task"}
    toolkit = FrankaToolkit(
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
        result = toolkit.execute_tool("move_delta", {"delta_xyz": delta})
        assert result.is_error
    assert env.moves == []
    assert env.observation_calls == 1
    assert len(events) == 1
    result = toolkit.execute_tool("move_delta", {"delta_xyz": [0.01, 0, 0]})
    assert not result.is_error
    assert "images" not in result.data
    assert "image_cam_path" in result.data and "image_wrist_path" in result.data
    assert len(result.images) == 2
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
    assert "finish" in {tool.name for tool in toolkit.list_tools()}
    assert toolkit.finish_result is None
    toolkit.close()


# Historical PR #172 schemas and return fields; do not regenerate from native tools.
_CONTRACT = json.loads(
    (Path(__file__).parent / "fixtures/pre_native_tool_contracts.json").read_text()
)


def test_tool_schema_and_description_contract():
    expected = copy.deepcopy(_CONTRACT["schemas"])
    for schema in expected.values():
        schema["additionalProperties"] = False
    # These existing optional inputs now explicitly advertise their None value.
    nullable = {
        "back_project": ("step",),
        "back_project_correspondence": (
            "third_person_row",
            "third_person_col",
            "wrist_row",
            "wrist_col",
            "pixels",
            "step",
        ),
    }
    for name, parameters in nullable.items():
        for parameter in parameters:
            schema = expected[name]["properties"][parameter]
            schema["type"] = [schema["type"], "null"]
    actual = {item.name: item.input_schema for item in tools.FRANKA_TOOLS}
    assert len(actual) == len(tools.FRANKA_TOOLS)
    assert actual == expected
    assert {item.name: item.description for item in tools.FRANKA_TOOLS} == _CONTRACT[
        "descriptions"
    ]
    for item in tools.FRANKA_TOOLS:
        for name, parameter in actual[item.name]["properties"].items():
            if "default" in parameter:
                assert (
                    item.args_schema.model_fields[name].default == parameter["default"]
                )


def _prepare_contract_perception(state):
    metadata = {
        "observation_camera_map": {"main": "wrist_cam", "extra_0": "external_cam"},
        "cameras": {
            name: {"intrinsic_K": [[100, 0, 2], [0, 100, 2], [0, 0, 1]]}
            for name in ("wrist_cam", "external_cam")
        },
    }
    images = {name: state.load(f"{name}.png") for name in ("wrist", "camera")}
    with state.record_step(state={"raw_base_state": state.get().state}) as step:
        for name, image in images.items():
            state.save(f"{name}.png", image, step=step)
            state.save(
                f"{name}_depth.npy", np.full((4, 4), 0.5, dtype=np.float32), step=step
            )
        state.save("camera_meta.json", metadata, step=step)


@pytest.mark.parametrize("case", _CONTRACT["cases"], ids=lambda case: case["name"])
def test_normal_return_contract(case, tmp_path, monkeypatch):
    monkeypatch.setattr(franka_toolkit, "get_output_dir", lambda: tmp_path)
    toolkit = FrankaToolkit(
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
