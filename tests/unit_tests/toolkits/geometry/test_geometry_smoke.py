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

"""Offline execution checks for the generic geometry tool and observation contract."""

import os
import subprocess
import sys
from functools import partial
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from rpent.robots.components.molmo_client import MolmoResult
from rpent.robots.components.sam3_client import Sam3Result
from rpent.session import EnvState
from toolkits.geometry.geometry_agent import GeometryAgent
from toolkits.geometry.observation import GeometryObservation


def _inputs(tmp_path, world, **kwargs):
    state = EnvState(tmp_path / "episode")
    with state.record_step(state={}):
        pass
    frame = GeometryObservation(
        camera="side/color",
        step=0,
        world_frame_id="workcell/base",
        rgb=np.zeros(world.shape, dtype=np.uint8),
        world_xyz=world,
        **kwargs,
    )
    calls = []

    def get_observation(**request):
        calls.append(request)
        return frame

    return state, frame, SimpleNamespace(get_observation=get_observation), calls


def _grounder(pixel, shape):
    return SimpleNamespace(
        ground=lambda *args: MolmoResult(
            found=True, point_xy=pixel, image_size=(shape[1], shape[0])
        )
    )


def test_imports_without_simulator_or_model_stacks(tmp_path):
    root = Path(__file__).resolve().parents[4]
    code = """
import sys
from importlib import import_module
for name in ("geometry", "part_geometry", "perception", "observation", "prompts"):
    import_module("toolkits.geometry." + name)
from toolkits.geometry.geometry_agent import GeometryAgent
assert GeometryAgent.geometry.name == "geometry"
assert GeometryAgent.geometry.readonly
assert not {"torch", "rlinf", "mujoco", "robosuite", "sam3"}.intersection(sys.modules)
assert not any(name.startswith("robots.") for name in sys.modules)
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=tmp_path,
        env={**os.environ, "PYTHONPATH": str(root)},
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_generic_tool_dispatch_and_model_adapter(tmp_path, monkeypatch):
    from pydantic_ai.models.test import TestModel

    import rpent.planner.base as base
    from rpent.dashboard.events import NullDashboardEventSink
    from rpent.memory.manager import MemoryManager
    from rpent.tools.toolkit import Toolkit
    from rpent.utils import templates

    rows, cols = np.indices((6, 8))
    world = np.stack((cols * 0.01, rows * 0.02, np.full(rows.shape, 12.0)), axis=-1)
    state, frame, source, calls = _inputs(tmp_path, world)
    monkeypatch.setattr(
        templates, "default_variables", lambda: {"output_dir": str(tmp_path)}
    )
    monkeypatch.setattr(
        base,
        "build_api_model",
        lambda *a: TestModel(custom_output_args={"point_xy": [5, 3]}),
    )
    agent = GeometryAgent(state=state, observations=source, model="test:geometry")
    parent = Toolkit(
        state=state,
        dashboard_events=NullDashboardEventSink(),
        memory=MemoryManager(root=tmp_path / "memory", memory_access="read_only"),
    )
    parent.add_tool(
        agent.geometry.with_handler(
            partial(agent.geometry, check_cancelled=parent.raise_if_cancelled)
        )
    )
    schema = parent._tools["geometry"].input_schema
    assert "enum" not in schema["properties"]["camera"]
    assert (
        "camera" in schema["required"] and "check_cancelled" not in schema["properties"]
    )
    data = parent.execute_tool(
        "geometry", {"query": "blue handle", "camera": frame.camera}
    ).data
    assert data["status"] == "usable", data
    np.testing.assert_allclose(data["world_xyz"], frame.world_xyz[3, 5])
    assert data["world_frame_id"] == "workcell/base"
    assert calls == [{"camera": frame.camera, "step": -1}]
    assert state.load(data["geometry_artifact"], step=0)["status"] == "usable"
    assert len(state.records()) == 1


def test_depth_projection_in_caller_coordinate_frame(tmp_path):
    world = np.zeros((3, 4, 3))
    state, _, _, _ = _inputs(tmp_path, world)
    pose = np.array(
        [[1, 0, 0, 0.1], [0, 0, 1, 0.2], [0, -1, 0, 0.3], [0, 0, 0, 1]], dtype=float
    )
    intrinsics = np.array([[100, 0, 1], [0, 100, 1], [0, 0, 1]], dtype=float)
    frame = GeometryObservation(
        camera="sensor-17",
        step=0,
        world_frame_id="fixture/y_up",
        rgb=np.zeros((3, 4, 3), dtype=np.uint8),
        depth_m=np.full((3, 4), 12.0),
        intrinsics=intrinsics,
        camera_to_world=pose,
        up_direction=np.array([0, 1, 0]),
    )
    source = SimpleNamespace(get_observation=lambda **kwargs: frame)
    agent = GeometryAgent(
        state=state, observations=source, grounder=_grounder((2, 1), (3, 4))
    )
    data = agent.geometry("marker", camera=frame.camera, step=0).data
    assert data["status"] == "usable", data
    np.testing.assert_allclose(data["world_xyz"], [0.22, 12.2, 0.3])
    assert data["world_frame_id"] == "fixture/y_up"


def test_snapshot_preserves_valid_origin_during_model_call(tmp_path):
    world = np.zeros((1, 1, 3))
    state, frame, source, calls = _inputs(tmp_path, world)

    def ground(*args):
        world[:] = 99
        return MolmoResult(found=True, point_xy=(0, 0), image_size=(1, 1))

    data = (
        GeometryAgent(
            state=state, observations=source, grounder=SimpleNamespace(ground=ground)
        )
        .geometry("origin marker", camera=frame.camera)
        .data
    )
    assert data["status"] == "usable", data
    assert data["world_xyz"] == [0, 0, 0]
    assert len(calls) == 1 and not frame.world_xyz.flags.writeable


@pytest.mark.parametrize("accept_second", [True, False])
def test_segmentation_point_fallback_has_bounded_calls(tmp_path, accept_second):
    state, frame, source, _ = _inputs(tmp_path, np.ones((6, 8, 3)))
    attempts = []

    def segment(image, **kwargs):
        attempts.append(kwargs)
        mask = np.ones((6, 8), dtype=bool)
        if len(attempts) == 1 or not accept_second:
            mask[3, 5] = False
        return Sam3Result(found=True, mask=mask, score=0.95)

    agent = GeometryAgent(
        state=state,
        observations=source,
        segmenter=SimpleNamespace(segment=segment),
        grounder=_grounder((5, 3), (6, 8)),
    )
    data = agent.geometry(
        "blue cup left of the red box",
        camera=frame.camera,
        measurement="visible_region",
    ).data
    assert len(attempts) == 2
    assert attempts[0]["text_prompt"] == "blue cup left of the red box"
    assert attempts[1]["point"] == [3, 5]
    assert data["calls"] == {"grounding": 1, "segment": 2, "back_project": 1}
    assert data["status"] == ("usable" if accept_second else "need_observation")
    assert len(state.records()) == 1


@pytest.mark.parametrize(
    "part,quantity",
    [
        ("rim", "rim_center"),
        ("bottom", "bottom_center"),
        ("plane", "plane_normal"),
        ("body", "axis"),
        ("inner_rim", "inner_rim_center"),
    ],
)
def test_part_geometry_smoke_in_rotated_frame(tmp_path, part, quantity):
    if part == "body":
        theta, z = np.meshgrid(
            np.linspace(0, np.radians(220), 32), np.linspace(0.8, 0.92, 20)
        )
        world = np.stack((0.04 * np.cos(theta), 0.04 * np.sin(theta), z), axis=-1)
        mask = np.ones(world.shape[:2], dtype=bool)
        pixel = (16, 10)
    else:
        rows, cols = np.indices((128, 128))
        x, y = (cols - 64) * 0.001, (rows - 64) * 0.001
        world = np.stack((x, y, np.full(x.shape, 0.8)), axis=-1)
        if part == "rim":
            mask = np.abs(np.hypot(x, y) - 0.04) < 0.0016
        elif part == "bottom":
            mask = np.hypot(x, y) <= 0.04
        elif part == "inner_rim":
            mask = (np.hypot(x, y) >= 0.032) & (np.hypot(x, y) <= 0.045)
        else:
            mask = (np.abs(x) < 0.04) & (np.abs(y) < 0.04)
        pixel = (
            (104, 64)
            if part == "rim"
            else (96, 64)
            if part == "inner_rim"
            else (64, 64)
        )
    rotation = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]])
    world = world @ rotation.T
    state, frame, source, calls = _inputs(
        tmp_path, world, up_direction=np.array([0, 1, 0])
    )
    segmenter = SimpleNamespace(
        segment=lambda *a, **kw: Sam3Result(found=True, mask=mask, score=0.95)
    )
    agent = GeometryAgent(
        state=state,
        observations=source,
        segmenter=segmenter,
        grounder=_grounder(pixel, mask.shape),
    )
    data = agent.geometry(
        "container beside the blue box",
        camera=frame.camera,
        step=0,
        required_geometry=[quantity],
    ).data
    assert data["status"] == "usable", data
    field = data["geometry"][quantity]
    value = field["value"]["direction"] if part == "body" else field["value"]
    assert np.asarray(value).shape == (3,) and np.isfinite(value).all()
    assert field["world_frame_id"] == frame.world_frame_id
    assert len(calls) == 1 and len(state.records()) == 1
    assert state.exists(data["parts"][part]["cleaned_points_artifact"], step=0)


def test_request_cancellation_stops_before_coordinate_read(tmp_path, monkeypatch):
    from rpent.tools.toolkit import ToolCancelled
    from toolkits.geometry import perception

    state, frame, source, _ = _inputs(tmp_path, np.ones((6, 8, 3)))
    cancelled = False

    def ground(*args):
        nonlocal cancelled
        cancelled = True
        return MolmoResult(found=True, point_xy=(5, 3), image_size=(8, 6))

    def check():
        if cancelled:
            raise ToolCancelled("requested stop")

    def unexpected(*args):
        pytest.fail("cancelled request must stop before coordinate reads")

    monkeypatch.setattr(perception, "back_project", unexpected)
    agent = GeometryAgent(
        state=state, observations=source, grounder=SimpleNamespace(ground=ground)
    )
    with pytest.raises(ToolCancelled, match="requested stop"):
        agent.geometry("target", camera=frame.camera, check_cancelled=check)


def test_observation_validation_and_source_identity(tmp_path):
    state, frame, source, _ = _inputs(tmp_path, np.ones((6, 8, 3)))
    data = (
        GeometryAgent(
            state=state, observations=source, grounder=_grounder((5, 3), (6, 8))
        )
        .geometry("target", camera="other")
        .data
    )
    assert data["status"] == "error" and data["code"] == "observation_unavailable"
    with pytest.raises(ValueError, match="world_xyz must match RGB"):
        GeometryObservation(
            camera="camera",
            step=0,
            world_frame_id="base",
            rgb=frame.rgb,
            world_xyz=np.ones((3, 4, 3)),
        )


def test_cleaning_uses_explicit_up_direction():
    from toolkits.geometry.geometry import clean_part_cloud

    x, z = np.meshgrid(np.linspace(-0.05, 0.05, 16), np.linspace(-0.05, 0.05, 16))
    context = np.column_stack((x.ravel(), np.zeros(x.size), z.ravel()))
    theta, y = np.meshgrid(np.linspace(0, 2 * np.pi, 32), np.linspace(0.02, 0.08, 12))
    body = np.column_stack(
        (0.02 * np.cos(theta).ravel(), y.ravel(), 0.02 * np.sin(theta).ravel())
    )
    points = np.vstack((context, body))
    args = {
        "part": "body",
        "anchor": np.array([0, 0.05, 0.02]),
        "context_points": context,
        "require_anchor_connection": False,
    }
    supplied = clean_part_cloud(points, up_direction=np.array([0, 1, 0]), **args)
    unspecified = clean_part_cloud(points, **args)
    assert "background_plane" in supplied.quality
    assert "background_plane" not in unspecified.quality
    assert np.isfinite(supplied.points).all()
