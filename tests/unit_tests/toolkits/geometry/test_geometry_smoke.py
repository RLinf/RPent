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

"""Offline smoke checks for geometry imports, delegation and part fitting."""

import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from robots.libero.tools import LiberoPrimitives, write_recipe_from_states
from rpent.robots.components.molmo_client import MolmoResult
from rpent.robots.components.sam3_client import Sam3Result
from rpent.session import EnvState
from toolkits.geometry.geometry_agent import GeometryAgent


def _save_state(tmp_path, world):
    state = EnvState(tmp_path / "episode")
    with state.record_step(state={}):
        state.save("agentview.png", np.zeros(world.shape, dtype=np.uint8))
        state.save("agentview_world.npz", world.astype(np.float32))
        state.save("agentview_depth.npz", np.ones(world.shape[:2], dtype=np.float32))
    return state


def test_imports_and_cli_smoke(tmp_path):
    root = Path(__file__).resolve().parents[4]
    code = """
import sys
from importlib import import_module
for name in (
    "geometry", "candidate_geometry", "reference_geometry",
    "evaluation.evaluation_scene",
):
    import_module("toolkits.geometry." + name)
from toolkits.geometry.geometry_agent import GeometryAgent
from toolkits.geometry.evaluation.experiment import main
assert GeometryAgent.geometry.name == "geometry"
assert GeometryAgent.geometry.readonly
assert main(["validate-manifest"]) == 0
assert not {"torch", "rlinf", "mujoco", "robosuite", "sam3"}.intersection(sys.modules)
"""
    for args in (
        ["-c", code],
        ["-m", "toolkits.geometry.evaluation.experiment", "--help"],
    ):
        result = subprocess.run(
            [sys.executable, *args],
            cwd=tmp_path,
            env={**os.environ, "PYTHONPATH": str(root)},
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode == 0, result.stdout + result.stderr


def test_parent_geometry_and_recipe_smoke(tmp_path, monkeypatch):
    from pydantic_ai.models.test import TestModel

    import rpent.planner.base as base
    from robots.libero.toolkit import LiberoToolkit
    from rpent.dashboard.events import NullDashboardEventSink
    from rpent.memory import MemoryManager
    from rpent.utils import templates

    rows, cols = np.indices((6, 8))
    world = np.stack((cols * 0.01, rows * 0.02, 0.8 + rows * 0.01), axis=-1)
    state = _save_state(tmp_path, world)

    def init(self, **kwargs):
        self._state = state
        self._primitives = LiberoPrimitives(
            env=object(),
            model=object(),
            sam3_client=None,
            check_cancelled=self.raise_if_cancelled,
        )

    monkeypatch.setattr(LiberoToolkit, "init_primitives", init)
    monkeypatch.setattr(
        templates, "default_variables", lambda: {"output_dir": str(tmp_path / "parent")}
    )
    monkeypatch.setattr(
        base,
        "build_api_model",
        lambda *a: TestModel(custom_output_args={"point_xy": [5, 3]}),
    )
    parent = LiberoToolkit(
        runtime_kwargs={},
        dashboard_events=NullDashboardEventSink(),
        memory=MemoryManager(tmp_path / "memory"),
        geometry_model="test:geometry",
        state_output_dir=tmp_path / "parent",
    )
    data = parent.execute_tool("geometry", {"query": "cup handle", "step": 0}).data
    assert data["status"] == "usable", data
    assert np.asarray(data["world_xyz"]).shape == (3,)
    assert np.isfinite(data["world_xyz"]).all()
    assert state.load(data["geometry_artifact"], step=0)["status"] == "usable"
    assert len(state.records()) == 1

    with state.record_step(state={}, terminated=True):
        pass
    name = write_recipe_from_states(state, "geometry", output_dir=tmp_path)
    commands = [json.loads(line) for line in (tmp_path / name).read_text().splitlines()]
    assert len(commands) == 1
    assert commands[0]["action"] == "geometry"
    assert commands[0]["query"] == "cup handle"


@pytest.mark.parametrize(
    "part,quantity",
    [
        ("rim", "rim_center"),
        ("bottom", "bottom_center"),
        ("plane", "plane_normal"),
        ("body", "axis"),
    ],
)
def test_part_geometry_smoke(tmp_path, part, quantity):
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
        else:
            mask = (np.abs(x) < 0.04) & (np.abs(y) < 0.04)
        pixel = (104, 64) if part == "rim" else (64, 64)
    state = _save_state(tmp_path, world)
    primitives = LiberoPrimitives(
        env=None,
        model=None,
        sam3_client=SimpleNamespace(
            segment=lambda *a, **kw: Sam3Result(found=True, mask=mask, score=0.95)
        ),
        check_cancelled=lambda: None,
    )
    agent = GeometryAgent(
        state=state,
        segment=primitives.segment,
        check_cancelled=lambda: None,
        grounder=SimpleNamespace(
            ground=lambda *a: MolmoResult(
                found=True, point_xy=pixel, image_size=(mask.shape[1], mask.shape[0])
            )
        ),
    )
    data = agent.geometry(
        "cup" if part != "plane" else "microwave top",
        step=0,
        required_geometry=[quantity],
    ).data
    assert data["status"] == "usable", data
    value = data["geometry"][quantity]["value"]
    vector = value["direction"] if part == "body" else value
    assert np.asarray(vector).shape == (3,)
    assert np.isfinite(vector).all()
    assert state.exists(data["parts"][part]["cleaned_points_artifact"], step=0)
    assert state.exists(data["geometry_artifact"], step=0)
