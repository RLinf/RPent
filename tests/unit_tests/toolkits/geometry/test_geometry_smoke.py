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

from rpent.robots.components.molmo_client import MolmoClient, MolmoResult
from rpent.robots.components.sam3_client import Sam3Result
from rpent.session import EnvState
from toolkits.geometry.geometry_tool import GeometryTool
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
from toolkits.geometry.geometry_tool import GeometryTool
assert GeometryTool.geometry.name == "geometry"
assert GeometryTool.geometry.readonly
assert not {"torch", "rlinf", "mujoco", "robosuite", "sam3"}.intersection(sys.modules)
assert "rpent.robots.components.grounding_agent" not in sys.modules
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


def test_generic_tool_dispatch_with_injected_molmo(tmp_path, monkeypatch):
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
    rpc_calls = []

    def call(method, *, kwargs, timeout_s):
        rpc_calls.append((method, kwargs))
        return {"point_xy": [5, 3], "image_size": [8, 6]}

    grounder = MolmoClient(SimpleNamespace(call=call), point_mode="geometry")
    geometry_tool = GeometryTool(
        state=state, observations=source, grounder=grounder, model="Molmo2-8B"
    )
    parent = Toolkit(
        state=state,
        dashboard_events=NullDashboardEventSink(),
        memory=MemoryManager(root=tmp_path / "memory", memory_access="read_only"),
    )
    parent.add_tool(
        geometry_tool.geometry.with_handler(
            partial(geometry_tool.geometry, check_cancelled=parent.raise_if_cancelled)
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
    assert data["model"] == "Molmo2-8B"
    assert rpc_calls[0][0] == "molmo.ground"
    assert rpc_calls[0][1]["point_mode"] == "geometry"
    assert rpc_calls[0][1]["query"] == "blue handle"
    assert calls == [{"camera": frame.camera, "step": -1}]
    assert state.load(data["geometry_artifact"], step=0)["status"] == "usable"
    assert len(state.records()) == 1


@pytest.mark.parametrize("image_size", [None, [8, 6]])
def test_molmo_miss_requests_observation_without_depth_or_segmentation(
    tmp_path, image_size
):
    state, frame, source, _ = _inputs(tmp_path, np.ones((6, 8, 3)))
    payload = {"answer": "not found"}
    if image_size is not None:
        payload["image_size"] = image_size
    grounder = MolmoClient(
        SimpleNamespace(call=lambda *a, **kw: payload), point_mode="geometry"
    )
    data = (
        GeometryTool(state=state, observations=source, grounder=grounder)
        .geometry("the hidden handle of the blue cup", camera=frame.camera)
        .data
    )
    assert data["status"] == "need_observation", data
    assert data["reason"] == "target_not_identified_in_source_image"
    assert data["world_xyz"] is None
    assert data["calls"] == {"grounding": 1, "segment": 0, "back_project": 0}


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
    geometry_tool = GeometryTool(
        state=state, observations=source, grounder=_grounder((2, 1), (3, 4))
    )
    data = geometry_tool.geometry("marker", camera=frame.camera, step=0).data
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
        GeometryTool(
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

    geometry_tool = GeometryTool(
        state=state,
        observations=source,
        segmenter=SimpleNamespace(segment=segment),
        grounder=_grounder((5, 3), (6, 8)),
    )
    data = geometry_tool.geometry(
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
    geometry_tool = GeometryTool(
        state=state,
        observations=source,
        segmenter=segmenter,
        grounder=_grounder(pixel, mask.shape),
    )
    data = geometry_tool.geometry(
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
    geometry_tool = GeometryTool(
        state=state, observations=source, grounder=SimpleNamespace(ground=ground)
    )
    with pytest.raises(ToolCancelled, match="requested stop"):
        geometry_tool.geometry("target", camera=frame.camera, check_cancelled=check)


@pytest.mark.parametrize("include_height", [False, True])
def test_unaccepted_circle_returns_estimate_with_warning(tmp_path, include_height):
    theta = np.broadcast_to(np.linspace(0, np.radians(120), 64), (8, 64))
    world = np.stack(
        (0.04 * np.cos(theta), 0.04 * np.sin(theta), np.full(theta.shape, 0.8)), axis=-1
    )
    state, frame, source, _ = _inputs(tmp_path, world)
    mask = np.ones(theta.shape, dtype=bool)

    def ground(image, query):
        found = query.startswith("the visible circular rim")
        return MolmoResult(
            found=found, point_xy=(32, 4) if found else None, image_size=(64, 8)
        )

    geometry_tool = GeometryTool(
        state=state,
        observations=source,
        grounder=SimpleNamespace(ground=ground),
        segmenter=SimpleNamespace(
            segment=lambda *a, **kw: Sam3Result(found=True, mask=mask, score=0.95)
        ),
    )
    requested = ["rim_center", "height"] if include_height else ["rim_center"]
    data = geometry_tool.geometry(
        "blue cup", camera=frame.camera, required_geometry=requested
    ).data
    field = data["geometry"]["rim_center"]
    assert data["status"] == ("partial" if include_height else "warning"), data
    np.testing.assert_allclose(field["value"], [0, 0, 0.8], atol=1e-5)
    assert field["fit_accepted"] is False
    assert "insufficient_circle_angular_coverage" in field["warnings"]
    assert {
        "quantity": "rim_center",
        "reason": "insufficient_circle_angular_coverage",
    } in data["warnings"]
    saved = state.load(data["geometry_artifact"], step=0)
    assert saved["geometry"]["rim_center"]["value"] == field["value"]
    assert saved["parts"]["rim"]["fit"]["parameters"]["center"] == field["value"]
    if include_height:
        assert data["geometry"]["height"]["value"] is None
        assert data["observation_request"]["missing_geometry"] == ["height"]
    else:
        assert "observation_request" not in data
    assert len(state.records()) == 1


def test_uncomputable_geometry_stays_unknown():
    from toolkits.geometry.geometry import fit_circle, fit_plane, geometry_fields

    circle = fit_circle(np.array([[0, 0, 0], [0.1, 0, 0]]))
    plane = fit_plane(np.column_stack((np.linspace(0, 0.1, 32), np.zeros((32, 2)))))
    fields, _ = geometry_fields(
        {"rim": circle, "plane": plane}, ["rim_center", "plane_normal"]
    )
    for field in fields.values():
        assert field["value"] is None and field["fit_accepted"] is False
        assert field["warnings"] == [] and field["reason"]


def test_circle_anchor_mismatch_keeps_unanchored_estimate():
    from toolkits.geometry.geometry import fit_circle, geometry_fields

    theta = np.linspace(0, 2 * np.pi, 180, endpoint=False)
    points = np.column_stack(
        (0.04 * np.cos(theta), 0.04 * np.sin(theta), np.full(theta.shape, 0.8))
    )
    matched = fit_circle(points, anchor=points[0])
    assert matched.accepted and not matched.warnings
    unmatched = fit_circle(points, anchor=np.array([0, 0, 1.0]))
    fields, _ = geometry_fields({"rim": unmatched}, ["rim_center"])
    field = fields["rim_center"]
    np.testing.assert_allclose(field["value"], matched.parameters["center"], atol=1e-6)
    assert field["fit_accepted"] is False
    assert "circle_anchor_constraint_relaxed" in field["warnings"]
    assert "circle_inconsistent_with_part_anchor" in field["warnings"]
    assert unmatched.quality["anchor_constraint_relaxed"] is True
    assert unmatched.quality["anchor_error_m"] > unmatched.quality["anchor_threshold_m"]


@pytest.mark.parametrize("quantity", ["plate_center", "rectangle_center"])
def test_face_contour_returns_warning_when_physical_boundary_is_missing(
    tmp_path, quantity
):
    rows, cols = np.indices((128, 128))
    x, y = (cols - 64) * 0.001, (rows - 64) * 0.001
    world = np.stack((x, y, np.full(x.shape, 0.8)), axis=-1)
    mask = (
        np.hypot(x, y) <= 0.04
        if quantity == "plate_center"
        else (np.abs(x) <= 0.04) & (np.abs(y) <= 0.03)
    )
    state, frame, source, _ = _inputs(tmp_path, world)
    geometry_tool = GeometryTool(
        state=state,
        observations=source,
        grounder=_grounder((64, 64), mask.shape),
        segmenter=SimpleNamespace(
            segment=lambda *a, **kw: Sam3Result(found=True, mask=mask, score=0.95)
        ),
    )
    data = geometry_tool.geometry(
        "box or plate on the table", camera=frame.camera, required_geometry=[quantity]
    ).data
    assert data["status"] == "warning", data
    field = data["geometry"][quantity]
    np.testing.assert_allclose(field["value"], [0, 0, 0.8], atol=0.001)
    assert field["fit_accepted"] is False
    assert "physical_boundary_not_verified" in field["warnings"]
    sample = next(iter(data["parts"].values()))
    extraction = sample["part_extraction"]
    assert extraction["n_supported_boundary_pixels"] == 0
    assert extraction["n_fallback_boundary_pixels"] > 0
    assert extraction["boundary_source"] == "unverified_segmented_face_contour"
    assert state.load(sample["fit_points_artifact"], step=0).shape[0] > 0
    assert "observation_request" not in data


def test_rectangle_and_body_keep_computed_parameters_when_checks_fail():
    from toolkits.geometry.geometry import FitOptions, fit_axisymmetric
    from toolkits.geometry.part_geometry import fit_rectangle, part_geometry_fields

    x, y = np.meshgrid(np.linspace(-0.04, 0.04, 20), np.linspace(-0.03, 0.03, 16))
    points = np.column_stack((x.ravel(), y.ravel(), np.full(x.size, 0.8)))
    # Three physical sides cannot pass the four-edge requirement.
    edges = points[(np.abs(points[:, 0]) > 0.039) | (points[:, 1] < -0.029)]
    rectangle = fit_rectangle(points, edges)
    field = part_geometry_fields({"rectangular_top": rectangle}, ["rectangle_center"])[
        "rectangle_center"
    ]
    assert field["value"] is not None and field["fit_accepted"] is False
    assert "four_complete_physical_edges_not_observed" in field["warnings"]
    theta, z = np.meshgrid(
        np.linspace(0, np.radians(220), 32), np.linspace(0.8, 0.92, 20)
    )
    body = fit_axisymmetric(
        np.column_stack(
            (0.04 * np.cos(theta).ravel(), 0.04 * np.sin(theta).ravel(), z.ravel())
        ),
        FitOptions(min_body_coverage_deg=360),
    )
    assert body.parameters and not body.accepted
    assert "insufficient_body_angular_coverage" in body.warnings


def test_inconsistent_parts_return_warning_instead_of_hiding_coordinates():
    from toolkits.geometry.geometry import ShapeFit, geometry_fields

    rim = ShapeFit(
        "circle",
        accepted=True,
        parameters={"center": [1, 0, 1], "normal": [0, 0, 1], "radius": 0.04},
    )
    bottom = ShapeFit(
        "circle",
        accepted=True,
        parameters={"center": [0, 0, 0], "normal": [0, 0, 1], "radius": 0.04},
    )
    fields, checks = geometry_fields(
        {"rim": rim, "bottom": bottom}, ["height", "body_center"]
    )
    assert checks and not all(check["consistent"] for check in checks)
    np.testing.assert_allclose(fields["body_center"]["value"], [0.5, 0, 0.5])
    assert fields["height"]["value"] == pytest.approx(np.sqrt(2))
    assert all(
        not field["fit_accepted"] and "inconsistent_part_fits" in field["warnings"]
        for field in fields.values()
    )


def test_observation_validation_and_source_identity(tmp_path):
    state, frame, source, _ = _inputs(tmp_path, np.ones((6, 8, 3)))
    data = (
        GeometryTool(
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


@pytest.mark.parametrize("angle", [0.0, 1.1])
def test_plane_at_material_anchor_wins_over_larger_side_face(angle):
    from toolkits.geometry.geometry import fit_plane

    x, y = np.meshgrid(np.linspace(-0.025, 0.025, 17), np.linspace(-0.025, 0.025, 17))
    face = np.column_stack((x.ravel(), y.ravel(), np.full(x.size, 0.03)))
    y, z = np.meshgrid(np.linspace(-0.04, 0.04, 32), np.linspace(-0.03, 0.03, 32))
    wall = np.column_stack((np.full(y.size, 0.04), y.ravel(), z.ravel()))
    rotation = np.array(
        [
            [np.cos(angle), 0, np.sin(angle)],
            [0, 1, 0],
            [-np.sin(angle), 0, np.cos(angle)],
        ]
    )
    translation = np.array([0.4, -0.2, 0.7])
    points = np.vstack((face, wall)) @ rotation.T + translation
    anchor = rotation @ np.array([0, 0, 0.03]) + translation
    dominant = fit_plane(points)
    selected = fit_plane(points, anchor=anchor)
    assert abs(np.asarray(dominant.parameters["normal"]) @ rotation[:, 0]) > 0.99
    normal = np.asarray(selected.parameters["normal"])
    assert abs(normal @ rotation[:, 2]) > 0.99
    assert abs((anchor - selected.parameters["point"]) @ normal) < 0.001
    assert "low_plane_inlier_ratio" in selected.warnings


def test_short_arc_refinement_does_not_extrapolate_a_metre_scale_circle():
    from toolkits.geometry.geometry import fit_circle

    angles = np.linspace(-0.02, 0.02, 100)
    rng = np.random.default_rng(12)
    points = np.column_stack(
        (
            2 * np.sin(angles),
            2 * (1 - np.cos(angles)) + rng.normal(0, 0.0001, len(angles)),
            np.zeros(len(angles)),
        )
    )
    fitted = fit_circle(points)
    assert fitted.parameters and not fitted.accepted
    assert np.isfinite(fitted.parameters["center"]).all()
    assert fitted.parameters["radius"] < 0.2
    assert np.linalg.norm(fitted.parameters["center"]) < 0.2
    assert "unstable_circle_fit" in fitted.warnings
    assert "insufficient_circle_angular_coverage" in fitted.warnings


def test_plane_keeps_estimate_when_no_plane_matches_the_anchor():
    from toolkits.geometry.geometry import fit_plane

    x, y = np.meshgrid(np.linspace(-0.03, 0.03, 12), np.linspace(-0.03, 0.03, 12))
    points = np.column_stack((x.ravel(), y.ravel(), np.zeros(x.size)))
    fitted = fit_plane(points, anchor=np.array([0, 0, 1.0]))
    assert fitted.parameters and not fitted.accepted
    np.testing.assert_allclose(fitted.parameters["point"][2], 0)
    assert "selected_plane_does_not_contain_visual_face_anchor" in fitted.warnings
