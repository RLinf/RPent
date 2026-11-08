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

"""Independent analytic scenes check the reference definitions, including refusals."""

from types import SimpleNamespace

import numpy as np
import pytest

from robots.libero.tools import LiberoPrimitives
from rpent.robots.components.molmo_client import MolmoResult
from rpent.robots.components.sam3_client import Sam3Result
from rpent.session import EnvState
from toolkits.geometry.geometry import FitOptions, fit_plane
from toolkits.geometry.geometry_agent import GeometryAgent
from toolkits.geometry.reference_geometry import fit_barrel_segment, fit_rectangle


def reference_agent(
    tmp_path, world, mask, pixel, *, camera=True, image=None, permissive=False
):
    state = EnvState(tmp_path / "episode")
    height, width = mask.shape
    with state.record_step(state={}):
        state.save(
            "agentview.png",
            np.zeros((*mask.shape, 3), dtype=np.uint8) if image is None else image,
        )
        state.save("agentview_world.npz", world.astype(np.float32))
        state.save("agentview_depth.npz", 2 - world[:, :, 2])
        if camera:
            transform = np.diag([1.0, -1.0, -1.0, 1.0])
            transform[2, 3] = 2
            state.save(
                "agentview_metadata.json", {"extrinsic_cam2world": transform.tolist()}
            )
    calls = []

    def ground(image, query):
        calls.append(("ground", query))
        return MolmoResult(found=True, point_xy=pixel, image_size=(width, height))

    def segment(image, **kwargs):
        calls.append(("segment", kwargs))
        return Sam3Result(found=True, mask=mask, score=0.96)

    primitive = LiberoPrimitives(
        env=None,
        model=None,
        sam3_client=SimpleNamespace(segment=segment),
        check_cancelled=lambda: None,
    )
    agent = GeometryAgent(
        state=state,
        segment=primitive.segment,
        check_cancelled=lambda: None,
        grounder=SimpleNamespace(ground=ground),
        permissive=permissive,
    )
    return state, agent, calls


def grid():
    rr, cc = np.indices((192, 192))
    x, y = (cc - 96) * 0.001, (rr - 96) * 0.001
    return x, y


def test_permissive_agent_returns_uncertified_face_center_without_requesting_view(
    tmp_path,
):
    x, y = grid()
    world = np.stack((x, y, np.full(x.shape, 0.81)), axis=-1)
    mask = np.hypot(x, y) <= 0.045
    _, strict, _ = reference_agent(tmp_path / "strict", world, mask, (96, 96))
    rejected = strict.localize("plate", required_geometry=["plate_center"]).data
    assert rejected["status"] == "need_observation"
    state, agent, _ = reference_agent(
        tmp_path / "candidate", world, mask, (96, 96), permissive=True
    )
    data = agent.localize("plate", required_geometry=["plate_center"]).data
    assert data["status"] == "usable", data
    assert "observation_request" not in data
    field = data["geometry"]["plate_center"]
    np.testing.assert_allclose(field["value"], [0, 0, 0.81], atol=0.002)
    assert field["strict_accepted"] is False
    assert field["evaluation_only"] is True
    assert field["estimate_method"] == "relaxed_planar_boundary_circle"
    part = data["parts"]["plate_surface"]
    assert part["fit"]["accepted"] is False
    assert part["fit"]["reason"] == "insufficient_finite_points"
    assert len(state.load(part["candidate_fit_points_artifact"], step=0)) >= 6


@pytest.mark.parametrize(
    "target",
    [
        "small ramekin",
        "black bowl",
        "gray and white textured mug",
        "yellow and white mug",
    ],
)
@pytest.mark.parametrize("ring_only", [True, False])
def test_inner_opening_center_stays_at_upper_edge_not_floor_or_outer_lip(
    tmp_path, target, ring_only
):
    x, y = grid()
    radius = np.hypot(x, y)
    z = np.where(radius <= 0.04, 0.74, np.where(radius < 0.048, 0.84, 0.70))
    world = np.stack((x, y, z), axis=-1)
    mask = (radius > 0.04) & (radius < 0.048) if ring_only else radius < 0.048
    state, agent, calls = reference_agent(tmp_path, world, mask, (137, 96))
    data = agent.localize(target, required_geometry=["inner_rim_center"]).data
    assert data["status"] == "usable", data
    np.testing.assert_allclose(
        data["geometry"]["inner_rim_center"]["value"], [0, 0, 0.84], atol=0.001
    )
    part = data["parts"]["inner_rim"]
    fit_points = state.load(part["fit_points_artifact"], step=0)
    assert fit_points[:, 2].min() > 0.83
    assert np.median(np.linalg.norm(fit_points[:, :2], axis=1)) < 0.042
    assert len(calls) == 2 and data["calls"]["segment"] == 1
    assert len(state.records()) == 1 and data["world_xyz"] is None


def test_outer_ring_without_resolved_inner_edge_is_not_renamed(tmp_path):
    x, y = grid()
    world = np.stack((x, y, np.full(x.shape, 0.84)), axis=-1)
    mask = np.hypot(x, y) < 0.048
    _, agent, _ = reference_agent(tmp_path, world, mask, (140, 96))
    data = agent.localize("mug", required_geometry=["inner_rim_center"]).data
    assert data["status"] == "need_observation"
    assert data["geometry"]["inner_rim_center"]["value"] is None
    assert "inner_edge_not_resolved" in data["parts"]["inner_rim"]["fit"]["reason"]


def test_inner_rgb_contour_supports_a_smooth_depth_transition(tmp_path):
    x, y = grid()
    r = np.hypot(x, y)
    z = 0.84 - 0.012 * np.maximum(0, 1 - (r / 0.04) ** 2)
    world = np.stack((x, y, np.where(r < 0.05, z, 0.7)), axis=-1)
    image = np.repeat(
        np.where(r < 0.04, 40, 220).astype(np.uint8)[..., None], 3, axis=2
    )
    _, agent, _ = reference_agent(tmp_path, world, r < 0.05, (137, 96), image=image)
    data = agent.localize("black bowl", required_geometry=["inner_rim_center"]).data
    assert data["status"] == "usable", data
    np.testing.assert_allclose(
        data["geometry"]["inner_rim_center"]["value"], [0, 0, 0.84], atol=0.001
    )
    assert (
        "interior_rgb_contour"
        in data["parts"]["inner_rim"]["candidate_extraction"]["method"]
    )


@pytest.mark.parametrize("quantity,tilt", [("plate_center", 0.0), ("lid_center", 0.35)])
def test_circular_face_center_uses_selected_face_height_even_with_raised_lip(
    tmp_path, quantity, tilt
):
    x, y = grid()
    radius = np.hypot(x, y)
    face = radius <= 0.045
    z = 0.81 + tilt * x
    # Projected disk is an ellipse for an actually tilted circle.
    face = np.hypot(x * np.sqrt(1 + tilt**2), y) <= 0.045
    rim = (~face) & (radius < 0.051)
    world = np.stack((x, y, np.where(face, z, np.where(rim, z + 0.014, 0.7))), axis=-1)
    state, agent, _ = reference_agent(tmp_path, world, face | rim, (96, 96))
    data = agent.localize(
        "plate" if quantity == "plate_center" else "soup can",
        required_geometry=[quantity],
    ).data
    assert data["status"] == "usable", data
    np.testing.assert_allclose(
        data["geometry"][quantity]["value"], [0, 0, 0.81], atol=0.0015
    )
    part = data["parts"]["plate_surface" if quantity == "plate_center" else "lid"]
    raw = state.load(part["observed_support_artifact"], step=0)
    assert np.max(np.abs(raw[:, 2] - 0.81 - tilt * raw[:, 0])) < 0.001


@pytest.mark.parametrize("target", ["cheese box", "microwave body"])
def test_rectangular_top_center_requires_physical_face_boundaries(tmp_path, target):
    x, y = grid()
    angle = 0.35
    u, v = np.cos(angle) * x + np.sin(angle) * y, -np.sin(angle) * x + np.cos(angle) * y
    mask = (np.abs(u) <= 0.060) & (np.abs(v) <= 0.032)
    world = np.stack((x, y, np.where(mask, 0.91 + 0.2 * x, 0.7)), axis=-1)
    _, agent, _ = reference_agent(tmp_path, world, mask, (96, 96))
    data = agent.localize(target, required_geometry=["rectangle_center"]).data
    assert data["status"] == "usable", data
    np.testing.assert_allclose(
        data["geometry"]["rectangle_center"]["value"], [0, 0, 0.91], atol=0.002
    )


@pytest.mark.parametrize(
    "cut", ["label_crop", "foreground_occlusion", "image_crop", "missing_camera"]
)
def test_incomplete_rectangular_face_never_uses_fragment_bbox(tmp_path, cut):
    x, y = grid()
    full = (np.abs(x) < 0.07) & (np.abs(y) < 0.04)
    mask = full & (x < 0.02)
    z = np.where(full, 0.9, 0.7)
    if cut == "foreground_occlusion":
        z[full & ~mask] = 1.05
    elif cut == "image_crop":
        mask = (x < 0.02) & (np.abs(y) < 0.04)
        z[mask] = 0.9
    world = np.stack((x, y, z), axis=-1)
    _, agent, _ = reference_agent(
        tmp_path, world, mask, (96, 96), camera=cut != "missing_camera"
    )
    data = agent.localize("microwave", required_geometry=["rectangle_center"]).data
    assert data["status"] == "need_observation", data
    assert data["geometry"]["rectangle_center"]["value"] is None


def test_rotated_rectangle_fit_is_tilt_aware_and_requires_all_four_edges():
    x, y = np.meshgrid(np.linspace(-0.08, 0.08, 45), np.linspace(-0.035, 0.035, 30))
    points = np.column_stack((x.ravel(), y.ravel(), np.zeros(x.size)))
    rotation = np.array([[0.8, 0, 0.6], [0, 1, 0], [-0.6, 0, 0.8]])
    points = points @ rotation.T + [0.1, -0.2, 0.9]
    boundary = ((np.abs(x) == 0.08) | (np.abs(y) == 0.035)).ravel()
    result = fit_rectangle(points, points[boundary])
    assert result.accepted, result
    np.testing.assert_allclose(result.parameters["center"], [0.1, -0.2, 0.9], atol=1e-5)
    incomplete = points[boundary & (x.ravel() < 0.079)]
    assert not fit_rectangle(points, incomplete).accepted


def barrel_cloud(axis):
    theta, t = np.meshgrid(
        np.linspace(0, np.radians(240), 60), np.linspace(-0.12, 0.2, 90)
    )
    radii = np.where(t <= 0.12, 0.036, np.maximum(0.015, 0.036 - (t - 0.12) * 0.7))
    points = np.stack(
        (radii * np.cos(theta), radii * np.sin(theta), t), axis=-1
    ).reshape(-1, 3)
    basis_x = np.cross(axis, [0, 1, 0])
    basis_x /= np.linalg.norm(basis_x)
    basis_y = np.cross(axis, basis_x)
    basis = np.column_stack((basis_x, basis_y, axis))
    return points @ basis.T + [0.3, -0.1, 0.9], basis


@pytest.mark.parametrize("axis", [np.array([1.0, 0.0, 0.0]), np.array([0.6, 0.0, 0.8])])
def test_bottle_barrel_center_excludes_neck_and_supports_horizontal_axis(axis):
    points, basis = barrel_cloud(axis)
    terminal = points[np.abs((points - [0.3, -0.1, 0.9]) @ axis + 0.12) < 1e-6]
    anchor = np.array([0.3, -0.1, 0.9]) + 0.036 * basis[:, 0]
    result = fit_barrel_segment(points, terminal, anchor)
    assert result.accepted, result
    np.testing.assert_allclose(
        result.parameters["center"], [0.3, -0.1, 0.9], atol=0.005
    )
    assert result.quality["endpoint_evidence"] == [
        "physical_terminal_edge",
        "radius_transition",
    ]


def test_bottle_visible_extent_alone_cannot_define_barrel_center():
    points, basis = barrel_cloud(np.array([1.0, 0.0, 0.0]))
    cropped = points[(points[:, 0] > 0.24) & (points[:, 0] < 0.38)]
    result = fit_barrel_segment(
        cropped, np.empty((0, 3)), np.array([0.3, -0.1, 0.9]) + 0.036 * basis[:, 0]
    )
    assert not result.accepted
    assert result.reason == "both_main_barrel_end_boundaries_not_observed"


@pytest.mark.parametrize("cropped", [False, True])
def test_bottle_localize_retains_neck_context_and_rejects_a_mask_crop(
    tmp_path, cropped
):
    rr, cc = np.indices((100, 160))
    t = (cc - 20) / 119 * 0.32 - 0.12
    theta = np.radians((rr - 20) / 59 * 160 - 80)
    radius = np.where(t <= 0.12, 0.036, np.maximum(0.015, 0.036 - (t - 0.12) * 0.7))
    full = (rr >= 20) & (rr < 80) & (cc >= 20) & (cc < 140)
    world = np.stack(
        (
            0.3 + t,
            radius * np.sin(theta),
            np.where(full, 0.84 + radius * np.cos(theta), 0.68),
        ),
        axis=-1,
    )
    mask = full & (t > -0.05) & (t < 0.06) if cropped else full
    state, agent, calls = reference_agent(tmp_path, world, mask, (65, 50))
    data = agent.localize("wine bottle", required_geometry=["body_segment_center"]).data
    assert "main cylindrical barrel" in calls[0][1]
    assert "shoulder, neck and base" in calls[1][1]["text_prompt"]
    if cropped:
        assert data["status"] == "need_observation", data
        assert data["geometry"]["body_segment_center"]["value"] is None
    else:
        assert data["status"] == "usable", data
        np.testing.assert_allclose(
            data["geometry"]["body_segment_center"]["value"], [0.3, 0, 0.84], atol=0.005
        )
        part = data["parts"]["bottle_segment"]
        points = state.load(part["fit_points_artifact"], step=0)
        indices = state.load(part["fit_inlier_indices_artifact"], step=0)
        assert len(indices) > 30 and max(indices) < len(points)


def test_collinear_plane_samples_cannot_create_a_rectangular_top():
    points = np.column_stack((np.linspace(0, 0.1, 60), np.zeros(60), np.ones(60)))
    assert not fit_plane(points, FitOptions()).accepted
    assert not fit_rectangle(points, points).accepted
