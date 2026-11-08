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

"""Offline part perception fixtures exercise the real cleanup/fitting chain."""

import json
from types import SimpleNamespace

import numpy as np
import pytest

from robots.libero.tools import LiberoPrimitives, write_recipe_from_states
from rpent.robots.components.molmo_client import MolmoResult
from rpent.robots.components.sam3_client import Sam3Result
from rpent.session import EnvState
from toolkits.geometry.geometry_agent import GeometryAgent


def make_agent(
    tmp_path, *, part="rim", arc=360, hide_other_parts=False, dtype=np.float32
):
    state = EnvState(tmp_path / "episode")
    rr, cc = np.indices((128, 128))
    x, y = (cc - 64) * 0.001, (rr - 64) * 0.001
    world = np.stack((x, y, np.full(x.shape, 0.8)), axis=-1).astype(dtype)
    if part == "rim":
        mask = np.abs(np.hypot(x, y) - 0.04) < 0.0016
        angle = np.degrees(np.mod(np.arctan2(y, x), 2 * np.pi))
        mask &= angle <= arc
    elif part == "bottom":
        mask = np.hypot(x, y) <= 0.04
    else:
        mask = np.zeros((128, 128), dtype=bool)
        mask[20:108, 20:108] = True
    with state.record_step(state={}):
        state.save("agentview.png", np.zeros((128, 128, 3), dtype=np.uint8))
        state.save("agentview_world.npz", world)
        state.save("agentview_depth.npz", np.ones((128, 128)))
    queries, sam_calls = [], []

    def ground(image, query):
        queries.append(query)
        visible = not hide_other_parts or "rim boundary" in query
        point = (104, 64) if part == "rim" else (64, 64)
        return MolmoResult(
            found=visible, point_xy=point if visible else None, image_size=(128, 128)
        )

    def segment(image, **kwargs):
        sam_calls.append(kwargs)
        return Sam3Result(found=True, mask=mask, score=0.95)

    primitives = LiberoPrimitives(
        env=object(),
        model=object(),
        sam3_client=SimpleNamespace(segment=segment),
        check_cancelled=lambda: None,
    )
    agent = GeometryAgent(
        state=state,
        segment=primitives.segment,
        check_cancelled=lambda: None,
        grounder=SimpleNamespace(ground=ground),
    )
    return state, agent, queries, sam_calls


def test_rim_circle_integration_uses_raw_boundary_not_mask_median(tmp_path):
    state, agent, queries, sam_calls = make_agent(tmp_path)
    result = agent.localize(
        "yellow and white mug", step=0, required_geometry=["rim_center", "radius"]
    ).data
    assert result["status"] == "usable", result
    center = result["geometry"]["rim_center"]
    np.testing.assert_allclose(center["value"], [0, 0, 0.8], atol=0.001)
    assert result["geometry"]["radius"]["value"] == pytest.approx(0.04, abs=0.0015)
    assert center["evidence_status"] == "inferred" and center["source_step"] == 0
    assert result["world_xyz"] is None
    assert len(queries) == len(sam_calls) == 1
    part = result["parts"]["rim"]
    cleaned = state.load(part["cleaned_points_artifact"], step=0)
    indices = state.load(part["fit_inlier_indices_artifact"], step=0)
    assert len(indices) >= 30 and len(cleaned) >= len(indices)
    assert part["fit"]["quality"]["angular_coverage_deg"] > 300
    assert center["source_refs"]["rim"]["mask_artifact"] == part["mask_artifact"]
    saved = state.load(result["geometry_artifact"], step=0)
    assert saved["geometry"]["rim_center"]["value"] == center["value"]
    assert len(state.records()) == 1


def test_short_arc_requests_more_observation_instead_of_using_a_low_residual(tmp_path):
    _, agent, _, _ = make_agent(tmp_path, arc=60)
    data = agent.localize("mug", required_geometry=["rim_center"]).data
    assert data["status"] == "need_observation"
    assert data["geometry"]["rim_center"]["value"] is None
    assert (
        data["parts"]["rim"]["fit"]["reason"] == "insufficient_circle_angular_coverage"
    )
    assert data["observation_request"]["missing_geometry"] == ["rim_center"]


def test_visible_bottom_disk_uses_its_boundary_and_keeps_bottom_plane(tmp_path):
    _, agent, _, _ = make_agent(tmp_path, part="bottom")
    data = agent.localize("mug", required_geometry=["bottom_center"]).data
    assert data["status"] == "usable", data
    np.testing.assert_allclose(
        data["geometry"]["bottom_center"]["value"], [0, 0, 0.8], atol=0.001
    )
    assert "background_plane" not in data["parts"]["bottom"]["cleaning"]


def test_missing_ends_keep_full_height_and_body_center_unknown(tmp_path):
    _, agent, queries, sam_calls = make_agent(tmp_path, hide_other_parts=True)
    data = agent.localize(
        "mug", required_geometry=["rim_center", "height", "body_center"]
    ).data
    assert data["status"] == "partial", data
    assert data["geometry"]["rim_center"]["value"] is not None
    assert data["geometry"]["height"]["value"] is None
    assert data["geometry"]["body_center"]["evidence_status"] == "unknown"
    assert len(queries) == 3 and len(sam_calls) == 1
    assert data["calls"]["grounding"] == 3
    assert data["observation_request"]["execution_owner"] == "parent_planner"


def test_plane_request_does_not_remove_target_as_background(tmp_path):
    _, agent, _, _ = make_agent(tmp_path, part="plane")
    data = agent.localize(
        "microwave top", required_geometry=["plane", "plane_normal"]
    ).data
    assert data["status"] == "usable", data
    normal = np.array(data["geometry"]["plane_normal"]["value"])
    assert abs(normal[2]) > 0.999
    assert "background_plane" not in data["parts"]["plane"]["cleaning"]
    assert data["geometry"]["plane"]["quality"]["plane"]["calibrated_error_m"] is None


def test_fit_trace_retains_quantized_source_dtype(tmp_path):
    _, agent, _, _ = make_agent(tmp_path, dtype=np.float16)
    data = agent.localize("mug", required_geometry=["rim_center"]).data
    assert data["status"] == "usable", data
    assert data["source_dtype"] == "float16"
    assert (
        data["geometry"]["rim_center"]["quality"]["rim"]["calibrated_error_m"] is None
    )


@pytest.mark.parametrize("partial", [False, True])
def test_fitted_recipe_preserves_requested_quantities_and_hides_internal_calls(
    tmp_path,
    partial,
):
    state, agent, _, _ = make_agent(tmp_path, hide_other_parts=partial)
    requested = ["rim_center", "height"] if partial else ["rim_center", "radius"]
    agent.localize("mug", required_geometry=requested)
    with state.record_step(state={}, terminated=True):
        pass
    name = write_recipe_from_states(state, "parts", output_dir=tmp_path)
    commands = [json.loads(line) for line in (tmp_path / name).read_text().splitlines()]
    assert commands == [
        {
            "action": "localize",
            "query": "mug",
            "camera": "agentview",
            "required_geometry": requested,
        }
    ]


def test_curved_body_request_runs_cleanup_and_axisymmetric_fit(tmp_path):
    state = EnvState(tmp_path / "body-episode")
    theta, z = np.meshgrid(
        np.linspace(0, np.radians(220), 32), np.linspace(0.8, 0.92, 20)
    )
    world = np.stack((0.04 * np.cos(theta), 0.04 * np.sin(theta), z), axis=-1).astype(
        np.float32
    )
    mask = np.ones((20, 32), dtype=bool)
    with state.record_step(state={}):
        state.save("agentview.png", np.zeros((20, 32, 3), dtype=np.uint8))
        state.save("agentview_world.npz", world)
        state.save("agentview_depth.npz", np.ones((20, 32)))
    primitives = LiberoPrimitives(
        env=object(),
        model=object(),
        check_cancelled=lambda: None,
        sam3_client=SimpleNamespace(
            segment=lambda *a, **kw: Sam3Result(found=True, mask=mask, score=0.9)
        ),
    )
    grounder = SimpleNamespace(
        ground=lambda *a: MolmoResult(
            found=True, point_xy=(16, 10), image_size=(32, 20)
        )
    )
    agent = GeometryAgent(
        state=state,
        segment=primitives.segment,
        check_cancelled=lambda: None,
        grounder=grounder,
    )
    data = agent.localize("cup", required_geometry=["axis", "body_radii"]).data
    assert data["status"] == "usable", data
    assert data["parts"]["body"]["fit"]["model"] == "cylinder"
    assert abs(data["geometry"]["axis"]["value"]["direction"][2]) > 0.99
    np.testing.assert_allclose(
        data["geometry"]["body_radii"]["value"]["radii"], 0.04, atol=0.001
    )
    assert data["parts"]["body"]["fit"]["parameters"]["height"] is None
