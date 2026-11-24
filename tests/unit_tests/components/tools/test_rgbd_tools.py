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

"""Offline contracts for calibrated measurements, target identity and passive views."""

from types import SimpleNamespace

import numpy as np
import pytest

from components.models.molmo.client import MolmoResult
from components.models.sam3.client import Sam3Result
from components.tools.rgbd.geometry.geometry_tool import GeometryTool
from components.tools.rgbd.geometry.local_point_agent import LocalPointAgent
from components.tools.rgbd.geometry.multiview import observe_with_alternatives
from components.tools.rgbd.observation import GeometryObservation
from rpent.session import EnvState
from rpent.tools import ToolResult


def test_back_project_measures_declared_raster_without_claiming_identity():
    from components.tools.rgbd.back_project import back_project
    from components.tools.rgbd.observation import GeometryObservation

    h, w = 24, 32
    rr, cc = np.indices((h, w))
    xyz = np.stack((cc / 100, 1 + rr / 100, np.zeros_like(rr)), axis=-1)
    valid = np.ones((h, w), bool)
    valid[3, 5] = False
    obs = GeometryObservation(
        "overhead",
        0,
        "rotated",
        np.zeros((h, w, 3), np.uint8),
        world_xyz=xyz,
        valid_mask=valid,
        up_direction=np.array([0.0, 1.0, 0.0]),
    )
    source = SimpleNamespace(get_observation=lambda **_: obs)

    def run(selection):
        return back_project(
            camera="overhead", step=0, selection=selection, observations=source
        ).data

    result = run({"image_shape": [12, 16], "points_rc": [[1, 2], [3, 4]]})
    assert result["status"] == "sampled" and "world_xyz" not in result
    assert result["samples_xyz"] == [None, xyz[7, 9].tolist()]
    points = {"image_shape": [12, 16], "points_rc": [[1, 2], [3, 4], [5, 6]]}
    filtered = run({**points, "height_m": [1.07, 1.07]})
    assert filtered["samples_xyz"] == [None, xyz[7, 9].tolist(), None]
    assert filtered["n_height_filtered"] == 1
    empty_points = run({**points, "height_m": [2, None]})
    assert empty_points["reason"] == "no_selected_points_in_height_interval"
    assert empty_points["status"] == "need_observation"
    assert empty_points["samples_xyz"] == [None, None, None]
    assert "low <= high" in run({**points, "height_m": [2, 1]})["error"]
    for bad, diagnostic in (
        ({"points_rc": [[1, 2], [1, 16]]}, "[1, 16] at index 1"),
        ({"box_rc": [4, 4, 13, 16]}, "bottom <= 12"),
    ):
        invalid = run({"image_shape": [12, 16], **bad})
        assert diagnostic in invalid["error"] and "[12, 16]" in invalid["error"]
        assert "surface_xyz" not in invalid and "samples_xyz" not in invalid
    sampled = run(
        {"image_shape": [24, 32], "box_rc": [2, 4, 12, 20], "height_m": [1.06, 1.08]}
    )
    assert sampled["n_valid"] == 48 and 1.06 <= sampled["surface_xyz"][1] <= 1.08
    assert run({"image_shape": [12, 12], "points_rc": [[1, 1]]})["status"] == "error"
    empty = run(
        {"image_shape": [24, 32], "box_rc": [2, 4, 12, 20], "height_m": [3.0, 4.0]}
    )
    assert empty["status"] == "need_observation" and "surface_xyz" not in empty


def test_text_segmentation_checks_the_positive_point_on_the_native_mask(tmp_path):
    from components.tools.rgbd.segment import SegmentTool

    xyz = np.ones((6, 8, 3))
    state = EnvState(tmp_path)
    with state.record_step(state={}):
        pass
    frame = GeometryObservation(
        "overhead", 0, "table", np.zeros((6, 8, 3), np.uint8), world_xyz=xyz
    )
    source = SimpleNamespace(get_observation=lambda **_: frame)
    mask = np.zeros((6, 8), dtype=bool)
    mask[2:, 4:] = True
    prompts = []

    def infer(image, *, text_prompt, point, min_score):
        prompts.append((text_prompt, point))
        return Sam3Result(found=text_prompt == "cup", score=0.9, mask=mask)

    tool = SegmentTool(
        observations=source, state=state, segmenter=SimpleNamespace(segment=infer)
    )
    args = {"camera": frame.camera, "image_shape": [3, 4]}
    text_only = tool.segment(**args, prompt="cup").data
    accepted = tool.segment(**args, prompt="cup", point=[1, 2]).data
    assert accepted["mode"] == "text_with_point_check" and accepted["point"] == [3, 5]
    assert accepted["segment_artifact"].startswith("segment_")
    unscaled = tool.segment(camera=frame.camera, prompt="cup", point=[1, 2]).data
    assert unscaled["status"] == "error" and "image_shape" in unscaled["error"]
    assert accepted["world_xyz"] == text_only["world_xyz"] == [1.0, 1.0, 1.0]
    assert state.load(accepted["mask_artifact"], step=0)[3, 5]
    for prompt, point, reason in (
        ("cup", [0, 2], "text_mask_misses_positive_point"),
        ("unseen object", [1, 2], "segmenter found no mask"),
    ):
        rejected = tool.segment(**args, prompt=prompt, point=point)
        assert rejected.data["status"] == "need_observation"
        assert not rejected.data["found"] and rejected.data["reason"] == reason
        assert (
            rejected.data["mask_artifact"] is None
            and rejected.data["world_xyz"] is None
        )
        assert not rejected.images
    invalid = tool.segment(**args, point=[3, 2]).data
    assert invalid["status"] == "error" and "rows 0..2" in invalid["error"]
    with pytest.raises(ValueError):
        tool.segment_observation(frame, b"", tag="invalid", prompt="cup", point=[-1, 5])
    assert prompts == [
        ("cup", None),
        ("cup", None),
        ("cup", None),
        ("unseen object", None),
    ]


@pytest.mark.parametrize(
    ("relation", "name", "scene_answer", "peer", "reason"),
    [
        ("between", "bowl", "yes", False, None),
        ("between", "ramekin", "yes", False, "independent_object_name_not_confirmed"),
        ("between", "bowl", "yes", True, "another_visually_compatible_instance"),
        ("left of the plate", "blue cup", "yes", False, None),
        (
            "left of the plate",
            "plate",
            "yes",
            False,
            "independent_object_name_not_confirmed",
        ),
        (
            "to the right of the plate",
            "blue cup",
            "no",
            False,
            "local_model_did_not_confirm_target",
        ),
    ],
)
def test_spatial_relation_checks_the_scene_and_the_isolated_target(
    monkeypatch, relation, name, scene_answer, peer, reason
):
    from io import BytesIO

    from PIL import Image

    from components.tools.rgbd.geometry import target_identity

    if relation == "between":
        subject = "patterned black bowl"
        relation = "between the red-rim plate and the silver gray ramekin"
    else:
        subject = "blue cup"
    query = f"{subject} {relation}"
    mask = np.zeros((100, 120), bool)
    mask[30:70, 10:50] = True
    other = np.roll(mask, 60, axis=1)
    obs = SimpleNamespace(
        valid_mask=np.ones_like(mask), rgb=np.full((100, 120, 3), 128, np.uint8)
    )
    obs.rgb[85:95, 108:118] = [255, 0, 0]  # Landmark outside the target crop.
    monkeypatch.setattr(
        target_identity, "_opening_evidence", lambda *args: {"resolved": True}
    )
    monkeypatch.setattr(
        target_identity,
        "similar_instances",
        lambda *args: [(other, False)] if peer else [],
    )
    questions = []

    def verify(image, prompt, *, isolated=False):
        questions.append((prompt, isolated))
        if isolated:
            return name if prompt is None else "yes"
        panels = np.asarray(Image.open(BytesIO(image)))
        context = panels[:, panels.shape[1] // 2 :]
        assert np.all(context == [255, 0, 0], axis=-1).any()
        return scene_answer

    out = target_identity.verify_region(
        obs, mask, query, SimpleNamespace(verify=verify), {"calls": {}}, [45, 20]
    )
    assert out["accepted"] is (reason is None)
    assert query in questions[0][0] and questions[0][1] is False
    if reason:
        assert out["reason"] == reason
    if scene_answer == "yes":
        assert (None, True) in questions
    if reason is None:
        assert out["relation_verified"] and (subject, True) in questions
    if peer:
        assert query in questions[-1][0]


def _obs(camera, x=0, frame="table", step=0):
    cam = np.eye(4)
    cam[0, 3] = x
    return GeometryObservation(
        camera,
        step,
        frame,
        np.zeros((8, 8, 3), np.uint8),
        depth_m=np.ones((8, 8)),
        intrinsics=np.eye(3),
        camera_to_world=cam,
        up_direction=np.array([0.0, 0.0, 1.0]),
    )


@pytest.mark.parametrize(
    "reason",
    ["another_occluded_colored_instance", "another_occluded_compatible_instance"],
)
def test_wrist_lookalike_cannot_resolve_occluded_target(reason):
    front = _obs("agentview")
    wrist = _obs("wrist", x=0.4)
    provider = SimpleNamespace(
        get_observation=lambda **kwargs: front,
        additional_observations=lambda primary: [wrist],
    )

    def measure(*args, observation, evidence, **kwargs):
        if observation is front:
            # The leftmost cube projects outside the wrist image.
            evidence["anchor_xyz"] = front.world_xyz[3, 0]
            return ToolResult(data={"status": "need_observation", "reason": reason})
        return ToolResult(
            data={
                "status": "usable",
                "found": True,
                "world_xyz": wrist.world_xyz[3, 0].tolist(),
            }
        )

    result = observe_with_alternatives(
        provider,
        measure,
        "the leftmost red cube",
        "agentview",
        0,
        "surface_point",
        None,
        lambda: None,
    ).data
    assert result["status"] == "insufficient_evidence"
    assert result.get("world_xyz") is None


@pytest.mark.parametrize("overlap", [True, False])
def test_only_synchronized_overlapping_material_can_complete_partial_geometry(overlap):
    front = _obs("front")
    side = _obs("side", 0.4)

    class Provider:
        def get_observation(self, **kwargs):
            return front

        def additional_observations(self, first):
            assert first is front
            yield _obs("crop")
            yield _obs("foreign", 0.2, frame="other")
            yield _obs("later", 0.3, step=2)
            yield side
            raise RuntimeError("next camera is offline")

    def measure(
        query,
        camera,
        step,
        measurement,
        fields,
        cancel,
        *,
        observation,
        evidence,
        direction_camera=None,
        projected_rc=None,
    ):
        primary = observation is front
        if not primary:
            np.testing.assert_array_equal(direction_camera, front.camera_to_world)
        shift = 0.0 if primary or overlap else 1.5
        evidence["points"] = np.arange(30).reshape(10, 3) * 0.001 + shift
        quantities = {
            k: {
                "value": v,
                "fit_accepted": v is not None,
                "source_camera": camera,
                "warnings": [],
                "reason": None,
            }
            for k, v in zip(fields, ([1.0, None] if primary else [None, 2.0]))
        }
        return ToolResult(
            data={
                "camera": camera,
                "step": step,
                "query": query,
                "status": "partial",
                "found": True,
                "geometry": quantities,
                "identity_evidence": {"depth": {"accepted": True}},
            }
        )

    result = observe_with_alternatives(
        Provider(),
        measure,
        "a box",
        "front",
        0,
        "surface_point",
        ["rectangle_center", "height_above_support"],
        lambda: None,
    ).data
    assert [item["reason"] for item in result["view_evidence"]["skipped"]][:3] == [
        "same_optical_view",
        "incompatible_calibration_or_world_frame",
        "different_pinned_step",
    ]
    if overlap:
        assert result["status"] == "usable"
        assert result["geometry"]["height_above_support"]["source_camera"] == "side"
    else:
        assert (
            result["view_evidence"]["skipped"][-1]["reason"]
            == "additional_observation_unavailable"
        )
        assert (
            result["status"] == "partial"
            and result["geometry"]["height_above_support"]["value"] is None
        )
        assert result["missing_geometry"] == ["height_above_support"]
        assert "observation_request" not in result


def test_geometry_verifies_identity_and_reuses_evidence_only_within_a_step(tmp_path):
    from scipy.ndimage import binary_dilation

    height, width = 70, 110
    rows, cols = np.indices((height, width))
    rgb = np.full((height, width, 3), 140, dtype=np.uint8)
    xyz = np.stack((cols * 0.002, rows * 0.002, rows * 0.0), axis=2)
    rgb[25:44, 63:82] = [215, 15, 13]
    xyz[25:44, 63:82, 2] = 0.038
    pose = np.diag([1.0, -1.0, -1.0, 1.0])
    pose[2, 3] = 1.0

    def snapshot(*, camera, step):
        return GeometryObservation(
            camera,
            step,
            "bench",
            rgb,
            world_xyz=xyz,
            camera_to_world=pose,
            up_direction=np.array([0.0, 0.0, 1.0]),
        )

    class Grounder:
        answer = "no"
        calls = 0

        def ground(self, image, query):
            self.calls += 1
            return MolmoResult(True, (72.0, 34.0), (width, height))

        def verify(self, image, query, *, isolated=False):
            return self.answer if query is not None else "a toy block"

    state = EnvState(tmp_path)
    backend = Grounder()
    tool = GeometryTool(
        state=state,
        observations=SimpleNamespace(get_observation=snapshot),
        grounder=LocalPointAgent(backend),
    )
    with state.record_step(state={}):
        pass

    def locate():
        return tool.geometry(query="the red cube", camera="overhead").data

    refused = locate()
    assert refused["status"] == "insufficient_evidence" and "world_xyz" not in refused
    calls = backend.calls
    backend.answer = "yes"
    assert locate() == refused and backend.calls == calls
    with state.record_step(state={}):
        pass
    accepted = locate()
    assert accepted["status"] == "usable" and accepted["step"] == 1
    row, col = accepted["pixel_rc"]
    np.testing.assert_allclose(accepted["world_xyz"], xyz[row, col])
    assert 63 <= col < 82 and backend.calls > calls

    sam = SimpleNamespace(
        segment=lambda *args, **kw: Sam3Result(
            found=True, mask=xyz[:, :, 2] > 0, score=0.95
        )
    )
    tool = GeometryTool(
        state=state,
        observations=SimpleNamespace(get_observation=snapshot),
        grounder=LocalPointAgent(backend),
        segmenter=sam,
    )
    plane = tool.geometry(
        "the red cube", "overhead", required_geometry=["plane_normal"]
    ).data
    assert (plane["measurement"], plane["definition"]) == (
        "part_geometry",
        "part_fitted_geometry",
    )
    mixed = tool.geometry(
        "the red cube",
        "overhead",
        required_geometry=["rectangle_center", "plane_normal"],
    ).data
    normal = mixed["geometry"]["plane_normal"]
    assert normal["fit_accepted"]
    np.testing.assert_allclose(normal["value"], [0.0, 0.0, 1.0], atol=1e-6)
    center = mixed["geometry"]["rectangle_center"]
    assert "value" not in center and center["warnings"]
    assert mixed["missing_geometry"] == ["rectangle_center"]
    heights = tool.geometry(
        "toy block",
        "overhead",
        required_geometry=["height_above_support", "supported_height"],
    ).data
    assert heights["geometry"]["height_above_support"]["value"] == pytest.approx(0.038)
    assert heights["missing_geometry"] == ["supported_height"]
    rgb[25:44, 15:34] = [215, 15, 13]
    xyz[25:44, 15:34, 2] = 0.038
    tool.reset_cache()
    ambiguous = tool.geometry(
        "the red cube", "overhead", required_geometry=["rectangle_center"]
    ).data
    assert ambiguous["status"] == "insufficient_evidence"
    assert "value" not in ambiguous["geometry"]["rectangle_center"]

    # A flat printed square can look like a cube if missing depth supplies its outline.
    rgb[25:44, 15:34] = 140
    xyz[..., 2] = 0
    square = rgb[..., 0] == 215
    xyz[binary_dilation(square, iterations=2) & ~square] = np.nan
    with state.record_step(state={}):
        pass
    printed = locate()
    assert printed["status"] == "insufficient_evidence" and "world_xyz" not in printed


def test_back_project_uses_the_calibrated_frame_and_optical_depth():
    from components.tools.rgbd.back_project import back_project

    pose = np.eye(4)
    pose[:3, 3] = [4.0, 2.0, 1.0]
    obs = GeometryObservation(
        "side",
        3,
        "station",
        np.zeros((4, 6, 3), np.uint8),
        depth_m=np.full((4, 6), 0.5),
        intrinsics=np.diag([2.0, 2.0, 1.0]),
        camera_to_world=pose,
    )
    data = back_project(
        "side",
        {"image_shape": [4, 6], "points_rc": [[2, 4]]},
        3,
        observations=SimpleNamespace(get_observation=lambda **_: obs),
    ).data
    assert data["world_frame_id"] == "station" and data["units"] == "m"
    np.testing.assert_allclose(data["samples_xyz"], [[5.0, 2.5, 1.5]])


def test_depth_dropouts_do_not_supply_face_or_rim_boundaries():
    from components.tools.rgbd.geometry.geometry import part_candidate_mask
    from components.tools.rgbd.geometry.measured_rectangle import observed_face

    mask = np.zeros((72, 88), bool)
    mask[15:56, 20:68] = True
    depth = np.where(mask, 0.9, 1.04)
    depth[30:35, 40:44] = 0  # Missing depth inside a solid face.
    valid = np.ones_like(mask)
    valid[24:28, 30:34] = False
    depth[~valid] = 2.0  # A finite placeholder can also be marked invalid.
    pose = np.diag([1.0, -1.0, -1.0, 1.0])
    pose[2, 3] = 1.4

    def frame():
        return GeometryObservation(
            "front",
            0,
            "bench",
            np.zeros((*depth.shape, 3), np.uint8),
            intrinsics=np.array([[240.0, 0, 44.0], [0, 240.0, 36.0], [0, 0, 1.0]]),
            camera_to_world=pose,
            up_direction=[0, 0, 1],
            depth_m=depth,
            valid_mask=valid,
        )

    obs = frame()
    visible = mask & obs.valid_mask
    face = observed_face(obs, visible, obs.world_xyz[visible])
    assert face["supported"].all()
    for part in ("rim", "bottom"):
        candidates = part_candidate_mask(
            visible,
            obs.world_xyz,
            obs.rgb,
            part,
            valid_mask=obs.valid_mask,
            camera_transform=pose,
        )
        assert candidates[15, 40]
        assert not candidates[29, 41] and not candidates[23, 31]

    depth[10:62, 60:75] = 0  # Hide one entire side and both of its corners.
    obs = frame()
    visible = mask & obs.valid_mask
    face = observed_face(obs, visible, obs.world_xyz[visible])
    assert face["supported"].sum() == 3


def test_rim_depth_edges_use_only_adjacent_pixels_in_the_frame():
    from components.tools.rgbd.geometry.geometry import part_candidate_mask

    mask = np.ones((12, 18), bool)
    world = np.ones((*mask.shape, 3))
    world[-1, :, 2] = 2
    world[:, -1, 2] = 2
    candidates = part_candidate_mask(
        mask,
        world,
        np.zeros_like(world, dtype=np.uint8),
        "rim",
        camera_transform=np.eye(4),
    )
    assert candidates[-2, 5] and candidates[5, -2]
    assert not candidates[0, 5] and not candidates[5, 0]


@pytest.mark.parametrize("with_wrist", [False, True])
def test_plate_clipped_by_two_edges_needs_a_second_view(tmp_path, with_wrist):
    rows, cols = np.indices((96, 128))
    k = np.array([[300.0, 0, 64], [0, 300.0, 48], [0, 0, 1]])

    def view(camera, col, row):
        # Both cameras see the same 12 cm plate centered at (0, 0, 0.03) m.
        depth = 0.72
        disc = np.hypot(cols - col, rows - row) <= 0.06 * k[0, 0] / depth
        rgb = np.full((*disc.shape, 3), 110, dtype=np.uint8)
        rgb[disc] = 225
        pose = np.diag([1.0, -1.0, -1.0, 1.0])
        pose[:3, 3] = [(64 - col) * depth / 300, (row - 48) * depth / 300, 0.75]
        return GeometryObservation(
            camera,
            0,
            "table",
            rgb,
            depth_m=np.where(disc, depth, 0.75),
            intrinsics=k,
            camera_to_world=pose,
            up_direction=[0, 0, 1],
        )

    clipped = view("front", 3, 3)
    state = EnvState(tmp_path)
    with state.record_step(state={}):
        pass
    tool = GeometryTool(
        observations=SimpleNamespace(
            get_observation=lambda **_: clipped,
            additional_observations=lambda _: (
                [view("wrist", 64, 48)] if with_wrist else []
            ),
        ),
        grounder=SimpleNamespace(
            ground=lambda *_: MolmoResult(True, (9.0, 9.0), (128, 96)),
            verify=lambda image, query, **_: "yes" if query else "plate",
        ),
        state=state,
    )
    out = tool.geometry("plate", "front", required_geometry=["plate_center"]).data
    field = out["geometry"]["plate_center"]
    if with_wrist:
        assert out["status"] == "usable" and field["source_camera"] == "wrist"
        np.testing.assert_allclose(field["value"], [0, 0, 0.03], atol=0.001)
    else:
        assert out["status"] == "insufficient_evidence" and "value" not in field
        assert "insufficient_circle_angular_coverage" in field["warnings"]


def test_part_measurements_keep_independent_fits_and_reject_inconsistent_ends():
    from components.tools.rgbd.geometry.geometry import ShapeFit, geometry_fields

    circle = {"normal": [0, 0, 1], "radius": 0.035}
    rim = ShapeFit("circle", True, {**circle, "center": [0, 0, 0.1]})
    bottom = ShapeFit("circle", True, {**circle, "center": [0, 0, 0]})
    body = ShapeFit(
        "cylinder",
        True,
        {
            "axis_point": [0.04, 0, 0],
            "axis_direction": [1, 0, 0],
            "visible_axial_range": [-0.04, 0.04],
            "visible_end_radii": [0.035, 0.035],
        },
    )
    fits = {"rim": rim, "bottom": bottom, "body": body}
    singles = ("rim_radius", "rim_center", "bottom_center", "body_radii", "axis")
    fields, _ = geometry_fields(fits, [*singles, "height"])
    for name, part in zip(singles, ("rim", "rim", "bottom", "body", "body")):
        alone, _ = geometry_fields({part: fits[part]}, [name])
        assert fields[name] == alone[name] and fields[name]["fit_accepted"]
    assert not fields["height"]["fit_accepted"]
    assert "inconsistent_part_fits" in fields["height"]["warnings"]

    body.warn("insufficient_body_angular_coverage")
    fields, _ = geometry_fields(fits, ["rim_radius", "height", "axis"])
    assert all(field["fit_accepted"] for field in fields.values())
    assert fields["height"]["value"] == pytest.approx(0.1)
    assert fields["axis"]["source_parts"] == ["rim", "bottom"]

    # The accepted body supplies the axis when an end has parameters but fails its fit.
    body = ShapeFit(
        "cylinder",
        True,
        {**body.parameters, "axis_direction": [0, 0, 1], "axis_point": [0, 0, 0]},
    )
    bottom.warn("insufficient_circle_angular_coverage")
    fields, _ = geometry_fields(
        {**fits, "body": body}, ["rim_radius", "height", "axis"]
    )
    assert fields["rim_radius"]["fit_accepted"] and not fields["height"]["fit_accepted"]
    assert fields["axis"]["fit_accepted"] and fields["axis"]["source_parts"] == ["body"]
