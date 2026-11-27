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

"""Offline contracts for calibrated measurements and target identity."""

import io
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
from PIL import Image
from scipy.ndimage import label

from rpent.robots.components.molmo_client import MolmoResult
from rpent.robots.components.sam3_client import Sam3Result
from rpent.session import EnvState
from toolkits.geometry.geometry_tool import GeometryTool
from toolkits.geometry.local_point_agent import LocalPointAgent
from toolkits.observation import GeometryObservation


def test_back_project_measures_declared_raster_without_claiming_identity():
    from toolkits.back_project import back_project
    from toolkits.observation import GeometryObservation

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


def test_segment_routes_text_and_scaled_point_prompts(tmp_path):
    from jsonschema import Draft202012Validator

    from toolkits.segment import SegmentTool

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
        return Sam3Result(
            found=text_prompt == "cup" or point == [3, 5], score=0.9, mask=mask
        )

    tool = SegmentTool(
        observations=source, state=state, segmenter=SimpleNamespace(segment=infer)
    )
    args = {"camera": frame.camera, "image_shape": [3, 4]}
    for schema in (
        tool.segment.input_schema,
        tool.segment.args_schema.model_json_schema(),
    ):
        validator = Draft202012Validator(schema)
        for valid in (
            {"prompt": "cup"},
            {"prompt": "cup", "point": None},
            {"point": [1, 2], "image_shape": [3, 4]},
            {"point": [1, 2], "image_shape": [3, 4], "prompt": " "},
        ):
            validator.validate({"camera": frame.camera, **valid})
        for invalid in (
            {},
            {"prompt": " "},
            {"point": [1, 2]},
            {"point": [1, 2], "image_shape": None},
            {"prompt": "cup", "point": [1, 2], "image_shape": [3, 4]},
        ):
            assert not validator.is_valid({"camera": frame.camera, **invalid})
    text_only = tool.segment(camera=frame.camera, prompt="cup").data
    point_only = tool.segment(**args, point=[1, 2]).data
    assert text_only["mode"] == "text" and text_only["point"] is None
    assert point_only["mode"] == "point" and point_only["point"] == [3, 5]
    assert point_only["segment_artifact"].startswith("segment_")
    unscaled = tool.segment(camera=frame.camera, point=[1, 2]).data
    assert unscaled["status"] == "error" and "image_shape" in unscaled["error"]
    assert point_only["world_xyz"] == text_only["world_xyz"] == [1.0, 1.0, 1.0]
    assert state.load(point_only["mask_artifact"], step=0)[3, 5]
    missing = tool.segment(camera=frame.camera, prompt="unseen object")
    assert not missing.data["found"]
    assert missing.data["reason"] == "segmenter found no mask"
    assert missing.data["mask_artifact"] is None and missing.data["world_xyz"] is None
    assert not missing.images
    invalid = tool.segment(**args, point=[3, 2]).data
    assert invalid["status"] == "error" and "rows 0..2" in invalid["error"]
    assert prompts == [("cup", None), (None, [3, 5]), ("unseen object", None)]


@pytest.mark.parametrize(
    "prompts",
    [{"prompt": "cup", "point": [1, 2], "image_shape": [6, 8]}, {"prompt": " "}],
)
def test_segment_rejects_combined_or_empty_prompts_before_observation(
    tmp_path, prompts
):
    from toolkits.segment import SegmentTool

    def unexpected_observation(**kwargs):
        pytest.fail("Invalid prompts should be rejected before loading an observation.")

    tool = SegmentTool(
        observations=SimpleNamespace(get_observation=unexpected_observation),
        state=EnvState(tmp_path),
        segmenter=None,
    )
    result = tool.segment(camera="overhead", **prompts).data
    assert result["status"] == "error" and "exactly one" in result["error"]


@pytest.mark.parametrize(("camera", "step"), [("wrist", 0), ("agentview", 1)])
def test_geometry_rejects_a_frame_from_a_different_camera_or_step(
    tmp_path, camera, step
):
    observation = GeometryObservation(
        camera,
        step,
        "table",
        np.zeros((1, 1, 3), np.uint8),
        world_xyz=np.ones((1, 1, 3)),
    )
    state = EnvState(tmp_path)
    with state.record_step(state={}):
        pass
    tool = GeometryTool(
        state=state,
        observations=SimpleNamespace(get_observation=lambda **_: observation),
        grounder=SimpleNamespace(),
    )
    result = tool.geometry(
        "cup",
        "agentview",
        object_prompt="cup",
        required_geometry=["ring_center"],
        feature="upper_inner_ring",
    ).data
    assert result["code"] == "observation_unavailable"
    assert (result["camera"], result["step"]) == ("agentview", 0)


class _PaintedMasks:
    """Return each painted patch as an instance in the supplied original RGB image."""

    def __init__(self, color):
        self.color = color
        self.prompts = []

    def segment_candidates(self, image, *, text_prompt, min_score):
        from rpent.robots.components.sam3_client import Sam3Result

        assert isinstance(image, bytes)
        with Image.open(io.BytesIO(image)) as im:
            image = np.array(im)
        self.prompts.append((text_prompt, image.shape[:2]))
        instances, count = label(np.all(image == self.color, axis=2))
        return [
            Sam3Result(found=True, score=0.9, mask=instances == i)
            for i in range(1, count + 1)
        ]


def test_geometry_shows_the_selected_mask_when_the_query_names_the_other_cube(tmp_path):
    rows, cols = np.indices((60, 100))
    left = (abs(rows - 30) <= 9) & (abs(cols - 22) <= 9)
    right = (abs(rows - 30) <= 9) & (abs(cols - 72) <= 9)
    rgb = np.full((*rows.shape, 3), 140, np.uint8)
    rgb[left | right] = [255, 0, 0]
    xyz = np.stack((cols * 0.002, rows * 0.002, (left | right) * 0.038), axis=-1)
    pose = np.diag([1.0, -1.0, -1.0, 1.0])
    pose[2, 3] = 1.0
    frame = GeometryObservation(
        "overhead",
        0,
        "bench",
        rgb,
        world_xyz=xyz,
        camera_to_world=pose,
        up_direction=[0, 0, 1],
    )
    state = EnvState(tmp_path)
    with state.record_step(state={}):
        pass
    candidates = Mock(return_value=[Sam3Result(True, mask=left, score=0.9)])
    ground = Mock()
    tool = GeometryTool(
        state=state,
        observations=SimpleNamespace(get_observation=lambda **_: frame),
        segmenter=SimpleNamespace(segment_candidates=candidates),
        grounder=SimpleNamespace(ground=ground),
    )

    def measure():
        return tool.geometry(
            "the red cube on the right",
            "overhead",
            object_prompt="red cube",
            required_geometry=["rectangle_center"],
            feature="upper_face",
        )

    result = measure()
    assert result.data["status"] == "usable"
    saved = state.load(result.data["geometry_artifact"], step=0)
    assert saved["identity_evidence"]["object"] == {"method": "sam3_single_candidate"}
    np.testing.assert_allclose(
        result.data["geometry"]["rectangle_center"]["value"], xyz[30, 22], atol=0.002
    )
    (png,) = result.images
    with Image.open(io.BytesIO(png)) as image:
        assert image.format == "PNG"
        overlay = np.asarray(image)
    np.testing.assert_array_equal(overlay[~left], rgb[~left])
    assert np.abs(overlay[left].astype(int) - rgb[left]).max() > 64
    assert measure() == result
    candidates.assert_called_once()
    ground.assert_not_called()


def test_geometry_uses_a_single_mask_directly_and_reuses_evidence_within_a_step(
    tmp_path,
):
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
        ground_queries = []
        point = (15.0, 34.0)

        def ground(self, image, query):
            self.ground_queries.append(query)
            return MolmoResult(self.point is not None, self.point, (width, height))

    state = EnvState(tmp_path)
    backend = Grounder()
    sam = _PaintedMasks([215, 15, 13])
    tool = GeometryTool(
        state=state,
        observations=SimpleNamespace(get_observation=snapshot),
        grounder=LocalPointAgent(backend),
        segmenter=sam,
    )
    with state.record_step(state={}):
        pass

    def fit_cube(query="the red cube"):
        return tool.geometry(
            query,
            "overhead",
            object_prompt="red cube",
            required_geometry=["rectangle_center"],
            feature="upper_face",
        ).data

    first = fit_cube()
    assert first["status"] == "usable" and not first["warnings"]
    np.testing.assert_allclose(
        first["geometry"]["rectangle_center"]["value"], xyz[34, 72], atol=0.002
    )
    detail = state.load(first["geometry_artifact"], step=0)
    assert detail["identity_evidence"]["object"]["method"] == "sam3_single_candidate"
    assert detail["calls"]["grounding"] == 0
    assert backend.ground_queries == []
    assert sam.prompts == [("red cube", (height, width))]
    backend.point = (72.0, 34.0)
    assert fit_cube() == first and len(sam.prompts) == 1
    with state.record_step(state={}):
        pass
    fitted = fit_cube()
    assert fitted["status"] == "usable" and fitted["step"] == 1
    assert backend.ground_queries == []
    assert sam.prompts == [("red cube", (height, width))] * 2

    for request, feature, code in (
        ([], "upper_face", "unsupported_geometry_quantity"),
        (["plane_normal"], "upper_face", "unsupported_geometry_quantity"),
        (["rectangle_center"], None, "invalid_geometry_feature"),
        (["rectangle_center"], "upper_inner_ring", "invalid_geometry_feature"),
        (["ring_center"], "lower_outer_ring", "invalid_geometry_feature"),
        (["disk_center"], "lower_face", "invalid_geometry_feature"),
        (["rectangle_center"], "largest_face", "invalid_geometry_feature"),
        (["base_center"], "supported_base", "invalid_geometry_feature"),
        (["base_center"], "supported_rectangular_base", "invalid_geometry_feature"),
        (["cylinder_base_center"], "supported_base", "unsupported_geometry_quantity"),
        (
            ["disk_center", "base_center"],
            "supported_round_base",
            "invalid_geometry_feature",
        ),
    ):
        result = tool.geometry(
            "the red cube",
            "overhead",
            required_geometry=request,
            feature=feature,
            object_prompt="red cube",
        ).data
        assert result["status"] == "error" and result["code"] == code
    assert len(sam.prompts) == 2

    assert fitted["feature"] == "upper_face"
    center = fitted["geometry"]["rectangle_center"]
    np.testing.assert_allclose(center["value"], xyz[34, 72], atol=0.002)
    assert center["fit_accepted"]
    detail = state.load(fitted["geometry_artifact"], step=1)
    assert detail["request"]["feature"] == "upper_face"
    assert "measurement" not in detail["request"]
    assert "world_xyz" not in fitted
    assert detail["calls"]["segment"] == 1
    assert detail["calls"]["back_project"] == 1
    for field in detail["geometry"].values():
        assert field["source_refs"]["object_mask"] == detail["object_mask_artifact"]
    wrong_shape = tool.geometry(
        "the red cube",
        "overhead",
        object_prompt="red cube",
        required_geometry=["ring_center"],
        feature="upper_outer_ring",
    ).data["geometry"]["ring_center"]
    assert not wrong_shape["fit_accepted"] and "value" not in wrong_shape
    rgb[26:44, 15:34] = [215, 15, 13]
    xyz[26:44, 15:34, 2] = 0.038
    tool.reset_cache()
    full_query = "the red cube to the right of the other cube"
    previous_calls = len(backend.ground_queries)
    selected = fit_cube(full_query)
    np.testing.assert_allclose(
        selected["geometry"]["rectangle_center"]["value"], xyz[34, 72], atol=0.002
    )
    assert backend.ground_queries[previous_calls:] == [full_query]
    assert sam.prompts[-1] == ("red cube", (height, width))
    backend.point = None
    fallback_fit = fit_cube()
    assert fallback_fit["status"] == "warning"
    assert fallback_fit["warnings"] == [
        {
            "stage": "object",
            "reason": "no_single_molmo_point",
            "source_camera": "overhead",
        }
    ]
    center = fallback_fit["geometry"]["rectangle_center"]
    assert center["fit_accepted"]
    np.testing.assert_allclose(center["value"], xyz[34, 72], atol=0.002)
    mixed = tool.geometry(
        "the red cube",
        "overhead",
        object_prompt="red cube",
        feature="upper_face",
        required_geometry=["rectangle_center", "disk_center"],
    ).data
    assert mixed["status"] == "partial" and mixed["missing_geometry"] == ["disk_center"]
    assert fallback_fit["warnings"][0] in mixed["warnings"]

    # Material identified in RGB still needs a real depth sample in the mask.
    xyz[rgb[..., 0] == 215] = np.nan
    with state.record_step(state={}):
        pass
    no_depth = fit_cube()
    assert no_depth["status"] == "insufficient_evidence"
    assert no_depth["reason"] == "no_valid_depth_in_object_mask"
    assert no_depth["warnings"] == fallback_fit["warnings"]
    assert no_depth["missing_geometry"] == ["rectangle_center"]
    assert "value" not in no_depth["geometry"]["rectangle_center"]


def test_back_project_uses_the_calibrated_frame_and_optical_depth():
    from toolkits.back_project import back_project

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
    from toolkits.geometry.geometry import observed_silhouette
    from toolkits.geometry.measured_rectangle import observed_face

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
    plane = (obs.world_xyz[15, 40], obs.up_direction)
    face = observed_face(obs, visible, object_mask=mask, plane=plane)
    assert face["supported"].all()
    boundary = observed_silhouette(visible, obs.valid_mask)
    assert boundary[15, 40]
    assert not boundary[29, 41] and not boundary[23, 31]

    depth[10:62, 60:75] = 0  # Hide one entire side and both of its corners.
    obs = frame()
    visible = mask & obs.valid_mask
    face = observed_face(obs, visible, object_mask=mask, plane=plane)
    assert face["supported"].sum() == 3


def test_silhouette_uses_only_adjacent_pixels_in_the_frame():
    from toolkits.geometry.geometry import observed_silhouette

    mask = np.ones((12, 18), bool)
    mask[-1, :] = False
    mask[:, -1] = False
    boundary = observed_silhouette(mask, np.ones_like(mask))
    assert boundary[-2, 5] and boundary[5, -2]
    assert not boundary[0, 5] and not boundary[5, 0]


def _stored_fit(state, result, quantity):
    evidence = state.load(result["geometry_artifact"], step=result["step"])
    return evidence["geometry"][quantity]


def _mask_tool(tmp_path, mask, x, y, z, *, from_below=False, check_cancelled=None):
    pose = np.eye(4)
    if not from_below:
        pose[1, 1] = pose[2, 2] = -1
        pose[2, 3] = 1
    rgb = np.zeros((*mask.shape, 3), np.uint8)
    rgb[mask] = 220
    obs = GeometryObservation(
        "front",
        0,
        "table",
        rgb,
        world_xyz=np.stack((x, y, z), axis=-1),
        camera_to_world=pose,
        up_direction=[0, 0, 1],
    )
    state = EnvState(tmp_path)
    with state.record_step(state={}):
        pass
    tool = GeometryTool(
        state=state,
        observations=SimpleNamespace(get_observation=lambda **_: obs),
        segmenter=_PaintedMasks([220, 220, 220]),
        grounder=SimpleNamespace(),
        check_cancelled=check_cancelled,
    )
    return state, tool


def _measure_mask(tmp_path, mask, x, y, z, quantity, feature, *, from_below=False):
    state, tool = _mask_tool(tmp_path, mask, x, y, z, from_below=from_below)
    result = tool.geometry(
        "target",
        "front",
        object_prompt="target",
        required_geometry=[quantity],
        feature=feature,
    ).data
    return result, _stored_fit(state, result, quantity)


def _fit_face_center(mask, x, y, z, quantity, feature):
    from toolkits.geometry.geometry import FitOptions
    from toolkits.geometry.object_features import fit_features
    from toolkits.geometry.rgbd_parts import object_points

    pose = np.diag([1.0, -1.0, -1.0, 1.0])
    pose[2, 3] = 1.0
    obs = GeometryObservation(
        "front",
        0,
        "table",
        np.zeros((*mask.shape, 3), np.uint8),
        world_xyz=np.stack((x, y, z), axis=-1),
        camera_to_world=pose,
        up_direction=[0, 0, 1],
    )
    mask.setflags(write=False)
    cloud = object_points(
        obs.world_xyz,
        obs.valid_mask,
        mask,
        pose[:3, 2],
        keep_small_faces=feature == "upper_face",
    )
    _, fit = next(
        fit_features([quantity], feature, obs, mask, cloud, FitOptions(), lambda: None)
    )
    return fit


@pytest.mark.parametrize("edge", ["step", "bevel", "asymmetric_bevel"])
def test_disk_center_measures_the_offset_planar_face_on_a_shallow_plate(tmp_path, edge):
    rows, cols = np.indices((151, 151))
    x, y = (cols - 75) * 0.0008, (75 - rows) * 0.0008
    mask = np.hypot(x, y) <= 0.055
    # The planar patch is offset from the enclosing circular outline.
    radius = np.hypot(x - 0.0032, y)
    if edge == "step":
        flat_radius = 0.038
        rise = 0.006 * (radius > flat_radius)
    else:
        flat_radius = 0.035
        width = 0.018
        if edge == "asymmetric_bevel":
            width *= 1 + 0.4 * np.cos(np.arctan2(y, x - 0.0032))
        rise = 0.008 * np.clip((radius - flat_radius) / width, 0, 1)
    z = np.where(mask, 0.05 + rise, 0.01)
    result, saved = _measure_mask(tmp_path, mask, x, y, z, "disk_center", "upper_face")
    assert result["status"] == "usable"
    np.testing.assert_allclose(saved["value"], [0.0032, 0, 0.05], atol=0.0006)
    assert saved["radius_m"] == pytest.approx(flat_radius, abs=0.001)
    assert saved["definition"] == "center_of_selected_observed_circular_planar_surface"


@pytest.mark.parametrize(
    "quantity,feature",
    [
        ("disk_center", "upper_face"),
        ("rectangle_center", "upper_face"),
        ("ring_center", "upper_outer_ring"),
    ],
)
def test_upper_feature_rejects_an_underside_seen_from_below(
    tmp_path, quantity, feature
):
    rows, cols = np.indices((101, 101))
    x, y = (cols - 50) * 0.001, (rows - 50) * 0.001
    mask = (
        np.hypot(x, y) <= 0.04
        if quantity != "rectangle_center"
        else (abs(x) <= 0.03) & (abs(y) <= 0.02)
    )
    z = np.where(mask, 0.50, 0.65)
    result, saved = _measure_mask(
        tmp_path, mask, x, y, z, quantity, feature, from_below=True
    )
    assert result["status"] == "insufficient_evidence"
    assert saved["value"] is None


@pytest.mark.parametrize(
    ("feature", "interior_z", "include_bottom", "radius_m"),
    [
        ("upper_inner_ring", 0.03, False, 0.038),
        ("upper_inner_ring", 0.03, True, 0.038),
        ("upper_outer_ring", 0.03, False, 0.040),
        ("upper_inner_ring", np.nan, False, None),
    ],
)
def test_thin_ring_uses_the_requested_measured_edge(
    tmp_path, feature, interior_z, include_bottom, radius_m
):
    rows, cols = np.indices((101, 101))
    x, y = (cols - 50) * 0.001, (50 - rows) * 0.001
    radius = np.hypot(x, y)
    mask = (0.038 <= radius) & (radius <= 0.040)
    z = np.select([mask, radius < 0.038], [0.05, interior_z], default=0.03)
    if include_bottom:
        mask |= radius < 0.038
    result, saved = _measure_mask(tmp_path, mask, x, y, z, "ring_center", feature)
    if radius_m is None:
        assert result["status"] == "insufficient_evidence"
        assert saved["value"] is None
        return
    assert result["status"] == "usable"
    np.testing.assert_allclose(saved["value"], [0, 0, 0.05], atol=0.001)
    assert saved["radius_m"] == pytest.approx(radius_m, abs=0.0008)


def test_inner_rim_can_cross_the_image_border(tmp_path):
    rows, cols = np.indices((115, 115))
    x, y = (cols - 28) * 0.001, (57 - rows) * 0.001
    radius = np.hypot(x, y)
    mask = (radius >= 0.030) & (radius <= 0.045)
    z = np.where(mask, 0.05, 0.02)
    result, saved = _measure_mask(
        tmp_path, mask, x, y, z, "ring_center", "upper_inner_ring"
    )
    assert result["status"] == "usable"
    np.testing.assert_allclose(saved["value"], [0, 0, 0.05], atol=0.001)
    assert saved["radius_m"] == pytest.approx(0.030, abs=0.001)
    assert saved["quality"]["angular_coverage_deg"] > 300


@pytest.mark.parametrize("cropped", [False, True])
def test_stepped_opening_preserves_the_height_of_the_upper_lip(tmp_path, cropped):
    rows, cols = np.indices((241, 241))
    x, y = (cols - 120) * 0.0005, (120 - rows) * 0.0005
    radius = np.hypot(x, y)
    lid = radius <= 0.055
    z = np.select(
        [radius < 0.006, radius < 0.040, lid], [0.02, 0.05, 0.10], default=0.0
    )
    view = np.s_[:, x[0] >= -0.008] if cropped else np.s_[:, :]
    result, saved = _measure_mask(
        tmp_path,
        lid[view],
        x[view],
        y[view],
        z[view],
        "ring_center",
        "upper_inner_ring",
    )
    if cropped:
        assert result["missing_geometry"] == ["ring_center"]
        assert saved["value"] is None and saved["reason"] == "partial_circular_arc"
    else:
        np.testing.assert_allclose(saved["value"], [0, 0, 0.10], atol=0.0006)


@pytest.mark.parametrize(
    "opening", ["thin_lip", "offset", "two_openings", "solid", "d_shaped"]
)
def test_inner_ring_uses_the_openings_own_depth_boundary(opening):
    pitch, half = (0.0005, 205) if opening == "thin_lip" else (0.001, 105)
    rows, cols = np.indices((2 * half + 1, 2 * half + 1))
    x, y = (cols - half) * pitch, (half - rows) * pitch
    lid = np.hypot(x, y) <= 0.100
    center_x, radius = (0.0, 0.0995) if opening == "thin_lip" else (0.080, 0.006)
    floor = (np.hypot(x - center_x, y) < radius) & (opening != "solid")
    if opening == "two_openings":
        floor |= np.hypot(x + center_x, y) < radius
    elif opening == "d_shaped":
        floor = (np.hypot(x, y) < 0.080) & (x <= 0.048)
    z = np.select([floor, lid], [0.030, 0.050], default=0.010)
    fit = _fit_face_center(lid, x, y, z, "ring_center", "upper_inner_ring")
    assert fit.accepted == (opening in {"thin_lip", "offset"}), fit.reason
    if opening == "two_openings":
        assert fit.reason == "ambiguous_inner_rings"
    elif opening == "d_shaped":
        assert fit.reason == "noncircular_measured_edge"
    if fit.accepted:
        np.testing.assert_allclose(
            fit.parameters["center"], [center_x, 0, 0.05], atol=0.0006
        )
        assert fit.parameters["radius"] == pytest.approx(radius, abs=0.0006)


@pytest.mark.parametrize(
    ("raised", "top_z"),
    [
        ("tab", 0.065),
        ("competing_rim", 0.065),
        ("neck", 0.065),
        ("neck", 0.055),
        ("taper", 0.060),
        ("rounded_taper", 0.060),
        ("dome", 0.060),
    ],
)
def test_outer_rim_checks_height_around_its_circumference(tmp_path, raised, top_z):
    rows, cols = np.indices((111, 111))
    x, y = (cols - 55) * 0.001, (55 - rows) * 0.001
    radius = np.hypot(x, y)
    mask = radius <= 0.045
    top = (
        (abs(x) <= 0.008) & (abs(y) <= 0.008)
        if raised == "tab"
        else (radius >= 0.039) & (radius <= 0.042)
    )
    if raised == "neck":
        top = (radius >= 0.012) & (radius <= 0.015)
    z = np.select([top, mask], [top_z, 0.05], default=0.01)
    center_x = 0.0
    if raised == "taper":
        z[mask] = np.interp(radius[mask], [0.015, 0.045], [top_z, 0.05])
    elif raised == "rounded_taper":
        center_x = 0.0035
        rise = np.clip((0.045 - np.hypot(x[mask] - center_x, y[mask])) / 0.030, 0, 1)
        z[mask] = 0.05 + (top_z - 0.05) * rise**2 * (3 - 2 * rise)
    elif raised == "dome":
        z[mask] = top_z - (top_z - 0.05) * (radius[mask] / 0.045) ** 2
    result, saved = _measure_mask(
        tmp_path, mask, x, y, z, "ring_center", "upper_outer_ring"
    )
    if raised == "tab":
        assert result["status"] == "usable"
        np.testing.assert_allclose(saved["value"], [0, 0, 0.05], atol=0.001)
        assert saved["radius_m"] == pytest.approx(0.045, abs=0.001)
    elif raised == "dome":
        assert result["status"] == "insufficient_evidence"
        assert saved["value"] is None
    elif raised in ("neck", "taper", "rounded_taper"):
        assert result["status"] == "usable"
        np.testing.assert_allclose(saved["value"], [center_x, 0, top_z], atol=0.001)
        assert saved["radius_m"] == pytest.approx(0.015, abs=0.001)
    else:
        assert saved["value"] is None or saved["value"][2] == pytest.approx(
            0.065, abs=0.001
        )


@pytest.mark.parametrize(
    "scene",
    [
        "raised_face",
        "foreground_aperture",
        "coplanar_mask",
        "half_hidden",
        "half_mask",
        "small_hole",
        "covered_center",
    ],
)
def test_disk_center_uses_the_observed_face_and_its_own_edges(scene):
    rows, cols = np.indices((131, 131))
    x, y = (cols - 65) * 0.001, (65 - rows) * 0.001
    disk = np.hypot(x, y) <= 0.05
    mask = disk.copy()
    height = np.where(disk, 0.05, 0.0)
    expected = [0.0, 0.0, 0.05]
    if scene == "raised_face":
        # The depth step separates a small, independently sampled upper face.
        height[np.hypot(x - 0.027, y) <= 0.010] = 0.07
        expected = [0.027, 0.0, 0.07]
    elif scene in {"foreground_aperture", "coplanar_mask"}:
        mask = np.hypot(x - 0.015, y) <= 0.024
        if scene == "foreground_aperture":
            height[~mask] = 0.10
    elif scene in {"half_hidden", "half_mask"}:
        mask &= x <= 0
        if scene == "half_hidden":
            height[x > 0] = 0.10
    else:
        center = np.hypot(x, y) < 0.010
        mask[center] = False
        height[center] = 0.10 if scene == "covered_center" else 0.0
    fit = _fit_face_center(mask, x, y, height, "disk_center", "upper_face")
    if scene in {"foreground_aperture", "coplanar_mask", "small_hole"}:
        assert not fit.accepted
    else:
        assert fit.accepted
        np.testing.assert_allclose(fit.parameters["center"], expected, atol=0.001)


@pytest.mark.parametrize("occluded", [False, True])
def test_disk_distinguishes_a_measured_chord_from_an_occluding_edge(occluded):
    rows, cols = np.indices((211, 211))
    x, y = (cols - 105) * 0.001, (105 - rows) * 0.001
    cut = x > 0.050
    face = (np.hypot(x, y) <= 0.100) & ~cut
    z = np.where(face, 0.050, 0.010)
    if occluded:
        z[cut] = 0.090
    fit = _fit_face_center(face, x, y, z, "disk_center", "upper_face")
    assert fit.accepted == occluded, fit.reason
    if occluded:
        np.testing.assert_allclose(fit.parameters["center"], [0, 0, 0.05], atol=0.001)
    else:
        assert fit.reason == "noncircular_measured_edge"


def test_two_opposite_strips_of_a_square_do_not_establish_a_disk():
    rows, cols = np.indices((421, 421))
    x, y = (cols - 210) * 0.0005, (210 - rows) * 0.0005
    plate = (abs(x) <= 0.100) & (abs(y) <= 0.100)
    z = np.select([plate, abs(y) <= 0.005], [0.05, 0.01], default=np.nan)
    fit = _fit_face_center(plate, x, y, z, "disk_center", "upper_face")
    assert not fit.accepted
    assert fit.reason == "insufficient_circle_angular_coverage"
    assert fit.quality["angular_coverage_deg"] == pytest.approx(11.4, abs=0.5)


@pytest.mark.parametrize(
    ("central", "opening_z", "dropout"),
    [
        (False, 0.05, False),
        (False, np.nan, False),
        (True, 0.05, False),
        (True, np.nan, False),
        (True, 0.05, True),
    ],
)
def test_rectangular_face_distinguishes_a_measured_cutout_from_missing_depth(
    central, opening_z, dropout
):
    rows, cols = np.indices((121, 141))
    x, y = (cols - 70) * 0.001, (60 - rows) * 0.001
    plate = (abs(x) <= 0.055) & (abs(y) <= 0.045)
    upper = (abs(x - 0.008) <= 0.030) & (abs(y) <= 0.025)
    opening = upper & (x > -0.004) & (y > -0.010)
    if central:
        opening = (abs(cols - 78) <= 5) & (abs(rows - 60) <= 5)
    height = np.select([upper, plate], [0.07, 0.05], default=0.01)
    height[opening] = opening_z
    if dropout:
        height[60, 78] = np.nan

    fit = _fit_face_center(plate, x, y, height, "rectangle_center", "upper_face")
    assert fit.accepted
    expected = [0, 0, 0.05] if np.isfinite(opening_z) else [0.008, 0, 0.07]
    np.testing.assert_allclose(fit.parameters["center"], expected, atol=0.001)


@pytest.mark.parametrize("background_z", [0.055, 0.045])
def test_rectangle_requires_an_outward_depth_drop_at_a_window(background_z):
    rows, cols = np.indices((85, 105))
    x, y = (cols - 49) * 0.001, (42 - rows) * 0.001
    visible = (abs(cols - 60) <= 30) & (abs(rows - 38) <= 20)
    height = np.where(visible, 0.050, background_z)
    fit = _fit_face_center(visible, x, y, height, "rectangle_center", "upper_face")
    assert fit.accepted == (background_z < 0.050)


@pytest.mark.parametrize(
    ("center_z", "dropout"), [(0.0, False), (0.0, True), (0.08, False), (np.nan, False)]
)
def test_disk_center_distinguishes_an_opening_at_coarse_pixel_spacing(
    center_z, dropout
):
    rows, cols = np.indices((53, 53))
    x, y = (cols - 25.5) * 0.005, (26.5 - rows) * 0.005
    radius = np.hypot(x, y)
    material = (radius <= 0.1) & (radius >= 0.020)
    z = np.select([material, radius < 0.020], [0.05, center_z], default=0.0)
    if dropout:
        z[26, 25] = np.nan
    scale = (1 - z) / (1 - 0.05)
    fit = _fit_face_center(
        material, x * scale, y * scale, z, "disk_center", "upper_face"
    )
    assert fit.accepted == (center_z != 0.0)
    if center_z == 0.0:
        assert fit.reason == "disk_center_over_opening"


@pytest.mark.parametrize("missing_cols, clipped", [(1, False), (2, False), (0, True)])
def test_rectangle_center_requires_a_measured_fourth_edge(missing_cols, clipped):
    rows, cols = np.indices((77, 101))
    x, y = (cols - 50) * 0.001, (38 - rows) * 0.001
    left = 0 if clipped else 20
    face = (cols >= left) & (cols <= 80) & (abs(rows - 38) <= 20)
    mask = face & (cols <= 80 - missing_cols)
    missing_edge_pixels = np.count_nonzero(face & ~mask)
    # A disconnected patch at the same height also has a pixel in the object mask.
    face[15:17, 78:81] = True
    mask[15, 79] = True
    fit = _fit_face_center(
        mask, x, y, np.where(face, 0.05, 0.01), "rectangle_center", "upper_face"
    )
    if missing_cols == 1:
        assert fit.accepted
        assert 0 < fit.quality["adjacent_edge_samples"] <= missing_edge_pixels
        np.testing.assert_allclose(fit.parameters["center"], [0, 0, 0.05], atol=0.0005)
    else:
        assert not fit.accepted and not fit.parameters
        if clipped:
            assert fit.reason == "rectangle_face_at_image_border"
            assert fit.quality["image_borders"] == ["left"]
        else:
            assert fit.reason == "rectangle_edges_not_observed"


@pytest.mark.parametrize("inset", [1, 2])
def test_rectangle_measures_all_four_edges_of_an_inset_mask(inset):
    from scipy.ndimage import binary_erosion

    rows, cols = np.indices((95, 115))
    x, y = (cols - 57) * 0.001, (47 - rows) * 0.001
    face = (abs(x) <= 0.035) & (abs(y) <= 0.022)
    mask = binary_erosion(face, iterations=inset)
    fit = _fit_face_center(
        mask, x, y, np.where(face, 0.05, 0.01), "rectangle_center", "upper_face"
    )
    assert fit.accepted == (inset == 1)
    if fit.accepted:
        np.testing.assert_allclose(
            fit.parameters["side_lengths_m"], [0.070, 0.044], atol=0.0008
        )
        assert fit.quality["adjacent_edge_samples"] > 0


@pytest.mark.parametrize(
    ("quantity", "feature"),
    [
        ("disk_center", "upper_face"),
        ("ring_center", "upper_outer_ring"),
        ("ring_center", "upper_inner_ring"),
    ],
)
@pytest.mark.parametrize(("inset", "outside_z"), [(1, 0.02), (1, 0.07), (2, 0.02)])
def test_circular_edge_one_pixel_outside_mask_needs_a_measured_drop(
    quantity, feature, inset, outside_z
):
    from scipy.ndimage import binary_erosion

    rows, cols = np.indices((95, 105))
    x, y = (cols - 55) * 0.001, (45 - rows) * 0.001
    face = np.hypot(x, y) <= 0.035
    if feature == "upper_inner_ring":
        face = ~face & (np.hypot(x, y) <= 0.042)
    mask = binary_erosion(face, iterations=inset)
    fit = _fit_face_center(
        mask, x, y, np.where(face, 0.05, outside_z), quantity, feature
    )
    assert fit.accepted == (inset == 1 and outside_z < 0.05)
    if fit.accepted:
        np.testing.assert_allclose(fit.parameters["center"], [0, 0, 0.05], atol=0.001)
        assert fit.parameters["radius"] == pytest.approx(0.035, abs=0.0008)


def test_rectangle_in_a_deep_tray_pocket():
    rows, cols = np.indices((115, 115))
    x, y = (cols - 57) * 0.001, (57 - rows) * 0.001
    tray = np.hypot(x, y) <= 0.050
    pocket = (abs(x - 0.008) <= 0.022) & (abs(y) <= 0.014)
    z = np.select([pocket, tray], [0.05, 0.07], default=0.01)
    fit = _fit_face_center(tray, x, y, z, "rectangle_center", "upper_face")
    assert fit.accepted, fit.reason
    np.testing.assert_allclose(fit.parameters["center"], [0.008, 0, 0.05], atol=0.001)


@pytest.mark.parametrize("quantity", ["rectangle_center", "disk_center"])
@pytest.mark.parametrize("slope", [0.0, 0.01, 0.2])
def test_upper_face_ranks_coplanar_pads_by_center_height(quantity, slope):
    rows, cols = np.indices((95, 201))
    x, y = (cols - 100) * 0.001, (47 - rows) * 0.001
    if quantity == "rectangle_center":
        lower = (abs(x + 0.060) <= 0.027) & (abs(y) <= 0.020)
        upper = (abs(x - 0.060) <= 0.015) & (abs(y) <= 0.012)
    else:
        lower = np.hypot((x + 0.060) * np.hypot(1, slope), y) <= 0.026
        upper = np.hypot((x - 0.060) * np.hypot(1, slope), y) <= 0.016
    bridge = (abs(x) <= 0.060) & (abs(y) <= 0.004)
    z = np.select([lower | upper, bridge], [0.060 + slope * x, 0.030], default=0.01)
    fit = _fit_face_center(lower | upper | bridge, x, y, z, quantity, "upper_face")
    if slope < 0.2:
        assert not fit.accepted and fit.reason == "ambiguous_upper_faces"
        return
    assert fit.accepted, fit.reason
    np.testing.assert_allclose(fit.parameters["center"], [0.060, 0, 0.072], atol=0.001)


@pytest.mark.parametrize("quantity", ["rectangle_center", "disk_center"])
def test_upper_face_tries_separate_pads_at_the_same_height(tmp_path, quantity):
    rows, cols = np.indices((121, 201))
    x, y = (cols - 100) * 0.001, (60 - rows) * 0.001
    triangle = (x >= -0.085) & (x <= -0.015) & (abs(y) <= (x + 0.085) * 0.6)
    pad = (
        (abs(x - 0.04) <= 0.018) & (abs(y) <= 0.012)
        if quantity == "rectangle_center"
        else np.hypot(x - 0.04, y) <= 0.017
    )
    bridge = (abs(y) <= 0.005) & (x > -0.020) & (x < 0.035)
    mask = triangle | pad | bridge
    z = np.select([triangle | pad, bridge], [0.055, 0.033], default=0.010)
    result, saved = _measure_mask(tmp_path, mask, x, y, z, quantity, "upper_face")
    assert result["status"] == "usable"
    np.testing.assert_allclose(saved["value"], [0.04, 0, 0.055], atol=0.001)


@pytest.mark.parametrize("edge", ["one_pixel", "two_pixels", "separated", "occluded"])
def test_clipped_rectangle_requires_four_in_frame_edge_segments(edge):
    from scipy.ndimage import binary_erosion

    rows, cols = np.indices((167, 179))
    # At the top plane the calibrated pixel pitch is 1 mm. Rotation clips the
    # left corner while leaving a measurable segment of each physical side.
    x = (cols - 89) * 0.001 + 0.038
    y = (83 - rows) * 0.001
    angle = np.deg2rad(35)
    along = x * np.cos(angle) + y * np.sin(angle)
    across = -x * np.sin(angle) + y * np.cos(angle)
    top = (abs(along) <= 0.052) & (abs(across) <= 0.032)
    mask = top.copy()
    z = np.where(top, 0.05, 0.0)
    if edge in ("one_pixel", "two_pixels"):
        mask = binary_erosion(
            top, iterations=1 if edge == "one_pixel" else 2, border_value=1
        )
    else:
        strip = (across > 0.032) & (across < 0.046) & (abs(along) < 0.062)
        if edge == "separated":
            # Two small clusters can span a side without establishing its line.
            strip &= (abs(along + 0.009) > 0.0022) & (abs(along - 0.030) > 0.0022)
        z[strip] = 0.08 if edge == "occluded" else np.nan
    fit = _fit_face_center(
        mask,
        (cols - 89) * (1 - z) / 950,
        (83 - rows) * (1 - z) / 950,
        z,
        "rectangle_center",
        "upper_face",
    )
    assert fit.quality["image_borders"] == ["left"]
    if edge == "one_pixel":
        assert fit.accepted and fit.quality["adjacent_edge_samples"] > 0
        np.testing.assert_allclose(
            fit.parameters["center"], [-0.038, 0, 0.05], atol=0.001
        )
        np.testing.assert_allclose(
            fit.parameters["side_lengths_m"], [0.104, 0.064], atol=0.002
        )
    else:
        assert not fit.accepted and not fit.parameters
        assert fit.reason == "rectangle_face_at_image_border"


@pytest.mark.parametrize("clipped", [False, True])
def test_disk_fit_uses_only_the_primary_frame_even_when_a_second_view_is_available(
    tmp_path, clipped
):
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

    frames = {
        "front": view("front", 3, 3) if clipped else view("front", 64, 48),
        "wrist": view("wrist", 64, 48),
    }
    reads = []

    def get_observation(*, camera, step):
        reads.append((camera, step))
        return frames[camera]

    state = EnvState(tmp_path)
    with state.record_step(state={}):
        pass
    tool = GeometryTool(
        observations=SimpleNamespace(
            get_observation=get_observation,
            additional_observations=lambda primary: [
                get_observation(camera="wrist", step=primary.step)
            ],
        ),
        segmenter=_PaintedMasks([225, 225, 225]),
        grounder=SimpleNamespace(),
        state=state,
    )
    out = tool.geometry(
        "plate",
        "front",
        required_geometry=["disk_center"],
        feature="upper_face",
        object_prompt="plate",
    ).data
    field = out["geometry"]["disk_center"]
    assert reads == [("front", 0)]
    assert field["source_camera"] == "front"
    record = state.load(out["geometry_artifact"], step=0)
    assert record["status"] == out["status"]
    assert record["found"] == out["found"] == (not clipped)
    if clipped:
        assert out["status"] == "insufficient_evidence" and "value" not in field
        assert record["missing_geometry"] == out["missing_geometry"] == ["disk_center"]
        assert "insufficient_circle_angular_coverage" in field["warnings"]
        assert (
            record["reason"]
            == field["reason"]
            == "insufficient_circle_angular_coverage"
        )
        rejected = record["geometry"]["disk_center"]
        assert rejected["value"] is None and field["evidence_status"] == "unknown"
        assert len(rejected["quality"]["candidate_parameters"]["center"]) == 3
    else:
        assert out["status"] == "usable"
        np.testing.assert_allclose(field["value"], [0, 0, 0.03], atol=0.001)
        fit = _stored_fit(state, out, "disk_center")
        assert fit["radius_m"] == pytest.approx(0.06, abs=0.002)
        np.testing.assert_allclose(fit["normal_world"], [0, 0, 1], atol=1e-6)


def test_geometry_fits_the_upper_outer_ring_of_a_cylinder(tmp_path):
    angles = np.linspace(0, 2 * np.pi, 144, endpoint=False)
    heights = np.linspace(0.2, 0.31, 28)
    mask = np.zeros((30, 146), dtype=bool)
    mask[1:-1, 1:-1] = True
    world = np.ones((*mask.shape, 3))
    world[1:-1, 1:-1, 0] = 0.04 * np.cos(angles)
    world[1:-1, 1:-1, 1] = 0.04 * np.sin(angles)
    world[1:-1, 1:-1, 2] = heights[:, None]
    rgb = np.zeros_like(world, dtype=np.uint8)
    rgb[mask] = 220
    obs = GeometryObservation(
        "front",
        0,
        "table",
        rgb,
        world_xyz=world,
        camera_to_world=np.eye(4),
        up_direction=[0, 0, 1],
    )
    state = EnvState(tmp_path)
    with state.record_step(state={}):
        pass
    segmenter = _PaintedMasks([220, 220, 220])
    tool = GeometryTool(
        state=state,
        observations=SimpleNamespace(get_observation=lambda **_: obs),
        segmenter=segmenter,
        grounder=SimpleNamespace(
            ground=lambda *_: MolmoResult(True, (50.0, 14.0), (146, 30))
        ),
    )

    result = tool.geometry(
        "cup",
        "front",
        required_geometry=["ring_center"],
        feature="upper_outer_ring",
        object_prompt="cup",
    ).data
    assert result["status"] == "usable"
    assert set(result["geometry"]) == {"ring_center"}
    field = result["geometry"]["ring_center"]
    np.testing.assert_allclose(field["value"], [0, 0, 0.31], atol=0.002)
    assert {"radius_m", "normal_world"}.isdisjoint(field)
    fit = _stored_fit(state, result, "ring_center")
    assert fit["radius_m"] == pytest.approx(0.04, abs=0.001)
    np.testing.assert_allclose(fit["normal_world"], [0, 0, 1], atol=1e-6)


@pytest.mark.parametrize("outer", [False, True])
def test_oblique_rim_uses_the_lip_plane_for_both_edges(tmp_path, outer):
    # Cast calibrated camera rays through a circular lip and its two sidewalls.
    # The front lip borders the body. Through the opening, the rays hit a lower plane.
    rows, cols = np.indices((180, 180))
    pose = np.eye(4)
    c = np.sqrt(0.5)
    pose[:3, :3] = [[1, 0, 0], [0, -c, c], [0, -c, -c]]
    pose[:3, 3] = [0, -0.34, 0.34]
    camera = pose[:3, 3]
    intrinsics = np.array([[650, 0, 90], [0, 650, 90], [0, 0, 1]])
    pixels = np.stack((cols, rows, np.ones_like(rows)), axis=-1)
    rays = pixels @ np.linalg.inv(intrinsics).T @ pose[:3, :3].T
    top = -camera[2] / rays[..., 2]
    xy = camera[:2] + top[..., None] * rays[..., :2]
    radial = np.linalg.norm(xy, axis=-1)
    background = (-0.14 - camera[2]) / rays[..., 2]
    depth = np.where((radial >= 0.03) & (radial <= 0.04), top, background)
    a = np.sum(rays[..., :2] ** 2, axis=-1)
    b = rays[..., :2] @ camera[:2]
    thickness = 0.08 if outer else 0.008
    for radius in (0.03, 0.04):
        discriminant = b**2 - a * (camera[:2] @ camera[:2] - radius**2)
        for sign in (-1, 1):
            distance = (-b + sign * np.sqrt(np.maximum(discriminant, 0))) / a
            height = camera[2] + distance * rays[..., 2]
            wall = (discriminant >= 0) & (height <= 0) & (height >= -thickness)
            depth[wall] = np.minimum(depth[wall], distance[wall])
    rgb = np.zeros((*rows.shape, 3), np.uint8)
    rgb[depth < background] = 220
    obs = GeometryObservation(
        "front",
        0,
        "table",
        rgb,
        depth_m=depth,
        intrinsics=intrinsics,
        camera_to_world=pose,
        up_direction=[0, 0, 1],
    )
    state = EnvState(tmp_path)
    with state.record_step(state={}):
        pass
    tool = GeometryTool(
        state=state,
        observations=SimpleNamespace(get_observation=lambda **_: obs),
        segmenter=_PaintedMasks([220, 220, 220]),
        grounder=SimpleNamespace(),
    )
    result = tool.geometry(
        "round opening",
        "front",
        object_prompt="round object",
        required_geometry=["ring_center"],
        feature="upper_outer_ring" if outer else "upper_inner_ring",
    ).data
    assert result["status"] == "usable"
    fit = _stored_fit(state, result, "ring_center")
    np.testing.assert_allclose(fit["value"], [0, 0, 0], atol=0.001)
    assert fit["radius_m"] == pytest.approx(0.04 if outer else 0.03, abs=0.001)


@pytest.mark.parametrize(
    "interior", ["recess", "solid", "through_hole", "missing_depth"]
)
def test_round_face_distinguishes_material_holes_and_depth_dropouts(tmp_path, interior):
    rows, cols = np.indices((84, 84))
    radius = np.hypot(cols - 42, rows - 42)
    mask = radius <= 26
    height = np.where(mask, 0.08, 0)
    if interior == "recess":
        height[radius < 18] = 0.03
    elif interior == "missing_depth":
        height[radius < 18] = np.nan
    elif interior == "through_hole":
        mask[radius < 18] = False
        height[radius < 18] = 0.0
    world = np.stack((cols * 0.002, rows * 0.002, height), axis=2)
    rgb = np.zeros((*mask.shape, 3), np.uint8)
    rgb[mask] = 220
    pose = np.diag([1.0, -1.0, -1.0, 1.0])
    pose[2, 3] = 1.0
    obs = GeometryObservation(
        "front",
        0,
        "table",
        rgb,
        world_xyz=world,
        camera_to_world=pose,
        up_direction=[0, 0, 1],
    )
    state = EnvState(tmp_path)
    with state.record_step(state={}):
        pass
    tool = GeometryTool(
        state=state,
        observations=SimpleNamespace(get_observation=lambda **_: obs),
        segmenter=_PaintedMasks([220, 220, 220]),
        grounder=SimpleNamespace(
            ground=lambda *_: MolmoResult(True, (42.0, 42.0), (84, 84))
        ),
    )
    out = tool.geometry(
        "round object",
        "front",
        object_prompt="round object",
        required_geometry=["ring_center"],
        feature="upper_inner_ring",
    ).data
    field = out["geometry"]["ring_center"]
    if interior in {"recess", "through_hole"}:
        assert field["fit_accepted"]
        assert _stored_fit(state, out, "ring_center")["radius_m"] == pytest.approx(
            0.036, abs=0.002
        )
        np.testing.assert_allclose(field["value"], [0.084, 0.084, 0.08], atol=0.002)
    else:
        assert out["status"] == "insufficient_evidence"
        assert field["reason"] == "inner_ring_not_observed"
    outer = tool.geometry(
        "round object",
        "front",
        object_prompt="round object",
        required_geometry=["ring_center"],
        feature="upper_outer_ring",
    ).data
    assert outer["geometry"]["ring_center"]["fit_accepted"]
    assert _stored_fit(state, outer, "ring_center")["radius_m"] == pytest.approx(
        0.052, abs=0.002
    )
    result = tool.geometry(
        "round object",
        "front",
        object_prompt="round object",
        required_geometry=["disk_center", "rectangle_center"],
        feature="upper_face",
    ).data
    disk, rectangle = (
        result["geometry"]["disk_center"],
        result["geometry"]["rectangle_center"],
    )
    assert not rectangle["fit_accepted"] and "value" not in rectangle
    if interior in {"recess", "solid"}:
        assert result["status"] == "partial" and disk["fit_accepted"]
        np.testing.assert_allclose(
            disk["value"],
            [0.084, 0.084, 0.03 if interior == "recess" else 0.08],
            atol=0.002,
        )
        assert _stored_fit(state, result, "disk_center")["radius_m"] == pytest.approx(
            0.036 if interior == "recess" else 0.052, abs=0.002
        )
        assert result["missing_geometry"] == ["rectangle_center"]
    else:
        assert result["status"] == "insufficient_evidence"
        assert not disk["fit_accepted"] and "value" not in disk


@pytest.mark.parametrize("quantity", ["disk_center", "rectangle_center"])
def test_fitter_selects_the_upper_face_even_when_the_lower_face_is_larger(quantity):
    from toolkits.geometry.geometry import FitOptions
    from toolkits.geometry.object_features import fit_features
    from toolkits.geometry.rgbd_parts import object_points

    rows, cols = np.indices((90, 140))
    left_col, right_col, row = 40, 98, 45
    if quantity == "disk_center":
        left = np.hypot(cols - left_col, rows - row) <= 21
        right = np.hypot(cols - right_col, rows - row) <= 18
    else:
        left = (abs(cols - left_col) <= 22) & (abs(rows - row) <= 13)
        right = (abs(cols - right_col) <= 18) & (abs(rows - row) <= 10)
    # A thin bridge connects the two levels in the object mask.
    bridge = (rows == row) & (cols > left_col) & (cols < right_col)
    mask = left | right | bridge
    heights = np.select([left, right, bridge], [0.12, 0.06, 0.09], default=0.0)
    # The lower right face has fewer samples and a larger physical area.
    scale = np.clip((cols - 66) / 12, 0, 1)
    x = 0.002 * (cols + np.maximum(cols - 72, 0))
    y = 0.002 * (rows + scale * (rows - row))
    world = np.stack((x, y, heights), axis=2)
    rgb = np.zeros((*mask.shape, 3), np.uint8)
    rgb[mask] = 220
    pose = np.diag([1.0, -1.0, -1.0, 1.0])
    pose[2, 3] = 1.0
    # Tilt both faces in a Y-up world frame. Normals must follow the fitted plane.
    c, s = np.cos(0.19), np.sin(0.19)
    rotation = np.array([[c, 0, s], [-s, 0, c], [0, -1, 0]])
    pose[:3] = rotation @ pose[:3]
    world = world @ rotation.T
    obs = GeometryObservation(
        "front",
        0,
        "table",
        rgb,
        world_xyz=world,
        camera_to_world=pose,
        up_direction=[0, 1, 0],
    )
    cloud = object_points(obs.world_xyz, obs.valid_mask, mask, pose[:3, 2])
    _, fit = next(
        fit_features(
            [quantity], "upper_face", obs, mask, cloud, FitOptions(), lambda: None
        )
    )
    assert fit.accepted, fit.reason
    params = fit.parameters
    np.testing.assert_allclose(params["center"], world[row, left_col], atol=0.002)
    np.testing.assert_allclose(params["normal"], rotation[:, 2], atol=1e-6)
    if quantity == "disk_center":
        assert params["radius"] == pytest.approx(0.042, abs=0.003)
    else:
        np.testing.assert_allclose(params["side_lengths_m"], [0.088, 0.052], atol=0.002)


@pytest.mark.parametrize(
    "view",
    [
        "supported",
        "narrow_tray_edge",
        "cutout",
        "asymmetric_cutout",
        "floating",
        "fine_contact",
        "small_clearance",
        "short_arc",
        "no_support",
    ],
)
def test_cylindrical_base_uses_the_sidewall_and_its_support(tmp_path, view):
    # Orthographic oblique rays see a handled cylindrical body on a raised tray.
    # Its underside is hidden. Everything outside the tray is a lower floor.
    scale = 5 if view in ("fine_contact", "small_clearance") else 1
    rows, cols = np.indices((160 * scale, 180 * scale))
    down = np.array([0, -0.5, -np.sqrt(0.75)])
    forward = np.array([0, np.sqrt(0.75), -0.5])
    pos = np.array([0, -0.4, 0.065 + 0.4 / np.sqrt(3)])
    x = (cols / scale - 88) * 0.001
    origin = pos + (rows / scale - 80)[:, :, None] * 0.001 * down
    origin[:, :, 0] = x

    def at_height(z):
        return origin + ((z - origin[:, :, 2]) / forward[2])[:, :, None] * forward

    plane = at_height(0.02)
    tray_radius = np.hypot(plane[:, :, 0], plane[:, :, 1])
    front_arc = (abs(x) < 0.010) & (plane[:, :, 1] < 0)
    tray = tray_radius < np.where(
        (view == "narrow_tray_edge") & front_arc, 0.031, 0.052
    )
    if view in ("cutout", "asymmetric_cutout"):
        inner_radius = np.where((view == "asymmetric_cutout") & front_arc, 0.032, 0.035)
        tray &= tray_radius >= inner_radius
    world = np.where(tray[:, :, None], plane, at_height(0.0))
    radius = 0.03
    gap = {"floating": 0.025, "small_clearance": 0.005}.get(view, 0.0)
    base_z = 0.02 + gap
    front_y = -np.sqrt(np.maximum(0, radius**2 - x**2))
    wall = origin + ((front_y - origin[:, :, 1]) / forward[1])[:, :, None] * forward
    side = (
        (abs(x) < radius) & (wall[:, :, 2] >= base_z) & (wall[:, :, 2] < base_z + 0.09)
    )
    cap = at_height(base_z + 0.09)
    top = np.hypot(cap[:, :, 0], cap[:, :, 1]) < radius
    handle = origin + ((-0.002 - origin[:, :, 1]) / forward[1])[:, :, None] * forward
    handle_h = handle[:, :, 2] - base_z
    grip = (x >= 0.028) & (x < 0.048) & (handle_h >= 0.023) & (handle_h < 0.065)
    grip &= (x > 0.043) | (handle_h < 0.029) | (handle_h > 0.06)
    mask = side | top | grip
    world[grip] = handle[grip]
    world[side] = wall[side]
    world[top] = cap[top]
    if view == "short_arc":
        mask &= abs(x) < 0.009
    if view == "no_support":
        world[~mask] = np.nan
    # Exercise the observation's up vector in a Y-up world frame.
    rotate = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]])
    pose = np.eye(4)
    pose[:3, :3] = rotate @ np.column_stack(([1, 0, 0], down, forward))
    pose[:3, 3] = rotate @ pos
    rgb = np.zeros((*mask.shape, 3), dtype=np.uint8)
    rgb[mask] = 220
    obs = GeometryObservation(
        "front",
        0,
        "world",
        rgb,
        world_xyz=world @ rotate.T,
        camera_to_world=pose,
        up_direction=rotate[:, 2],
    )
    state = EnvState(tmp_path)
    with state.record_step(state={}):
        pass
    tool = GeometryTool(
        state=state,
        observations=SimpleNamespace(get_observation=lambda **_: obs),
        grounder=SimpleNamespace(),
        segmenter=_PaintedMasks([220, 220, 220]),
    )
    result = tool.geometry(
        "upright mug",
        "front",
        object_prompt="mug",
        required_geometry=["base_center"],
        feature="supported_round_base",
    ).data
    field = result["geometry"]["base_center"]
    if view in ("supported", "narrow_tray_edge", "fine_contact"):
        assert result["status"] == "usable", field
        np.testing.assert_allclose(field["value"], rotate @ [0, 0, 0.02], atol=0.002)
        assert field["evidence_status"] == "inferred"
        assert set(result["geometry"]) == {"base_center"}
        raw = _stored_fit(state, result, "base_center")
        assert raw["quality"]["sidewall_radius_m"] == pytest.approx(radius, abs=0.002)
        assert "radius_m" not in raw
    else:
        assert result["status"] == "insufficient_evidence" and "value" not in field
        if view in ("cutout", "asymmetric_cutout", "small_clearance"):
            assert field["reason"] == "base_support_gap_observed"


@pytest.mark.parametrize(
    "view",
    ["resting", "floating", "eccentric", "no_support", "open_gap", "small_platform"],
)
def test_round_base_needs_a_concentric_body_on_a_local_support(view):
    from toolkits.geometry.geometry import FitOptions
    from toolkits.geometry.object_features import fit_features
    from toolkits.geometry.rgbd_parts import object_points

    rows, cols = np.indices((140, 150))
    x, y = (cols - 73) * 0.001, (rows - 70) * 0.001
    radius = 0.052
    radial_sq = (x**2 + y**2) / radius**2
    mask = radial_sq <= 1
    # The symmetry axis is tilted relative to the raised support. Dropping the
    # rim center vertically would miss the axis intersection by over 6 mm.
    tilt = 0.17
    axis = np.array([np.sin(tilt), 0.0, np.cos(tilt)])
    across = np.array([np.cos(tilt), 0.0, -np.sin(tilt)])
    base = np.array([0.018, -0.011, 0.025])
    rise = 0.038 - 0.033 * (1 - radial_sq)
    if view == "eccentric":
        rise += 0.022 * (x / radius) * (1 - radial_sq)
    if view == "floating":
        rise += 0.025
    surface = (
        base + x[..., None] * across + y[..., None] * [0, 1, 0] + rise[..., None] * axis
    )
    world = surface.copy()
    world[~mask, 2] = np.nan if view == "no_support" else base[2]
    pose = np.diag([1.0, -1.0, -1.0, 1.0])
    pose[2, 3] = 1.0
    if view in {"open_gap", "small_platform"}:
        camera = base + [0, 0, 0.40]
        pose[:3, 3] = camera
        rim_plane = (
            base + x[..., None] * across + y[..., None] * [0, 1, 0] + 0.038 * axis
        )
        rays = rim_plane - camera
        counter = camera + rays * ((base[2] - camera[2]) / rays[..., 2])[..., None]
        radial = np.linalg.norm((counter - base)[..., :2], axis=2)
        gap = radial < radius + 0.008 if view == "open_gap" else radial > radius + 0.010
        height = base[2] - gap * 0.05
        background = camera + rays * ((height - camera[2]) / rays[..., 2])[..., None]
        world[~mask] = background[~mask]
    obs = GeometryObservation(
        "front",
        0,
        "world",
        np.zeros((*mask.shape, 3), np.uint8),
        world_xyz=world,
        camera_to_world=pose,
        up_direction=[0, 0, 1],
    )
    cloud = object_points(world, obs.valid_mask, mask, pose[:3, 2])
    name, fit = next(
        fit_features(
            ["base_center"],
            "supported_round_base",
            obs,
            mask,
            cloud,
            FitOptions(),
            lambda: None,
        )
    )
    assert name == "base_center"
    if view in {"resting", "small_platform"}:
        assert fit.accepted, fit.reason
        assert "base_contact_occluded" in fit.warnings
        np.testing.assert_allclose(fit.parameters["center"], base, atol=0.001)
    else:
        assert not fit.accepted and "center" not in fit.parameters
        if view == "open_gap":
            assert fit.reason == "base_support_gap_observed"


def test_noisy_tilted_faces_keep_two_distinct_levels():
    from toolkits.geometry.rgbd_parts import face_candidates

    rows, cols = np.indices((22, 29))
    x, y = (cols - 14) * 0.002, (rows - 10) * 0.002
    z = 0.05 + 0.18 * x + 0.0013 * np.sin(0.8 * cols) * np.cos(0.6 * rows)
    lower = np.stack((x, y, z), axis=-1).reshape(-1, 3)
    cloud = np.concatenate([lower, lower + [0.08, 0, 0.035]])
    faces = face_candidates(cloud, np.array([0.0, 0.0, 1.0]))
    heights = [support[:, 2].mean() for _, _, support in faces]
    np.testing.assert_allclose(heights, [0.085, 0.05], atol=0.001)


@pytest.mark.parametrize("stop_at", ["plane_fit", "measurement_complete"])
def test_cancelled_face_measurement_is_not_published_or_cached(
    tmp_path, monkeypatch, stop_at
):
    from rpent.tools.toolkit import ToolCancelled
    from toolkits.geometry import rgbd_parts

    rows, cols = np.indices((40, 40))
    face = (abs(rows - 20) < 10) & (abs(cols - 20) < 12)
    stopped = False

    def check():
        if stopped:
            raise ToolCancelled()

    state, tool = _mask_tool(
        tmp_path, face, cols * 0.001, rows * 0.001, face * 0.05, check_cancelled=check
    )
    owner, name = (
        (rgbd_parts, "_plane") if stop_at == "plane_fit" else (tool, "_measure")
    )
    original = getattr(owner, name)

    def stop_after(*args, **kwargs):
        nonlocal stopped
        assert not stopped, "started another fit after cancellation"
        result = original(*args, **kwargs)
        stopped = True
        return result

    def measure():
        return tool.geometry(
            "block",
            "front",
            object_prompt="block",
            required_geometry=["rectangle_center"],
            feature="upper_face",
        )

    with monkeypatch.context() as patch:
        patch.setattr(owner, name, stop_after)
        with pytest.raises(ToolCancelled):
            measure()
    assert not any(path.endswith(".json") for path in state.get(0).artifacts)
    stopped = False
    assert measure().data["status"] == "usable"


@pytest.mark.parametrize(
    (
        "point",
        "image_size",
        "hit_indices",
        "selected_index",
        "warning",
    ),
    [
        ((11.25, 3.75), (14, 12), [0], 0, None),
        ((3, 4), (14, 12), [2], 2, None),
        ((5, 4), (14, 12), [1, 2], 1, "molmo_point_hits_multiple_object_masks"),
        # A point in a hole in the mask still misses the object.
        ((10, 4), (14, 12), [], 0, "molmo_point_outside_object_masks"),
        (None, (14, 12), None, 0, "no_single_molmo_point"),
        ((11, 4), (12, 14), None, 0, "molmo_point_does_not_match_source_image"),
        ((14, 4), (14, 12), None, 0, "molmo_point_does_not_match_source_image"),
    ],
)
def test_geometry_selects_unique_molmo_hits_and_warns_on_sam_fallback(
    tmp_path, point, image_size, hit_indices, selected_index, warning
):
    shape = (12, 14)
    masks = []
    for x, score in ((9, 0.99), (4, 0.85), (2, 0.45)):
        mask = np.zeros(shape, bool)
        mask[2:7, x : x + 5] = True
        masks.append(Sam3Result(True, mask=mask, score=score))
    masks[0].mask[4, 10] = False
    state = EnvState(tmp_path)
    with state.record_step(state={}):
        pass
    rgb = np.zeros((*shape, 3), np.uint8)
    obs = GeometryObservation(
        "front",
        0,
        "table",
        rgb,
        world_xyz=np.ones((*shape, 3)),
        camera_to_world=np.eye(4),
        up_direction=[0, 0, 1],
    )

    stages = []

    def ground(image, query):
        assert stages == ["object"]
        assert image == obs.png_bytes()
        assert query == "the cup next to the bowl"
        stages.append("grounding")
        return MolmoResult(point is not None, point, image_size)

    def segment(image, *, text_prompt, min_score):
        assert text_prompt == "white cup" and min_score == 0.4
        assert image == obs.png_bytes()
        stages.append("object")
        return masks

    tool = GeometryTool(
        state=state,
        observations=SimpleNamespace(get_observation=lambda **_: obs),
        grounder=SimpleNamespace(ground=ground),
        segmenter=SimpleNamespace(segment_candidates=segment),
    )
    result = tool.geometry(
        "the cup next to the bowl",
        "front",
        object_prompt="  white cup  ",
        required_geometry=["disk_center"],
        feature="upper_face",
    ).data
    detail = state.load(result["geometry_artifact"], step=0)
    assert detail["calls"] == {
        "segment": 1,
        "grounding": 1,
        "back_project": 1,
    }
    attempt = detail["segment_attempts"][0]
    assert attempt.get("point_hit_indices") == hit_indices
    assert attempt.get("selected_index") == selected_index
    assert result["status"] == "insufficient_evidence"
    assert result["reason"] == "too_few_valid_object_points"
    assert result["missing_geometry"] == ["disk_center"]
    assert stages == ["object", "grounding"]
    assert detail["identity_evidence"]["object"]["accepted"] is (warning is None)
    assert attempt.get("reason") == warning
    expected_warnings = (
        [{"stage": "object", "reason": warning, "source_camera": "front"}]
        if warning
        else []
    )
    assert result.get("warnings", []) == expected_warnings
    if hit_indices is not None:
        assert attempt["point_rc"] == [round(point[1]), round(point[0])]
    np.testing.assert_array_equal(
        state.load(detail["object_mask_artifact"], step=0),
        masks[selected_index].mask,
    )


@pytest.mark.parametrize(
    ("outcome", "reason"),
    [
        ("valid", None),
        ("no_point", "no_single_molmo_point"),
        ("wrong_image", "molmo_point_does_not_match_source_image"),
        ("no_mask", "below threshold"),
        ("wrong_shape", "invalid_sam3_point_mask"),
        ("empty", "invalid_sam3_point_mask"),
        ("miss", "sam3_point_mask_misses_point"),
    ],
)
def test_geometry_point_prompt_recovers_only_an_aligned_mask(tmp_path, outcome, reason):
    rows, cols = np.indices((40, 60))
    mask = (abs(rows - 17) <= 9) & (abs(cols - 38) <= 12)
    xyz = np.stack((cols * 0.002, rows * 0.002, mask * 0.04), axis=-1)
    pose = np.diag([1.0, -1.0, -1.0, 1.0])
    pose[2, 3] = 1.0
    obs = GeometryObservation(
        "front",
        0,
        "table",
        np.zeros((40, 60, 3), np.uint8),
        world_xyz=xyz,
        camera_to_world=pose,
        up_direction=[0, 0, 1],
    )
    calls = []

    def candidates(image, *, text_prompt, min_score):
        assert image == obs.png_bytes() and text_prompt == "box"
        calls.append("text")
        return []

    def ground(image, query):
        assert image == obs.png_bytes() and query == "black box next to the cup"
        calls.append("ground")
        if outcome == "no_point":
            return MolmoResult(False)
        size = (40, 60) if outcome == "wrong_image" else (60, 40)
        return MolmoResult(True, (32.7, 12.2), size)

    def segment(image, *, point):
        assert image == obs.png_bytes() and point == [12, 33]
        calls.append("point")
        if outcome == "no_mask":
            return Sam3Result(False, reason="below threshold")
        proposed = mask.copy()
        if outcome == "wrong_shape":
            proposed = proposed.T
        elif outcome == "empty":
            proposed[:] = False
        elif outcome == "miss":
            proposed[tuple(point)] = False
        return Sam3Result(True, score=0.9, mask=proposed)

    state = EnvState(tmp_path)
    with state.record_step(state={}):
        pass
    tool = GeometryTool(
        state=state,
        observations=SimpleNamespace(get_observation=lambda **_: obs),
        grounder=LocalPointAgent(SimpleNamespace(ground=ground)),
        segmenter=SimpleNamespace(segment_candidates=candidates, segment=segment),
    )
    measurement = tool.geometry(
        "black box next to the cup",
        "front",
        object_prompt="box",
        required_geometry=["rectangle_center"],
        feature="upper_face",
    )
    result = measurement.data
    prompted = outcome not in {"no_point", "wrong_image"}
    assert calls == (["text", "ground", "point"] if prompted else ["text", "ground"])
    detail = state.load(result["geometry_artifact"], step=0)
    assert detail["calls"]["segment"] == 1 + prompted
    assert detail["calls"]["grounding"] == 1
    assert [a["mode"] for a in detail["segment_attempts"]] == (
        ["text", "point"] if prompted else ["text"]
    )
    field = result["geometry"]["rectangle_center"]
    if reason:
        assert (
            result["status"] == "insufficient_evidence" and result["reason"] == reason
        )
        assert not field["fit_accepted"] and "value" not in field
        assert "object_mask_artifact" not in detail
        assert not measurement.images
    else:
        assert result["status"] == "usable" and field["fit_accepted"]
        assert len(measurement.images) == 1
        np.testing.assert_allclose(field["value"], xyz[17, 38], atol=0.002)
        assert detail["identity_evidence"]["object"]["method"] == "sam3_point_prompt"
        np.testing.assert_array_equal(
            state.load(detail["object_mask_artifact"], step=0), mask
        )
