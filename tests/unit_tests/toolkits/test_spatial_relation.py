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

"""Offline contracts for metric point and segmentation relations."""

from types import SimpleNamespace

import numpy as np
import pytest

from rpent.robots.components.sam3_client import Sam3Result
from rpent.session import EnvState
from toolkits.observation import GeometryObservation
from toolkits.segment import SegmentTool
from toolkits.spatial_relation import spatial_relation


def _view(
    camera="overhead",
    *,
    step=0,
    frame="table",
    xyz=(0, 0, 0),
    shape=(4, 5),
    up=(0, 1, 0),
    valid_mask=None,
):
    return GeometryObservation(
        camera=camera,
        step=step,
        world_frame_id=frame,
        rgb=np.zeros((*shape, 3), dtype=np.uint8),
        world_xyz=np.broadcast_to(xyz, (*shape, 3)),
        up_direction=up,
        valid_mask=valid_mask,
    )


def _provider(*views):
    by_camera = {view.camera: view for view in views}
    return SimpleNamespace(get_observation=lambda *, camera, step: by_camera[camera])


def _point(xyz, frame="table"):
    return {"xyz": xyz, "world_frame_id": frame}


def _measure(state, observations, **options):
    arguments = {
        "camera": "overhead",
        "reference": _point([0, 0, 0]),
        "target": _point([3, 2, -4]),
        "forward_xyz": [1, 4, 0],
    }
    return spatial_relation(
        **(arguments | options), observations=observations, state=state
    )


def _segment(state, observations, camera, mask):
    segmenter = SimpleNamespace(
        segment=lambda *args, **kwargs: Sam3Result(found=True, mask=mask, score=0.9)
    )
    segment = SegmentTool(observations=observations, state=state, segmenter=segmenter)
    result = segment.segment(camera=camera, prompt="object").data
    assert result["found"]
    return {"segment_artifact": result["segment_artifact"]}, result["mask_artifact"]


def test_point_distance_and_signed_offsets_follow_a_non_z_up_frame(tmp_path):
    observations = _provider(_view(step=4))
    state = EnvState(tmp_path)
    result = _measure(state, observations).data

    assert result["status"] == "measured"
    assert result["step"] == 4 and result["world_frame_id"] == "table"
    assert result["units"] == "m"
    assert result["distance_m"] == pytest.approx(np.sqrt(29))
    assert result["forward_delta_m"] == 3
    assert result["height_delta_m"] == 2
    assert result["right_delta_m"] == -4
    assert result["forward_xyz"] == [1, 0, 0]
    assert result["up_xyz"] == [0, 1, 0]
    assert result["right_xyz"] == [0, 0, 1]
    assert result["directions"] == {
        "front_back": "front",
        "left_right": "left",
        "above_below": "above",
    }

    reverse = _measure(
        state,
        observations,
        reference=_point([3, 2, -4]),
        target=_point([0, 0, 0]),
    ).data
    assert reverse["distance_m"] == result["distance_m"]
    assert reverse["directions"] == {
        "front_back": "back",
        "left_right": "right",
        "above_below": "below",
    }


def test_tolerance_changes_direction_labels_without_rounding_metrics(tmp_path):
    data = _measure(
        EnvState(tmp_path),
        _provider(_view()),
        target=_point([0, 0.0005, -0.001]),
    ).data
    assert set(data["directions"].values()) == {"aligned"}
    assert data["height_delta_m"] == 0.0005
    assert data["right_delta_m"] == -0.001
    assert data["distance_m"] > 0


@pytest.mark.parametrize(
    ("options", "message"),
    [
        ({"target": _point([3, 2, -4], "other")}, "Expected world frame"),
        ({"forward_xyz": [0, 2, 0]}, "nonzero horizontal component"),
        ({"forward_xyz": [0, 0, 0]}, "nonzero horizontal component"),
        ({"step": 7}, "different camera or step"),
    ],
)
def test_invalid_frame_axis_or_step_returns_no_metrics(tmp_path, options, message):
    result = _measure(EnvState(tmp_path), _provider(_view()), **options)
    assert result.is_error
    assert message in result.data["error"]
    assert "distance_m" not in result.data


def test_missing_up_direction_requests_an_observation(tmp_path):
    result = _measure(EnvState(tmp_path), _provider(_view(up=None)))
    assert result.data["status"] == "need_observation"
    assert result.data["reason"] == "missing_up_direction"
    assert "height_delta_m" not in result.data


def test_masks_use_valid_visible_points_from_their_own_cameras(tmp_path):
    state = EnvState(tmp_path)
    with state.record_step(state={}) as step:
        pass
    a = np.zeros((4, 5, 3))
    a[:3, :4] = [1, 2, 3]
    a[0, 0] = [60, 60, 60]
    valid = np.ones((4, 5), dtype=bool)
    valid[0, 0] = False
    b = np.zeros((6, 7, 3))
    b[1:5, 1:5] = [4, 6, 3]
    overhead = _view(step=step, xyz=a, valid_mask=valid)
    wrist = _view("wrist", step=step, xyz=b, shape=(6, 7))
    observations = _provider(overhead, wrist)
    a_mask = np.zeros((4, 5), dtype=bool)
    a_mask[:3, :4] = True
    b_mask = np.zeros((6, 7), dtype=bool)
    b_mask[1:5, 1:5] = True
    reference, _ = _segment(state, observations, "overhead", a_mask)
    target, _ = _segment(state, observations, "wrist", b_mask)
    recorded = set(state.get(step).artifacts)

    result = _measure(state, observations, reference=reference, target=target).data

    assert result["status"] == "measured"
    assert result["reference"]["xyz"] == [1, 2, 3]
    assert result["reference"]["n_valid"] == 11
    assert result["target"] == {
        "camera": "wrist",
        "xyz": [4, 6, 3],
        "definition": "visible_mask_coordinate_median",
        "segment_artifact": target["segment_artifact"],
        "n_valid": 16,
    }
    assert result["distance_m"] == 5
    assert result["height_delta_m"] == 4
    assert state.latest_step == step and state.get(step).artifacts == recorded


@pytest.mark.parametrize("count", [0, 9])
def test_empty_or_sparse_masks_request_an_observation(tmp_path, count):
    state = EnvState(tmp_path)
    with state.record_step(state={}) as step:
        pass
    observations = _provider(_view(step=step))
    target, mask_name = _segment(
        state, observations, "overhead", np.ones((4, 5), dtype=bool)
    )
    mask = np.zeros((4, 5), dtype=bool)
    mask.flat[:count] = True
    state.save(mask_name, mask, step=step)

    result = _measure(state, observations, target=target).data
    assert result["status"] == "need_observation"
    assert result["reason"] == "insufficient_valid_mask_points"
    assert result["target"]["n_valid"] == count and result["target"]["xyz"] is None
    assert "distance_m" not in result


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"source_step": 2}, "different source step"),
        ({"world_frame_id": "other"}, "Expected world frame"),
        ({"image_shape": [5, 4]}, "does not match its recorded image shape"),
        ({"mask_artifact": "wrong.npz"}, "Expected the saved mask"),
    ],
)
def test_segmentation_provenance_must_match(tmp_path, change, message):
    state = EnvState(tmp_path)
    with state.record_step(state={}) as step:
        pass
    observations = _provider(_view(step=step))
    target, _ = _segment(state, observations, "overhead", np.ones((4, 5), dtype=bool))
    artifact = target["segment_artifact"]
    state.save(artifact, state.load(artifact, step=step) | change, step=step)

    result = _measure(state, observations, target=target)
    assert result.is_error
    assert message in result.data["error"]


def test_metric_mask_median_with_anisotropic_calibration_and_rigid_pose(tmp_path):
    state = EnvState(tmp_path)
    with state.record_step(state={}) as step:
        pass
    depth = np.full((5, 8), 2.0)
    depth[1, 1] = float("nan")
    depth[3, 5] = float("inf")
    depth[1, 2] = 0
    depth[3, 4] = -1
    mask = np.zeros(depth.shape, dtype=bool)
    mask[1:4, 1:6] = True
    with np.errstate(invalid="ignore"):
        view = GeometryObservation(
            camera="overhead",
            step=step,
            world_frame_id="table",
            rgb=np.zeros((*depth.shape, 3), np.uint8),
            depth_m=depth,
            intrinsics=np.array([[4, 0, 2], [0, 8, 1], [0, 0, 1]]),
            camera_to_world=np.array(
                [
                    [0, -1, 0, 10],
                    [1, 0, 0, 20],
                    [0, 0, 1, 30],
                    [0, 0, 0, 1],
                ]
            ),
            up_direction=[0, 0, 1],
        )
    provider = _provider(view)
    target, mask_name = _segment(state, provider, "overhead", mask)
    reference = _point([7.75, 17.5, 29])
    # The symmetric 11 valid pixels have median column 3 and row 2.
    # The optical point is (0.5, 0.25, 2), mapped by (-y, x, z) + (10,20,30).
    result = _measure(
        state, provider, reference=reference, target=target, forward_xyz=[1, 0, 0]
    ).data
    assert result["target"]["n_valid"] == 11
    assert result["target"]["xyz"] == [9.75, 20.5, 32]
    assert result["distance_m"] == pytest.approx(22**0.5)
    assert result["height_delta_m"] == 3
    assert result["forward_delta_m"] == 2
    assert result["right_delta_m"] == -3
    reverse = _measure(
        state, provider, reference=target, target=reference, forward_xyz=[1, 0, 0]
    ).data
    assert reverse["distance_m"] == result["distance_m"]
    assert reverse["directions"] == {
        "front_back": "back",
        "left_right": "right",
        "above_below": "below",
    }
    np.testing.assert_array_equal(view.depth_m, depth)
    np.testing.assert_array_equal(state.load(mask_name, step=step), mask)
