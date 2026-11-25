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

"""Offline contracts for crop images and their full-frame pixel coordinates."""

import io
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from toolkits.crop_zoom import crop_zoom
from toolkits.observation import GeometryObservation


@pytest.fixture
def observation():
    rows, cols = np.indices((12, 18), dtype=np.uint8)
    return GeometryObservation(
        camera="wrist",
        step=4,
        world_frame_id="table",
        rgb=np.stack((rows, cols, rows + cols), axis=-1),
        depth_m=np.ones(rows.shape),
        intrinsics=np.eye(3),
        camera_to_world=np.eye(4),
    )


def _crop(observation, **overrides):
    arguments = {"camera": "wrist", "image_shape": [12, 18], "box_rc": [2, 3, 8, 13]}
    return crop_zoom(
        **(arguments | overrides),
        observations=SimpleNamespace(get_observation=lambda **_: observation),
    )


def _decode(result):
    (png,) = result.images
    with Image.open(io.BytesIO(png)) as image:
        assert image.format == "PNG"
        return np.asarray(image)


def test_zoom_preserves_native_pixels_and_resolves_latest_step(observation):
    result = _crop(observation, box_rc=[2, 3, 12, 18], zoom=3)
    assert result.data == {
        "status": "cropped",
        "camera": "wrist",
        "step": 4,
        "image_shape": [12, 18],
        "source_box_rc": [2, 3, 12, 18],
        "crop_shape": [30, 45],
        "zoom": 3,
        "grid_step": None,
        "marks": [],
    }
    expected = observation.rgb[2:12, 3:18].repeat(3, axis=0).repeat(3, axis=1)
    np.testing.assert_array_equal(_decode(result), expected)


def test_grid_and_numbered_marks_on_a_resized_full_frame(observation):
    original = observation.rgb.copy()
    arguments = {"image_shape": [8, 12], "box_rc": [1, 3, 8, 12], "zoom": 3}
    clean = _decode(_crop(observation, **arguments))
    grid = _decode(_crop(observation, **arguments, grid_step=4))
    result = _crop(observation, **arguments, grid_step=4, marks_rc=[[1, 3], [7, 11]])
    annotated = _decode(result)
    assert result.data["source_box_rc"] == [1, 4, 12, 18]
    assert result.data["crop_shape"] == [33, 42]
    assert result.data["marks"] == [
        {"number": 1, "source_point_rc": [2, 5], "crop_point_rc": [4, 4]},
        {"number": 2, "source_point_rc": [11, 17], "crop_point_rc": [31, 40]},
    ]
    assert np.all(grid[12, 7] > clean[12, 7])
    np.testing.assert_array_equal(grid[13, 7], clean[13, 7])
    for mark in result.data["marks"]:
        row, col = mark["crop_point_rc"]
        assert (annotated[row, col] != grid[row, col]).any()
        np.testing.assert_array_equal(
            clean[row, col], original[tuple(mark["source_point_rc"])]
        )
    np.testing.assert_array_equal(observation.rgb, original)


def test_grid_is_visible_on_a_white_surface():
    obs = GeometryObservation(
        camera="wrist",
        step=4,
        world_frame_id="table",
        rgb=np.full((12, 18, 3), 255, dtype=np.uint8),
        world_xyz=np.zeros((12, 18, 3)),
    )
    grid = _decode(_crop(obs, grid_step=4))
    assert grid[8, 7, 0] < 255
    assert (grid[9, 7] == 255).all()


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"box_rc": [2, 3, 13, 18]}, "box_rc"),
        ({"box_rc": [2, 3, 2, 4]}, "box_rc"),
        ({"image_shape": [12, 19]}, "original aspect ratio"),
        ({"marks_rc": [[2, 13]]}, "inside box_rc"),
        ({"camera": "other"}, "different camera or step"),
        ({"step": 3}, "different camera or step"),
    ],
)
def test_invalid_selection_or_source_returns_no_image(observation, overrides, message):
    result = _crop(observation, **overrides)
    assert result.data["status"] == "error"
    assert message in result.data["error"]
    assert result.images == []


def test_crop_caps_output_size_at_native_resolution():
    obs = GeometryObservation(
        camera="wrist",
        step=4,
        world_frame_id="table",
        rgb=np.zeros((2, 1025, 3), dtype=np.uint8),
        world_xyz=np.zeros((2, 1025, 3)),
    )
    arguments = {"image_shape": [2, 1025], "zoom": 2}
    fits = _crop(obs, **arguments, box_rc=[0, 0, 2, 512])
    assert _decode(fits).shape == (4, 1024, 3)
    oversized = _crop(obs, **arguments, box_rc=[0, 0, 2, 513])
    assert "maximum edge of 1024 pixels" in oversized.data["error"]
    assert oversized.images == []


def test_adjacent_mark_centres_stay_visible_at_zoom_one(observation):
    result = _crop(observation, zoom=1, marks_rc=[[3, 4], [3, 7]])
    image = _decode(result)
    for mark in result.data["marks"]:
        np.testing.assert_array_equal(
            image[tuple(mark["crop_point_rc"])], [255, 220, 0]
        )


def test_two_preview_marks_at_the_same_native_pixel_are_rejected(observation):
    result = _crop(
        observation,
        image_shape=[24, 36],
        box_rc=[0, 0, 24, 36],
        marks_rc=[[4, 6], [5, 7]],
    )
    assert result.is_error and "same native pixel" in result.data["error"]
    assert result.images == []


@pytest.mark.parametrize(
    ("size", "marks"),
    [(20, [[18, 18], [18, 19], [19, 19]]), (72, [[6 * i, 6 * i] for i in range(11)])],
)
def test_numbered_labels_fit_edges_without_overlap(size, marks):
    rgb = np.full((size, size, 3), 255, dtype=np.uint8)
    obs = GeometryObservation("wrist", 4, "table", rgb, world_xyz=np.zeros(rgb.shape))
    args = {
        "image_shape": [size, size],
        "box_rc": [0, 0, size, size],
        "marks_rc": marks,
        "zoom": 1,
    }
    result = _crop(obs, **args)
    assert result == _crop(obs, **args)
    image = _decode(result)
    boxes = result.data["label_boxes_rc"]
    assert len(boxes) == len(marks)
    assert sum(box is not None for box in boxes) >= 2
    if size == 72:
        assert all(box is not None for box in boxes)
        assert boxes[9][3] - boxes[9][1] > boxes[0][3] - boxes[0][1]
    drawn = []
    for box in (box for box in boxes if box is not None):
        top, left, bottom, right = box
        assert 0 <= top < bottom <= size and 0 <= left < right <= size
        assert not any(top <= r < bottom and left <= c < right for r, c in marks)
        assert all(
            bottom <= t or b <= top or right <= l or q <= left for t, l, b, q in drawn
        )
        assert np.all(image[top:bottom, left:right] == [0, 0, 0], axis=-1).any()
        drawn.append(box)


def test_tiny_roi_retains_dots_and_reports_omitted_numbers(observation):
    result = _crop(observation, zoom=1, box_rc=[4, 5, 5, 6], marks_rc=[[4, 5]])
    assert result.data["status"] == "cropped"
    assert result.data["label_boxes_rc"] == [None]
    assert "zoom" in result.data["label_hint"]
    assert result.data["marks"][0]["source_point_rc"] == [4, 5]
    np.testing.assert_array_equal(_decode(result)[0, 0], [255, 220, 0])
