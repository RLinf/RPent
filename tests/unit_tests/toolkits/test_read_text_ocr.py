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

"""Offline tests for OCR image selection, word boxes and backend failures."""

import io
import subprocess
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from toolkits.observation import GeometryObservation
from toolkits.read_text_ocr import read_text_ocr

HEADER = "level\tpage_num\tblock_num\tpar_num\tline_num\tword_num\tleft\ttop\twidth\theight\tconf\ttext\n"


def _word(text, x=0, y=0, width=10, height=10, confidence=90, line=1):
    return f"5\t1\t1\t1\t{line}\t1\t{x}\t{y}\t{width}\t{height}\t{confidence}\t{text}\n"


def _observation(*, shape=(100, 160)):
    rgb = np.zeros((*shape, 3), dtype=np.uint8)
    rgb[8:68, 20:100] = [80, 120, 180]
    return GeometryObservation(
        camera="panel",
        step=5,
        world_frame_id="recording",
        rgb=rgb,
        world_xyz=np.zeros(rgb.shape, dtype=float),
    )


def _ocr(observation, **arguments):
    return read_text_ocr(
        camera="panel",
        observations=SimpleNamespace(get_observation=lambda **_: observation),
        **arguments,
    )


def test_preview_crop_and_upscaled_word_boxes_return_native_coordinates():
    obs = _observation()
    source = obs.rgb.copy()
    tsv = (
        HEADER
        + "1\t1\t0\t0\t0\t0\t0\t0\t240\t180\t-1\t\n"
        + _word("TEMP", 11, 7, 20, 21, 92.5)
        + _word("27", 235, 7, 20, 21, 80)
        + _word("noise", 50, 30, confidence=34)
        + _word("MIN", -4, 90, 22, 18, 75, line=2)
    )

    def recognize(png, layout):
        assert layout == "sparse"
        with Image.open(io.BytesIO(png)) as image:
            assert image.format == "PNG" and image.size == (240, 180)
            np.testing.assert_array_equal(
                np.asarray(image), np.broadcast_to([80, 120, 180], (180, 240, 3))
            )
        return tsv

    result = _ocr(
        obs,
        region={"image_shape": [50, 80], "box_rc": [4, 10, 34, 50]},
        layout="sparse",
        upscale=3,
        min_confidence=0.75,
        recognize=recognize,
    )
    assert result.data == {
        "status": "read",
        "camera": "panel",
        "step": 5,
        "image_shape": [100, 160],
        "source_box_rc": [8, 20, 68, 100],
        "text": "TEMP 27\nMIN",
        "words": [
            {"text": "TEMP", "confidence": 0.925, "box_rc": [10, 23, 18, 31]},
            {"text": "27", "confidence": 0.8, "box_rc": [10, 98, 18, 100]},
            {"text": "MIN", "confidence": 0.75, "box_rc": [38, 20, 44, 26]},
        ],
        "truncated": False,
    }
    assert result.images == []
    np.testing.assert_array_equal(obs.rgb, source)


def test_full_frame_and_no_confident_text():
    def recognize(png, layout):
        assert layout == "block"
        with Image.open(io.BytesIO(png)) as image:
            assert image.size == (160, 100)
        return HEADER + _word("uncertain", confidence=49)

    result = _ocr(_observation(), upscale=1, recognize=recognize).data
    assert result["status"] == "no_text"
    assert result["source_box_rc"] == [0, 0, 100, 160]
    assert result["text"] == "" and result["words"] == []


def test_reports_truncation_and_keeps_initial_words_in_order():
    data = _ocr(
        _observation(),
        recognize=lambda *_: HEADER + "".join(_word(str(i)) for i in range(70)),
    ).data
    assert data["truncated"] is True
    assert [word["text"] for word in data["words"]] == [str(i) for i in range(64)]
    assert data["text"].split()[-1] == "63"


@pytest.mark.parametrize(
    ("arguments", "error"),
    [
        ({"region": {"image_shape": [50, 80], "box_rc": [0, 0, 51, 80]}}, "box_rc"),
        (
            {"region": {"image_shape": [50, 82], "box_rc": [0, 0, 50, 80]}},
            "aspect ratio",
        ),
        ({"step": 3}, "different camera or step"),
    ],
)
def test_rejects_bad_selection_before_starting_ocr(arguments, error):
    def recognize(*_):
        pytest.fail("OCR must not process a mismatched image selection")

    result = _ocr(_observation(), recognize=recognize, **arguments)
    assert result.is_error and error in result.data["error"]


def test_limits_processed_image_size():
    result = _ocr(_observation(shape=(6, 1025)))
    assert result.is_error and "2048 pixels" in result.data["error"]


@pytest.mark.parametrize(
    ("tsv", "error"),
    [
        ("words only", "OCR TSV"),
        (HEADER + _word("TEMP", confidence="unknown"), "numeric confidence"),
    ],
)
def test_invalid_backend_response_is_reported(tsv, error):
    result = _ocr(_observation(), recognize=lambda *_: tsv)
    assert result.is_error and error in result.data["error"]


def test_missing_executable_returns_actionable_error(monkeypatch):
    monkeypatch.setattr("toolkits.read_text_ocr.shutil.which", lambda _: None)
    result = _ocr(_observation())
    assert result.is_error and "English language data" in result.data["error"]


@pytest.mark.parametrize("failure", ["timeout", "exit"])
def test_local_engine_errors_are_returned_without_text(monkeypatch, failure):
    monkeypatch.setattr(
        "toolkits.read_text_ocr.shutil.which", lambda _: "/local/tesseract"
    )

    def run(argv, *, input, capture_output, timeout, check):
        assert argv == [
            "/local/tesseract",
            "stdin",
            "stdout",
            "-l",
            "eng",
            "--psm",
            "7",
            "tsv",
        ]
        with Image.open(io.BytesIO(input)) as image:
            assert image.format == "PNG"
        assert capture_output and timeout == 10 and check is False
        if failure == "timeout":
            raise subprocess.TimeoutExpired(argv, timeout)
        return subprocess.CompletedProcess(argv, 1, b"", b"missing eng.traineddata")

    monkeypatch.setattr("toolkits.read_text_ocr.subprocess.run", run)
    result = _ocr(_observation(), layout="line")
    expected = "timed out" if failure == "timeout" else "missing eng.traineddata"
    assert result.is_error and expected in result.data["error"]
    assert "words" not in result.data


def test_truncated_word_record_is_a_backend_error():
    data = _ocr(
        _observation(), recognize=lambda *_: HEADER + "5\t1\t1\t1\t1\t1\t2\t4\n"
    ).data
    assert data["status"] == "error"
    assert "complete word record" in data["error"]


def test_real_local_tesseract_reads_only_the_requested_roi():
    import shutil

    from PIL import ImageDraw, ImageFont

    executable = shutil.which("tesseract")
    if executable is None:
        pytest.skip("Tesseract is not installed")
    languages = subprocess.run(
        [executable, "--list-langs"],
        capture_output=True,
        text=True,
        timeout=10,
        check=True,
    ).stdout.splitlines()
    if "eng" not in languages:
        pytest.skip("English Tesseract language data is not installed")
    image = Image.new("RGB", (720, 360), "white")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default(size=54)
    draw.text((120, 93), "TEMP 27", fill="black", font=font)
    draw.text((120, 215), "MODE 42", fill="black", font=font)
    rgb = np.asarray(image)
    obs = GeometryObservation(
        "panel", 5, "recording", rgb, world_xyz=np.zeros(rgb.shape)
    )
    data = _ocr(
        obs,
        layout="line",
        region={"image_shape": [360, 720], "box_rc": [70, 90, 180, 430]},
    ).data
    assert data["status"] == "read" and data["text"] == "TEMP 27"
    assert [word["text"] for word in data["words"]] == ["TEMP", "27"]
    for word in data["words"]:
        top, left, bottom, right = word["box_rc"]
        assert 70 <= top < bottom <= 180 and 90 <= left < right <= 430
        assert 0.5 <= word["confidence"] <= 1
    np.testing.assert_array_equal(obs.rgb, rgb)
