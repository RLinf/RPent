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

"""Tests for segmenting a recorded video into map builds."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from rpent.tools.build_map import (
    build_maps,
    build_maps_from_video,
    estimated_segments,
    iter_video_frames,
    segment_names,
)
from toolkits.mapping.client import MapBuildResult, MapFrame, MapSessionStatus


class _FakeMapClient:
    """Record sessions and frames without a mapping service."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.sessions: list[dict[str, Any]] = []
        self.stopped = 0
        self._current: dict[str, Any] | None = None

    def start_session(self, *, map_name: str) -> MapSessionStatus:
        self._current = {"name": map_name, "frames": []}
        self.sessions.append(self._current)
        return MapSessionStatus(session_id=map_name, running=True, frames=0)

    def push_frame(
        self, image: Any, *, pose: list[float] | None, timestamp_s: float
    ) -> MapSessionStatus:
        assert self._current is not None
        assert pose is None
        self._current["frames"].append((np.asarray(image), timestamp_s))
        return MapSessionStatus(
            session_id=str(self._current["name"]),
            running=True,
            frames=len(self._current["frames"]),
        )

    def build(self) -> MapBuildResult:
        assert self._current is not None
        path = self.root / str(self._current["name"])
        path.mkdir(parents=True, exist_ok=True)
        frames = len(self._current["frames"])
        result = MapBuildResult(
            map_path=str(path),
            frames=frames,
            metadata={
                "backend": "frame_archive",
                "has_depth": False,
                "frames": frames,
            },
        )
        self._current = None
        return result

    def load_frame(self, map_path: str, index: int) -> MapFrame:
        import io

        from PIL import Image

        buffer = io.BytesIO()
        Image.new("RGB", (4, 4), (10, 20, 30)).save(buffer, format="PNG")
        return MapFrame(
            index=index,
            image=buffer.getvalue(),
            source_size=(4, 4),
            timestamp_s=index * 0.2,
            pushed_base_pose=None,
        )

    def stop_session(self) -> None:
        self.stopped += 1
        self._current = None


def _frames(count: int) -> list[tuple[np.ndarray, float]]:
    return [
        (np.full((4, 4, 3), index, dtype=np.uint8), index * 0.2)
        for index in range(count)
    ]


def test_long_recording_numbers_every_map(tmp_path: Path) -> None:
    client = _FakeMapClient(tmp_path)

    maps = build_maps(
        _frames(5),
        client,  # type: ignore[arg-type]
        map_name="workspace",
        segment_frames=2,
        segments=3,
    )

    assert [session["name"] for session in client.sessions] == [
        "workspace_00",
        "workspace_01",
        "workspace_02",
    ]
    assert [len(session["frames"]) for session in client.sessions] == [2, 2, 1]
    assert [entry["map_path"] for entry in maps] == [
        str(tmp_path / "workspace_00"),
        str(tmp_path / "workspace_01"),
        str(tmp_path / "workspace_02"),
    ]
    assert [entry["frames_pushed"] for entry in maps] == [2, 2, 1]


def test_single_segment_keeps_the_requested_name(tmp_path: Path) -> None:
    client = _FakeMapClient(tmp_path)

    maps = build_maps(
        _frames(3),
        client,  # type: ignore[arg-type]
        map_name="workspace",
        segment_frames=8,
        segments=1,
    )

    assert [session["name"] for session in client.sessions] == ["workspace"]
    assert maps[0]["frames"] == 3
    assert [timestamp for _, timestamp in client.sessions[0]["frames"]] == [
        0.0,
        0.2,
        0.4,
    ]


def test_unknown_length_numbers_every_map(tmp_path: Path) -> None:
    client = _FakeMapClient(tmp_path)

    build_maps(
        _frames(2),
        client,  # type: ignore[arg-type]
        map_name="workspace",
        segment_frames=8,
        segments=None,
    )

    assert [session["name"] for session in client.sessions] == ["workspace_00"]


def test_empty_input_is_rejected(tmp_path: Path) -> None:
    client = _FakeMapClient(tmp_path)

    with pytest.raises(ValueError, match="no sampled frames"):
        build_maps(
            [],
            client,  # type: ignore[arg-type]
            map_name="workspace",
            segments=1,
        )
    assert client.sessions == []
    assert client.stopped == 0


def test_labels_require_a_perception_client(tmp_path: Path) -> None:
    client = _FakeMapClient(tmp_path)

    with pytest.raises(ValueError, match="perception client"):
        build_maps(
            _frames(2),
            client,  # type: ignore[arg-type]
            map_name="workspace",
            segments=1,
            labels=["chair"],
        )


def test_grounding_failure_keeps_the_built_map(tmp_path: Path) -> None:
    client = _FakeMapClient(tmp_path)

    class _BrokenPerception:
        def detect_segment(self, *args: Any, **kwargs: Any) -> Any:
            raise RuntimeError("perception service is down")

    maps = build_maps(
        _frames(2),
        client,  # type: ignore[arg-type]
        map_name="workspace",
        segments=1,
        labels=["chair"],
        perception_client=_BrokenPerception(),  # type: ignore[arg-type]
    )

    assert maps[0]["frames"] == 2
    assert "perception service is down" in maps[0]["annotation_error"]


def test_segment_names() -> None:
    assert segment_names("workspace", 1) == ["workspace"]
    assert segment_names("workspace", 2) == ["workspace_00", "workspace_01"]
    assert segment_names("workspace", None) == ["workspace_00"]


def test_video_sampling_and_segments(tmp_path: Path) -> None:
    cv2 = pytest.importorskip("cv2")
    path = tmp_path / "walk.avi"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 30.0, (16, 16))
    if not writer.isOpened():
        pytest.skip("OpenCV cannot write test videos here")
    try:
        for index in range(12):
            writer.write(np.full((16, 16, 3), index * 10, dtype=np.uint8))
    finally:
        writer.release()

    sampled = list(iter_video_frames(path, fps=10.0))
    assert len(sampled) == 4
    assert [round(timestamp, 3) for _, timestamp in sampled] == [0.0, 0.1, 0.2, 0.3]
    assert sampled[0][0].shape == (16, 16, 3)

    client = _FakeMapClient(tmp_path / "maps")
    maps = build_maps_from_video(
        path,
        client,  # type: ignore[arg-type]
        map_name="walk",
        fps=10.0,
        segment_frames=2,
    )

    assert [session["name"] for session in client.sessions] == ["walk_00", "walk_01"]
    assert [entry["frames_pushed"] for entry in maps] == [2, 2]


def test_unreadable_video_is_rejected(tmp_path: Path) -> None:
    pytest.importorskip("cv2")
    path = tmp_path / "empty.avi"
    path.write_bytes(b"")
    with pytest.raises(ValueError, match="could not open video"):
        estimated_segments(path, fps=5.0, segment_frames=2)
