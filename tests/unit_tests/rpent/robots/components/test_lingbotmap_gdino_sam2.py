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

"""Offline tests for the LingBot-Map and Grounding DINO + SAM 2 services."""

from __future__ import annotations

import base64
import io
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from PIL import Image

from rpent.robots.components.gdino_sam2_client import (
    GdinoSam2Client,
)
from rpent.robots.components.lingbotmap_client import LingBotMapClient
from rpent.robots.components.lingbotmap_server import (
    FrameArchiveBackend,
    LingBotMapFacade,
    build_lingbot_backend,
)
from rpent.utils.rpc import RpcClient


class _LocalRpcClient(RpcClient):
    """Dispatch calls directly into a facade's registered RPC methods."""

    def __init__(self, facade: Any) -> None:
        self._facade = facade

    def call(
        self, method: str, args: tuple = (), kwargs: dict | None = None, **_: Any
    ) -> Any:
        handler = getattr(self._facade, "_rpc")[method]
        return handler(*(args or ()), **(kwargs or {}))


def _png_bytes(array: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    Image.fromarray(np.asarray(array, dtype=np.uint8)).save(buffer, format="PNG")
    return buffer.getvalue()


def _frame(fill: int = 128) -> np.ndarray:
    return np.full((24, 32, 3), fill, dtype=np.uint8)


@pytest.fixture
def map_client(tmp_path: Path) -> LingBotMapClient:
    facade = LingBotMapFacade(FrameArchiveBackend(str(tmp_path / "maps")))
    return LingBotMapClient(_LocalRpcClient(facade))


class TestLingBotMapService:
    def test_start_push_build_roundtrip(self, map_client: LingBotMapClient) -> None:
        status = map_client.start_session(map_name="office")
        assert status.running and status.frames == 0

        for index in range(3):
            pushed = map_client.push_frame(
                _frame(index * 30),
                pose=[0.0, 0.0, 0.1 * index, 0, 0, 0],
                timestamp_s=1.0 + index,
            )
            assert pushed.frames == index + 1

        built = map_client.build()
        assert built.frames == 3
        assert (Path(built.map_path) / "map.json").is_file()
        assert len(list((Path(built.map_path) / "frames").glob("*.png"))) == 3
        assert map_client.session_id is None

    def test_push_without_pose_estimates_from_frames(
        self, map_client: LingBotMapClient, tmp_path: Path
    ) -> None:
        map_client.start_session(map_name="no_odom")
        map_client.push_frame(_frame(), pose=None, timestamp_s=1.0)
        built = map_client.build()
        import json

        poses = json.loads((Path(built.map_path) / "poses.json").read_text())
        assert poses[0]["pose"] is None

    def test_build_refuses_empty_session(self, map_client: LingBotMapClient) -> None:
        map_client.start_session(map_name="empty")
        with pytest.raises(RuntimeError, match="no frames"):
            map_client.build()

    def test_push_frame_rejects_bad_pose(self, map_client: LingBotMapClient) -> None:
        map_client.start_session(map_name="bad_pose")
        with pytest.raises(ValueError, match="flat 6-vector"):
            map_client.push_frame(_frame(), pose=[1.0, 2.0], timestamp_s=1.0)
        with pytest.raises(ValueError, match="finite"):
            map_client.push_frame(_frame(), pose=[float("nan")] * 6, timestamp_s=1.0)

    def test_client_blocks_second_session_until_resolved(
        self, map_client: LingBotMapClient
    ) -> None:
        map_client.start_session(map_name="one")
        with pytest.raises(RuntimeError, match="already active"):
            map_client.start_session(map_name="two")
        assert map_client.session_id is not None
        map_client.stop_session()
        assert map_client.session_id is None
        status = map_client.start_session(map_name="two")
        assert status.running

    def test_push_without_session_is_refused(
        self, map_client: LingBotMapClient
    ) -> None:
        with pytest.raises(RuntimeError, match="no active mapping session"):
            map_client.push_frame(_frame(), pose=None, timestamp_s=1.0)

    def test_load_metadata_reads_built_map(self, map_client: LingBotMapClient) -> None:
        map_client.start_session(map_name="meta")
        map_client.push_frame(_frame(), pose=None, timestamp_s=1.0)
        built = map_client.build()
        metadata = map_client.load_map_metadata(built.map_path)
        assert metadata["map_name"] == "meta"
        assert metadata["frames"] == 1

    def test_build_writes_version_two_contract(
        self, map_client: LingBotMapClient
    ) -> None:
        map_client.start_session(map_name="contract")
        map_client.push_frame(
            _frame(10), pose=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0], timestamp_s=1.0
        )
        map_client.push_frame(
            _frame(20), pose=[1.0, 0.0, 0.0, 0.0, 0.0, 0.0], timestamp_s=2.0
        )
        built = map_client.build()
        map_dir = Path(built.map_path)
        metadata = json.loads((map_dir / "map.json").read_text())

        assert metadata["schema_version"] == 2
        assert metadata["backend"] == "frame_archive"
        assert metadata["frames"] == 2
        assert metadata["has_depth"] is False
        assert metadata["has_world_points"] is False
        assert metadata["preprocess"] is None
        assert metadata["camera_to_world"] is None
        assert metadata["keyframes"]["path"] == "keyframe_index.json"
        assert metadata["keyframes"]["count"] == 2
        assert metadata["pushed_poses"]["status"] == "present"
        entries = json.loads((map_dir / "keyframe_index.json").read_text())
        assert [entry["index"] for entry in entries] == [0, 1]
        assert entries[0]["source_size"] == [24, 32]
        assert entries[0]["pushed_base_pose"] == [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        assert entries[0]["model_keyframe"] is None

    def test_load_frame_reads_archived_source(
        self, map_client: LingBotMapClient
    ) -> None:
        map_client.start_session(map_name="frames")
        map_client.push_frame(
            _frame(77), pose=[1.0, 2.0, 0.3, 0.0, 0.0, 0.1], timestamp_s=4.0
        )
        built = map_client.build()
        frame = map_client.load_frame(built.map_path, 0)
        assert frame.index == 0
        assert frame.source_size == (24, 32)
        assert frame.image.startswith(b"\x89PNG\r\n\x1a\n")
        assert frame.timestamp_s == 4.0
        assert frame.pushed_base_pose == [1.0, 2.0, 0.3, 0.0, 0.0, 0.1]
        assert np.asarray(Image.open(io.BytesIO(frame.image))).max() == 77

    def test_load_depth_refused_for_archive(self, map_client: LingBotMapClient) -> None:
        map_client.start_session(map_name="no_depth")
        map_client.push_frame(_frame(), pose=None, timestamp_s=1.0)
        built = map_client.build()
        with pytest.raises(FileNotFoundError, match="no depth"):
            map_client.load_depth(built.map_path, 0)

    def test_save_annotations_writes_the_artifact_set(
        self, map_client: LingBotMapClient
    ) -> None:
        map_client.start_session(map_name="annotations")
        map_client.push_frame(_frame(), pose=None, timestamp_s=1.0)
        built = map_client.build()
        map_dir = Path(built.map_path)
        mask = _png_bytes(np.full((24, 32), 255, dtype=np.uint8))
        objects = {"schema_version": 1, "labels": ["chair"], "objects": []}

        result = map_client.save_annotations(
            built.map_path,
            objects=objects,
            masks={"000000_chair.png": mask},
            semantic_points_npz=b"npz-payload",
            semantic_points_ply=b"ply-payload",
        )

        assert result["masks_written"] == 1
        assert result["semantic_artifacts"] == [
            "semantic_points.npz",
            "semantic_points.ply",
        ]
        assert (map_dir / "annotations" / "000000_chair.png").read_bytes() == mask
        assert json.loads((map_dir / "objects.json").read_text()) == objects
        assert (map_dir / "semantic_points.npz").read_bytes() == b"npz-payload"
        assert (map_dir / "semantic_points.ply").read_bytes() == b"ply-payload"

    def test_save_annotations_rejects_invalid_artifacts(
        self, map_client: LingBotMapClient
    ) -> None:
        map_client.start_session(map_name="guarded")
        map_client.push_frame(_frame(), pose=None, timestamp_s=1.0)
        built = map_client.build()
        image = _png_bytes(np.full((4, 4), 255, dtype=np.uint8))

        with pytest.raises(ValueError, match="invalid annotation file name"):
            map_client.save_annotations(
                built.map_path, objects={}, masks={"../evil.png": image}
            )
        with pytest.raises(ValueError, match="not a PNG"):
            map_client.save_annotations(
                built.map_path, objects={}, masks={"fake.png": b"plain text"}
            )
        with pytest.raises(ValueError, match="direct child"):
            map_client.save_annotations(str(Path(built.map_path).parent), objects={})

    def test_backend_selection_defaults_to_archive(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("LINGBOTMAP_CHECKPOINT_PATH", raising=False)
        backend = build_lingbot_backend(str(tmp_path / "maps"))
        assert isinstance(backend, FrameArchiveBackend)

    def test_backend_selection_falls_back_when_package_missing(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("LINGBOTMAP_CHECKPOINT_PATH", str(tmp_path / "ckpt.pt"))
        (tmp_path / "ckpt.pt").touch()
        backend = build_lingbot_backend(str(tmp_path / "maps"))
        assert isinstance(backend, FrameArchiveBackend)

    def test_backend_selection_refuses_missing_checkpoint(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # The real package is not importable offline; simulate its presence with
        # a stub module so only the missing-checkpoint path is exercised.
        import sys
        import types

        stub = types.ModuleType("lingbot_map")
        models = types.ModuleType("lingbot_map.models")
        gct = types.ModuleType("lingbot_map.models.gct_stream")

        class _StubGCTStream:  # pragma: no cover - constructor is never reached
            def __init__(self, **_: Any) -> None:
                raise AssertionError("model must not load without a checkpoint")

        gct.GCTStream = _StubGCTStream
        monkeypatch.setitem(sys.modules, "lingbot_map", stub)
        monkeypatch.setitem(sys.modules, "lingbot_map.models", models)
        monkeypatch.setitem(sys.modules, "lingbot_map.models.gct_stream", gct)
        monkeypatch.setenv("LINGBOTMAP_CHECKPOINT_PATH", str(tmp_path / "absent.pt"))
        with pytest.raises(FileNotFoundError, match="LingBot-Map checkpoint"):
            build_lingbot_backend(str(tmp_path / "maps"))


class _FakePerceptionFacade:
    """Deterministic perception backend shaped like the real RPC response."""

    _rpc = {"gdino_sam2.detect_segment": None}  # replaced per instance below

    def __init__(self) -> None:
        self._rpc = {"gdino_sam2.detect_segment": self.detect_segment}

    def detect_segment(
        self,
        *,
        image_base64: str,
        labels: list[str],
        min_score: float = 0.3,
    ) -> dict[str, Any]:
        image = np.asarray(Image.open(io.BytesIO(base64.b64decode(image_base64))))
        mask = image[..., 0] > 100
        detections = [
            {
                "label": label,
                "score": 0.9,
                "box": [2.0, 2.0, 30.0, 20.0],
                "mask_png_base64": base64.b64encode(_png_bytes(mask * 255)).decode(),
            }
            for label in labels
        ]
        return {
            "found": bool(detections),
            "detections": detections,
            "image_shape": [image.shape[0], image.shape[1]],
        }


class TestGdinoSam2Client:
    def test_detect_segment_decodes_masks(self) -> None:
        client = GdinoSam2Client(_LocalRpcClient(_FakePerceptionFacade()))
        result = client.detect_segment(_frame(200), labels=["chair", "table"])
        assert result.found
        assert [d.label for d in result.detections] == ["chair", "table"]
        assert result.image_shape == (24, 32)
        for detection in result.detections:
            assert detection.mask.shape == (24, 32)
            assert detection.mask.sum() == 24 * 32

    def test_client_validates_labels(self) -> None:
        client = GdinoSam2Client(_LocalRpcClient(_FakePerceptionFacade()))
        with pytest.raises(ValueError, match="non-empty strings"):
            client.detect_segment(_frame(), labels=[])
        with pytest.raises(ValueError, match="non-empty strings"):
            client.detect_segment(_frame(), labels=["  "])
        with pytest.raises(ValueError, match="at most 16"):
            client.detect_segment(_frame(), labels=[f"obj{i}" for i in range(17)])
        with pytest.raises(ValueError, match="min_score"):
            client.detect_segment(_frame(), labels=["chair"], min_score=1.5)
