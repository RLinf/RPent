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

"""Offline tests for the map localization RPC service."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from rpent.tools.localize import LocalizationResult
from rpent.utils.rpc import RpcClient
from toolkits.localization.client import LocalizeClient
from toolkits.localization.server import LocalizeFacade


class _LocalRpcClient(RpcClient):
    """Dispatch calls directly into a facade's registered RPC methods."""

    def __init__(self, facade: Any) -> None:
        self._facade = facade

    def call(
        self, method: str, args: tuple = (), kwargs: dict | None = None, **_: Any
    ) -> Any:
        handler = self._facade._rpc[method]
        return handler(*(args or ()), **(kwargs or {}))


class _StubLocalizer:
    """Return a fixed pose and record the arguments it receives."""

    def __init__(
        self,
        map_path: str = "/maps/workspace",
        *,
        metadata: dict[str, Any] | None = None,
        pose: np.ndarray | None = None,
        status: str = "ok",
    ) -> None:
        self.map_path = map_path
        self.frame = "model"
        self.scale = {"metric": False, "meters_per_unit": None}
        self.source = SimpleNamespace(
            frame_count=5, metadata={"frames": 5} if metadata is None else metadata
        )
        self.model_indices = [0, 2, 4]
        self.intrinsic = np.eye(3)
        self._pose = pose
        self._status = status
        self.calls: list[dict[str, Any]] = []

    def localize(self, image: np.ndarray, **kwargs: Any) -> LocalizationResult:
        self.calls.append({"image": image, **kwargs})
        if self._pose is None:
            pose = np.eye(4)
            pose[0, 3] = 1.5
        else:
            pose = np.asarray(self._pose, dtype=np.float64).copy()
        if self._status != "ok":
            pose = None
        return LocalizationResult(
            status=self._status,
            frame="model",
            pose=pose,
            keyframe=2,
            matches=80,
            correspondences=70,
            inliers=60,
            inlier_ratio=0.857,
            reprojection_rms_px=2.5,
            query_keypoints=320,
            candidates_checked=3,
            selection=kwargs.get("selection", "exhaustive"),
            ambiguous=False,
            latency_ms=12.0,
            scale=dict(self.scale),
            reason=None if pose is not None else "too few inliers",
        )


@pytest.fixture
def stub() -> _StubLocalizer:
    return _StubLocalizer()


@pytest.fixture
def client(stub: _StubLocalizer) -> LocalizeClient:
    return LocalizeClient(
        _LocalRpcClient(LocalizeFacade("/maps/workspace", localizer=stub))
    )


def test_localize_round_trip_preserves_pose_and_options(
    client: LocalizeClient, stub: _StubLocalizer
) -> None:
    prior = np.eye(4)
    prior[2, 3] = 0.4

    result = client.localize(
        np.full((8, 12, 3), 90, dtype=np.uint8),
        prior_pose=prior,
        exhaustive=True,
        exclude=[0, 1],
    )

    assert result.status == "ok"
    assert result.ok
    assert result.keyframe == 2
    assert result.pose is not None
    assert result.pose[0, 3] == pytest.approx(1.5)
    call = stub.calls[0]
    assert call["image"].shape == (8, 12, 3)
    assert call["image"].dtype == np.uint8
    assert call["exhaustive"] is True
    assert call["exclude"] == [0, 1]
    assert np.allclose(call["prior_pose"], prior)


def test_localize_server_rejects_invalid_payloads(stub: _StubLocalizer) -> None:
    facade = LocalizeFacade("/maps/workspace", localizer=stub)

    with pytest.raises(ValueError, match="image_base64"):
        facade.localize(image_base64="")
    with pytest.raises(ValueError, match="invalid image data"):
        facade.localize(image_base64="not-base64!")
    with pytest.raises(ValueError, match="prior_pose must be 4x4"):
        facade.localize(
            image_base64=_encoded_frame(),
            prior_pose=[[1.0, 0.0], [0.0, 1.0]],
        )
    with pytest.raises(ValueError, match="exclude entries"):
        facade.localize(image_base64=_encoded_frame(), exclude=["frame"])
    with pytest.raises(ValueError, match="min_inlier_ratio"):
        facade.localize(image_base64=_encoded_frame(), min_inlier_ratio=2.0)
    assert stub.calls == []


def test_localize_server_accepts_frame_zero_as_prior(
    client: LocalizeClient, stub: _StubLocalizer
) -> None:
    client.localize(np.zeros((6, 6, 3), dtype=np.uint8), prior_index=0)

    assert stub.calls[0]["prior_index"] == 0


def test_info_reports_map_geometry(client: LocalizeClient) -> None:
    info = client.info()

    assert info["map_path"] == "/maps/workspace"
    assert info["frame"] == "model"
    assert info["frames"] == 5
    assert info["model_keyframes"] == 3
    assert info["intrinsic"] == np.eye(3).tolist()
    assert info["scale"] == {"metric": False, "meters_per_unit": None}


def test_result_from_dict_rejects_a_malformed_pose() -> None:
    payload = {
        "status": "ok",
        "pose": [[1.0, 0.0, 0.0]],
        "scale": {"metric": False, "meters_per_unit": None},
    }

    with pytest.raises(ValueError, match="pose must be 4x4"):
        LocalizationResult.from_dict(payload)


def _encoded_frame() -> str:
    import base64
    import io

    from PIL import Image

    buffer = io.BytesIO()
    Image.fromarray(np.zeros((6, 6, 3), dtype=np.uint8)).save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def _clearance_map(tmp_path: Path) -> tuple[str, dict[str, Any]]:
    """Write a tiny map with a floor and a wall row ahead of the origin."""
    height, width = 24, 32
    rows = np.arange(height)[:, None]
    cols = np.arange(width)[None, :]
    # The floor spans x -0.2 .. 2.1 and y -0.8 .. 0.75; a wall crosses the
    # corridor one model unit ahead. The checks stride the pixel grid by four,
    # so the wall sits on a row and column set the stride keeps.
    points = np.stack(
        np.broadcast_arrays(
            0.1 * (rows - 2),
            0.05 * (cols - 16),
            np.full((height, width), -0.2),
        ),
        axis=-1,
    )
    points[20, :, 0] = 1.0
    points[20, :, 1] = 0.05 * (np.arange(width) - 16)
    points[20, :, 2] = -0.1
    metadata = {
        "schema_version": 2,
        "frames": 1,
        "gravity": {"status": "estimated", "up": [0.0, 0.0, 1.0]},
        "floor": {"status": "estimated", "offset": -0.2, "camera_height": 0.2},
        "scale": {"metric": False, "meters_per_unit": None},
    }
    root = tmp_path / "map"
    root.mkdir()
    (root / "map.json").write_text(json.dumps(metadata))
    np.save(root / "world_points.npy", points[None, None].astype(np.float32))
    return str(root), metadata


#: A camera at the map origin looking along model ``+x``.
LOOK_ALONG_X = np.asarray(
    [
        [0.0, 0.0, 1.0, 0.0],
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
)


def test_check_path_is_registered_as_a_mutating_rpc(
    stub: _StubLocalizer,
) -> None:
    facade = LocalizeFacade("/maps/workspace", localizer=stub)

    assert "localize.check_path" in facade._rpc
    assert "localize.check_path" not in facade._readonly_methods
    assert "localize.info" in facade._readonly_methods


def test_check_path_with_pose_reports_the_corridor(tmp_path: Path) -> None:
    map_path, metadata = _clearance_map(tmp_path)
    stub = _StubLocalizer(map_path, metadata=metadata)
    client = LocalizeClient(_LocalRpcClient(LocalizeFacade(map_path, localizer=stub)))

    result = client.check_path(
        pose=LOOK_ALONG_X,
        unit="map_units",
        length=2.0,
        half_width=0.5,
        min_height=0.05,
    )

    assert result.status == "blocked"
    assert result.unit == "map_units"
    assert result.localization is None
    assert result.keyframe is None
    assert result.nearest is not None
    assert result.nearest["distance"] == pytest.approx(1.0)
    assert stub.calls == []


def test_check_path_with_image_forwards_localization_options(tmp_path: Path) -> None:
    map_path, metadata = _clearance_map(tmp_path)
    stub = _StubLocalizer(map_path, metadata=metadata, pose=LOOK_ALONG_X)
    client = LocalizeClient(_LocalRpcClient(LocalizeFacade(map_path, localizer=stub)))

    result = client.check_path(
        np.zeros((6, 6, 3), dtype=np.uint8),
        prior_index=2,
        unit="map_units",
        length=2.0,
        half_width=0.5,
        min_height=0.05,
    )

    assert stub.calls[0]["prior_index"] == 2
    assert result.status == "blocked"
    assert result.keyframe == 2
    assert result.localization is not None
    assert result.localization["status"] == "ok"
    assert result.localization["inliers"] == 60


def test_check_path_without_a_pose_is_unlocalized(tmp_path: Path) -> None:
    map_path, metadata = _clearance_map(tmp_path)
    stub = _StubLocalizer(map_path, metadata=metadata, status="lost")
    client = LocalizeClient(_LocalRpcClient(LocalizeFacade(map_path, localizer=stub)))

    result = client.check_path(np.zeros((6, 6, 3), dtype=np.uint8))

    assert result.status == "unlocalized"
    assert result.pose is None
    assert result.localization is not None
    assert result.localization["status"] == "lost"
    assert result.localization["reason"] == "too few inliers"


def test_check_path_requires_exactly_one_source(
    stub: _StubLocalizer, tmp_path: Path
) -> None:
    map_path, metadata = _clearance_map(tmp_path)
    stub = _StubLocalizer(map_path, metadata=metadata)
    facade = LocalizeFacade(map_path, localizer=stub)

    with pytest.raises(ValueError, match="exactly one"):
        facade.check_path()
    with pytest.raises(ValueError, match="exactly one"):
        facade.check_path(image_base64=_encoded_frame(), pose=LOOK_ALONG_X.tolist())
    with pytest.raises(ValueError, match="apply only to image_base64"):
        facade.check_path(pose=LOOK_ALONG_X.tolist(), exhaustive=True)
    with pytest.raises(ValueError, match="pose must be 4x4"):
        facade.check_path(pose=[[1.0, 0.0], [0.0, 1.0]])


def test_check_path_client_rejects_malformed_response() -> None:
    class _StaticClient(RpcClient):
        def call(
            self, method: str, args: tuple = (), kwargs: dict | None = None, **_
        ) -> Any:
            assert method == "localize.check_path"
            return {"status": "clear"}

    client = LocalizeClient(_StaticClient())

    with pytest.raises(RuntimeError, match="check_path"):
        client.check_path(pose=LOOK_ALONG_X)


def test_info_reports_clearance_availability(tmp_path: Path) -> None:
    map_path, metadata = _clearance_map(tmp_path)
    stub = _StubLocalizer(map_path, metadata=metadata)
    client = LocalizeClient(_LocalRpcClient(LocalizeFacade(map_path, localizer=stub)))

    clearance = client.info()["clearance"]

    assert clearance["available"] is True
    assert clearance["frame"] == "model"
    assert clearance["camera_height"] == pytest.approx(0.2)
    assert clearance["units"] == ["camera_heights", "map_units", "m"]
    assert clearance["defaults"]["length"] == pytest.approx(2.0)
