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

"""CPU visual localization RPC service for local or remote deployment."""

from __future__ import annotations

import argparse
import io
import math
import time
from pathlib import Path

import numpy as np

from rpent.utils.rpc import RpcFacade


def camera_from_metadata(meta):
    """Require rectified OpenCV pinhole intrinsics for native PnP."""
    if not isinstance(meta, dict) or meta.get("calibrated") is not True:
        raise ValueError("Calibrated camera required")
    if (
        meta.get("model") != "PINHOLE"
        or meta.get("pixel_convention") != "opencv_center"
    ):
        raise ValueError("Rectified OpenCV PINHOLE camera required")
    width, height = meta.get("width"), meta.get("height")
    if any(type(v) is not int or not 32 <= v <= 4096 for v in (width, height)):
        raise ValueError("Camera size must be 32..4096 pixels")
    params = np.asarray(meta.get("params"), dtype=float)
    if (
        params.shape != (4,)
        or not np.isfinite(params).all()
        or min(params[:2]) <= 0
        or not 0 <= params[2] < width
        or not 0 <= params[3] < height
    ):
        raise ValueError("Invalid camera intrinsics")
    return meta


def validate_node_id(node_id: int) -> None:
    if type(node_id) is not int or node_id <= 0:
        raise ValueError("node_id must be a positive integer")


def validate_map_query(label=None, limit=None, center_node_id=None, radius_m=None):
    if label is not None and (not isinstance(label, str) or not 1 <= len(label) <= 128):
        raise ValueError("label must contain 1..128 characters")
    if limit is not None and (type(limit) is not int or not 1 <= limit <= 100):
        raise ValueError("limit must be an integer from 1 to 100")
    if center_node_id is not None:
        validate_node_id(center_node_id)
    if radius_m is not None:
        if (
            type(radius_m) not in (int, float)
            or not math.isfinite(radius_m)
            or not 0.1 <= radius_m <= 100
            or center_node_id is None
        ):
            raise ValueError(
                "radius_m requires center_node_id and must be 0.1..100 meters"
            )


class Localizer:
    """Adapt native localization results and discard late responses."""

    def __init__(self, database, max_processing_ms=2000):
        import lekiwi_localization

        if not np.isfinite(max_processing_ms) or max_processing_ms <= 0:
            raise ValueError("max_processing_ms must be positive and finite")
        self.timeout = max_processing_ms / 1000
        self._native = lekiwi_localization.Localizer(
            str(Path(database).expanduser().resolve())
        )

    def _timed_call(self, method, *args):
        started = time.monotonic()
        result = method(*args)
        elapsed = time.monotonic() - started
        # Native calls cannot be interrupted; discard results that arrive too late.
        if elapsed > self.timeout:
            raise TimeoutError("RTAB-Map processing deadline exceeded")
        return result, elapsed * 1000

    def localize(self, rgb, camera, packet):
        result, elapsed_ms = self._timed_call(
            self._native.localize,
            rgb,
            camera,
            packet["stamp_ns"],
            packet["T_base_camera"],
            packet["odometry"],
        )
        result["processing_ms"] = elapsed_ms
        return result

    def goal(self, target: int | str) -> dict:
        """Resolve an existing RTAB-Map node or label using its native graph API."""
        result, _ = self._timed_call(self._native.goal, target)
        if result["status"] != "goal":
            raise ValueError(f"Unknown RTAB-Map node or label: {target}")
        return result

    def list_map_nodes(
        self, label=None, limit=None, center_node_id=None, radius_m=None
    ) -> dict:
        """List graph nodes in ID order, optionally within a node-centered radius."""
        validate_map_query(label, limit, center_node_id, radius_m)
        nodes, _ = self._timed_call(self._native.list_nodes)
        center = None
        if center_node_id is not None:
            center = next((n for n in nodes if n["node_id"] == center_node_id), None)
            if center is None:
                raise ValueError(f"Unknown RTAB-Map node: {center_node_id}")
        selected = []
        for node in sorted(nodes, key=lambda n: n["node_id"]):
            if label is not None and node["label"] != label:
                continue
            if radius_m is not None:
                xy = np.asarray(node["pose"][:2]) - center["pose"][:2]
                if np.linalg.norm(xy) > radius_m:
                    continue
            selected.append(node)
            if limit is not None and len(selected) > limit:
                break
        return {
            "frame_id": "map",
            "nodes": selected[:limit],
            "nodes_truncated": limit is not None and len(selected) > limit,
            "center": center,
        }

    def inspect_map_node(self, node_id: int) -> dict:
        """Read a bounded historical image and the node's optimized map pose."""
        from PIL import Image

        from robots.lekiwi.map_view import encode_png

        validate_node_id(node_id)
        result, _ = self._timed_call(self._native.inspect_node, node_id)
        result = dict(result, frame_id="map", image_kind="historical")
        encoded = result.pop("image")
        result["image_available"] = encoded is not None
        if encoded is not None:
            with Image.open(io.BytesIO(encoded.tobytes())) as image:
                result["image_png"] = encode_png(image.convert("RGB"))
        return result

    def close(self):
        # The native instance serializes close with active calls and owns its state.
        self._native.close()


def localize_frame(localizer, packet: dict, jpeg) -> dict:
    """Decode one image and attach its source coordinates to the native result."""
    from PIL import Image

    camera_meta = packet["camera"]
    camera = camera_from_metadata(camera_meta)
    with Image.open(io.BytesIO(jpeg.tobytes())) as decoded:
        if decoded.size != (camera["width"], camera["height"]):
            raise ValueError("Image dimensions do not match camera parameters")
        rgb = np.array(decoded.convert("RGB"))
    result = localizer.localize(rgb, camera, packet)
    result.update(
        stamp_ns=packet["stamp_ns"],
        camera_frame_id=packet["frame_id"],
        map_frame_id="map",
        clock_domain=packet["clock_domain"],
        camera_calibrated=camera_meta["calibrated"],
    )
    return result


class LocalizationFacade(RpcFacade):
    """Serialize localization through the framework's RPC dispatch lock."""

    def __init__(self, localizer) -> None:
        super().__init__()
        self._localizer = localizer
        self._rpc["localization.localize"] = self.localize
        self._rpc["localization.goal"] = self.goal
        self._rpc["localization.list_map_nodes"] = self.list_map_nodes
        self._rpc["localization.inspect_map_node"] = self.inspect_map_node

    def goal(self, target: int | str) -> dict:
        """Read a node goal from the same database used for localization."""
        if type(target) not in (int, str):
            raise ValueError("Target must be a node ID or label")
        return self._localizer.goal(target)

    def list_map_nodes(self, **kwargs) -> dict:
        return self._localizer.list_map_nodes(**kwargs)

    def inspect_map_node(self, node_id: int) -> dict:
        return self._localizer.inspect_map_node(node_id)

    def localize(self, packet: dict, jpeg) -> dict:
        return localize_frame(self._localizer, packet, jpeg)


def serve_localization(
    *,
    database: Path,
    bind: str,
    port: int,
    max_processing_ms: float = 2000.0,
) -> None:
    """Serve localization requests until the operator stops the service."""
    localizer = Localizer(database, max_processing_ms=max_processing_ms)
    try:
        LocalizationFacade(localizer).serve(transport="http", host=bind, port=port)
    finally:
        localizer.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["serve"])
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--bind", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18125)
    parser.add_argument("--max-processing-ms", type=float, default=2000.0)
    args = parser.parse_args(argv)

    return serve_localization(
        database=args.database,
        bind=args.bind,
        port=args.port,
        max_processing_ms=args.max_processing_ms,
    )


if __name__ == "__main__":
    main()
