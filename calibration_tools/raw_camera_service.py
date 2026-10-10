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
"""Serve lossless RealSense frames for fixed-camera calibration."""

from __future__ import annotations

import argparse
import base64
import json
import logging
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2
import numpy as np
from common import Record

LOGGER = logging.getLogger(__name__)
ROLES = ("base", "d455")


class Stream:
    """Own one RealSense pipeline and publish fresh frame snapshots."""

    def __init__(
        self,
        role: str,
        serial: str,
        width: int,
        height: int,
        fps: int,
    ) -> None:
        """Start the color stream of one explicitly selected camera."""
        import pyrealsense2 as rs

        self.role, self.serial = role, serial
        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_device(serial)
        config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._raw: Record | None = None
        self.pipeline.start(config)
        try:
            self._thread = threading.Thread(target=self._capture, daemon=True)
            self._thread.start()
        except Exception:
            self.pipeline.stop()
            raise

    def _capture(self) -> None:
        """Publish lossless color images and intrinsic metadata for acquisition."""
        while not self._stop.is_set():
            try:
                frames = self.pipeline.wait_for_frames(3000)
                received = time.time()
                frame = frames.get_color_frame()
                if not frame:
                    continue
                image = np.asanyarray(frame.get_data()).copy()
                intrinsics = frame.profile.as_video_stream_profile().get_intrinsics()
                ok, png = cv2.imencode(".png", image)
                if not ok:
                    raise ValueError("Cannot encode camera color frame")
                raw = {
                    "serial": self.serial,
                    "host_received_s": received,
                    "frame_number": frame.get_frame_number(),
                    "camera_timestamp_ms": frame.get_timestamp(),
                    "timestamp_domain": str(frame.get_frame_timestamp_domain()),
                    "width": intrinsics.width,
                    "height": intrinsics.height,
                    "K": [
                        [intrinsics.fx, 0, intrinsics.ppx],
                        [0, intrinsics.fy, intrinsics.ppy],
                        [0, 0, 1],
                    ],
                    "distortion_model": str(intrinsics.model),
                    "distortion": list(intrinsics.coeffs),
                    "png_b64": base64.b64encode(png).decode(),
                }
                with self._lock:
                    self._raw = raw
            except (RuntimeError, ValueError, cv2.error) as error:
                if not self._stop.is_set():
                    LOGGER.warning("Camera %s: %s", self.role, error)
                    self._stop.wait(0.2)

    def snapshot(self) -> Record:
        """Return a fresh immutable-by-convention snapshot or reject stale frames."""
        with self._lock:
            raw = self._raw
        if raw is None or not 0 <= time.time() - raw["host_received_s"] <= 0.5:
            raise ValueError("No fresh camera frame")
        return raw

    def close(self) -> None:
        """Stop the pipeline and join its capture thread."""
        self._stop.set()
        try:
            self.pipeline.stop()
        except RuntimeError as error:
            LOGGER.debug("Pipeline already stopped: %s", error)
        self._thread.join(timeout=4)


def make_handler(streams: dict[str, Stream]) -> type[BaseHTTPRequestHandler]:
    """Build the raw camera handler consumed by the collector."""

    class Handler(BaseHTTPRequestHandler):
        """Serve camera data from preexisting pipelines."""

        def do_GET(self) -> None:
            """Serve one fresh raw snapshot for the requested fixed camera."""
            path = self.path.split("?", 1)[0]
            role = path.removeprefix("/raw/")
            if role not in streams or path != f"/raw/{role}":
                self.send_error(404, "Unknown camera endpoint")
                return
            try:
                raw = streams[role].snapshot()
            except ValueError as error:
                self.send_error(503, str(error))
                return
            body = json.dumps(raw, allow_nan=False).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format: str, *args: object) -> None:
            """Route HTTP access messages to the module logger."""
            LOGGER.debug(format, *args)

    return Handler


def main() -> None:
    """Open only explicitly configured cameras and close them on every exit path."""
    parser = argparse.ArgumentParser(description=__doc__)
    for role in ROLES:
        parser.add_argument(f"--{role.replace('_', '-')}-serial")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=15)
    args = parser.parse_args()
    devices = {
        role: getattr(args, role + "_serial")
        for role in ROLES
        if getattr(args, role + "_serial")
    }
    if not devices or len(set(devices.values())) != len(devices):
        parser.error(
            "Specify at least one camera, with a different serial for each role"
        )
    if min(args.width, args.height, args.fps) <= 0 or not 1 <= args.port <= 65535:
        parser.error("Image size and FPS must be positive; port must be 1..65535")
    logging.basicConfig(level=logging.INFO)
    streams: dict[str, Stream] = {}
    try:
        for role, serial in devices.items():
            streams[role] = Stream(
                role,
                serial,
                args.width,
                args.height,
                args.fps,
            )
        with ThreadingHTTPServer(
            ("127.0.0.1", args.port), make_handler(streams)
        ) as server:
            LOGGER.info("Camera service http://127.0.0.1:%d", args.port)
            server.serve_forever()
    except KeyboardInterrupt:
        LOGGER.info("Camera service stopped")
    finally:
        for stream in reversed(list(streams.values())):
            stream.close()


if __name__ == "__main__":
    main()
