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
"""Stationary hand-eye acquisition with explicit per-session configuration."""

from __future__ import annotations

import argparse
import base64
import json
import logging
import re
import shlex
import shutil
import subprocess
import tempfile
import threading
import time
import urllib.request
from dataclasses import dataclass
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import cv2
import numpy as np
from common import (
    Record,
    check_camera,
    check_opencv,
    check_state,
    delta,
    sample_poses,
    transform,
    write_json,
)

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class CaptureConfig:
    """Immutable camera, board, and read-only robot reader configuration."""

    arm: str
    camera_serial: str
    camera_url: str
    reader_command: tuple[str, ...]
    squares_x: int = 6
    squares_y: int = 8
    square_m: float = 0.025
    marker_m: float = 0.018
    dictionary: str = "DICT_4X4_100"

    def __post_init__(self) -> None:
        """Reject invalid geometry and missing device configuration."""
        check_opencv()
        if self.arm not in ("left", "right"):
            raise ValueError("Arm must be left or right")
        if not self.camera_serial or not self.reader_command:
            raise ValueError("Camera serial and robot reader command are required")
        if (
            min(self.squares_x, self.squares_y) < 3
            or not 0 < self.marker_m < self.square_m
        ):
            raise ValueError(
                "Board requires at least 3 x 3 squares and 0 < marker_m < square_m"
            )
        if (self.squares_x - 1) * (self.squares_y - 1) < 15:
            raise ValueError("Board must provide at least 15 internal corners")
        if not hasattr(cv2.aruco, self.dictionary):
            raise ValueError(f"Unknown ArUco dictionary: {self.dictionary}")

    @property
    def board_metadata(self) -> Record:
        """Serializable board geometry in meters."""
        return {
            "squares": [self.squares_x, self.squares_y],
            "square_m": self.square_m,
            "marker_m": self.marker_m,
            "dictionary": self.dictionary,
            "legacy": False,
        }


class Collector:
    """Own acquisition state without mutating other imported modules."""

    def __init__(
        self, config: CaptureConfig, root: Path, *, resume: bool = False
    ) -> None:
        """Initialize an isolated session; hardware is read only on request."""
        self.config = config
        self.root = root
        self.pose_key = f"T_{config.arm}_base_ee"
        self.samples: list[Record] = []
        self.reference: Record | None = None
        self.next_index = 1
        self.last: Record = {}
        self.lock = threading.Lock()
        self.board = cv2.aruco.CharucoBoard(
            (config.squares_x, config.squares_y),
            config.square_m,
            config.marker_m,
            cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, config.dictionary)),
        )
        self.board.setLegacyPattern(False)
        if resume:
            self._resume()
        elif root.exists():
            raise FileExistsError("Output directory already exists; use --resume")

    def _resume(self) -> None:
        """Load saved samples without changing their identities or files."""
        if not self.root.is_dir():
            raise ValueError("--resume requires an existing session directory")
        reference: Record | None = None
        indices: set[int] = set()
        for parent in (self.root, self.root / "excluded"):
            for folder in sorted(parent.glob("sample_*")):
                match = re.fullmatch(r"sample_([0-9]+)", folder.name)
                if not match or not folder.is_dir() or folder.is_symlink():
                    raise ValueError(f"Invalid sample directory: {folder}")
                index = int(match[1])
                if (
                    index < 1
                    or index in indices
                    or folder.name != f"sample_{index:03d}"
                ):
                    raise ValueError(f"Invalid or duplicate sample index: {folder}")
                indices.add(index)
                self.next_index = max(self.next_index, index + 1)
                record = json.loads(
                    (folder / "sample.json").read_text(encoding="utf-8")
                )
                if type(record["index"]) is not int or record["index"] != index:
                    raise ValueError(f"Sample index does not match: {folder}")
                if reference is None:
                    reference = record
                sample_poses(
                    record,
                    reference,
                    arm=self.config.arm,
                    serial=self.config.camera_serial,
                    board=self.config.board_metadata,
                    source=folder,
                )
                for name in ("color.png", "annotated.png"):
                    if not (folder / name).is_file():
                        raise ValueError(f"Missing sample image: {folder / name}")
                if parent == self.root:
                    self.samples.append(record)
        self.samples.sort(key=lambda sample: sample["index"])
        self.reference = reference
        self.last = {"resumed": True, "count": len(self.samples)}

    def delete_sample(self, index: int) -> Record:
        """Exclude a selected sample, preserving its files for manual recovery."""
        if type(index) is not int or index < 1:
            raise ValueError("Sample index must be a positive integer")
        sample = next((s for s in self.samples if s["index"] == index), None)
        if sample is None:
            raise ValueError(f"No active sample with index {index}")
        source = self.root / f"sample_{index:03d}"
        destination = self.root / "excluded" / source.name
        if destination.exists():
            raise FileExistsError(f"Excluded sample already exists: {destination}")
        destination.parent.mkdir(exist_ok=True)
        source.rename(destination)
        self.samples.remove(sample)
        LOGGER.info("Excluded sample %d; files retained at %s", index, destination)
        return {
            "deleted": True,
            "index": index,
            "count": len(self.samples),
            "archived": str(destination),
        }

    def read_state(self) -> Record:
        """Run the configured readOnce probe with a bounded local timeout."""
        result = subprocess.run(
            self.config.reader_command,
            capture_output=True,
            text=True,
            timeout=15,
            check=True,
        )
        state = json.loads(result.stdout)
        check_state(state)
        return state

    def sample(self) -> Record:
        """Acquire one stationary pose/image pair and persist it after validation."""
        start = time.time()
        before = self.read_state()
        boundary = time.time()
        # Wait for a frame whose host receipt is later than the first read completed.
        time.sleep(0.15)
        with urllib.request.urlopen(self.config.camera_url, timeout=5) as r:
            camera = json.load(r)
        if camera["serial"] != self.config.camera_serial:
            raise ValueError("相机序列号不匹配")
        if (
            not boundary < camera["host_received_s"] <= time.time()
            or time.time() - camera["host_received_s"] > 0.3
        ):
            raise ValueError("相机帧时间不在当前采样窗口内")
        after = self.read_state()
        a, b = transform(before), transform(after)
        drift_m, drift_rad = delta(a, b)
        if drift_m > 0.001 or drift_rad > 0.005:
            raise ValueError(
                f"采样期间位姿变化过大：{drift_m * 1000:.2f} mm, {np.degrees(drift_rad):.2f} deg"
            )
        if time.time() - start > 3:
            raise ValueError("采样窗口超过 3 秒，请检查网络后重试")
        if before["F_T_EE"] != after["F_T_EE"]:
            raise ValueError("末端坐标定义在采样期间发生变化")
        if self.reference is not None and not np.allclose(
            before["F_T_EE"],
            self.reference["robot_before"]["F_T_EE"],
            atol=1e-8,
            rtol=0,
        ):
            raise ValueError("末端坐标定义与先前样本不同")
        for old in self.samples:
            dm, dr = delta(a, np.array(old[self.pose_key]))
            if dm < 0.005 and dr < np.deg2rad(5):
                raise ValueError("与已有姿态过于接近，请改变姿态后采集；不要重复点击")
        check_camera(camera, self.reference["camera"] if self.reference else camera)
        png = base64.b64decode(camera.pop("png_b64"), validate=True)
        im = cv2.imdecode(np.frombuffer(png, np.uint8), cv2.IMREAD_COLOR)
        if im is None or im.shape[:2] != (camera["height"], camera["width"]):
            raise ValueError(
                "Image decoding failed or dimensions do not match metadata"
            )
        K = np.array(camera["K"], dtype=float)
        dist = np.array(camera["distortion"], dtype=float)
        if np.any(dist != 0) and camera["distortion_model"] not in [
            "distortion.brown_conrady"
        ]:
            raise ValueError(
                "Unsupported nonzero distortion: "
                f"model={camera['distortion_model']}, coefficients={dist.tolist()}. "
                "Convert to an OpenCV-compatible model before sampling."
            )
        params = cv2.aruco.CharucoParameters()
        params.cameraMatrix, params.distCoeffs = K, dist
        cc, ci, mc, mi = cv2.aruco.CharucoDetector(self.board, params).detectBoard(im)
        n = 0 if ci is None else len(ci)
        if n < 15 or self.board.checkCharucoCornersCollinear(ci):
            raise ValueError(f"有效角点不足或共线：{n}，建议调整板角度和照明")
        obj, img = self.board.matchImagePoints(cc, ci)
        ok, rv, tv = cv2.solvePnP(obj, img, K, dist)
        if not ok:
            raise ValueError("PnP 求解失败")
        rot = cv2.Rodrigues(rv)[0]
        if np.min((obj.reshape(-1, 3) @ rot.T + tv.reshape(3))[:, 2]) <= 0:
            raise ValueError("PnP 棋盘位于相机后方")
        proj = cv2.projectPoints(obj, rv, tv, K, dist)[0]
        rms = float(
            np.sqrt(
                np.mean(np.sum((proj.reshape(-1, 2) - img.reshape(-1, 2)) ** 2, axis=1))
            )
        )
        if not np.isfinite(rms) or rms > 1:
            raise ValueError(f"重投影 RMS {rms:.3f} px 超过 1 px")
        t = np.eye(4)
        t[:3, :3] = rot
        t[:3, 3] = tv.ravel()
        record = {
            "index": self.next_index,
            "host_start_s": start,
            "host_end_s": time.time(),
            "robot_before": before,
            "robot_after": after,
            "camera": camera,
            self.pose_key: a.tolist(),
            "T_camera_board": t.tolist(),
            "arm": self.config.arm,
            "calibration_mode": "eye_to_hand",
            "charuco_ids": ci.flatten().tolist(),
            "charuco_corners_px": cc.reshape(-1, 2).tolist(),
            "corners": n,
            "markers": 0 if mi is None else len(mi),
            "reprojection_rms_px": rms,
            "drift_m": drift_m,
            "drift_rad": drift_rad,
            "board": self.config.board_metadata,
            "convention": "T_A_B maps B to A; ee is Franka O_T_EE, not an assumed Robotiq fingertip TCP",
            "sampling": "stationary bracketed reads; not hardware synchronized",
        }
        self.root.mkdir(parents=True, exist_ok=True)
        folder = self.root / f"sample_{record['index']:03d}"
        temporary = Path(tempfile.mkdtemp(prefix=".sample_", dir=self.root))
        try:
            (temporary / "color.png").write_bytes(png)
            cv2.aruco.drawDetectedCornersCharuco(im, cc, ci)
            cv2.drawFrameAxes(im, K, dist, rv, tv, 0.05)
            if not cv2.imwrite(str(temporary / "annotated.png"), im):
                raise OSError("Failed to write annotated image")
            write_json(temporary / "sample.json", record)
            if folder.exists():
                raise FileExistsError(f"Sample directory already exists: {folder}")
            temporary.rename(folder)
        finally:
            if temporary.exists():
                shutil.rmtree(temporary)
        self.samples.append(record)
        if self.reference is None:
            self.reference = record
        self.next_index += 1
        return {
            "accepted": True,
            "count": len(self.samples),
            "corners": n,
            "reprojection_rms_px": round(rms, 3),
            "saved": str(folder),
        }

    def status(self) -> Record:
        """Return the current session count and last acquisition result."""
        return {
            "count": len(self.samples),
            "session": str(self.root),
            "last": self.last,
            "samples": [
                {
                    "index": sample["index"],
                    "corners": sample["corners"],
                    "reprojection_rms_px": round(sample["reprojection_rms_px"], 3),
                }
                for sample in self.samples
            ],
        }

    def preview(self) -> bytes:
        """Return a fresh annotated JPEG without recording a calibration sample."""
        with urllib.request.urlopen(self.config.camera_url, timeout=3) as response:
            camera = json.load(response)
        if (
            camera["serial"] != self.config.camera_serial
            or not 0 <= time.time() - camera["host_received_s"] <= 1
        ):
            raise ValueError("Wrong camera or stale frame")
        image = cv2.imdecode(
            np.frombuffer(base64.b64decode(camera["png_b64"], validate=True), np.uint8),
            cv2.IMREAD_COLOR,
        )
        if image is None:
            raise ValueError("Camera image cannot be decoded")
        corners, identifiers, _, _ = cv2.aruco.CharucoDetector(self.board).detectBoard(
            image
        )
        if identifiers is not None:
            cv2.aruco.drawDetectedCornersCharuco(image, corners, identifiers)
        ok, encoded = cv2.imencode(".jpg", image)
        if not ok:
            raise ValueError("Preview cannot be encoded")
        return encoded.tobytes()


def make_handler(collector: Collector) -> type[BaseHTTPRequestHandler]:
    """Build a request handler bound to one collector instance."""

    class Handler(BaseHTTPRequestHandler):
        """Serve previews and serialize sampling requests."""

        def reply(self, code: int, body: bytes, kind: str = "application/json") -> None:
            """Send an uncached response with an explicit body size."""
            self.send_response(code)
            self.send_header("Content-Type", kind)
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:
            """Serve the session page, status, or camera preview."""
            if self.path == "/":
                self.reply(
                    200,
                    Path(__file__).with_suffix(".html").read_bytes(),
                    "text/html; charset=utf-8",
                )
            elif self.path == "/status":
                with collector.lock:
                    status = collector.status()
                self.reply(200, json.dumps(status, allow_nan=False).encode())
            elif match := re.fullmatch(r"/samples/([0-9]+)/annotated\.png", self.path):
                index = int(match[1])
                with collector.lock:
                    if not any(s["index"] == index for s in collector.samples):
                        self.send_error(404)
                        return
                    try:
                        data = (
                            collector.root / f"sample_{index:03d}" / "annotated.png"
                        ).read_bytes()
                    except OSError as error:
                        self.reply(503, json.dumps({"error": str(error)}).encode())
                        return
                self.reply(200, data, "image/png")
            elif self.path.startswith("/frame.jpg"):
                try:
                    self.reply(200, collector.preview(), "image/jpeg")
                except (ValueError, OSError, cv2.error, KeyError) as error:
                    self.reply(503, json.dumps({"error": str(error)}).encode())
            else:
                self.send_error(404)

        def do_POST(self) -> None:
            """Serialize sample capture and recoverable exclusion."""
            deletion = re.fullmatch(r"/samples/([0-9]+)/delete", self.path)
            if self.path != "/sample" and deletion is None:
                self.send_error(404)
                return
            if not collector.lock.acquire(blocking=False):
                self.reply(409, b'{"error":"sample operation in progress"}')
                return
            try:
                try:
                    collector.last = (
                        collector.delete_sample(int(deletion[1]))
                        if deletion
                        else collector.sample()
                    )
                    code = 200
                except (
                    ValueError,
                    OSError,
                    KeyError,
                    cv2.error,
                    subprocess.SubprocessError,
                ) as error:
                    LOGGER.warning("Sample operation rejected: %s", error)
                    collector.last = {"accepted": False, "error": str(error)}
                    code = 422
                self.reply(code, json.dumps(collector.last, allow_nan=False).encode())
            finally:
                collector.lock.release()

        def log_message(self, format: str, *args: object) -> None:
            """Route HTTP access messages to the module logger."""
            LOGGER.debug(format, *args)

    return Handler


def add_arguments(parser: argparse.ArgumentParser) -> None:
    """Add the device and board options for fixed-camera acquisition."""
    parser.add_argument("--arm", choices=("left", "right"), required=True)
    parser.add_argument("--camera-serial", required=True)
    parser.add_argument("--camera-url", required=True, help="Raw camera JSON endpoint")
    parser.add_argument(
        "--reader", required=True, help="Path to read_franka_state on the reader host"
    )
    parser.add_argument("--robot-ip", required=True)
    parser.add_argument("--ssh-host", help="SSH alias; omit to run the reader locally")
    parser.add_argument(
        "--library-dir",
        help="Optional libfranka shared-library directory on the reader host",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--resume", action="store_true", help="Resume --output with unchanged setup"
    )
    parser.add_argument("--port", type=int, default=8767)
    parser.add_argument("--squares-x", type=int, default=6)
    parser.add_argument("--squares-y", type=int, default=8)
    parser.add_argument("--square-m", type=float, default=0.025)
    parser.add_argument("--marker-m", type=float, default=0.018)
    parser.add_argument("--dictionary", default="DICT_4X4_100")


def build_config(args: argparse.Namespace) -> CaptureConfig:
    """Construct a reader command, quoting remote shell arguments explicitly."""
    command = [args.reader, args.robot_ip]
    if args.library_dir:
        command = ["env", f"LD_LIBRARY_PATH={args.library_dir}", *command]
    if args.ssh_host:
        if args.ssh_host.startswith("-"):
            raise ValueError("SSH host cannot start with an option prefix")
        command = [
            "ssh",
            "-o",
            "BatchMode=yes",
            "-o",
            "ConnectTimeout=5",
            args.ssh_host,
            shlex.join(["timeout", "10", *command]),
        ]
    return CaptureConfig(
        args.arm,
        args.camera_serial,
        args.camera_url,
        tuple(command),
        args.squares_x,
        args.squares_y,
        args.square_m,
        args.marker_m,
        args.dictionary,
    )


def serve(collector: Collector, port: int) -> None:
    """Serve a local-only collector and release the socket on shutdown."""
    if not 1 <= port <= 65535:
        raise ValueError("Port must be between 1 and 65535")
    with ThreadingHTTPServer(("127.0.0.1", port), make_handler(collector)) as server:
        LOGGER.info("Session %s; collector http://127.0.0.1:%d", collector.root, port)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            LOGGER.info("Collector stopped")


def main() -> None:
    """Run a configured collector; imports never open hardware or create files."""
    parser = argparse.ArgumentParser(description=__doc__)
    add_arguments(parser)
    args = parser.parse_args()
    if args.resume and args.output is None:
        parser.error("--resume requires --output")
    config = build_config(args)
    root = args.output or Path(__file__).resolve().parent / "sessions" / (
        datetime.now().strftime("%Y%m%d_%H%M%S_%f") + f"_{config.arm}_eye_to_hand"
    )
    try:
        collector = Collector(config, root, resume=args.resume)
    except (ValueError, OSError, KeyError) as error:
        parser.error(str(error))
    logging.basicConfig(level=logging.INFO)
    serve(collector, args.port)


if __name__ == "__main__":
    main()
