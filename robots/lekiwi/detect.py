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

"""Generate a LeKiwi site YAML by probing the serial bus and the cameras.

Run on the robot's computer, with the ``lekiwi`` extra installed:

    rpent-lekiwi-detect --output /path/to/robot.yaml

Nothing moves: the wheel bus is only pinged (lerobot's
``FeetechMotorsBus.scan_port`` broadcasts a ping at every baud rate) and each
camera is opened through the RLinf ``WebcamCamera`` driver to confirm it
produces a frame. Camera discovery and probing follow the same contract as
RLinf's own webcam smoke test (``toolkits/lekiwi/test_webcam.py``):
``WebcamCamera.discover()`` names the devices and a short-lived
``WebcamCamera`` connection decides which one delivers frames. Geometry and
drive limits are policy, not discovery, so they are written as conservative
defaults and named in the report as fields to review.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

#: Wheel motor IDs (left, back, right). Must match ``WHEEL_IDS`` in
#: rlinf/robotics/parts/mobility/lekiwi.py at the installed RLinf revision.
EXPECTED_WHEEL_IDS = (7, 8, 9)

_PLACEHOLDER_PORT = "/dev/ttyACM0"
_PLACEHOLDER_SERIAL = "<front-camera-serial>"


def list_serial_ports(dev_root: Path = Path("/dev")) -> list[str]:
    """Candidate wheel-bus ports, most stable spelling first.

    ``/dev/serial/by-id`` names survive re-enumeration, so they are preferred
    over the plain ``/dev/ttyACM*`` / ``/dev/ttyUSB*`` nodes they point at.
    """
    candidates: list[str] = []
    by_id = dev_root / "serial" / "by-id"
    if by_id.is_dir():
        candidates.extend(str(entry) for entry in sorted(by_id.iterdir()))
    for pattern in ("ttyACM*", "ttyUSB*", "tty.usb*"):
        candidates.extend(str(entry) for entry in sorted(dev_root.glob(pattern)))
    unique: dict[str, str] = {}
    for path in candidates:
        unique.setdefault(_realpath(path), path)
    return list(unique.values())


def _realpath(path: str) -> str:
    import os

    return os.path.realpath(path)


def scan_wheel_bus(port: str) -> dict[int, list[int]]:
    """Ping every baud rate on ``port``; returns ``{baudrate: [motor ids]}``."""
    from lerobot.motors.feetech import FeetechMotorsBus

    return FeetechMotorsBus.scan_port(port)


def pick_wheel_port(
    ports: list[str],
) -> tuple[str | None, dict[str, dict[int, list[int]]]]:
    """Return the port whose bus answers for every wheel, plus all scan reports.

    A port where only some wheel ids answer is reported but not chosen: the
    env server's bus handshake pings every wheel, so a partial port would fail
    at startup anyway. A ``None`` port means no candidate is the wheel bus.
    """
    reports: dict[str, dict[int, list[int]]] = {}
    for port in ports:
        try:
            found = scan_wheel_bus(port)
        except Exception:  # noqa: BLE001 - a busy or absent port is a result
            found = {}
        if found:
            reports[port] = found
    expected = set(EXPECTED_WHEEL_IDS)

    def wheels_found(port: str) -> int:
        ids = {motor_id for ids in reports[port].values() for motor_id in ids}
        return len(expected & ids)

    for port in reports:
        if wheels_found(port) == len(expected):
            return port, reports
    return None, reports


def discover_cameras() -> list[str]:
    """Camera serials from the RLinf webcam driver's own discovery.

    ``WebcamCamera.discover()`` returns serials the driver itself resolves
    (``video0``-style names on Linux, numeric indices elsewhere), so whatever
    it lists here is a valid ``serial`` value in the site YAML. The list can
    include metadata nodes that carry no frames; :func:`probe_camera` filters
    those out.
    """
    from rlinf.robotics.parts.cameras.webcam import WebcamCamera

    return sorted(WebcamCamera.discover())


def probe_camera(serial: str) -> dict[str, Any]:
    """Open one camera through the RLinf driver and confirm it delivers a frame.

    Follows the smoke-test contract in ``toolkits/lekiwi/test_webcam.py``:
    connect, wait for a frame within 3 s, read one observation, disconnect.
    An error is reported with its text: a missing ``libGL`` or an unopenable
    device reads differently from a node that opens but never frames.
    """
    from rlinf.robotics.parts.cameras import CameraInfo
    from rlinf.robotics.parts.cameras.webcam import WebcamCamera

    camera = WebcamCamera(
        CameraInfo(name="detect", serial_number=serial, camera_type="webcam")
    )
    try:
        camera.connect()
        if not camera.is_ready(timeout=3.0):
            return {"serial": serial, "ok": False, "error": "no frame within 3 s"}
        frame = camera.get_observation()["frame"]
        height, width = frame.shape[:2]
        return {"serial": serial, "ok": True, "width": width, "height": height}
    except Exception as exc:  # noqa: BLE001 - an unopenable node is a result
        return {"serial": serial, "ok": False, "error": str(exc)}
    finally:
        camera.disconnect()


def build_site_config(port: str | None, camera_serials: list[str]) -> dict[str, Any]:
    """Assemble the site YAML structure from the detection results."""
    serials = camera_serials or [_PLACEHOLDER_SERIAL]
    devices: dict[str, dict[str, Any]] = {}
    for index, serial in enumerate(serials):
        name = "front" if index == 0 else f"extra_{index}"
        devices[name] = {
            "serial": serial,
            "type": "webcam",
            "main": index == 0,
        }
    return {
        "robot": {
            "port": port or _PLACEHOLDER_PORT,
            "wheel_radius": 0.05,
            "base_radius": 0.125,
            "max_raw": 3000,
        },
        "cameras": {"devices": devices},
        "drive": {
            "step_frequency": 10.0,
            "max_linear_velocity": 0.3,
            "max_angular_velocity": 60.0,
            "frame_size": [128, 128],
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Detect the LeKiwi wheel bus and cameras; write a site YAML."
    )
    parser.add_argument(
        "--output", default="lekiwi-site.yaml", help="path of the YAML to write"
    )
    parser.add_argument(
        "--skip-motors",
        action="store_true",
        help="do not ping the serial bus; emit a placeholder port",
    )
    parser.add_argument(
        "--skip-cameras",
        action="store_true",
        help="do not open the cameras; emit a placeholder serial",
    )
    args = parser.parse_args(argv)

    ports: list[str] = []
    wheel_port: str | None = None
    bus_reports: dict[str, dict[int, list[int]]] = {}
    if not args.skip_motors:
        ports = list_serial_ports()
        wheel_port, bus_reports = pick_wheel_port(ports)

    camera_serials: list[str] = []
    probed: list[dict[str, Any]] = []
    if not args.skip_cameras:
        probed = [probe_camera(serial) for serial in discover_cameras()]
        camera_serials = [report["serial"] for report in probed if report["ok"]]

    config = build_site_config(wheel_port, camera_serials)

    print("== LeKiwi site detection ==")
    if not args.skip_motors:
        print(f"serial ports scanned: {ports or 'none found'}")
        for port, found in bus_reports.items():
            ids = sorted({motor_id for ids in found.values() for motor_id in ids})
            print(f"  {port}: motor ids {ids}")
        if wheel_port is None:
            print(
                "no port answered for wheel ids "
                f"{list(EXPECTED_WHEEL_IDS)}; check the USB cable and power, "
                "or pass --skip-motors to emit a placeholder"
            )
        else:
            print(f"wheel bus: {wheel_port}")
    if not args.skip_cameras:
        if probed:
            for report in probed:
                if report["ok"]:
                    size = f"{report['width']}x{report['height']}"
                    print(f"camera {report['serial']}: ok at {size}")
                else:
                    error = report.get("error") or "no frame"
                    print(f"camera {report['serial']}: failed ({error})")
        else:
            print(
                "no cameras found by WebcamCamera.discover(); plug the camera in "
                "and retry, or fill cameras.devices.front.serial by hand"
            )

    output = Path(args.output)
    _write_yaml(output, config)
    print(f"wrote {output}")

    print("review before the first run:")
    print("  - robot.wheel_radius / robot.base_radius: measure with a ruler (m)")
    print("  - drive.max_linear_velocity / max_angular_velocity: policy; start low")
    print("  - drive.frame_size: camera output size; raise from 128x128 if needed")
    print("  - drive.step_frequency: control rate; 10 Hz suits the STS3215 bus")
    return 0


def _write_yaml(path: Path, config: dict[str, Any]) -> None:
    import yaml

    path.write_text(yaml.safe_dump(config, sort_keys=False))
    return None


if __name__ == "__main__":
    raise SystemExit(main())
