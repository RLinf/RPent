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

"""Offline tests for the LeKiwi site-config detection script."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import yaml

from robots.lekiwi import detect


def _fake_module(name: str, **attrs: Any) -> ModuleType:
    module = ModuleType(name)
    for key, value in attrs.items():
        setattr(module, key, value)
    return module


def _make_dev_tree(tmp_path: Path) -> Path:
    dev = tmp_path / "dev"
    (dev / "serial" / "by-id").mkdir(parents=True)
    return dev


def test_list_serial_ports_prefers_by_id_spelling(tmp_path: Path) -> None:
    dev = _make_dev_tree(tmp_path)
    (dev / "ttyACM0").touch()
    (dev / "ttyACM1").touch()
    # by-id names are symlinks to the tty nodes; use a relative target so the
    # realpath resolution stays inside the fake dev tree.
    os.symlink("../../ttyACM0", dev / "serial" / "by-id" / "usb-FTDI_FT232R_ABC-if00")

    ports = detect.list_serial_ports(dev)

    assert str(dev / "serial" / "by-id" / "usb-FTDI_FT232R_ABC-if00") in ports
    assert str(dev / "ttyACM1") in ports
    # The symlink and its target dedupe to one stable entry.
    assert str(dev / "ttyACM0") not in ports


def test_pick_wheel_port_requires_every_wheel_id(monkeypatch) -> None:
    scans = {
        "/dev/ttyACM0": {1000000: [7, 8, 9]},
        "/dev/ttyUSB0": {1000000: [7]},
    }
    monkeypatch.setattr(detect, "scan_wheel_bus", lambda port: scans[port])

    port, reports = detect.pick_wheel_port(["/dev/ttyACM0", "/dev/ttyUSB0"])

    assert port == "/dev/ttyACM0"
    assert set(reports) == set(scans)


def test_pick_wheel_port_reports_none_without_full_wheel_set(monkeypatch) -> None:
    monkeypatch.setattr(detect, "scan_wheel_bus", lambda port: {1000000: [7]})

    port, _reports = detect.pick_wheel_port(["/dev/ttyUSB0"])

    # Partial overlap is not a wheel bus: report nothing rather than a guess.
    assert port is None


def test_pick_wheel_port_tolerates_a_busy_port(monkeypatch) -> None:
    def scan(port: str) -> dict:
        if port == "/dev/ttyUSB0":
            raise RuntimeError("device busy")
        return {1000000: [7, 8, 9]}

    monkeypatch.setattr(detect, "scan_wheel_bus", scan)

    port, reports = detect.pick_wheel_port(["/dev/ttyUSB0", "/dev/ttyACM0"])

    assert port == "/dev/ttyACM0"
    assert "/dev/ttyUSB0" not in reports


def test_build_site_config_marks_first_camera_main() -> None:
    config = detect.build_site_config(
        "/dev/ttyACM0",
        ["video0", "video2"],
    )

    devices = config["cameras"]["devices"]
    assert set(devices) == {"front", "extra_1"}
    assert devices["front"] == {
        "serial": "video0",
        "type": "webcam",
        "main": True,
    }
    assert devices["extra_1"]["main"] is False
    assert config["robot"]["port"] == "/dev/ttyACM0"
    assert config["drive"]["max_linear_velocity"] == 0.3


def test_build_site_config_emits_placeholders_when_detection_finds_nothing() -> None:
    config = detect.build_site_config(None, [])

    assert config["robot"]["port"] == "/dev/ttyACM0"
    assert config["cameras"]["devices"]["front"]["main"] is True


def test_probe_camera_reports_why_it_failed(monkeypatch) -> None:
    # A connect failure must surface its own text: on the robot, a cv2 that
    # cannot import (missing libGL) read as "no frame" and hid the real error.
    class MissingLibGLCamera:
        def __init__(self, _info: object) -> None:
            return None

        def connect(self) -> None:
            raise ImportError("libGL.so.1: cannot open shared object file")

        def disconnect(self) -> None:
            return None

    monkeypatch.setitem(
        sys.modules,
        "rlinf.robotics.parts.cameras",
        _fake_module("rlinf.robotics.parts.cameras", CameraInfo=lambda **_k: None),
    )
    monkeypatch.setitem(
        sys.modules,
        "rlinf.robotics.parts.cameras.webcam",
        _fake_module(
            "rlinf.robotics.parts.cameras.webcam", WebcamCamera=MissingLibGLCamera
        ),
    )

    report = detect.probe_camera("usb-USB_Camera_USB_Camera_2505280051-video-index0")

    assert report["ok"] is False
    assert "libGL.so.1" in report["error"]


def test_main_writes_a_config_loadable_by_runtime_config(
    fake_rlinf_realworld_modules, tmp_path: Path, monkeypatch, capsys
) -> None:
    monkeypatch.setattr(detect, "list_serial_ports", lambda *a, **k: ["/dev/ttyACM0"])
    monkeypatch.setattr(detect, "scan_wheel_bus", lambda port: {1000000: [7, 8, 9]})
    monkeypatch.setattr(detect, "discover_cameras", lambda: ["video0", "video1"])
    monkeypatch.setattr(
        detect,
        "probe_camera",
        lambda serial: {
            "serial": serial,
            "ok": serial == "video0",
            "width": 640,
            "height": 480,
            "error": None if serial == "video0" else "no frame within 3 s",
        },
    )
    output = tmp_path / "site.yaml"

    exit_code = detect.main(["--output", str(output)])

    assert exit_code == 0
    config = yaml.safe_load(output.read_text())
    assert config["robot"]["port"] == "/dev/ttyACM0"
    # Only the camera that delivered a frame is written.
    assert config["cameras"]["devices"]["front"]["serial"] == "video0"
    assert config["cameras"]["devices"]["front"]["main"] is True

    report = capsys.readouterr().out
    assert "wheel bus: /dev/ttyACM0" in report
    assert "camera video1: failed (no frame within 3 s)" in report
    assert "review before the first run" in report

    # The generated YAML must validate against the same loader the env server
    # uses, so a machine-generated file cannot drift from the example schema.
    from robots.lekiwi.runtime_config import load_runtime_config

    runtime = load_runtime_config(output)
    assert runtime.rlinf.env.eval.main_image_key == "front"
    hardware = runtime.rlinf.cluster.node_groups[0].hardware.configs[0]
    assert hardware["port"] == "/dev/ttyACM0"
    assert hardware["camera_serials"] == ["video0"]


def test_main_reports_a_missing_wheel_bus(monkeypatch, capsys, tmp_path: Path) -> None:
    monkeypatch.setattr(detect, "list_serial_ports", lambda *a, **k: ["/dev/ttyUSB0"])
    monkeypatch.setattr(detect, "scan_wheel_bus", lambda port: {})
    monkeypatch.setattr(detect, "discover_cameras", lambda: [])
    output = tmp_path / "site.yaml"

    detect.main(["--output", str(output)])

    report = capsys.readouterr().out
    assert "no port answered" in report
    config = yaml.safe_load(output.read_text())
    assert config["robot"]["port"] == "/dev/ttyACM0"  # placeholder
