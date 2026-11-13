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

"""Save stereo frames from the Env's camera; never open another video device."""

from __future__ import annotations

import json
import shutil
import tempfile
import threading
import time
from pathlib import Path


class StereoRecording:
    """One bounded image recording sharing the Env-owned native frame cache."""

    def __init__(self, camera, config: dict, identity: str) -> None:
        self.camera, self.config = camera, config
        self.state = {
            "identity": identity,
            "phase": "starting",
            "format": "stereo_png",
            "frames": 0,
        }
        self.ending = threading.Event()
        self.thread = threading.Thread(target=self.run, daemon=True)

    @property
    def alive(self) -> bool:
        return self.thread.is_alive()

    def start(self) -> None:
        self.thread.start()

    def stop(self) -> None:
        self.ending.set()
        self.thread.join()

    def run(self) -> None:
        import queue

        import cv2

        try:
            root = Path(self.config["root"])
            root.mkdir(parents=True, exist_ok=True)
            directory = Path(tempfile.mkdtemp(prefix="capture-", dir=root))
            for side in ("left", "right"):
                (directory / side).mkdir()
            shutil.copyfile(self.config["calibration"], directory / "calibration.conf")
            width, height = self.camera.camera_info.resolution
            offset_ns = time.time_ns() - time.monotonic_ns()
            metadata = {
                "eye_size": [width // 2, height],
                "serial": self.config["serial"],
                "calibration": "calibration.conf",
            }
            (directory / "stereo.json").write_text(json.dumps(metadata, indent=2))
            self.state["path"] = str(directory)
            started = last_received = time.monotonic()
            previous = 0.0
            while (
                not self.ending.is_set()
                and time.monotonic() - started
                < self.config["maximum_recording_seconds"]
            ):
                try:
                    stamp, image = self.camera.get_frame(
                        timeout=0.5, max_age=0.5, return_timestamp=True
                    )
                except queue.Empty:
                    if time.monotonic() - last_received > 3:
                        raise RuntimeError("Stereo frame stream stopped")
                    self.ending.wait(0.01)
                    continue
                if (
                    stamp <= previous
                    or stamp - previous < 1 / self.config["sample_fps"]
                ):
                    self.ending.wait(0.01)
                    continue
                previous, last_received = stamp, time.monotonic()
                index = self.state["frames"]
                stamp_ns = round(stamp * 1e9) + offset_ns
                name = f"{stamp_ns // 10**9:010d}.{stamp_ns % 10**9:09d}.png"
                names = [f"left/{name}", f"right/{name}"]
                for name, eye in zip(
                    names, (image[:, : width // 2], image[:, width // 2 :])
                ):
                    if not cv2.imwrite(
                        str(directory / name), eye, [cv2.IMWRITE_PNG_COMPRESSION, 1]
                    ):
                        raise OSError(f"Could not write {name}")
                self.state.update(frames=index + 1, phase="recording")
            self.state["phase"] = "recorded"
        except Exception as exc:
            self.state.update(phase="error", error=str(exc))
