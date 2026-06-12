"""File-backed client for the interactive driver.

This preserves the original protocol: the tool process writes
``command.json`` and waits for the driver to append to ``states.json``.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any


class FileDriverClient:
    """Client for the existing workdir-backed driver protocol and artifacts."""

    def __init__(self, workdir: str | os.PathLike):
        self.workdir = Path(workdir)

    @property
    def command_path(self) -> Path:
        return self.workdir / "command.json"

    def load_states(self) -> list:
        path = self.workdir / "states.json"
        if not path.exists():
            return []
        try:
            with open(path) as f:
                arr = json.load(f)
            if isinstance(arr, list):
                return arr
        except Exception:
            pass
        return []

    def latest_step(self) -> int | None:
        arr = self.load_states()
        if not arr:
            return None
        return len(arr) - 1

    def load_step(self, step: int | None = None) -> dict:
        arr = self.load_states()
        if not arr:
            raise FileNotFoundError(f"no states.json entries in {self.workdir}")
        nn = len(arr) - 1 if step is None else int(step)
        if nn < 0 or nn >= len(arr) or arr[nn] is None:
            raise IndexError(f"step {nn} not present in states.json (len={len(arr)})")
        entry = arr[nn]
        if not isinstance(entry, dict):
            raise ValueError(f"states.json step {nn} is not an object")
        return entry

    def get_image_paths(self, step: int) -> dict[str, Path]:
        nn_str = f"{step:02d}"
        image_path = self.workdir / "images" / f"image_{nn_str}.png"
        image_cam_path = self.workdir / "images_cam" / f"image_cam_{nn_str}.png"
        paths: dict[str, Path] = {}
        if image_path.exists():
            paths["image"] = image_path
        if image_cam_path.exists():
            paths["image_cam"] = image_cam_path
        return paths

    def load_image(self, step: int, kind: str = "agent") -> bytes | None:
        nn_str = f"{int(step):02d}"
        if kind == "agent":
            path = self.workdir / "images" / f"image_{nn_str}.png"
        elif kind == "camera":
            path = self.workdir / "images_cam" / f"image_cam_{nn_str}.png"
        else:
            raise ValueError(f"unknown image kind: {kind}")
        if not path.exists():
            return None
        return path.read_bytes()

    def load_camera_meta(self) -> dict[str, Any]:
        path = self.workdir / "camera_meta.json"
        with open(path) as f:
            meta = json.load(f)
        return meta

    def load_depth(self, step: int) -> Any:
        import numpy as np

        depth_path = self.workdir / "depths" / f"depth_{step:02d}.npy"
        return np.load(depth_path)

    def request(
        self,
        method: str,
        params: dict | None = None,
        *,
        timeout_s: float | None = None,
    ) -> dict:
        """Perform one file-transport request.

        File transport owns the driver command and artifact boundary. Tool-level
        interpretation stays in ``tools/frontend.py``.
        """
        params = params or {}
        if method != "send_command":
            return {"error": f"file transport does not handle method: {method}"}
        current_step = params.get("current_step")
        if current_step is None:
            current_step = self.latest_step()
        if current_step is None:
            return {"error": "no states.json (or empty); driver not ready"}
        return self.send_command(
            params.get("command"),
            current_step=int(current_step),
            timeout_s=timeout_s if timeout_s is not None else 600.0,
        )

    def send_command(
        self,
        command: dict,
        *,
        current_step: int | None = None,
        timeout_s: float = 600.0,
    ) -> dict:
        """Write one command file and wait until the state trace advances."""
        if not self.workdir.exists():
            return {"error": f"WORKDIR {self.workdir} missing; driver not started"}
        if not isinstance(command, dict):
            return {"error": "send_command requires object param 'command'"}
        if current_step is None:
            current_step = self.latest_step()
        if current_step is None:
            return {"error": "no states.json (or empty); driver not ready"}

        next_step = current_step + 1

        tmp_path = self.workdir / "command.json.tmp"
        with open(tmp_path, "w") as f:
            json.dump(command, f)
        os.replace(tmp_path, self.command_path)

        t0 = time.time()
        while True:
            latest = self.latest_step()
            if latest is not None and latest >= next_step:
                break
            time.sleep(0.5)
            if time.time() - t0 > timeout_s:
                return {
                    "error": (
                        f"timeout after {timeout_s}s waiting for step {next_step} "
                        f"in states.json (still at step {latest})"
                    ),
                    "command_sent": command,
                }

        return {
            "step": next_step,
            "agent_elapsed_s": round(time.time() - t0, 1),
        }

    def close(self) -> None:
        return None


# Compatibility name from the command-only abstraction.
FileTransportClient = FileDriverClient
