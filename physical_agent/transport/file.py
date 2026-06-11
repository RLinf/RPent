"""File-based transport for the interactive driver.

This preserves the original protocol: the tool process writes
``command.json`` and waits for the driver to append to ``states.json``.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path


class FileTransportClient:
    """Client for the existing workdir-backed driver protocol."""

    def __init__(self, workdir: str | os.PathLike):
        self.workdir = Path(workdir)

    @property
    def command_path(self) -> Path:
        return self.workdir / "command.json"

    def _load_states(self) -> list:
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

    def _latest_step(self) -> int | None:
        arr = self._load_states()
        if not arr:
            return None
        return len(arr) - 1

    def request(
        self,
        method: str,
        params: dict | None = None,
        *,
        timeout_s: float | None = None,
    ) -> dict:
        """Perform one file-transport request.

        File transport only owns command delivery. Tool-level state,
        camera, and depth interpretation stays in ``tools/frontend.py``.
        """
        params = params or {}
        if method != "send_command":
            return {"error": f"file transport does not handle method: {method}"}
        current_step = params.get("current_step")
        if current_step is None:
            current_step = self._latest_step()
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
        current_step: int,
        timeout_s: float = 600.0,
    ) -> dict:
        """Write one command file and wait until the state trace advances."""
        if not self.workdir.exists():
            return {"error": f"WORKDIR {self.workdir} missing; driver not started"}
        if not isinstance(command, dict):
            return {"error": "send_command requires object param 'command'"}

        next_step = current_step + 1

        tmp_path = self.workdir / "command.json.tmp"
        with open(tmp_path, "w") as f:
            json.dump(command, f)
        os.replace(tmp_path, self.command_path)

        t0 = time.time()
        while True:
            latest = self._latest_step()
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
