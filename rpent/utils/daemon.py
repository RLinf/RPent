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

"""Subprocess RPC servers: spawn (parent side), death-watch (child side)."""

from __future__ import annotations

import os
import select
import socket
import subprocess
import threading
from typing import Callable

from rpent.utils.config import get_repo_root, get_rlinf_repo_path
from rpent.utils.logging import get_logger

logger = get_logger("daemon")


# ---------------------------------------------------------------------------
# Child side
# ---------------------------------------------------------------------------


class ParentDeathWatcher:
    """Cancellable stdin EOF watcher that never touches ``BufferedReader``."""

    def __init__(self, on_death: Callable[[], None], fd: int = 0) -> None:
        self._on_death = on_death
        self._fd = fd
        self._stop = threading.Event()
        # Owners stop and join this thread. A startup exception must not keep
        # the interpreter alive; raw fd reads are safe during finalization.
        self._thread = threading.Thread(
            target=self._run, name="parent-watch", daemon=True
        )
        self._thread.start()

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                ready, _, _ = select.select([self._fd], [], [], 0.1)
            except (OSError, ValueError):
                return
            if not ready:
                continue
            try:
                data = os.read(self._fd, 4096)
            except OSError:
                return
            if not data:
                if not self._stop.is_set():
                    self._on_death()
                return

    def close(self) -> None:
        """Cancel polling and wait for the watcher to finish."""
        self._stop.set()
        if self._thread is not threading.current_thread():
            self._thread.join(timeout=1.0)


def watch_parent_death(on_death: Callable[[], None]) -> ParentDeathWatcher:
    """Call ``on_death()`` once, from a background thread, when stdin hits EOF.

    The owner must close the returned handle during teardown. ProcessDaemon
    closes the pipe for graceful shutdown, or the OS closes it on parent death.
    Terminal input is polled without blocking; /dev/null produces immediate EOF.
    """

    return ParentDeathWatcher(on_death)


# ---------------------------------------------------------------------------
# Parent side
# ---------------------------------------------------------------------------


def pick_free_port(host: str = "127.0.0.1") -> int:
    """Return an unused TCP port on ``host``.

    There's a small race between the socket closing and the child binding;
    on 127.0.0.1 it's near-zero in practice.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((host, 0))
        return int(s.getsockname()[1])


class ProcessDaemon:
    """Wraps a subprocess server with ready-detection and lifecycle."""

    def __init__(
        self,
        name: str,
        cmd: list[str],
        *,
        env_overrides: dict[str, str] | None = None,
        log_path: str | None = None,
        cwd: str | None = None,
    ) -> None:
        self.name = name
        self.cmd = cmd
        self.subprocess_env = os.environ.copy()
        self.subprocess_env.update(env_overrides or {})
        # Put the RLinf checkout first on the child's PYTHONPATH
        rlinf_path = str(get_rlinf_repo_path())
        pythonpath = self.subprocess_env.get("PYTHONPATH", "")
        self.subprocess_env["PYTHONPATH"] = (
            f"{rlinf_path}{os.pathsep}{pythonpath}" if pythonpath else rlinf_path
        )
        self.log_path = log_path
        self.cwd = cwd
        self._proc: subprocess.Popen | None = None
        self._log_f = None

    def poll(self) -> int | None:
        """Return the subprocess exit code, or None if it's still running.

        None also before ``start()``. A non-None value means the child has
        exited — used by ``wait_for_ready`` to fail fast on early crashes.
        """
        return self._proc.poll() if self._proc is not None else None

    def start(self) -> None:
        """Spawn the subprocess. Returns immediately; does not wait for readiness."""
        self._log_f = (
            open(self.log_path, "a") if self.log_path else open(os.devnull, "w")
        )
        self._proc = subprocess.Popen(
            self.cmd,
            # stdin pipe lets the child detect our death via EOF; see
            # watch_parent_death above.
            stdin=subprocess.PIPE,
            stdout=self._log_f,
            stderr=subprocess.STDOUT,
            env=self.subprocess_env,
            cwd=self.cwd or get_repo_root(),
        )
        logger.info("%s spawned (pid=%s)", self.name, self._proc.pid)

    def stop(self, timeout: float = 15.0) -> None:
        if self._proc is not None and self._proc.poll() is None:
            # Closing the parent-watch pipe lets cooperative servers unwind
            # their RPC/facade cleanup and exit with status 0.
            if "--parent-watch" in self.cmd and self._proc.stdin is not None:
                try:
                    self._proc.stdin.close()
                    self._proc.wait(timeout=timeout)
                except (BrokenPipeError, OSError, subprocess.TimeoutExpired):
                    pass
            if self._proc.poll() is None:
                self._proc.terminate()
                try:
                    self._proc.wait(timeout=timeout)
                except subprocess.TimeoutExpired:
                    self._proc.kill()
                    self._proc.wait()
        if self._proc is not None and self._proc.stdin is not None:
            self._proc.stdin.close()
        if self._log_f is not None:
            try:
                self._log_f.close()
            except Exception:
                pass
            self._log_f = None
