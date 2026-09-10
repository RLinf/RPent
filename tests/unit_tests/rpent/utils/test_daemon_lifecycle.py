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

from __future__ import annotations

import os
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from rpent.utils.daemon import ProcessDaemon
from rpent.utils.rpc import wait_for_ready


def _wait_until(
    predicate: Callable[[], bool],
    *,
    timeout_s: float = 3.0,
) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.01)
    raise AssertionError("condition was not met before timeout")


def _python_daemon(
    tmp_path: Path,
    script: str,
    *,
    name: str = "contract-daemon",
) -> tuple[ProcessDaemon, Path]:
    log_path = tmp_path / f"{name}.log"
    return (
        ProcessDaemon(
            name,
            [sys.executable, "-c", script],
            log_path=str(log_path),
            cwd=str(tmp_path),
        ),
        log_path,
    )


class UnavailableClient:
    def call(self, *args: Any, **kwargs: Any) -> None:
        del args, kwargs
        raise ConnectionError("not ready")


def test_daemon_starts_logs_and_stops_idempotently(tmp_path: Path) -> None:
    daemon, log_path = _python_daemon(
        tmp_path,
        "import time; print('ready', flush=True); time.sleep(30)",
    )

    try:
        assert daemon.poll() is None
        daemon.start()
        _wait_until(lambda: log_path.exists() and "ready" in log_path.read_text())
        assert daemon.poll() is None
    finally:
        daemon.stop(timeout=1.0)

    assert daemon.poll() is not None
    assert log_path.read_text().strip() == "ready"
    daemon.stop(timeout=0.01)


def test_wait_for_ready_fails_fast_when_daemon_exits(tmp_path: Path) -> None:
    daemon, _ = _python_daemon(
        tmp_path,
        "import sys; sys.exit(7)",
        name="crashing-daemon",
    )
    daemon.start()

    try:
        with pytest.raises(RuntimeError, match="exited with code 7"):
            wait_for_ready(
                UnavailableClient(),
                timeout_s=3.0,
                poll_interval_s=0.01,
                daemon=daemon,
            )
    finally:
        daemon.stop(timeout=1.0)


@pytest.mark.skipif(sys.platform == "win32", reason="uses POSIX SIGTERM semantics")
def test_daemon_force_kills_after_terminate_timeout(tmp_path: Path) -> None:
    daemon, log_path = _python_daemon(
        tmp_path,
        "import signal, time; "
        "signal.signal(signal.SIGTERM, signal.SIG_IGN); "
        "print('ignoring-term', flush=True); "
        "time.sleep(30)",
        name="stubborn-daemon",
    )
    daemon.start()

    try:
        _wait_until(
            lambda: log_path.exists() and "ignoring-term" in log_path.read_text()
        )
        daemon.stop(timeout=0.05)
        _wait_until(lambda: daemon.poll() is not None)
    finally:
        daemon.stop(timeout=1.0)


def test_daemon_prepends_checkout_already_on_pythonpath(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkout = tmp_path / "rlinf"
    checkout.mkdir()
    resolved = str(checkout.resolve())
    monkeypatch.setenv("RPENT_RLINF_ROOT", str(checkout))
    # The checkout is already inherited on the path: it is prepended again
    # anyway — duplicate entries are harmless (Python ignores them).
    monkeypatch.setenv("PYTHONPATH", os.pathsep.join(("/other", resolved)))
    daemon = ProcessDaemon("env-probe", [sys.executable, "-c", "pass"])
    assert daemon.subprocess_env["PYTHONPATH"] == os.pathsep.join(
        (resolved, "/other", resolved)
    )


def test_daemon_sets_pythonpath_when_parent_has_none(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkout = tmp_path / "rlinf"
    checkout.mkdir()
    monkeypatch.setenv("RPENT_RLINF_ROOT", str(checkout))
    monkeypatch.delenv("PYTHONPATH", raising=False)
    daemon = ProcessDaemon("env-probe", [sys.executable, "-c", "pass"])
    assert daemon.subprocess_env["PYTHONPATH"] == str(checkout.resolve())


def test_daemon_prepends_rlinf_checkout_to_pythonpath(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkout = tmp_path / "rlinf"
    checkout.mkdir()
    monkeypatch.setenv("RPENT_RLINF_ROOT", str(checkout))
    monkeypatch.setenv("PYTHONPATH", "/other")
    daemon = ProcessDaemon("env-probe", [sys.executable, "-c", "pass"])
    assert daemon.subprocess_env["PYTHONPATH"] == os.pathsep.join(
        (str(checkout.resolve()), "/other")
    )


def test_daemon_prepends_missing_checkout_harmlessly(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    missing = tmp_path / "missing"
    monkeypatch.setenv("RPENT_RLINF_ROOT", str(missing))
    monkeypatch.setenv("PYTHONPATH", "/other")
    daemon = ProcessDaemon("env-probe", [sys.executable, "-c", "pass"])
    # Python ignores non-existent PYTHONPATH entries, so the child then
    # imports the installed rlinf package; the path is prepended anyway.
    assert daemon.subprocess_env["PYTHONPATH"] == os.pathsep.join(
        (str(missing.resolve()), "/other")
    )


def test_spawned_daemon_inherits_rlinf_pythonpath(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkout = tmp_path / "rlinf"
    checkout.mkdir()
    monkeypatch.setenv("RPENT_RLINF_ROOT", str(checkout))
    log_path = tmp_path / "pythonpath-probe.log"
    daemon = ProcessDaemon(
        "pythonpath-probe",
        [
            sys.executable,
            "-c",
            "import os; print('PYTHONPATH=' + os.environ.get('PYTHONPATH', ''))",
        ],
        log_path=str(log_path),
        cwd=str(tmp_path),
    )
    daemon.start()

    def _printed() -> bool:
        return log_path.exists() and "PYTHONPATH=" in log_path.read_text()

    try:
        _wait_until(_printed)
    finally:
        daemon.stop(timeout=1.0)

    value = log_path.read_text().strip().partition("PYTHONPATH=")[2]
    assert value.split(os.pathsep)[0] == str(checkout.resolve())
