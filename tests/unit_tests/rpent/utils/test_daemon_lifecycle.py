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
import subprocess
import sys
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from rpent.utils.daemon import ParentDeathWatcher, ProcessDaemon
from rpent.utils.rpc import make_rpc_client, wait_for_endpoint, wait_for_ready


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


def test_failed_spawn_closes_log(monkeypatch, tmp_path):
    logs = []

    def fail_spawn(*args, **kwargs):
        logs.append(kwargs["stdout"])
        raise OSError("cannot spawn")

    monkeypatch.setattr(subprocess, "Popen", fail_spawn)
    daemon, _ = _python_daemon(tmp_path, "pass")
    with pytest.raises(OSError, match="cannot spawn"):
        daemon.start()
    assert logs[0].closed
    daemon.stop()


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


def test_parent_watcher_uses_raw_fd_and_stops_without_fatal_thread() -> None:
    read_fd, write_fd = os.pipe()
    calls: list[bool] = []
    try:
        watcher = ParentDeathWatcher(lambda: calls.append(True), fd=read_fd)
        os.close(write_fd)
        _wait_until(lambda: calls == [True])
        watcher.close()
        assert not watcher._thread.is_alive()
    finally:
        for fd in (read_fd, write_fd):
            try:
                os.close(fd)
            except OSError:
                pass


@pytest.mark.parametrize("eof", [False, True])
def test_watcher_subprocess_eof_and_cancellation_exit_cleanly(eof):
    script = """
import threading
from rpent.utils.daemon import watch_parent_death
done = threading.Event()
watcher = watch_parent_death(done.set)
print("ready", flush=True)
if EOF:
    assert done.wait(5)
watcher.close()
assert not watcher._thread.is_alive()
assert done.is_set() == EOF
""".replace("EOF", repr(eof))
    proc = subprocess.Popen(
        [sys.executable, "-c", script],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        assert proc.stdout.readline().strip() == "ready"
        if eof:
            proc.stdin.close()
        assert proc.wait(timeout=10) == 0
        assert "Fatal Python error" not in proc.stderr.read()
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait()
        proc.stdin.close()
        proc.stdout.close()
        proc.stderr.close()


def test_daemon_parent_watch_exits_zero_on_stop(tmp_path):
    script = """
import threading
from rpent.utils.daemon import watch_parent_death
done = threading.Event()
watcher = watch_parent_death(done.set)
print("ready", flush=True)
try:
    done.wait()
finally:
    watcher.close()
"""
    daemon, log = _python_daemon(tmp_path, script)
    daemon.cmd.append("--parent-watch")
    daemon.start()
    try:
        _wait_until(lambda: "ready" in log.read_text())
    finally:
        daemon.stop(timeout=2)
    assert daemon.poll() == 0
    assert "Fatal Python error" not in log.read_text()


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


def test_wait_for_endpoint_returns_the_port_a_server_bound(tmp_path: Path) -> None:
    log_path = tmp_path / "facade.log"
    daemon = ProcessDaemon(
        "facade",
        [
            sys.executable,
            "-c",
            "from rpent.utils.rpc import RpcFacade; "
            "RpcFacade().serve(transport='http', host='127.0.0.1', port=0, "
            "parent_watch=True)",
            "--parent-watch",
        ],
        log_path=str(log_path),
    )
    daemon.start()
    try:
        endpoint = wait_for_endpoint(daemon, timeout_s=30.0)
        assert not endpoint.endswith(":0")
        wait_for_ready(make_rpc_client(endpoint), timeout_s=30.0, daemon=daemon)
    finally:
        daemon.stop(timeout=2.0)


def test_wait_for_endpoint_fails_fast_when_daemon_exits(tmp_path: Path) -> None:
    daemon, _ = _python_daemon(tmp_path, "import sys; sys.exit(3)")
    daemon.start()
    try:
        with pytest.raises(RuntimeError, match="exit code 3"):
            wait_for_endpoint(daemon, timeout_s=30.0)
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


@pytest.mark.skipif(sys.platform != "linux", reason="uses Linux owned process groups")
def test_owned_process_group_reaps_cold_children_without_stopping_other_daemons(
    tmp_path,
):
    import json

    grandchild = (
        "import signal,time; "
        "signal.signal(signal.SIGTERM,signal.SIG_IGN); time.sleep(60)"
    )
    ids = tmp_path / "children.json"
    child = (
        "import json,os,signal,subprocess,sys,time; from pathlib import Path; "
        "signal.signal(signal.SIGTERM,signal.SIG_IGN); "
        f"p=subprocess.Popen([sys.executable,'-c',{grandchild!r}]); "
        f"ids=Path({str(ids)!r}); temporary=ids.with_suffix('.tmp'); "
        "temporary.write_text(json.dumps([os.getpid(),p.pid])); temporary.replace(ids); "
        "time.sleep(60)"
    )
    parent = (
        "import subprocess,sys,time; "
        f"subprocess.Popen([sys.executable,'-c',{child!r}]); time.sleep(60)"
    )
    daemon = ProcessDaemon(
        "cold-owned-tree",
        [sys.executable, "-c", parent],
        log_path=str(tmp_path / "owned.log"),
        cwd=str(tmp_path),
        start_new_session=True,
    )
    foreign = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])

    def running(pid):
        stat = Path(f"/proc/{pid}/stat")
        if not stat.exists():
            return False
        return stat.read_text().rpartition(")")[2].split()[0] != "Z"

    try:
        daemon.start()
        _wait_until(ids.exists)
        children = json.loads(ids.read_text())
        assert all(running(pid) for pid in children)
        daemon.stop(timeout=0.1)
        assert daemon.poll() is not None
        assert not any(running(pid) for pid in children)
        assert foreign.poll() is None
        daemon.stop(timeout=0.1)
    finally:
        daemon.stop(timeout=0.1)
        foreign.terminate()
        foreign.wait(timeout=3)


def test_process_group_checks_coalesce_pending_requests(monkeypatch):
    from concurrent.futures import ThreadPoolExecutor

    from rpent.utils.daemon import _ProcessGroupScanner

    scanner = _ProcessGroupScanner()
    entered, finish = threading.Event(), threading.Event()
    snapshots = []

    def snapshot(groups):
        snapshots.append(set(groups))
        if len(snapshots) == 1:
            entered.set()
            assert finish.wait(timeout=5)
        return {group: [group * 10] for group in groups}

    monkeypatch.setattr(scanner, "_snapshot", snapshot)
    with ThreadPoolExecutor(max_workers=32) as pool:
        first = pool.submit(scanner.live, 1)
        assert entered.wait(timeout=3)
        rest = [pool.submit(scanner.live, group) for group in range(2, 33)]
        _wait_until(lambda: len(scanner._pending) == 31)
        finish.set()
        assert first.result(timeout=3) == [10]
        assert [future.result(timeout=3) for future in rest] == [
            [group * 10] for group in range(2, 33)
        ]
    assert snapshots == [{1}, set(range(2, 33))]


def test_process_group_check_never_reuses_an_earlier_zero_snapshot(monkeypatch):
    from concurrent.futures import ThreadPoolExecutor

    from rpent.utils.daemon import _ProcessGroupScanner

    scanner = _ProcessGroupScanner()
    entered, finish = threading.Event(), threading.Event()
    count = 0

    def snapshot(groups):
        nonlocal count
        count += 1
        if count == 1:
            entered.set()
            assert finish.wait(timeout=5)
            return {}
        return {7: [701]}

    monkeypatch.setattr(scanner, "_snapshot", snapshot)
    with ThreadPoolExecutor(max_workers=2) as pool:
        older = pool.submit(scanner.live, 7)
        assert entered.wait(timeout=3)
        newer = pool.submit(scanner.live, 7)
        _wait_until(lambda: len(scanner._pending) == 1)
        finish.set()
        assert older.result(timeout=3) == []
        assert newer.result(timeout=3) == [701]
    assert count == 2


def test_process_group_scan_failure_reaches_callers_and_does_not_wedge_next_check(
    monkeypatch,
):
    from rpent.utils.daemon import _ProcessGroupScanner

    scanner = _ProcessGroupScanner()

    def broken(groups):
        raise OSError("procfs unavailable")

    monkeypatch.setattr(scanner, "_snapshot", broken)
    with pytest.raises(OSError, match="procfs unavailable"):
        scanner.live(7)
    monkeypatch.setattr(scanner, "_snapshot", lambda groups: {7: [701]})
    assert scanner.live(7) == [701]


@pytest.mark.skipif(sys.platform != "linux", reason="uses Linux process groups")
def test_missing_process_group_does_not_scan_proc(monkeypatch):
    from rpent.utils import daemon as module

    def unexpected(group):
        raise AssertionError("an absent process group needs no procfs enumeration")

    monkeypatch.setattr(module._PROCESS_GROUPS, "live", unexpected)
    assert ProcessDaemon._live_process_group(10_000_000) == []


@pytest.mark.skipif(sys.platform != "linux", reason="uses Linux process identity")
def test_process_group_reuse_cannot_signal_a_foreign_group(tmp_path):
    foreign = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(60)"],
        start_new_session=True,
    )
    owner = ProcessDaemon(
        "retired-owner", [sys.executable, "-c", "pass"], start_new_session=True
    )
    owner._process_group = foreign.pid
    owner._process_group_start_ticks = owner._leader_start_ticks(foreign.pid) + 1
    try:
        with pytest.raises(RuntimeError, match="identifier was reused"):
            owner._stop_process_group(timeout=0.1)
        assert foreign.poll() is None
    finally:
        owner._process_group = None
        foreign.terminate()
        foreign.wait(timeout=3)


@pytest.mark.skipif(sys.platform != "linux", reason="uses Linux owned process groups")
def test_concurrent_owned_process_groups_stop_without_cross_signals(tmp_path):
    from concurrent.futures import ThreadPoolExecutor

    daemons = []
    foreign = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    try:
        for index in range(8):
            script = (
                "import signal,time; "
                "signal.signal(signal.SIGTERM,signal.SIG_IGN); "
                "print('ready',flush=True);time.sleep(60)"
            )
            daemon = ProcessDaemon(
                f"owned-{index}",
                [sys.executable, "-c", script],
                log_path=str(tmp_path / f"owned-{index}.log"),
                cwd=str(tmp_path),
                start_new_session=True,
            )
            daemon.start()
            daemons.append(daemon)
        _wait_until(
            lambda: all(
                "ready" in (tmp_path / f"owned-{index}.log").read_text()
                for index in range(8)
            )
        )
        with ThreadPoolExecutor(max_workers=8) as pool:
            stops = [pool.submit(daemon.stop, timeout=0.1) for daemon in daemons]
            for stop in stops:
                stop.result(timeout=10)
        assert all(daemon.poll() is not None for daemon in daemons)
        assert all(daemon._process_group is None for daemon in daemons)
        assert foreign.poll() is None
    finally:
        for daemon in daemons:
            daemon.stop(timeout=0.1)
        foreign.terminate()
        foreign.wait(timeout=3)
