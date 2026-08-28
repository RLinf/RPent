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

import fcntl
import hashlib
import json
import os
import signal
import stat
import subprocess
import threading
import time
from pathlib import Path
from unittest.mock import Mock

import pytest

from rpent.reproduce.robocasa import preliminary_driver
from rpent.reproduce.robocasa.deadline_supervisor import (
    CONTRACT_NAME,
    DEADLINE_PROTOCOL,
    FREEZE_NAME,
    GATE_NAME,
    RECEIPT_PREFIX,
    DeadlineController,
    DeadlineError,
    _atomic_json,
    _ObservedProcess,
    _open_gate,
    _read_trusted_json,
    _seal_committed_prefix,
    normalize_deadline_outcome,
    process_identity,
    scan_committed_prefix,
)
from rpent.reproduce.robocasa.executor import _timeout_driver_exit_expected


def _contract(
    workdir: Path,
    *,
    deadline_ns: int,
    started_ns: int | None = None,
    run_id: str = "run-1",
) -> tuple[dict, str]:
    descriptor = _open_gate(workdir / GATE_NAME, create=True)
    os.close(descriptor)
    identity = process_identity(os.getpid())
    assert identity is not None
    if started_ns is None:
        started_ns = min(time.monotonic_ns(), deadline_ns - 1)
    timeout_ns = deadline_ns - started_ns
    contract = {
        "schema_version": 1,
        "protocol": DEADLINE_PROTOCOL,
        "run_id": run_id,
        "nonce": "0" * 32,
        "started_monotonic_ns": started_ns,
        "deadline_monotonic_ns": deadline_ns,
        "timeout_ns": timeout_ns,
        "driver": {key: identity[key] for key in ("pid", "pgid", "start_time_ticks")},
        "external_deadline_sha256": "a" * 64,
    }
    digest = _atomic_json(
        workdir / CONTRACT_NAME,
        contract,
    )
    return contract, digest


def _publish(path: str | Path) -> None:
    Path(path).write_text("", encoding="utf-8")


def test_commit_gate_accepts_only_pre_deadline_publication(tmp_path: Path):
    workdir = tmp_path / "workdir"
    workdir.mkdir(mode=0o700)
    _contract(workdir, deadline_ns=time.monotonic_ns() + 1_000_000_000)

    marker = workdir / "done_01.flag"
    preliminary_driver._publish_done_before_deadline(_publish, marker, run_id="run-1")

    assert marker.is_file()
    receipt, _digest = _read_trusted_json(workdir / f"{RECEIPT_PREFIX}01.json")
    assert receipt["done_published_monotonic_ns"] < receipt["deadline_monotonic_ns"]


def test_commit_gate_rejects_command_that_finishes_after_deadline(tmp_path: Path):
    workdir = tmp_path / "workdir"
    workdir.mkdir(mode=0o700)
    _contract(workdir, deadline_ns=time.monotonic_ns() - 1)

    marker = workdir / "done_01.flag"
    with pytest.raises(SystemExit) as error:
        preliminary_driver._publish_done_before_deadline(
            _publish, marker, run_id="run-1"
        )

    assert error.value.code == 75
    assert not marker.exists()
    assert not (workdir / f"{RECEIPT_PREFIX}01.json").exists()


@pytest.mark.parametrize("tamper", ["extra_field", "driver_start_time"])
def test_commit_gate_rejects_non_exact_or_stale_driver_contract(
    tmp_path: Path, tamper: str
):
    workdir = tmp_path / "workdir"
    workdir.mkdir(mode=0o700)
    contract, _digest = _contract(
        workdir, deadline_ns=time.monotonic_ns() + 1_000_000_000
    )
    if tamper == "extra_field":
        contract["unexpected"] = True
    else:
        contract["driver"]["start_time_ticks"] += 1
    _atomic_json(workdir / CONTRACT_NAME, contract)

    marker = workdir / "done_01.flag"
    with pytest.raises(DeadlineError, match="does not match"):
        preliminary_driver._publish_done_before_deadline(
            _publish, marker, run_id="run-1"
        )

    assert not marker.exists()
    assert not (workdir / f"{RECEIPT_PREFIX}01.json").exists()


def test_commit_gate_removes_and_fsyncs_a_marker_published_across_deadline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    workdir = tmp_path / "workdir"
    workdir.mkdir(mode=0o700)
    (workdir / "done_00.flag").write_text("", encoding="utf-8")
    (workdir / "state_00.json").write_text(
        json.dumps({"step": 0, "success": False}) + "\n", encoding="utf-8"
    )
    (workdir / "state_01.json").write_text(
        json.dumps({"step": 1, "success": True}) + "\n", encoding="utf-8"
    )
    (workdir / "log_01.json").write_text(
        json.dumps({"command": {"action": "release"}}) + "\n",
        encoding="utf-8",
    )
    (workdir / "command_trace.jsonl").write_text(
        '{"action":"release"}\n', encoding="utf-8"
    )
    contract, contract_sha256 = _contract(workdir, deadline_ns=200, started_ns=100)
    clock = iter((199, 201))
    monkeypatch.setattr(preliminary_driver.time, "monotonic_ns", lambda: next(clock))

    marker = workdir / "done_01.flag"
    original_fsync = os.fsync
    directory_fsyncs: list[bool] = []

    def observe_fsync(descriptor: int) -> None:
        if stat.S_ISDIR(os.fstat(descriptor).st_mode):
            directory_fsyncs.append(marker.exists())
        original_fsync(descriptor)

    monkeypatch.setattr(
        "rpent.reproduce.robocasa.deadline_supervisor.os.fsync", observe_fsync
    )
    with pytest.raises(SystemExit) as error:
        preliminary_driver._publish_done_before_deadline(
            _publish, marker, run_id="run-1"
        )

    assert error.value.code == 75
    assert not marker.exists()
    assert directory_fsyncs[-1] is False
    receipt, _digest = _read_trusted_json(workdir / f"{RECEIPT_PREFIX}01.json")
    assert receipt["done_published_monotonic_ns"] == 201
    assert _seal_committed_prefix(workdir, contract, contract_sha256) == []
    prefix = scan_committed_prefix(workdir)
    assert prefix["prefix_step"] == 0
    assert prefix["success"] is False
    assert prefix["command_trace_sha256"] == hashlib.sha256(b"").hexdigest()
    assert "state_01.json" not in prefix["files_sha256"]
    assert "log_01.json" not in prefix["files_sha256"]
    gate_descriptor = _open_gate(workdir / GATE_NAME, create=False)
    try:
        fcntl.flock(gate_descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
    finally:
        os.close(gate_descriptor)


@pytest.mark.parametrize(("terminal_ns", "fires"), [(199, False), (200, True)])
def test_observed_planner_terminal_order_wins_over_delayed_watcher(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    terminal_ns: int,
    fires: bool,
):
    controller = DeadlineController(
        workdir=tmp_path,
        run_id="run-1",
        driver_identity={"pid": 1, "pgid": 1, "start_time_ticks": 1},
        timeout_seconds=1,
        kill_after_seconds=1,
        external_sha256="a" * 64,
    )
    controller.contract = {"deadline_monotonic_ns": 200}
    clock = iter((terminal_ns, 300, 300))
    monkeypatch.setattr(
        "rpent.reproduce.robocasa.deadline_supervisor._time.monotonic_ns",
        lambda: next(clock),
    )
    freeze = Mock()
    monkeypatch.setattr(controller, "_freeze", freeze)

    observed = _ObservedProcess(Mock(poll=Mock(return_value=0)), controller)
    assert observed.poll() == 0
    controller._watch()

    assert controller.cancelled_monotonic_ns == terminal_ns
    assert controller.fired is fires
    assert freeze.call_count == int(fires)


def test_terminal_timestamp_and_cancellation_are_linearized_against_watcher(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    workdir = tmp_path / "workdir"
    workdir.mkdir(mode=0o700)
    controller = DeadlineController(
        workdir=workdir,
        run_id="run-1",
        driver_identity={"pid": 1, "pgid": 1, "start_time_ticks": 1},
        timeout_seconds=1,
        kill_after_seconds=1,
        external_sha256="a" * 64,
    )
    controller.contract = {"deadline_monotonic_ns": 200}
    timestamp_started = threading.Event()
    release_timestamp = threading.Event()

    def clock() -> int:
        if threading.current_thread().name == "terminal-observer":
            timestamp_started.set()
            assert release_timestamp.wait(timeout=2)
            return 199
        return 300

    monkeypatch.setattr(
        "rpent.reproduce.robocasa.deadline_supervisor._time.monotonic_ns", clock
    )
    freeze = Mock()
    monkeypatch.setattr(controller, "_freeze", freeze)
    observer = threading.Thread(
        target=controller.note_planner_terminal,
        name="terminal-observer",
    )
    watcher = threading.Thread(target=controller._watch, name="deadline-watcher")

    observer.start()
    assert timestamp_started.wait(timeout=2)
    watcher.start()
    release_timestamp.set()
    observer.join(timeout=2)
    watcher.join(timeout=2)

    assert not observer.is_alive()
    assert not watcher.is_alive()
    assert controller.cancelled_monotonic_ns == 199
    assert controller.fired is False
    freeze.assert_not_called()


@pytest.mark.parametrize(
    ("return_code", "driver_stop", "success_latched", "expected"),
    [
        (
            9,
            {
                "observed_state": "Z",
                "supervisor_stopped": False,
                "kill_sent": False,
                "kill_observed": True,
            },
            False,
            False,
        ),
        (
            75,
            {
                "observed_state": "Z",
                "supervisor_stopped": False,
                "kill_sent": False,
                "kill_observed": True,
            },
            False,
            True,
        ),
        (
            75,
            {
                "observed_state": "T",
                "supervisor_stopped": True,
                "kill_sent": True,
                "kill_observed": True,
            },
            False,
            False,
        ),
        (
            -signal.SIGKILL,
            {
                "observed_state": "T",
                "supervisor_stopped": True,
                "kill_sent": True,
                "kill_observed": True,
            },
            False,
            True,
        ),
        (
            -signal.SIGKILL,
            {
                "observed_state": "T",
                "supervisor_stopped": True,
                "kill_sent": False,
                "kill_observed": False,
            },
            False,
            False,
        ),
        (
            0,
            {
                "observed_state": "Z",
                "supervisor_stopped": False,
                "kill_sent": False,
                "kill_observed": True,
            },
            True,
            True,
        ),
        (
            0,
            {
                "observed_state": "Z",
                "supervisor_stopped": False,
                "kill_sent": False,
                "kill_observed": True,
            },
            False,
            False,
        ),
    ],
)
def test_timeout_driver_exit_requires_adapter_75_or_confirmed_supervisor_kill(
    return_code: int, driver_stop: dict, success_latched: bool, expected: bool
):
    assert (
        _timeout_driver_exit_expected(
            {
                "driver_stop": driver_stop,
                "driver_return_code_before_stop": return_code,
                "success_at_deadline": success_latched,
            }
        )
        is expected
    )


@pytest.mark.parametrize(
    ("provider", "web_count", "raw_cause", "raw_rc", "expected"),
    [
        (None, 0, "planner_protocol_error", 23, ("planner_timeout", 20)),
        (
            {"violation": "forbidden_web_search"},
            0,
            "planner_protocol_error",
            23,
            ("planner_protocol_error", 23),
        ),
        (None, 1, "planner_protocol_error", 23, ("planner_protocol_error", 23)),
        (None, 0, "operator_interrupted", 130, ("operator_interrupted", 130)),
        (None, 0, "planner_completed", 0, ("planner_timeout", 20)),
    ],
)
def test_outer_deadline_preserves_only_higher_priority_failures(
    provider, web_count: int, raw_cause: str, raw_rc: int, expected
):
    assert (
        normalize_deadline_outcome(
            raw_cause,
            raw_rc,
            supervisor_fired=True,
            provider_violation=provider,
            forbidden_web_search_event_count=web_count,
        )
        == expected
    )


def test_driver_that_crashed_before_timeout_is_not_a_supervisor_termination(
    tmp_path: Path,
):
    workdir = tmp_path / "workdir"
    workdir.mkdir(mode=0o700)
    (workdir / "done_00.flag").write_text("", encoding="utf-8")
    (workdir / "state_00.json").write_text(
        json.dumps({"step": 0, "success": False}) + "\n", encoding="utf-8"
    )
    (workdir / "command_trace.jsonl").write_text("", encoding="utf-8")
    process = subprocess.Popen(
        ["/bin/sh", "-c", "read _value; exit 9"],
        stdin=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    identity = process_identity(process.pid)
    assert identity is not None
    driver_identity = {
        key: identity[key] for key in ("pid", "pgid", "start_time_ticks")
    }
    assert process.stdin is not None
    process.stdin.write("exit\n")
    process.stdin.close()
    exit_deadline = time.monotonic() + 2
    while time.monotonic() < exit_deadline:
        observed = process_identity(process.pid)
        if observed is not None and observed["state"] == "Z":
            break
        time.sleep(0.005)
    else:
        pytest.fail("driver fixture did not become a zombie")
    controller = DeadlineController(
        workdir=workdir,
        run_id="run-1",
        driver_identity=driver_identity,
        timeout_seconds=0.03,
        kill_after_seconds=1,
        external_sha256="a" * 64,
    )
    try:
        controller.start(time.monotonic())
        limit = time.monotonic() + 2
        while not controller.fired and time.monotonic() < limit:
            time.sleep(0.005)
        controller.finish()
        return_code = process.wait(timeout=2)
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=2)

    assert controller.error is None
    freeze, _digest = _read_trusted_json(workdir / FREEZE_NAME)
    assert freeze["driver_stop"] == {
        "observed_state": "Z",
        "supervisor_stopped": False,
        "kill_sent": False,
        "kill_observed": True,
    }
    assert return_code == 9
    assert not _timeout_driver_exit_expected(
        {
            "driver_stop": freeze["driver_stop"],
            "driver_return_code_before_stop": return_code,
            "success_at_deadline": freeze["success"],
        }
    )


def test_deadline_controller_freezes_and_kills_the_committed_prefix(tmp_path: Path):
    workdir = tmp_path / "workdir"
    workdir.mkdir(mode=0o700)
    (workdir / "done_00.flag").write_text("", encoding="utf-8")
    (workdir / "state_00.json").write_text(
        json.dumps({"step": 0, "success": False}) + "\n", encoding="utf-8"
    )
    (workdir / "command_trace.jsonl").write_text("", encoding="utf-8")
    process = subprocess.Popen(["sleep", "60"], start_new_session=True)
    try:
        identity = process_identity(process.pid)
        assert identity is not None
        driver_identity = {
            key: identity[key] for key in ("pid", "pgid", "start_time_ticks")
        }
        controller = DeadlineController(
            workdir=workdir,
            run_id="run-1",
            driver_identity=driver_identity,
            timeout_seconds=0.2,
            kill_after_seconds=1,
            external_sha256="a" * 64,
        )
        controller.start(time.monotonic())
        assert controller.contract is not None
        assert controller.contract_sha256 is not None
        (workdir / "state_01.json").write_text(
            json.dumps({"step": 1, "success": True}) + "\n", encoding="utf-8"
        )
        (workdir / "log_01.json").write_text(
            json.dumps({"command": {"action": "release"}}) + "\n",
            encoding="utf-8",
        )
        (workdir / "command_trace.jsonl").write_text(
            '{"action":"release"}\n', encoding="utf-8"
        )
        _atomic_json(
            workdir / f"{RECEIPT_PREFIX}01.json",
            {
                "schema_version": 1,
                "protocol": DEADLINE_PROTOCOL,
                "run_id": "run-1",
                "nonce": controller.contract["nonce"],
                "step": 1,
                "deadline_monotonic_ns": controller.contract["deadline_monotonic_ns"],
                "done_published_monotonic_ns": controller.contract[
                    "deadline_monotonic_ns"
                ]
                - 1,
                "contract_sha256": controller.contract_sha256,
            },
        )
        (workdir / "done_01.flag").write_text("", encoding="utf-8")
        limit = time.monotonic() + 2
        while not controller.fired and time.monotonic() < limit:
            time.sleep(0.01)
        controller.finish()
        process.wait(timeout=2)
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=2)

    assert controller.error is None
    freeze, digest = _read_trusted_json(workdir / FREEZE_NAME)
    assert digest == controller.freeze_sha256
    assert freeze["prefix_step"] == 1
    assert freeze["success"] is True
    assert freeze["driver_stop"]["observed_state"] in {"T", "t"}
    assert freeze["driver_stop"]["supervisor_stopped"] is True
    assert freeze["driver_stop"]["kill_sent"] is True
    assert freeze["driver_stop"]["kill_observed"] is True
    assert process.returncode == -signal.SIGKILL
