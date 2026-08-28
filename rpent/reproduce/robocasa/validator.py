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

"""One canonical validator shared by resume decisions and summaries."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .artifacts import CellResult, Completion, Integrity, Outcome, artifact_directory
from .deadline_supervisor import (
    DEADLINE_PROTOCOL,
    FREEZE_NAME,
    SEAL_NAME,
    deadline_receipt_problem,
    driver_stop_problem,
    normalize_deadline_outcome,
    timeout_driver_exit_problem,
)
from .protocol import (
    CELLS,
    EMPTY_MEMORY_TASKS,
    PAPER_REFERENCES,
    PROTOCOL_ID,
    SPLITS,
    Cell,
    command_problem,
)
from .provenance import RUN_MANIFEST_NAME, load_run_manifest

_ALLOWED_AGENT_EVENT_TYPES = frozenset(
    {
        "thread.started",
        "turn.started",
        "turn.completed",
        "turn.failed",
        "item.started",
        "item.updated",
        "item.completed",
        "error",
    }
)
_ALLOWED_AGENT_ITEM_TYPES = frozenset(
    {"agent_message", "reasoning", "command_execution", "image_view", "todo_list"}
)
_HEX = frozenset("0123456789abcdef")
_CONTRACT_FIELDS = frozenset(
    {
        "schema_version",
        "protocol",
        "run_id",
        "nonce",
        "started_monotonic_ns",
        "deadline_monotonic_ns",
        "timeout_ns",
        "driver",
        "external_deadline_sha256",
    }
)
_SUPERVISOR_FIELDS = frozenset(
    {
        "protocol",
        "run_id",
        "nonce",
        "fired",
        "contract_sha256",
        "freeze_sha256",
        "error",
    }
)
_SEAL_FIELDS = frozenset(
    {
        "schema_version",
        "protocol",
        "run_id",
        "nonce",
        "deadline_monotonic_ns",
        "sealed_monotonic_ns",
        "contract_sha256",
    }
)
_FREEZE_PREFIX_FIELDS = frozenset(
    {
        "prefix_step",
        "success",
        "final_state_sha256",
        "command_trace_sha256",
        "raw_command_trace_sha256",
        "files_sha256",
    }
)
_FREEZE_FIELDS = frozenset(
    {
        "schema_version",
        "protocol",
        "run_id",
        "nonce",
        "deadline_monotonic_ns",
        "frozen_monotonic_ns",
        "contract_sha256",
        "seal_sha256",
        "driver",
        "driver_stop",
        "dropped_done_steps",
        *_FREEZE_PREFIX_FIELDS,
    }
)
_PLANNER_STATUS_FIELDS = frozenset(
    {
        "schema_version",
        "state",
        "started_at",
        "finished_at",
        "elapsed_seconds",
        "timeout_seconds",
        "kill_after_seconds",
        "termination_cause",
        "wrapper_rc",
        "raw_termination_cause",
        "raw_wrapper_rc",
        "child_pid",
        "child_pgid",
        "child_rc",
        "child_signal",
        "deadline_fired",
        "external_deadline_fired",
        "interrupted_signal",
        "forwarded_signal_sent",
        "term_sent",
        "kill_sent",
        "orphan_cleanup_term_sent",
        "parent_pid",
        "parent_death_signal",
        "parent_death_signal_enabled",
        "parent_death_observed",
        "subreaper_enabled",
        "descendant_cleanup",
        "spawn_error",
        "supervision_error",
        "protocol_violation_detected",
        "policy_stop_triggered",
        "policy_stop_source",
        "provider_protocol_violation",
        "live_codex_protocol_violation",
        "stdout_total_lines",
        "stdout_blank_lines",
        "stdout_json_events",
        "malformed_jsonl_lines",
        "terminal_event_count",
        "terminal_event_counts",
        "terminal_event",
        "forbidden_web_search_event_count",
        "forbidden_web_search_event_counts",
        "forbidden_web_search_event",
        "deadline_supervisor",
    }
)
_CLEANUP_FIELDS = frozenset(
    {
        "subreaper_enabled",
        "complete",
        "detected",
        "term_sent",
        "kill_sent",
        "reaped",
        "remaining",
        "errors",
    }
)


@dataclass(frozen=True)
class Validation:
    cell: Cell
    result: CellResult
    artifact_problems: tuple[str, ...] = ()
    protocol_problems: tuple[str, ...] = ()
    provenance_problems: tuple[str, ...] = ()
    audit: dict[str, Any] | None = None

    @property
    def problems(self) -> tuple[str, ...]:
        return (
            self.artifact_problems + self.protocol_problems + self.provenance_problems
        )

    @property
    def resumable(self) -> bool:
        return not self.result.canonical


def _read_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8", errors="strict"))
    if not isinstance(value, dict):
        raise ValueError("root is not an object")
    return value


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _validate_agent_log(path: Path, audit: dict[str, Any]) -> list[str]:
    problems: list[str] = []
    timed_out = audit.get("termination_cause") == "planner_timeout"
    try:
        with path.open(encoding="utf-8", errors="replace") as stream:
            lines = list(stream)
    except Exception as exc:
        return [f"evidence.agent_log is unreadable: {exc}"]
    if not lines and not timed_out:
        problems.append("evidence.agent_log has no events")
    saw_turn_completed = False
    reconnect_lines: list[int] = []
    for number, raw in enumerate(lines, 1):
        if not raw.strip():
            if not timed_out:
                problems.append(f"agent log line {number} is blank")
            continue
        try:
            event = json.loads(raw)
        except Exception as exc:
            if not timed_out:
                problems.append(f"agent log line {number} is invalid JSON: {exc}")
            continue
        if not isinstance(event, dict):
            if not timed_out:
                problems.append(f"agent log line {number} is not an event object")
            continue
        event_type = event.get("type")
        if event_type not in _ALLOWED_AGENT_EVENT_TYPES:
            problems.append(
                f"agent log line {number} has prohibited or unknown "
                f"event type {event_type!r}"
            )
            continue
        if event_type == "turn.completed":
            saw_turn_completed = True
        if event_type == "turn.failed" and not timed_out:
            problems.append(
                f"agent log line {number} has prohibited event type {event_type!r}"
            )
            continue
        if event_type == "error":
            message = event.get("message")
            if not isinstance(message, str) or not message.startswith(
                "Reconnecting..."
            ):
                problems.append(
                    f"agent log line {number} has a non-recoverable error event"
                )
            else:
                reconnect_lines.append(number)
            continue
        if event_type.startswith("item."):
            item = event.get("item")
            if not isinstance(item, dict):
                problems.append(
                    f"agent log line {number} has item event without item object"
                )
                continue
            item_type = item.get("type")
            if item_type not in _ALLOWED_AGENT_ITEM_TYPES:
                problems.append(
                    f"agent log line {number} has prohibited or unknown "
                    f"item type {item_type!r}"
                )
    if not timed_out and not saw_turn_completed:
        problems.append("evidence.agent_log has no turn.completed event")
    if reconnect_lines and not timed_out and not saw_turn_completed:
        problems.append("agent transport reconnect was not followed by turn.completed")
    return problems


def _agent_log_summary(path: Path) -> dict[str, Any] | None:
    """Rebuild the external wrapper's JSONL counters from archived bytes."""
    try:
        with path.open(encoding="utf-8", errors="replace") as stream:
            lines = list(stream)
    except OSError:
        return None
    blank_lines = 0
    json_events = 0
    malformed_lines = 0
    terminal_events: list[dict[str, Any]] = []
    web_events: list[dict[str, Any]] = []
    for line_number, raw in enumerate(lines, 1):
        if not raw.strip():
            blank_lines += 1
            continue
        try:
            event = json.loads(raw)
        except (json.JSONDecodeError, UnicodeError):
            malformed_lines += 1
            continue
        if not isinstance(event, dict):
            malformed_lines += 1
            continue
        json_events += 1
        event_type = event.get("type")
        if event_type in {"turn.completed", "turn.failed"}:
            terminal_events.append({"type": event_type, "line_number": line_number})
        item = event.get("item")
        if event_type in {"item.started", "item.completed"} and isinstance(item, dict):
            item_type = item.get("type")
            if isinstance(item_type, str) and "web_search" in (
                item_type.strip().lower().replace("-", "_")
            ):
                web_events.append(
                    {
                        "type": event_type,
                        "item_type": item_type,
                        "line_number": line_number,
                    }
                )
    return {
        "stdout_total_lines": len(lines),
        "stdout_blank_lines": blank_lines,
        "stdout_json_events": json_events,
        "malformed_jsonl_lines": malformed_lines,
        "terminal_event_count": len(terminal_events),
        "terminal_event_counts": {
            event_type: sum(event["type"] == event_type for event in terminal_events)
            for event_type in ("turn.completed", "turn.failed")
        },
        "terminal_event": terminal_events[-1] if terminal_events else None,
        "forbidden_web_search_event_count": len(web_events),
        "forbidden_web_search_event_counts": {
            event_type: sum(event["type"] == event_type for event in web_events)
            for event_type in ("item.completed", "item.started")
        },
        "forbidden_web_search_event": web_events[0] if web_events else None,
    }


def _planner_raw_outcome(status: dict[str, Any]) -> tuple[str, int]:
    """Replay the frozen external deadline wrapper's classify() function."""
    if (
        status.get("provider_protocol_violation") is not None
        or status.get("forbidden_web_search_event_count") != 0
    ):
        return "planner_protocol_error", 23
    interrupted = status.get("interrupted_signal")
    if type(interrupted) is int:
        return "operator_interrupted", 128 + interrupted
    if status.get("external_deadline_fired") is True:
        return "planner_timeout", 20
    counts = status.get("terminal_event_counts")
    failed = counts.get("turn.failed") if isinstance(counts, dict) else None
    if type(failed) is int and failed:
        return "planner_failed", 21
    if status.get("child_rc") != 0:
        return "planner_failed", 22
    completed = counts.get("turn.completed") if isinstance(counts, dict) else None
    if (
        status.get("malformed_jsonl_lines") != 0
        or status.get("terminal_event_count") != 1
        or completed != 1
    ):
        return "planner_protocol_error", 23
    return "planner_completed", 0


def _planner_status_semantic_problems(
    status: dict[str, Any],
    agent_log: Path,
    *,
    timed_out: bool,
) -> list[str]:
    problems: list[str] = []
    summary = _agent_log_summary(agent_log)
    if summary is None:
        problems.append("planner status cannot be replayed from evidence.agent_log")
    elif any(status.get(key) != value for key, value in summary.items()):
        problems.append(
            "planner status JSONL summary disagrees with evidence.agent_log"
        )

    child_rc = status.get("child_rc")
    child_signal = status.get("child_signal")
    if type(child_rc) is not int or child_rc < 0:
        problems.append("planner status child_rc must be a non-negative integer")
    if child_signal is not None and (
        type(child_signal) is not int
        or child_signal <= 0
        or child_rc != 128 + child_signal
    ):
        problems.append("planner status child signal and shell return code disagree")
    for key in ("child_pid", "child_pgid", "parent_pid"):
        if type(status.get(key)) is not int or status[key] <= 0:
            problems.append(f"planner status {key} must be a positive integer")
    if status.get("child_pid") != status.get("child_pgid"):
        problems.append("planner status child pid and process group must agree")

    raw_cause, raw_rc = _planner_raw_outcome(status)
    if (
        status.get("raw_termination_cause") != raw_cause
        or status.get("raw_wrapper_rc") != raw_rc
    ):
        problems.append("planner raw outcome disagrees with replayed classification")
    normalized_cause, normalized_rc = normalize_deadline_outcome(
        raw_cause,
        raw_rc,
        supervisor_fired=timed_out,
        provider_violation=status.get("provider_protocol_violation"),
        forbidden_web_search_event_count=status.get("forbidden_web_search_event_count"),
    )
    if (
        status.get("termination_cause") != normalized_cause
        or status.get("wrapper_rc") != normalized_rc
    ):
        problems.append("planner final outcome disagrees with deadline normalization")
    return problems


def _parse_trace(path: Path) -> tuple[list[dict[str, Any]], list[str]]:
    commands: list[dict[str, Any]] = []
    problems: list[str] = []
    try:
        lines = path.read_text(encoding="utf-8", errors="strict").splitlines()
    except Exception as exc:
        return [], [f"trace is unreadable: {exc}"]
    for number, raw in enumerate(lines, 1):
        if not raw.strip():
            problems.append(f"trace line {number} is blank")
            continue
        try:
            command = json.loads(raw)
        except Exception as exc:
            problems.append(f"trace line {number} is invalid JSON: {exc}")
            continue
        if problem := command_problem(command):
            problems.append(f"trace line {number} is invalid: {problem}")
            continue
        commands.append(command)
    return commands, problems


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in _HEX for character in value)
    )


def _driver_problem(value: Any) -> str | None:
    if not isinstance(value, dict) or set(value) != {
        "pid",
        "pgid",
        "start_time_ticks",
    }:
        return "driver identity has an invalid schema"
    if any(type(value[key]) is not int or value[key] <= 0 for key in value):
        return "driver identity values must be positive integers"
    return None


def _trusted_digest(path: Path) -> str:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{path.name} is not a regular non-symlink file")
    return _digest(path)


def _canonical_trace(commands: list[dict[str, Any]]) -> bytes:
    return b"".join(
        (json.dumps(command, ensure_ascii=False, separators=(",", ":")) + "\n").encode(
            "utf-8"
        )
        for command in commands
    )


def _receipt_problems(
    receipt: dict[str, Any],
    *,
    step: int,
    contract: dict[str, Any],
    contract_sha256: str,
    prefix: bool,
) -> list[str]:
    problems: list[str] = []
    published = receipt.get("done_published_monotonic_ns")
    problem = deadline_receipt_problem(
        receipt,
        step=step,
        contract=contract,
        contract_sha256=contract_sha256,
        require_before_deadline=prefix,
    )
    if problem is not None:
        problems.append(f"deadline receipt {step} {problem}")
    elif not prefix and published < contract["deadline_monotonic_ns"]:
        problems.append(f"on-time deadline receipt {step} has no committed marker")
    return problems


def _validate_frozen_prefix(
    log_directory: Path,
    directory: Path,
    audit: dict[str, Any],
    commands: list[dict[str, Any]],
    contract: dict[str, Any],
    contract_sha256: str,
    freeze: dict[str, Any],
) -> list[str]:
    problems: list[str] = []
    prefix_step = freeze.get("prefix_step")
    if type(prefix_step) is not int or prefix_step < 0:
        return ["deadline freeze prefix_step must be a non-negative integer"]

    expected_names = {
        *(f"done_{step:02d}.flag" for step in range(prefix_step + 1)),
        *(f"state_{step:02d}.json" for step in range(prefix_step + 1)),
        *(f"log_{step:02d}.json" for step in range(1, prefix_step + 1)),
        *(f"_deadline_commit_{step:02d}.json" for step in range(1, prefix_step + 1)),
    }
    files = freeze.get("files_sha256")
    if (
        not isinstance(files, dict)
        or set(files) != expected_names
        or any(not _is_sha256(value) for value in files.values())
    ):
        return ["deadline freeze files_sha256 is not the exact canonical prefix"]

    actual_done = {path.name for path in log_directory.glob("done_*.flag")}
    expected_done = {f"done_{step:02d}.flag" for step in range(prefix_step + 1)}
    if actual_done != expected_done:
        problems.append("archived deadline prefix has missing or extra done markers")

    for prefix, suffix, minimum_step in (
        ("state_", ".json", 0),
        ("log_", ".json", 1),
        ("_deadline_commit_", ".json", 1),
    ):
        for path in log_directory.glob(f"{prefix}*{suffix}"):
            raw_step = path.name[len(prefix) : -len(suffix)]
            try:
                step = int(raw_step)
            except ValueError:
                problems.append(
                    f"archived deadline journal has invalid name {path.name}"
                )
                continue
            if path.name != f"{prefix}{step:02d}{suffix}":
                problems.append(
                    f"archived deadline journal has invalid name {path.name}"
                )
            if step < minimum_step or (
                step <= prefix_step and path.name not in expected_names
            ):
                problems.append(
                    f"archived deadline journal is outside the canonical prefix/tail: "
                    f"{path.name}"
                )
            try:
                _trusted_digest(path)
            except Exception as exc:
                problems.append(
                    f"archived deadline journal {path.name} is unsafe: {exc}"
                )

    for name, expected_digest in files.items():
        path = log_directory / name
        try:
            actual_digest = _trusted_digest(path)
        except Exception as exc:
            problems.append(f"frozen journal {name} is missing or unsafe: {exc}")
            continue
        if actual_digest != expected_digest:
            problems.append(f"frozen journal {name} hash disagrees with freeze")
        if name.startswith("done_"):
            try:
                if path.stat().st_size != 0:
                    problems.append(f"frozen marker {name} must be empty")
            except OSError as exc:
                problems.append(f"frozen marker {name} cannot be inspected: {exc}")

    rebuilt: list[dict[str, Any]] = []
    final_state: dict[str, Any] | None = None
    task_language = audit.get("task_language")
    for step in range(prefix_step + 1):
        suffix = f"{step:02d}"
        try:
            state = _read_object(log_directory / f"state_{suffix}.json")
        except Exception as exc:
            problems.append(f"frozen state {step} is invalid JSON: {exc}")
            continue
        if state.get("step") != step or not isinstance(state.get("success"), bool):
            problems.append(f"frozen state {step} has invalid step/success fields")
        if (
            not isinstance(state.get("task_language"), str)
            or not state["task_language"]
            or state["task_language"] != task_language
        ):
            problems.append(
                f"frozen state {step} task_language disagrees with canonical audit"
            )
        final_state = state
        if step == 0:
            continue
        try:
            log = _read_object(log_directory / f"log_{suffix}.json")
        except Exception as exc:
            problems.append(f"frozen log {step} is invalid JSON: {exc}")
            continue
        command = log.get("command")
        if not isinstance(command, dict) or (problem := command_problem(command)):
            detail = (
                problem if isinstance(command, dict) else "command is not an object"
            )
            problems.append(f"frozen log {step} has invalid command: {detail}")
        else:
            rebuilt.append(command)
        try:
            receipt = _read_object(log_directory / f"_deadline_commit_{suffix}.json")
        except Exception as exc:
            problems.append(f"deadline receipt {step} is invalid JSON: {exc}")
        else:
            problems.extend(
                _receipt_problems(
                    receipt,
                    step=step,
                    contract=contract,
                    contract_sha256=contract_sha256,
                    prefix=True,
                )
            )

    for receipt_path in log_directory.glob("_deadline_commit_*.json"):
        raw_step = receipt_path.name[len("_deadline_commit_") : -5]
        if not raw_step.isdigit() or int(raw_step) <= prefix_step:
            continue
        try:
            receipt = _read_object(receipt_path)
        except Exception as exc:
            problems.append(f"tail deadline receipt is invalid JSON: {exc}")
            continue
        problems.extend(
            _receipt_problems(
                receipt,
                step=int(raw_step),
                contract=contract,
                contract_sha256=contract_sha256,
                prefix=False,
            )
        )

    trace_bytes = _canonical_trace(rebuilt)
    trace_sha256 = hashlib.sha256(trace_bytes).hexdigest()
    raw_trace = log_directory / "command_trace.jsonl"
    try:
        raw_trace_sha256 = _trusted_digest(raw_trace)
    except Exception as exc:
        problems.append(f"raw command trace is missing or unsafe: {exc}")
    else:
        if raw_trace_sha256 != freeze.get("raw_command_trace_sha256"):
            problems.append("raw command trace hash disagrees with freeze")
    try:
        canonical_trace_sha256 = _trusted_digest(
            directory / str(audit.get("command_trace"))
        )
    except Exception as exc:
        problems.append(f"canonical command trace is missing or unsafe: {exc}")
    else:
        if canonical_trace_sha256 != trace_sha256:
            problems.append("canonical command trace differs from frozen prefix logs")
    if rebuilt != commands or trace_sha256 != freeze.get("command_trace_sha256"):
        problems.append("deadline freeze command trace does not match rebuilt prefix")
    if final_state is None:
        problems.append("deadline freeze has no final state")
    else:
        final_state_path = log_directory / f"state_{prefix_step:02d}.json"
        try:
            final_state_sha256 = _trusted_digest(final_state_path)
        except Exception as exc:
            problems.append(f"deadline final state is missing or unsafe: {exc}")
        else:
            if final_state_sha256 != freeze.get("final_state_sha256"):
                problems.append("deadline freeze final state hash disagrees")
        if final_state.get("success") is not freeze.get("success"):
            problems.append("deadline freeze success disagrees with final state")
        if final_state.get("success") is not audit.get("success"):
            problems.append("canonical audit success disagrees with frozen final state")
    return problems


def _validate_evidence(
    directory: Path,
    cell: Cell,
    audit: dict[str, Any],
    commands: list[dict[str, Any]],
    configuration: dict[str, Any] | None,
) -> list[str]:
    problems: list[str] = []
    evidence = audit.get("evidence")
    if not isinstance(evidence, dict):
        return ["evidence must be an object"]
    timed_out = audit.get("termination_cause") == "planner_timeout"
    expected_keys = {"agent_log", "planner_status", "deadline_contract"}
    if timed_out:
        expected_keys.update({"deadline_seal", "deadline_freeze"})
    if set(evidence) != expected_keys:
        problems.append("evidence must contain exactly the deadline protocol files")

    paths: dict[str, Path] = {}
    digests: dict[str, str] = {}
    for key in expected_keys:
        item = evidence.get(key)
        if not isinstance(item, dict) or set(item) != {"path", "sha256"}:
            problems.append(f"evidence.{key} must contain exactly path and sha256")
            continue
        relative, digest = item.get("path"), item.get("sha256")
        if (
            not isinstance(relative, str)
            or not relative
            or Path(relative).is_absolute()
            or ".." in Path(relative).parts
        ):
            problems.append(f"evidence.{key}.path must be a confined relative path")
            continue
        path = directory / relative
        try:
            unsafe_component = any(
                candidate.is_symlink()
                for candidate in (
                    directory / Path(*Path(relative).parts[:index])
                    for index in range(1, len(Path(relative).parts) + 1)
                )
            )
            actual_digest = _trusted_digest(path)
        except Exception as exc:
            problems.append(f"evidence.{key} is missing or unsafe: {exc}")
            continue
        if unsafe_component or not _is_sha256(digest) or actual_digest != digest:
            problems.append(f"evidence.{key} is unsafe or hash-mismatched")
            continue
        paths[key] = path
        digests[key] = digest
    if set(paths) != expected_keys:
        return problems
    if len({path.parent for path in paths.values()}) != 1:
        problems.append("deadline evidence files must share one archived log directory")
        return problems
    log_directory = paths["deadline_contract"].parent
    problems.extend(_validate_agent_log(paths["agent_log"], audit))

    objects: dict[str, dict[str, Any]] = {}
    for key in expected_keys - {"agent_log"}:
        try:
            objects[key] = _read_object(paths[key])
        except Exception as exc:
            problems.append(f"evidence.{key} is invalid JSON: {exc}")
    if set(objects) != expected_keys - {"agent_log"}:
        return problems

    runtime = configuration.get("runtime") if isinstance(configuration, dict) else None
    inputs = runtime.get("inputs_sha256") if isinstance(runtime, dict) else None
    expected_external_sha256 = (
        inputs.get("deadline") if isinstance(inputs, dict) else None
    )
    expected_kill_after = (
        runtime.get("kill_after_seconds") if isinstance(runtime, dict) else None
    )
    expected_timeout = SPLITS[cell.split].timeout_seconds
    expected_timeout_ns = expected_timeout * 1_000_000_000
    if not _is_sha256(expected_external_sha256):
        problems.append("root runtime deadline identity is missing or invalid")

    deadline = audit.get("deadline")
    contract = objects["deadline_contract"]
    contract_sha256 = digests["deadline_contract"]
    driver = contract.get("driver")
    nonce = contract.get("nonce")
    started_ns = contract.get("started_monotonic_ns")
    deadline_ns = contract.get("deadline_monotonic_ns")
    timeout_ns = contract.get("timeout_ns")
    if set(contract) != _CONTRACT_FIELDS:
        problems.append("deadline contract has an invalid schema")
    if problem := _driver_problem(driver):
        problems.append(problem)
    if (
        contract.get("schema_version") != 1
        or contract.get("protocol") != DEADLINE_PROTOCOL
        or contract.get("run_id") != audit.get("run_id")
        or not isinstance(nonce, str)
        or len(nonce) != 32
        or any(character not in _HEX for character in nonce)
        or type(started_ns) is not int
        or started_ns <= 0
        or type(deadline_ns) is not int
        or type(timeout_ns) is not int
        or timeout_ns != expected_timeout_ns
        or deadline_ns != started_ns + timeout_ns
        or contract.get("external_deadline_sha256") != expected_external_sha256
        or not isinstance(deadline, dict)
        or contract_sha256 != deadline.get("contract_sha256")
        or deadline_ns != deadline.get("deadline_monotonic_ns")
        or timeout_ns != deadline.get("timeout_ns")
        or driver != deadline.get("driver")
    ):
        problems.append("deadline contract disagrees with audit/root protocol")

    status = objects["planner_status"]
    supervisor = status.get("deadline_supervisor")
    cleanup = status.get("descendant_cleanup")
    if set(status) != _PLANNER_STATUS_FIELDS:
        problems.append("evidence.planner_status has an invalid schema")
    if (
        status.get("schema_version") != 1
        or status.get("state") != "finished"
        or status.get("termination_cause") != audit.get("termination_cause")
        or status.get("wrapper_rc") != (20 if timed_out else 0)
        or status.get("timeout_seconds") != expected_timeout
        or status.get("kill_after_seconds") != expected_kill_after
        or status.get("deadline_fired") is not timed_out
        or type(status.get("external_deadline_fired")) is not bool
        or status.get("interrupted_signal") is not None
        or status.get("forwarded_signal_sent") is not False
        or status.get("parent_death_signal") != 15
        or status.get("parent_death_signal_enabled") is not True
        or status.get("subreaper_enabled") is not True
        or status.get("parent_death_observed") is not False
        or status.get("spawn_error") is not None
        or status.get("supervision_error") is not None
        or status.get("protocol_violation_detected") is not False
        or status.get("policy_stop_triggered") is not False
        or status.get("policy_stop_source") is not None
        or status.get("provider_protocol_violation") is not None
        or status.get("live_codex_protocol_violation") is not None
    ):
        problems.append("evidence.planner_status is not a canonical planner outcome")
    problems.extend(
        _planner_status_semantic_problems(
            status,
            paths["agent_log"],
            timed_out=timed_out,
        )
    )
    if audit.get("agent_exit_code") != status.get("wrapper_rc"):
        problems.append("audit agent_exit_code disagrees with planner wrapper status")
    if (
        not isinstance(cleanup, dict)
        or set(cleanup) != _CLEANUP_FIELDS
        or cleanup.get("subreaper_enabled") is not True
        or cleanup.get("complete") is not True
        or cleanup.get("remaining") != []
        or cleanup.get("errors") != []
        or status.get("term_sent") is not bool(cleanup.get("term_sent"))
        or status.get("kill_sent") is not bool(cleanup.get("kill_sent"))
        or status.get("orphan_cleanup_term_sent") is not bool(cleanup.get("term_sent"))
    ):
        problems.append("evidence.planner_status descendant cleanup is incomplete")
    if (
        not isinstance(supervisor, dict)
        or set(supervisor) != _SUPERVISOR_FIELDS
        or supervisor.get("protocol") != DEADLINE_PROTOCOL
        or supervisor.get("run_id") != audit.get("run_id")
        or supervisor.get("nonce") != nonce
        or supervisor.get("fired") is not timed_out
        or supervisor.get("contract_sha256") != contract_sha256
        or supervisor.get("freeze_sha256")
        != (digests.get("deadline_freeze") if timed_out else None)
        or supervisor.get("error") is not None
    ):
        problems.append("planner deadline supervisor disagrees with contract/audit")

    if not timed_out:
        for name in (SEAL_NAME, FREEZE_NAME):
            path = log_directory / name
            if path.exists() or path.is_symlink():
                problems.append(
                    f"completed planner archive unexpectedly contains {name}"
                )
        return problems

    seal = objects["deadline_seal"]
    seal_sha256 = digests["deadline_seal"]
    sealed_ns = seal.get("sealed_monotonic_ns")
    if set(seal) != _SEAL_FIELDS:
        problems.append("deadline seal has an invalid schema")
    if (
        seal.get("schema_version") != 1
        or seal.get("protocol") != DEADLINE_PROTOCOL
        or seal.get("run_id") != audit.get("run_id")
        or seal.get("nonce") != nonce
        or seal.get("deadline_monotonic_ns") != deadline_ns
        or seal.get("contract_sha256") != contract_sha256
        or type(sealed_ns) is not int
        or sealed_ns < deadline_ns
        or not isinstance(deadline, dict)
        or seal_sha256 != deadline.get("seal_sha256")
    ):
        problems.append("deadline seal disagrees with contract/audit")

    freeze = objects["deadline_freeze"]
    freeze_sha256 = digests["deadline_freeze"]
    frozen_ns = freeze.get("frozen_monotonic_ns")
    driver_stop = freeze.get("driver_stop")
    dropped = freeze.get("dropped_done_steps")
    if set(freeze) != _FREEZE_FIELDS:
        problems.append("deadline freeze has an invalid schema")
    if problem := _driver_problem(freeze.get("driver")):
        problems.append(f"deadline freeze {problem}")
    prefix_step = freeze.get("prefix_step")
    dropped_valid = (
        isinstance(dropped, list)
        and all(type(step) is int for step in dropped)
        and dropped == sorted(set(dropped))
        and type(prefix_step) is int
        and all(step > prefix_step for step in dropped)
    )
    if (
        freeze.get("schema_version") != 1
        or freeze.get("protocol") != DEADLINE_PROTOCOL
        or freeze.get("run_id") != audit.get("run_id")
        or freeze.get("nonce") != nonce
        or freeze.get("deadline_monotonic_ns") != deadline_ns
        or freeze.get("contract_sha256") != contract_sha256
        or freeze.get("seal_sha256") != seal_sha256
        or freeze.get("driver") != driver
        or type(frozen_ns) is not int
        or type(sealed_ns) is not int
        or frozen_ns < sealed_ns
        or driver_stop_problem(driver_stop) is not None
        or not dropped_valid
        or freeze_sha256 != deadline.get("freeze_sha256")
        or freeze.get("prefix_step") != audit.get("steps")
        or freeze.get("success") is not audit.get("success")
        or deadline.get("prefix_step") != freeze.get("prefix_step")
        or deadline.get("success_at_deadline") is not freeze.get("success")
        or deadline.get("driver_stop") != driver_stop
    ):
        problems.append("deadline freeze disagrees with sealed canonical audit")
    problems.extend(
        _validate_frozen_prefix(
            log_directory,
            directory,
            audit,
            commands,
            contract,
            contract_sha256,
            freeze,
        )
    )
    return problems


def _validate_audit_provenance(
    audit: dict[str, Any], configuration: dict[str, Any]
) -> list[str]:
    """Bind every cell's recorded identities to the root run configuration."""
    problems: list[str] = []
    planner = configuration.get("planner")
    memory = configuration.get("memory")
    checkpoint = configuration.get("checkpoint")
    runtime = configuration.get("runtime")
    if not all(
        isinstance(item, dict) for item in (planner, memory, checkpoint, runtime)
    ):
        return ["root run configuration identities are malformed"]
    assert isinstance(planner, dict)
    assert isinstance(memory, dict)
    assert isinstance(checkpoint, dict)
    assert isinstance(runtime, dict)

    planner_fields = {
        "planner_profile": "profile",
        "planner_backend": "backend",
        "planner_model": "model",
        "planner_reasoning_effort": "reasoning_effort",
        "planner_auth_mode": "auth_mode",
        "planner_provider": "provider",
        "planner_endpoint_identity": "endpoint_identity",
    }
    for audit_name, config_name in planner_fields.items():
        if audit.get(audit_name) != planner.get(config_name):
            problems.append(f"audit {audit_name} differs from the root run manifest")

    if audit.get("preliminary") is not configuration.get("preliminary"):
        problems.append("audit preliminary flag differs from the root run manifest")
    if audit.get("release_ready") is not False:
        problems.append("preliminary runtime audit must set release_ready=false")
    if audit.get("runtime_root") != runtime.get("root"):
        problems.append("audit runtime_root differs from the root run manifest")

    inputs = runtime.get("inputs_sha256")
    navview = runtime.get("navview")
    if not isinstance(inputs, dict) or not isinstance(navview, dict):
        problems.append("root runtime input identity is malformed")
    else:
        expected_scripts = {
            "external_driver": inputs.get("driver"),
            "external_deadline": inputs.get("deadline"),
            "driver_adapter": inputs.get("preliminary_driver_adapter"),
            "deadline_supervisor": inputs.get("deadline_supervisor_adapter"),
            "artifact_builder": inputs.get("artifact_builder"),
        }
        if audit.get("runtime_scripts") != expected_scripts:
            problems.append("audit runtime_scripts differ from the root run manifest")
        expected_navview = {"xml_sha256": inputs.get("navview_xml"), **navview}
        if audit.get("navview") != expected_navview:
            problems.append("audit navview identity differs from the root run manifest")

    expected_memory = {
        "kind": memory.get("source"),
        "revision": memory.get("revision"),
        "manifest_sha256": memory.get("manifest_sha256"),
    }
    if audit.get("memory_source") != expected_memory:
        problems.append("audit memory_source differs from the root run manifest")

    recorded_checkpoint = audit.get("checkpoint")
    if not isinstance(recorded_checkpoint, dict):
        problems.append("audit checkpoint identity is missing")
    else:
        for key in (
            "checkpoint_id",
            "authority_manifest_sha256",
            "fingerprint",
        ):
            if recorded_checkpoint.get(key) != checkpoint.get(key):
                problems.append(
                    f"audit checkpoint.{key} differs from the root manifest"
                )
        files = recorded_checkpoint.get("files")
        shard_names = (
            "model-00001-of-00003.safetensors",
            "model-00002-of-00003.safetensors",
            "model-00003-of-00003.safetensors",
        )
        expected_shards = (
            {name: files.get(f"model/{name}") for name in shard_names}
            if isinstance(files, dict)
            else None
        )
        if audit.get("checkpoint_sha256") != expected_shards:
            problems.append(
                "audit checkpoint shard hashes disagree with attested files"
            )
    return problems


def _validate_protocol(
    directory: Path,
    cell: Cell,
    audit: dict[str, Any],
    commands: list[dict[str, Any]],
    trace_name: str,
    run_config_sha256: str | None,
    run_configuration: dict[str, Any] | None,
) -> list[str]:
    problems: list[str] = []
    required = {
        "suite": "robocasa365",
        "source": "harness",
        "commit_protocol": "done_marker_v1",
        "memory_role": "formal_evaluation",
        "command_trace": trace_name,
    }
    for key, expected in required.items():
        if audit.get(key) != expected:
            problems.append(f"{key}={audit.get(key)!r}, expected {expected!r}")
    if audit.get("protocol_id") != PROTOCOL_ID:
        problems.append(f"protocol_id must be {PROTOCOL_ID}")
    if run_config_sha256 is None or audit.get("run_config_sha256") != run_config_sha256:
        problems.append("audit run_config_sha256 must match the root run manifest")
    if audit.get("split") != cell.split.value:
        problems.append(f"split must be {cell.split.value}")
    if type(audit.get("schema_version")) is not int or audit.get("schema_version") != 1:
        problems.append("schema_version must be integer 1")
    if audit.get("valid") is not True:
        problems.append("valid must be true")
    for key, expected in (
        ("global_memory", False),
        ("task_notes_available", False),
        ("perception_isolation", True),
        ("reset_enabled", False),
    ):
        if audit.get(key) is not expected:
            problems.append(f"{key} must be {expected}")
    expected_reset_seed = (
        4_200_000 + cell.seed if cell.split.value == "composite_seen" else None
    )
    if audit.get("reset_seed") != expected_reset_seed:
        problems.append(
            f"reset_seed must be protocol-derived value {expected_reset_seed!r}"
        )
    if audit.get("artifact_errors") != []:
        problems.append("artifact_errors must be empty")
    if type(audit.get("steps")) is not int or audit.get("steps") != len(commands):
        problems.append(f"steps must equal trace command count {len(commands)}")
    for key in ("trace_recovered_from_logs", "rollout_timeout"):
        if not isinstance(audit.get(key), bool):
            problems.append(f"{key} must be boolean")
    dropped = audit.get("incomplete_commands_dropped")
    if type(dropped) is not int or dropped < 0:
        problems.append("incomplete_commands_dropped must be a non-negative integer")
    if type(audit.get("agent_exit_code")) is not int:
        problems.append("agent_exit_code must be an integer")
    success = audit.get("success")
    if not isinstance(success, bool):
        problems.append("success must be boolean")
    failure_type = audit.get("failure_type")
    if success is True and failure_type is not None:
        problems.append("successful rollout must have null failure_type")
    if success is False and failure_type not in {"task_failure", "rollout_timeout"}:
        problems.append("failed rollout has invalid failure_type")
    timeout = audit.get("rollout_timeout") is True
    if success is False and (failure_type == "rollout_timeout") != timeout:
        problems.append("rollout_timeout flag and failure_type disagree")
    deadline = audit.get("deadline")
    if not isinstance(deadline, dict) or set(deadline) != {
        "protocol",
        "contract_sha256",
        "seal_sha256",
        "freeze_sha256",
        "deadline_monotonic_ns",
        "timeout_ns",
        "driver",
        "driver_stop",
        "driver_return_code_before_stop",
        "prefix_step",
        "success_at_deadline",
    }:
        problems.append("deadline attestation has an invalid schema")
    else:
        contract_sha256 = deadline.get("contract_sha256")
        seal_sha256 = deadline.get("seal_sha256")
        freeze_sha256 = deadline.get("freeze_sha256")
        expected_timeout_ns = SPLITS[cell.split].timeout_seconds * 1_000_000_000
        if (
            deadline.get("protocol") != DEADLINE_PROTOCOL
            or not _is_sha256(contract_sha256)
            or type(deadline.get("deadline_monotonic_ns")) is not int
            or deadline["deadline_monotonic_ns"] <= 0
            or deadline.get("timeout_ns") != expected_timeout_ns
            or _driver_problem(deadline.get("driver")) is not None
        ):
            problems.append("deadline contract identity is invalid")
        if timeout:
            if (
                not _is_sha256(seal_sha256)
                or not _is_sha256(freeze_sha256)
                or not isinstance(deadline.get("driver_stop"), dict)
                or timeout_driver_exit_problem(
                    deadline.get("driver_stop"),
                    deadline.get("driver_return_code_before_stop"),
                    success_latched=deadline.get("success_at_deadline"),
                )
                is not None
                or deadline.get("prefix_step") != len(commands)
                or deadline.get("success_at_deadline") is not success
            ):
                problems.append("timeout deadline freeze identity is invalid")
        elif (
            seal_sha256 is not None
            or freeze_sha256 is not None
            or deadline.get("driver_stop") is not None
            or deadline.get("driver_return_code_before_stop") is not None
            or deadline.get("prefix_step") is not None
            or deadline.get("success_at_deadline") is not None
        ):
            problems.append("completed rollout must not contain a deadline freeze")
    expected_termination = "planner_timeout" if timeout else "planner_completed"
    if audit.get("termination_cause") != expected_termination:
        problems.append(f"termination_cause must be {expected_termination}")
    if audit.get("infra_status") != "ok":
        problems.append("infra_status must be ok")
    expected_planner_status = "timeout" if timeout else "completed"
    if audit.get("planner_status") != expected_planner_status:
        problems.append(f"planner_status must be {expected_planner_status}")
    if audit.get("environment_success") is not success:
        problems.append("environment_success must agree with success")
    for key in (
        "planner_profile",
        "planner_backend",
        "planner_model",
        "planner_reasoning_effort",
        "planner_auth_mode",
        "planner_provider",
        "planner_endpoint_identity",
    ):
        if not isinstance(audit.get(key), str) or not audit[key]:
            problems.append(f"{key} must be a non-empty string")
    preliminary = audit.get("preliminary")
    release_ready = audit.get("release_ready")
    if not isinstance(preliminary, bool) or not isinstance(release_ready, bool):
        problems.append("preliminary and release_ready must be boolean")
    elif preliminary and release_ready:
        problems.append("preliminary results cannot claim release_ready")
    expected_memory = {
        "seed": 0,
        "audit": f"{cell.task}_s0.json",
        "command_trace": f"{cell.task}_s0.jsonl",
    }
    if cell.task in EMPTY_MEMORY_TASKS:
        if (
            audit.get("task_memory") is not None
            or audit.get("task_memory_available") is not False
        ):
            problems.append("empty-whitelist task memory must be null and unavailable")
    elif (
        audit.get("task_memory") != expected_memory
        or audit.get("task_memory_available") is not True
    ):
        problems.append(
            "task memory must be the exact task-matched successful seed-0 pair"
        )
    language = audit.get("task_language")
    if not isinstance(language, str) or not language.strip():
        problems.append("task_language must be non-empty")
    else:
        for number, command in enumerate(commands, 1):
            if command["action"] == "vla_act" and command.get("prompt") != language:
                problems.append(
                    f"trace command {number} does not use task_language verbatim"
                )
    problems.extend(
        _validate_evidence(directory, cell, audit, commands, run_configuration)
    )
    return problems


def validate_cell(
    root: Path,
    cell: Cell,
    *,
    expected_run_config_sha256: str | None = None,
) -> Validation:
    """Validate completion commit, hashes, audit identity, trace and protocol."""
    run_manifest, run_manifest_problem = load_run_manifest(root)
    provenance: list[str] = []
    run_config_sha256 = None
    run_configuration: dict[str, Any] | None = None
    if run_manifest_problem is not None:
        provenance.append(run_manifest_problem)
    else:
        assert run_manifest is not None
        run_config_sha256 = run_manifest["run_config_sha256"]
        run_configuration = run_manifest["config"]
        if (
            expected_run_config_sha256 is not None
            and run_config_sha256 != expected_run_config_sha256
        ):
            provenance.append("root run manifest differs from the active run")
    try:
        directory = artifact_directory(root, cell)
    except ValueError as exc:
        result = CellResult(Completion.INCOMPLETE, Outcome.UNKNOWN, Integrity.UNKNOWN)
        return Validation(
            cell,
            result,
            artifact_problems=(f"unsafe artifact directory: {exc}",),
            provenance_problems=tuple(provenance),
        )
    commit_path = directory / f"{cell.tag}.completed.json"
    if not commit_path.is_file() or commit_path.is_symlink():
        result = CellResult(Completion.INCOMPLETE, Outcome.UNKNOWN, Integrity.UNKNOWN)
        return Validation(
            cell,
            result,
            artifact_problems=("missing completion manifest",),
            provenance_problems=tuple(provenance),
        )
    artifact: list[str] = []
    protocol: list[str] = []
    try:
        commit = _read_object(commit_path)
    except Exception as exc:
        result = CellResult(
            Completion.COMPLETED, Outcome.UNKNOWN, Integrity.INVALID, str(exc)
        )
        return Validation(
            cell,
            result,
            artifact_problems=(f"invalid completion manifest: {exc}",),
            provenance_problems=tuple(provenance),
        )
    if (
        type(commit.get("schema_version")) is not int
        or commit.get("schema_version") != 1
    ):
        artifact.append("completion manifest schema_version must be integer 1")
    identity = commit.get("cell")
    expected_identity = {
        "split": cell.split.value,
        "task": cell.task,
        "seed": cell.seed,
    }
    if identity != expected_identity:
        artifact.append("completion manifest identity mismatch")
    if (
        run_config_sha256 is None
        or commit.get("run_config_sha256") != run_config_sha256
    ):
        provenance.append(
            "completion run_config_sha256 must match the root run manifest"
        )
    files = commit.get("files")
    audit_name, trace_name = f"{cell.tag}.json", f"{cell.tag}.jsonl"
    if not isinstance(files, dict) or set(files) != {audit_name, trace_name}:
        artifact.append("completion manifest file set mismatch")
        files = {}
    for name, digest in files.items():
        path = directory / name
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or not path.is_file()
            or path.is_symlink()
            or _digest(path) != digest
        ):
            artifact.append(f"missing, unsafe, or hash-mismatched artifact: {name}")
    try:
        audit = _read_object(directory / audit_name)
    except Exception as exc:
        audit = None
        artifact.append(f"invalid audit: {exc}")
    commands, trace_problems = _parse_trace(directory / trace_name)
    protocol.extend(trace_problems)
    success: bool | None = None
    if audit is not None:
        if (
            audit.get("task") != cell.task
            or type(audit.get("seed")) is not int
            or audit.get("seed") != cell.seed
        ):
            artifact.append("audit identity mismatch")
        success = (
            audit.get("success") if isinstance(audit.get("success"), bool) else None
        )
        try:
            protocol.extend(
                _validate_protocol(
                    directory,
                    cell,
                    audit,
                    commands,
                    trace_name,
                    run_config_sha256,
                    run_configuration,
                )
            )
        except Exception as exc:
            protocol.append(
                f"protocol validation failed closed: {type(exc).__name__}: {exc}"
            )
        if run_configuration is not None:
            try:
                provenance.extend(_validate_audit_provenance(audit, run_configuration))
            except Exception as exc:
                provenance.append(
                    f"provenance validation failed closed: {type(exc).__name__}: {exc}"
                )
    recorded = commit.get("result")
    expected_outcome = (
        Outcome.SUCCESS
        if success is True
        else Outcome.FAILURE
        if success is False
        else None
    )
    if not isinstance(recorded, dict):
        artifact.append("completion result is not an object")
    else:
        if recorded.get("completion") != Completion.COMPLETED.value:
            artifact.append("completion result does not claim completed")
        if recorded.get("integrity") != Integrity.VALID.value:
            artifact.append("completion result does not claim valid")
        if (
            expected_outcome is None
            or recorded.get("outcome") != expected_outcome.value
        ):
            artifact.append("completion outcome disagrees with audit")
    if artifact or protocol or provenance:
        reason = "; ".join(artifact + protocol + provenance)
        result = CellResult(
            Completion.COMPLETED, Outcome.UNKNOWN, Integrity.INVALID, reason
        )
    else:
        assert expected_outcome is not None
        result = CellResult(Completion.COMPLETED, expected_outcome, Integrity.VALID)
    return Validation(
        cell,
        result,
        artifact_problems=tuple(artifact),
        protocol_problems=tuple(protocol),
        provenance_problems=tuple(provenance),
        audit=audit,
    )


def validate_run(root: Path) -> dict[str, Any]:
    """Summarize all 340 cells exclusively from :func:`validate_cell`."""
    run_manifest, run_manifest_problem = load_run_manifest(root)
    run_config_sha256 = (
        run_manifest["run_config_sha256"] if run_manifest is not None else None
    )
    validations = [
        validate_cell(
            root,
            cell,
            expected_run_config_sha256=run_config_sha256,
        )
        for cell in CELLS
    ]
    missing = [
        item for item in validations if item.result.completion is Completion.INCOMPLETE
    ]
    invalid = [
        item
        for item in validations
        if (item.artifact_problems or item.provenance_problems)
        and item.result.completion is Completion.COMPLETED
    ]
    violations = [item for item in validations if item.protocol_problems]
    provenance_violations = [
        item
        for item in validations
        if item.provenance_problems and item.result.completion is Completion.COMPLETED
    ]
    split_rows: dict[str, Any] = {}
    task_rates: list[float] = []
    for split, spec in SPLITS.items():
        split_items = [item for item in validations if item.cell.split is split]
        details = []
        for task in spec.tasks:
            items = [
                item
                for item in split_items
                if item.cell.task == task and item.result.canonical
            ]
            successes = sum(item.result.outcome is Outcome.SUCCESS for item in items)
            rate = 100.0 * successes / len(spec.seeds)
            details.append(
                {
                    "task": task,
                    "successes": successes,
                    "completed": len(items),
                    "trials": len(spec.seeds),
                    "rate": rate,
                }
            )
            task_rates.append(rate)
        canonical = [item for item in split_items if item.result.canonical]
        successes = sum(item.result.outcome is Outcome.SUCCESS for item in canonical)
        expected = len(spec.tasks) * len(spec.seeds)
        rate = 100.0 * successes / expected
        split_rows[split.value] = {
            "expected": expected,
            "completed": len(canonical),
            "successes": successes,
            "rate": rate,
            "observed_rate": 100.0 * successes / len(canonical) if canonical else 0.0,
            "tasks_detail": details,
        }
    counts = Counter(
        "canonical_success"
        if item.result.canonical and item.result.outcome is Outcome.SUCCESS
        else "canonical_failure"
        if item.result.canonical
        else "invalid"
        if item.result.integrity is Integrity.INVALID
        else "incomplete"
        for item in validations
    )
    canonical = [item for item in validations if item.result.canonical]
    successes = sum(item.result.outcome is Outcome.SUCCESS for item in canonical)
    complete = run_manifest_problem is None and len(canonical) == len(CELLS)
    # The checked-in runtime is explicitly preliminary. A future formal runtime
    # needs a new, independently reviewed manifest schema before publication.
    root_release_ready = False
    publication_ready = (
        complete
        and root_release_ready
        and all(
            item.audit is not None
            and item.audit.get("release_ready") is True
            and item.audit.get("preliminary") is False
            for item in canonical
        )
    )
    deltas = (
        {
            name: {
                split: split_rows[split]["rate"] - reference[split]
                for split in split_rows
            }
            for name, reference in PAPER_REFERENCES.items()
        }
        if publication_ready
        else None
    )
    return {
        "expected": len(CELLS),
        "complete": complete,
        "run_manifest": str(Path(root) / RUN_MANIFEST_NAME),
        "run_config_sha256": run_config_sha256,
        "run_manifest_problems": (
            [] if run_manifest_problem is None else [run_manifest_problem]
        ),
        "publication_ready": publication_ready,
        "root_release_ready": root_release_ready,
        "comparison_ready": publication_ready,
        "counts": dict(counts),
        "splits": split_rows,
        "overall": {
            "successes": successes,
            "completed": len(canonical),
            "rollouts": len(CELLS),
            "rollout_weighted_rate": 100.0 * successes / len(CELLS),
            "task_weighted_rate": sum(task_rates) / len(task_rates),
        },
        "paper": PAPER_REFERENCES,
        "delta_percentage_points": deltas,
        "missing": [_issue(item) for item in missing],
        "invalid": [_issue(item) for item in invalid],
        "protocol_violations": [_issue(item) for item in violations],
        "provenance_violations": [_issue(item) for item in provenance_violations],
        "resume": [item.cell.tag for item in validations if item.resumable],
    }


def _identity(cell: Cell) -> dict[str, Any]:
    return {"split": cell.split.value, "task": cell.task, "seed": cell.seed}


def _issue(validation: Validation) -> dict[str, Any]:
    return {**_identity(validation.cell), "problems": list(validation.problems)}


summarize = validate_run
