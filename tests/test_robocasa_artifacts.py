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

import hashlib
import json
from pathlib import Path

import pytest

from rpent.reproduce.robocasa.artifacts import (
    CellResult,
    Completion,
    Integrity,
    Outcome,
    publish_completed_cell,
    secure_artifact_subdirectory,
)
from rpent.reproduce.robocasa.deadline_supervisor import DEADLINE_PROTOCOL
from rpent.reproduce.robocasa.protocol import PROTOCOL_ID, SPLITS, cell_for
from rpent.reproduce.robocasa.provenance import make_run_manifest
from rpent.reproduce.robocasa.validator import (
    _agent_log_summary,
    validate_cell,
    validate_run,
)

FIXTURE_SHA256 = "a" * 64
PLANNER_TRANSPORT = {
    "auth_mode": "api-key",
    "provider": "rpent_responses_api",
    "endpoint_identity": "responses_api:https://planner.example/v1",
    "credential_broker": False,
    "credential_broker_protocol": None,
}


def _isolation_attestation() -> dict:
    return {
        "schema_version": 2,
        "planner_transport": dict(PLANNER_TRANSPORT),
    }


def _run_configuration(root: Path) -> dict:
    return {
        "protocol_id": PROTOCOL_ID,
        "runtime_kind": "preliminary_external_snapshot",
        "preliminary": True,
        "planner": {
            "profile": "codex-gpt55-xhigh",
            "backend": "codex",
            "model": "gpt-5.5",
            "reasoning_effort": "xhigh",
            **PLANNER_TRANSPORT,
            "wire_api": "responses",
            "provider_retry_policy": {
                "request_max_retries": 12,
                "stream_max_retries": 120,
                "stream_idle_timeout_ms": 330_000,
            },
            "credential_boundary": "cell_codex_only_generated_shell_excluded",
            "network_policy": {},
            "isolation": {},
            "codex": {},
            "isolation_preflight": _isolation_attestation(),
        },
        "memory": {
            "source": "fixture-memory",
            "revision": None,
            "manifest_sha256": FIXTURE_SHA256,
            "global_memory": False,
            "task_notes": False,
        },
        "checkpoint": {
            "checkpoint_id": "RLDX-1-FT-RC365",
            "authority_manifest_sha256": FIXTURE_SHA256,
            "fingerprint": FIXTURE_SHA256,
        },
        "runtime": {
            "root": str((root / "runtime").resolve()),
            "inputs_sha256": {
                "driver": FIXTURE_SHA256,
                "deadline": FIXTURE_SHA256,
                "preliminary_driver_adapter": FIXTURE_SHA256,
                "deadline_supervisor_adapter": FIXTURE_SHA256,
                "artifact_builder": FIXTURE_SHA256,
                "navview_xml": FIXTURE_SHA256,
            },
            "navview": {
                "camera": "mobilebase0_navview",
                "mode": "fixed",
                "pos": "0.2 0 1.6",
                "xyaxes": "0 -1 0 0.643 0 0.766",
                "fovy": "75",
            },
            "startup_timeout_seconds": 2400,
            "kill_after_seconds": 15,
            "driver_policy": {},
            "source_trees": {},
            "source_git": {},
        },
        "implementation_sha256": {"fixture.py": FIXTURE_SHA256},
    }


def _run_digest(root: Path) -> str:
    path = root / "_run_manifest.json"
    if not path.exists():
        root.mkdir(parents=True, exist_ok=True)
        manifest = make_run_manifest(_run_configuration(root))
        path.write_text(json.dumps(manifest) + "\n", encoding="utf-8")
        path.chmod(0o600)
    return json.loads(path.read_text(encoding="utf-8"))["run_config_sha256"]


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")


def _planner_status(
    *,
    timeout: bool,
    timeout_seconds: int,
    contract_sha256: str,
    freeze_sha256: str | None,
    nonce: str,
) -> dict:
    termination = "planner_timeout" if timeout else "planner_completed"
    wrapper_rc = 20 if timeout else 0
    return {
        "schema_version": 1,
        "state": "finished",
        "started_at": "2026-01-01T00:00:00Z",
        "finished_at": "2026-01-01T00:30:00Z",
        "elapsed_seconds": float(timeout_seconds),
        "timeout_seconds": timeout_seconds,
        "kill_after_seconds": 15,
        "termination_cause": termination,
        "wrapper_rc": wrapper_rc,
        "raw_termination_cause": termination,
        "raw_wrapper_rc": wrapper_rc,
        "child_pid": 9001,
        "child_pgid": 9001,
        "child_rc": 0,
        "child_signal": None,
        "deadline_fired": timeout,
        "external_deadline_fired": timeout,
        "interrupted_signal": None,
        "forwarded_signal_sent": False,
        "term_sent": False,
        "kill_sent": False,
        "orphan_cleanup_term_sent": False,
        "parent_pid": 8001,
        "parent_death_signal": 15,
        "parent_death_signal_enabled": True,
        "parent_death_observed": False,
        "subreaper_enabled": True,
        "descendant_cleanup": {
            "subreaper_enabled": True,
            "complete": True,
            "detected": [],
            "term_sent": [],
            "kill_sent": [],
            "reaped": [],
            "remaining": [],
            "errors": [],
        },
        "spawn_error": None,
        "supervision_error": None,
        "protocol_violation_detected": False,
        "policy_stop_triggered": False,
        "policy_stop_source": None,
        "provider_protocol_violation": None,
        "live_codex_protocol_violation": None,
        "stdout_total_lines": 1,
        "stdout_blank_lines": 0,
        "stdout_json_events": 1,
        "malformed_jsonl_lines": 0,
        "terminal_event_count": 1,
        "terminal_event_counts": {"turn.completed": 1, "turn.failed": 0},
        "terminal_event": {"type": "turn.completed", "line_number": 1},
        "forbidden_web_search_event_count": 0,
        "forbidden_web_search_event_counts": {
            "item.completed": 0,
            "item.started": 0,
        },
        "forbidden_web_search_event": None,
        "deadline_supervisor": {
            "protocol": DEADLINE_PROTOCOL,
            "run_id": "run-1",
            "nonce": nonce,
            "fired": timeout,
            "contract_sha256": contract_sha256,
            "freeze_sha256": freeze_sha256,
            "error": None,
        },
    }


def _audit(root: Path, cell, success: bool = True, *, timeout: bool = False):
    directory = root / cell.split.value / cell.task
    logs = directory / "run_logs" / cell.tag / "run-1"
    logs.mkdir(parents=True, exist_ok=True)
    agent = logs / "agent.log"
    status = logs / "planner_status.json"
    contract = logs / "_deadline_contract.json"
    seal = logs / "_deadline_seal.json"
    freeze = logs / "_deadline_freeze.json"
    raw_trace = logs / "command_trace.jsonl"
    agent.write_text('{"type":"turn.completed"}\n', encoding="utf-8")
    nonce = "0123456789abcdef0123456789abcdef"
    driver = {"pid": 7001, "pgid": 7001, "start_time_ticks": 123456}
    timeout_seconds = SPLITS[cell.split].timeout_seconds
    timeout_ns = timeout_seconds * 1_000_000_000
    started_ns = 1_000_000_000
    deadline_ns = started_ns + timeout_ns
    contract_value = {
        "schema_version": 1,
        "protocol": DEADLINE_PROTOCOL,
        "run_id": "run-1",
        "nonce": nonce,
        "started_monotonic_ns": started_ns,
        "deadline_monotonic_ns": deadline_ns,
        "timeout_ns": timeout_ns,
        "driver": driver,
        "external_deadline_sha256": FIXTURE_SHA256,
    }
    _write_json(contract, contract_value)
    contract_sha256 = _digest(contract)
    seal_sha256 = None
    freeze_sha256 = None
    driver_stop = None
    if timeout:
        trace = b'{"action":"release"}\n'
        raw_trace.write_bytes(trace)
        (logs / "done_00.flag").write_bytes(b"")
        (logs / "done_01.flag").write_bytes(b"")
        _write_json(
            logs / "state_00.json",
            {"step": 0, "success": False, "task_language": "open the drawer"},
        )
        _write_json(
            logs / "state_01.json",
            {"step": 1, "success": success, "task_language": "open the drawer"},
        )
        _write_json(logs / "log_01.json", {"command": {"action": "release"}})
        _write_json(
            logs / "_deadline_commit_01.json",
            {
                "schema_version": 1,
                "protocol": DEADLINE_PROTOCOL,
                "run_id": "run-1",
                "nonce": nonce,
                "step": 1,
                "deadline_monotonic_ns": deadline_ns,
                "done_published_monotonic_ns": deadline_ns - 1,
                "contract_sha256": contract_sha256,
            },
        )
        seal_value = {
            "schema_version": 1,
            "protocol": DEADLINE_PROTOCOL,
            "run_id": "run-1",
            "nonce": nonce,
            "deadline_monotonic_ns": deadline_ns,
            "sealed_monotonic_ns": deadline_ns + 1,
            "contract_sha256": contract_sha256,
        }
        _write_json(seal, seal_value)
        seal_sha256 = _digest(seal)
        files = {
            name: _digest(logs / name)
            for name in (
                "done_00.flag",
                "done_01.flag",
                "state_00.json",
                "state_01.json",
                "log_01.json",
                "_deadline_commit_01.json",
            )
        }
        driver_stop = {
            "observed_state": "T",
            "supervisor_stopped": True,
            "kill_sent": True,
            "kill_observed": True,
        }
        freeze_value = {
            "schema_version": 1,
            "protocol": DEADLINE_PROTOCOL,
            "run_id": "run-1",
            "nonce": nonce,
            "deadline_monotonic_ns": deadline_ns,
            "frozen_monotonic_ns": deadline_ns + 2,
            "contract_sha256": contract_sha256,
            "seal_sha256": seal_sha256,
            "driver": driver,
            "driver_stop": driver_stop,
            "dropped_done_steps": [],
            "prefix_step": 1,
            "success": success,
            "final_state_sha256": files["state_01.json"],
            "command_trace_sha256": hashlib.sha256(trace).hexdigest(),
            "raw_command_trace_sha256": _digest(raw_trace),
            "files_sha256": files,
        }
        _write_json(freeze, freeze_value)
        freeze_sha256 = _digest(freeze)
    _write_json(
        status,
        _planner_status(
            timeout=timeout,
            timeout_seconds=timeout_seconds,
            contract_sha256=contract_sha256,
            freeze_sha256=freeze_sha256,
            nonce=nonce,
        ),
    )

    configuration = _run_configuration(root)
    checkpoint_files = {
        f"model/model-{index:05d}-of-00003.safetensors": FIXTURE_SHA256
        for index in range(1, 4)
    }
    return {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "suite": "robocasa365",
        "source": "harness",
        "split": cell.split.value,
        "task": cell.task,
        "seed": cell.seed,
        "success": success,
        "valid": True,
        "failure_type": None
        if success
        else "rollout_timeout"
        if timeout
        else "task_failure",
        "steps": 1,
        "commit_protocol": "done_marker_v1",
        "trace_recovered_from_logs": False,
        "incomplete_commands_dropped": 0,
        "agent_exit_code": 20 if timeout else 0,
        "termination_cause": "planner_timeout" if timeout else "planner_completed",
        "rollout_timeout": timeout,
        "infra_status": "ok",
        "planner_status": "timeout" if timeout else "completed",
        "environment_success": success,
        "planner_profile": "codex-gpt55-xhigh",
        "planner_backend": "codex",
        "planner_model": "gpt-5.5",
        "planner_reasoning_effort": "xhigh",
        "planner_auth_mode": PLANNER_TRANSPORT["auth_mode"],
        "planner_provider": PLANNER_TRANSPORT["provider"],
        "planner_endpoint_identity": PLANNER_TRANSPORT["endpoint_identity"],
        "preliminary": True,
        "release_ready": False,
        "run_id": "run-1",
        "run_config_sha256": _run_digest(root),
        "runtime_root": configuration["runtime"]["root"],
        "runtime_scripts": {
            "external_driver": FIXTURE_SHA256,
            "external_deadline": FIXTURE_SHA256,
            "driver_adapter": FIXTURE_SHA256,
            "deadline_supervisor": FIXTURE_SHA256,
            "artifact_builder": FIXTURE_SHA256,
        },
        "deadline": {
            "protocol": DEADLINE_PROTOCOL,
            "contract_sha256": contract_sha256,
            "seal_sha256": seal_sha256,
            "freeze_sha256": freeze_sha256,
            "deadline_monotonic_ns": deadline_ns,
            "timeout_ns": timeout_ns,
            "driver": driver,
            "driver_stop": driver_stop,
            "driver_return_code_before_stop": -9 if timeout else None,
            "prefix_step": 1 if timeout else None,
            "success_at_deadline": success if timeout else None,
        },
        "memory_source": {
            "kind": "fixture-memory",
            "revision": None,
            "manifest_sha256": FIXTURE_SHA256,
        },
        "navview": {
            "xml_sha256": FIXTURE_SHA256,
            **configuration["runtime"]["navview"],
        },
        "checkpoint": {
            **configuration["checkpoint"],
            "files": checkpoint_files,
        },
        "checkpoint_sha256": {
            f"model-{index:05d}-of-00003.safetensors": FIXTURE_SHA256
            for index in range(1, 4)
        },
        "task_language": "open the drawer",
        "task_memory": {
            "seed": 0,
            "audit": f"{cell.task}_s0.json",
            "command_trace": f"{cell.task}_s0.jsonl",
        },
        "task_memory_available": True,
        "task_notes_available": False,
        "memory_role": "formal_evaluation",
        "global_memory": False,
        "perception_isolation": True,
        "reset_enabled": False,
        "reset_seed": 4_200_000 + cell.seed
        if cell.split.value == "composite_seen"
        else None,
        "command_trace": f"{cell.tag}.jsonl",
        "artifact_errors": [],
        "evidence": {
            "agent_log": {
                "path": str(agent.relative_to(directory)),
                "sha256": _digest(agent),
            },
            "planner_status": {
                "path": str(status.relative_to(directory)),
                "sha256": _digest(status),
            },
            "deadline_contract": {
                "path": str(contract.relative_to(directory)),
                "sha256": contract_sha256,
            },
            **(
                {
                    "deadline_seal": {
                        "path": str(seal.relative_to(directory)),
                        "sha256": seal_sha256,
                    },
                    "deadline_freeze": {
                        "path": str(freeze.relative_to(directory)),
                        "sha256": freeze_sha256,
                    },
                }
                if timeout
                else {}
            ),
        },
    }


def _set_agent_events(root: Path, cell, audit, events) -> None:
    path = root / cell.split.value / cell.task / audit["evidence"]["agent_log"]["path"]
    path.write_text(
        "".join(json.dumps(event) + "\n" for event in events),
        encoding="utf-8",
    )
    audit["evidence"]["agent_log"]["sha256"] = hashlib.sha256(
        path.read_bytes()
    ).hexdigest()
    summary = _agent_log_summary(path)
    assert summary is not None
    status_path = _evidence_path(root, cell, audit, "planner_status")
    status = _read_json(status_path)
    status.update(summary)
    _sync_status_evidence(root, cell, audit, status)


def _evidence_path(root: Path, cell, audit: dict, key: str) -> Path:
    directory = root / cell.split.value / cell.task
    return directory / audit["evidence"][key]["path"]


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _sync_status_evidence(root: Path, cell, audit: dict, status: dict) -> None:
    status_path = _evidence_path(root, cell, audit, "planner_status")
    _write_json(status_path, status)
    audit["evidence"]["planner_status"]["sha256"] = _digest(status_path)


def _refresh_freeze(root: Path, cell, audit: dict) -> None:
    freeze_path = _evidence_path(root, cell, audit, "deadline_freeze")
    freeze = _read_json(freeze_path)
    log_directory = freeze_path.parent
    files = freeze.get("files_sha256")
    if isinstance(files, dict):
        for name in files:
            path = log_directory / name
            if path.is_file():
                files[name] = _digest(path)
    prefix_step = freeze.get("prefix_step")
    if type(prefix_step) is int and prefix_step >= 0:
        final_state = log_directory / f"state_{prefix_step:02d}.json"
        if final_state.is_file():
            freeze["final_state_sha256"] = _digest(final_state)
        commands = []
        for step in range(1, prefix_step + 1):
            log_path = log_directory / f"log_{step:02d}.json"
            if log_path.is_file():
                commands.append(_read_json(log_path).get("command"))
        trace = b"".join(
            (json.dumps(command, separators=(",", ":")) + "\n").encode("utf-8")
            for command in commands
        )
        freeze["command_trace_sha256"] = hashlib.sha256(trace).hexdigest()
    raw_trace = log_directory / "command_trace.jsonl"
    if raw_trace.is_file():
        freeze["raw_command_trace_sha256"] = _digest(raw_trace)
    _write_json(freeze_path, freeze)
    freeze_sha256 = _digest(freeze_path)
    audit["evidence"]["deadline_freeze"]["sha256"] = freeze_sha256
    audit["deadline"]["freeze_sha256"] = freeze_sha256
    status_path = _evidence_path(root, cell, audit, "planner_status")
    status = _read_json(status_path)
    status["deadline_supervisor"]["freeze_sha256"] = freeze_sha256
    _sync_status_evidence(root, cell, audit, status)


def _publish_for_validation(root: Path, cell, audit: dict):
    result = CellResult(
        Completion.COMPLETED,
        Outcome.SUCCESS if audit.get("success") is True else Outcome.FAILURE,
        Integrity.VALID,
    )
    publish_completed_cell(root, cell, result, audit, [{"action": "release"}])
    return validate_cell(root, cell)


def test_only_completed_valid_results_are_published_and_validated(tmp_path: Path):
    cell = cell_for("atomic", "OpenDrawer", 1)
    result = CellResult(Completion.COMPLETED, Outcome.SUCCESS, Integrity.VALID)
    publish_completed_cell(
        tmp_path,
        cell,
        result,
        _audit(tmp_path, cell),
        [{"action": "release"}],
    )
    validation = validate_cell(tmp_path, cell)
    assert validation.result == result
    assert not validation.resumable
    summary = validate_run(tmp_path)
    assert summary["counts"]["canonical_success"] == 1
    assert summary["expected"] == 340
    assert summary["splits"]["atomic"]["expected"] == 180
    assert summary["splits"]["composite_seen"]["expected"] == 80
    assert summary["splits"]["composite_unseen"]["expected"] == 80
    assert len(summary["paper"]) == 3
    assert summary["delta_percentage_points"] is None
    assert summary["publication_ready"] is False
    assert len(summary["missing"]) == 339
    assert len(summary["resume"]) == 339
    drawer = next(
        row
        for row in summary["splits"]["atomic"]["tasks_detail"]
        if row["task"] == "OpenDrawer"
    )
    assert drawer == {
        "task": "OpenDrawer",
        "successes": 1,
        "completed": 1,
        "trials": 10,
        "rate": 10.0,
    }


def test_incomplete_or_invalid_results_are_never_canonical(tmp_path: Path):
    cell = cell_for("atomic", "OpenDrawer", 1)
    incomplete = CellResult(Completion.INCOMPLETE, Outcome.UNKNOWN, Integrity.UNKNOWN)
    with pytest.raises(ValueError, match="only completed"):
        publish_completed_cell(tmp_path, cell, incomplete, {}, [])
    assert validate_cell(tmp_path, cell).resumable


def test_manifest_hash_tampering_is_resume_eligible(tmp_path: Path):
    cell = cell_for("atomic", "OpenDrawer", 1)
    publish_completed_cell(
        tmp_path,
        cell,
        CellResult(Completion.COMPLETED, Outcome.FAILURE, Integrity.VALID),
        _audit(tmp_path, cell, False),
        [{"action": "release"}],
    )
    artifact = tmp_path / "atomic" / cell.task / f"{cell.tag}.jsonl"
    artifact.write_text('{"action":"navigate_to"}\n')
    validation = validate_cell(tmp_path, cell)
    assert validation.result.integrity is Integrity.INVALID
    assert validation.resumable


def test_cell_result_rejects_contradictory_axes():
    with pytest.raises(ValueError):
        CellResult(Completion.INCOMPLETE, Outcome.SUCCESS, Integrity.UNKNOWN)


@pytest.mark.parametrize(
    ("field", "value", "problem"),
    [
        ("global_memory", True, "global_memory"),
        ("task_notes_available", True, "task_notes_available"),
        ("termination_cause", "planner_timeout", "termination_cause"),
        ("steps", 2, "steps"),
    ],
)
def test_protocol_fields_fail_closed(tmp_path: Path, field, value, problem):
    cell = cell_for("atomic", "OpenDrawer", 1)
    audit = _audit(tmp_path, cell)
    audit[field] = value
    publish_completed_cell(
        tmp_path,
        cell,
        CellResult(Completion.COMPLETED, Outcome.SUCCESS, Integrity.VALID),
        audit,
        [{"action": "vla_act", "prompt": "open the drawer"}],
    )
    validation = validate_cell(tmp_path, cell)
    assert any(problem in item for item in validation.protocol_problems)


def test_vla_prompt_memory_and_web_search_are_strict(tmp_path: Path):
    cell = cell_for("atomic", "OpenDrawer", 1)
    audit = _audit(tmp_path, cell)
    agent_path = (
        tmp_path / "atomic" / cell.task / audit["evidence"]["agent_log"]["path"]
    )
    agent_path.write_text(
        json.dumps({"type": "item.completed", "item": {"type": "web_search"}}) + "\n"
    )
    audit["evidence"]["agent_log"]["sha256"] = hashlib.sha256(
        agent_path.read_bytes()
    ).hexdigest()
    audit["task_memory"]["audit"] = "Other_s0.json"
    publish_completed_cell(
        tmp_path,
        cell,
        CellResult(Completion.COMPLETED, Outcome.SUCCESS, Integrity.VALID),
        audit,
        [{"action": "vla_act", "prompt": "open the drawer"}],
    )
    problems = validate_cell(tmp_path, cell).protocol_problems
    assert any("task memory" in item for item in problems)
    assert any("web_search" in item for item in problems)

    with pytest.raises(ValueError, match="task_language"):
        publish_completed_cell(
            tmp_path,
            cell,
            CellResult(Completion.COMPLETED, Outcome.SUCCESS, Integrity.VALID),
            _audit(tmp_path, cell),
            [{"action": "vla_act", "prompt": "different"}],
        )


def test_agent_log_accepts_only_frozen_event_and_item_allowlists(tmp_path: Path):
    cell = cell_for("atomic", "OpenDrawer", 1)
    audit = _audit(tmp_path, cell)
    events = [
        {"type": "thread.started", "thread_id": "thread-1"},
        {"type": "turn.started"},
    ]
    events.extend(
        {
            "type": event_type,
            "item": {"type": item_type, "id": f"{event_type}-{item_type}"},
        }
        for event_type, item_type in zip(
            (
                "item.started",
                "item.updated",
                "item.completed",
                "item.completed",
                "item.completed",
            ),
            (
                "agent_message",
                "reasoning",
                "command_execution",
                "image_view",
                "todo_list",
            ),
            strict=True,
        )
    )
    events.append({"type": "turn.completed"})
    _set_agent_events(tmp_path, cell, audit, events)
    publish_completed_cell(
        tmp_path,
        cell,
        CellResult(Completion.COMPLETED, Outcome.SUCCESS, Integrity.VALID),
        audit,
        [{"action": "release"}],
    )

    assert validate_cell(tmp_path, cell).result.canonical


def test_agent_log_accepts_recovered_transport_reconnect(tmp_path: Path):
    cell = cell_for("atomic", "OpenDrawer", 1)
    audit = _audit(tmp_path, cell)
    _set_agent_events(
        tmp_path,
        cell,
        audit,
        [
            {
                "type": "error",
                "message": "Reconnecting... 1/5 (stream disconnected before completion)",
            },
            {"type": "turn.completed"},
        ],
    )
    publish_completed_cell(
        tmp_path,
        cell,
        CellResult(Completion.COMPLETED, Outcome.SUCCESS, Integrity.VALID),
        audit,
        [{"action": "release"}],
    )

    assert validate_cell(tmp_path, cell).result.canonical


@pytest.mark.parametrize("event_type", ["turn.failed", "error", "future.event"])
def test_agent_log_rejects_failed_and_unknown_top_level_events(
    tmp_path: Path, event_type
):
    cell = cell_for("atomic", "OpenDrawer", 1)
    audit = _audit(tmp_path, cell)
    _set_agent_events(
        tmp_path,
        cell,
        audit,
        [{"type": event_type}, {"type": "turn.completed"}],
    )
    publish_completed_cell(
        tmp_path,
        cell,
        CellResult(Completion.COMPLETED, Outcome.SUCCESS, Integrity.VALID),
        audit,
        [{"action": "release"}],
    )

    problems = validate_cell(tmp_path, cell).protocol_problems
    assert any(event_type in problem for problem in problems)


@pytest.mark.parametrize(
    "item_type",
    [
        "file_change",
        "mcp_tool_call",
        "collab_tool_call",
        "web_search",
        "future_item",
    ],
)
def test_agent_log_rejects_side_effecting_and_unknown_items(tmp_path: Path, item_type):
    cell = cell_for("atomic", "OpenDrawer", 1)
    audit = _audit(tmp_path, cell)
    _set_agent_events(
        tmp_path,
        cell,
        audit,
        [
            {"type": "item.completed", "item": {"type": item_type}},
            {"type": "turn.completed"},
        ],
    )
    publish_completed_cell(
        tmp_path,
        cell,
        CellResult(Completion.COMPLETED, Outcome.SUCCESS, Integrity.VALID),
        audit,
        [{"action": "release"}],
    )

    problems = validate_cell(tmp_path, cell).protocol_problems
    assert any(item_type in problem for problem in problems)


@pytest.mark.parametrize(
    "event_type", ["item.started", "item.updated", "item.completed"]
)
@pytest.mark.parametrize("item", [None, "not-an-object"])
def test_every_item_event_requires_an_item_object(tmp_path: Path, event_type, item):
    cell = cell_for("atomic", "OpenDrawer", 1)
    audit = _audit(tmp_path, cell)
    _set_agent_events(
        tmp_path,
        cell,
        audit,
        [
            {"type": event_type, "item": item},
            {"type": "turn.completed"},
        ],
    )
    publish_completed_cell(
        tmp_path,
        cell,
        CellResult(Completion.COMPLETED, Outcome.SUCCESS, Integrity.VALID),
        audit,
        [{"action": "release"}],
    )

    problems = validate_cell(tmp_path, cell).protocol_problems
    assert any("without item object" in problem for problem in problems)


def test_completed_planner_requires_turn_completed_event(tmp_path: Path):
    cell = cell_for("atomic", "OpenDrawer", 1)
    audit = _audit(tmp_path, cell)
    _set_agent_events(tmp_path, cell, audit, [{"type": "turn.started"}])
    publish_completed_cell(
        tmp_path,
        cell,
        CellResult(Completion.COMPLETED, Outcome.SUCCESS, Integrity.VALID),
        audit,
        [{"action": "release"}],
    )

    problems = validate_cell(tmp_path, cell).protocol_problems
    assert any("no turn.completed" in problem for problem in problems)


def test_planner_timeout_may_omit_turn_completed_event(tmp_path: Path):
    cell = cell_for("atomic", "OpenDrawer", 1)
    audit = _audit(tmp_path, cell, success=False, timeout=True)
    _set_agent_events(tmp_path, cell, audit, [{"type": "turn.started"}])
    publish_completed_cell(
        tmp_path,
        cell,
        CellResult(Completion.COMPLETED, Outcome.FAILURE, Integrity.VALID),
        audit,
        [{"action": "release"}],
    )

    assert validate_cell(tmp_path, cell).result.canonical


def test_planner_timeout_takes_priority_over_simultaneous_turn_failed(
    tmp_path: Path,
):
    cell = cell_for("atomic", "OpenDrawer", 1)
    audit = _audit(tmp_path, cell, success=False, timeout=True)
    _set_agent_events(tmp_path, cell, audit, [{"type": "turn.failed"}])
    publish_completed_cell(
        tmp_path,
        cell,
        CellResult(Completion.COMPLETED, Outcome.FAILURE, Integrity.VALID),
        audit,
        [{"action": "release"}],
    )

    assert validate_cell(tmp_path, cell).result.canonical


def test_planner_timeout_does_not_reclassify_reconnect_as_infrastructure(
    tmp_path: Path,
):
    cell = cell_for("atomic", "OpenDrawer", 1)
    audit = _audit(tmp_path, cell, success=False, timeout=True)
    _set_agent_events(
        tmp_path,
        cell,
        audit,
        [
            {"type": "turn.started"},
            {
                "type": "error",
                "message": "Reconnecting... 1/5 (stream disconnected before completion)",
            },
        ],
    )
    publish_completed_cell(
        tmp_path,
        cell,
        CellResult(Completion.COMPLETED, Outcome.FAILURE, Integrity.VALID),
        audit,
        [{"action": "release"}],
    )

    assert validate_cell(tmp_path, cell).result.canonical


def test_success_committed_before_planner_timeout_remains_canonical(
    tmp_path: Path,
):
    cell = cell_for("atomic", "OpenDrawer", 1)
    audit = _audit(tmp_path, cell, success=True, timeout=True)
    audit["agent_exit_code"] = 20
    _set_agent_events(
        tmp_path,
        cell,
        audit,
        [
            {"type": "turn.started"},
            {
                "type": "error",
                "message": "Reconnecting... 1/5 (stream disconnected before completion)",
            },
        ],
    )
    publish_completed_cell(
        tmp_path,
        cell,
        CellResult(Completion.COMPLETED, Outcome.SUCCESS, Integrity.VALID),
        audit,
        [{"action": "release"}],
    )

    assert validate_cell(tmp_path, cell).result.canonical
    summary = validate_run(tmp_path)
    assert summary["counts"]["canonical_success"] == 1
    assert summary["overall"]["completed"] == 1
    assert cell.tag not in summary["resume"]


@pytest.mark.parametrize("mutation", ["missing", "extra"])
def test_deadline_evidence_requires_the_exact_file_set(tmp_path: Path, mutation: str):
    cell = cell_for("atomic", "OpenDrawer", 1)
    audit = _audit(tmp_path, cell)
    if mutation == "missing":
        del audit["evidence"]["deadline_contract"]
    else:
        audit["evidence"]["unexpected"] = dict(audit["evidence"]["agent_log"])

    validation = _publish_for_validation(tmp_path, cell, audit)

    assert not validation.result.canonical
    assert any(
        "exactly the deadline protocol files" in item for item in validation.problems
    )


@pytest.mark.parametrize("key", ["deadline_seal", "deadline_freeze"])
def test_timeout_requires_both_seal_and_freeze_evidence(tmp_path: Path, key: str):
    cell = cell_for("atomic", "OpenDrawer", 1)
    audit = _audit(tmp_path, cell, success=False, timeout=True)
    del audit["evidence"][key]

    validation = _publish_for_validation(tmp_path, cell, audit)

    assert not validation.result.canonical
    assert any(
        "exactly the deadline protocol files" in item for item in validation.problems
    )


@pytest.mark.parametrize("name", ["_deadline_seal.json", "_deadline_freeze.json"])
def test_completed_archive_rejects_any_deadline_seal_or_freeze(
    tmp_path: Path, name: str
):
    cell = cell_for("atomic", "OpenDrawer", 1)
    audit = _audit(tmp_path, cell)
    log_directory = _evidence_path(tmp_path, cell, audit, "deadline_contract").parent
    _write_json(log_directory / name, {"unexpected": True})

    validation = _publish_for_validation(tmp_path, cell, audit)

    assert not validation.result.canonical
    assert any("unexpectedly contains" in item for item in validation.problems)


@pytest.mark.parametrize(
    ("key", "timeout", "problem"),
    [
        ("deadline_contract", False, "deadline contract has an invalid schema"),
        ("planner_status", False, "planner_status has an invalid schema"),
        ("deadline_seal", True, "deadline seal has an invalid schema"),
        ("deadline_freeze", True, "deadline freeze has an invalid schema"),
    ],
)
def test_deadline_evidence_objects_have_exact_schemas(
    tmp_path: Path, key: str, timeout: bool, problem: str
):
    cell = cell_for("atomic", "OpenDrawer", 1)
    audit = _audit(tmp_path, cell, success=not timeout, timeout=timeout)
    path = _evidence_path(tmp_path, cell, audit, key)
    value = _read_json(path)
    value["unexpected"] = True
    _write_json(path, value)
    digest = _digest(path)
    audit["evidence"][key]["sha256"] = digest
    if key == "deadline_contract":
        audit["deadline"]["contract_sha256"] = digest
        status = _read_json(_evidence_path(tmp_path, cell, audit, "planner_status"))
        status["deadline_supervisor"]["contract_sha256"] = digest
        _sync_status_evidence(tmp_path, cell, audit, status)
    elif key == "deadline_seal":
        audit["deadline"]["seal_sha256"] = digest
    elif key == "deadline_freeze":
        audit["deadline"]["freeze_sha256"] = digest
        status = _read_json(_evidence_path(tmp_path, cell, audit, "planner_status"))
        status["deadline_supervisor"]["freeze_sha256"] = digest
        _sync_status_evidence(tmp_path, cell, audit, status)

    validation = _publish_for_validation(tmp_path, cell, audit)

    assert not validation.result.canonical
    assert any(problem in item for item in validation.problems)


@pytest.mark.parametrize("field", ["timeout_ns", "external_deadline_sha256"])
def test_deadline_contract_binds_protocol_timeout_and_runtime_source(
    tmp_path: Path, field: str
):
    cell = cell_for("atomic", "OpenDrawer", 1)
    audit = _audit(tmp_path, cell)
    path = _evidence_path(tmp_path, cell, audit, "deadline_contract")
    contract = _read_json(path)
    if field == "timeout_ns":
        contract[field] += 1
        contract["deadline_monotonic_ns"] += 1
        audit["deadline"]["timeout_ns"] += 1
        audit["deadline"]["deadline_monotonic_ns"] += 1
    else:
        contract[field] = "b" * 64
    _write_json(path, contract)
    digest = _digest(path)
    audit["evidence"]["deadline_contract"]["sha256"] = digest
    audit["deadline"]["contract_sha256"] = digest
    status = _read_json(_evidence_path(tmp_path, cell, audit, "planner_status"))
    status["deadline_supervisor"]["contract_sha256"] = digest
    _sync_status_evidence(tmp_path, cell, audit, status)

    validation = _publish_for_validation(tmp_path, cell, audit)

    assert not validation.result.canonical
    assert any("root protocol" in item for item in validation.problems)


@pytest.mark.parametrize(
    "name",
    [
        "done_01.flag",
        "state_01.json",
        "log_01.json",
        "_deadline_commit_01.json",
        "command_trace.jsonl",
    ],
)
def test_timeout_rejects_missing_frozen_journal_file(tmp_path: Path, name: str):
    cell = cell_for("atomic", "OpenDrawer", 1)
    audit = _audit(tmp_path, cell, success=False, timeout=True)
    log_directory = _evidence_path(tmp_path, cell, audit, "deadline_freeze").parent
    (log_directory / name).unlink()

    validation = _publish_for_validation(tmp_path, cell, audit)

    assert not validation.result.canonical
    assert any(
        phrase in " ".join(validation.problems)
        for phrase in ("journal", "done markers", "command trace")
    )


def test_timeout_rejects_late_receipt_even_when_all_hashes_match(tmp_path: Path):
    cell = cell_for("atomic", "OpenDrawer", 1)
    audit = _audit(tmp_path, cell, success=False, timeout=True)
    receipt_path = (
        _evidence_path(tmp_path, cell, audit, "deadline_freeze").parent
        / "_deadline_commit_01.json"
    )
    receipt = _read_json(receipt_path)
    receipt["done_published_monotonic_ns"] = receipt["deadline_monotonic_ns"]
    _write_json(receipt_path, receipt)
    _refresh_freeze(tmp_path, cell, audit)

    validation = _publish_for_validation(tmp_path, cell, audit)

    assert not validation.result.canonical
    assert any("published after the deadline" in item for item in validation.problems)


def test_timeout_rejects_receipt_before_contract_start_when_hashes_match(
    tmp_path: Path,
):
    cell = cell_for("atomic", "OpenDrawer", 1)
    audit = _audit(tmp_path, cell, success=False, timeout=True)
    receipt_path = (
        _evidence_path(tmp_path, cell, audit, "deadline_freeze").parent
        / "_deadline_commit_01.json"
    )
    receipt = _read_json(receipt_path)
    receipt["done_published_monotonic_ns"] = 0
    _write_json(receipt_path, receipt)
    _refresh_freeze(tmp_path, cell, audit)

    validation = _publish_for_validation(tmp_path, cell, audit)

    assert not validation.result.canonical
    assert any(
        "before the deadline contract started" in item for item in validation.problems
    )


@pytest.mark.parametrize("target", ["success", "command"])
def test_timeout_rebuilds_success_and_trace_from_frozen_journal(
    tmp_path: Path, target: str
):
    cell = cell_for("atomic", "OpenDrawer", 1)
    audit = _audit(tmp_path, cell, success=True, timeout=True)
    log_directory = _evidence_path(tmp_path, cell, audit, "deadline_freeze").parent
    if target == "success":
        state_path = log_directory / "state_01.json"
        state = _read_json(state_path)
        state["success"] = False
        _write_json(state_path, state)
    else:
        _write_json(
            log_directory / "log_01.json",
            {"command": {"action": "release", "steps": 2}},
        )
    _refresh_freeze(tmp_path, cell, audit)

    validation = _publish_for_validation(tmp_path, cell, audit)

    assert not validation.result.canonical
    assert any(
        phrase in " ".join(validation.problems)
        for phrase in ("success disagrees", "command trace")
    )


def test_timeout_rejects_extra_done_marker_outside_frozen_prefix(tmp_path: Path):
    cell = cell_for("atomic", "OpenDrawer", 1)
    audit = _audit(tmp_path, cell, success=False, timeout=True)
    log_directory = _evidence_path(tmp_path, cell, audit, "deadline_freeze").parent
    (log_directory / "done_02.flag").write_bytes(b"")

    validation = _publish_for_validation(tmp_path, cell, audit)

    assert not validation.result.canonical
    assert any("extra done markers" in item for item in validation.problems)


@pytest.mark.parametrize("dropped", [[2, 2], [3, 2], [1]])
def test_timeout_rejects_noncanonical_dropped_done_steps(
    tmp_path: Path, dropped: list[int]
):
    cell = cell_for("atomic", "OpenDrawer", 1)
    audit = _audit(tmp_path, cell, success=False, timeout=True)
    freeze_path = _evidence_path(tmp_path, cell, audit, "deadline_freeze")
    freeze = _read_json(freeze_path)
    freeze["dropped_done_steps"] = dropped
    _write_json(freeze_path, freeze)
    _refresh_freeze(tmp_path, cell, audit)

    validation = _publish_for_validation(tmp_path, cell, audit)

    assert not validation.result.canonical
    assert any("sealed canonical audit" in item for item in validation.problems)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("kill_sent", False),
        ("kill_observed", False),
        ("supervisor_stopped", False),
    ],
)
def test_timeout_rejects_invalid_driver_termination_proof(
    tmp_path: Path, field: str, value: bool
):
    cell = cell_for("atomic", "OpenDrawer", 1)
    audit = _audit(tmp_path, cell, success=False, timeout=True)
    freeze_path = _evidence_path(tmp_path, cell, audit, "deadline_freeze")
    freeze = _read_json(freeze_path)
    freeze["driver_stop"][field] = value
    audit["deadline"]["driver_stop"] = dict(freeze["driver_stop"])
    _write_json(freeze_path, freeze)
    _refresh_freeze(tmp_path, cell, audit)

    validation = _publish_for_validation(tmp_path, cell, audit)

    assert not validation.result.canonical
    assert any("sealed canonical audit" in item for item in validation.problems)


@pytest.mark.parametrize(("return_code", "canonical"), [(75, True), (9, False)])
def test_timeout_binds_passive_driver_exit_to_adapter_return_code(
    tmp_path: Path, return_code: int, canonical: bool
):
    cell = cell_for("atomic", "OpenDrawer", 1)
    audit = _audit(tmp_path, cell, success=False, timeout=True)
    freeze_path = _evidence_path(tmp_path, cell, audit, "deadline_freeze")
    freeze = _read_json(freeze_path)
    passive_stop = {
        "observed_state": "Z",
        "supervisor_stopped": False,
        "kill_sent": False,
        "kill_observed": True,
    }
    freeze["driver_stop"] = passive_stop
    audit["deadline"]["driver_stop"] = dict(passive_stop)
    audit["deadline"]["driver_return_code_before_stop"] = return_code
    _write_json(freeze_path, freeze)
    _refresh_freeze(tmp_path, cell, audit)

    validation = _publish_for_validation(tmp_path, cell, audit)

    assert validation.result.canonical is canonical
    if not canonical:
        assert any(
            "timeout deadline freeze identity" in item for item in validation.problems
        )


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("log_00.json", {"command": {"action": "release"}}),
        ("_deadline_commit_00.json", {"unexpected": True}),
        ("state_-1.json", {"step": -1, "success": False}),
    ],
)
def test_timeout_rejects_journals_outside_the_canonical_prefix_or_tail(
    tmp_path: Path, name: str, value: dict
):
    cell = cell_for("atomic", "OpenDrawer", 1)
    audit = _audit(tmp_path, cell, success=False, timeout=True)
    log_directory = _evidence_path(tmp_path, cell, audit, "deadline_freeze").parent
    _write_json(log_directory / name, value)

    validation = _publish_for_validation(tmp_path, cell, audit)

    assert not validation.result.canonical
    assert any(
        "outside the canonical prefix/tail" in item for item in validation.problems
    )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("raw_termination_cause", "planner_failed"),
        ("raw_wrapper_rc", 22),
        ("stdout_total_lines", 999),
        ("terminal_event_count", 0),
    ],
)
def test_planner_status_is_replayed_from_raw_outcome_and_agent_log(
    tmp_path: Path, field: str, value
):
    cell = cell_for("atomic", "OpenDrawer", 1)
    audit = _audit(tmp_path, cell)
    status_path = _evidence_path(tmp_path, cell, audit, "planner_status")
    status = _read_json(status_path)
    status[field] = value
    _sync_status_evidence(tmp_path, cell, audit, status)

    validation = _publish_for_validation(tmp_path, cell, audit)

    assert not validation.result.canonical
    assert any("planner" in item for item in validation.problems)


def test_timeout_accepts_external_completion_observed_at_supervisor_deadline(
    tmp_path: Path,
):
    cell = cell_for("atomic", "OpenDrawer", 1)
    audit = _audit(tmp_path, cell, success=False, timeout=True)
    status_path = _evidence_path(tmp_path, cell, audit, "planner_status")
    status = _read_json(status_path)
    status["external_deadline_fired"] = False
    status["raw_termination_cause"] = "planner_completed"
    status["raw_wrapper_rc"] = 0
    _sync_status_evidence(tmp_path, cell, audit, status)

    validation = _publish_for_validation(tmp_path, cell, audit)

    assert validation.result.canonical


@pytest.mark.parametrize("raw_agent_log", ["", "{truncated-json\n"])
def test_timeout_accepts_missing_or_truncated_terminal_at_supervisor_deadline(
    tmp_path: Path, raw_agent_log: str
):
    cell = cell_for("atomic", "OpenDrawer", 1)
    audit = _audit(tmp_path, cell, success=False, timeout=True)
    agent_path = _evidence_path(tmp_path, cell, audit, "agent_log")
    agent_path.write_text(raw_agent_log, encoding="utf-8")
    audit["evidence"]["agent_log"]["sha256"] = _digest(agent_path)
    summary = _agent_log_summary(agent_path)
    assert summary is not None
    status_path = _evidence_path(tmp_path, cell, audit, "planner_status")
    status = _read_json(status_path)
    status.update(summary)
    status["external_deadline_fired"] = False
    status["raw_termination_cause"] = "planner_protocol_error"
    status["raw_wrapper_rc"] = 23
    _sync_status_evidence(tmp_path, cell, audit, status)

    validation = _publish_for_validation(tmp_path, cell, audit)

    assert validation.result.canonical


def test_audit_agent_exit_code_must_match_planner_wrapper(tmp_path: Path):
    cell = cell_for("atomic", "OpenDrawer", 1)
    audit = _audit(tmp_path, cell, success=False, timeout=True)
    audit["agent_exit_code"] = 0

    validation = _publish_for_validation(tmp_path, cell, audit)

    assert not validation.result.canonical
    assert any("agent_exit_code" in item for item in validation.problems)


def test_timeout_binds_audit_task_language_to_every_frozen_state(tmp_path: Path):
    cell = cell_for("atomic", "OpenDrawer", 1)
    audit = _audit(tmp_path, cell, success=False, timeout=True)
    audit["task_language"] = "close the fridge"

    validation = _publish_for_validation(tmp_path, cell, audit)

    assert not validation.result.canonical
    assert any("task_language" in item for item in validation.problems)


def test_old_pre_deadline_preliminary_artifact_fails_closed(tmp_path: Path):
    cell = cell_for("atomic", "OpenDrawer", 1)
    audit = _audit(tmp_path, cell)
    del audit["deadline"]
    del audit["evidence"]["deadline_contract"]

    validation = _publish_for_validation(tmp_path, cell, audit)

    assert not validation.result.canonical
    assert validation.resumable
    assert any("deadline" in item for item in validation.problems)


@pytest.mark.parametrize(
    "command",
    [
        {"action": "move_to", "target": [0, 0, 0]},
        {"action": "move_to", "xyz": [0, 0]},
        {"action": "release", "steps": True},
        {"action": "move_base", "steps": 0},
        {"action": "set_gripper", "gripper": "hold"},
        {"action": "rotate_pitch", "gripper": "hold"},
        {"action": "move_to", "xyz": [0, 0, 0], "tol": 2.0},
        {"action": "vla_act", "prompt": "open the drawer", "max_chunks": 1},
    ],
)
def test_command_schemas_fail_closed(tmp_path: Path, command):
    cell = cell_for("atomic", "OpenDrawer", 1)
    with pytest.raises(ValueError, match="command trace line 1 is invalid"):
        publish_completed_cell(
            tmp_path,
            cell,
            CellResult(Completion.COMPLETED, Outcome.SUCCESS, Integrity.VALID),
            _audit(tmp_path, cell),
            [command],
        )


def test_empty_unseen_memory_must_be_null(tmp_path: Path):
    cell = cell_for("composite_unseen", "HeatKebabSandwich", 1)
    audit = _audit(tmp_path, cell)
    audit["task_language"] = "heat kebab"
    publish_completed_cell(
        tmp_path,
        cell,
        CellResult(Completion.COMPLETED, Outcome.SUCCESS, Integrity.VALID),
        audit,
        [{"action": "release"}],
    )
    assert any(
        "empty-whitelist" in item
        for item in validate_cell(tmp_path, cell).protocol_problems
    )
    audit = _audit(tmp_path, cell)
    audit["task_language"] = "heat kebab"
    audit["task_memory"] = None
    audit["task_memory_available"] = False
    publish_completed_cell(
        tmp_path,
        cell,
        CellResult(Completion.COMPLETED, Outcome.SUCCESS, Integrity.VALID),
        audit,
        [{"action": "release"}],
    )
    assert validate_cell(tmp_path, cell).result.canonical


def _minimal_publication(cell):
    return (
        CellResult(Completion.COMPLETED, Outcome.SUCCESS, Integrity.VALID),
        {
            "task": cell.task,
            "seed": cell.seed,
            "success": True,
            "run_config_sha256": "a" * 64,
            "task_language": "open the drawer",
        },
        [{"action": "release"}],
    )


def test_publisher_securely_creates_split_and_task_directories(tmp_path: Path):
    root = tmp_path / "results"
    root.mkdir(mode=0o700)
    cell = cell_for("atomic", "OpenDrawer", 1)
    result, audit, commands = _minimal_publication(cell)

    publish_completed_cell(root, cell, result, audit, commands)

    split = root / "atomic"
    task = split / cell.task
    for directory in (split, task):
        metadata = directory.lstat()
        assert directory.is_dir() and not directory.is_symlink()
        assert metadata.st_uid == __import__("os").geteuid()
        assert metadata.st_mode & 0o022 == 0
    assert (task / f"{cell.tag}.completed.json").is_file()
    assert not list(task.glob(f".{cell.tag}.*"))


@pytest.mark.parametrize("component", ["root", "split", "task"])
def test_publisher_rejects_symlink_directory_components(tmp_path: Path, component: str):
    cell = cell_for("atomic", "OpenDrawer", 1)
    outside = tmp_path / "outside"
    outside.mkdir(mode=0o700)
    root = tmp_path / "results"
    if component == "root":
        root.symlink_to(outside, target_is_directory=True)
    else:
        root.mkdir(mode=0o700)
        split = root / "atomic"
        if component == "split":
            split.symlink_to(outside, target_is_directory=True)
        else:
            split.mkdir(mode=0o700)
            (split / cell.task).symlink_to(outside, target_is_directory=True)
    result, audit, commands = _minimal_publication(cell)

    with pytest.raises(ValueError, match="artifact"):
        publish_completed_cell(root, cell, result, audit, commands)

    assert not list(outside.rglob("*.completed.json"))


@pytest.mark.parametrize("component", ["root", "split", "task"])
def test_publisher_rejects_non_directory_components(tmp_path: Path, component: str):
    cell = cell_for("atomic", "OpenDrawer", 1)
    root = tmp_path / "results"
    if component == "root":
        root.write_text("not a directory\n", encoding="utf-8")
    else:
        root.mkdir(mode=0o700)
        split = root / "atomic"
        if component == "split":
            split.write_text("not a directory\n", encoding="utf-8")
        else:
            split.mkdir(mode=0o700)
            (split / cell.task).write_text("not a directory\n", encoding="utf-8")
    result, audit, commands = _minimal_publication(cell)

    with pytest.raises(ValueError, match="artifact"):
        publish_completed_cell(root, cell, result, audit, commands)


@pytest.mark.parametrize("component", ["root", "split", "task"])
@pytest.mark.parametrize("unsafe_mode", [0o720, 0o702])
def test_publisher_rejects_group_or_world_writable_components(
    tmp_path: Path, component: str, unsafe_mode: int
):
    cell = cell_for("atomic", "OpenDrawer", 1)
    root = tmp_path / "results"
    split = root / "atomic"
    task = split / cell.task
    task.mkdir(parents=True, mode=0o700)
    for directory in (root, split, task):
        directory.chmod(0o700)
    {"root": root, "split": split, "task": task}[component].chmod(unsafe_mode)
    result, audit, commands = _minimal_publication(cell)

    with pytest.raises(ValueError, match="writable by group or other"):
        publish_completed_cell(root, cell, result, audit, commands)


@pytest.mark.parametrize(
    ("component", "mismatched_call"), [("root", 1), ("split", 2), ("task", 3)]
)
def test_publisher_rejects_directory_not_owned_by_current_uid(
    tmp_path: Path, monkeypatch, component: str, mismatched_call: int
):
    from rpent.reproduce.robocasa import artifacts as artifact_module

    cell = cell_for("atomic", "OpenDrawer", 1)
    root = tmp_path / "results"
    (root / "atomic" / cell.task).mkdir(parents=True, mode=0o700)
    actual_uid = artifact_module.os.geteuid()
    calls = 0

    def observed_uid():
        nonlocal calls
        calls += 1
        return actual_uid + int(calls == mismatched_call)

    monkeypatch.setattr(artifact_module.os, "geteuid", observed_uid)
    result, audit, commands = _minimal_publication(cell)

    with pytest.raises(ValueError, match=f"artifact {component} must be owned"):
        publish_completed_cell(root, cell, result, audit, commands)


def test_log_archive_rejects_symlink_descendant(tmp_path: Path):
    cell = cell_for("atomic", "OpenDrawer", 1)
    root = tmp_path / "results"
    task = root / "atomic" / cell.task
    task.mkdir(parents=True, mode=0o700)
    root.chmod(0o700)
    (root / "atomic").chmod(0o700)
    outside = tmp_path / "outside-logs"
    outside.mkdir(mode=0o700)
    (task / "run_logs").symlink_to(outside, target_is_directory=True)

    with pytest.raises(ValueError, match="artifact subdirectory"):
        with secure_artifact_subdirectory(root, cell, "run_logs", cell.tag, "run-1"):
            pass


def test_validator_rejects_symlink_task_without_following_it(tmp_path: Path):
    cell = cell_for("atomic", "OpenDrawer", 1)
    root = tmp_path / "results"
    _run_digest(root)
    split = root / "atomic"
    split.mkdir(mode=0o700)
    outside = tmp_path / "outside-result"
    outside.mkdir(mode=0o700)
    (outside / f"{cell.tag}.completed.json").write_text("{}\n", encoding="utf-8")
    (split / cell.task).symlink_to(outside, target_is_directory=True)

    validation = validate_cell(root, cell)

    assert not validation.result.canonical
    assert any("unsafe artifact directory" in item for item in validation.problems)


@pytest.mark.parametrize(
    "tamper",
    [
        "runtime_scripts",
        "memory_source",
        "navview",
        "checkpoint",
        "preliminary",
        "planner_auth_mode",
        "planner_provider",
        "planner_endpoint_identity",
    ],
)
def test_cell_audit_identity_must_match_root_manifest(tmp_path: Path, tamper: str):
    cell = cell_for("atomic", "OpenDrawer", 1)
    audit = _audit(tmp_path, cell)
    if tamper == "preliminary":
        audit["preliminary"] = False
    elif tamper == "checkpoint":
        audit["checkpoint"]["fingerprint"] = "b" * 64
    elif tamper.startswith("planner_"):
        audit[tamper] = f"tampered-{audit[tamper]}"
    else:
        audit[tamper] = {**audit[tamper], "tampered": True}
    publish_completed_cell(
        tmp_path,
        cell,
        CellResult(Completion.COMPLETED, Outcome.SUCCESS, Integrity.VALID),
        audit,
        [{"action": "release"}],
    )

    validation = validate_cell(tmp_path, cell)

    assert validation.provenance_problems
    assert not validation.result.canonical
