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

import json
import os
from pathlib import Path

import pytest

from rpent.evaluation import (
    RunFinalizationContext,
    verdict_fields,
    write_json_atomic,
)


def _context(**overrides):
    values = {
        "output_dir": Path("/run"),
        "robot_name": "robot",
        "task_desc": {},
        "environment_success": None,
        "agent_error": None,
        "elapsed_s": 1.0,
        "planner": "api",
        "model": None,
        "reasoning_effort": "high",
        "max_turns": 10,
        "planner_timeout_s": None,
        "finish_result": None,
        "stats": {},
    }
    values.update(overrides)
    return RunFinalizationContext(**values)


@pytest.mark.parametrize(
    ("success", "agent_error", "finish_result", "expected"),
    [
        (True, None, None, (True, False, None, "solved")),
        (True, "RuntimeError: late failure", None, (True, False, None, "solved")),
        (False, None, None, (False, False, "task_failed", "unfinished")),
        (False, None, {"status": "failure"}, (False, False, "task_failed", "finished")),
        (
            None,
            "Codex SDK timed out",
            None,
            (None, True, "agent_error", "planner_error"),
        ),
        (None, None, None, (None, True, "no_verdict", "unfinished")),
        (
            None,
            None,
            {"operator_aborted": True},
            (None, True, "no_verdict", "operator_aborted"),
        ),
    ],
)
def test_the_verdict_derives_from_the_environment_outcome(
    success, agent_error, finish_result, expected
):
    context = _context(
        environment_success=success,
        agent_error=agent_error,
        finish_result=finish_result,
        truth_source="environment",
    )

    assert (
        context.accepted,
        context.retryable,
        context.failure_category,
        context.stop_reason,
    ) == expected
    assert verdict_fields(context) == {
        "accepted": context.accepted,
        "retryable": context.retryable,
        "failure_category": context.failure_category,
        "stop_reason": context.stop_reason,
        "truth_source": "environment",
    }


def test_an_unknown_truth_source_is_rejected():
    with pytest.raises(ValueError, match="truth_source must be one of"):
        _context(truth_source="simulator")


def test_the_exit_status_reports_no_verdict_separately(tmp_path: Path) -> None:
    from rpent.cli.main import _exit_code

    assert [_exit_code(accepted) for accepted in (True, False, None)] == [0, 1, 75]


def test_write_json_atomic_replaces_complete_record(tmp_path: Path) -> None:
    destination = tmp_path / "result.json"
    destination.write_text('{"status": "old"}\n', encoding="utf-8")

    result = write_json_atomic(destination, {"status": "complete", "success": True})

    assert result == destination
    assert json.loads(destination.read_text(encoding="utf-8")) == {
        "status": "complete",
        "success": True,
    }
    assert not (tmp_path / ".result.json.tmp").exists()


def test_write_json_atomic_cleans_temporary_file_on_replace_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    destination = tmp_path / "result.json"
    destination.write_text('{"status": "old"}\n', encoding="utf-8")

    def fail_replace(source: Path, target: Path) -> None:
        del source, target
        raise OSError("replace failed")

    monkeypatch.setattr(os, "replace", fail_replace)

    with pytest.raises(OSError, match="replace failed"):
        write_json_atomic(destination, {"status": "new"})

    assert json.loads(destination.read_text(encoding="utf-8")) == {"status": "old"}
    assert not (tmp_path / ".result.json.tmp").exists()
