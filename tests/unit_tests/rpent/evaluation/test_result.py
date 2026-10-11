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

from rpent.evaluation import write_json_atomic


def test_operator_outcomes_are_unique_across_dashboard_runs(tmp_path: Path) -> None:
    from robots.dual_franka.robot_spec import DUAL_FRANKA_DASHBOARD_SPEC
    from rpent.dashboard.state import DashboardState
    from rpent.evaluation.result import record_operator_outcome

    paths = []
    for run in ("dashboard_a", "dashboard_b"):
        state = DashboardState(
            output_dir=tmp_path / run, dashboard_spec=DUAL_FRANKA_DASHBOARD_SPEC
        )
        state.shared_services_ready()
        state.request_task({"task_id": 3})
        task = state.wait_for_task(0)
        kwargs = {
            "state_output_dir": task.output_dir / "sessions" / "session_001",
            "session_number": 1,
            "verdict": {"operator_verdict": "failure"},
            "messages": [],
            "planner_error": None,
        }
        evidence = record_operator_outcome(**kwargs)
        before = evidence.read_bytes()
        with pytest.raises(FileExistsError):
            record_operator_outcome(**kwargs)
        assert evidence.read_bytes() == before
        paths.append(evidence)
    assert paths[0] != paths[1] and all(path.is_file() for path in paths)


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
