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
from pathlib import Path

import pytest

from rpent.evaluation import TrialRecorder


def recorder(root: Path, **options) -> TrialRecorder:
    return TrialRecorder(
        root,
        trial_id="robotwin/tidy_table/s0",
        method="agent",
        task="tidy_table",
        clock=lambda: 120.0,
        **options,
    )


def test_first_success_stop_and_drain_are_distinct(tmp_path):
    trial = recorder(tmp_path, budget={"wall_s": 10})
    trial.start(at=100)
    trial.record_native_verdict(True, source="native", observed_at=108.5)
    trial.stop("task_complete", at=109)
    trial.record_native_verdict(False, source="native", observed_at=109.5)
    trial.quiesce(at=111)
    record = json.loads(trial.finish().read_text())
    assert record["status"] == "success"
    assert record["native_verdict"] is True
    assert record["last_native_observation"]["value"] is False
    assert record["timing"] == {
        "run_s": 11,
        "success_s": 8.5,
        "stop_s": 9,
        "tail_s": 2,
        "observed_s": 11,
    }


def test_late_success_cannot_rescue_a_timeout(tmp_path):
    trial = recorder(tmp_path, budget={"wall_s": 10})
    trial.start(at=100)
    trial.record_native_verdict(True, source="native", observed_at=111)
    trial.stop("timeout", at=112)
    record = json.loads(trial.finish().read_text())
    assert record["status"] == "failure"
    assert record["native_verdict"] is None
    assert record["last_native_observation"]["observed_s"] == 11
    assert record["timing"] == {
        "run_s": None,
        "success_s": None,
        "stop_s": 12,
        "tail_s": None,
        "observed_s": 20,
    }


@pytest.mark.parametrize(
    ("evidence", "frozen", "extra", "status", "success_s"),
    (
        (True, True, {}, "success", None),
        (True, False, {}, "unresolved", None),
        (False, True, {}, "failure", None),
        (None, True, {}, "unresolved", None),
        (None, True, {"failure": True}, "failure", None),
        (None, True, {"infra_invalid": True}, "infra_invalid", None),
    ),
)
def test_terminal_evidence_and_unknowns(
    tmp_path, evidence, frozen, extra, status, success_s
):
    trial = recorder(tmp_path)
    trial.start(at=100)
    trial.stop("end", at=110)
    trial.record_native_verdict(evidence, source="native", confirmed_only=frozen)
    record = json.loads(trial.finish(**extra).read_text())
    assert record["status"] == status
    assert record["timing"]["success_s"] == success_s


def test_attempt_paths_preserve_conditions_and_infrastructure_retries(tmp_path):
    failed = recorder(tmp_path)
    failed.stop("simulator_boot_error")
    failed.quiesce()
    failed_path = failed.finish(infra_invalid=True, error_type="ConnectionError")
    completed = recorder(tmp_path)
    completed.start(at=120)
    completed.record_native_verdict(False, source="native")
    completed.stop("done")
    completed.quiesce()
    completed_path = completed.finish()
    assert failed_path != completed_path
    assert completed.finish() == completed_path
    first, second = (json.loads(p.read_text()) for p in (failed_path, completed_path))
    assert first["status"] == "infra_invalid"
    assert first["error_type"] == "ConnectionError"
    assert all(value is None for value in first["timing"].values())
    assert first["trial_id"] == second["trial_id"]
    assert first["run_id"] != second["run_id"]
    assert second["status"] == "failure"
    assert second["timing"]["run_s"] == 0
    assert completed_path == (
        tmp_path / "metrics" / completed.run_id / "episode_metrics.json"
    )


@pytest.mark.parametrize("wall_s", [-1, True, float("nan"), float("inf"), "10"])
def test_invalid_wall_budget_does_not_silently_affect_success(tmp_path, wall_s):
    with pytest.raises(ValueError, match="wall_s"):
        recorder(tmp_path, budget={"wall_s": wall_s})


def test_reset_and_model_endpoint_bind_metadata_before_timer_starts(tmp_path):
    trial = recorder(tmp_path)
    trial.set_instruction_version("sha256:123")
    trial.set_checkpoint("worker/fastwam")
    trial.start()
    with pytest.raises(ValueError, match="before trial start"):
        trial.set_instruction_version("sha256:456")
    record = json.loads(trial.finish().read_text())
    assert record["instruction_version"] == "sha256:123"
    assert record["checkpoint"] == "worker/fastwam"


def test_shared_entry_points_require_no_robot_or_planner_extras():
    # -S runs with only the standard library and the current checkout.
    import subprocess
    import sys

    subprocess.run(
        [sys.executable, "-S", "-c", "from rpent.evaluation import TrialRecorder"],
        check=True,
    )


def test_operator_assisted_success_is_retained_as_raw_evidence(tmp_path):
    trial = recorder(tmp_path)
    trial.start()
    trial.record_native_verdict(True, source="native")
    trial.stop("operator")
    trial.quiesce()
    record = json.loads(trial.finish(manual_intervention=True).read_text())
    assert record["status"] == "unresolved"
    assert record["manual_intervention"] is True
    assert record["native_verdict"] is True
    assert record["timing"]["success_s"] is None
