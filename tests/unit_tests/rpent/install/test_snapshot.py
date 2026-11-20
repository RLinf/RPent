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

"""The run manifest records the run and the installation that produced it."""

import json

import pytest

from rpent.install.fingerprint import fingerprint_sha256, write_fingerprint
from rpent.install.snapshot import (
    finish_run_manifest,
    installation_snapshot,
    start_run_manifest,
)


def _fingerprint():
    return {
        "schema": 6,
        "targets": ["infra-minimal"],
        "python": "3.11.14",
        "platform": "linux-x86_64",
        "components": {"infra-minimal": {"manifest_sha256": "digest"}},
        "source_commit": "abc123",
        "source_dirty": False,
    }


def test_snapshot_of_a_venv_without_a_fingerprint(tmp_path):
    venv = tmp_path / "venv"
    venv.mkdir()

    snapshot = installation_snapshot(venv)

    assert snapshot["targets"] is None
    assert snapshot["install_sha256"] is None
    assert snapshot["venv"] == str(venv.resolve())
    assert snapshot["components"] == []


def test_snapshot_identifies_the_installed_targets(tmp_path):
    venv = tmp_path / "venv"
    venv.mkdir()
    fingerprint = _fingerprint()
    write_fingerprint(venv, fingerprint)

    snapshot = installation_snapshot(venv)

    assert snapshot["targets"] == ["infra-minimal"]
    assert snapshot["install_sha256"] == fingerprint_sha256(fingerprint)
    assert snapshot["components"] == ["infra-minimal"]
    assert snapshot["source_commit"] == "abc123"


def test_start_records_the_run_and_claims_the_directory(tmp_path):
    venv = tmp_path / "venv"
    venv.mkdir()
    write_fingerprint(venv, _fingerprint())

    path = start_run_manifest(
        tmp_path,
        robot="libero",
        planner="api",
        model=None,
        task={"suite": "libero_10_task", "task": 0},
        venv=venv,
    )
    record = json.loads(path.read_text(encoding="utf-8"))

    assert record["schema"] == 1
    assert record["robot"] == "libero"
    assert record["task"] == {"suite": "libero_10_task", "task": 0}
    assert record["environment"]["targets"] == ["infra-minimal"]
    assert record["artifacts"] == {}

    # One directory holds one run: a second start must not overwrite it.
    with pytest.raises(FileExistsError, match="already records a run"):
        start_run_manifest(
            tmp_path, robot="libero", planner="api", model=None, task={}, venv=venv
        )


def test_finish_records_artifacts_and_the_verdict(tmp_path):
    start_run_manifest(tmp_path, robot="libero", planner="api", model="m", task={})

    path = finish_run_manifest(
        tmp_path,
        artifacts={"transcript": tmp_path / "transcript.json", "result": None},
        verdict={"accepted": True, "truth_source": "environment"},
        error=None,
    )
    record = json.loads(path.read_text(encoding="utf-8"))

    assert record["artifacts"] == {"transcript": str(tmp_path / "transcript.json")}
    assert (record["accepted"], record["truth_source"]) == (True, "environment")
    assert record["error"] is None
    assert record["artifact_error"] is None
    assert record["started_at"].endswith("Z")
    assert record["finished_at"].endswith("Z")


def test_finish_records_an_artifact_failure_apart_from_the_agent_error(tmp_path):
    start_run_manifest(tmp_path, robot="libero", planner="api", model="m", task={})

    path = finish_run_manifest(
        tmp_path,
        artifacts={"transcript": None},
        verdict={"accepted": True, "truth_source": "environment"},
        error=None,
        artifact_error="result finalization failed: OSError: disk full",
    )
    record = json.loads(path.read_text(encoding="utf-8"))

    # The verdict and `error` describe the run; the failure to write the run's
    # own artifacts is reported on its own so neither field contradicts the other.
    assert record["accepted"] is True
    assert record["error"] is None
    assert record["artifact_error"] == "result finalization failed: OSError: disk full"
