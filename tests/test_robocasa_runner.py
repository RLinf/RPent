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
import threading
import time
from types import SimpleNamespace

import pytest

from rpent.reproduce.robocasa import runner
from rpent.reproduce.robocasa.executor import (
    EXIT_CANONICAL,
    EXIT_RETRYABLE_INFRA,
    Execution,
)
from rpent.reproduce.robocasa.protocol import Split, cell_for


def _validation(resumable: bool):
    return SimpleNamespace(
        resumable=resumable,
        provenance_problems=(),
        result=SimpleNamespace(
            canonical=not resumable,
            completion=SimpleNamespace(
                value="incomplete" if resumable else "completed"
            ),
        ),
    )


def _manifest(_config):
    return {"run_config_sha256": "a" * 64}


def test_named_selections_cover_the_frozen_matrix():
    full = runner.select_cells("full")
    assert len(full) == 340
    assert {cell.split for cell in full[:180]} == {Split.ATOMIC}
    assert {cell.split for cell in full[180:260]} == {Split.COMPOSITE_SEEN}
    assert {cell.split for cell in full[260:]} == {Split.COMPOSITE_UNSEEN}
    assert len(runner.select_cells("atomic")) == 180
    assert len(runner.select_cells("composite_seen")) == 80
    assert len(runner.select_cells("composite_unseen")) == 80
    assert len(runner.select_cells("smoke-v1")) == 4
    assert {cell.split for cell in runner.SMOKE_CELLS} == set(Split)
    with pytest.raises(ValueError, match="selection"):
        runner.select_cells("unknown")


def test_runner_retries_infrastructure_and_resumes_canonical_cells(
    tmp_path, monkeypatch
):
    cell = cell_for("atomic", "OpenDrawer", 1)
    complete: set = set()
    attempts = 0

    monkeypatch.setattr(
        runner,
        "validate_cell",
        lambda _root, candidate, **_kwargs: _validation(candidate not in complete),
    )
    monkeypatch.setattr(runner, "ensure_execution_run_manifest", _manifest)

    def execute(_config, candidate, _gpu):
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            return Execution(
                candidate,
                EXIT_RETRYABLE_INFRA,
                "planner_failed",
                False,
                tmp_path / f"attempt-{attempts}",
            )
        complete.add(candidate)
        return Execution(
            candidate,
            EXIT_CANONICAL,
            "planner_completed",
            True,
            tmp_path / "attempt-3",
        )

    config = SimpleNamespace(results_root=tmp_path / "results")
    report = runner.run_cells(
        config,
        [cell],
        gpus=(0,),
        max_attempts=3,
        retry_backoff_seconds=0,
        executor=execute,
    )
    assert report["complete"] is True
    assert report["gpus"] == [0]
    assert report["started_at"].endswith("Z")
    assert report["finished_at"].endswith("Z")
    assert report["elapsed_seconds"] >= 0
    assert report["canonical"] == 1
    assert len(report["attempts"]) == 3

    resumed = runner.run_cells(
        config,
        [cell],
        gpus=(0,),
        executor=lambda *_args: pytest.fail("canonical cell was rerun"),
    )
    assert resumed["complete"] is True
    assert resumed["skipped_canonical"] == 1


def test_runner_accepts_canonical_timeout_without_fatal_stop(tmp_path, monkeypatch):
    cell = cell_for("atomic", "OpenDrawer", 1)
    complete: set = set()
    monkeypatch.setattr(
        runner,
        "validate_cell",
        lambda _root, candidate, **_kwargs: _validation(candidate not in complete),
    )
    monkeypatch.setattr(runner, "ensure_execution_run_manifest", _manifest)

    def execute(_config, candidate, _gpu):
        complete.add(candidate)
        return Execution(
            candidate,
            EXIT_CANONICAL,
            "planner_timeout",
            True,
            tmp_path / "timeout-attempt",
        )

    config = SimpleNamespace(results_root=tmp_path / "results")
    report = runner.run_cells(
        config,
        [cell],
        gpus=(12000,),
        executor=execute,
    )

    assert report["complete"] is True
    assert report["fatal"] is None
    assert not (config.results_root / "_FATAL_STOP.json").exists()


def test_runner_persists_executor_exceptions_without_leaking_message(
    tmp_path, monkeypatch
):
    cell = cell_for("atomic", "OpenDrawer", 1)
    monkeypatch.setattr(
        runner,
        "validate_cell",
        lambda _root, _cell, **_kwargs: _validation(True),
    )
    monkeypatch.setattr(runner, "ensure_execution_run_manifest", _manifest)
    config = SimpleNamespace(results_root=tmp_path / "results")

    def explode(_config, _cell, _gpu):
        raise RuntimeError("sensitive upstream response")

    report = runner.run_cells(
        config,
        [cell],
        gpus=(0,),
        retry_backoff_seconds=0,
        executor=explode,
    )
    fatal_path = config.results_root / "_FATAL_STOP.json"
    fatal = json.loads(fatal_path.read_text(encoding="utf-8"))
    assert report["complete"] is False
    assert fatal["termination_cause"] == "executor_exception"
    assert fatal["message"] == "RuntimeError"
    assert "sensitive" not in fatal_path.read_text(encoding="utf-8")

    with pytest.raises(RuntimeError, match="persistent fatal stop"):
        runner.run_cells(config, [cell], gpus=(0,), executor=explode)


@pytest.mark.parametrize("gpus", [(), (0, 0), (-1,)])
def test_runner_rejects_invalid_gpu_sets(tmp_path, gpus):
    config = SimpleNamespace(results_root=tmp_path)
    with pytest.raises(ValueError, match="gpus"):
        runner.run_cells(config, [], gpus=gpus)


def test_runner_uses_both_gpus_with_one_serial_worker_each(tmp_path, monkeypatch):
    cells = tuple(runner.select_cells("atomic")[:4])
    complete: set = set()
    active = {0: 0, 1: 0}
    peak = {0: 0, 1: 0}
    seen: set[int] = set()
    calls = 0
    lock = threading.Lock()
    first_wave = threading.Barrier(2, timeout=5)

    monkeypatch.setattr(
        runner,
        "validate_cell",
        lambda _root, candidate, **_kwargs: _validation(candidate not in complete),
    )
    monkeypatch.setattr(runner, "ensure_execution_run_manifest", _manifest)

    def execute(_config, candidate, gpu):
        nonlocal calls
        with lock:
            calls += 1
            wave = calls
            active[gpu] += 1
            peak[gpu] = max(peak[gpu], active[gpu])
            seen.add(gpu)
        try:
            if wave <= 2:
                first_wave.wait()
            time.sleep(0.01)
            with lock:
                complete.add(candidate)
            return Execution(
                candidate,
                EXIT_CANONICAL,
                "planner_completed",
                True,
                tmp_path / f"gpu-{gpu}-{candidate.tag}",
            )
        finally:
            with lock:
                active[gpu] -= 1

    report = runner.run_cells(
        SimpleNamespace(results_root=tmp_path / "results"),
        cells,
        gpus=(0, 1),
        executor=execute,
    )

    assert report["complete"] is True
    assert seen == {0, 1}
    assert peak == {0: 1, 1: 1}


def test_runner_rejects_concurrent_use_of_same_results_root(tmp_path):
    results_root = tmp_path / "results"
    config = SimpleNamespace(results_root=results_root)

    with runner._runner_locks(results_root, (12001,)):
        with pytest.raises(RuntimeError, match="results root .* already locked"):
            runner.run_cells(config, [], gpus=(12002,))


def test_runner_rejects_concurrent_use_of_same_gpu_across_roots(tmp_path):
    first_root = tmp_path / "first-results"
    second = SimpleNamespace(results_root=tmp_path / "second-results")

    with runner._runner_locks(first_root, (12003,)):
        with pytest.raises(RuntimeError, match="RoboCasa GPU 12003.*already locked"):
            runner.run_cells(second, [], gpus=(12003,))


@pytest.mark.parametrize("conflict", ["results", "gpu"])
def test_lock_conflict_rejects_before_preflight_can_write(tmp_path, conflict):
    first_root = tmp_path / "first-results"
    second_root = first_root if conflict == "results" else tmp_path / "second-results"
    locked_gpu = 12004 if conflict == "results" else 12005
    requested_gpu = 12006 if conflict == "results" else locked_gpu
    config = SimpleNamespace(results_root=second_root)
    called = False

    def preflight():
        nonlocal called
        called = True
        directory = second_root / "_preflight"
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "should-not-exist.json").write_text("{}\n", encoding="utf-8")
        return {"ok": True}

    with runner._runner_locks(first_root, (locked_gpu,)):
        with pytest.raises(RuntimeError, match="already locked"):
            runner.run_cells(
                config,
                [],
                gpus=(requested_gpu,),
                preflight=preflight,
            )

    assert called is False
    assert not (second_root / "_preflight/should-not-exist.json").exists()


def test_failed_preflight_releases_runner_locks(tmp_path):
    results_root = tmp_path / "results"
    config = SimpleNamespace(results_root=results_root)

    with pytest.raises(runner.PreflightFailed) as error:
        runner.run_cells(
            config,
            [],
            gpus=(12007,),
            preflight=lambda: {"ok": False, "problems": ["fixture"]},
        )

    assert error.value.report["problems"] == ["fixture"]
    with runner._runner_locks(results_root, (12007,)):
        pass


def test_standalone_preflight_cannot_mutate_a_locked_results_root(tmp_path):
    results_root = tmp_path / "results"
    called = False

    def preflight():
        nonlocal called
        called = True
        return {"ok": True}

    with runner._runner_locks(results_root, (12008,)):
        with pytest.raises(RuntimeError, match="results root .* already locked"):
            runner.run_locked_preflight(results_root, preflight)

    assert called is False
