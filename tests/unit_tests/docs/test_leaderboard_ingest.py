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

"""Unit tests for the standalone leaderboard ingestion and validation module."""

from __future__ import annotations

import ast
import copy
import difflib
import json
import runpy
import sys
from pathlib import Path
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
INGEST_PATH = REPO_ROOT / "docs/_ext/leaderboard_ingest.py"
INGEST = runpy.run_path(str(INGEST_PATH))
ValidationError = INGEST["LeaderboardValidationError"]


def _write_json(path: Path, payload: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _bilingual(text: str) -> dict[str, str]:
    return {"en": text, "zh": text}


def _make_result(
    view_id: str,
    configuration_id: str,
    rate: Any,
    source_id: str = "src-1",
    **extra: Any,
) -> dict[str, Any]:
    return {
        "id": f"{view_id}--{configuration_id}",
        "view_id": view_id,
        "configuration_id": configuration_id,
        "rate": rate,
        "source_ids": [source_id],
        **extra,
    }


def _make_cost(
    benchmark_id: str,
    configuration_id: str,
    mean_elapsed_seconds: Any,
    total_output_tokens: Any,
    **extra: Any,
) -> dict[str, Any]:
    return {
        "id": f"{benchmark_id}--{configuration_id}",
        "benchmark_id": benchmark_id,
        "configuration_id": configuration_id,
        "mean_elapsed_seconds": mean_elapsed_seconds,
        "total_output_tokens": total_output_tokens,
        **extra,
    }


def _make_dataset() -> dict[str, Any]:
    robotwin_note = {
        "id": "note-1",
        "configuration_ids": ["cfg-rpent"],
        "text": _bilingual("Note"),
    }
    robocasa_views = [
        {"id": "robocasa", "label": _bilingual("Overall")},
        {"id": "robocasa-atomic-seen", "label": _bilingual("Atomic")},
    ]
    robotwin_benchmark = {
        "id": "robotwin",
        "name": "RoboTwin",
        "heading": _bilingual("C2R"),
        "default_view": "robotwin",
        "views": [{"id": "robotwin", "label": _bilingual("C2R")}],
        "notes": [robotwin_note],
    }
    robocasa_benchmark = {
        "id": "robocasa",
        "name": "RoboCasa",
        "heading": _bilingual("Target50"),
        "default_view": "robocasa",
        "views": robocasa_views,
    }
    return {
        "schema_version": 4,
        "footer_notes": [{"id": "footer-1", "text": _bilingual("Footer")}],
        "configurations": [
            {"id": "cfg-rpent", "kind": "rpent", "backend": "API", "model": "Model-1"},
            {"id": "cfg-ext", "kind": "external", "model": "External-1"},
        ],
        "benchmarks": [robotwin_benchmark, robocasa_benchmark],
        "results": [
            _make_result("robotwin", "cfg-rpent", "50.00", successes=1, episodes=2)
        ],
        "cost_results": {
            "source_ids": ["src-1"],
            "rows": [_make_cost("robotwin", "cfg-rpent", 12.5, 1000)],
        },
        "sources": [{"id": "src-1", "note": "English-only source note."}],
        "palette": {
            "light": {"best": "#1", "rpent": "#2", "reference": "#3"},
            "dark": {"best": "#4", "rpent": "#5", "reference": "#6"},
        },
    }


def _make_transcript(
    task_name: str,
    seed: int,
    *,
    success: Any = True,
    error: Any = None,
    include_error: bool = True,
    elapsed_s: float = 10.0,
    output_tokens: int = 20,
    model: Any = "openai-chat:test-model",
    robotwin_seeds: bool = False,
) -> dict[str, Any]:
    seed_fields = (
        {"requested_seed": seed, "initial_native_seed": seed}
        if robotwin_seeds
        else {"seed": seed}
    )
    record: dict[str, Any] = {
        "model": model,
        "task_name": task_name,
        "task_config": "demo_randomized",
        **seed_fields,
        "elapsed_s": elapsed_s,
        "environment_success": success,
        "stats": {
            "total_input_tokens": 100,
            "total_output_tokens": output_tokens,
            "total_cached_input_tokens": 40,
        },
    }
    if include_error:
        record["error"] = error
    return record


def _make_split_summary(successes: int) -> dict[str, int]:
    return {"successes": successes, "valid_cells": 1, "expected_cells": 1}


def test_module_uses_only_stdlib():
    tree = ast.parse(INGEST_PATH.read_text(encoding="utf-8"))
    imported_modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_modules.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported_modules.add(node.module.split(".")[0])
    assert imported_modules <= sys.stdlib_module_names


def test_real_results_roundtrip_and_compact_upsert():
    raw_text = INGEST["DATA_PATH"].read_text(encoding="utf-8")
    dataset = INGEST["load_data"](INGEST["DATA_PATH"])
    assert INGEST["dump_data"](dataset) == raw_text

    new_config = {
        "id": "sim-api",
        "kind": "rpent",
        "backend": "API",
        "model": "Sim",
        "insert_after": "deepseek-v4-1-flash-api-xhigh",
    }
    new_result = _make_result(
        "robotwin",
        "sim-api",
        "60.00",
        "src-sim",
        successes=150,
        episodes=250,
    )
    new_cost = _make_cost(
        "robotwin",
        "sim-api",
        123.45,
        500000,
        source_ids=["src-sim"],
    )
    payload = {
        "sources": [{"id": "src-sim", "note": "Synthetic API run."}],
        "configurations": [new_config],
        "results": [new_result],
        "cost_results": [new_cost],
    }
    after_text = INGEST["dump_data"](INGEST["upsert_data"](dataset, payload))
    diff_lines = list(
        difflib.unified_diff(
            raw_text.splitlines(),
            after_text.splitlines(),
            lineterm="",
        )
    )
    removed = [
        line
        for line in diff_lines
        if line.startswith("-") and not line.startswith("---")
    ]
    added = [
        line
        for line in diff_lines
        if line.startswith("+") and not line.startswith("+++")
    ]
    assert len(removed) <= 3
    assert len(added) < 25


def test_validate_rules_exemptions_and_cost_guards():
    base = _make_dataset()
    INGEST["validate_data"](base)

    nullable = copy.deepcopy(base)
    nullable["results"][0]["rate"] = None
    nullable["cost_results"]["rows"][0]["mean_elapsed_seconds"] = None
    INGEST["validate_data"](nullable)

    for exempt_patch in (
        {"successes_estimated": True},
        {"status": "derived"},
        {"id": "robocasa--cfg-rpent", "view_id": "robocasa"},
    ):
        exempt = copy.deepcopy(base)
        exempt["results"][0].update(
            successes=1,
            episodes=3,
            rate="99.00",
            **exempt_patch,
        )
        INGEST["validate_data"](exempt)

    def _clear_labels(config: dict[str, Any], config_id: str, kind: str) -> None:
        config.clear()
        config.update(id=config_id, kind=kind)

    invalid_rates = ("97", "97.00\n", "100.01", 50.0, True)
    invalid_numbers = (True, "12", -1, float("nan"))
    mutations = [
        lambda data: data.update(schema_version=True),
        lambda data: data["benchmarks"][0].pop("name"),
        lambda data: data["results"][0].pop("rate"),
        lambda data: data["cost_results"]["rows"][0].pop("mean_elapsed_seconds"),
        lambda data: data["benchmarks"][0].update(default_view=[]),
        lambda data: data["results"][0].update(view_id=[]),
        lambda data: data["results"][0].update(derived_from=[[]]),
        lambda data: data["results"][0].update(derived_from=None),
        lambda data: data["cost_results"]["rows"][0].update(benchmark_id=[]),
        lambda data: data["palette"]["dark"].pop("reference"),
        lambda data: _clear_labels(data["configurations"][0], "cfg-rpent", "rpent"),
        lambda data: _clear_labels(data["configurations"][1], "cfg-ext", "external"),
        lambda data: data["benchmarks"][0]["heading"].update(zh=" "),
        lambda data: data["benchmarks"][0].update(summary=123),
        lambda data: data["results"][0].update(evaluation_note={"en": "ok", "zh": ""}),
        lambda data: data["results"][0].update(evaluation_note=""),
        lambda data: data["results"][0].update(successes=1, episodes=3, rate="33.34"),
        *(
            lambda data, bad_rate=bad_rate: data["results"][0].update(rate=bad_rate)
            for bad_rate in invalid_rates
        ),
        *(
            lambda data, bad_num=bad_num: data["cost_results"]["rows"][0].update(
                total_output_tokens=bad_num
            )
            for bad_num in invalid_numbers
        ),
    ]
    for mutate in mutations:
        broken = copy.deepcopy(base)
        mutate(broken)
        with pytest.raises(ValidationError):
            INGEST["validate_data"](broken)


def test_robotwin_termination_classification_parity():
    classify = INGEST["_classify_robotwin_termination"]
    assert classify(None, True) == "completed"
    assert classify("ModelAPIError: Request timed out.", True) == "completed"
    assert classify(None, False) == "completed"
    assert classify("  ", False) == "completed"
    assert classify("planner timed out after 4800s", False) == "planner_timeout"
    assert classify("agent sdk timed out", False) == "planner_timeout"
    assert classify("codex sdk timed out", False) == "planner_timeout"
    assert classify("episode timed out", False) == "planner_timeout"
    assert (
        classify("planner response truncated (length): limit exceeded", False)
        == "planner_response_truncated"
    )
    assert (
        classify("ModelAPIError: Request timed out.", False) == "infrastructure_error"
    )
    assert classify("ModelHTTPError: status_code: 400", False) == "infrastructure_error"


def test_robotwin_append_model_consistency_and_count_as_failure(tmp_path: Path):
    manifest_data = {
        "task_config": "demo_randomized",
        "tasks": {"task_a": [{"seed": 1}, {"seed": 2}], "task_b": [{"seed": 3}]},
    }
    manifest_file = _write_json(tmp_path / "manifest.json", manifest_data)
    run_dir = tmp_path / "robotwin_run"
    run_dir.mkdir()

    appended_transcript = run_dir / "transcript_task_a_s1.json"
    attempt_1 = json.dumps(
        _make_transcript(
            "task_a",
            1,
            success=False,
            error="HTTP 500",
            robotwin_seeds=True,
        )
    )
    attempt_2 = json.dumps(
        _make_transcript(
            "task_a",
            1,
            include_error=False,
            elapsed_s=10.0,
            output_tokens=100,
            robotwin_seeds=True,
        )
    )
    appended_transcript.write_text(attempt_1 + attempt_2, encoding="utf-8")

    final_record, in_file_retries = INGEST["load_transcript_final_record"](
        appended_transcript
    )
    assert final_record["environment_success"] is True
    assert in_file_retries == 1

    _write_json(
        run_dir / "transcript_task_a_s2.json",
        _make_transcript(
            "task_a",
            2,
            success=False,
            error="planner response truncated",
            elapsed_s=20.0,
            output_tokens=200,
            robotwin_seeds=True,
        ),
    )
    _write_json(
        run_dir / "transcript_task_b_s3.json",
        _make_transcript(
            "task_b",
            3,
            success=False,
            error="planner timed out after 120s",
            elapsed_s=30.0,
            output_tokens=300,
            robotwin_seeds=True,
        ),
    )

    aggregated = INGEST["aggregate_robotwin"](
        run_dir,
        manifest_path=manifest_file,
        expected_model="openai-chat:test-model",
        configuration_id="cfg-rpent",
        source_ids=["src-1"],
        protocol_id="proto-1",
    )
    result_row = aggregated["results"][0]
    cost_row = aggregated["cost_results"][0]
    token_summary = aggregated["token_summary"]

    assert result_row["rate"] == "33.33"
    assert result_row["task_failures"] == 1
    assert result_row["episode_timeouts"] == 1
    assert "mean_total_tokens" not in cost_row
    assert cost_row["mean_elapsed_seconds"] == 20.0
    assert cost_row["total_output_tokens"] == 600
    assert token_summary["model"] == "openai-chat:test-model"
    assert token_summary["cost_episodes"] == 3
    assert token_summary["excluded_in_file_attempts"] == 1
    assert token_summary["counted_as_failure_cells"] == []

    output_file = tmp_path / "agg_rt.json"
    base_cli = [
        "aggregate",
        "--benchmark",
        "robotwin",
        "--input-dir",
        str(run_dir),
        "--manifest",
        str(manifest_file),
    ]
    assert (
        INGEST["main"](
            [
                *base_cli,
                "--expected-model",
                "openai-chat:test-model",
                "--output",
                str(output_file),
            ]
        )
        == 0
    )
    assert (
        json.loads(output_file.read_text(encoding="utf-8"))["results"][0]["rate"]
        == "33.33"
    )
    assert INGEST["main"]([*base_cli, "--planner", "codex"]) == 1
    assert (
        INGEST["main"]([*base_cli, "--expected-model", "openai-chat:other-model"]) == 1
    )

    # Test --count-as-failure after removing one exhausted cell's transcript.
    (run_dir / "transcript_task_b_s3.json").unlink()
    exhausted_out = tmp_path / "agg_rt_exhausted.json"
    assert (
        INGEST["main"](
            [
                *base_cli,
                "--count-as-failure",
                "task_b_s3",
                "--output",
                str(exhausted_out),
            ]
        )
        == 0
    )
    exhausted = json.loads(exhausted_out.read_text(encoding="utf-8"))
    assert exhausted["results"][0]["episodes"] == 3
    assert exhausted["results"][0]["successes"] == 1
    assert exhausted["results"][0]["task_failures"] == 2
    assert exhausted["results"][0]["episode_timeouts"] == 0
    assert exhausted["results"][0]["rate"] == "33.33"
    assert exhausted["cost_results"][0]["mean_elapsed_seconds"] == 15.0
    assert exhausted["cost_results"][0]["total_output_tokens"] == 300
    assert exhausted["token_summary"]["cost_episodes"] == 2
    assert exhausted["token_summary"]["counted_as_failure_cells"] == ["task_b_s3"]

    for bad_spec in (
        ["badspec"],
        ["unknown_s1"],
        ["task_b_s3", "task_b_s3"],
        ["task_a_s1"],
    ):
        with pytest.raises(ValueError):
            INGEST["aggregate_robotwin"](
                run_dir,
                manifest_path=manifest_file,
                count_as_failure=bad_spec,
            )

    # Restore transcript_task_b_s3.json and verify bad records are rejected.
    bad_records = (
        _make_transcript(
            "task_b",
            3,
            success=False,
            error="ModelHTTPError: status_code: 500",
            robotwin_seeds=True,
        ),
        _make_transcript(
            "task_b",
            3,
            success=False,
            error=500,
            robotwin_seeds=True,
        ),
        _make_transcript("task_b", 3, success=None, robotwin_seeds=True),
        {
            **_make_transcript("task_b", 3, robotwin_seeds=True),
            "initial_native_seed": 9,
        },
        {
            **_make_transcript("task_b", 3, robotwin_seeds=True),
            "planner": "codex",
        },
        _make_transcript(
            "task_b",
            3,
            model="openai-chat:different-model",
            robotwin_seeds=True,
        ),
        _make_transcript("task_b", 3, model="", robotwin_seeds=True),
    )
    for bad_record in bad_records:
        _write_json(run_dir / "transcript_task_b_s3.json", bad_record)
        with pytest.raises(ValueError):
            INGEST["aggregate_robotwin"](run_dir, manifest_path=manifest_file)


def test_robocasa_four_views_model_guards_and_cli(tmp_path: Path):
    summary_data = {
        "protocol_id": "robocasa-harness-vla-v2",
        "valid_cells": 3,
        "expected_cells": 3,
        "splits": {
            "atomic": _make_split_summary(1),
            "composite_seen": _make_split_summary(1),
            "composite_unseen": _make_split_summary(0),
        },
        "overall": {
            "metric": "task_weighted_success_rate",
            "tasks": 3,
            "success_rate": 0.666667,
        },
    }
    summary_file = _write_json(tmp_path / "summary.json", summary_data)
    cells = (
        ("atomic", "Task1", 10.0, 100),
        ("composite_seen", "Task2", 20.0, 200),
        ("composite_unseen", "Task3", 30.0, 300),
    )
    manifest_file = _write_json(
        tmp_path / "target50.json",
        {
            "splits": {
                split_name: {"tasks": [task_name], "seeds": [1]}
                for split_name, task_name, _, _ in cells
            }
        },
    )
    run_root = tmp_path / "robocasa_run"
    for split_name, task_name, elapsed_s, output_tokens in cells:
        _write_json(
            run_root / split_name / f"{task_name}_s1/transcript_rc.json",
            _make_transcript(
                task_name,
                1,
                elapsed_s=elapsed_s,
                output_tokens=output_tokens,
            ),
        )

    aggregated = INGEST["aggregate_robocasa"](
        run_root,
        summary_file,
        manifest_path=manifest_file,
        expected_model="openai-chat:test-model",
        configuration_id="cfg-rpent",
        source_ids=["src-1"],
    )
    rates = {row["view_id"]: row["rate"] for row in aggregated["results"]}
    assert rates == {
        "robocasa": "66.67",
        "robocasa-atomic-seen": "100.00",
        "robocasa-composite-seen": "100.00",
        "robocasa-composite-unseen": "0.00",
    }
    assert "mean_total_tokens" not in aggregated["cost_results"][0]
    assert aggregated["cost_results"][0]["total_output_tokens"] == 600
    assert aggregated["token_summary"]["model"] == "openai-chat:test-model"

    base_cli = [
        "aggregate",
        "--benchmark",
        "robocasa",
        "--input-dir",
        str(run_root),
        "--target50-summary",
        str(summary_file),
    ]
    assert INGEST["main"]([*base_cli, "--manifest", str(manifest_file)]) == 0
    assert INGEST["main"]([*base_cli, "--planner", "claude_code"]) == 1
    assert (
        INGEST["main"](
            [
                *base_cli,
                "--manifest",
                str(manifest_file),
                "--count-as-failure",
                "Task1_s1",
            ]
        )
        == 1
    )

    assert INGEST["ROBOCASA_MANIFEST_PATH"].is_file()
    with pytest.raises(ValueError):
        INGEST["aggregate_robocasa"](run_root, summary_file)

    result_path = run_root / "atomic/Task1_s1/result.json"
    _write_json(result_path, {"planner": {"backend": "codex"}})
    with pytest.raises(ValueError, match="Non-API"):
        INGEST["aggregate_robocasa"](
            run_root, summary_file, manifest_path=manifest_file
        )
    _write_json(
        result_path,
        {"planner": {"backend": "api", "model": "openai-chat:wrong-model"}},
    )
    with pytest.raises(ValueError, match="result.json model"):
        INGEST["aggregate_robocasa"](
            run_root, summary_file, manifest_path=manifest_file
        )
    result_path.unlink()

    _write_json(
        run_root / "atomic/Task1_s1/transcript_rc.json",
        _make_transcript("WrongTask", 1),
    )
    with pytest.raises(ValueError, match="!="):
        INGEST["aggregate_robocasa"](
            run_root, summary_file, manifest_path=manifest_file
        )

    (run_root / "composite_unseen/Task3_s1/transcript_rc.json").unlink()
    _write_json(run_root / "stray/transcript_stray.json", _make_transcript("Task3", 1))
    with pytest.raises(ValueError):
        INGEST["aggregate_robocasa"](
            run_root, summary_file, manifest_path=manifest_file
        )


def test_upsert_anchor_no_overwrite_and_cli(tmp_path: Path, capsys):
    dataset = _make_dataset()
    dataset["benchmarks"][1]["notes"] = []
    dataset["sources"][0]["extra"] = [{"id": {"nested": 1}}]
    data_file = tmp_path / "results.json"
    INGEST["write_data"](data_file, dataset)

    assert INGEST["main"](["validate", "--data", str(data_file)]) == 0
    assert "OK:" in capsys.readouterr().out

    new_config = {
        "id": "cfg-rpent-2",
        "kind": "rpent",
        "model": "Model-2",
        "insert_after": "cfg-rpent",
    }
    robocasa_note = {
        "id": "rc-note-1",
        "configuration_ids": ["cfg-rpent"],
        "text": _bilingual("RC"),
    }
    payload_data = {
        "sources": [{"id": "src-2", "note": "Run 2"}],
        "configurations": [new_config],
        "benchmark_notes": {"robocasa": [robocasa_note]},
        "results": [
            _make_result("robocasa", "cfg-rpent", "55.00", "src-2", episodes=340)
        ],
        "cost_results": [
            _make_cost("robocasa", "cfg-rpent", 40.0, 500, source_ids=["src-2"])
        ],
    }
    payload_file = _write_json(tmp_path / "payload.json", payload_data)
    before_text = data_file.read_text(encoding="utf-8")

    base_cli = ["upsert", "--data", str(data_file), "--payload", str(payload_file)]
    assert INGEST["main"]([*base_cli, "--dry-run"]) == 0
    assert '+    {"id": "robocasa--cfg-rpent"' in capsys.readouterr().out
    assert data_file.read_text(encoding="utf-8") == before_text

    assert INGEST["main"]([*base_cli, "--in-place"]) == 0
    assert '\n          "id": "rc-note-1",' in data_file.read_text(encoding="utf-8")
    updated = INGEST["load_data"](data_file)
    assert [config["id"] for config in updated["configurations"]] == [
        "cfg-rpent",
        "cfg-rpent-2",
        "cfg-ext",
    ]
    assert INGEST["main"]([*base_cli, "--in-place"]) == 1
    assert "error:" in capsys.readouterr().err

    with pytest.raises(ValidationError):
        INGEST["upsert_data"]({"schema_version": 3}, {})
    bad_upsert = {
        "results": [{"id": "x", "view_id": [], "configuration_id": "cfg-rpent"}]
    }
    with pytest.raises(ValidationError):
        INGEST["upsert_data"](updated, bad_upsert)
    assert "src-2" in INGEST["dump_data"](
        INGEST["upsert_data"](
            _make_dataset(), {"sources": [{"id": "src-2", "note": "x"}]}
        )
    )
    for removed_flag in ("--allow-update-config", "--json"):
        with pytest.raises(SystemExit):
            INGEST["main"]([*base_cli, "--dry-run", removed_flag])
