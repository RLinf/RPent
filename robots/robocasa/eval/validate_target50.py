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

"""Validate and summarize a complete RoboCasa Target50 result directory."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from robots.robocasa.memory import selection_errors

DEFAULT_MANIFEST = Path(__file__).with_name("target50_v2.json")


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("top-level JSON value must be an object")
    return value


def validate_results(
    results_root: Path | str,
    *,
    manifest_path: Path | str = DEFAULT_MANIFEST,
) -> tuple[dict[str, Any], list[str]]:
    """Validate expected result files and return ``(summary, errors)``."""
    root = Path(results_root)
    manifest = _load_json(Path(manifest_path))
    errors: list[str] = []
    task_rates: list[float] = []
    valid_cells = 0
    split_summaries: dict[str, dict[str, Any]] = {}

    reference = manifest["planner_reference"]
    runtime_protocol = manifest["runtime_protocol"]
    is_v2 = manifest["protocol_id"] == "robocasa-harness-vla-v2"
    for split_name, split in manifest["splits"].items():
        split_successes = 0
        split_valid = 0
        for task_name in split["tasks"]:
            task_successes = 0
            for seed in split["seeds"]:
                relative = Path(split_name) / f"{task_name}_s{seed}" / "result.json"
                path = root / relative
                if not path.is_file():
                    errors.append(f"{relative}: missing result.json")
                    continue
                try:
                    result = _load_json(path)
                except (OSError, ValueError, json.JSONDecodeError) as exc:
                    errors.append(f"{relative}: invalid JSON: {exc}")
                    continue

                expected = {
                    "schema_version": runtime_protocol["result_schema_version"],
                    "protocol_id": manifest["protocol_id"],
                    "evaluation_split": split_name,
                    "task_name": task_name,
                    "environment_split": manifest["environment_split"],
                    "seed": seed,
                    "success_source": manifest["success_source"],
                }
                errors_before_cell = len(errors)
                for key, expected_value in expected.items():
                    if result.get(key) != expected_value:
                        errors.append(
                            f"{relative}: {key}={result.get(key)!r}, "
                            f"expected {expected_value!r}"
                        )

                success = result.get("success")
                if not isinstance(success, bool):
                    errors.append(f"{relative}: success must be a boolean")
                if result.get("valid") is not True:
                    errors.append(f"{relative}: result is not valid")
                if result.get("termination_reason") not in {
                    "completed",
                    "planner_timeout",
                }:
                    errors.append(f"{relative}: invalid termination_reason")

                planner = result.get("planner", {})
                expected_planner = {
                    "backend": reference["planner"],
                    "model": reference["model"],
                    "reasoning_effort": reference["reasoning_effort"],
                    "max_turns": reference["max_turns"],
                }
                if planner != expected_planner:
                    errors.append(
                        f"{relative}: planner profile does not match the manifest"
                    )

                runtime = result.get("runtime", {})
                if runtime.get("cell_timeout_seconds") != split["timeout_seconds"]:
                    errors.append(
                        f"{relative}: cell timeout does not match the manifest"
                    )
                if (
                    runtime.get("rldx_max_chunks")
                    != runtime_protocol["rldx_max_chunks"]
                ):
                    errors.append(
                        f"{relative}: RLDX max_chunks does not match the manifest"
                    )
                if (
                    runtime.get("rldx_settle_patience")
                    != runtime_protocol["rldx_settle_patience"]
                ):
                    errors.append(
                        f"{relative}: RLDX settle_patience does not match the manifest"
                    )
                if (
                    runtime.get("rldx_action_steps_per_chunk")
                    != runtime_protocol["rldx_action_steps_per_chunk"]
                ):
                    errors.append(
                        f"{relative}: RLDX action steps do not match the manifest"
                    )

                if is_v2:
                    memory = result.get("memory", {})
                    if not isinstance(memory, dict):
                        memory = {}
                    if memory.get("policy") != "task-global":
                        errors.append(
                            f"{relative}: memory policy does not match the evaluation"
                        )
                    errors.extend(
                        f"{relative}: {message}"
                        for message in selection_errors(
                            memory, task_name, manifest["environment_split"]
                        )
                    )
                    selected = memory.get("selected_files")
                    if memory.get("audit_status") != "ok":
                        errors.append(
                            f"{relative}: memory read audit is missing or invalid"
                        )
                    reads = memory.get("read_files")
                    if (
                        not isinstance(selected, list)
                        or not isinstance(reads, list)
                        or not all(isinstance(name, str) for name in reads)
                        or reads != sorted(set(reads))
                        or any(name not in selected for name in reads)
                    ):
                        errors.append(
                            f"{relative}: memory read_files must be a sorted subset of selected files"
                        )

                if (
                    len(errors) == errors_before_cell
                    and result.get("valid") is True
                    and isinstance(success, bool)
                ):
                    valid_cells += 1
                    split_valid += 1
                    task_successes += int(success)
                    split_successes += int(success)

            task_rates.append(task_successes / len(split["seeds"]))

        split_summaries[split_name] = {
            "successes": split_successes,
            "valid_cells": split_valid,
            "expected_cells": split["cell_count"],
            "success_rate": round(split_successes / split["cell_count"], 6),
        }

    summary = {
        "protocol_id": manifest["protocol_id"],
        "valid_cells": valid_cells,
        "expected_cells": manifest["total_cells"],
        "splits": split_summaries,
        "overall": {
            "metric": "task_weighted_success_rate",
            "tasks": manifest["total_tasks"],
            "success_rate": round(sum(task_rates) / len(task_rates), 6),
        },
    }
    return summary, errors


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "results_root",
        type=Path,
        help="Directory containing <split>/<Task>_s<seed>/result.json files.",
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the Target50 result validator."""
    args = _build_parser().parse_args(argv)
    try:
        summary, errors = validate_results(
            args.results_root,
            manifest_path=args.manifest,
        )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"manifest error: {exc}", file=sys.stderr)
        return 2

    print(json.dumps(summary, indent=2))
    if errors:
        print(f"validation failed with {len(errors)} error(s):", file=sys.stderr)
        for error in errors:
            print(f"- {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
