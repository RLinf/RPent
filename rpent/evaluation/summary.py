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
"""Summarize versioned episode metrics against a predeclared test manifest."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping
from pathlib import Path
from statistics import mean, median, quantiles
from typing import Any

from rpent.evaluation.result import write_json_atomic


def _distribution(values: list[float], expected: int) -> dict[str, Any]:
    """Report the linear-interpolated 90th percentile of measured durations."""
    result: dict[str, Any] = {
        "measured": len(values),
        "missing": expected - len(values),
        "median": None,
        "p90": None,
    }
    if values:
        result["median"] = median(values)
        result["p90"] = (
            quantiles(values, n=10, method="inclusive")[8]
            if len(values) > 1
            else values[0]
        )
    return result


def _token_breakdown(records: list[dict]) -> dict:
    coverage: Counter[str] = Counter()
    breakdown: dict[tuple, dict] = {}
    for record in records:
        ledger = record["token_ledger"]
        coverage[ledger["coverage"]] += 1
        for entry in ledger["entries"]:
            # Partial episodes retain observed subtotals in their own rows.
            fields = ("provider", "model", "tokenizer", "modality", "stage", "module")
            identity = {f: entry.get(f) for f in fields}
            identity["episode_coverage"] = ledger["coverage"]
            key = tuple(identity.values())
            row = breakdown.setdefault(
                key, {**identity, "input_tokens": None, "output_tokens": None}
            )
            for side in ("input_tokens", "output_tokens"):
                count = entry.get(side)
                if count is not None:
                    row[side] = (row[side] or 0) + count
    return {
        "episode_counts": {
            s: coverage[s] for s in ("complete", "partial", "unknown", "not_applicable")
        },
        "observed_totals": list(breakdown.values()),
    }


def _unresolved_environment_start(record: Mapping[str, Any]) -> bool:
    """Whether this unscored attempt ended during environment initialization."""
    return (
        record["status"] == "unresolved"
        and record.get("stop_reason") == "environment_initialization_failed"
        and not record.get("manual_intervention", False)
        and record["timing"]["observed_s"] is None
        and record["native_verdict"] is None
    )


def _infrastructure_reviews(spec: dict, attempts: list[dict]) -> list[dict]:
    """Validate external rulings against this condition's immutable attempts."""
    reviews = spec.get("infrastructure_reviews", {})
    if not isinstance(reviews, Mapping):
        raise ValueError("infrastructure_reviews must map run IDs to reasons")
    by_id = {record["run_id"]: record for record in attempts}
    audit = []
    for run_id, reason in reviews.items():
        if not isinstance(run_id, str) or run_id not in by_id:
            raise ValueError(
                f"Unknown infrastructure review run_id {run_id!r} for {spec['trial_id']}"
            )
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError(
                f"An infrastructure review requires a nonempty reason for {run_id!r}"
            )
        if not _unresolved_environment_start(by_id[run_id]):
            raise ValueError(
                f"Infrastructure review requires an unresolved environment initialization failure for {run_id!r}"
            )
        original = by_id[run_id]
        for replacement in attempts:
            if replacement["status"] not in {"success", "failure"} or replacement.get(
                "manual_intervention"
            ):
                continue
            for field in ("environment", "budget", "checkpoint", "instruction_version"):
                if original.get(field) is not None and original[
                    field
                ] != replacement.get(field):
                    raise ValueError(f"Mismatched {field} for replacement of {run_id}")
        audit.append(
            {
                "trial_id": spec["trial_id"],
                "method": spec["method"],
                "run_id": run_id,
                "original_status": "unresolved",
                "effective_status": "infra_invalid",
                "reason": reason.strip(),
            }
        )
    return audit


def _cell(planned: list[tuple[dict, list[dict]]]) -> dict:
    accepted = []
    infra = 0
    assisted = 0
    for spec, attempts in planned:
        scores = []
        for record in attempts:
            if record.get("manual_intervention", False):
                assisted += 1
            elif record["status"] == "infra_invalid" or record["run_id"] in spec.get(
                "infrastructure_reviews", {}
            ):
                infra += 1
            else:
                scores.append(record)
        if len(scores) > 1:
            raise ValueError(
                f"Multiple non-infrastructure attempts for {spec['trial_id']} / {spec['method']}"
            )
        if scores and scores[0]["status"] != "unresolved":
            accepted.append(scores[0])
    successes = [r for r in accepted if r["status"] == "success"]

    def durations(group: list[dict], field: str) -> dict:
        return _distribution(
            [r["timing"][field] for r in group if r["timing"][field] is not None],
            len(group),
        )

    return {
        "planned": len(planned),
        "valid": len(accepted),
        "successes": len(successes),
        "failures": len(accepted) - len(successes),
        "infra_invalid_attempts": infra,
        "assisted_attempts": assisted,
        "unresolved": len(planned) - len(accepted),
        "native_sr": len(successes) / len(accepted) if accepted else None,
        "run_s": durations(accepted, "run_s"),
        "success_s": durations(successes, "success_s"),
        "token_usage": _token_breakdown(accepted),
    }


def summarize(trials: Iterable[Mapping[str, Any]], records: Iterable[dict]) -> dict:
    """Match each method and planned condition to its unique scored attempt.

    A replacement may accompany infrastructure-invalid attempts. Reject duplicates,
    unplanned runs and conflicting trial metadata instead of choosing a result.
    """
    planned = {}
    for trial in trials:
        spec = dict(trial)
        for field in ("method", "trial_id", "task", "scene", "seed"):
            if field not in spec:
                raise ValueError(f"Missing manifest field {field!r}")
        key = (spec["method"], spec["trial_id"])
        if key in planned:
            raise ValueError(f"Duplicate manifest entry {key}")
        planned[key] = (spec, [])
    if not planned:
        raise ValueError("Empty trial manifest")
    seen = set()
    for record in records:
        key = (record["method"], record["trial_id"])
        if record["schema_version"] != 1 or key not in planned:
            raise ValueError(f"Unsupported or unplanned episode {key}")
        if record["status"] not in {
            "success",
            "failure",
            "unresolved",
            "infra_invalid",
        }:
            raise ValueError(f"Invalid episode status for {key}")
        rid = record["run_id"]
        if rid in seen:
            raise ValueError(f"Duplicate run_id {rid}")
        seen.add(rid)
        spec, attempts = planned[key]
        for field in (
            "task",
            "scene",
            "seed",
            "budget",
            "checkpoint",
            "environment",
            "instruction_version",
        ):
            # A pre-reset attempt can lack the instruction and worker checkpoint
            # that ready episodes verify. Its registered task, seed, and budgets
            # are checked even when a reviewer later rules it invalid.
            if (
                field in spec
                and field in {"instruction_version", "checkpoint"}
                and record.get(field) is None
                and _unresolved_environment_start(record)
            ):
                continue
            if field in spec and spec[field] != record[field]:
                raise ValueError(f"Mismatched {field} for {key}")
        attempts.append(record)
    # Older single-benchmark manifests may omit the environment. A missing
    # episode in a multi-benchmark batch needs an explicit assignment.
    known_environments = defaultdict(set)
    for spec, attempts in planned.values():
        environments = {
            e
            for e in (
                spec.get("environment"),
                *(r.get("environment") for r in attempts),
            )
            if e is not None
        }
        if len(environments) > 1:
            raise ValueError(f"Conflicting environments for {spec['trial_id']}")
        if environments:
            spec["environment"] = environments.pop()
            known_environments[spec["method"]].add(spec["environment"])
    cells = defaultdict(list)
    reviews = []
    for spec, attempts in planned.values():
        reviews.extend(_infrastructure_reviews(spec, attempts))
        method = spec["method"]
        environment = spec.get("environment")
        if environment is None:
            if len(known_environments[method]) > 1:
                raise ValueError(
                    f"Specify environment in manifest for {spec['trial_id']}"
                )
            environment = next(iter(known_environments[method]), None)
        cells[(method, environment, spec["task"], spec["scene"])].append(
            (spec, attempts)
        )
    rows = []
    methods = defaultdict(list)
    for (method, environment, task, scene), items in cells.items():
        identity = {"environment": environment} if environment is not None else {}
        row = {
            "method": method,
            **identity,
            "task": task,
            "scene": scene,
            **_cell(items),
        }
        rows.append(row)
        methods[(method, environment)].append(row)
    by_method = []
    for (method, environment), group in methods.items():
        complete = all(row["unresolved"] == 0 for row in group)
        per_task = defaultdict(list)
        if complete:
            for row in group:
                per_task[row["task"]].append(row["native_sr"])
        by_method.append(
            {
                "method": method,
                **({"environment": environment} if environment is not None else {}),
                "complete": complete,
                "native_sr_macro": mean(mean(rates) for rates in per_task.values())
                if complete
                else None,
            }
        )
    return {
        "schema_version": 1,
        "cells": rows,
        "methods": by_method,
        "infrastructure_reviews": reviews,
    }


def main() -> None:
    """Write a JSON summary of the episode files under the supplied run directory."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path, help="JSON object with a trials array")
    parser.add_argument("runs", type=Path, help="Root of the episode run directories")
    parser.add_argument("output", type=Path, help="Destination summary JSON")
    args = parser.parse_args()
    try:
        if not args.runs.is_dir():
            raise ValueError(f"Missing runs directory: {args.runs}")
        manifest = json.loads(args.manifest.read_text())
        result = summarize(
            manifest["trials"],
            (
                json.loads(path.read_text())
                for path in sorted(args.runs.rglob("episode_metrics.json"))
            ),
        )
        write_json_atomic(args.output, result)
    except (ValueError, KeyError, OSError, TypeError) as exc:
        parser.exit(2, f"{exc}\n")


if __name__ == "__main__":
    main()
