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

"""Outer BEHAVIOR Explore harness.

Run as:

    python -m robots.behavior.harness explore --attempts 3 -- <behavior args>

Each attempt is a separate standard RPent process:

    rpent --robot behavior --behavior-mode explore --output-dir <attempt_dir> ...

The harness never passes main ``--explore`` and never resets inside a running
planner invocation; restart-env semantics come from process isolation.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from collections.abc import Iterable, Mapping, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

_FORBIDDEN_RPENT_FLAGS = {
    "--env",
    "--explore",
    "--output-dir",
    "--robot",
    "--behavior-mode",
}


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def _positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def _default_output_dir() -> Path:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return Path("logs") / f"{stamp}_behavior_explore_outer"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m robots.behavior.harness",
        description="BEHAVIOR outer harness commands.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    explore = subparsers.add_parser(
        "explore",
        description="Run independent BEHAVIOR Explore attempts.",
    )
    explore.add_argument("--attempts", type=_positive_int, default=1)
    explore.add_argument(
        "--output-dir",
        type=Path,
        default=_default_output_dir(),
        help="Outer harness output root. Each attempt receives a child output dir.",
    )
    explore.add_argument(
        "--rpent-executable",
        default=os.environ.get("RPENT_EXECUTABLE", "rpent"),
        help="RPent console script or executable path.",
    )
    explore.add_argument(
        "--cwd",
        type=Path,
        default=None,
        help="Working directory for each RPent attempt. Defaults to the current cwd.",
    )
    explore.add_argument(
        "--timeout-s",
        type=_positive_float,
        default=None,
        help="Optional wall-clock timeout per attempt.",
    )
    explore.add_argument(
        "--stop-on-explicit-success",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Stop after a terminal receipt explicitly reports success.",
    )
    explore.add_argument(
        "--dry-run",
        action="store_true",
        help="Write the attempt argv summary without launching RPent.",
    )
    return parser


def _normalize_passthrough(values: Sequence[str]) -> list[str]:
    passthrough = list(values)
    if passthrough and passthrough[0] == "--":
        passthrough = passthrough[1:]
    seen_forbidden = [
        value
        for value in passthrough
        if value in _FORBIDDEN_RPENT_FLAGS
        or any(value.startswith(f"{flag}=") for flag in _FORBIDDEN_RPENT_FLAGS)
    ]
    if seen_forbidden:
        raise ValueError(
            "the outer harness owns these RPent flags: "
            + ", ".join(sorted(set(seen_forbidden)))
        )
    return passthrough


def _attempt_argv(
    *,
    rpent_executable: str,
    attempt_dir: Path,
    passthrough: Sequence[str],
) -> list[str]:
    return [
        rpent_executable,
        "--robot",
        "behavior",
        "--behavior-mode",
        "explore",
        "--output-dir",
        str(attempt_dir),
        *passthrough,
    ]


def _iter_json_objects(path: Path) -> Iterable[Mapping[str, Any]]:
    if path.stat().st_size > 50_000_000:
        return
    if path.suffix == ".jsonl":
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    value = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(value, Mapping):
                    yield value
        return
    if path.suffix == ".json":
        try:
            with path.open(encoding="utf-8") as handle:
                value = json.load(handle)
        except (json.JSONDecodeError, OSError):
            return
        if isinstance(value, Mapping):
            yield value
        elif isinstance(value, list):
            for item in value:
                if isinstance(item, Mapping):
                    yield item


def _nested_get(value: Mapping[str, Any], path: Sequence[str]) -> Any:
    current: Any = value
    for key in path:
        if not isinstance(current, Mapping) or key not in current:
            return None
        current = current[key]
    return current


def _first_bool(
    value: Mapping[str, Any], paths: Sequence[Sequence[str]]
) -> bool | None:
    for path in paths:
        item = _nested_get(value, path)
        if isinstance(item, bool):
            return item
    return None


def _terminal_score(path: Path, value: Mapping[str, Any]) -> int:
    score = 0
    lower_name = path.name.lower()
    if any(token in lower_name for token in ("terminal", "receipt", "manifest")):
        score += 2
    if any(key in value for key in ("_finish", "finish", "terminal", "task_success")):
        score += 3
    if any(key in value for key in ("official", "done", "info_done", "receipt")):
        score += 1
    return score


def _summarize_receipt(
    path: Path, value: Mapping[str, Any], root: Path
) -> dict[str, Any]:
    task_success = _first_bool(
        value,
        (
            ("task_success",),
            ("finish", "task_success"),
            ("receipt", "task_success"),
            ("result", "task_success"),
        ),
    )
    official_success = _first_bool(
        value,
        (
            ("official", "success"),
            ("done", "success"),
            ("info_done", "success"),
            ("finish", "official", "success"),
            ("receipt", "official", "success"),
            ("result", "official", "success"),
        ),
    )
    terminal = _first_bool(
        value,
        (
            ("_finish",),
            ("terminal",),
            ("finish", "_finish"),
            ("receipt", "_finish"),
            ("result", "_finish"),
        ),
    )
    reason = (
        _nested_get(value, ("stop_reason",))
        or _nested_get(value, ("reason",))
        or _nested_get(value, ("finish", "reason"))
        or _nested_get(value, ("receipt", "stop_reason"))
        or _nested_get(value, ("result", "stop_reason"))
    )
    return {
        "path": str(path.relative_to(root)),
        "terminal": terminal,
        "task_success": task_success,
        "official_success": official_success,
        "stop_reason": reason if isinstance(reason, str) else None,
    }


def _collect_terminal_receipts(attempt_dir: Path) -> list[dict[str, Any]]:
    candidates: list[tuple[int, Path, Mapping[str, Any]]] = []
    if not attempt_dir.exists():
        return []
    for path in attempt_dir.rglob("*"):
        if path.suffix not in {".json", ".jsonl"} or not path.is_file():
            continue
        for value in _iter_json_objects(path):
            score = _terminal_score(path, value)
            if score > 0:
                candidates.append((score, path, value))
    candidates.sort(key=lambda item: (-item[0], str(item[1])))
    return [
        _summarize_receipt(path, value, attempt_dir)
        for _, path, value in candidates[:20]
    ]


def _explicit_success(receipts: Sequence[Mapping[str, Any]]) -> bool:
    return any(
        receipt.get("task_success") is True or receipt.get("official_success") is True
        for receipt in receipts
    )


def run_explore(args: argparse.Namespace, passthrough: Sequence[str]) -> int:
    passthrough = _normalize_passthrough(passthrough)
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    attempts: list[dict[str, Any]] = []
    summary_path = output_dir / "explore_harness_summary.json"

    for attempt_index in range(1, args.attempts + 1):
        attempt_dir = output_dir / f"attempt_{attempt_index:03d}"
        attempt_dir.mkdir(parents=True, exist_ok=True)
        argv = _attempt_argv(
            rpent_executable=args.rpent_executable,
            attempt_dir=attempt_dir,
            passthrough=passthrough,
        )
        started_at = time.time()
        attempt: dict[str, Any] = {
            "attempt_index": attempt_index,
            "output_dir": str(attempt_dir),
            "argv": argv,
            "returncode": None,
            "timed_out": False,
            "elapsed_s": None,
            "terminal_receipts": [],
            "explicit_success": False,
        }
        attempts.append(attempt)
        if args.dry_run:
            attempt["returncode"] = 0
            attempt["elapsed_s"] = 0.0
            continue

        stdout_path = attempt_dir / "stdout.log"
        stderr_path = attempt_dir / "stderr.log"
        with (
            stdout_path.open("w", encoding="utf-8") as stdout,
            stderr_path.open(
                "w",
                encoding="utf-8",
            ) as stderr,
        ):
            try:
                completed = subprocess.run(
                    argv,
                    cwd=str(args.cwd.expanduser().resolve()) if args.cwd else None,
                    stdout=stdout,
                    stderr=stderr,
                    timeout=args.timeout_s,
                    check=False,
                    shell=False,
                )
                attempt["returncode"] = completed.returncode
            except subprocess.TimeoutExpired:
                attempt["returncode"] = 124
                attempt["timed_out"] = True
        attempt["elapsed_s"] = round(time.time() - started_at, 1)
        receipts = _collect_terminal_receipts(attempt_dir)
        attempt["terminal_receipts"] = receipts
        attempt["explicit_success"] = _explicit_success(receipts)
        if args.stop_on_explicit_success and attempt["explicit_success"]:
            break

    successful_attempts = [
        attempt["attempt_index"] for attempt in attempts if attempt["explicit_success"]
    ]
    summary = {
        "schema_version": 1,
        "kind": "behavior_explore_outer_harness_summary",
        "dry_run": bool(args.dry_run),
        "output_dir": str(output_dir),
        "attempts_requested": args.attempts,
        "attempts_run": len(attempts),
        "successful_attempts": successful_attempts,
        "success_source": "explicit terminal receipt fields only",
        "attempts": attempts,
    }
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, default=str)
        handle.write("\n")
    print(json.dumps(summary, indent=2, default=str))
    if args.dry_run:
        return 0
    return 0 if successful_attempts else 1


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args, passthrough = parser.parse_known_args(argv)
    try:
        if args.command == "explore":
            return run_explore(args, passthrough)
    except ValueError as exc:
        parser.error(str(exc))
    parser.error(f"unsupported command: {args.command}")
    return 2


if __name__ == "__main__":
    sys.exit(main())
