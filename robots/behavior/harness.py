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
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

from robots.behavior.task_specs import get_task_spec
from robots.behavior.terminal_success import validate_terminal_success_receipt
from rpent.memory import MemoryManager

_FORBIDDEN_RPENT_FLAGS = {
    "--env",
    "--explore",
    "--output-dir",
    "--robot",
    "--behavior-mode",
    "--memory-dir",
    "--memory-profile",
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
        "--memory-dir",
        type=Path,
        default=None,
        help="Official MemoryManager corpus root (default: <output-dir>/memory).",
    )
    explore.add_argument(
        "--auto-merge-memory",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Merge the shared attempt inbox with MemoryManager after the run.",
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
    memory_dir: Path,
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
        "--memory-profile",
        "local",
        "--memory-dir",
        str(memory_dir),
        *passthrough,
    ]


def _cell_tag_from_passthrough(passthrough: Sequence[str]) -> str:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--task-name")
    parser.add_argument("--public-seed", type=int)
    parser.add_argument("--seed", type=int)
    identity, _ = parser.parse_known_args(list(passthrough))
    if not identity.task_name:
        raise ValueError("Explore passthrough requires --task-name")
    public_seed = (
        identity.public_seed if identity.public_seed is not None else identity.seed
    )
    if public_seed is None:
        raise ValueError("Explore passthrough requires --public-seed or --seed")
    if (
        identity.public_seed is not None
        and identity.seed is not None
        and identity.public_seed != identity.seed
    ):
        raise ValueError("--public-seed and --seed disagree")
    return get_task_spec(identity.task_name).tag(public_seed)


def _collect_terminal_receipts(attempt_dir: Path) -> list[dict[str, Any]]:
    receipt_path = attempt_dir / "terminal_receipt.json"
    if not receipt_path.is_file() or receipt_path.is_symlink():
        return []
    try:
        if receipt_path.stat().st_size > 1_000_000:
            raise ValueError("terminal receipt exceeds 1 MB")
        with receipt_path.open(encoding="utf-8") as handle:
            value = json.load(handle)
    except (json.JSONDecodeError, OSError, ValueError) as exc:
        return [
            {
                "path": receipt_path.name,
                "terminal": False,
                "task_success": False,
                "official_success": False,
                "valid": False,
                "validation_error": str(exc),
            }
        ]
    validation = validate_terminal_success_receipt(
        tool_name="finish",
        step=0,
        result=value,
        output_dir=attempt_dir,
    )
    return [
        {
            "path": receipt_path.name,
            "terminal": validation.valid,
            "task_success": validation.valid,
            "official_success": validation.valid,
            "valid": validation.valid,
            "validation_error": validation.reason,
        }
    ]


def run_explore(args: argparse.Namespace, passthrough: Sequence[str]) -> int:
    passthrough = _normalize_passthrough(passthrough)
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    memory_dir = (
        args.memory_dir.expanduser().resolve()
        if args.memory_dir is not None
        else (output_dir / "memory").resolve()
    )
    cell_tag = _cell_tag_from_passthrough(passthrough)
    attempts: list[dict[str, Any]] = []
    summary_path = output_dir / "explore_harness_summary.json"

    for attempt_index in range(1, args.attempts + 1):
        attempt_dir = output_dir / f"attempt_{attempt_index:03d}"
        attempt_dir.mkdir(parents=True, exist_ok=True)
        argv = _attempt_argv(
            rpent_executable=args.rpent_executable,
            attempt_dir=attempt_dir,
            memory_dir=memory_dir,
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
        attempt["explicit_success"] = bool(
            len(receipts) == 1 and receipts[0].get("valid") is True
        )
        if args.stop_on_explicit_success and attempt["explicit_success"]:
            break

    successful_attempts = [
        attempt["attempt_index"] for attempt in attempts if attempt["explicit_success"]
    ]
    merge_result: dict[str, Any] | None = None
    merge_error: str | None = None
    merge_candidates = [
        attempt for attempt in attempts if attempt.get("returncode") == 0
    ]
    if args.auto_merge_memory and not args.dry_run and merge_candidates:
        selected = next(
            (
                attempt
                for attempt in merge_candidates
                if attempt.get("explicit_success") is True
            ),
            merge_candidates[-1],
        )
        try:
            merge_result = MemoryManager(memory_dir).merge_memory(
                cell_tag=cell_tag,
                run_state_dir=selected["output_dir"],
                solved=bool(selected.get("explicit_success")),
            )
        except Exception as exc:
            merge_error = f"{type(exc).__name__}: {exc}"
    summary = {
        "schema_version": 1,
        "kind": "behavior_explore_outer_harness_summary",
        "dry_run": bool(args.dry_run),
        "output_dir": str(output_dir),
        "memory_dir": str(memory_dir),
        "memory_cell_tag": cell_tag,
        "memory_merge": merge_result,
        "memory_merge_error": merge_error,
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
    if merge_error is not None:
        return 1
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
