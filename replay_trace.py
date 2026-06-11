#!/usr/bin/env python3
"""Replay a JSONL command trace into a running PhysicalAgent REPL driver.

The historical files are often named ``recipe_*.jsonl``, but they are really
traces of commands to send to ``repl_driver.py``. This tool reads each JSON
line, writes the command to ``command.json``, waits for the next entry in
``states.json``, then prints a compact result summary.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

from physicalagent.config import get_default_workdir_prefix


def _fmt(value: Any, digits: int = 4) -> str:
    if isinstance(value, int | float):
        return f"{value:.{digits}f}"
    return "?" if value is None else str(value)


def _load_trace(path: Path) -> list[dict[str, Any]]:
    commands: list[dict[str, Any]] = []
    with path.open() as f:
        for line_no, raw in enumerate(f, 1):
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise SystemExit(f"{path}:{line_no}: invalid JSON: {exc}") from exc
            if not isinstance(record, dict):
                raise SystemExit(f"{path}:{line_no}: expected a JSON object")
            command = record.get("command") if "command" in record else record
            if not isinstance(command, dict):
                raise SystemExit(f"{path}:{line_no}: record.command must be an object")
            commands.append(command)
    return commands


def _write_command(workdir: Path, command: dict[str, Any]) -> None:
    tmp_path = workdir / "command.json.tmp"
    cmd_path = workdir / "command.json"
    with tmp_path.open("w") as f:
        json.dump(command, f)
    tmp_path.replace(cmd_path)


def _load_states(states_path: Path) -> list:
    """Return parsed states.json as a list (empty on error / missing)."""
    if not states_path.exists():
        return []
    try:
        with states_path.open() as f:
            arr = json.load(f)
        if isinstance(arr, list):
            return arr
    except Exception:
        pass
    return []


def _wait_for_step(states_path: Path, step: int, timeout_s: float, poll_s: float) -> dict:
    """Block until ``states_path`` has an entry at index ``step``. Return it."""
    start = time.time()
    while True:
        arr = _load_states(states_path)
        if step < len(arr) and isinstance(arr[step], dict):
            return arr[step]
        if time.time() - start > timeout_s:
            raise TimeoutError(
                f"timed out waiting for step {step} in {states_path} "
                f"(have {len(arr)} entries)"
            )
        time.sleep(poll_s)


def _summarize(entry: dict, action: str, elapsed_s: float) -> tuple[str, bool]:
    result = entry.get("result", {}) if isinstance(entry, dict) else {}
    if not isinstance(result, dict):
        result = {}

    terminated = bool(result.get("libero_terminated") or entry.get("libero_terminated"))
    extras: list[str] = []
    if action == "move_to":
        extras.append(f"dist={_fmt(result.get('final_dist_m'))}")
        extras.append(f"steps={_fmt(result.get('steps_used'), digits=0)}")
    elif action == "pi0_pick":
        extras.append(f"chunks={_fmt(result.get('chunks_used'), digits=0)}")
        extras.append(f"peak_lift={_fmt(result.get('peak_lift_m'))}")
    elif action == "release":
        extras.append(f"grip_open={_fmt(result.get('final_gripper_opening'))}")
    elif "success" in result:
        extras.append(f"success={result.get('success')}")

    summary = f"done in {elapsed_s:.1f}s  libero_term={terminated}"
    if extras:
        summary += "  " + "  ".join(extras)
    return summary, terminated


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Replay a JSONL command trace into a running repl_driver.py workdir. "
            "Files historically named recipe_*.jsonl are accepted."
        )
    )
    parser.add_argument("trace", type=Path, help="JSONL command trace to replay")
    parser.add_argument(
        "--workdir",
        type=Path,
        default=Path(get_default_workdir_prefix()),
        help="REPL workdir containing command.json + states.json",
    )
    parser.add_argument(
        "--start-step",
        type=int,
        default=1,
        help="First REPL step number expected for the first command",
    )
    parser.add_argument(
        "--timeout-s",
        type=float,
        default=600.0,
        help="Seconds to wait for each new states.json entry",
    )
    parser.add_argument(
        "--poll-s",
        type=float,
        default=0.5,
        help="Polling interval while waiting for states.json",
    )
    parser.add_argument(
        "--keep-note",
        action="store_true",
        help="Keep the optional note field when sending commands",
    )
    parser.add_argument(
        "--continue-after-terminated",
        action="store_true",
        help="Do not stop early when a command reports libero_terminated=True",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without writing to command.json",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_argparser().parse_args(argv)
    if args.start_step < 1:
        print("--start-step must be >= 1", file=sys.stderr)
        return 2
    if not args.trace.exists():
        print(f"trace not found: {args.trace}", file=sys.stderr)
        return 2
    if not args.dry_run and not args.workdir.is_dir():
        print(f"workdir not found: {args.workdir}", file=sys.stderr)
        return 2

    commands = _load_trace(args.trace)
    print(f"[trace] loaded {len(commands)} commands from {args.trace}")

    states_path = args.workdir / "states.json"

    for offset, command in enumerate(commands):
        step = args.start_step + offset
        step_id = f"{step:02d}"
        action = str(command.get("action", "?"))
        clean = dict(command) if args.keep_note else {
            key: value for key, value in command.items() if key != "note"
        }

        if not args.dry_run:
            existing = _load_states(states_path)
            if step < len(existing) and isinstance(existing[step], dict):
                print(
                    f"[trace] refusing to replay step {step_id}: states.json already "
                    f"has entry {step}. Use --start-step for the next pending step "
                    f"or clear the workdir.",
                    file=sys.stderr,
                )
                return 2

        print(f"[trace] step {step_id}: {action}", flush=True)
        if args.dry_run:
            print(json.dumps(clean, default=str))
            continue

        start = time.time()
        _write_command(args.workdir, clean)
        try:
            entry = _wait_for_step(
                states_path, step,
                timeout_s=args.timeout_s, poll_s=args.poll_s,
            )
        except TimeoutError as exc:
            print(f"[trace] TIMEOUT: {exc}", file=sys.stderr, flush=True)
            return 2

        summary, terminated = _summarize(
            entry, action=action, elapsed_s=time.time() - start,
        )
        print(f"[trace] step {step_id}: {summary}", flush=True)
        if terminated and not args.continue_after_terminated:
            print(f"[trace] libero_terminated=True at step {step_id}; stopping")
            break

    print("[trace] done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())