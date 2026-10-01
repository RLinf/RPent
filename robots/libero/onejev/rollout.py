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
"""Run independent OneJev LIBERO rollouts with one shared Pi0.5 service."""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from robots.libero.robot_spec import _spawn_vla_server, get_robot_spec
from rpent.dashboard.events import NullDashboardEventSink
from rpent.evaluation import write_json_atomic
from rpent.planner.check import LlmCheckRequest, check_llm
from rpent.planner.onejev_client import DEFAULT_ONEJEV_MODEL
from rpent.robots.runtime import (
    stop_owned_daemons,
    try_spawn_server,
    try_wait_server,
)
from rpent.utils.config import get_repo_root
from rpent.utils.daemon import ProcessDaemon, pick_free_port
from rpent.utils.logging import get_logger, init_output_dir

logger = get_logger("onejev_rollout")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    get_robot_spec().add_cli_args(parser, False)
    parser.set_defaults(
        planner="onejev", explore=False, memory_profile="local", memory_dir=None
    )
    parser.add_argument("--rounds", type=int, default=1)
    parser.add_argument(
        "--seed-step",
        type=int,
        default=1,
        help="Increment per round; 0 repeats the starting seed.",
    )
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", default=DEFAULT_ONEJEV_MODEL)
    parser.add_argument("--max-turns", type=int, default=40)
    parser.add_argument("--planner-timeout-s", type=int, default=1200)
    parser.add_argument("--check-timeout-s", type=int, default=120)
    parser.add_argument("--vla-startup-timeout-s", type=int, default=300)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=get_repo_root() / "logs" / "libero_pro_one_jev",
        help="Log root; suite/task, timestamp and round/seed directories are added.",
    )
    parser.add_argument("--verbose", action="store_true")
    return parser


def _create_run_directory(args: argparse.Namespace) -> Path:
    parent = args.output_dir.expanduser().resolve() / f"{args.suite}-{args.task}"
    parent.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    directory = parent / timestamp
    try:
        directory.mkdir()
    except FileExistsError:
        # Concurrent runs in one second must never overwrite each other.
        directory = parent / f"{timestamp}-{datetime.now().strftime('%f')}"
        directory.mkdir()
    return directory


def _round_command(
    args: argparse.Namespace, row: dict[str, Any], root: Path, vla_endpoint: str
) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "rpent.cli.main",
        "--robot",
        "libero",
        "--planner",
        "onejev",
        "--model",
        args.model,
        "--base-url",
        args.base_url,
        "--memory-profile",
        "local",
        "--suite",
        args.suite,
        "--task",
        str(args.task),
        "--seed",
        str(row["seed"]),
        "--max-turns",
        str(args.max_turns),
        "--planner-timeout-s",
        str(args.planner_timeout_s),
        "--max-episode-steps",
        str(args.max_episode_steps),
        "--output-dir",
        str(root / row["output_dir"]),
        "--vla-endpoint",
        vla_endpoint,
    ]
    for flag, value in (
        ("--libero-type", args.libero_type),
        ("--cuda-device", args.cuda_device),
        ("--onejev-config", args.onejev_config),
        ("--env-endpoint", args.env_endpoint),
        ("--flywheel-root", args.flywheel_root),
    ):
        if value is not None:
            command.extend((flag, str(value)))
    if args.collect_flywheel_data:
        command.append("--collect-flywheel-data")
    if args.verbose:
        command.append("--verbose")
    return command


def _stop_round(process: subprocess.Popen) -> None:
    """Let the CLI clean up, then stop only its owned process group if needed."""
    for sig, timeout in (
        (signal.SIGINT, 20),
        (signal.SIGTERM, 10),
        (signal.SIGKILL, 5),
    ):
        if process.poll() is not None:
            return
        try:
            os.killpg(process.pid, sig)
        except ProcessLookupError:
            return
        try:
            process.wait(timeout=timeout)
            return
        except subprocess.TimeoutExpired:
            continue


def _execute_round(command: list[str], console_path: Path) -> int:
    with console_path.open("w", encoding="utf-8") as console:
        with subprocess.Popen(
            command,
            cwd=get_repo_root(),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            text=True,
            bufsize=1,
        ) as process:
            try:
                for line in process.stdout:
                    console.write(line)
                    console.flush()
                    sys.stdout.write(line)
                    sys.stdout.flush()
                return process.wait()
            finally:
                _stop_round(process)


def _collect_round(directory: Path, exit_code: int | None) -> dict[str, Any]:
    """Read public step artifacts; never infer success from planner prose."""
    manifest_path = directory / "states.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    steps = manifest.get("steps", [])
    success = any(step["terminated"] for step in steps) if steps else None
    instruction = next(
        (
            step.get("extras", {}).get("task_language")
            for step in steps
            if step.get("extras", {}).get("task_language")
        ),
        None,
    )
    outcome_path = directory / "onejev_outcome.json"
    outcome = json.loads(outcome_path.read_text()) if outcome_path.exists() else {}
    error = outcome.get("error")
    if exit_code != 0:
        error = (
            error or f"RPent process exited with code {exit_code}; inspect console log"
        )
    elif success is None:
        error = error or "No public observation recorded; inspect console log"
    return {
        "status": "error" if error else "success" if success else "failure",
        "success": success,
        "task_instruction": instruction,
        "exit_code": exit_code,
        "error": error,
        "finish": outcome.get("finish"),
        "stats": outcome.get("stats", {}),
    }


def _save_summary(root: Path, summary: dict[str, Any]) -> None:
    rounds = summary["rounds"]
    summary["finished_rounds"] = sum(
        row["status"] not in {"pending", "running"} for row in rounds
    )
    summary["successful_rounds"] = sum(row["success"] is True for row in rounds)
    summary["success_rate"] = (
        summary["successful_rounds"] / len(rounds)
        if summary["status"] == "completed"
        else None
    )
    write_json_atomic(root / "summary.json", summary)


def _interrupt(signum, frame) -> None:
    raise KeyboardInterrupt


def main() -> int:
    """Check OneJev, share Pi0.5 across rounds, and persist a live summary."""
    parser = _parser()
    args = parser.parse_args()
    for name in (
        "rounds",
        "max_turns",
        "max_episode_steps",
        "planner_timeout_s",
        "check_timeout_s",
        "vla_startup_timeout_s",
    ):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.task < 0 or args.seed < 0 or args.seed_step < 0:
        parser.error("task, starting seed and seed step must be non-negative")
    if not args.suite.replace("_", "").isalnum():
        parser.error("suite must contain only letters, numbers and underscores")
    if args.env_endpoint and args.rounds > 1 and args.seed_step:
        parser.error("A borrowed env has a fixed seed; use --seed-step 0")
    spec = get_robot_spec()
    try:
        spec.validate_args(args)
        spec.parse_config(args)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))

    root = _create_run_directory(args)
    init_output_dir(root, verbose=args.verbose)
    logger.info("rollout logs: %s", root)
    summary: dict[str, Any] = {
        "status": "starting",
        "started_at": datetime.now().astimezone().isoformat(),
        "task_description": {
            "suite": args.suite,
            "task_id": args.task,
            "instruction": None,
        },
        "requested_rounds": args.rounds,
        "configuration": vars(args).copy(),
        "rounds": [
            {
                "round": index + 1,
                "seed": args.seed + index * args.seed_step,
                "output_dir": f"round_{index + 1:03d}_seed_{args.seed + index * args.seed_step}",
                "console_log": f"round_{index + 1:03d}_seed_{args.seed + index * args.seed_step}.console.log",
                "status": "pending",
                "success": None,
            }
            for index in range(args.rounds)
        ],
    }
    summary["configuration"]["output_dir"] = str(args.output_dir)
    _save_summary(root, summary)
    daemons: dict[str, ProcessDaemon] = {}
    rpc = None
    events = NullDashboardEventSink()
    started = time.monotonic()
    signal.signal(signal.SIGTERM, _interrupt)
    try:
        check = check_llm(
            LlmCheckRequest(
                planner="onejev",
                model=args.model,
                base_url=args.base_url,
                timeout_s=args.check_timeout_s,
            )
        )
        summary["service_check"] = check.as_dict()
        _save_summary(root, summary)
        if not check.ok:
            raise RuntimeError(f"OneJev service check failed: {check.detail}")
        port = pick_free_port() if not args.vla_endpoint else None
        endpoint = args.vla_endpoint or f"http://127.0.0.1:{port}"
        daemon, rpc = try_spawn_server(
            daemons,
            events,
            "vla",
            lambda: _spawn_vla_server(args, root, port=port),
        )
        try_wait_server(daemons, events, "vla", rpc, daemon, args.vla_startup_timeout_s)
        summary["vla_endpoint"] = endpoint
        summary["status"] = "running"
        _save_summary(root, summary)
        for row in summary["rounds"]:
            if daemon is not None and daemon.poll() is not None:
                raise RuntimeError(
                    "Shared Pi0.5 service exited; inspect vla_server.log"
                )
            row.update(
                status="running", started_at=datetime.now().astimezone().isoformat()
            )
            _save_summary(root, summary)
            logger.info("round %d/%d: seed %d", row["round"], args.rounds, row["seed"])
            command = _round_command(args, row, root, endpoint)
            row["command"] = command
            round_started = time.monotonic()
            try:
                exit_code = _execute_round(command, root / row["console_log"])
                row.update(_collect_round(root / row["output_dir"], exit_code))
            except KeyboardInterrupt:
                row.update(_collect_round(root / row["output_dir"], None))
                row.update(
                    status="interrupted", error="Rollout interrupted by operator"
                )
                raise
            except Exception as exc:
                row.update(status="error", error=f"{type(exc).__name__}: {exc}")
                logger.error("round %d failed: %s", row["round"], row["error"])
            finally:
                row["elapsed_s"] = round(time.monotonic() - round_started, 3)
                row["ended_at"] = datetime.now().astimezone().isoformat()
                if (
                    row.get("task_instruction")
                    and not summary["task_description"]["instruction"]
                ):
                    summary["task_description"]["instruction"] = row["task_instruction"]
                _save_summary(root, summary)
            logger.info(
                "round %d: status=%s success=%s",
                row["round"],
                row["status"],
                row["success"],
            )
        summary["status"] = "completed"
    except KeyboardInterrupt:
        summary.update(
            status="interrupted", error="Rollout batch interrupted by operator"
        )
    except Exception as exc:
        summary.update(status="error", error=f"{type(exc).__name__}: {exc}")
        logger.error("rollout batch stopped: %s", summary["error"])
    finally:
        try:
            if rpc is not None:
                rpc.close()
        finally:
            stop_owned_daemons(daemons, events)
        summary["elapsed_s"] = round(time.monotonic() - started, 3)
        summary["ended_at"] = datetime.now().astimezone().isoformat()
        _save_summary(root, summary)
        logger.info("summary: %s", root / "summary.json")
    if summary["status"] == "interrupted":
        return 130
    return int(
        summary["status"] == "error"
        or any(row["status"] == "error" for row in summary["rounds"])
    )


if __name__ == "__main__":
    sys.exit(main())
