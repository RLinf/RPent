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

"""Command line entry point for one EFE-compatible trial session."""

from __future__ import annotations

import argparse
from pathlib import Path

from robots.efe.constants import SANDBOX_CHOICES


def main(argv: list[str] | None = None) -> int:
    """Run one EFE trial session. Return a process status, not an evaluation score."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--instruction-file", required=True, type=Path)
    parser.add_argument("--workdir", type=Path, default=Path.cwd())
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--model", required=True)
    parser.add_argument("--timeout-s", type=int, default=600)
    parser.add_argument("--max-turns", type=int, default=200)
    parser.add_argument(
        "--reasoning-effort",
        choices=("none", "low", "medium", "high", "xhigh"),
        default="none",
    )
    parser.add_argument("--memory-dir", type=Path, default=None)
    parser.add_argument("--explore-sessions", type=int, default=1)
    parser.add_argument(
        "--no-skills",
        dest="skills",
        action="store_false",
        help="Register only the raw EFE tools, without high-level skills.",
    )
    parser.add_argument(
        "--codex-sandbox",
        choices=SANDBOX_CHOICES,
        default="read-only",
        help=(
            "Codex shell sandbox. read-only keeps shell commands away from the "
            "simulator, so only RPent tools act on the episode."
        ),
    )
    parser.add_argument(
        "--efe-base-url",
        default=None,
        help="Simulator service URL. Default: probe sim:8000, then localhost:8000.",
    )
    parser.add_argument(
        "--efe-connect-timeout-s",
        type=float,
        default=3600.0,
        help="How long to wait for the simulator to answer.",
    )
    parser.set_defaults(skills=True)
    args = parser.parse_args(argv)

    from robots.efe.runner import run_efe

    try:
        report = run_efe(
            args.instruction_file.read_text(encoding="utf-8"),
            workdir=args.workdir,
            output_dir=args.output_dir,
            model=args.model,
            timeout_s=args.timeout_s,
            max_turns=args.max_turns,
            reasoning_effort=args.reasoning_effort,
            memory_dir=args.memory_dir,
            explore_sessions=args.explore_sessions,
            skills=args.skills,
            sandbox=args.codex_sandbox,
            efe_base_url=args.efe_base_url,
            connect_timeout_s=args.efe_connect_timeout_s,
        )
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    if report["error"]:
        return 1
    if report.get("budget_stop"):
        # The planner used its whole time budget. The episode still stands and
        # the verifier grades it, so this is a clean stop: Harbor reads a
        # non-zero agent exit as a failed trial and rescans the whole run.log
        # for transient messages, which can invalidate a valid reward.
        print(f"planner budget used: {report['budget_stop']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
