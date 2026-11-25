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

"""Resolve standalone RPent sampling options and share them with batch collection."""

import argparse
import os
import sys
from datetime import datetime

from rpent.planner.base import REASONING_EFFORTS
from rpent.utils.config import get_repo_root

OPTIONS = (
    ("--sam3-timeout-s", float, 120),
    ("--max-actions-per-call", int, 3),
    ("--reasoning-history", int, -1),
    ("--image-history", int, -1),
    ("--image_size", int, 1024),
    ("--max-turns", int, 100),
    ("--max-tokens", int, 32768),
    ("--context-window", int, 262144),
    ("--compaction-fraction", float, 0.7),
    ("--compaction-max-messages", int, 80),
    ("--compaction-keep-messages", int, 10),
    ("--reasoning-effort", str, None),
    ("--temperature", float, 1.0),
    ("--top-p", float, 0.95),
    ("--top-k", int, 20),
    ("--min-p", float, 0.0),
    ("--presence-penalty", float, 0.0),
    ("--repetition-penalty", float, 1.0),
    ("--max-episode-steps", int, 10000),
    ("--planner-timeout-s", int, 7200),
)


def option_name(flag: str) -> str:
    """Return the argument destination for a rollout flag."""
    return flag[2:].replace("-", "_")


def planner_defaults() -> dict:
    """Keep standalone simulator defaults local to the RL robot."""
    return {
        option_name(flag): default
        for flag, _, default in OPTIONS
        if flag in {"--max-tokens", "--context-window", "--compaction-fraction"}
    }


def add_options(parser: argparse.ArgumentParser) -> None:
    """Add the flags that can override the standalone rollout defaults."""
    for flag, kind, _ in OPTIONS:
        choices = (
            REASONING_EFFORTS
            if flag == "--reasoning-effort"
            else (1024, 512, 256)
            if flag == "--image_size"
            else None
        )
        parser.add_argument(flag, type=kind, choices=choices)
    thinking = parser.add_mutually_exclusive_group()
    thinking.add_argument(
        "--no-thinking", "--no-reasoning", dest="no_thinking", action="store_true"
    )
    thinking.add_argument(
        "--thinking", "--reasoning", dest="no_thinking", action="store_false"
    )
    parser.set_defaults(no_thinking=None)


def resolve_options(args: argparse.Namespace) -> None:
    """Apply environment and model defaults underneath explicit arguments."""
    model = os.environ.get("RPENT_MODEL", "").lower()
    for flag, kind, default in OPTIONS:
        name = option_name(flag)
        if getattr(args, name) is not None:
            continue
        if flag == "--reasoning-effort":
            default = "xhigh" if not model or "qwen3.8" in model else "none"
        value = os.environ.get(f"RPENT_{name.upper()}")
        setattr(args, name, kind(value) if value else default)
    if args.no_thinking is None:
        args.no_thinking = False


def episode_options(args: argparse.Namespace) -> list[str]:
    """Serialize resolved options so a batch keeps the settings it recorded."""
    values = [
        item
        for flag, _, _ in OPTIONS
        for item in (flag, str(getattr(args, option_name(flag))))
    ]
    return [*values, "--no-thinking" if args.no_thinking else "--thinking"]


def main() -> None:
    """Run one simulator episode with the standalone model endpoint."""
    parser = argparse.ArgumentParser(add_help=False)
    add_options(parser)
    args, remaining = parser.parse_known_args()
    if "-h" in remaining or "--help" in remaining:
        os.execv(
            sys.executable,
            [sys.executable, "-m", "rpent.cli.main", "--robot", "rl", "--help"],
        )
    for key in ("RPENT_MODEL", "RPENT_BASE_URL"):
        if not os.environ.get(key):
            parser.error(f"set {key} before starting the rollout")
    os.environ.setdefault("OPENAI_API_KEY", "local")
    resolve_options(args)
    output = os.environ.get("RPENT_OUTPUT_DIR") or str(
        get_repo_root() / "logs" / datetime.now().strftime("rl-%Y%m%d-%H%M%S")
    )
    command = [
        sys.executable,
        "-m",
        "rpent.cli.main",
        "--robot",
        "rl",
        "--planner",
        "api",
        "--model",
        os.environ["RPENT_MODEL"],
        "--base-url",
        os.environ["RPENT_BASE_URL"],
        "--no-strict-tools",
        "--output-dir",
        output,
        *episode_options(args),
    ]
    if sam3 := os.environ.get("RPENT_SAM3_ENDPOINT"):
        command.extend(("--sam3-endpoint", sam3))
    os.execv(sys.executable, [*command, *remaining])


if __name__ == "__main__":
    main()
