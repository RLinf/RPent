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

"""Cosmos-specific runtime configuration without official model dependencies."""

import argparse
import os
from pathlib import Path

from rpent.robots.components.cosmos_policy import USES_SESSIONS as USES_SESSIONS
from rpent.robots.components.cosmos_policy.adapter import ADAPTERS as ADAPTERS

OPTION_FIELDS = (
    "cosmos_text_embeddings",
    "predict_future",
    "cosmos_cached_instructions_only",
)


def add_arguments(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group(
        "Cosmos Policy local options",
        "For local Cosmos deployment; all options below are optional.\n"
        "Append to your platform/task/planner command (replace example paths):\n"
        "  --am-backend cosmos-policy --am-checkpoint /models/cosmos \\\n"
        "    --am-python /envs/cosmos/bin/python --am-root /src/cosmos\n"
        "COSMOS_POLICY_PYTHON and COSMOS_POLICY_ROOT provide path defaults.",
    )
    group.add_argument(
        "--cosmos-text-embeddings",
        metavar="PATH",
        help="T5 embedding cache (default: checkpoint's libero_t5_embeddings.pkl)",
    )
    group.add_argument(
        "--predict-future",
        action="store_true",
        help="Enable future-state/value outputs (currently supported by Cosmos Policy)",
    )
    group.add_argument(
        "--cosmos-cached-instructions-only",
        action="store_true",
        help="Use only cached Cosmos instructions",
    )
    # Existing launch commands retain their aliases without crowding --help.
    for alias, field in (
        ("--wam-text-embeddings", "cosmos_text_embeddings"),
        ("--cosmos-predict-future", "predict_future"),
        ("--wam-predict-future", "predict_future"),
        ("--wam-cached-instructions-only", "cosmos_cached_instructions_only"),
    ):
        group.add_argument(
            alias,
            dest=field,
            action="store" if field == "cosmos_text_embeddings" else "store_true",
            default=argparse.SUPPRESS,
            help=argparse.SUPPRESS,
        )


def validate_options(args: argparse.Namespace) -> None:
    """Resolve Cosmos environment defaults and require its working directory."""
    checkpoint = Path(args.am_checkpoint).expanduser().resolve()
    if not checkpoint.exists():
        raise ValueError(
            "--am-checkpoint must reference a local Cosmos Policy checkpoint"
        )
    args.am_checkpoint = str(checkpoint)
    args.am_python = args.am_python or os.getenv("COSMOS_POLICY_PYTHON")
    args.am_root = args.am_root or os.getenv("COSMOS_POLICY_ROOT")
    if not args.am_root:
        raise ValueError("--am-root must reference the provisioned Cosmos environment")


def worker_arguments(args: argparse.Namespace) -> list[str]:
    """Return Cosmos-specific worker arguments."""
    result = []
    for field, option in (
        ("predict_future", "--predict-future"),
        ("cosmos_cached_instructions_only", "--cached-instructions-only"),
    ):
        if getattr(args, field, False):
            result.append(option)
    if args.cosmos_text_embeddings:
        result += [
            "--text-embeddings",
            str(Path(args.cosmos_text_embeddings).expanduser().absolute()),
        ]
    return result
