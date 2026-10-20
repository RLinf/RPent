# Copyright 2026 The RPent Authors.
# Licensed under the Apache License, Version 2.0 (the "License");

"""Cosmos-specific runtime configuration without official model dependencies."""

import argparse
import os
from pathlib import Path

from rpent.robots.components.cosmos_policy import USES_SESSIONS as USES_SESSIONS
from rpent.robots.components.cosmos_policy.adapter import ADAPTERS as ADAPTERS

OPTION_FIELDS = (
    "cosmos_text_embeddings",
    "cosmos_predict_future",
    "cosmos_cached_instructions_only",
)


def add_arguments(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group(
        "Cosmos Policy local options",
        "For local Cosmos deployment; all options below are optional.\n"
        "Append to your platform/task/planner command (replace example paths):\n"
        "  --wam-backend cosmos-policy --wam-checkpoint /models/cosmos \\\n"
        "    --wam-python /envs/cosmos/bin/python --wam-root /src/cosmos\n"
        "COSMOS_POLICY_PYTHON and COSMOS_POLICY_ROOT provide path defaults.",
    )
    group.add_argument(
        "--cosmos-text-embeddings",
        metavar="PATH",
        help="T5 embedding cache (default: checkpoint's libero_t5_embeddings.pkl)",
    )
    group.add_argument(
        "--cosmos-predict-future",
        action="store_true",
        help="Enable Cosmos future-state/value prediction",
    )
    group.add_argument(
        "--cosmos-cached-instructions-only",
        action="store_true",
        help="Use only cached Cosmos instructions",
    )
    # Existing launch commands retain their aliases without crowding --help.
    for field in OPTION_FIELDS:
        group.add_argument(
            "--" + field.replace("cosmos_", "wam_", 1).replace("_", "-"),
            dest=field,
            action="store" if field == "cosmos_text_embeddings" else "store_true",
            default=argparse.SUPPRESS,
            help=argparse.SUPPRESS,
        )


def validate_options(args: argparse.Namespace) -> None:
    """Resolve Cosmos environment defaults and require its working directory."""
    args.wam_python = args.wam_python or os.getenv("COSMOS_POLICY_PYTHON")
    args.wam_root = args.wam_root or os.getenv("COSMOS_POLICY_ROOT")
    if not args.wam_root:
        raise ValueError("--wam-root must reference the provisioned Cosmos environment")


def worker_arguments(args: argparse.Namespace) -> list[str]:
    """Return Cosmos-specific worker arguments."""
    result = []
    for field, option in (
        ("cosmos_predict_future", "--predict-future"),
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
