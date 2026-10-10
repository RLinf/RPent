# Copyright 2026 The RPent Authors.
# Licensed under the Apache License, Version 2.0 (the "License");

"""Cosmos-specific runtime configuration without official model dependencies."""

import argparse
import os
from pathlib import Path

from rpent.robots.components.cosmos_policy import USES_SESSIONS as USES_SESSIONS
from rpent.robots.components.cosmos_policy.adapter import ADAPTERS as ADAPTERS

OPTION_FIELDS = (
    "wam_text_embeddings",
    "wam_predict_future",
    "wam_cached_instructions_only",
)


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--wam-text-embeddings", help="Cosmos T5 embedding cache")
    parser.add_argument(
        "--wam-predict-future",
        action="store_true",
        help="Enable Cosmos future-state/value prediction",
    )
    parser.add_argument(
        "--wam-cached-instructions-only",
        action="store_true",
        help="Use only cached Cosmos instructions",
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
        ("wam_predict_future", "--predict-future"),
        ("wam_cached_instructions_only", "--cached-instructions-only"),
    ):
        if getattr(args, field, False):
            result.append(option)
    if args.wam_text_embeddings:
        result += [
            "--text-embeddings",
            str(Path(args.wam_text_embeddings).expanduser().absolute()),
        ]
    return result
