# Copyright 2026 The RPent Authors.
# Licensed under the Apache License, Version 2.0 (the "License");

"""Fast-WAM-specific runtime configuration for registered platform pairs."""

import argparse
from pathlib import Path

from rpent.robots.components.fast_wam import USES_SESSIONS as USES_SESSIONS
from rpent.robots.components.fast_wam.adapter import ADAPTERS as ADAPTERS

OPTION_FIELDS = ("fast_wam_config", "fast_wam_dataset_stats")


def add_arguments(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group(
        "Fast-WAM local options",
        "For local Fast-WAM deployment; both files below are required.\n"
        "Append to your platform/task/planner command (replace example paths):\n"
        "  --wam-backend fast-wam --wam-checkpoint /models/fast-wam.pt \\\n"
        "    --wam-python /envs/fast-wam/bin/python \\\n"
        "    --fast-wam-config /models/worker.yaml \\\n"
        "    --fast-wam-dataset-stats /models/dataset_stats.json",
    )
    group.add_argument(
        "--fast-wam-config",
        metavar="YAML",
        help="Resolved model/processor YAML (required)",
    )
    group.add_argument(
        "--fast-wam-dataset-stats",
        metavar="JSON",
        help="Checkpoint dataset statistics (required)",
    )
    for field in OPTION_FIELDS:
        group.add_argument(
            "--" + field.removeprefix("fast_").replace("_", "-"),
            dest=field,
            default=argparse.SUPPRESS,
            help=argparse.SUPPRESS,
        )


def validate_options(args: argparse.Namespace) -> None:
    """Require the model configuration and dataset statistics files."""
    for field in OPTION_FIELDS:
        value = getattr(args, field, None)
        if not value or not Path(value).expanduser().is_file():
            raise ValueError(f"--{field.replace('_', '-')} must reference a local file")


def worker_arguments(args: argparse.Namespace) -> list[str]:
    """Return Fast-WAM-specific worker arguments."""
    return [
        "--config",
        str(Path(args.fast_wam_config).expanduser().resolve()),
        "--dataset-stats",
        str(Path(args.fast_wam_dataset_stats).expanduser().resolve()),
    ]
