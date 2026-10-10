# Copyright 2026 The RPent Authors.
# Licensed under the Apache License, Version 2.0 (the "License");

"""Fast-WAM-specific runtime configuration for registered platform pairs."""

import argparse
from pathlib import Path

from rpent.robots.components.fast_wam import USES_SESSIONS as USES_SESSIONS
from rpent.robots.components.fast_wam.adapter import ADAPTERS as ADAPTERS

OPTION_FIELDS = ("wam_config", "wam_dataset_stats")


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--wam-config", help="Resolved Fast-WAM model/processor YAML")
    parser.add_argument(
        "--wam-dataset-stats", help="Fast-WAM checkpoint dataset statistics"
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
        str(Path(args.wam_config).expanduser().resolve()),
        "--dataset-stats",
        str(Path(args.wam_dataset_stats).expanduser().resolve()),
    ]
