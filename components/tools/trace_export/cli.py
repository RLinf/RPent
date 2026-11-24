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

"""Command-line export of recorded code-policy traces to dataset rows."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from components.tools.trace_export.export import export
from components.tools.trace_export.trace import TraceError
from rpent.utils.logging import get_logger

logger = get_logger("trace_export")


def _parse_methods(raw: Sequence[str] | None) -> list[str] | None:
    if raw is None:
        return None
    return [part.strip() for item in raw for part in item.split(",")]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m components.tools.trace_export",
        description=(
            "Validate recorded code-policy traces and export them as model-agnostic "
            "dataset rows. An episode is exported whole or quarantined whole."
        ),
    )
    parser.add_argument(
        "sources",
        metavar="SOURCE",
        type=Path,
        nargs="+",
        help=(
            "Episode directory or parent directory to search recursively for "
            "trace v1 episodes."
        ),
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        required=True,
        help="New output directory for rows.jsonl, images/ and export-audit.json.",
    )
    parser.add_argument(
        "--robo-method",
        action="append",
        default=None,
        help=(
            "Fallback public robo method name(s) used when episode.json omits "
            "robo_methods. Accepts comma-separated values or repeated flags."
        ),
    )
    parser.add_argument(
        "--include-unsuccessful",
        action="store_true",
        help=(
            "Also export episodes without an environment-confirmed success, "
            "marked with environment_success=false."
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the exporter and return the process exit status.

    Args:
        argv: Command-line arguments, or ``None`` to read ``sys.argv``.

    Returns:
        0 when every episode was exported, 1 when the export failed, nothing was
        exported, or any episode was quarantined.
    """
    args = _parser().parse_args(argv)
    try:
        audit = export(
            args.sources,
            args.output,
            _parse_methods(args.robo_method),
            allow_unsuccessful=args.include_unsuccessful,
        )
    except (TraceError, FileExistsError) as exc:
        logger.error("Trace export failed: %s", exc)
        return 1
    logger.info(
        "Exported %d rows from %d episodes to %s",
        audit["rows"],
        audit["accepted_episodes"],
        args.output,
    )
    for item in audit["quarantined_episodes"]:
        logger.warning("Quarantined %s: %s", item["episode"], item["reason"])
    return 0 if audit["rows"] and not audit["quarantined_episodes"] else 1
