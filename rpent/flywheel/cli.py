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

"""Command-line entry point for RPent Flywheel."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="rpent-flywheel")
    commands = parser.add_subparsers(dest="command", required=True)

    validate = commands.add_parser("validate", help="validate one raw episode")
    validate.add_argument("episode", type=Path)

    export = commands.add_parser(
        "export-lerobot", help="export successful episodes to LeRobot"
    )
    export.add_argument(
        "--data-root",
        type=Path,
        default=Path("datacollection"),
        help="root containing raw Flywheel episodes",
    )
    export.add_argument("--suite", required=True)
    export.add_argument("--task", type=int, required=True)
    export.add_argument("--dataset-id")
    export.add_argument(
        "--output-root",
        type=Path,
        help="parent directory for the exported LeRobot dataset",
    )

    args = parser.parse_args(argv)
    if args.command == "validate":
        from rpent.flywheel.episode import validate_episode

        result = validate_episode(args.episode)
    else:
        from rpent.flywheel.export import export_lerobot

        result = export_lerobot(
            args.data_root,
            suite=args.suite,
            task_id=args.task,
            dataset_id=args.dataset_id,
            output_root=args.output_root,
        )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
