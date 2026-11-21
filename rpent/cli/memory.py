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

"""Standalone command-line maintenance interface for local memory."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

try:
    from rpent.memory import MemoryManager
except ModuleNotFoundError as exc:
    # The memory store is part of the planner layer, so a core-only environment
    # still parses arguments and explains what to install.
    _MISSING_PLANNER_LAYER: ModuleNotFoundError | None = exc
else:
    _MISSING_PLANNER_LAYER = None


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="rpent-memory")
    parser.add_argument(
        "--memory-dir",
        type=Path,
        required=True,
        help="Local corpus root; point it at the corpus of the robot or "
        "simulator you are maintaining, for example memory/<robot>. The former "
        "implicit default was removed, so pass it explicitly.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    merge = subparsers.add_parser("merge", help="Publish one completed cell.")
    merge.add_argument("--cell", required=True)
    merge.add_argument("--output-dir", type=Path, required=True)
    merge.add_argument(
        "--solved",
        action="store_true",
        help="Mark the cell as solved so its task audit and recipe are published.",
    )
    subparsers.add_parser("validate", help="Validate published memory leaves.")
    subparsers.add_parser("build-index", help="Rebuild MEMORY.md.")
    return parser


def main() -> int:
    args = _parser().parse_args()
    if _MISSING_PLANNER_LAYER is not None:
        print(
            "rpent-memory needs the planner layer, which this environment does "
            f"not provide ({_MISSING_PLANNER_LAYER}); install it with "
            '`pip install "rpent[planner]"`',
            file=sys.stderr,
        )
        return 1
    manager = MemoryManager(args.memory_dir)
    if args.command == "merge":
        result = manager.merge_memory(
            cell_tag=args.cell,
            run_state_dir=args.output_dir,
            solved=args.solved,
        )
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return 0
    if args.command == "validate":
        problems = manager.validate()
        if problems:
            print("\n".join(problems))
            return 1
        print("local memory is valid")
        return 0

    index = manager.rebuild_index()
    if index is None:
        print("no indexable memory leaves; MEMORY.md not regenerated")
    else:
        print(index)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
