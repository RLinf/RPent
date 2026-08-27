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
from pathlib import Path

from rpent.memory import MemoryManager
from rpent.utils.config import get_memory_dir


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="rpent-memory")
    parser.add_argument(
        "--memory-dir",
        type=Path,
        default=get_memory_dir("libero"),
        help="Local corpus root (default: resources/libero/memory).",
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
    print(index)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
