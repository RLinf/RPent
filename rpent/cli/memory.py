"""Standalone command-line maintenance interface for local memory."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from rpent.memory.layered import build_index, merge_cell, validate_corpus
from rpent.utils.config import get_memory_dir


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="rpent-memory")
    parser.add_argument("--memory-dir", type=Path, default=get_memory_dir("libero"),
                        help="Local corpus root (default: resources/libero/memory).")
    subparsers = parser.add_subparsers(dest="command", required=True)

    merge = subparsers.add_parser("merge", help="Publish one completed cell.")
    merge.add_argument("--cell", required=True)
    merge.add_argument("--output-dir", type=Path, required=True)
    subparsers.add_parser("validate", help="Validate published memory leaves.")
    subparsers.add_parser("build-index", help="Rebuild MEMORY.md.")
    return parser


def main() -> int:
    args = _parser().parse_args()
    if args.command == "merge":
        result = merge_cell(
            memory_dir=args.memory_dir,
            cell_tag=args.cell,
            output_dir=args.output_dir,
        )
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return 0
    if args.command == "validate":
        problems = validate_corpus(args.memory_dir)
        if problems:
            print("\n".join(problems))
            return 1
        print("local memory is valid")
        return 0
    index = build_index(args.memory_dir)
    print(index)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
