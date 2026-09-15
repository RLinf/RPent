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

"""Convert the reviewed PR #130 memory package to the current RoboCasa layout."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from robots.robocasa.memory import GLOBAL_FILE, TaskMemory, task_files


def migrate(source: Path, output: Path) -> dict:
    """Copy the 103 reviewed files byte-for-byte and emit their hash manifest."""
    source = source.resolve()
    output = output.resolve()
    if output == source or output.is_relative_to(source):
        raise ValueError("output must be outside the source package")
    files: dict[str, bytes] = {GLOBAL_FILE: (source / "GLOBAL_MEMORY.md").read_bytes()}
    audits = sorted((source / "task").glob("*_s0.json"))
    markdown = sorted((source / "task").glob("*.md"))
    recipes = sorted((source / "task").glob("recipe_*_s0.jsonl"))
    if (len(audits), len(recipes), len(markdown)) != (43, 43, 16):
        raise ValueError("expected 43 JSON/JSONL pairs and 16 task Markdown files")
    for path in audits:
        task = path.name.removesuffix("_s0.json")
        names = task_files(task)
        audit = json.loads(path.read_text())
        if (
            audit.get("task") != task
            or audit.get("seed") != 0
            or audit.get("success") is not True
        ):
            raise ValueError(f"invalid reviewed seed-0 audit: {path.name}")
        recipe = source / "task" / f"recipe_{task}_s0.jsonl"
        lines = recipe.read_text().splitlines()
        if not lines or any(not isinstance(json.loads(line), dict) for line in lines):
            raise ValueError(f"invalid recipe JSONL: {recipe.name}")
        files[names[0]] = path.read_bytes()
        files[names[1]] = recipe.read_bytes()
    for path in markdown:
        files[task_files(path.stem)[2]] = path.read_bytes()
    if len(files) != 103:
        raise ValueError("expected exactly 103 memory files")
    manifest = {
        "schema_version": 1,
        "source_package": "pr130_robocasa_local_changes_package_20260901",
        "files": {
            name: hashlib.sha256(data).hexdigest()
            for name, data in sorted(files.items())
        },
    }
    if output.exists() and any(output.iterdir()):
        raise ValueError("output directory must be empty; do not overlay old memory")
    for name, data in files.items():
        path = output / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    (output / "CORPUS.json").write_text(json.dumps(manifest, indent=2) + "\n")
    TaskMemory.load(output, "StirVegetables")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        required=True,
        help="Package memory directory containing task/ and GLOBAL_MEMORY.md",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Empty destination robocasa/ directory",
    )
    args = parser.parse_args()
    manifest = migrate(args.source, args.output)
    print(f"Verified {len(manifest['files'])} memory files in {args.output}")


if __name__ == "__main__":
    main()
