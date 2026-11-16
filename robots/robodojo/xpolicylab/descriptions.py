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

"""Save a verified snapshot of the official public task descriptions."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from urllib.request import urlopen

SOURCE = "https://robodojo-benchmark.com/assets/index-bXab2prJ.js"
SHA256 = "9473aa9b6f82833671ecb44dc2021483c0dd9c9c0631e32e68ee2c2e37824069"
PAGE = "https://robodojo-benchmark.com/leaderboard/detail-description"


def extract_descriptions(source: str) -> dict:
    """Extract literal public descriptions without executing downloaded JavaScript."""
    quoted = r'"(?:[^"\\]|\\.)*"'
    pattern = r"\{name:(" + quoted + r"),description:(" + quoted + r"),scoring:"
    entries = {}
    for match in re.finditer(pattern, source):
        name, description = json.loads(match[1]), json.loads(match[2])
        scoring, _ = json.JSONDecoder().raw_decode(source[match.end() :])
        if not isinstance(scoring, list) or not all(
            isinstance(s, str) for s in scoring
        ):
            raise ValueError(f"Invalid public scoring description for {name}")
        if name in entries:
            raise ValueError(f"Duplicate public task description: {name}")
        entries[name] = {"name": name, "description": description, "scoring": scoring}
    return entries


def main() -> None:
    """Download once during installation; evaluation uses the frozen local file."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    with urlopen(SOURCE, timeout=30) as response:
        content = response.read()
    if hashlib.sha256(content).hexdigest() != SHA256:
        raise ValueError(
            "Official asset changed; review the new source before using it"
        )
    entries = extract_descriptions(content.decode())
    if len(entries) != 42:
        raise ValueError("Expected the verified 42-task public description catalog")
    entries["_source"] = {"page": PAGE, "asset": SOURCE, "sha256": SHA256}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(entries, stream, indent=2, ensure_ascii=False)
        stream.write("\n")


if __name__ == "__main__":
    main()
