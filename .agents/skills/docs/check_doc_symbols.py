#!/usr/bin/env python3
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

"""Check that every identifier RPent development docs put in code font exists in code.

The Sphinx build checks markup and ``:doc:`` targets, not whether an API class,
hook, or module name still exists. This script scans single-word inline literals
on Concepts & Development pages and reports names that appear nowhere in the
shipped codebase.

Genuinely illustrative names (such as ``MyRobot`` or ``my_robot`` in an extension
tutorial) go in ``ALLOWED`` below or in ``--allow``.

Usage::

    python3 .agents/skills/docs/check_doc_symbols.py           # all development docs
    python3 .agents/skills/docs/check_doc_symbols.py FILE...   # subset
    python3 .agents/skills/docs/check_doc_symbols.py --allow Foo

Exit code is 1 when any unknown name is reported, 0 otherwise.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path


def repo_root() -> Path:
    """The RPent checkout containing the working directory, or this script's repo."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            capture_output=True,
            text=True,
            check=True,
        )
        return Path(out.stdout.strip())
    except (subprocess.CalledProcessError, FileNotFoundError):
        return Path(__file__).resolve().parents[3]


ROOT = repo_root()

#: Pages that document RPent's own APIs and extension contracts.
DOC_ROOTS = tuple(f"docs/source-{lang}/rst_source/development" for lang in ("en", "zh"))

#: Shipped code, tests, configs, and docs config where valid names may appear.
CODE_ROOTS = (
    "rpent",
    "robots",
    "components",
    "tests",
    "pyproject.toml",
    ".github/workflows",
    "docs/source-en/conf.py",
    "docs/source-zh/conf.py",
)

CODE_SUFFIXES = {".py", ".yaml", ".yml", ".sh", ".toml", ".cfg", ".txt", ".md", ".json"}

#: Illustrative names invented in tutorials and snippets.
ALLOWED: set[str] = {
    "MyEnvFacade",
    "MyPlanner",
    "MyRobot",
    "MyRobotPrimitives",
    "MyToolkit",
    "execute_action_chunk_of_length_20",  # counter-example tool name in add_primitive.rst
    "my_planner",
    "my_robot",
    "my_tool",
    "myrobot",
}

#: External project identifiers referenced in integration pages.
EXTERNAL: set[str] = set()

#: Known drift waiting to be fixed in docs; delete entries once fixed.
KNOWN_DRIFT: set[str] = set()

IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z_0-9]*$")
LITERAL = re.compile(r"``([^`\n]+)``")
WORD = re.compile(r"[A-Za-z_][A-Za-z_0-9]*")


def vocabulary(root: Path) -> set[str]:
    """Every word spelled in RPent's shipped code and configuration."""
    known: set[str] = set()
    for entry in CODE_ROOTS:
        target = root / entry
        if target.is_file():
            files = [target]
        elif target.is_dir():
            files = [
                path
                for path in target.rglob("*")
                if path.is_file() and path.suffix in CODE_SUFFIXES
            ]
        else:
            continue
        for path in files:
            try:
                known.update(WORD.findall(path.read_text(encoding="utf-8")))
            except (UnicodeDecodeError, OSError):
                continue
    return known


def literals(path: Path) -> list[tuple[int, str]]:
    """Every single-word inline literal on a page, with its 1-based line number."""
    found = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        for token in LITERAL.findall(line):
            token = token.strip()
            if IDENTIFIER.match(token):
                found.append((number, token))
    return found


def collect(targets: list[str], root: Path | None = None) -> list[Path]:
    base = root or ROOT
    paths: list[Path] = []
    for target in targets:
        candidate = Path(target)
        if not candidate.is_absolute():
            candidate = base / candidate
        if candidate.is_dir():
            paths.extend(sorted(candidate.rglob("*.rst")))
        elif candidate.is_file():
            paths.append(candidate)
        else:
            print(f"warning: {target} does not exist", file=sys.stderr)
    return paths


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "targets",
        nargs="*",
        default=list(DOC_ROOTS),
        help="RST files or directories (default: development docs)",
    )
    parser.add_argument(
        "--allow",
        action="append",
        default=[],
        metavar="NAME",
        help="a name that is illustrative rather than real; repeatable",
    )
    args = parser.parse_args(argv)

    root = repo_root()
    known = vocabulary(root) | ALLOWED | EXTERNAL | KNOWN_DRIFT | set(args.allow)
    pages = collect(args.targets or list(DOC_ROOTS), root)

    unknown: list[str] = []
    for page in pages:
        rel = page.relative_to(root) if page.is_relative_to(root) else page
        for number, token in literals(page):
            if token not in known:
                unknown.append(f"{rel}:{number}: {token}")

    for line in unknown:
        print(line)

    if unknown:
        print(
            f"\n{len(unknown)} name(s) in code font that the repository does not "
            "define. Rename, delete, or add to ALLOWED if illustrative.",
            file=sys.stderr,
        )
        return 1
    print(f"Every name in code font across {len(pages)} pages exists in the code.")
    if KNOWN_DRIFT:
        print(
            f"{len(KNOWN_DRIFT)} name(s) are known drift waiting to be fixed: "
            f"{', '.join(sorted(KNOWN_DRIFT))}."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
