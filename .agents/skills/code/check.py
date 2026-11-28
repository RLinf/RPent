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

"""Check changed RPent code: lint, package layout, headers, test layout, shell syntax.

Usage::

    python3 .agents/skills/code/check.py            # files changed against main
    python3 .agents/skills/code/check.py FILE ...

``pre-commit`` runs Ruff with ``--fix``, so it may rewrite files in place; run
the check again after it does.
"""

from __future__ import annotations

import datetime
import re
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import route  # noqa: E402

HEADER = re.compile(r"# Copyright (\d{4}) The RPent Authors\.")
PACKAGE_ROOTS = ("rpent", "robots", "components")
STANDALONE_SCRIPT_DIRS = frozenset({"robots/dual_franka/calibration"})


def check_pre_commit(paths: list[str]) -> list[route.Finding]:
    """Run pre-commit (or ruff fallback) on the changed files."""
    if shutil.which("pre-commit"):
        result = subprocess.run(
            ["pre-commit", "run", "--files", *paths],
            cwd=route.REPO_ROOT,
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            return []
        print(result.stdout + result.stderr)
        return [
            route.Finding(
                "major",
                ".pre-commit-config.yaml",
                None,
                "code/pre-commit",
                "pre-commit failed on the changed files (output above); fix what it reports",
            )
        ]

    py_paths = [p for p in paths if p.endswith(".py")]
    if py_paths and shutil.which("ruff"):
        findings: list[route.Finding] = []
        check_res = subprocess.run(
            ["ruff", "check", "--preview", *py_paths],
            cwd=route.REPO_ROOT,
            capture_output=True,
            text=True,
        )
        if check_res.returncode != 0:
            print(check_res.stdout + check_res.stderr)
            findings.append(
                route.Finding(
                    "major",
                    "pyproject.toml",
                    None,
                    "code/pre-commit",
                    "ruff check --preview failed on changed Python files (output above)",
                )
            )
        fmt_res = subprocess.run(
            ["ruff", "format", "--check", *py_paths],
            cwd=route.REPO_ROOT,
            capture_output=True,
            text=True,
        )
        if fmt_res.returncode != 0:
            print(fmt_res.stdout + fmt_res.stderr)
            findings.append(
                route.Finding(
                    "major",
                    "pyproject.toml",
                    None,
                    "code/pre-commit",
                    "ruff format --check failed on changed Python files (output above)",
                )
            )
        return findings

    print(
        "skipped: pre-commit (and ruff) not installed (uv pip install pre-commit==4.6.2)"
    )
    return []


def check_init(root: Path | None = None) -> list[route.Finding]:
    """Every directory with Python files under rpent/, robots/, components/ needs __init__.py."""
    base = root or route.REPO_ROOT
    findings: list[route.Finding] = []
    for pkg in PACKAGE_ROOTS:
        pkg_dir = base / pkg
        if not pkg_dir.is_dir():
            continue
        for directory in sorted(
            [pkg_dir, *(p for p in pkg_dir.rglob("*") if p.is_dir())]
        ):
            if "__pycache__" in directory.parts:
                continue
            rel_dir = directory.relative_to(base).as_posix()
            if rel_dir in STANDALONE_SCRIPT_DIRS:
                continue
            if (
                any(directory.glob("*.py"))
                and not (directory / "__init__.py").is_file()
            ):
                findings.append(
                    route.Finding(
                        "major",
                        rel_dir,
                        None,
                        "code/init",
                        "has Python files but no __init__.py",
                    )
                )
    return findings


def check_header(path: str, root: Path | None = None) -> list[route.Finding]:
    """Check Apache-2.0 RPent copyright header on a new Python file."""
    base = root or route.REPO_ROOT
    head = (base / path).read_text(encoding="utf-8").splitlines()[:5]
    years = [m[1] for line in head if (m := HEADER.match(line))]
    if not years:
        return [
            route.Finding(
                "major",
                path,
                1,
                "code/license",
                "new file without '# Copyright <year> The RPent Authors.'; "
                "a vendored file keeps its upstream header",
            )
        ]
    year = str(datetime.date.today().year)
    if years[0] != year:
        return [
            route.Finding(
                "minor",
                path,
                1,
                "code/license",
                f"new file carries {years[0]}; use {year} unless it moved from another path",
            )
        ]
    return []


def check_test_layout(change: route.Change) -> list[route.Finding]:
    """New unit tests should mirror module layout instead of sitting directly in tests/unit_tests/."""
    if change.status == "A" and re.fullmatch(
        r"tests/unit_tests/test_\w+\.py", change.path
    ):
        return [
            route.Finding(
                "minor",
                change.path,
                None,
                "code/test-layout",
                "new test file directly under tests/unit_tests/; mirror the component "
                "under tests/unit_tests/{rpent,robots,components,skills}/",
            )
        ]
    return []


def check_shell_syntax(path: str, root: Path | None = None) -> list[route.Finding]:
    """Verify changed shell scripts parse cleanly under bash -n."""
    base = root or route.REPO_ROOT
    result = subprocess.run(
        ["bash", "-n", path], cwd=base, capture_output=True, text=True
    )
    if result.returncode == 0:
        return []
    return [
        route.Finding(
            "critical", path, None, "code/shell-syntax", result.stderr.strip()
        )
    ]


def main(argv: list[str] | None = None) -> int:
    args = route.check_parser(__doc__.splitlines()[0]).parse_args(argv)
    changes = route.select("code", args).changes
    present = [c for c in changes if c.status != "D"]
    findings = check_pre_commit([c.path for c in present]) if present else []
    if any(c.path.startswith(tuple(f"{r}/" for r in PACKAGE_ROOTS)) for c in changes):
        findings += check_init()
    for change in present:
        if change.status == "A" and change.path.endswith(".py"):
            findings += check_header(change.path)
        findings += check_test_layout(change)
        if change.path.endswith(".sh"):
            findings += check_shell_syntax(change.path)
    return route.report("code", findings)


if __name__ == "__main__":
    raise SystemExit(main())
