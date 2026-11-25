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

"""Reproduce the Read the Docs build gate locally, for both languages.

``docs/source-en/.readthedocs.yaml`` and ``docs/source-zh/.readthedocs.yaml``
both set ``fail_on_warning: true``, so any Sphinx warning fails the Read the
Docs check. The two projects build independently, so a warning triggered only
by a Chinese page breaks the Chinese build while the English build stays green.
Always build both.

This script reuses the current Python environment when Sphinx and RPent's docs
extensions are already installed, or provisions a docs-only virtualenv from
``docs/requirements.txt``, builds each language, and reports every warning.

Usage::

    python3 .agents/skills/docs/build_docs.py             # en + zh
    python3 .agents/skills/docs/build_docs.py --lang zh
    python3 .agents/skills/docs/build_docs.py --keep-output

Exit code is 1 when either language reports a warning, 0 otherwise.
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REQUIRED_DOCS_MODULES = (
    "sphinx",
    "pydata_sphinx_theme",
    "sphinx_copybutton",
    "sphinx_design",
    "sphinx_reredirects",
    "sphinx_sitemap",
)
# A warning record starts at "<path>[:<line>]: WARNING" or a bare "WARNING:".
RECORD_START = re.compile(r"^(?:\S+: )?(?:WARNING|ERROR|SEVERE)[:\s]")


def repo_root() -> Path:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            capture_output=True,
            text=True,
            check=True,
        )
        return Path(out.stdout.strip())
    except (subprocess.CalledProcessError, FileNotFoundError):
        return Path.cwd()


def _has_docs_stack(python: Path) -> bool:
    if not python.exists():
        return False
    probe = subprocess.run(
        [
            str(python),
            "-c",
            "; ".join(f"import {mod}" for mod in REQUIRED_DOCS_MODULES),
        ],
        capture_output=True,
    )
    return probe.returncode == 0


def ensure_venv(venv: Path, requirements: Path, *, explicit: bool = False) -> Path:
    """Return a Python with RPent's Sphinx stack, provisioning ``venv`` if needed."""
    python = venv / "bin" / "python"
    if _has_docs_stack(python):
        return python
    if not explicit:
        current = Path(sys.executable)
        if _has_docs_stack(current):
            return current

    if not python.exists():
        print(f"[build_docs] creating {venv}")
        if shutil.which("uv"):
            subprocess.run(["uv", "venv", str(venv), "--python", "3.11"], check=True)
        else:
            subprocess.run([sys.executable, "-m", "venv", str(venv)], check=True)

    print(f"[build_docs] installing {requirements.name}")
    env = {**os.environ, "UV_HTTP_TIMEOUT": "180"}
    if shutil.which("uv"):
        subprocess.run(
            ["uv", "pip", "install", "--python", str(python), "-r", str(requirements)],
            check=True,
            env=env,
        )
    else:
        subprocess.run(
            [str(python), "-m", "pip", "install", "-q", "-r", str(requirements)],
            check=True,
        )
    return python


def split_records(stderr: str) -> list[str]:
    """Group Sphinx's stderr into one string per warning."""
    records: list[str] = []
    for line in stderr.splitlines():
        if RECORD_START.match(line) or not records:
            records.append(line)
        else:
            records[-1] += "\n" + line
    return [r for r in records if r.strip()]


def build(python: Path, source: Path, output: Path) -> list[str]:
    """Build one language and return the Sphinx warnings."""
    result = subprocess.run(
        [
            str(python),
            "-m",
            "sphinx",
            "-b",
            "html",
            "-E",
            "-N",
            "-W",
            "--keep-going",
            "-q",
            str(source),
            str(output),
        ],
        capture_output=True,
        text=True,
        env={**os.environ, "LC_ALL": "C.UTF-8", "LANG": "C.UTF-8"},
    )
    records = [r for r in split_records(result.stderr) if RECORD_START.match(r)]
    if result.returncode != 0 and not records:
        detail = (result.stderr.strip() or result.stdout.strip())[-2000:]
        records.append(
            f"sphinx-build exited {result.returncode}\n{detail}"
            if detail
            else f"sphinx-build exited {result.returncode}"
        )
    return records


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--lang", choices=("en", "zh", "both"), default="both")
    parser.add_argument(
        "--venv",
        type=Path,
        default=os.environ.get("DOCS_VENV"),
        help="virtualenv to build in (default: active Python if ready, else <repo>/.docs-venv)",
    )
    parser.add_argument(
        "--keep-output", action="store_true", help="keep the generated HTML"
    )
    args = parser.parse_args(argv)

    root = repo_root()
    explicit = args.venv is not None
    venv = args.venv or root / ".docs-venv"
    python = ensure_venv(venv, root / "docs" / "requirements.txt", explicit=explicit)

    languages = ("en", "zh") if args.lang == "both" else (args.lang,)
    destination = (
        root / "docs" / "build" / "check"
        if args.keep_output
        else Path(tempfile.mkdtemp(prefix="rpent-docs-"))
    )

    total = 0
    for language in languages:
        source = root / "docs" / f"source-{language}"
        print(f"[build_docs] building {source.name}")
        records = build(python, source, destination / language)
        total += len(records)
        for record in records:
            print(record)
        print(f"[build_docs] {source.name}: {len(records)} warning(s)")

    if not args.keep_output:
        shutil.rmtree(destination, ignore_errors=True)

    if total:
        print(
            f"\n{total} warning(s) — Read the Docs builds with fail_on_warning, "
            "so this fails the docs check.",
            file=sys.stderr,
        )
        return 1
    print("\nBoth builds are clean." if len(languages) > 1 else "\nBuild is clean.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
