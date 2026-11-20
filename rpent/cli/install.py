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

"""``rpent-install``: inspect, check, install, and diagnose components.

Every environment is described by the components named on the command line::

    rpent-install install robot-alpha --venv .venv-alpha
    rpent-install install infra-dev --venv .venv-dev

Repository checks and CI selection live under ``tests/tooling/``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from rpent.install.compile import resolve_resolution
from rpent.install.discovery import discover
from rpent.install.doctor import FAIL, check_installation, inspect_installation
from rpent.install.errors import InstallConfigError, InstallError
from rpent.install.fingerprint import current_venv
from rpent.install.installer import find_uv, install, resolve_targets
from rpent.install.requirements import (
    read_core_dependencies,
    read_planner_dependencies,
)
from rpent.install.resolve import Resolution, choose_python
from rpent.utils.config import get_repo_root


def command_list(args: argparse.Namespace) -> int:
    """Print every component and what it requires."""
    catalog = discover(args.root)
    headings = {
        "robot": "Robots",
        "sim": "Simulators",
        "model": "Models",
        "tool": "Tools",
        "infra": "Infrastructure",
    }
    for kind, heading in headings.items():
        components = sorted(
            (
                component
                for component in catalog.components.values()
                if component.kind == kind
            ),
            key=lambda component: component.name,
        )
        if not components:
            continue
        print(heading)
        for component in components:
            requires = ", ".join(component.requires) or "-"
            print(f"  {component.name:<20} requires {requires}")
        print()
    return 0


def _report_resolution(
    label: str,
    resolution: Resolution,
    *,
    root: Path,
    uv: str | None,
) -> bool:
    """Print one resolution's verdict and return whether it failed."""
    if not resolution.ok:
        print(f"{label}: FAIL")
        for error in resolution.errors:
            print(f"  {error}")
        return True
    count = len(resolution.components)
    noun = "component" if count == 1 else "components"
    if uv is None:
        print(f"{label}: ok, {count} {noun}")
        return False
    python = choose_python(resolution.components.values())
    try:
        resolved_text = resolve_resolution(
            resolution,
            root=root,
            uv=uv,
            python=python,
            core_dependencies=read_core_dependencies(root),
            planner_dependencies=read_planner_dependencies(root),
        )
    except InstallError as exc:
        print(f"{label}: FAIL")
        print(f"  {exc}")
        return True
    pinned = sum(
        1
        for line in resolved_text.splitlines()
        if line and not line.startswith((" ", "#"))
    )
    print(f"{label}: ok, {count} {noun}, uv resolved {pinned} packages")
    return False


def command_check(args: argparse.Namespace) -> int:
    """Resolve component environments without installing anything.

    Several components on one command line describe the single environment
    ``install`` would create from them; ``--all`` checks every component's own
    environment.
    """
    catalog = discover(args.root)
    names = (
        sorted(catalog.components) if args.all else list(dict.fromkeys(args.targets))
    )
    if not names:
        print("pass a component name or --all", file=sys.stderr)
        return 2
    unknown = [name for name in names if name not in catalog.components]
    for name in unknown:
        known = ", ".join(sorted(catalog.components))
        print(f"unknown component {name!r}; known components: {known}", file=sys.stderr)
    failed = bool(unknown)
    uv = find_uv() if args.resolve else None
    targets = [name for name in names if name in catalog.components]
    common = {"root": args.root, "uv": uv}

    def resolve_names(names: list[str]) -> Resolution:
        return resolve_targets(names, catalog, root=args.root)

    if len(targets) > 1 and not args.all:
        # Naming several components on one command line asks for the
        # environment ``install`` would create from them, so they resolve
        # together: components that each resolve alone may still refuse to
        # share an environment. ``--all`` keeps asking about each component's
        # own environment, which is also the unit the CI matrix installs.
        resolution = resolve_names(targets)
        failed |= _report_resolution(" ".join(targets), resolution, **common)
        if not resolution.ok:
            # Attribute the failure to the manifests that need changing.
            for name in targets:
                failed |= _report_resolution(name, resolve_names([name]), **common)
        return 1 if failed else 0

    for name in targets:
        failed |= _report_resolution(name, resolve_names([name]), **common)
    return 1 if failed else 0


def command_install(args: argparse.Namespace) -> int:
    """Create or reuse a virtualenv and install the named components."""
    install(
        args.targets,
        root=args.root,
        venv=args.venv,
        dry_run=args.dry_run,
        force=args.force,
        verbose=True,
    )
    return 0


def command_doctor(args: argparse.Namespace) -> int:
    """Check the current or selected virtualenv."""
    venv = args.venv or current_venv()
    if venv is None:
        print(
            "no active virtualenv; activate one or pass --venv",
            file=sys.stderr,
        )
        return 2
    if args.fingerprint:
        report = inspect_installation(venv)
        if report is None:
            print(f"{venv} has no installation fingerprint", file=sys.stderr)
            return 1
        print(f"venv: {report['venv']}")
        print(f"sha256: {report['sha256']}")
        for key in sorted(report["fingerprint"]):
            value = report["fingerprint"][key]
            if isinstance(value, dict):
                print(f"{key}:")
                for nested, nested_value in sorted(value.items()):
                    print(f"  {nested}: {nested_value}")
            else:
                print(f"{key}: {value}")
        return 0
    checks = check_installation(venv, root=args.root)
    for check in checks:
        print(f"{check.status:<4} {check.message}")
    return 1 if any(check.status == FAIL for check in checks) else 0


def build_parser() -> argparse.ArgumentParser:
    """Return the argument parser for ``rpent-install``."""
    parser = argparse.ArgumentParser(
        prog="rpent-install",
        description="Install and diagnose RPent component environments.",
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=None,
        help="RPent checkout to read manifests from; defaults to this checkout.",
    )
    subcommands = parser.add_subparsers(dest="command", required=True)

    listing = subcommands.add_parser("list", help="list components")
    listing.set_defaults(handler=command_list)

    check = subcommands.add_parser(
        "check",
        help="check environments without installing",
        description=(
            "Named components are checked as the one environment `install` "
            "would create from them; --all checks each component's own "
            "environment instead."
        ),
    )
    check.add_argument(
        "targets",
        nargs="*",
        metavar="COMPONENT",
        help="components that form one environment",
    )
    check.add_argument(
        "--all",
        action="store_true",
        help="check every component in an environment of its own",
    )
    check.add_argument(
        "--resolve",
        action="store_true",
        help="also compile each environment with uv to prove it resolves",
    )
    check.set_defaults(handler=command_check)

    installer = subcommands.add_parser(
        "install",
        help="install one or more components and their requirements into a virtualenv",
    )
    installer.add_argument(
        "targets",
        nargs="+",
        metavar="COMPONENT",
        help="component names; the environment also contains their closure",
    )
    installer.add_argument("--venv", type=Path, help="virtualenv path")
    installer.add_argument(
        "--dry-run", action="store_true", help="print requirements only"
    )
    installer.add_argument(
        "--force",
        action="store_true",
        help=(
            "replace the virtualenv at --venv when it holds other components "
            "or was not created by rpent-install"
        ),
    )
    installer.set_defaults(handler=command_install)

    doctor = subcommands.add_parser("doctor", help="diagnose an installed venv")
    doctor.add_argument("--venv", type=Path)
    doctor.add_argument(
        "--fingerprint",
        action="store_true",
        help="print the stored installation fingerprint instead of checking",
    )
    doctor.set_defaults(handler=command_doctor)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run ``rpent-install`` and return a process exit code."""
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.root is None:
        args.root = get_repo_root()
        # A wheel install puts this package in site-packages, where none of the
        # manifests the installer reads exist. Say so instead of failing on a
        # path the user never named.
        if not (args.root / "pyproject.toml").is_file():
            parser.error(
                f"{args.root} is not an RPent checkout; rpent-install reads "
                "component manifests from a source checkout. Run it from one or "
                "pass --root"
            )
    try:
        return args.handler(args)
    except (InstallError, InstallConfigError) as exc:
        print(str(exc), file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
