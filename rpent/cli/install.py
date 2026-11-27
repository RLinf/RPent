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

"""``rpent-install``: list, check, install, and diagnose component environments."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from rpent.install.errors import InstallConfigError, InstallError
from rpent.install.installer import (
    FAIL,
    check_installation,
    current_venv,
    find_uv,
    inspect_installation,
    install,
    resolve_targets,
)
from rpent.install.manifest import discover
from rpent.install.resolve import (
    Resolution,
    choose_python,
    read_core_dependencies,
    read_planner_dependencies,
    resolve_resolution,
)
from rpent.utils.config import get_repo_root


def _checkout_root(start: Path) -> Path | None:
    """Return the RPent checkout containing ``start``, if any."""
    for candidate in (start, *start.parents):
        if (candidate / "pyproject.toml").is_file() and (
            (candidate / "components").is_dir() or (candidate / "robots").is_dir()
        ):
            return candidate
    return None


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
            (c for c in catalog.components.values() if c.kind == kind),
            key=lambda c: c.name,
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
    label: str, resolution: Resolution, *, root: Path, uv: str | None
) -> bool:
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
    """Validate component manifests and resolutions without installing."""
    catalog = discover(args.root)
    if args.all and args.targets:
        print(
            "--all checks every component; do not name components with it",
            file=sys.stderr,
        )
        return 2
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

    if len(targets) > 1 and not args.all:
        resolution = resolve_targets(targets, catalog, root=args.root)
        failed |= _report_resolution(
            " ".join(targets), resolution, root=args.root, uv=uv
        )
        if not resolution.ok:
            for name in targets:
                failed |= _report_resolution(
                    name,
                    resolve_targets([name], catalog, root=args.root),
                    root=args.root,
                    uv=uv,
                )
        return 1 if failed else 0

    for name in targets:
        failed |= _report_resolution(
            name,
            resolve_targets([name], catalog, root=args.root),
            root=args.root,
            uv=uv,
        )
    return 1 if failed else 0


def command_install(args: argparse.Namespace) -> int:
    """Create or update a virtualenv for the named components."""
    install(
        args.targets,
        root=args.root,
        venv=args.venv,
        python_version=args.python,
        dry_run=args.dry_run,
        force=args.force,
        index_url=args.index_url,
        torch_backend=args.torch_backend,
        verbose=True,
    )
    return 0


def command_doctor(args: argparse.Namespace) -> int:
    """Diagnose the active or selected virtualenv."""
    venv = args.venv or current_venv()
    if venv is None:
        print("no active virtualenv; activate one or pass --venv", file=sys.stderr)
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
        help="RPent checkout containing component manifests.",
    )
    subcommands = parser.add_subparsers(dest="command", required=True)

    listing = subcommands.add_parser("list", help="list available components")
    listing.set_defaults(handler=command_list)

    check = subcommands.add_parser("check", help="check components without installing")
    check.add_argument("targets", nargs="*", metavar="COMPONENT")
    check.add_argument(
        "--all", action="store_true", help="check every component individually"
    )
    check.add_argument(
        "--resolve",
        action="store_true",
        help="also verify dependency resolution with uv",
    )
    check.set_defaults(handler=command_check)

    installer = subcommands.add_parser(
        "install", help="install components and their requirements into a virtualenv"
    )
    installer.add_argument("targets", nargs="+", metavar="COMPONENT")
    installer.add_argument("--venv", type=Path, help="target virtualenv path")
    installer.add_argument(
        "--python",
        default=None,
        help="Python version to provision for the virtualenv (e.g. 3.11 or 3.12)",
    )
    installer.add_argument(
        "--dry-run", action="store_true", help="print generated requirements only"
    )
    installer.add_argument(
        "--force",
        action="store_true",
        help="recreate an existing virtualenv at --venv",
    )
    installer.add_argument(
        "--index-url",
        default=None,
        help="override the primary Python package index URL (e.g. a PyPI mirror)",
    )
    installer.add_argument(
        "--torch-backend",
        default=None,
        help="set UV_TORCH_BACKEND (e.g. auto, cpu, cu124, cu126, cu128)",
    )
    installer.set_defaults(handler=command_install)

    doctor = subcommands.add_parser("doctor", help="diagnose an installed virtualenv")
    doctor.add_argument("--venv", type=Path)
    doctor.add_argument(
        "--fingerprint",
        action="store_true",
        help="print the stored installation fingerprint",
    )
    doctor.set_defaults(handler=command_doctor)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run ``rpent-install`` and return a process exit code."""
    parser = build_parser()
    args = parser.parse_args(argv)
    needs_checkout = not (args.command == "doctor" and args.fingerprint)
    if needs_checkout:
        if args.root is None:
            default_root = os.environ.get("RPENT_DEFAULT_ROOT")
            args.root = (
                _checkout_root(Path.cwd())
                or (Path(default_root) if default_root else None)
                or get_repo_root()
            )
        args.root = args.root.resolve()
        if _checkout_root(args.root) != args.root:
            parser.error(
                f"{args.root} is not an RPent checkout; run from a checkout or pass --root"
            )
    try:
        return args.handler(args)
    except (InstallError, InstallConfigError) as exc:
        print(str(exc), file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
