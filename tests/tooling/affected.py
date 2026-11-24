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

"""Map changed paths to the components, tests and environments they affect.

`fast ci` narrows which tests a pull request runs and `full ci` narrows which
environments it installs from the same answer, so the two levels cannot drift: a
path under a component directory names that component; an end-to-end suite
directory names the robot whose integration it runs; the affected environments
are the components whose ``requires`` closure contains the named component, and
the affected tests are the suites that mirror it.

A path outside every component directory belongs to the core, the installer, or
the install data, so it can change every installation. An unknown component
name is treated the same way rather than silently ignored. The answer then
names every component, every environment, and the whole unit suite. When no
path is given there is nothing to narrow the answer with, so the same happens.

Changes confined to the repository's install data are narrower: they decide
which environment can be built rather than how a component behaves, so they
select the repository contract tests and the environments whose closure reads
the changed files. Unrecognised install data widens back to every environment.

Narrowing is fail-open throughout: a change no test directory covers, or a
component whose tests do not exist yet, widens the scope instead of reporting a
smaller one.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import asdict, dataclass, replace
from pathlib import Path

from rpent.install.discovery import Catalog, discover
from rpent.install.errors import InstallConfigError
from rpent.install.manifest import COMPONENT_DIRECTORIES
from rpent.install.resolve import resolve
from tests.tooling.matrix import MATRICES

REPO_ROOT = Path(__file__).resolve().parents[2]

#: Component owner directories and the kind they hold. The installer owns this
#: table, so a new component kind becomes visible here instead of drifting.
#: Infrastructure manifests are install data, which the scope handles on its
#: own, so they are not component owners here.
COMPONENT_ROOTS = tuple(
    (directory, kind)
    for kind, directory in COMPONENT_DIRECTORIES.items()
    if kind != "infra"
)
#: Tests every component group contributes to, relative to the repository root.
SHARED_TESTS = ("tests/unit_tests/conftest.py",)
#: Robot toolkits are exercised once per slice, so a slice change reruns them.
ROBOT_SHARED_TESTS = (
    "tests/unit_tests/robots/test_tool_schema_contracts.py",
    "tests/unit_tests/robots/test_toolkit_contracts.py",
)
#: Environments a pull request installs when a change affects the core. The
#: core-only environment proves the change without pulling the full GPU matrix
#: into every core edit; nightly and release runs still install everything.
PR_PILOT_COMPONENTS = ("infra-minimal",)
#: Paths whose change can alter every installation, so a PR installs every
#: environment instead of the pilot set. Install data changes narrow their own
#: environment set first; a path here in a mixed change still widens it. The
#: checkout launcher decides how every environment bootstraps, the GPU launcher
#: decides how every end-to-end suite runs, and this tooling decides which
#: environments a change runs, so all three widen the same way.
PR_ALL_INSTALL_PREFIXES = (
    "rpent/install/",
    "components/infra/",
    ".github/",
    "rpent-install.sh",
    "tests/tooling/",
    "tests/e2e_tests/",
)
#: The repository's install data: the infrastructure manifests that decide what
#: each environment contains, and the packaging settings that decide what the
#: core adds to every one of them.
INSTALL_DATA_PREFIXES = ("components/infra/",)
INSTALL_DATA_ROOT_FILES = ("pyproject.toml",)
#: Repository contract tests read that data. Installer unit tests live under
#: ``tests/unit_tests/rpent/install/`` and use synthetic manifests, so the two
#: groups stay distinguishable by path.
INSTALL_DATA_TESTS = "tests/unit_tests/installs"
#: Tests for the repository checks and the selection logic that reads the
#: install data. They live under ``tests/unit_tests/`` so the default unit run
#: collects them, and changing the checks can change what every other test path
#: means, so they run with every scope that is not narrowed to a component.
TOOLING_TESTS = "tests/unit_tests/tooling"


@dataclass(frozen=True)
class Scope:
    """What a change can affect, in the form CI jobs consume."""

    #: Component names the change can affect.
    components: tuple[str, ...]
    tests: tuple[str, ...]
    #: Components to install, one environment each. For a component change
    #: these are the components whose closure contains it.
    installs: tuple[str, ...]
    #: True when nothing narrowed the answer to part of the repository.
    full: bool
    #: Components a pull request installs. For a component change this equals
    #: ``installs``; for a core change it is the pilot set, while ``installs``
    #: stays complete for nightly and release runs. Installer and install-data
    #: changes keep the full set because they can alter any environment.
    pr_installs: tuple[str, ...] = ()
    #: True when every changed path is prose, so only the docs check applies.
    docs: bool = False

    def as_dict(self) -> dict[str, object]:
        """Return the JSON-ready form printed by the CLI."""
        return asdict(self)


def _is_docs(path: str) -> bool:
    """Return True when ``path`` only changes documentation."""
    return path.startswith("docs/") or path.endswith((".md", ".rst"))


def _e2e_component(path: str) -> str | None:
    """Return the robot whose end-to-end suite holds ``path``, if any.

    ``tests/e2e_tests/<suite>/`` runs the integration of ``robot-<suite>``, so
    a change inside one reaches that robot rather than the whole repository.
    """
    parts = path.split("/")
    if len(parts) > 3 and parts[:2] == ["tests", "e2e_tests"]:
        return f"robot-{parts[2]}"
    return None


def _component_for(path: str) -> tuple[str, str] | None:
    """Return ``(kind, name)`` when ``path`` belongs to one component."""
    parts = path.split("/")
    for root, kind in COMPONENT_ROOTS:
        root_parts = root.split("/")
        if parts[: len(root_parts)] == root_parts and len(parts) > len(root_parts):
            return kind, f"{kind}-{parts[len(root_parts)]}"
    return None


def _is_install_data(path: str) -> bool:
    """Return whether ``path`` changes the repository's install data."""
    if path in INSTALL_DATA_ROOT_FILES:
        return True
    if path.startswith(INSTALL_DATA_PREFIXES):
        return True
    return path == INSTALL_DATA_TESTS or path.startswith(f"{INSTALL_DATA_TESTS}/")


def _installs_containing(names: set[str], catalog: Catalog) -> list[str] | None:
    """Return the components whose closure contains one of ``names``.

    ``None`` widens the answer when an environment cannot be resolved: ``check``
    reports that component separately, and CI must install it rather than skip
    it.
    """
    if not names <= set(catalog.components):
        return None
    found: list[str] = []
    for name in sorted(catalog.components):
        try:
            resolution = resolve([name], catalog)
        except InstallConfigError:
            return None
        if names & set(resolution.components):
            found.append(name)
    return found


def _installs_for_install_data(paths: list[str], catalog: Catalog) -> tuple[str, ...]:
    """Return the components whose installation reads ``paths``, or all of them.

    An infrastructure manifest names the components whose closure installs it.
    Packaging settings feed every environment, and an unrecognised file widens
    the answer rather than guessing which environments it reaches.
    """
    every = tuple(sorted(catalog.components))
    selected: set[str] = set()
    for path in paths:
        if path in INSTALL_DATA_ROOT_FILES:
            return every
        if path == INSTALL_DATA_TESTS or path.startswith(f"{INSTALL_DATA_TESTS}/"):
            # A test-only change installs nothing; the unit job still runs it.
            continue
        parts = path.split("/")
        if len(parts) >= 3 and parts[:2] == ["components", "infra"]:
            required = _installs_containing({f"infra-{parts[2]}"}, catalog)
            if required is None:
                return every
            selected.update(required)
        else:
            return every
    return tuple(sorted(selected))


def _pr_installs(paths: list[str], components: tuple[str, ...]) -> tuple[str, ...]:
    """Return the components a pull request should install for ``paths``.

    A component change never reaches this helper: its environment set is
    already the resolution closure. For a core change, installing every GPU
    environment proves little beyond what nightly proves and costs a
    self-hosted runner slot, so the PR installs the core-only pilot. The
    installer, workflows, packaging, this tooling, the end-to-end launcher, and
    a mixed change that still names one of them keep the full set; install data
    on its own narrows before this helper runs.
    """
    available = tuple(sorted(components))
    if not paths:
        return available
    if any(path.startswith(PR_ALL_INSTALL_PREFIXES) for path in paths):
        return available
    if all(path.startswith(("rpent/", "tests/")) for path in paths):
        pilots = tuple(name for name in PR_PILOT_COMPONENTS if name in components)
        return pilots or available
    return available


def affected_components(paths: list[str], catalog: Catalog) -> set[str] | None:
    """Return the components ``paths`` touch, or ``None`` when all can change.

    ``None`` means the change is not confined to a known component: the core,
    the installer, install data, or an unowned path.
    """
    names: set[str] = set()
    for path in paths:
        name = _e2e_component(path)
        if name is None:
            owner = _component_for(path)
            if owner is None:
                return None
            _, name = owner
        if name not in catalog.components:
            return None
        names.add(name)
    return names


def _tests_for(kind: str, name: str, root: Path) -> list[str]:
    """Return the existing test paths that mirror one component."""
    subject = name.split("-", 1)[1] if "-" in name else name
    if kind == "robot":
        candidates = [
            f"tests/unit_tests/robots/{subject}",
            *ROBOT_SHARED_TESTS,
        ]
    elif kind == "model":
        # A model service is exercised through its wire ends as well as its own
        # suite, and the RPC loopbacks are the only coverage of some models.
        candidates = [
            "tests/unit_tests/components/models",
            "tests/unit_tests/rpent/utils/rpc",
        ]
    elif kind == "tool":
        candidates = ["tests/unit_tests/components/tools"]
    else:
        # Simulator code is imported by robot slices and model servers, which
        # have no per-simulator suite; nothing mirrors it closely enough.
        candidates = []
    return [path for path in candidates if (root / path).exists()]


def affected_scope(paths: list[str], catalog: Catalog, root: Path) -> Scope:
    """Return the components, tests and environments ``paths`` can change."""
    if paths and all(_is_docs(path) for path in paths):
        return Scope(components=(), tests=(), installs=(), full=False, docs=True)
    if paths and all(_is_install_data(path) for path in paths):
        # Install data is repository-level: it changes which environment can be
        # built, not a component's behavior. The contract tests under
        # ``tests/unit_tests/installs/`` own the cross-file checks, the runner
        # partition under ``tests/unit_tests/tooling/`` covers every component,
        # and the environments whose closure reads the changed files are the
        # ones a pull request installs.
        installs = _installs_for_install_data(paths, catalog)
        return Scope(
            components=(),
            tests=(INSTALL_DATA_TESTS, TOOLING_TESTS),
            installs=installs,
            full=False,
            pr_installs=installs,
        )
    names = affected_components(paths, catalog)
    if names is None or not paths:
        return Scope(
            components=tuple(sorted(catalog.components)),
            tests=("tests/unit_tests", TOOLING_TESTS),
            installs=tuple(sorted(catalog.components)),
            full=True,
            pr_installs=_pr_installs(paths, tuple(catalog.components)),
        )
    tests: set[str] = {path for path in SHARED_TESTS if (root / path).exists()}
    for name in sorted(names):
        mirrors = _tests_for(catalog.components[name].kind, name, root)
        if not mirrors:
            # A component with no test directory must not shrink the suite to
            # nothing.
            tests.add("tests/unit_tests")
        tests.update(mirrors)
    installs = _installs_containing(names, catalog)
    if installs is None:
        installs = sorted(names)
    # A robot's suite exercises the components its closure installs, so a
    # change to one of them also runs the suites of the robots that depend on
    # it; some models are covered only there.
    for dependent in installs:
        component = catalog.components.get(dependent)
        if component is not None and component.kind == "robot":
            tests.update(_tests_for("robot", dependent, root))
    return Scope(
        components=tuple(sorted(names)),
        tests=tuple(sorted(tests)),
        installs=tuple(installs),
        full=False,
        pr_installs=tuple(installs),
    )


def changed_paths(root: Path, base: str, head: str) -> list[str]:
    """Return the repository-relative paths that differ between two refs.

    Deleted paths count too: a pull request that only deletes packaging data or
    a component file still changes what the checks have to prove, and dropping
    it would leave the remaining documentation to make the change look inert.
    """
    result = subprocess.run(
        ["git", "diff", "--name-only", "--diff-filter=ACDMRT", base, head],
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise SystemExit(
            f"git diff {base} {head} failed: {result.stderr.strip() or 'unknown error'}"
        )
    return [line for line in result.stdout.splitlines() if line.strip()]


def _runner_components(runner: str, catalog: Catalog) -> set[str]:
    """Return every component ``runner``'s environments install."""
    allowed: set[str] = set()
    for entry in MATRICES[runner]:
        allowed.update(entry)
        allowed.update(resolve(entry, catalog).components)
    return allowed


def main(argv: list[str] | None = None, *, root: Path = REPO_ROOT) -> int:
    """Print the affected scope as a JSON object."""
    parser = argparse.ArgumentParser(
        prog="tests.tooling.affected",
        description="Map changed paths to affected components, tests and environments.",
    )
    parser.add_argument(
        "paths",
        nargs="*",
        help="changed paths; read from git when --base is given and none are "
        "passed; everything is affected when neither is given",
    )
    parser.add_argument("--base", help="git revision to diff from")
    parser.add_argument("--head", default="HEAD", help="git revision to diff to")
    parser.add_argument(
        "--runner",
        choices=sorted(MATRICES),
        help="restrict the environments to the ones one runner installs",
    )
    args = parser.parse_args(argv)
    if args.paths and args.base:
        parser.error("pass either changed paths or --base, not both")
    paths = args.paths or (
        changed_paths(root, args.base, args.head) if args.base else []
    )
    catalog = discover(root)
    scope = affected_scope(paths, catalog, root)
    if args.runner is not None:
        allowed = _runner_components(args.runner, catalog)
        scope = replace(
            scope,
            components=tuple(name for name in scope.components if name in allowed),
            installs=tuple(name for name in scope.installs if name in allowed),
            pr_installs=tuple(name for name in scope.pr_installs if name in allowed),
        )
    print(json.dumps(scope.as_dict()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
