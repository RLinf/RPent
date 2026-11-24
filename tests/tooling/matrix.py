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

"""Assign every component to the runner that installs it in full CI.

Components are discovered, but the workflow needs a static matrix. The table
below is authored on purpose: each entry is one fresh environment, named by the
components it installs, and a generated matrix would follow the components it
is supposed to cover, so a component that nobody classified would quietly
install nowhere. ``--check`` resolves every entry and compares its coverage with
the discovered components, so adding a component is also a decision about where
it is installed.

An entry may name several components: the environment then contains them and
their ``requires`` closure. Coverage is by closure, so the model services a
robot environment already installs need no entry of their own.

Hosted runners install the entries whose dependencies pull no CUDA packages; the
GPU runner installs everything that does.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from rpent.install.discovery import Catalog, discover
from rpent.install.requirements import (
    read_core_dependencies,
    read_planner_dependencies,
)
from rpent.install.resolve import resolve

REPO_ROOT = Path(__file__).resolve().parents[2]

#: CPU-only environments: installable on a hosted runner within its disk and
#: time limits.
HOSTED = (
    ("infra-dev",),
    ("infra-minimal",),
    ("infra-unit_cpu",),
    # EFE reuses the network services the benchmark provides, so its
    # environment needs no CUDA packages.
    ("robot-efe",),
)

#: Environments that resolve CUDA packages; they need the embodied GPU runner.
GPU = (
    ("infra-rlinf",),
    ("model-apxinf",),
    ("model-cosmos3",),
    ("model-cosmos_policy",),
    ("model-fast_wam",),
    ("model-molmo",),
    ("model-uniwam",),
    # The adapter installs no package of its own; the XPolicyLab runtime comes
    # from the deployment's resource root.
    ("model-xpolicylab",),
    ("robot-dual_franka",),
    ("robot-franka",),
    ("robot-libero",),
    ("robot-robocasa",),
    ("robot-robodojo",),
    ("robot-robotwin",),
    ("robot-yam",),
    ("sim-libero",),
    ("sim-libero_plus",),
    ("sim-robocasa",),
    ("sim-robodojo",),
    ("sim-robotwin",),
)

MATRICES = {"hosted": HOSTED, "gpu": GPU}


def problems(
    catalog: Catalog,
    matrices: dict[str, tuple[tuple[str, ...], ...]] | None = None,
) -> list[str]:
    """Return every way the table disagrees with ``catalog``."""
    table = MATRICES if matrices is None else matrices
    found: list[str] = []
    covered: set[str] = set()
    seen: dict[tuple[str, ...], str] = {}
    for runner, entries in table.items():
        for entry in entries:
            label = " ".join(entry)
            if not entry:
                found.append(f"{runner} lists an empty environment")
                continue
            if entry in seen:
                found.append(
                    f"{runner} lists environment {label!r} again; {seen[entry]} "
                    "already installs it"
                )
                continue
            seen[entry] = runner
            unknown = sorted(set(entry) - set(catalog.components))
            if unknown:
                found.append(
                    f"{runner} lists environment {label!r}, but "
                    f"{', '.join(unknown)} does not exist; remove it from "
                    "tests/tooling/matrix.py"
                )
                continue
            resolution = resolve(
                entry,
                catalog,
                core_dependencies=read_core_dependencies(catalog.root),
                planner_dependencies=read_planner_dependencies(catalog.root),
            )
            if not resolution.ok:
                found.append(
                    f"environment {label!r} cannot be installed: "
                    f"{'; '.join(resolution.errors)}"
                )
            covered.update(resolution.components)
    for name in sorted(set(catalog.components) - covered):
        found.append(
            f"component {name!r} is not installed by any environment; add it to "
            "HOSTED or GPU in tests/tooling/matrix.py"
        )
    return found


def main(argv: list[str] | None = None, *, root: Path = REPO_ROOT) -> int:
    """Check the table or print one runner's environments as JSON."""
    parser = argparse.ArgumentParser(
        prog="tests.tooling.matrix",
        description="Assign components to the runner that installs them.",
    )
    parser.add_argument(
        "--runner",
        choices=sorted(MATRICES),
        help="print one runner's environments as a JSON array of arrays",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="fail when the table and the discovered components disagree",
    )
    args = parser.parse_args(argv)

    if not args.check and args.runner is None:
        parser.error("pass --check or --runner")
    if args.check:
        found = problems(discover(root))
        for problem in found:
            print(f"error: {problem}", file=sys.stderr)
        if found:
            return 1
    if args.runner is not None:
        # Read the table without discovering components: the workflow needs the
        # matrix before any install runs, and --check owns the drift check.
        print(json.dumps([list(entry) for entry in MATRICES[args.runner]]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
