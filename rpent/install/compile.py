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

"""Check component dependencies with a temporary ``uv pip compile`` resolution."""

from __future__ import annotations

import tempfile
from collections.abc import Iterable
from pathlib import Path

from rpent.install.installer import index_options, run_command
from rpent.install.manifest import DEFAULT_MANYLINUX, Component
from rpent.install.requirements import render_requirements
from rpent.install.resolve import Resolution

#: The wheel target every environment resolves against. RPent supports one
#: platform; ``manylinux`` in a manifest raises the baseline within it.
PLATFORM_TAG = "x86_64-manylinux_{baseline}"


def manylinux_baseline(components: Iterable[Component]) -> str:
    """Return the newest glibc baseline any component in the closure needs."""
    baselines = [component.manylinux for component in components]
    if not baselines:
        return DEFAULT_MANYLINUX
    return max(baselines, key=_baseline_key)


def _baseline_key(value: str) -> tuple[int, ...]:
    return tuple(int(part) for part in value.split("_"))


def compile_resolution(
    resolution: Resolution,
    *,
    root: Path,
    uv: str,
    output: Path,
    python: str,
    core_dependencies: tuple[str, ...] = (),
    planner_dependencies: tuple[str, ...] = (),
) -> str:
    """Compile one component environment into ``output``.

    ``python`` is the ``major.minor`` interpreter the environment would run.
    """
    platform_tag = PLATFORM_TAG.format(
        baseline=manylinux_baseline(resolution.components.values())
    )
    requirements = render_requirements(
        resolution,
        core_dependencies=core_dependencies,
        planner_dependencies=planner_dependencies,
    )
    name = "+".join(resolution.targets)
    with tempfile.TemporaryDirectory(prefix="rpent-requirements-") as directory:
        source = Path(directory) / f"{name}.txt"
        source.write_text(requirements, encoding="utf-8")
        command = [
            uv,
            "pip",
            "compile",
            str(source),
            "--python-version",
            python,
            "--python-platform",
            platform_tag,
            "--no-header",
            "--no-annotate",
            "--output-file",
            str(output),
            *index_options(resolution),
        ]
        for component in resolution.components.values():
            for override in component.overrides:
                # Manifests resolve overrides to absolute paths when they are
                # loaded, so uv's working directory does not matter here.
                command.extend(["--overrides", str(override)])
        # Capture output so a resolution failure reports uv's reason (a missing
        # branch, an unreachable index) instead of only the command line. uv
        # discovers ``[tool.uv]`` from its working directory, so compile in the
        # checkout rather than wherever the CLI was invoked.
        run_command(command, capture=True, cwd=root)
    return output.read_text(encoding="utf-8")


def resolve_resolution(
    resolution: Resolution,
    *,
    root: Path,
    uv: str,
    python: str,
    core_dependencies: tuple[str, ...] = (),
    planner_dependencies: tuple[str, ...] = (),
) -> str:
    """Compile ``resolution`` into a temporary file and return the uv output.

    ``rpent-install check --resolve`` uses this to prove that the declared
    dependencies have a solution; it delegates to :func:`compile_resolution` so
    there is only one resolver.
    """
    with tempfile.TemporaryDirectory(prefix="rpent-resolve-") as directory:
        output = Path(directory) / "requirements.txt"
        return compile_resolution(
            resolution,
            root=root,
            uv=uv,
            output=output,
            python=python,
            core_dependencies=core_dependencies,
            planner_dependencies=planner_dependencies,
        )
