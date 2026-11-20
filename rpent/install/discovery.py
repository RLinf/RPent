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

"""Discover component manifests in a source checkout."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from rpent.install.errors import InstallConfigError
from rpent.install.manifest import (
    COMPONENT_DIRECTORIES,
    Component,
    load_component,
)
from rpent.utils.config import get_repo_root

#: Every component root: the glob that finds one ``component.toml`` per
#: component directory, and the kind that directory declares.
COMPONENT_ROOTS = tuple(
    (f"{directory}/*/component.toml", kind)
    for kind, directory in COMPONENT_DIRECTORIES.items()
)


@dataclass(frozen=True)
class Catalog:
    """Components available in one checkout."""

    root: Path
    components: dict[str, Component]


def _is_metadata_directory(directory: Path) -> bool:
    """Return whether ``directory`` is AppleDouble or other private metadata.

    Network filesystems store macOS extended attributes as ``._<name>`` entries
    next to the original, so a synced checkout can gain directories that are not
    components. An underscore marks the same kind of private entry, which is how
    the repository layout names them.
    """
    return directory.name.startswith((".", "_"))


def _component_manifests(root: Path) -> Iterable[tuple[Path, str]]:
    seen: set[Path] = set()
    for pattern, kind in COMPONENT_ROOTS:
        for manifest in sorted(root.glob(pattern)):
            if _is_metadata_directory(manifest.parent):
                continue
            resolved = manifest.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            yield manifest, kind


def discover_components(root: Path | None = None) -> dict[str, Component]:
    """Load every component manifest under ``root``.

    Raises:
        InstallConfigError: a manifest is invalid, duplicated, or references a
            component that does not exist.
    """
    repo_root = (root or get_repo_root()).resolve()
    components: dict[str, Component] = {}
    for manifest, kind in _component_manifests(repo_root):
        component = load_component(manifest, kind)
        previous = components.get(component.name)
        if previous is not None:
            raise InstallConfigError(
                f"component {component.name!r} is declared twice: "
                f"{previous.manifest_path} and {manifest}"
            )
        components[component.name] = component
    _check_references(components)
    # ``env_conflicts`` stays as each manifest writes it: resolution checks the
    # relation from both sides, so declaring a conflict once is enough.
    return components


def _check_references(components: dict[str, Component]) -> None:
    """Require every declared relationship to name a known component.

    ``imports`` additionally stays a robot-slice relation: model services share
    a wire contract instead of importing each other's modules, and components
    without code cannot import anything.
    """
    for component in components.values():
        for field in ("requires", "env_conflicts", "imports"):
            for other in getattr(component, field):
                if other not in components:
                    raise InstallConfigError(
                        f"{component.manifest_path}: {field} references unknown "
                        f"component {other!r}"
                    )
                if other == component.name:
                    raise InstallConfigError(
                        f"{component.manifest_path}: {field} names the component itself"
                    )
        for other in component.imports:
            if component.kind != "robot":
                raise InstallConfigError(
                    f"{component.manifest_path}: imports is only for robot "
                    f"slices; {component.kind} components share a contract "
                    "instead of importing each other"
                )
            if components[other].kind != "robot":
                raise InstallConfigError(
                    f"{component.manifest_path}: imports {other!r} must name a "
                    f"robot component, not {components[other].kind!r}"
                )
            if other not in component.requires and component.name not in (
                components[other].requires
            ):
                raise InstallConfigError(
                    f"{component.manifest_path}: imports {other!r}, but neither "
                    "component requires the other; add it to requires so the "
                    "install closure includes both"
                )


def discover(root: Path | None = None) -> Catalog:
    """Return every component in one checkout."""
    repo_root = (root or get_repo_root()).resolve()
    return Catalog(root=repo_root, components=discover_components(repo_root))
