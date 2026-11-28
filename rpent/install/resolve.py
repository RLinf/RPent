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

"""Resolve components into one environment's requirements and check compatibility."""

from __future__ import annotations

import hashlib
import json
import tempfile
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from rpent.install.errors import InstallConfigError
from rpent.install.manifest import DEFAULT_MANYLINUX, Catalog, Component

if TYPE_CHECKING:
    from packaging.specifiers import SpecifierSet

try:  # Python 3.11+
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10
    import tomli as tomllib

#: Python minors RPent supports, newest first; mirrors ``requires-python``.
SUPPORTED_PYTHONS = ("3.12", "3.11", "3.10")
PLATFORM_TAG = "x86_64-manylinux_{baseline}"


@dataclass
class Resolution:
    """Resolved component closure, merged index settings, and static errors."""

    targets: tuple[str, ...]
    components: dict[str, Component] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)
    index_url: str | None = None
    extra_index_urls: tuple[str, ...] = ()
    index_strategy: str | None = None

    @property
    def ok(self) -> bool:
        """Return whether static checks passed."""
        return not self.errors

    @property
    def uses_planner(self) -> bool:
        """Return whether the environment installs ``rpent[planner]``."""
        return any(component.planner for component in self.components.values())

    def dependencies(self) -> tuple[str, ...]:
        """Return every direct dependency in the closure, deduplicated."""
        return tuple(
            dict.fromkeys(
                requirement
                for component in self.components.values()
                for requirement in component.dependencies
            )
        )


def _read_pyproject(root: Path) -> dict[str, Any]:
    path = root / "pyproject.toml"
    try:
        with path.open("rb") as file:
            return tomllib.load(file)
    except (OSError, UnicodeDecodeError, tomllib.TOMLDecodeError) as exc:
        raise InstallConfigError(f"{path}: cannot read pyproject.toml: {exc}") from exc


def _dependency_strings(value: Any, path: Path, key: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not all(
        isinstance(item, str) and item for item in value
    ):
        raise InstallConfigError(f"{path}: {key} must be a list of requirement strings")
    return tuple(value)


def read_core_dependencies(root: Path) -> tuple[str, ...]:
    """Return ``[project].dependencies`` from ``pyproject.toml``."""
    project = _read_pyproject(root).get("project")
    if not isinstance(project, dict):
        raise InstallConfigError(f"{root / 'pyproject.toml'}: missing [project] table")
    return _dependency_strings(
        project.get("dependencies"), root / "pyproject.toml", "[project].dependencies"
    )


def read_planner_dependencies(root: Path) -> tuple[str, ...]:
    """Return ``[project.optional-dependencies].planner``, or ``()``."""
    project = _read_pyproject(root).get("project")
    if not isinstance(project, dict):
        raise InstallConfigError(f"{root / 'pyproject.toml'}: missing [project] table")
    optional = project.get("optional-dependencies")
    planner = optional.get("planner") if isinstance(optional, dict) else None
    if planner is None:
        return ()
    return _dependency_strings(
        planner, root / "pyproject.toml", "[project.optional-dependencies].planner"
    )


def read_uv_settings(root: Path) -> dict[str, Any]:
    """Return ``[tool.uv]`` from ``pyproject.toml``, or ``{}``."""
    path = root / "pyproject.toml"
    if not path.is_file():
        return {}
    tool = _read_pyproject(root).get("tool")
    settings = tool.get("uv") if isinstance(tool, dict) else None
    return dict(settings) if isinstance(settings, dict) else {}


def _canonical(value: str) -> tuple[str, SpecifierSet, str] | None:
    from packaging.requirements import InvalidRequirement, Requirement
    from packaging.utils import canonicalize_name

    try:
        requirement = Requirement(value)
    except InvalidRequirement:
        return None
    if requirement.marker is not None:
        return None
    return (
        canonicalize_name(requirement.name),
        requirement.specifier,
        str(requirement.url) if requirement.url else "",
    )


def _pin_conflicts(
    requirements: list[tuple[str, str]],
) -> dict[str, list[tuple[str, str]]]:
    """Return ``package -> [(owner, pin)]`` when exact pins or URLs disagree."""
    pins: dict[str, list[tuple[str, str]]] = {}
    for owner, value in requirements:
        identity = _canonical(value)
        if identity is None:
            continue
        package, specifier, url = identity
        if url:
            pin = url
        elif len(specifier) == 1:
            item = next(iter(specifier))
            if item.operator not in {"==", "==="} or item.version.endswith(".*"):
                continue
            pin = f"{item.operator}{item.version}"
        else:
            continue
        pins.setdefault(package, []).append((owner, pin))
    return {
        package: sorted(set(entries))
        for package, entries in sorted(pins.items())
        if len({pin for _, pin in entries}) > 1
    }


def _override_conflicts(components: dict[str, Component]) -> list[str]:
    """Report packages constrained by override files from multiple components."""
    from packaging.requirements import InvalidRequirement, Requirement
    from packaging.utils import canonicalize_name

    owners: dict[str, list[str]] = {}
    for name, component in sorted(components.items()):
        packages: set[str] = set()
        for path in component.overrides:
            try:
                text = path.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError) as exc:
                raise InstallConfigError(
                    f"{path}: cannot read override file: {exc}"
                ) from exc
            for line in text.splitlines():
                value = line.strip()
                if not value or value.startswith("#"):
                    continue
                try:
                    req = Requirement(value)
                except InvalidRequirement:
                    continue
                if req.marker is None:
                    packages.add(canonicalize_name(req.name))
        for package in packages:
            owners.setdefault(package, []).append(name)
    return [
        f"components {', '.join(repr(name) for name in names)} both override "
        f"{package!r} in one environment"
        for package, names in sorted(owners.items())
        if len(names) > 1
    ]


def _minor_candidates(minor: str, components: Iterable[Component]) -> list[str]:
    """Return candidate versions within ``minor`` to handle patch-level bounds."""
    from packaging.specifiers import InvalidSpecifier, SpecifierSet
    from packaging.version import InvalidVersion, Version

    candidates = [f"{minor}.0", f"{minor}.99"]
    minor_tuple = tuple(int(part) for part in minor.split("."))
    for component in components:
        if not component.python:
            continue
        try:
            specifier = SpecifierSet(component.python)
        except InvalidSpecifier:
            continue
        for item in specifier:
            try:
                ver = Version(item.version.rstrip(".*"))
            except InvalidVersion:
                continue
            if (ver.major, ver.minor) == minor_tuple:
                candidates.append(str(ver))
                candidates.append(f"{ver.major}.{ver.minor}.{ver.micro + 1}")
    return list(dict.fromkeys(candidates))


def _supports_version(component: Component, version: str) -> bool:
    from packaging.specifiers import InvalidSpecifier, SpecifierSet

    if component.python is None:
        return True
    try:
        supported = SpecifierSet(component.python)
    except InvalidSpecifier:
        return True
    return supported.contains(version, prereleases=True)


def choose_python(components: Iterable[Component]) -> str:
    """Return the newest supported Python minor (or patch) every component accepts."""
    items = list(components)
    for minor in SUPPORTED_PYTHONS:
        if all(_supports_version(component, minor) for component in items):
            return minor
        # Handle patch-level bounds such as ``>=3.11.4,<3.12``.
        for candidate in _minor_candidates(minor, items):
            if all(_supports_version(component, candidate) for component in items):
                return minor
    return SUPPORTED_PYTHONS[0]


def python_errors(
    version: str,
    components: Iterable[Component],
    *,
    owner: str = "this environment",
) -> list[str]:
    """Return errors for components incompatible with Python ``version``."""
    from packaging.specifiers import InvalidSpecifier, SpecifierSet
    from packaging.version import InvalidVersion, Version

    try:
        parsed = Version(version)
    except InvalidVersion:
        return [f"invalid or unknown Python version {version!r} for {owner}"]
    errors: list[str] = []
    for component in sorted(components, key=lambda item: item.name):
        if component.python is None:
            continue
        try:
            supported = SpecifierSet(component.python)
        except InvalidSpecifier as exc:
            errors.append(
                f"{component.manifest_path}: invalid python range "
                f"{component.python!r}: {exc}"
            )
            continue
        if not supported.contains(parsed, prereleases=True):
            # When checking a bare minor such as "3.11" (before venv creation),
            # accept if a patch version within that minor satisfies the range.
            if len(parsed.release) == 2 and any(
                supported.contains(candidate, prereleases=True)
                for candidate in _minor_candidates(version, [component])
            ):
                continue
            errors.append(
                f"component {component.name!r} requires Python "
                f"{component.python}, but {owner} uses {version}"
            )
    return errors


def _merge_index_settings(
    resolution: Resolution, components: Iterable[Component]
) -> None:
    """Merge index settings across the closure; union ``extra-index-urls``."""
    extra: list[str] = []
    for component in sorted(components, key=lambda item: item.name):
        if component.index_url is not None:
            if resolution.index_url not in (None, component.index_url):
                resolution.errors.append(
                    f"component {component.name!r} selects index "
                    f"{component.index_url!r}, but another component selects "
                    f"{resolution.index_url!r}; install them in separate environments"
                )
            resolution.index_url = component.index_url
        if component.index_strategy is not None:
            if resolution.index_strategy not in (None, component.index_strategy):
                resolution.errors.append(
                    f"component {component.name!r} selects index strategy "
                    f"{component.index_strategy!r}, but another component selects "
                    f"{resolution.index_strategy!r}; install them in separate environments"
                )
            resolution.index_strategy = component.index_strategy
        extra.extend(component.extra_index_urls)
    resolution.extra_index_urls = tuple(dict.fromkeys(extra))


def resolve(
    targets: Iterable[str],
    catalog: Catalog,
    *,
    core_dependencies: tuple[str, ...] = (),
    planner_dependencies: tuple[str, ...] = (),
    python: str | None = None,
) -> Resolution:
    """Resolve ``targets`` and their ``requires`` closure into one environment."""
    names = tuple(sorted(set(targets)))
    resolution = Resolution(targets=names)
    if not names:
        resolution.errors.append("name at least one component to install")
        return resolution

    for name in names:
        if name not in catalog.components:
            resolution.errors.append(
                f"unknown component {name!r}; known components: "
                f"{', '.join(sorted(catalog.components))}"
            )

    pending = list(names)
    while pending:
        name = pending.pop(0)
        if name in resolution.components:
            continue
        component = catalog.components.get(name)
        if component is None:
            continue
        resolution.components[name] = component
        pending.extend(component.requires)

    ordered = dict(sorted(resolution.components.items()))
    resolution.components = ordered

    for component in ordered.values():
        for required in component.requires:
            if required in component.env_conflicts or (
                required in ordered
                and component.name in ordered[required].env_conflicts
            ):
                resolution.errors.append(
                    f"component {component.name!r} requires {required!r} but "
                    "the two declare env_conflicts; fix their manifests"
                )

    check_python = python if python is not None else choose_python(ordered.values())
    resolution.errors.extend(
        python_errors(check_python, ordered.values(), owner="the environment")
    )
    _merge_index_settings(resolution, ordered.values())

    env_names = list(ordered)
    for idx, left_name in enumerate(env_names):
        for right_name in env_names[idx + 1 :]:
            left, right = ordered[left_name], ordered[right_name]
            if right_name in left.env_conflicts or left_name in right.env_conflicts:
                resolution.errors.append(
                    f"components {left.name!r} and {right.name!r} cannot share an "
                    f"environment (env_conflicts); install each separately: "
                    f"`rpent-install install {left.name}` and "
                    f"`rpent-install install {right.name}`"
                )

    requirements = [
        (component.name, req)
        for component in ordered.values()
        for req in component.dependencies
    ]
    requirements.extend(("rpent", req) for req in core_dependencies)
    if resolution.uses_planner:
        if not planner_dependencies:
            needs_planner = [name for name, comp in ordered.items() if comp.planner]
            resolution.errors.append(
                "[project.optional-dependencies].planner is missing from "
                "pyproject.toml, but these components declare 'planner = true': "
                + ", ".join(needs_planner)
            )
        requirements.extend(("rpent[planner]", req) for req in planner_dependencies)

    for package, pins in _pin_conflicts(requirements).items():
        lines = "\n".join(f"  {owner}: {package} {pin}" for owner, pin in pins)
        resolution.errors.append(f"conflicting pins for {package!r}\n{lines}")
    resolution.errors.extend(_override_conflicts(ordered))
    resolution.errors = sorted(dict.fromkeys(resolution.errors))
    return resolution


def render_requirements(
    resolution: Resolution,
    *,
    core_dependencies: tuple[str, ...] = (),
    planner_dependencies: tuple[str, ...] = (),
) -> str:
    """Render a deterministic requirements file for ``resolution``."""
    target_args = " ".join(resolution.targets)
    lines = [
        f"# Generated by rpent-install for: {target_args}",
        f"# Rerun: rpent-install install {target_args}",
    ]
    core = sorted(dict.fromkeys(core_dependencies))
    if core:
        lines.extend(["", "# RPent core", *core])
    if resolution.uses_planner and planner_dependencies:
        lines.extend(
            [
                "",
                "# RPent planner (rpent[planner])",
                *sorted(dict.fromkeys(planner_dependencies)),
            ]
        )
    for name, component in resolution.components.items():
        deps = sorted(dict.fromkeys(component.dependencies))
        if deps:
            lines.extend(["", f"# component {name}", *deps])
    return "\n".join(lines) + "\n"


def _requirement_lines(path: Path) -> list[str]:
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise InstallConfigError(f"{path}: cannot read override file: {exc}") from exc
    return [
        line.rstrip()
        for line in text.splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]


def requirements_sha256(
    resolution: Resolution,
    *,
    root: Path,
    core_dependencies: tuple[str, ...] = (),
    planner_dependencies: tuple[str, ...] = (),
) -> str:
    """Hash every manifest, override, and index input for ``resolution``."""
    repo_root = root.resolve()
    overrides: list[dict[str, Any]] = []
    seen: set[Path] = set()
    for component in resolution.components.values():
        for override in component.overrides:
            path = override if override.is_absolute() else repo_root / override
            if path in seen:
                continue
            seen.add(path)
            display = (
                str(path.relative_to(repo_root))
                if path.is_relative_to(repo_root)
                else str(path)
            )
            overrides.append({"path": display, "lines": _requirement_lines(path)})
    overrides.sort(key=lambda item: item["path"])

    payload = json.dumps(
        {
            "components": {
                name: comp.manifest_sha256
                for name, comp in sorted(resolution.components.items())
            },
            "core_dependencies": sorted(core_dependencies),
            "planner_dependencies": (
                sorted(planner_dependencies) if resolution.uses_planner else []
            ),
            "overrides": overrides,
            "index": {
                "url": resolution.index_url,
                "extra_urls": sorted(resolution.extra_index_urls),
                "strategy": resolution.index_strategy,
            },
            "uv_settings": read_uv_settings(repo_root),
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def index_options(
    resolution: Resolution, *, index_url_override: str | None = None
) -> list[str]:
    """Return ``uv`` index flags for ``resolution``."""
    options: list[str] = []
    index_url = index_url_override or resolution.index_url
    if index_url:
        options.extend(["--index-url", index_url])
    for url in resolution.extra_index_urls:
        options.extend(["--extra-index-url", url])
    if resolution.index_strategy:
        options.extend(["--index-strategy", resolution.index_strategy])
    return options


def manylinux_baseline(components: Iterable[Component]) -> str:
    """Return the newest glibc baseline required by ``components``."""
    baselines = [component.manylinux for component in components]
    if not baselines:
        return DEFAULT_MANYLINUX
    return max(baselines, key=lambda v: tuple(int(p) for p in v.split("_")))


def resolve_resolution(
    resolution: Resolution,
    *,
    root: Path,
    uv: str,
    python: str,
    core_dependencies: tuple[str, ...] = (),
    planner_dependencies: tuple[str, ...] = (),
) -> str:
    """Verify that ``resolution`` compiles with ``uv pip compile``."""
    from rpent.install.installer import run_command

    platform_tag = PLATFORM_TAG.format(
        baseline=manylinux_baseline(resolution.components.values())
    )
    requirements = render_requirements(
        resolution,
        core_dependencies=core_dependencies,
        planner_dependencies=planner_dependencies,
    )
    with tempfile.TemporaryDirectory(prefix="rpent-resolve-") as directory:
        source = Path(directory) / "requirements.in"
        output = Path(directory) / "requirements.txt"
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
                command.extend(["--override", str(override)])
        run_command(command, capture=True, cwd=root)
        return output.read_text(encoding="utf-8")
