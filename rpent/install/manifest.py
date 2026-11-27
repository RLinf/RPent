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

"""Typed component manifests (``component.toml``) and checkout discovery."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rpent.install.errors import InstallConfigError
from rpent.utils.config import get_repo_root

try:  # Python 3.11+
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10
    import tomli as tomllib

SCHEMA = 2
KINDS = ("robot", "sim", "model", "tool", "infra")
COMPONENT_DIRECTORIES = {
    "robot": "robots",
    "sim": "components/sim",
    "model": "components/models",
    "tool": "components/tools",
    "infra": "components/infra",
}
COMPONENT_ROOTS = tuple(
    (f"{directory}/*/component.toml", kind)
    for kind, directory in COMPONENT_DIRECTORIES.items()
)
INDEX_STRATEGIES = ("first-index", "unsafe-first-match", "unsafe-best-match")
DEFAULT_MANYLINUX = "2_31"
FIELDS = frozenset(
    {
        "schema",
        "name",
        "probe",
        "python",
        "manylinux",
        "planner",
        "requires",
        "imports",
        "env_conflicts",
        "dependencies",
        "overrides",
        "index-url",
        "extra-index-urls",
        "index-strategy",
    }
)
_FULL_COMMIT = re.compile(r"^[0-9a-f]{40}$")
_VERSION_TAG = re.compile(r"^v?\d[\w.+-]*$")
_INTEGRATION_BRANCH = re.compile(
    r"^(?:rpent(?:_v\d+\.\d+(?:\.\d+)?|/[a-z][a-z0-9_-]*)?|release/v\d+\.\d+(?:\.\d+)?)$"
)
_MANYLINUX = re.compile(r"^\d+_\d+$")


@dataclass(frozen=True)
class Component:
    """One install unit declared by a ``component.toml`` manifest."""

    name: str
    kind: str
    path: Path
    manifest: Path
    probe: str
    python: str | None
    manylinux: str
    planner: bool
    requires: tuple[str, ...]
    imports: tuple[str, ...]
    env_conflicts: tuple[str, ...]
    dependencies: tuple[str, ...]
    overrides: tuple[Path, ...]
    index_url: str | None
    extra_index_urls: tuple[str, ...]
    index_strategy: str | None
    manifest_sha256: str

    @property
    def manifest_path(self) -> Path:
        """Return the manifest path."""
        return self.manifest

    @property
    def is_infra(self) -> bool:
        """Return whether this component belongs to the ``infra`` kind."""
        return self.kind == "infra"


@dataclass(frozen=True)
class Catalog:
    """Components available in one checkout."""

    root: Path
    components: dict[str, Component]


def prefix(kind: str) -> str:
    """Return the name prefix for ``kind``."""
    return f"{kind}-"


def _read_toml(path: Path) -> dict[str, Any]:
    try:
        with path.open("rb") as file:
            return tomllib.load(file)
    except OSError as exc:
        raise InstallConfigError(f"{path}: cannot read file: {exc}") from exc
    except tomllib.TOMLDecodeError as exc:
        raise InstallConfigError(f"{path}: invalid TOML: {exc}") from exc
    except UnicodeDecodeError as exc:
        raise InstallConfigError(f"{path}: is not valid UTF-8: {exc}") from exc


def _optional_string(data: dict[str, Any], key: str, path: Path) -> str | None:
    value = data.get(key)
    if value is None:
        return None
    if not isinstance(value, str) or not value:
        raise InstallConfigError(f"{path}: '{key}' must be a non-empty string")
    return value


def _string_list(data: dict[str, Any], key: str, path: Path) -> tuple[str, ...]:
    value = data.get(key)
    if value is None:
        return ()
    if not isinstance(value, list) or not all(
        isinstance(item, str) and item for item in value
    ):
        raise InstallConfigError(f"{path}: '{key}' must be a list of strings")
    return tuple(value)


def _git_ref(requirement: str) -> str | None:
    """Return the ref of a ``name @ git+...@ref`` requirement, if any."""
    marker = requirement.find("git+")
    if marker < 0:
        return None
    target = (
        requirement[marker + len("git+") :].split(";", 1)[0].split("#", 1)[0].strip()
    )
    _, scheme_sep, rest = target.partition("://")
    if scheme_sep:
        _, slash, path = rest.partition("/")
        if not slash:
            return None
        ref_region = path
    else:
        ref_region = target
    _, separator, ref = ref_region.rpartition("@")
    if not separator:
        return None
    return ref or None


def _check_git_dependency(requirement: str, path: Path) -> None:
    """Reject unpinned or generic branch-based git requirements."""
    if "git+" not in requirement:
        return
    ref = _git_ref(requirement)
    if ref is None:
        raise InstallConfigError(
            f"{path}: dependency {requirement!r} has no git ref; pin a "
            "40-character commit SHA, a version tag such as 'v1.2.3', or a "
            "dedicated integration/release branch ('rpent', 'rpent_v<M>.<m>', "
            "'rpent/<name>', or 'release/v<M>.<m>')"
        )
    if (
        _FULL_COMMIT.fullmatch(ref)
        or _VERSION_TAG.fullmatch(ref)
        or _INTEGRATION_BRANCH.fullmatch(ref)
    ):
        return
    raise InstallConfigError(
        f"{path}: dependency {requirement!r} pins git ref {ref!r}; use a "
        "40-character commit SHA, a version tag such as 'v1.2.3', or a "
        "dedicated integration/release branch ('rpent', 'rpent_v<M>.<m>', "
        "'rpent/<name>', or 'release/v<M>.<m>') instead of a generic branch"
    )


def load_component(path: Path, kind: str) -> Component:
    """Load and validate one ``<component>/component.toml`` manifest."""
    data = _read_toml(path)
    schema = data.get("schema")
    if schema != SCHEMA:
        raise InstallConfigError(
            f"{path}: unsupported schema {schema!r}; expected {SCHEMA}"
        )
    unknown = sorted(set(data) - FIELDS)
    if unknown:
        raise InstallConfigError(
            f"{path}: unknown field(s): {', '.join(unknown)}; "
            f"expected {', '.join(sorted(FIELDS))}"
        )
    expected_name = f"{prefix(kind)}{path.parent.name}"
    name = data.get("name")
    if not isinstance(name, str) or not name:
        raise InstallConfigError(f"{path}: 'name' must be a non-empty string")
    if name != expected_name:
        raise InstallConfigError(
            f"{path}: name {name!r} must be {expected_name!r} to match its location"
        )
    probe = data.get("probe")
    if not isinstance(probe, str) or not probe:
        raise InstallConfigError(f"{path}: 'probe' must be a non-empty import path")

    directory = path.parent
    declared_overrides = _string_list(data, "overrides", path)
    overrides: list[Path] = []
    for relative in declared_overrides:
        if Path(relative).is_absolute():
            raise InstallConfigError(
                f"{path}: override {relative!r} must be relative to {directory}"
            )
        override = directory / relative
        if not override.is_file():
            raise InstallConfigError(f"{path}: override file not found: {override}")
        overrides.append(override)

    index_strategy = _optional_string(data, "index-strategy", path)
    if index_strategy is not None and index_strategy not in INDEX_STRATEGIES:
        raise InstallConfigError(
            f"{path}: 'index-strategy' must be one of {', '.join(INDEX_STRATEGIES)}"
        )
    manylinux = _optional_string(data, "manylinux", path) or DEFAULT_MANYLINUX
    if not _MANYLINUX.fullmatch(manylinux):
        raise InstallConfigError(
            f"{path}: 'manylinux' must name a glibc baseline such as '2_31'"
        )
    planner = data.get("planner", False)
    if not isinstance(planner, bool):
        raise InstallConfigError(f"{path}: 'planner' must be true or false")

    dependencies = _string_list(data, "dependencies", path)
    for requirement in dependencies:
        _check_git_dependency(requirement, path)

    return Component(
        name=name,
        kind=kind,
        path=directory,
        manifest=path,
        probe=probe,
        python=_optional_string(data, "python", path),
        manylinux=manylinux,
        planner=planner,
        requires=_string_list(data, "requires", path),
        imports=_string_list(data, "imports", path),
        env_conflicts=_string_list(data, "env_conflicts", path),
        dependencies=dependencies,
        overrides=tuple(overrides),
        index_url=_optional_string(data, "index-url", path),
        extra_index_urls=_string_list(data, "extra-index-urls", path),
        index_strategy=index_strategy,
        manifest_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
    )


def _component_manifests(root: Path) -> Iterable[tuple[Path, str]]:
    seen: set[Path] = set()
    for pattern, kind in COMPONENT_ROOTS:
        for manifest in sorted(root.glob(pattern)):
            if manifest.parent.name.startswith((".", "_")):
                continue
            resolved = manifest.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            yield manifest, kind


def _check_references(components: dict[str, Component]) -> None:
    """Validate cross-component references declared in manifests."""
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
                    f"{component.manifest_path}: imports is only allowed for robot "
                    f"components, not {component.kind!r}"
                )
            if components[other].kind != "robot":
                raise InstallConfigError(
                    f"{component.manifest_path}: imports {other!r} must name a "
                    f"robot component, not {components[other].kind!r}"
                )
            if (
                other not in component.requires
                and component.name not in components[other].requires
            ):
                raise InstallConfigError(
                    f"{component.manifest_path}: imports {other!r}, but neither "
                    "component requires the other"
                )


def discover_components(root: Path | None = None) -> dict[str, Component]:
    """Load and validate every component manifest under ``root``."""
    repo_root = (root or get_repo_root()).resolve()
    components: dict[str, Component] = {}
    for manifest, kind in _component_manifests(repo_root):
        component = load_component(manifest, kind)
        if component.name in components:
            raise InstallConfigError(
                f"component {component.name!r} is declared twice: "
                f"{components[component.name].manifest_path} and {manifest}"
            )
        components[component.name] = component
    _check_references(components)
    return components


def discover(root: Path | None = None) -> Catalog:
    """Return every component available in one checkout."""
    repo_root = (root or get_repo_root()).resolve()
    return Catalog(root=repo_root, components=discover_components(repo_root))
