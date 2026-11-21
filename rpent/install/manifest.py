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

"""Typed component manifests (``component.toml``).

A component is the only install unit. Its name carries both its kind and where
its code lives:

==============  ================================  ====================
prefix          location                          example
==============  ================================  ====================
``robot-``      ``robots/<name>/``                ``robot-alpha``
``sim-``        ``components/sim/<name>/``        ``sim-alpha``
``model-``      ``components/models/<name>/``     ``model-openpi``
``tool-``       ``components/tools/<name>/``      ``tool-flywheel``
``infra-``      ``components/infra/<name>/``      ``infra-unit_cpu``
==============  ================================  ====================

Every component is one directory holding its ``component.toml``; the directory
also holds the component's code unless it owns none, which infra components and
a separately packaged simulator variant do. An environment contains exactly the
components named on the command line plus the closure of their ``requires``. A
manifest owns its component's packages and nothing else adds or removes them.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rpent.install.errors import InstallConfigError

try:  # Python 3.11+
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10
    import tomli as tomllib

#: Manifest layout version. A stale manifest fails to load instead of being
#: read as if its fields still described an installation.
SCHEMA = 2
#: Component kinds in the order ``rpent-install list`` prints them.
KINDS = ("robot", "sim", "model", "tool", "infra")
#: Kind -> the directory holding one ``<name>/component.toml`` per component,
#: relative to the checkout root.
COMPONENT_DIRECTORIES = {
    "robot": "robots",
    "sim": "components/sim",
    "model": "components/models",
    "tool": "components/tools",
    "infra": "components/infra",
}
#: ``uv``'s multi-index strategies; see ``--index-strategy`` in ``uv pip compile``.
INDEX_STRATEGIES = ("first-index", "unsafe-first-match", "unsafe-best-match")
#: The glibc baseline a component's packages are resolved against when it does
#: not ask for a newer one. Current GPU wheels are published for baselines newer
#: than uv's default ``manylinux_2_17`` target.
DEFAULT_MANYLINUX = "2_31"
#: Fields a manifest may declare. Anything else is a typo or belongs to a layout
#: the schema version does not describe, so it is rejected rather than silently
#: ignored.
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
#: A git dependency names a commit or a version tag. A branch moves, so it would
#: install different code tomorrow; a tag can move too, but a version tag is how
#: upstreams name releases and the manifest accepts that trade.
_FULL_COMMIT = re.compile(r"^[0-9a-f]{40}$")
_VERSION_TAG = re.compile(r"^v?\d[\w.+-]*$")
_MANYLINUX = re.compile(r"^\d+_\d+$")


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


def prefix(kind: str) -> str:
    """Return the name prefix that marks ``kind``."""
    return f"{kind}-"


def _schema(data: dict[str, Any], path: Path) -> None:
    schema = data.get("schema")
    if schema != SCHEMA:
        raise InstallConfigError(
            f"{path}: unsupported schema {schema!r}; expected {SCHEMA}"
        )


def _fields(data: dict[str, Any], path: Path) -> None:
    unknown = sorted(set(data) - FIELDS)
    if unknown:
        raise InstallConfigError(
            f"{path}: unknown field(s): {', '.join(unknown)}; "
            f"expected {', '.join(sorted(FIELDS))}"
        )


def _name(data: dict[str, Any], path: Path, expected: str) -> str:
    """Return the component name and check it matches ``expected``."""
    value = data.get("name")
    if not isinstance(value, str) or not value:
        raise InstallConfigError(f"{path}: 'name' must be a non-empty string")
    if value != expected:
        raise InstallConfigError(
            f"{path}: name {value!r} must be {expected!r}, so the name says "
            "the component kind and matches its location"
        )
    return value


def _optional_string(data: dict[str, Any], key: str, path: Path) -> str | None:
    value = data.get(key)
    if value is None:
        return None
    if not isinstance(value, str) or not value:
        raise InstallConfigError(f"{path}: '{key}' must be a non-empty string")
    return value


def _string_list(
    data: dict[str, Any], key: str, path: Path, *, required: bool = False
) -> tuple[str, ...]:
    value = data.get(key)
    if value is None:
        if required:
            raise InstallConfigError(f"{path}: missing required field '{key}'")
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
    target = requirement[marker + len("git+") :].strip()
    _, separator, ref = target.rpartition("@")
    if not separator:
        return None
    return ref.split("#", 1)[0] or None


def _check_git_dependency(requirement: str, path: Path) -> None:
    """Reject a branch ref; a commit or a version tag names installable code."""
    if "git+" not in requirement:
        return
    ref = _git_ref(requirement)
    if ref is None:
        raise InstallConfigError(
            f"{path}: dependency {requirement!r} has no git ref; pin the "
            "40-character commit or a version tag such as 'v1.2.3' to install"
        )
    if _FULL_COMMIT.fullmatch(ref) or _VERSION_TAG.fullmatch(ref):
        return
    raise InstallConfigError(
        f"{path}: dependency {requirement!r} pins git ref {ref!r}; use the "
        "40-character commit or a version tag such as 'v1.2.3' instead. A "
        "branch moves, so it does not name the code an installation contains"
    )


def _dependency_list(
    data: dict[str, Any], key: str, path: Path, *, required: bool = False
) -> tuple[str, ...]:
    """Read a dependency list and validate its direct git references."""
    values = _string_list(data, key, path, required=required)
    for value in values:
        _check_git_dependency(value, path)
    return values


@dataclass(frozen=True)
class Component:
    """One install unit declared by a manifest."""

    name: str
    kind: str
    #: The component's directory: it holds the manifest and, unless the
    #: component owns no code, its package.
    path: Path
    manifest: Path
    probe: str
    python: str | None
    #: The oldest glibc baseline this component's packages are published for.
    manylinux: str
    #: Whether the component imports ``rpent[planner]``. The environment then
    #: installs the planner layer on top of the ``rpent`` runtime layer.
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
        """Return whether this component installs packages without owning code."""
        return self.kind == "infra"


def load_component(path: Path, kind: str) -> Component:
    """Load one ``<component directory>/component.toml``.

    The name must be ``kind``'s prefix plus the directory name, so a component
    cannot sit in one place and call itself another. A manifest that is invalid
    or does not match its directory raises ``InstallConfigError``.
    """
    data = _read_toml(path)
    _schema(data, path)
    _fields(data, path)
    name = _name(data, path, f"{prefix(kind)}{path.parent.name}")
    return _component(data, path, name=name, kind=kind)


def _component(
    data: dict[str, Any],
    path: Path,
    *,
    name: str,
    kind: str,
) -> Component:
    """Build a component from parsed data, validating shared fields."""
    directory = path.parent
    probe = data.get("probe")
    if not isinstance(probe, str) or not probe:
        raise InstallConfigError(f"{path}: 'probe' must be a non-empty import path")
    declared_overrides = _string_list(data, "overrides", path)
    for relative in declared_overrides:
        if Path(relative).is_absolute():
            raise InstallConfigError(
                f"{path}: override {relative!r} must be relative to the "
                "component directory, so the checkout can move"
            )
    overrides = tuple(directory / relative for relative in declared_overrides)
    for override in overrides:
        if not override.is_file():
            raise InstallConfigError(f"{path}: override file not found: {override}")
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
        dependencies=_dependency_list(data, "dependencies", path),
        overrides=overrides,
        index_url=_optional_string(data, "index-url", path),
        extra_index_urls=_string_list(data, "extra-index-urls", path),
        index_strategy=index_strategy,
        manifest_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
    )
