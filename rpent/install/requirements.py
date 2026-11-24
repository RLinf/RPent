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

"""Render deterministic requirements files and installation input hashes.

An installation is identified by the hash of every input that decides what it
contains: the target components, every manifest in their ``requires`` closure,
the override files those components apply, and the repository's ``[tool.uv]``
resolution settings. Requirement files are hashed by the lines uv reads, so
editing a comment does not make an installation look stale.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from rpent.install.errors import InstallConfigError
from rpent.install.fingerprint import FINGERPRINT_SCHEMA_FIELD
from rpent.install.resolve import Resolution

try:  # Python 3.11+
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10
    import tomli as tomllib


#: Identifies the layout of :func:`requirements_inputs`. Bump it when a key
#: changes meaning or the repository layout moves the files the inputs name, not
#: merely when one is added: ``doctor`` uses it to tell an installation recorded
#: by an older input mode from one whose inputs drifted. It is recorded under
#: ``FINGERPRINT_SCHEMA_FIELD``, never as ``schema``: that name belongs to the
#: manifest layout version.
FINGERPRINT_SCHEMA = 8


def _read_project(root: Path) -> dict[str, Any]:
    """Return the ``[project]`` table of the checkout's pyproject.toml."""
    path = root / "pyproject.toml"
    try:
        with path.open("rb") as file:
            data = tomllib.load(file)
    except (OSError, UnicodeDecodeError, tomllib.TOMLDecodeError) as exc:
        raise InstallConfigError(
            f"{path}: cannot read core dependencies: {exc}"
        ) from exc
    project = data.get("project")
    if not isinstance(project, dict):
        raise InstallConfigError(f"{path}: missing [project] table")
    return project


def _dependency_strings(value: Any, path: Path, key: str) -> tuple[str, ...]:
    """Return a ``pyproject.toml`` dependency list as a tuple."""
    if not isinstance(value, list) or not all(
        isinstance(item, str) and item for item in value
    ):
        raise InstallConfigError(f"{path}: {key} must be a list of requirement strings")
    return tuple(value)


def read_core_dependencies(root: Path) -> tuple[str, ...]:
    """Return ``[project].dependencies``: the ``rpent`` runtime layer."""
    project = _read_project(root)
    return _dependency_strings(
        project.get("dependencies"), root / "pyproject.toml", "[project].dependencies"
    )


def read_planner_dependencies(root: Path) -> tuple[str, ...]:
    """Return ``[project.optional-dependencies].planner``, or ``()``.

    A checkout without the planner extra returns nothing; ``resolve`` reports a
    component that declares ``planner = true`` there, so a plain environment
    that installs only the runtime layer reads as empty rather than failing.
    """
    project = _read_project(root)
    optional = project.get("optional-dependencies")
    planner = optional.get("planner") if isinstance(optional, dict) else None
    if planner is None:
        return ()
    return _dependency_strings(
        planner, root / "pyproject.toml", "[project.optional-dependencies].planner"
    )


def read_uv_settings(root: Path) -> dict[str, Any]:
    """Return the ``[tool.uv]`` settings that apply to installs in this checkout.

    ``uv`` reads these settings from the project it runs in, so they decide how
    a requirement file resolves.
    """
    path = root / "pyproject.toml"
    try:
        with path.open("rb") as file:
            data = tomllib.load(file)
    except FileNotFoundError:
        return {}
    except (OSError, UnicodeDecodeError, tomllib.TOMLDecodeError) as exc:
        raise InstallConfigError(f"{path}: cannot read [tool.uv]: {exc}") from exc
    tool = data.get("tool")
    settings = tool.get("uv") if isinstance(tool, dict) else None
    if not isinstance(settings, dict):
        return {}
    return dict(settings)


def override_files(resolution: Resolution, root: Path) -> list[tuple[str, Path]]:
    """Return ``(display, path)`` for every override in the environment.

    ``display`` is relative to ``root`` while the file lives in the checkout,
    so an installation fingerprint does not depend on where the checkout is.
    """
    files: list[tuple[str, Path]] = []
    seen: set[Path] = set()
    for component in resolution.components.values():
        for override in component.overrides:
            path = override if override.is_absolute() else root / override
            if path in seen:
                continue
            seen.add(path)
            display = (
                str(path.relative_to(root)) if path.is_relative_to(root) else str(path)
            )
            files.append((display, path))
    # The fingerprint hashes this list, so its order must not depend on the
    # order components were resolved in.
    files.sort(key=lambda entry: entry[0])
    return files


def read_requirement_text(path: Path) -> str | None:
    """Return the text of a requirement file, or ``None`` when it is unreadable.

    A missing or unreadable file is reported as absent so callers can treat it
    as a changed input. A file that is not UTF-8 raises instead: uv rejects it
    too, and hashing bytes nobody can resolve would hide the real problem
    behind a drift report.
    """
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        raise InstallConfigError(
            f"{path}: requirement file is not UTF-8: {exc}"
        ) from exc
    except OSError:
        return None


def _requirement_lines_sha256(text: str) -> str:
    """Hash the lines of a requirement file that a resolver reads.

    Blank lines and comments do not change a resolution, so editing them must
    not make an installation look stale. Every other line is hashed as written,
    because a changed version, marker, URL, or option does change it.
    """
    lines = [
        line.rstrip()
        for line in text.splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    return hashlib.sha256("\n".join(lines).encode("utf-8")).hexdigest()


def _override_inputs(resolution: Resolution, root: Path) -> list[dict[str, str]]:
    """Return the override files an installation applies, with their hashes."""
    entries: list[dict[str, str]] = []
    for display, path in override_files(resolution, root):
        text = read_requirement_text(path)
        if text is None:
            # The installer hands the file to uv, which fails on a missing
            # override; leaving it out keeps the hash true to what was read.
            continue
        entries.append({"path": display, "sha256": _requirement_lines_sha256(text)})
    return entries


def _planner_layer(resolution: Resolution, dependencies: tuple[str, ...]) -> list[str]:
    """Return the planner requirements the environment installs, sorted."""
    if not resolution.uses_planner:
        return []
    return sorted(dict.fromkeys(dependencies))


def render_requirements(
    resolution: Resolution,
    *,
    core_dependencies: tuple[str, ...] = (),
    planner_dependencies: tuple[str, ...] = (),
) -> str:
    """Render a stable requirements file for the environment.

    ``planner_dependencies`` joins the file only when a component in the
    environment declares ``planner = true``.
    """
    targets = ", ".join(resolution.targets)
    lines = [
        f"# Generated by rpent-install for: {targets}.",
        "# Do not edit; change a component manifest and rerun",
        f"# `rpent-install install {targets}`.",
    ]
    core = sorted(dict.fromkeys(core_dependencies))
    if core:
        lines.append("")
        lines.append("# RPent core")
        lines.extend(core)
    planner = _planner_layer(resolution, planner_dependencies)
    if planner:
        lines.append("")
        lines.append("# RPent planner (rpent[planner])")
        lines.extend(planner)
    for name, component in resolution.components.items():
        dependencies = sorted(dict.fromkeys(component.dependencies))
        if not dependencies:
            continue
        lines.append("")
        lines.append(f"# component {name}")
        lines.extend(dependencies)
    return "\n".join(lines) + "\n"


def requirements_inputs(
    resolution: Resolution,
    *,
    root: Path,
    core_dependencies: tuple[str, ...] = (),
    planner_dependencies: tuple[str, ...] = (),
) -> dict[str, Any]:
    """Return the canonical inputs whose hash identifies an installation.

    ``root`` locates component override files and repository uv settings. The
    planner layer is recorded only when the environment installs it, so
    changing ``rpent[planner]`` leaves runtime-only environments current.
    The typed targets are not an input: different target names can describe
    the same closure, and the closure plus the manifests decide the packages.
    """
    # Override paths are recorded relative to ``root``. Resolving it here makes
    # that relative form independent of how the caller spelled the checkout.
    root = root.resolve()
    return {
        FINGERPRINT_SCHEMA_FIELD: FINGERPRINT_SCHEMA,
        "components": {
            name: {"manifest_sha256": component.manifest_sha256}
            for name, component in sorted(resolution.components.items())
        },
        "core_dependencies": sorted(core_dependencies),
        "planner_dependencies": _planner_layer(resolution, planner_dependencies),
        # Override contents can change without changing the manifest.
        "overrides": _override_inputs(resolution, root),
        "index": {
            "url": resolution.index_url,
            "extra_urls": sorted(resolution.extra_index_urls),
            "strategy": resolution.index_strategy,
        },
        "uv_settings": read_uv_settings(root),
    }


def requirements_sha256(
    resolution: Resolution,
    *,
    root: Path,
    core_dependencies: tuple[str, ...] = (),
    planner_dependencies: tuple[str, ...] = (),
) -> str:
    """Hash every input that determines what the installation contains."""
    payload = json.dumps(
        requirements_inputs(
            resolution,
            root=root,
            core_dependencies=core_dependencies,
            planner_dependencies=planner_dependencies,
        ),
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
