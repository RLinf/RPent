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

"""Fixtures shared by the installer units, repository contracts and tooling."""

from pathlib import Path

import pytest

from rpent.install.manifest import (
    COMPONENT_DIRECTORIES,
    DEFAULT_MANYLINUX,
    KINDS,
    Catalog,
    Component,
    discover,
)

#: Repository root: this file sits one directory below it.
REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def catalog() -> Catalog:
    """Load the checked-in components once per module."""
    return discover(REPO_ROOT)


@pytest.fixture
def write_toml(tmp_path):
    """Write a TOML file and return its path."""

    def _write(relative, text):
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text.strip() + "\n", encoding="utf-8")
        return path

    return _write


@pytest.fixture
def make_component(tmp_path):
    """Build a component without touching the filesystem."""

    def _make(name, **values):
        prefix = name.split("-", 1)[0]
        kind = values.pop("kind", prefix if prefix in KINDS else "robot")
        subject = name.split("-", 1)[1] if "-" in name else name
        directory = tmp_path / COMPONENT_DIRECTORIES[kind] / subject
        return Component(
            name=name,
            kind=kind,
            path=directory,
            manifest=directory / "component.toml",
            probe=values.pop("probe", f"rpent.{subject}"),
            python=values.pop("python", None),
            requires=tuple(values.pop("requires", ())),
            imports=tuple(values.pop("imports", ())),
            env_conflicts=tuple(values.pop("env_conflicts", ())),
            dependencies=tuple(values.pop("dependencies", ())),
            overrides=tuple(values.pop("overrides", ())),
            index_url=values.pop("index_url", None),
            extra_index_urls=tuple(values.pop("extra_index_urls", ())),
            index_strategy=values.pop("index_strategy", None),
            manylinux=values.pop("manylinux", DEFAULT_MANYLINUX),
            planner=values.pop("planner", False),
            manifest_sha256=values.pop("manifest_sha256", name),
        )

    return _make


@pytest.fixture
def make_catalog(tmp_path, make_component):
    """Build a catalog from components."""

    def _make(components=()):
        return Catalog(
            root=Path(tmp_path),
            components={component.name: component for component in components},
        )

    return _make
