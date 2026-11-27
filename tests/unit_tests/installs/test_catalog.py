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

"""Contracts over the checked-in component manifests and repository layout."""

import pytest

from rpent.install.installer import probe_modules
from rpent.install.manifest import KINDS
from rpent.install.resolve import (
    SUPPORTED_PYTHONS,
    read_core_dependencies,
    read_planner_dependencies,
    resolve,
)


def test_every_component_manifest_lives_where_its_name_declares_it(catalog):
    for name, component in sorted(catalog.components.items()):
        assert component.name == name
        assert name.startswith(f"{component.kind}-")
        assert component.manifest == component.path / "component.toml"


def test_every_kind_has_at_least_one_component(catalog):
    assert {c.kind for c in catalog.components.values()} == set(KINDS)


def test_every_component_resolves_with_its_requires_closure(catalog):
    core = read_core_dependencies(catalog.root)
    planner = read_planner_dependencies(catalog.root)
    for name in sorted(catalog.components):
        resolution = resolve(
            [name], catalog, core_dependencies=core, planner_dependencies=planner
        )
        assert resolution.ok, f"component {name!r}: {'; '.join(resolution.errors)}"
        assert name in resolution.components


def test_documented_multi_component_and_variant_installs_resolve(catalog):
    core = read_core_dependencies(catalog.root)
    planner = read_planner_dependencies(catalog.root)

    # Standalone OpenPI server requires infra-rlinf alongside model-openpi.
    openpi = resolve(
        ["model-openpi", "infra-rlinf"],
        catalog,
        core_dependencies=core,
        planner_dependencies=planner,
    )
    assert openpi.ok, openpi.errors

    # LIBERO-plus provides a single-environment robot component.
    libero_plus = resolve(
        ["robot-libero_plus"],
        catalog,
        core_dependencies=core,
        planner_dependencies=planner,
    )
    assert libero_plus.ok, libero_plus.errors
    assert "sim-libero_plus" in libero_plus.components
    assert "sim-libero" not in libero_plus.components

    # sim-robodojo carries its own RLinf fork and overrides for standalone use.
    sim_robodojo = catalog.components["sim-robodojo"]
    assert any(dep.startswith("rlinf @") for dep in sim_robodojo.dependencies)
    assert sim_robodojo.overrides


@pytest.mark.parametrize("kind", ["robot", "sim", "model", "tool"])
def test_every_probe_names_modules_the_install_can_import(catalog, kind):
    for component in catalog.components.values():
        if component.kind != kind:
            continue
        modules = probe_modules(component.probe, component.kind, source=component.path)
        if component.probe.split(".")[0] not in {"robots", "components"}:
            assert modules == (component.probe,)
            continue
        for module in modules:
            path = catalog.root / module.replace(".", "/")
            assert path.is_dir() or path.with_suffix(".py").is_file(), (
                f"{component.name}: probe {component.probe!r} imports missing {module!r}"
            )
            assert path.is_relative_to(component.path)


def test_core_and_planner_dependencies_are_disjoint(catalog):
    core = read_core_dependencies(catalog.root)
    planner = read_planner_dependencies(catalog.root)
    assert core and planner
    assert not set(core) & set(planner)


def test_supported_pythons_mirrors_requires_python(catalog):
    from packaging.specifiers import SpecifierSet

    try:
        import tomllib
    except ImportError:
        import tomli as tomllib

    with (catalog.root / "pyproject.toml").open("rb") as file:
        spec = SpecifierSet(tomllib.load(file)["project"]["requires-python"])
    minors = {f"3.{m}" for m in range(8, 14) if spec.contains(f"3.{m}")}
    assert set(SUPPORTED_PYTHONS) == minors
