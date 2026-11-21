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

"""The checked-in component manifests agree with each other and with the tree."""

import pytest

from rpent.install.installer import probe_modules
from rpent.install.manifest import KINDS
from rpent.install.requirements import (
    read_core_dependencies,
    read_planner_dependencies,
)
from rpent.install.resolve import SUPPORTED_PYTHONS, resolve


def test_every_component_manifest_lives_where_its_name_declares_it(catalog):
    for name, component in sorted(catalog.components.items()):
        assert component.name == name
        assert name.startswith(f"{component.kind}-")
        # One directory per component holds its manifest and, unless it owns no
        # code, its package.
        assert component.manifest == component.path / "component.toml"


def test_every_kind_has_at_least_one_component(catalog):
    kinds = {component.kind for component in catalog.components.values()}

    assert kinds == set(KINDS)


def test_every_component_resolves_with_its_requires_closure(catalog):
    core = read_core_dependencies(catalog.root)
    planner = read_planner_dependencies(catalog.root)

    for name in sorted(catalog.components):
        resolution = resolve(
            [name],
            catalog,
            core_dependencies=core,
            planner_dependencies=planner,
        )

        assert resolution.ok, f"component {name!r}: {'; '.join(resolution.errors)}"
        assert name in resolution.components


@pytest.mark.parametrize("kind", ["robot", "sim", "model", "tool"])
def test_every_probe_names_modules_the_install_can_import(catalog, kind):
    # A probe that names checkout code must point inside the component's own
    # directory, and every module the probe derives for its kind must be one:
    # the install appends the role modules to whatever the manifest names, so a
    # probe that already ends in a role module produces a path that cannot
    # import. Only the roles the component ships are probed, because a tool is
    # either a process with a CLI or a capability library. Probes such as
    # `openpi` name the installed upstream distribution and are checked by the
    # install itself.
    for component in catalog.components.values():
        if component.kind != kind:
            continue
        modules = probe_modules(component.probe, component.kind, source=component.path)
        if component.probe.split(".")[0] not in {"robots", "components"}:
            assert modules == (component.probe,), (
                f"{component.name}: probe {component.probe!r} names an upstream "
                "distribution but gains role modules"
            )
            continue
        for module in modules:
            path = catalog.root / module.replace(".", "/")
            assert path.is_dir() or path.with_suffix(".py").is_file(), (
                f"{component.name}: probe {component.probe!r} imports "
                f"{module!r}, which is not a module in the checkout"
            )
            assert path.is_relative_to(component.path), (
                f"{component.name}: probe {component.probe!r} points outside "
                f"{component.path}"
            )


def test_core_dependencies_are_declared(catalog):
    assert read_core_dependencies(catalog.root)
    assert read_planner_dependencies(catalog.root)
    # The runtime layer must not be the whole distribution: a checkout that
    # declared its planner packages as runtime dependencies would install them
    # into every simulator and model environment.
    assert not set(read_core_dependencies(catalog.root)) & set(
        read_planner_dependencies(catalog.root)
    )


def test_supported_pythons_mirrors_requires_python(catalog):
    # ``choose_python`` selects from this literal, so a change to the package's
    # declared range without changing it would create environments the built
    # distribution does not support.
    from packaging.specifiers import SpecifierSet

    try:
        import tomllib
    except ImportError:  # Python 3.10
        import tomli as tomllib

    with (catalog.root / "pyproject.toml").open("rb") as file:
        requires_python = tomllib.load(file)["project"]["requires-python"]
    specifier = SpecifierSet(requires_python)
    minors = {
        f"3.{minor}" for minor in range(8, 14) if specifier.contains(f"3.{minor}")
    }

    assert set(SUPPORTED_PYTHONS) == minors
