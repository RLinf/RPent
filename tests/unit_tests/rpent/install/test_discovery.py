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

"""Discovery finds every component root and rejects inconsistent declarations."""

import pytest

from rpent.install.discovery import discover, discover_components
from rpent.install.errors import InstallConfigError
from rpent.install.manifest import COMPONENT_DIRECTORIES, SCHEMA


def _component(write_toml, kind, short, *, conflicts=(), requires=(), imports=()):
    name = f"{kind}-{short}"
    rendered = ", ".join(f'"{item}"' for item in conflicts)
    required = ", ".join(f'"{item}"' for item in requires)
    imported = ", ".join(f'"{item}"' for item in imports)
    return write_toml(
        f"{COMPONENT_DIRECTORIES[kind]}/{short}/component.toml",
        f"""
        schema = {SCHEMA}
        name = "{name}"
        probe = "{name.replace("-", ".")}"
        requires = [{required}]
        env_conflicts = [{rendered}]
        imports = [{imported}]
        """,
    )


def _component_without_code(write_toml, kind, short):
    """Write a manifest for a component that installs packages and no code."""
    return write_toml(
        f"{COMPONENT_DIRECTORIES[kind]}/{short}/component.toml",
        f"""
        schema = {SCHEMA}
        name = "{kind}-{short}"
        probe = "{short}"
        """,
    )


def test_discovery_loads_every_kind(tmp_path, write_toml):
    _component(write_toml, "robot", "alpha", conflicts=["robot-beta"])
    _component(write_toml, "robot", "beta", conflicts=["robot-alpha"])
    _component(write_toml, "sim", "alpha")
    _component(write_toml, "model", "alpha")
    _component(write_toml, "tool", "alpha")
    _component(write_toml, "infra", "alpha")

    catalog = discover(tmp_path)

    assert catalog.root == tmp_path.resolve()
    assert sorted(catalog.components) == [
        "infra-alpha",
        "model-alpha",
        "robot-alpha",
        "robot-beta",
        "sim-alpha",
        "tool-alpha",
    ]
    assert catalog.components["sim-alpha"].kind == "sim"
    assert catalog.components["infra-alpha"].is_infra


def test_a_component_without_code_is_discovered_from_its_directory(
    tmp_path, write_toml
):
    _component_without_code(write_toml, "sim", "variant_plus")
    _component_without_code(write_toml, "infra", "runtime")

    catalog = discover(tmp_path)

    assert sorted(catalog.components) == ["infra-runtime", "sim-variant_plus"]
    assert catalog.components["sim-variant_plus"].kind == "sim"
    assert (
        catalog.components["sim-variant_plus"].path
        == (tmp_path / "components" / "sim" / "variant_plus").resolve()
    )


def test_component_name_must_match_its_directory(tmp_path, write_toml):
    write_toml(
        "robots/alpha/component.toml",
        f"""
        schema = {SCHEMA}
        name = "robot-beta"
        probe = "robots.alpha"
        """,
    )

    with pytest.raises(InstallConfigError, match="must be 'robot-alpha'"):
        discover_components(tmp_path)


def test_component_manifest_in_the_wrong_kind_of_directory_is_rejected(
    tmp_path, write_toml
):
    write_toml(
        "components/models/alpha/component.toml",
        f"""
        schema = {SCHEMA}
        name = "robot-alpha"
        probe = "robots.alpha"
        """,
    )

    with pytest.raises(InstallConfigError, match="must be 'model-alpha'"):
        discover_components(tmp_path)


def test_a_component_name_is_derived_from_its_location(tmp_path, write_toml):
    _component(write_toml, "robot", "alpha")
    _component(write_toml, "model", "alpha")

    components = discover_components(tmp_path)

    assert components["robot-alpha"].path == (tmp_path / "robots" / "alpha").resolve()
    assert (
        components["model-alpha"].path
        == (tmp_path / "components" / "models" / "alpha").resolve()
    )
    assert components["robot-alpha"].manifest.name == "component.toml"


def test_conflicts_must_reference_known_components(tmp_path, write_toml):
    _component(write_toml, "robot", "alpha", conflicts=["robot-missing"])

    with pytest.raises(InstallConfigError, match="unknown component"):
        discover_components(tmp_path)


def test_conflicts_must_not_name_the_component_itself(tmp_path, write_toml):
    _component(write_toml, "robot", "alpha", conflicts=["robot-alpha"])

    with pytest.raises(InstallConfigError, match="names the component itself"):
        discover_components(tmp_path)


def test_requires_must_reference_known_components(tmp_path, write_toml):
    _component(write_toml, "robot", "alpha", requires=["model-missing"])

    with pytest.raises(InstallConfigError, match="requires references unknown"):
        discover_components(tmp_path)


def test_imports_must_be_declared_in_requires(tmp_path, write_toml):
    _component(write_toml, "robot", "alpha", imports=["robot-beta"])
    _component(write_toml, "robot", "beta")

    with pytest.raises(InstallConfigError, match="neither.*requires the other"):
        discover_components(tmp_path)


def test_imports_may_rely_on_the_other_components_requires(tmp_path, write_toml):
    _component(write_toml, "robot", "alpha", imports=["robot-beta"])
    _component(write_toml, "robot", "beta", requires=["robot-alpha"])

    components = discover_components(tmp_path)

    assert components["robot-alpha"].imports == ("robot-beta",)


def test_imports_must_name_a_robot_component(tmp_path, write_toml):
    _component(
        write_toml, "robot", "alpha", requires=["sim-alpha"], imports=["sim-alpha"]
    )
    _component(write_toml, "sim", "alpha")

    with pytest.raises(InstallConfigError, match="must name a robot component"):
        discover_components(tmp_path)


def test_imports_are_only_allowed_on_robot_slices(tmp_path, write_toml):
    _component(
        write_toml, "model", "alpha", requires=["model-beta"], imports=["model-beta"]
    )
    _component(write_toml, "model", "beta", requires=["model-alpha"])

    with pytest.raises(InstallConfigError, match="imports is only for robot slices"):
        discover_components(tmp_path)


def test_apple_double_metadata_is_ignored(tmp_path, write_toml):
    _component(write_toml, "robot", "alpha")
    _component_without_code(write_toml, "infra", "dev")
    # A network filesystem stores macOS extended attributes as ``._<name>``
    # entries next to the original, and the layout gives a leading underscore
    # the same meaning; neither names a component.
    write_toml(
        "components/infra/._dev/component.toml",
        f"""
        schema = {SCHEMA}
        name = "infra-._dev"
        probe = "json"
        """,
    )
    (tmp_path / "robots" / "._alpha").mkdir()

    assert sorted(discover_components(tmp_path)) == ["infra-dev", "robot-alpha"]


def test_empty_checkout_has_no_components(tmp_path):
    assert discover_components(tmp_path) == {}
