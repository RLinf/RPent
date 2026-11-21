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

"""Component manifests parse into install inputs."""

import pytest

from rpent.install.errors import InstallConfigError
from rpent.install.manifest import (
    DEFAULT_MANYLINUX,
    SCHEMA,
    load_component,
)


def test_component_manifest_reads_declarations(write_toml):
    write_toml("robots/alpha/overrides.txt", "torch==2.7")
    path = write_toml(
        "robots/alpha/component.toml",
        f"""
        schema = {SCHEMA}
        name = "robot-alpha"
        probe = "robots.alpha.robot_spec"
        python = ">=3.11,<3.12"
        planner = true
        requires = ["model-openpi"]
        imports = ["robot-beta"]
        env_conflicts = ["robot-beta"]
        overrides = ["overrides.txt"]
        index-url = "https://example.invalid/simple"
        extra-index-urls = ["https://example.invalid/extra"]
        index-strategy = "unsafe-first-match"
        dependencies = ["numpy>=2,<3"]
        """,
    )

    component = load_component(path, "robot")

    assert component.name == "robot-alpha"
    assert component.kind == "robot"
    assert component.path == path.parent
    assert component.python == ">=3.11,<3.12"
    assert component.planner is True
    assert component.requires == ("model-openpi",)
    assert component.imports == ("robot-beta",)
    assert component.env_conflicts == ("robot-beta",)
    assert component.overrides == (path.parent / "overrides.txt",)
    assert component.dependencies == ("numpy>=2,<3",)
    assert component.index_url == "https://example.invalid/simple"
    assert component.extra_index_urls == ("https://example.invalid/extra",)
    assert component.index_strategy == "unsafe-first-match"
    assert not component.is_infra


def test_platforms_is_not_a_manifest_field(write_toml):
    # RPent supports one platform, so a manifest cannot narrow or widen it.
    # The rejected field is a typo or a stale manifest, not a decision.
    path = write_toml(
        "robots/alpha/component.toml",
        f"""
        schema = {SCHEMA}
        name = "robot-alpha"
        probe = "robots.alpha"
        platforms = ["linux-x86_64"]
        """,
    )

    with pytest.raises(InstallConfigError, match="unknown field.*platforms"):
        load_component(path, "robot")


def test_component_name_must_name_kind_and_directory(write_toml):
    path = write_toml(
        "robots/alpha/component.toml",
        f"""
        schema = {SCHEMA}
        name = "robot-beta"
        probe = "robots.beta"
        """,
    )

    with pytest.raises(InstallConfigError, match="must be 'robot-alpha'"):
        load_component(path, "robot")


def test_component_kind_comes_from_the_directory_root(write_toml):
    path = write_toml(
        "components/models/sam3/component.toml",
        f"""
        schema = {SCHEMA}
        name = "model-sam3"
        probe = "components.models.sam3.server"
        """,
    )

    assert load_component(path, "model").kind == "model"


def test_infra_manifest_lives_in_a_directory_without_code(write_toml):
    # An infra component owns no code, but it is still one directory: the
    # manifest is all it contributes.
    path = write_toml(
        "components/infra/dev/component.toml",
        f"""
        schema = {SCHEMA}
        name = "infra-dev"
        probe = "pytest"
        dependencies = ["pytest"]
        """,
    )

    component = load_component(path, "infra")

    assert component.kind == "infra"
    assert component.path == path.parent
    assert component.is_infra
    assert component.dependencies == ("pytest",)


def test_component_name_must_match_its_directory(write_toml):
    path = write_toml(
        "components/infra/dev/component.toml",
        f"""
        schema = {SCHEMA}
        name = "infra-unit"
        probe = "pytest"
        """,
    )

    with pytest.raises(InstallConfigError, match="must be 'infra-dev'"):
        load_component(path, "infra")


def test_component_manifest_must_match_the_kind_of_its_directory(write_toml):
    # The same manifest read as a sim would have to be ``sim-dev``, so a
    # misplaced directory is reported instead of producing a second name.
    path = write_toml(
        "components/infra/dev/component.toml",
        f"""
        schema = {SCHEMA}
        name = "infra-dev"
        probe = "pytest"
        """,
    )

    with pytest.raises(InstallConfigError, match="must be 'sim-dev'"):
        load_component(path, "sim")


def test_a_packaged_simulator_variant_keeps_its_kind(write_toml):
    # A component that installs packages for code it does not own declares the
    # kind it serves, so `check` still reports the right Python and platform
    # range and `list` groups it with its kind.
    path = write_toml(
        "components/sim/libero_plus/component.toml",
        f"""
        schema = {SCHEMA}
        name = "sim-libero_plus"
        probe = "liberoplus"
        """,
    )

    component = load_component(path, "sim")

    assert component.kind == "sim"
    assert component.path == path.parent
    assert not component.is_infra


def test_unknown_fields_and_old_schema_are_rejected(write_toml):
    stale = write_toml(
        "robots/alpha/component.toml",
        """
        schema = 1
        name = "robot-alpha"
        probe = "robots.alpha"
        kind = "robot"
        placement = ["in-env"]
        """,
    )
    typo = write_toml(
        "robots/beta/component.toml",
        f"""
        schema = {SCHEMA}
        name = "robot-beta"
        probe = "robots.beta"
        docs = "docs/beta.rst"
        """,
    )

    with pytest.raises(InstallConfigError, match="unsupported schema 1"):
        load_component(stale, "robot")
    with pytest.raises(InstallConfigError, match="unknown field.*docs"):
        load_component(typo, "robot")


def test_git_dependencies_accept_a_commit_or_a_version_tag(write_toml):
    branch = write_toml(
        "robots/alpha/component.toml",
        f"""
        schema = {SCHEMA}
        name = "robot-alpha"
        probe = "robots.alpha"
        dependencies = ["pkg @ git+https://github.com/example/pkg.git@main"]
        """,
    )
    commit = write_toml(
        "robots/beta/component.toml",
        f"""
        schema = {SCHEMA}
        name = "robot-beta"
        probe = "robots.beta"
        dependencies = [
          "pkg @ git+https://github.com/example/pkg.git@0123456789abcdef0123456789abcdef01234567",
        ]
        """,
    )
    tag = write_toml(
        "robots/gamma/component.toml",
        f"""
        schema = {SCHEMA}
        name = "robot-gamma"
        probe = "robots.gamma"
        dependencies = ["pkg @ git+https://github.com/example/pkg.git@v1.2.3"]
        """,
    )

    with pytest.raises(InstallConfigError, match="pins git ref 'main'"):
        load_component(branch, "robot")
    # A version tag is how upstreams name releases; a branch names code that
    # changes from one day to the next.
    assert load_component(tag, "robot").dependencies == (
        "pkg @ git+https://github.com/example/pkg.git@v1.2.3",
    )
    assert load_component(commit, "robot").dependencies[0].endswith("01234567")


def test_git_dependency_without_a_ref_is_rejected(write_toml):
    path = write_toml(
        "robots/alpha/component.toml",
        f"""
        schema = {SCHEMA}
        name = "robot-alpha"
        probe = "robots.alpha"
        dependencies = ["pkg @ git+https://github.com/example/pkg.git"]
        """,
    )

    with pytest.raises(InstallConfigError, match="has no git ref"):
        load_component(path, "robot")


def test_component_rejects_an_absolute_override(write_toml):
    override = write_toml("robots/alpha/overrides.txt", "torch==2.7")
    path = write_toml(
        "robots/alpha/component.toml",
        f"""
        schema = {SCHEMA}
        name = "robot-alpha"
        probe = "robots.alpha"
        overrides = ["{override}"]
        """,
    )

    with pytest.raises(InstallConfigError, match="must be relative to the component"):
        load_component(path, "robot")


def test_component_rejects_missing_override(write_toml):
    path = write_toml(
        "robots/alpha/component.toml",
        f"""
        schema = {SCHEMA}
        name = "robot-alpha"
        probe = "robots.alpha"
        overrides = ["missing.txt"]
        """,
    )

    with pytest.raises(InstallConfigError, match="override file not found"):
        load_component(path, "robot")


def test_index_strategy_is_optional_and_validated(write_toml):
    default = load_component(
        write_toml(
            "components/infra/dev/component.toml",
            f"""
            schema = {SCHEMA}
            name = "infra-dev"
            probe = "pytest"
            """,
        ),
        "infra",
    )
    invalid = write_toml(
        "components/infra/unit/component.toml",
        f"""
        schema = {SCHEMA}
        name = "infra-unit"
        probe = "pytest"
        index-strategy = "best-match"
        """,
    )

    assert default.index_strategy is None
    with pytest.raises(InstallConfigError, match="'index-strategy' must be one of"):
        load_component(invalid, "infra")


def test_planner_defaults_to_the_runtime_layer_only(write_toml):
    default = load_component(
        write_toml(
            "components/sim/alpha/component.toml",
            f"""
            schema = {SCHEMA}
            name = "sim-alpha"
            probe = "components.sim.alpha"
            """,
        ),
        "sim",
    )
    invalid = write_toml(
        "components/sim/beta/component.toml",
        f"""
        schema = {SCHEMA}
        name = "sim-beta"
        probe = "components.sim.beta"
        planner = "yes"
        """,
    )

    assert default.planner is False
    with pytest.raises(InstallConfigError, match="'planner' must be true or false"):
        load_component(invalid, "sim")


def test_manylinux_defaults_to_the_repository_baseline(write_toml):
    default = load_component(
        write_toml(
            "components/infra/dev/component.toml",
            f"""
            schema = {SCHEMA}
            name = "infra-dev"
            probe = "pytest"
            """,
        ),
        "infra",
    )
    raised = load_component(
        write_toml(
            "components/infra/gpu/component.toml",
            f"""
            schema = {SCHEMA}
            name = "infra-gpu"
            probe = "pytest"
            manylinux = "2_35"
            """,
        ),
        "infra",
    )
    invalid = write_toml(
        "components/infra/unit/component.toml",
        f"""
        schema = {SCHEMA}
        name = "infra-unit"
        probe = "pytest"
        manylinux = "manylinux2.35"
        """,
    )

    assert default.manylinux == DEFAULT_MANYLINUX
    assert raised.manylinux == "2_35"
    with pytest.raises(InstallConfigError, match="'manylinux' must name a glibc"):
        load_component(invalid, "infra")
