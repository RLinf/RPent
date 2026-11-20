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

"""Component targets resolve into one environment's contents and diagnostics."""

from rpent.install.resolve import resolve


def _catalog(make_catalog, *components):
    return make_catalog(components=list(components))


def test_requires_closure_adds_components(make_component, make_catalog):
    robot = make_component("robot-alpha", requires=["model-sam3"])
    sam3 = make_component("model-sam3")
    catalog = _catalog(make_catalog, robot, sam3)

    resolution = resolve(["robot-alpha"], catalog)

    assert resolution.ok, resolution.errors
    assert sorted(resolution.components) == ["model-sam3", "robot-alpha"]
    assert resolution.targets == ("robot-alpha",)


def test_targets_are_sorted_and_deduplicated(make_component, make_catalog):
    beta = make_component("robot-beta")
    alpha = make_component("robot-alpha")
    catalog = _catalog(make_catalog, alpha, beta)

    resolution = resolve(["robot-beta", "robot-alpha", "robot-beta"], catalog)

    assert resolution.ok
    assert resolution.targets == ("robot-alpha", "robot-beta")


def test_mutual_requires_is_a_symmetric_closure(make_component, make_catalog):
    # `requires` describes co-installation, which is symmetric; a mutual pair
    # must resolve once per component instead of being rejected as a cycle.
    alpha = make_component("robot-alpha", requires=["robot-beta"])
    beta = make_component("robot-beta", requires=["robot-alpha"])
    catalog = _catalog(make_catalog, alpha, beta)

    resolution = resolve(["robot-alpha"], catalog)

    assert resolution.ok
    assert sorted(resolution.components) == ["robot-alpha", "robot-beta"]


def test_unknown_target_is_reported(make_component, make_catalog):
    catalog = _catalog(make_catalog, make_component("robot-alpha"))

    resolution = resolve(["robot-missing"], catalog)

    assert not resolution.ok
    assert "unknown component 'robot-missing'" in resolution.errors[0]
    assert "robot-alpha" in resolution.errors[0]


def test_no_targets_is_an_error(make_catalog):
    resolution = resolve([], _catalog(make_catalog))

    assert not resolution.ok
    assert "name at least one component" in resolution.errors[0]


def test_python_ranges_are_checked(make_component, make_catalog):
    robot = make_component("robot-alpha", python=">=3.12")
    catalog = _catalog(make_catalog, robot)

    resolution = resolve(["robot-alpha"], catalog, python="3.11.9")

    assert not resolution.ok
    assert any("requires Python >=3.12" in error for error in resolution.errors)


def test_without_python_a_supported_range_resolves(make_component, make_catalog):
    robot = make_component("robot-alpha", python=">=3.12")

    resolution = resolve(["robot-alpha"], _catalog(make_catalog, robot))

    assert resolution.ok, resolution.errors


def test_without_python_an_unsupported_range_is_reported(make_component, make_catalog):
    # A check does not create the environment, but a closure no supported
    # interpreter can run must fail here: ``install`` would select one anyway
    # and reject the component after the command has started.
    robot = make_component("robot-alpha", python=">=3.13")

    resolution = resolve(["robot-alpha"], _catalog(make_catalog, robot))

    assert not resolution.ok
    assert any("requires Python >=3.13" in error for error in resolution.errors)


def test_without_python_disjoint_ranges_are_reported(make_component, make_catalog):
    old = make_component("robot-old", python=">=3.10,<3.11")
    new = make_component("robot-new", python=">=3.11,<3.12")

    resolution = resolve(["robot-old", "robot-new"], _catalog(make_catalog, old, new))

    assert not resolution.ok
    for name in ("robot-old", "robot-new"):
        assert any(
            name in error and "requires Python" in error for error in resolution.errors
        )


def test_a_component_conflict_in_one_environment_is_reported(
    make_component, make_catalog
):
    # One side naming the other is enough; resolution checks both directions.
    alpha = make_component("robot-alpha", env_conflicts=["robot-beta"])
    beta = make_component("robot-beta")
    catalog = _catalog(make_catalog, alpha, beta)

    resolution = resolve(["robot-alpha", "robot-beta"], catalog)

    assert not resolution.ok
    assert "cannot share an environment" in resolution.errors[0]
    assert "`rpent-install install robot-alpha`" in resolution.errors[0]
    assert "`rpent-install install robot-beta`" in resolution.errors[0]


def test_conflict_details_name_the_disagreeing_pins(make_component, make_catalog):
    alpha = make_component(
        "robot-alpha",
        env_conflicts=["model-molmo"],
        dependencies=["transformers==4.57.0"],
    )
    molmo = make_component("model-molmo", dependencies=["transformers==5.0.1"])
    catalog = _catalog(make_catalog, alpha, molmo)

    resolution = resolve(["robot-alpha", "model-molmo"], catalog)

    assert "robot-alpha: transformers ==4.57.0" in resolution.errors[0]
    assert "model-molmo: transformers ==5.0.1" in resolution.errors[0]


def test_conflict_reaches_components_pulled_in_by_requires(
    make_component, make_catalog
):
    robot = make_component("robot-alpha", requires=["model-policy"])
    policy = make_component("model-policy", env_conflicts=["model-grounder"])
    grounder = make_component("model-grounder")
    catalog = _catalog(make_catalog, robot, policy, grounder)

    resolution = resolve(["robot-alpha", "model-grounder"], catalog)

    assert not resolution.ok
    assert any(
        "model-grounder" in error and "model-policy" in error
        for error in resolution.errors
    )


def test_declared_conflicts_in_separate_environments_are_allowed(
    make_component, make_catalog
):
    alpha = make_component("robot-alpha", env_conflicts=["sim-robotwin"])
    robotwin = make_component("sim-robotwin", env_conflicts=["robot-alpha"])
    catalog = _catalog(make_catalog, alpha, robotwin)

    assert resolve(["robot-alpha"], catalog).ok
    assert resolve(["sim-robotwin"], catalog).ok


def test_requiring_a_conflicting_component_is_rejected(make_component, make_catalog):
    alpha = make_component(
        "robot-alpha", requires=["infra-rlinf"], env_conflicts=["infra-rlinf"]
    )
    rlinf = make_component("infra-rlinf")
    catalog = _catalog(make_catalog, alpha, rlinf)

    resolution = resolve(["robot-alpha"], catalog)

    assert not resolution.ok
    assert any("declare env_conflicts" in error for error in resolution.errors)


def test_incompatible_pins_are_reported(make_component, make_catalog):
    alpha = make_component("robot-alpha", dependencies=["torch==2.7.0"])
    beta = make_component("robot-beta", dependencies=["torch==2.8.0"])
    catalog = _catalog(make_catalog, alpha, beta)

    resolution = resolve(["robot-alpha", "robot-beta"], catalog)

    assert any("conflicting pins for 'torch'" in error for error in resolution.errors)


def test_core_dependencies_take_part_in_pin_checks(make_component, make_catalog):
    alpha = make_component("robot-alpha", dependencies=["numpy==2.1.0"])

    resolution = resolve(
        ["robot-alpha"],
        _catalog(make_catalog, alpha),
        core_dependencies=("numpy==2.0.0",),
    )

    assert any("conflicting pins for 'numpy'" in error for error in resolution.errors)
    assert any("rpent: numpy ==2.0.0" in error for error in resolution.errors)


def test_planner_dependencies_take_part_in_pin_checks(make_component, make_catalog):
    alpha = make_component(
        "robot-alpha", planner=True, dependencies=["pydantic==2.9.0"]
    )

    resolution = resolve(
        ["robot-alpha"],
        _catalog(make_catalog, alpha),
        planner_dependencies=("pydantic==2.8.0",),
    )

    assert any("conflicting pins for 'pydantic'" in error for error in resolution.errors)
    assert any("rpent[planner]: pydantic ==2.8.0" in error for error in resolution.errors)


def test_planner_pins_do_not_constrain_a_runtime_only_environment(
    make_component, make_catalog
):
    # A simulator environment installs neither layer's planner packages, so a
    # separate pin there is not a conflict.
    alpha = make_component("sim-alpha", dependencies=["pydantic==2.9.0"])

    resolution = resolve(
        ["sim-alpha"],
        _catalog(make_catalog, alpha),
        planner_dependencies=("pydantic==2.8.0",),
    )

    assert resolution.ok, resolution.errors


def test_a_planner_component_needs_the_planner_extra(make_component, make_catalog):
    alpha = make_component("robot-alpha", planner=True)

    resolution = resolve(["robot-alpha"], _catalog(make_catalog, alpha))

    assert any(
        "[project.optional-dependencies].planner is missing" in error
        for error in resolution.errors
    )


def test_overlapping_specifiers_are_compatible(make_component, make_catalog):
    alpha = make_component("robot-alpha", dependencies=["huggingface-hub>=0.24,<1"])
    beta = make_component("robot-beta", dependencies=["huggingface-hub>=0.34,<2"])
    catalog = _catalog(make_catalog, alpha, beta)

    assert resolve(["robot-alpha", "robot-beta"], catalog).ok


def test_range_pins_are_left_to_uv(make_component, make_catalog):
    # `>1.0` and `<1.0.1` overlap at 1.0.post1; sampling boundaries used to
    # report a false conflict, so ranges are not compared statically.
    alpha = make_component("robot-alpha", dependencies=["pkg>1.0"])
    beta = make_component("robot-beta", dependencies=["pkg<1.0.1"])
    catalog = _catalog(make_catalog, alpha, beta)

    assert resolve(["robot-alpha", "robot-beta"], catalog).ok


def test_marker_scoped_pins_select_one_wheel_and_are_ignored(
    make_component, make_catalog
):
    marker = ' ; python_version == "3.11"'
    wheel = f"franky @ https://example.test/franky.whl{marker}"
    alpha = make_component("robot-alpha", dependencies=[wheel])
    beta = make_component("robot-beta", dependencies=[wheel])
    catalog = _catalog(make_catalog, alpha, beta)

    assert resolve(["robot-alpha", "robot-beta"], catalog).ok


def test_different_url_pins_are_reported(make_component, make_catalog):
    alpha = make_component("robot-alpha", dependencies=["pkg @ https://a.test/pkg.whl"])
    beta = make_component("robot-beta", dependencies=["pkg @ https://b.test/pkg.whl"])
    catalog = _catalog(make_catalog, alpha, beta)

    errors = resolve(["robot-alpha", "robot-beta"], catalog).errors

    assert any("conflicting pins for 'pkg'" in error for error in errors)


def test_dependencies_are_deduplicated_in_component_order(make_component, make_catalog):
    alpha = make_component("robot-alpha", dependencies=["numpy>=2", "torch==2.8.0"])
    beta = make_component("model-beta", dependencies=["numpy>=2", "pillow"])
    catalog = _catalog(make_catalog, alpha, beta)

    resolution = resolve(["robot-alpha", "model-beta"], catalog)

    assert resolution.dependencies() == ("numpy>=2", "pillow", "torch==2.8.0")


def test_index_settings_merge_across_components(make_component, make_catalog):
    alpha = make_component(
        "infra-unit_cpu",
        index_url="https://download.pytorch.org/whl/cpu",
        extra_index_urls=("https://pypi.org/simple",),
        index_strategy="unsafe-first-match",
    )
    beta = make_component(
        "infra-dev",
        index_url="https://download.pytorch.org/whl/cpu",
        extra_index_urls=("https://pypi.org/simple", "https://extra.test/simple"),
        index_strategy="unsafe-first-match",
    )
    catalog = _catalog(make_catalog, alpha, beta)

    resolution = resolve(["infra-unit_cpu", "infra-dev"], catalog)

    assert resolution.ok, resolution.errors
    assert resolution.index_url == "https://download.pytorch.org/whl/cpu"
    assert resolution.extra_index_urls == (
        "https://pypi.org/simple",
        "https://extra.test/simple",
    )
    assert resolution.index_strategy == "unsafe-first-match"


def test_components_selecting_different_indexes_are_reported(
    make_component, make_catalog
):
    alpha = make_component("infra-a", index_url="https://a.test/simple")
    beta = make_component("infra-b", index_url="https://b.test/simple")
    catalog = _catalog(make_catalog, alpha, beta)

    resolution = resolve(["infra-a", "infra-b"], catalog)

    assert not resolution.ok
    assert any("selects index" in error for error in resolution.errors)


def test_components_selecting_different_index_strategies_are_reported(
    make_component, make_catalog
):
    alpha = make_component("infra-a", index_strategy="unsafe-first-match")
    beta = make_component("infra-b", index_strategy="first-index")
    catalog = _catalog(make_catalog, alpha, beta)

    resolution = resolve(["infra-a", "infra-b"], catalog)

    assert any("selects index strategy" in error for error in resolution.errors)


def test_two_components_overriding_one_package_are_reported(
    make_component, make_catalog, tmp_path
):
    # uv merges every override into one resolution, so the winner would depend
    # on file order instead of on which component declared the constraint.
    shared = tmp_path / "shared.txt"
    shared.write_text("numpy==2.0.0\n", encoding="utf-8")
    alpha = make_component("robot-alpha", overrides=[shared])
    beta = make_component("model-beta", overrides=[shared])
    catalog = _catalog(make_catalog, alpha, beta)

    resolution = resolve(["robot-alpha", "model-beta"], catalog)

    assert not resolution.ok
    assert any("both override 'numpy'" in error for error in resolution.errors)


def test_marker_scoped_overrides_select_one_target_and_are_ignored(
    make_component, make_catalog, tmp_path
):
    linux = tmp_path / "linux.txt"
    linux.write_text('torch==2.4.0; sys_platform == "linux"\n', encoding="utf-8")
    darwin = tmp_path / "darwin.txt"
    darwin.write_text('torch==2.5.0; sys_platform == "darwin"\n', encoding="utf-8")
    alpha = make_component("robot-alpha", overrides=[linux])
    beta = make_component("model-beta", overrides=[darwin])
    catalog = _catalog(make_catalog, alpha, beta)

    resolution = resolve(["robot-alpha", "model-beta"], catalog)

    assert resolution.ok, resolution.errors
