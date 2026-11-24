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

"""Changed paths narrow CI scope, and anything unrecognised widens it."""

import json
import subprocess

from tests.tooling import affected


def test_paths_map_to_their_component(make_catalog, make_component):
    robot = make_component("robot-alpha")
    sim = make_component("sim-alpha")
    model = make_component("model-alpha")
    tool = make_component("tool-alpha")
    catalog = make_catalog(components=[robot, sim, model, tool])

    assert affected.affected_components(["robots/alpha/spec.py"], catalog) == {
        "robot-alpha"
    }
    assert affected.affected_components(
        ["components/sim/alpha/env_server.py"], catalog
    ) == {"sim-alpha"}
    assert affected.affected_components(
        ["components/models/alpha/server.py"], catalog
    ) == {"model-alpha"}
    assert affected.affected_components(
        ["components/tools/alpha/exporter.py"], catalog
    ) == {"tool-alpha"}


def test_an_unowned_path_means_everything(make_catalog, make_component):
    catalog = make_catalog(components=[make_component("robot-alpha")])

    assert affected.affected_components(["rpent/session/session.py"], catalog) is None
    assert affected.affected_components(["robots/missing/x.py"], catalog) is None


def test_end_to_end_suite_change_narrows_to_its_robot(
    tmp_path, make_catalog, make_component
):
    catalog = make_catalog(
        components=[
            make_component("robot-alpha"),
            make_component("robot-beta"),
        ]
    )
    for shared in affected.ROBOT_SHARED_TESTS:
        (tmp_path / shared).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / shared).write_text("", encoding="utf-8")

    scope = affected.affected_scope(
        ["tests/e2e_tests/alpha/test_policy_chain.py"], catalog, tmp_path
    )

    assert scope.components == ("robot-alpha",)
    assert scope.installs == ("robot-alpha",)
    assert not scope.full


def test_end_to_end_launcher_change_runs_every_environment(
    tmp_path, make_catalog, make_component
):
    catalog = make_catalog(
        components=[
            make_component("infra-minimal"),
            make_component("robot-alpha"),
        ]
    )

    for path in ("tests/e2e_tests/run_gpu_suite.sh", "tests/e2e_tests/common.py"):
        scope = affected.affected_scope([path], catalog, tmp_path)

        assert scope.full
        assert scope.pr_installs == ("infra-minimal", "robot-alpha")


def test_robot_change_narrows_to_its_slice_tests(
    tmp_path, make_catalog, make_component
):
    catalog = make_catalog(
        components=[
            make_component("robot-alpha"),
            make_component("robot-beta"),
        ]
    )
    slice_tests = tmp_path / "tests" / "unit_tests" / "robots" / "alpha"
    slice_tests.mkdir(parents=True)
    for shared in affected.ROBOT_SHARED_TESTS:
        (tmp_path / shared).write_text("", encoding="utf-8")

    scope = affected.affected_scope(["robots/alpha/spec.py"], catalog, tmp_path)

    assert scope.components == ("robot-alpha",)
    assert "tests/unit_tests/robots/alpha" in scope.tests
    assert "tests/unit_tests/robots/test_toolkit_contracts.py" in scope.tests
    assert "tests/unit_tests/robots/beta" not in scope.tests
    assert not scope.full


def test_model_change_installs_every_environment_that_requires_it(
    tmp_path, make_catalog, make_component
):
    catalog = make_catalog(
        components=[
            make_component("model-alpha"),
            make_component("robot-beta", requires=("model-alpha",)),
            make_component("robot-gamma"),
        ]
    )
    (tmp_path / "tests/unit_tests/components/models").mkdir(parents=True)

    scope = affected.affected_scope(
        ["components/models/alpha/server.py"], catalog, tmp_path
    )

    assert scope.components == ("model-alpha",)
    assert scope.installs == ("model-alpha", "robot-beta")
    # A component change already installs exactly the environments a full run
    # would install for it.
    assert scope.pr_installs == scope.installs
    assert not scope.full


def test_component_change_runs_the_robot_suites_that_depend_on_it(
    tmp_path, make_catalog, make_component
):
    # A robot's suite exercises the components its closure installs, so a model
    # change has to run the suites of the robots that depend on the model.
    catalog = make_catalog(
        components=[
            make_component("model-alpha"),
            make_component("robot-beta", requires=("model-alpha",)),
        ]
    )
    (tmp_path / "tests/unit_tests/components/models").mkdir(parents=True)
    (tmp_path / "tests/unit_tests/robots/beta").mkdir(parents=True)

    scope = affected.affected_scope(
        ["components/models/alpha/server.py"], catalog, tmp_path
    )

    assert "tests/unit_tests/robots/beta" in scope.tests


def test_model_change_runs_the_loopback_suites(tmp_path, make_catalog, make_component):
    # Some model services are tested only through their RPC wire ends, so a
    # model change selects those suites next to the model's own directory.
    catalog = make_catalog(components=[make_component("model-alpha")])
    (tmp_path / "tests/unit_tests/components/models").mkdir(parents=True)
    (tmp_path / "tests/unit_tests/rpent/utils/rpc").mkdir(parents=True)

    scope = affected.affected_scope(
        ["components/models/alpha/adapter/__init__.py"], catalog, tmp_path
    )

    assert scope.tests == (
        "tests/unit_tests/components/models",
        "tests/unit_tests/rpent/utils/rpc",
    )


def test_component_without_tests_runs_the_whole_suite(
    tmp_path, make_catalog, make_component
):
    catalog = make_catalog(components=[make_component("sim-alpha")])

    scope = affected.affected_scope(
        ["components/sim/alpha/env_server.py"], catalog, tmp_path
    )

    assert scope.tests == ("tests/unit_tests",)


def test_docs_only_change_needs_nothing(tmp_path, make_catalog, make_component):
    catalog = make_catalog(components=[make_component("robot-alpha")])

    scope = affected.affected_scope(
        ["README.md", "docs/source-en/index.rst"], catalog, tmp_path
    )

    assert scope.docs
    assert scope.tests == ()
    assert scope.installs == ()
    assert scope.pr_installs == ()
    assert not scope.full


def test_unmapped_path_selects_every_component_and_environment(
    tmp_path, make_catalog, make_component
):
    catalog = make_catalog(
        components=[
            make_component("infra-minimal"),
            make_component("robot-alpha"),
        ]
    )

    for path in ("setup.py", "MANIFEST.in"):
        scope = affected.affected_scope([path], catalog, tmp_path)

        assert scope.full
        assert scope.components == ("infra-minimal", "robot-alpha")
        assert scope.tests == ("tests/unit_tests", "tests/unit_tests/tooling")
        assert scope.installs == ("infra-minimal", "robot-alpha")
        # Packaging decides what every environment installs, so a pull request
        # keeps the full environment set rather than the pilot one.
        assert scope.pr_installs == ("infra-minimal", "robot-alpha")


def test_core_change_installs_the_pilot_environment_in_a_pull_request(
    tmp_path, make_catalog, make_component
):
    catalog = make_catalog(
        components=[
            make_component("infra-minimal"),
            make_component("robot-alpha"),
            make_component("sim-alpha"),
        ]
    )

    for path in ("rpent/session/session.py", "tests/unit_tests/rpent/test_session.py"):
        scope = affected.affected_scope([path], catalog, tmp_path)

        assert scope.full
        assert scope.installs == ("infra-minimal", "robot-alpha", "sim-alpha")
        assert scope.pr_installs == ("infra-minimal",)


def test_core_change_without_the_pilot_installs_every_environment(
    tmp_path, make_catalog, make_component
):
    catalog = make_catalog(components=[make_component("robot-alpha")])

    scope = affected.affected_scope(["rpent/session/session.py"], catalog, tmp_path)

    assert scope.pr_installs == ("robot-alpha",)


def test_installer_change_installs_every_environment_in_a_pull_request(
    tmp_path, make_catalog, make_component
):
    catalog = make_catalog(
        components=[
            make_component("infra-minimal"),
            make_component("robot-alpha"),
        ]
    )

    scope = affected.affected_scope(["rpent/install/installer.py"], catalog, tmp_path)

    assert scope.full
    assert scope.tests == ("tests/unit_tests", "tests/unit_tests/tooling")
    assert scope.pr_installs == ("infra-minimal", "robot-alpha")


def test_tooling_change_installs_every_environment_and_runs_itself(
    tmp_path, make_catalog, make_component
):
    catalog = make_catalog(
        components=[
            make_component("infra-minimal"),
            make_component("robot-alpha"),
        ]
    )

    for path in ("tests/tooling/layout.py", "tests/tooling/run.sh"):
        scope = affected.affected_scope([path], catalog, tmp_path)

        assert scope.full
        assert scope.tests == ("tests/unit_tests", "tests/unit_tests/tooling")
        assert scope.pr_installs == ("infra-minimal", "robot-alpha")


def _install_data_catalog(make_catalog, make_component):
    """Build a catalog where one robot installs the infrastructure component."""
    return make_catalog(
        components=[
            make_component("infra-minimal"),
            make_component("infra-rlinf"),
            make_component("robot-alpha", requires=("infra-rlinf",)),
        ]
    )


def test_install_data_change_runs_the_repository_contract_tests(
    tmp_path, make_catalog, make_component
):
    catalog = _install_data_catalog(make_catalog, make_component)

    scope = affected.affected_scope(
        ["components/infra/rlinf/component.toml"], catalog, tmp_path
    )

    assert not scope.full
    assert scope.components == ()
    assert scope.tests == ("tests/unit_tests/installs", "tests/unit_tests/tooling")
    assert scope.installs == ("infra-rlinf", "robot-alpha")
    assert scope.pr_installs == scope.installs


def test_install_data_change_reaches_the_environments_that_require_it(
    tmp_path, make_catalog, make_component
):
    catalog = _install_data_catalog(make_catalog, make_component)

    scope = affected.affected_scope(
        ["components/infra/minimal/component.toml"], catalog, tmp_path
    )

    assert scope.installs == ("infra-minimal",)


def test_packaging_settings_keep_every_environment(
    tmp_path, make_catalog, make_component
):
    catalog = _install_data_catalog(make_catalog, make_component)

    scope = affected.affected_scope(["pyproject.toml"], catalog, tmp_path)

    assert not scope.full
    assert scope.components == ()
    assert scope.tests == ("tests/unit_tests/installs", "tests/unit_tests/tooling")
    assert scope.installs == ("infra-minimal", "infra-rlinf", "robot-alpha")
    assert scope.pr_installs == scope.installs


def test_repository_contract_test_change_installs_nothing(
    tmp_path, make_catalog, make_component
):
    catalog = _install_data_catalog(make_catalog, make_component)

    scope = affected.affected_scope(
        ["tests/unit_tests/installs/test_catalog.py"], catalog, tmp_path
    )

    assert scope.tests == ("tests/unit_tests/installs", "tests/unit_tests/tooling")
    assert scope.installs == ()
    assert scope.pr_installs == ()
    assert not scope.full


def test_unrecognised_install_data_widens_to_every_environment(
    tmp_path, make_catalog, make_component
):
    catalog = _install_data_catalog(make_catalog, make_component)

    scope = affected.affected_scope(
        ["components/infra/matrix/component.toml"], catalog, tmp_path
    )

    assert scope.tests == ("tests/unit_tests/installs", "tests/unit_tests/tooling")
    assert scope.installs == ("infra-minimal", "infra-rlinf", "robot-alpha")


def test_mixed_install_and_core_change_widens_to_everything(
    tmp_path, make_catalog, make_component
):
    catalog = _install_data_catalog(make_catalog, make_component)

    scope = affected.affected_scope(
        ["components/infra/minimal/component.toml", "rpent/session/session.py"],
        catalog,
        tmp_path,
    )

    assert scope.full
    assert scope.tests == ("tests/unit_tests", "tests/unit_tests/tooling")
    assert scope.installs == ("infra-minimal", "infra-rlinf", "robot-alpha")
    assert scope.pr_installs == ("infra-minimal", "infra-rlinf", "robot-alpha")


def test_main_prints_scope_as_json(tmp_path, write_toml, capsys):
    write_toml(
        "pyproject.toml",
        """
        [project]
        name = "rpent"
        dependencies = []
        """,
    )
    write_toml(
        "robots/alpha/component.toml",
        """
        schema = 2
        name = "robot-alpha"
        probe = "json"
        """,
    )
    (tmp_path / "tests/unit_tests/robots/alpha").mkdir(parents=True)

    assert affected.main(["robots/alpha/spec.py"], root=tmp_path) == 0

    scope = json.loads(capsys.readouterr().out)
    assert scope["components"] == ["robot-alpha"]
    assert scope["installs"] == ["robot-alpha"]
    assert scope["tests"] == ["tests/unit_tests/robots/alpha"]
    assert not scope["full"]


def test_runner_filter_narrows_every_environment_set(
    tmp_path, write_toml, capsys, monkeypatch
):
    write_toml(
        "pyproject.toml",
        """
        [project]
        name = "rpent"
        dependencies = []
        """,
    )
    write_toml(
        "robots/alpha/component.toml",
        """
        schema = 2
        name = "robot-alpha"
        probe = "json"
        """,
    )
    write_toml(
        "components/infra/minimal/component.toml",
        """
        schema = 2
        name = "infra-minimal"
        probe = "json"
        """,
    )
    monkeypatch.setattr(affected, "MATRICES", {"gpu": (("robot-alpha",),)})

    assert (
        affected.main(["rpent/session/session.py", "--runner", "gpu"], root=tmp_path)
        == 0
    )

    scope = json.loads(capsys.readouterr().out)
    assert scope["components"] == ["robot-alpha"]
    assert scope["installs"] == ["robot-alpha"]
    assert scope["pr_installs"] == []


def test_changed_paths_report_deleted_files(tmp_path):
    def git(*args):
        subprocess.run(
            ["git", "-c", "user.name=t", "-c", "user.email=t@example.com", *args],
            cwd=tmp_path,
            check=True,
            capture_output=True,
        )

    git("init", "-q")
    (tmp_path / "gone.py").write_text("x = 1\n")
    git("add", ".")
    git("commit", "-qm", "base")
    (tmp_path / "gone.py").unlink()
    (tmp_path / "page.md").write_text("# page\n")
    git("add", "-A")
    git("commit", "-qm", "change")

    # A deleted file is a change: filtering it out would let a pull request
    # that removes a component or packaging file look documentation-only.
    assert sorted(affected.changed_paths(tmp_path, "HEAD~1", "HEAD")) == [
        "gone.py",
        "page.md",
    ]
