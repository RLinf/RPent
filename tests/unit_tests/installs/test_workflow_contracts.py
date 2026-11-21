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

"""CI workflows keep the checks the repository rules and releases rely on.

The scopes these workflows pass around are covered by ``test_affected``; these
tests read the workflow files themselves, because a dropped trigger, a missing
``needs``, or a widened wheel scope is invisible to a Python import. They sit
with the install-data contracts because the workflows and that data are read
together.
"""

import os
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
WORKFLOWS = REPO_ROOT / ".github" / "workflows"


def _workflow(name):
    return yaml.safe_load((WORKFLOWS / name).read_text(encoding="utf-8"))


def _triggers(workflow):
    # YAML 1.1 reads the top-level ``on:`` key as the boolean True.
    for key in ("on", True):
        if key in workflow:
            return workflow[key]
    raise AssertionError("workflow has no trigger section")


def _steps(workflow, job):
    return workflow["jobs"][job]["steps"]


def _step(workflow, job, name):
    for step in _steps(workflow, job):
        if step.get("name") == name:
            return step
    raise AssertionError(f"job {job!r} has no step {name!r}")


def test_fast_ci_runs_on_every_pull_request():
    triggers = _triggers(_workflow("fast-ci.yml"))

    # A ``paths:`` filter would leave the required check pending instead of
    # skipping, so the workflow must always start and narrow inside the scope
    # job.
    assert "paths" not in triggers["pull_request"]
    assert "paths" not in triggers["push"]
    assert triggers["push"]["branches"] == ["main"]


def test_fast_ci_has_one_aggregate_required_check():
    workflow = _workflow("fast-ci.yml")
    aggregate = workflow["jobs"]["unittest"]

    assert aggregate["if"] == "always()"
    assert set(aggregate["needs"]) == {"changes", "unit", "package", "full"}
    results = _steps(workflow, "unittest")[0]["run"]
    for job in ("changes", "unit", "package", "full"):
        assert f"{job}:${{{{ needs.{job}.result }}}}" in results


def test_fast_ci_scopes_full_ci_with_the_pull_request_environments():
    script = _step(_workflow("fast-ci.yml"), "changes", "Map changed paths to a scope")[
        "run"
    ]

    # A core change keeps the full environment set for nightly and release
    # runs, while a pull request installs the pilot set; a component change
    # already narrows ``installs`` to its closure.
    assert 'scope.get("pr_installs", scope["installs"])' in script
    assert (
        'e2e = sorted(suites) if set(selected) == set(scope["installs"]) else []'
        in (script)
    )
    # Install data changes narrow the environment set but still call full CI for
    # those environments.
    assert 'bool(scope["components"] or scope["installs"])' in script


def test_fast_ci_runs_installer_and_install_data_tests_in_the_dev_environment():
    workflow = _workflow("fast-ci.yml")

    # Both groups need only the core and `infra-dev` dependencies; every other
    # test path may import a model server, which needs the CPU Torch
    # environment.
    script = _step(workflow, "changes", "Map changed paths to a scope")["run"]
    dev_tests = (
        '{"tests/unit_tests/rpent/install", "tests/unit_tests/installs", '
        '"tests/unit_tests/tooling"}'
    )
    assert dev_tests in script
    assert (
        'components = "infra-dev" if tests and tests <= dev_tests '
        'else "infra-unit_cpu"' in script
    )
    assert workflow["jobs"]["unit"]["with"]["components"] == (
        "${{ needs.changes.outputs.components }}"
    )


def test_fast_ci_checks_the_package_for_packaging_changes():
    script = _step(_workflow("fast-ci.yml"), "changes", "Map changed paths to a scope")[
        "run"
    ]

    # ``setup.py`` chooses the packages in the wheel and sdist, so it changes
    # what the package job has to verify.
    assert '{"pyproject.toml", "setup.py", "MANIFEST.in"}' in script


def test_full_ci_does_not_run_on_tags():
    # A tag push used to start full CI and publish in parallel. The publish
    # workflow now calls full CI itself, so that it verifies the exact commit
    # it releases before building.
    triggers = _triggers(_workflow("full-ci.yml"))

    assert "push" not in triggers
    assert set(triggers) == {"schedule", "workflow_dispatch", "workflow_call"}


def test_full_ci_covers_all_environments_for_a_release():
    workflow = _workflow("full-ci.yml")

    assert workflow["jobs"]["unit"]["with"]["components"] == "infra-unit_cpu"
    called = _step(workflow, "scope", "Select the environments and suites")["run"]
    assert '"$EVENT" == "workflow_call" && "$CALLED_FULL" != "true"' in called


def test_publish_verifies_before_building_the_core_wheel():
    workflow = _workflow("publish.yml")

    assert _triggers(workflow)["push"]["tags"] == ["v*"]
    assert workflow["jobs"]["verify"]["uses"] == "./.github/workflows/full-ci.yml"
    assert workflow["jobs"]["verify"]["with"] == {"full": True}
    assert workflow["jobs"]["build"]["needs"] == "verify"
    assert workflow["jobs"]["publish"]["needs"] == "build"
    assert _step(workflow, "build", "Build sdist and wheel")["env"][
        "RPENT_WHEEL_SCOPE"
    ] == ("core")


def test_package_checks_the_core_wheel_and_the_complete_sdist():
    workflow = _workflow("package.yml")

    assert _step(workflow, "build", "Build sdist and wheel")["env"][
        "RPENT_WHEEL_SCOPE"
    ] == ("core")
    wheel_check = _step(workflow, "build", "Check the wheel install")["run"]
    assert "robots" in wheel_check and "components" in wheel_check
    # The sdist is the supported source install, so it must carry the component
    # packages the wheel deliberately omits.
    sdist = _step(workflow, "build", "Install the sdist and upgrade it to the wheel")[
        "run"
    ]
    assert "import components" in sdist
    assert "import robots" in sdist
    source = _step(workflow, "build", "Check source installs outside the checkout")[
        "run"
    ]
    # A source-scope wheel is what `pip install .` produces; it must carry the
    # files the installed package opens, and an editable install must expose
    # the checkout's packages and components from another directory.
    assert "--no-deps -e ." in source
    for relative in ("target50.json", "demo_randomized.json"):
        assert relative in source
    assert "rpent-install" in source


def test_unit_workflow_defaults_to_the_cpu_environment():
    workflow = _workflow("unit.yml")
    inputs = _triggers(workflow)["workflow_call"]["inputs"]

    # ``infra-dev`` deliberately has no Torch; the CPU environment supplies it
    # so the full unit suite can import model servers.
    assert inputs["components"]["default"] == "infra-unit_cpu"
    install = _step(workflow, "unit", "Install the test environment from its component")
    assert "if" not in install
    assert 'install "${{ inputs.components }}" --venv .venv' in install["run"]
    minimal = _step(workflow, "minimal-imports", "Install the minimal environment")
    assert "if" not in minimal
    assert "install infra-minimal --venv .venv-minimal" in minimal["run"]
    assert _step(workflow, "unit", "Report skipped tests")["if"] == "always()"


def test_env_workflow_installs_every_component_of_an_environment():
    workflow = _workflow("env.yml")
    matrix = workflow["jobs"]["gpu"]["strategy"]["matrix"]

    # Each matrix entry is the component list one install names, so a
    # multi-target environment is installed and doctored as one venv.
    assert matrix["targets"] == "${{ fromJSON(inputs.gpu) }}"
    install = _step(workflow, "gpu", "Fresh install from the components")["run"]
    assert "install ${{ join(matrix.targets, ' ') }}" in install
    assert "--venv \".venv-${{ join(matrix.targets, '+') }}\"" in install
    assert workflow["jobs"]["gpu"]["name"] == (
        "Fresh install ${{ join(matrix.targets, '+') }}"
    )


def test_workflows_bootstrap_the_tools_through_launchers():
    # The bootstrap interpreter and its extra dependency live in one place, so a
    # change reaches the workflows, the pre-commit hook, and the GPU suite
    # together instead of drifting between copies.
    launchers = (
        REPO_ROOT / "rpent-install.sh",
        REPO_ROOT / "tests" / "tooling" / "run.sh",
    )
    for launcher in launchers:
        assert launcher.is_file()
        assert os.access(launcher, os.X_OK)
    for path in [*WORKFLOWS.glob("*.yml"), REPO_ROOT / ".pre-commit-config.yaml"]:
        assert "--no-project" not in path.read_text(encoding="utf-8"), path.name
