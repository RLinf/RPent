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

"""``check --resolve`` compiles a component environment with uv, temporarily."""

import os
import stat
from pathlib import Path

import pytest

from rpent.cli.install import main as env_main
from rpent.install.compile import DEFAULT_MANYLINUX, resolve_resolution
from rpent.install.discovery import discover
from rpent.install.errors import InstallError
from rpent.install.installer import resolve_targets
from rpent.install.manifest import SCHEMA
from rpent.install.requirements import read_core_dependencies
from rpent.install.resolve import choose_python


def test_the_default_root_must_be_a_checkout(tmp_path, monkeypatch, capsys):
    # A wheel install resolves the package root to site-packages, which holds
    # no manifests; the CLI has to say that instead of failing on a path the
    # user never named. The working directory is not a checkout here, so the
    # package root is the only candidate.
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("rpent.cli.install.get_repo_root", lambda: tmp_path)

    with pytest.raises(SystemExit) as excinfo:
        env_main(["list"])

    assert excinfo.value.code == 2
    assert "is not an RPent checkout" in capsys.readouterr().err


def test_the_default_root_is_the_checkout_of_the_working_directory(
    tmp_path, monkeypatch, write_toml, capsys
):
    # ``pip install rpent`` leaves the package in site-packages, so the
    # documented flow runs the CLI from a checkout; the working directory has
    # to name it.
    write_toml("pyproject.toml", '[project]\nname = "rpent"\ndependencies = []\n')
    (tmp_path / "components").mkdir()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        "rpent.cli.install.get_repo_root", lambda: tmp_path / "site-packages"
    )

    assert env_main(["list"]) == 0
    assert "is not an RPent checkout" not in capsys.readouterr().err


@pytest.fixture
def repo(tmp_path, write_toml):
    write_toml(
        "pyproject.toml",
        """
        [project]
        name = "rpent"
        dependencies = ["packaging"]
        """,
    )
    write_toml(
        "robots/alpha/component.toml",
        f"""
        schema = {SCHEMA}
        name = "robot-alpha"
        probe = "json"
        dependencies = ["numpy>=2"]
        """,
    )
    write_toml(
        "robots/beta/component.toml",
        f"""
        schema = {SCHEMA}
        name = "robot-beta"
        probe = "json"
        """,
    )
    return tmp_path


@pytest.fixture
def fake_uv(tmp_path, monkeypatch):
    """A uv stand-in that copies the compiled source to ``--output-file``."""
    uv = tmp_path / "uv"
    uv.write_text(
        """#!/bin/sh
set -eu
pwd -P >> "$(dirname "$0")/uv-pwd"
printf '%s\\n' "$@" > "$(dirname "$0")/uv-args"
[ "$1 $2" = "pip compile" ]
source=$3
output=
previous=
for argument in "$@"; do
  if [ "$previous" = "--output-file" ]; then
    output=$argument
  fi
  previous=$argument
done
[ -n "$output" ]
if grep -q 'numpy>=9' "$source"; then
  echo "no solution for numpy>=9" >&2
  exit 1
fi
cp "$source" "$output"
""",
        encoding="utf-8",
    )
    uv.chmod(uv.stat().st_mode | stat.S_IXUSR)
    monkeypatch.setenv("RPENT_UV", str(uv))
    return uv


def _arguments(fake_uv):
    return (fake_uv.parent / "uv-args").read_text(encoding="utf-8").splitlines()


def _resolution(repo, *targets):
    catalog = discover(repo)
    resolution = resolve_targets(targets, catalog, root=repo)
    assert resolution.ok, resolution.errors
    return resolution


def test_check_resolves_from_the_checkout(repo, fake_uv, monkeypatch):
    # uv discovers ``[tool.uv]`` from its working directory, so dependencies
    # resolved from a subdirectory would record versions the checkout does not
    # select.
    monkeypatch.chdir(repo / "robots")

    assert env_main(["--root", str(repo), "check", "robot-alpha", "--resolve"]) == 0

    assert (repo / "uv-pwd").read_text(encoding="utf-8").splitlines() == [
        str(repo.resolve())
    ]


def test_check_resolves_override_files_against_the_checkout(repo, fake_uv, write_toml):
    # uv resolves relative paths against its working directory, so dependencies
    # resolved from a subdirectory would otherwise miss the override.
    write_toml(
        "robots/alpha/component.toml",
        f"""
        schema = {SCHEMA}
        name = "robot-alpha"
        probe = "json"
        dependencies = ["numpy>=2"]
        overrides = ["requirements/override.txt"]
        """,
    )
    write_toml("robots/alpha/requirements/override.txt", "numpy<3\n")

    assert env_main(["--root", str(repo), "check", "robot-alpha", "--resolve"]) == 0

    arguments = _arguments(fake_uv)
    value = arguments[arguments.index("--overrides") + 1]
    assert value == str(repo / "robots/alpha/requirements/override.txt")
    assert os.path.isabs(value)


def test_check_uses_the_default_manylinux_baseline(repo, fake_uv):
    assert env_main(["--root", str(repo), "check", "robot-alpha", "--resolve"]) == 0

    arguments = _arguments(fake_uv)
    assert arguments[arguments.index("--python-platform") + 1] == (
        f"x86_64-manylinux_{DEFAULT_MANYLINUX}"
    )


def test_check_targets_the_newest_baseline_in_the_closure(repo, fake_uv, write_toml):
    # A GPU component may publish wheels for a newer glibc than the repository
    # default; resolving the closure against the default would report packages
    # uv cannot install on the target.
    write_toml(
        "components/sim/alpha/component.toml",
        f"""
        schema = {SCHEMA}
        name = "sim-alpha"
        probe = "json"
        manylinux = "2_35"
        """,
    )
    write_toml(
        "robots/alpha/component.toml",
        f"""
        schema = {SCHEMA}
        name = "robot-alpha"
        probe = "json"
        requires = ["sim-alpha"]
        dependencies = ["numpy>=2"]
        """,
    )

    assert env_main(["--root", str(repo), "check", "robot-alpha", "--resolve"]) == 0

    arguments = _arguments(fake_uv)
    assert arguments[arguments.index("--python-platform") + 1] == (
        "x86_64-manylinux_2_35"
    )


def test_check_passes_the_components_index_settings(repo, fake_uv, write_toml):
    write_toml(
        "components/infra/unit_cpu/component.toml",
        f"""
        schema = {SCHEMA}
        name = "infra-unit_cpu"
        probe = "json"
        index-url = "https://download.pytorch.org/whl/cpu"
        extra-index-urls = ["https://pypi.org/simple"]
        index-strategy = "unsafe-first-match"
        """,
    )

    assert env_main(["--root", str(repo), "check", "infra-unit_cpu", "--resolve"]) == 0

    arguments = _arguments(fake_uv)
    assert arguments[arguments.index("--index-url") + 1] == (
        "https://download.pytorch.org/whl/cpu"
    )
    assert arguments[arguments.index("--extra-index-url") + 1] == (
        "https://pypi.org/simple"
    )
    assert arguments[arguments.index("--index-strategy") + 1] == "unsafe-first-match"


def test_resolve_resolution_cleans_up_compiled_requirements(repo, fake_uv):
    resolution = _resolution(repo, "robot-alpha")

    text = resolve_resolution(
        resolution,
        root=repo,
        uv=str(fake_uv),
        python=choose_python(resolution.components.values()),
        core_dependencies=read_core_dependencies(repo),
    )

    assert "numpy>=2" in text
    arguments = _arguments(fake_uv)
    assert not Path(arguments[arguments.index("--output-file") + 1]).exists()
    assert not Path(arguments[2]).exists()


def test_check_resolve_reports_the_uv_solution(repo, fake_uv, capsys):
    assert env_main(["--root", str(repo), "check", "robot-alpha", "--resolve"]) == 0

    # `packaging` from RPent core plus `numpy>=2` from the component.
    assert "robot-alpha: ok, 1 component, uv resolved 2 packages" in (
        capsys.readouterr().out
    )


def test_check_resolve_reports_a_uv_failure(repo, fake_uv, monkeypatch, capsys):
    def fail(*args, **kwargs):
        raise InstallError("no solution for numpy>=9")

    monkeypatch.setattr("rpent.cli.install.resolve_resolution", fail)

    assert env_main(["--root", str(repo), "check", "robot-alpha", "--resolve"]) == 1

    output = capsys.readouterr().out
    assert "robot-alpha: FAIL" in output
    assert "no solution for numpy>=9" in output


def test_check_without_resolve_reports_counts_without_uv(repo, monkeypatch, capsys):
    monkeypatch.setenv("RPENT_UV", "/nonexistent/uv")

    assert env_main(["--root", str(repo), "check", "robot-alpha"]) == 0

    assert "robot-alpha: ok, 1 component" in capsys.readouterr().out


def test_check_all_reports_failures_and_continues(repo, fake_uv, write_toml, capsys):
    write_toml(
        "robots/gamma/component.toml",
        f"""
        schema = {SCHEMA}
        name = "robot-gamma"
        probe = "json"
        dependencies = ["numpy>=9"]
        """,
    )

    assert env_main(["--root", str(repo), "check", "--all", "--resolve"]) == 1

    output = capsys.readouterr().out
    assert "robot-gamma: FAIL" in output
    assert "no solution for numpy>=9" in output
    assert "robot-alpha: ok" in output
    assert "robot-beta: ok" in output


def test_check_reports_unknown_components(repo, capsys):
    assert env_main(["--root", str(repo), "check", "robot-missing"]) == 1

    assert "unknown component 'robot-missing'" in capsys.readouterr().err


def _conflicting_components(write_toml):
    """Declare two components that each resolve alone but not together."""
    for name, other in (("alpha", "robot-beta"), ("beta", "robot-alpha")):
        write_toml(
            f"robots/{name}/component.toml",
            f"""
            schema = {SCHEMA}
            name = "robot-{name}"
            probe = "json"
            env_conflicts = ["{other}"]
            """,
        )


def test_check_resolves_named_components_as_one_environment(repo, write_toml, capsys):
    # ``install a b`` creates one environment, so ``check a b`` has to judge
    # that environment: two components that each resolve alone can still refuse
    # to share one.
    _conflicting_components(write_toml)

    assert env_main(["--root", str(repo), "check", "robot-alpha", "robot-beta"]) == 1

    output = capsys.readouterr().out
    assert "robot-alpha robot-beta: FAIL" in output
    assert "cannot share an environment" in output
    # The components are still reported one by one, so the failure points at
    # the manifests that need changing.
    assert "robot-alpha: ok, 1 component" in output
    assert "robot-beta: ok, 1 component" in output


def test_check_all_keeps_one_environment_per_component(repo, write_toml, capsys):
    _conflicting_components(write_toml)

    assert env_main(["--root", str(repo), "check", "--all"]) == 0

    output = capsys.readouterr().out
    assert "robot-alpha: ok, 1 component" in output
    assert "robot-beta: ok, 1 component" in output


def test_check_all_refuses_named_components(repo, capsys):
    # ``--all`` checks every component on its own, so a named component would
    # be dropped without a word, including a name that is a typo.
    assert env_main(["--root", str(repo), "check", "--all", "robot-missing"]) == 2

    assert "do not name components with it" in capsys.readouterr().err
