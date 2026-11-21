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

"""The installer resolves targets, renders requirements, and records fingerprints."""

import os
import stat
import subprocess
import sys
from pathlib import Path

import pytest

from rpent.install.discovery import discover
from rpent.install.errors import InstallError
from rpent.install.fingerprint import FINGERPRINT_SCHEMA_FIELD, read_fingerprint
from rpent.install.installer import (
    default_venv,
    environment_id,
    find_uv,
    index_options,
    install,
    probe_command,
    probe_source,
    requirements_file,
    resolve_targets,
)
from rpent.install.manifest import SCHEMA
from rpent.install.requirements import (
    FINGERPRINT_SCHEMA,
    read_core_dependencies,
    requirements_sha256,
)
from rpent.install.resolve import choose_python, resolve


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
    write_toml(
        "components/infra/dev/component.toml",
        f"""
        schema = {SCHEMA}
        name = "infra-dev"
        probe = "json"
        dependencies = ["pytest"]
        """,
    )
    return tmp_path


def test_dry_run_renders_requirements_without_uv(repo, monkeypatch, capsys):
    monkeypatch.delenv("RPENT_UV", raising=False)
    monkeypatch.setattr("shutil.which", lambda name: None)

    result = install(["robot-alpha"], root=repo, dry_run=True, verbose=False)

    assert result["targets"] == ["robot-alpha"]
    assert "packaging" in result["requirements"]
    assert "numpy>=2" in result["requirements"]
    assert "packaging" in capsys.readouterr().out


def test_environment_id_is_order_independent():
    assert environment_id(["robot-beta", "robot-alpha", "robot-alpha"]) == (
        "robot-alpha+robot-beta"
    )


def test_default_paths_are_named_after_the_target_set(repo):
    targets = ["robot-beta", "robot-alpha"]
    venv = default_venv(repo, targets)

    assert venv == repo / ".rpent" / "venvs" / "robot-alpha+robot-beta"
    # The rendered requirements belong to the environment, not to the checkout.
    assert requirements_file(venv) == venv / "rpent-install.requirements.txt"


def test_a_target_set_resolves_its_closure_together(repo):
    catalog = discover(repo)

    resolution = resolve_targets(["robot-alpha", "infra-dev"], catalog, root=repo)

    assert resolution.ok, resolution.errors
    assert resolution.targets == ("infra-dev", "robot-alpha")
    assert sorted(resolution.components) == ["infra-dev", "robot-alpha"]


def test_unknown_target_lists_known_components(repo):
    with pytest.raises(InstallError, match="known components: infra-dev"):
        install(["robot-missing"], root=repo, dry_run=True, verbose=False)


def test_choose_python_uses_the_newest_supported_minor(make_component):
    # The interpreter comes from the manifests, not from the shell that runs
    # the installer, so the same checkout builds the same environment anywhere.
    component = make_component("robot-alpha", python=">=3.10,<3.15")

    assert choose_python([component]) == "3.12"


def test_choose_python_picks_the_highest_supported_minor(make_component):
    component = make_component("robot-alpha", python=">=3.9,<3.12")

    assert choose_python([component]) == "3.11"


def test_index_options_follow_the_merged_resolution(make_component, make_catalog):
    component = make_component(
        "infra-unit_cpu",
        index_url="https://index.test/simple",
        extra_index_urls=("https://extra.test/simple",),
        index_strategy="unsafe-first-match",
    )
    resolution = resolve(["infra-unit_cpu"], make_catalog(components=[component]))

    assert index_options(resolution) == [
        "--index-url",
        "https://index.test/simple",
        "--extra-index-url",
        "https://extra.test/simple",
        "--index-strategy",
        "unsafe-first-match",
    ]


def test_find_uv_prefers_override(monkeypatch):
    monkeypatch.setenv("RPENT_UV", "/custom/uv")

    assert find_uv() == "/custom/uv"


def _fake_uv(tmp_path, monkeypatch):
    """Create a uv stand-in and return it with a reader for install arguments."""
    uv = tmp_path / "uv"
    uv.write_text(
        """#!/bin/sh
set -eu
pwd -P >> "$(dirname "$0")/uv-pwd"
if [ "$1" = "venv" ]; then
  for last; do :; done
  mkdir -p "$last/bin"
  : > "$last/pyvenv.cfg"
  cat > "$last/bin/python" <<'STUB'
#!/bin/sh
case "$*" in
  *importlib.metadata*) ;;
  *platform*) printf '3.11.9\n' ;;
esac
exit 0
STUB
  chmod +x "$last/bin/python"
elif [ "$1 $2" = "pip install" ]; then
  { printf '%s\n' "$@"; echo --; } >> "$(dirname "$0")/uv-install-args"
elif [ "$1 $2" = "pip freeze" ]; then
  echo "alpha==1.0.0"
fi
exit 0
""",
        encoding="utf-8",
    )
    uv.chmod(uv.stat().st_mode | stat.S_IXUSR)
    monkeypatch.setenv("RPENT_UV", str(uv))

    def install_arguments():
        """Return each ``uv pip install`` invocation's arguments."""
        text = (tmp_path / "uv-install-args").read_text(encoding="utf-8")
        return [block.splitlines() for block in text.split("--\n") if block.strip()]

    return uv, install_arguments


def _uv_working_directories(tmp_path):
    """Return the directory each ``uv`` invocation of the fake ran in."""
    return (tmp_path / "uv-pwd").read_text(encoding="utf-8").splitlines()


def test_install_records_a_fingerprint(repo, tmp_path, monkeypatch, capsys):
    _fake_uv(tmp_path, monkeypatch)
    venv = tmp_path / "venv"

    fingerprint = install(["robot-alpha"], root=repo, venv=venv, verbose=True)

    assert read_fingerprint(venv) == fingerprint
    assert fingerprint[FINGERPRINT_SCHEMA_FIELD] == FINGERPRINT_SCHEMA
    assert fingerprint["targets"] == ["robot-alpha"]
    assert sorted(fingerprint["components"]) == ["robot-alpha"]
    assert fingerprint["platform"] == "linux-x86_64"
    assert fingerprint["source_commit"] is None
    assert os.path.isfile(venv / "bin" / "python")
    # The rendered requirements and the resolved versions stay with the
    # environment, so what it was installed from remains readable.
    assert requirements_file(venv).is_file()
    assert (venv / "rpent-install.freeze.txt").read_text(encoding="utf-8") == (
        "alpha==1.0.0\n"
    )
    assert "probe robot-alpha: import json" in capsys.readouterr().out


def test_install_installs_the_requirements_then_the_checkout_as_editable(
    repo, tmp_path, monkeypatch
):
    uv, install_arguments = _fake_uv(tmp_path, monkeypatch)

    venv = tmp_path / "venv"
    install(["robot-alpha"], root=repo, venv=venv)

    requirements_arguments, editable_arguments = install_arguments()
    assert requirements_arguments[requirements_arguments.index("-r") + 1] == str(
        requirements_file(venv)
    )
    requirements = requirements_file(venv).read_text(encoding="utf-8")
    assert "numpy>=2" in requirements
    assert "packaging" in requirements
    assert editable_arguments[-3:] == ["--no-deps", "-e", str(repo.resolve())]
    assert requirements_arguments[:2] == ["pip", "install"]
    assert editable_arguments[:2] == ["pip", "install"]
    assert requirements_arguments[2:4] == [
        "--python",
        str(tmp_path / "venv" / "bin" / "python"),
    ]
    # The checkout's own package is installed editable, without dependencies, so
    # the requirements file stays the only source of third-party packages.
    assert str(uv) not in requirements_arguments


def test_install_records_the_inputs_doctor_recomputes(repo, tmp_path, monkeypatch):
    _fake_uv(tmp_path, monkeypatch)

    fingerprint = install(["robot-alpha"], root=repo, venv=tmp_path / "venv")

    resolution = resolve_targets(["robot-alpha"], discover(repo), root=repo)
    assert fingerprint["requirements_sha256"] == requirements_sha256(
        resolution,
        root=repo,
        core_dependencies=read_core_dependencies(repo),
    )


def test_install_passes_override_files_by_absolute_path(
    repo, tmp_path, write_toml, monkeypatch
):
    # uv resolves relative paths against its working directory, and the CLI may
    # be run from a subdirectory, so manifests are resolved against the
    # checkout before being handed over.
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
    _, install_arguments = _fake_uv(tmp_path, monkeypatch)

    install(["robot-alpha"], root=repo, venv=tmp_path / "venv", verbose=False)

    arguments = install_arguments()[0]
    value = arguments[arguments.index("--override") + 1]
    assert value == str(repo / "robots/alpha/requirements/override.txt")
    assert os.path.isabs(value)


def test_install_runs_uv_inside_the_checkout(repo, tmp_path, monkeypatch):
    # uv discovers ``[tool.uv]`` by walking up from its working directory, so
    # installing from a subdirectory must not resolve with other settings.
    _fake_uv(tmp_path, monkeypatch)
    monkeypatch.chdir(repo / "robots")

    install(["robot-alpha"], root=repo, venv=tmp_path / "venv", verbose=False)

    assert set(_uv_working_directories(tmp_path)) == {str(repo.resolve())}


def test_install_reuses_a_venv_holding_the_same_closure(
    repo, tmp_path, write_toml, monkeypatch
):
    # ``robot-alpha`` requires ``infra-dev``, so naming both describes the same
    # environment as naming only the slice. The typed names identify the
    # directory and the record, not the environment.
    write_toml(
        "robots/alpha/component.toml",
        f"""
        schema = {SCHEMA}
        name = "robot-alpha"
        probe = "json"
        requires = ["infra-dev"]
        dependencies = ["numpy>=2"]
        """,
    )
    _fake_uv(tmp_path, monkeypatch)
    venv = tmp_path / "venv"
    install(["robot-alpha"], root=repo, venv=venv, verbose=False)

    fingerprint = install(
        ["robot-alpha", "infra-dev"], root=repo, venv=venv, verbose=False
    )

    assert fingerprint["targets"] == ["infra-dev", "robot-alpha"]


def test_install_refuses_a_venv_holding_another_closure(repo, tmp_path, monkeypatch):
    _fake_uv(tmp_path, monkeypatch)
    venv = tmp_path / "venv"
    install(["robot-alpha"], root=repo, venv=venv, verbose=False)

    with pytest.raises(InstallError, match="holds a different environment"):
        install(["robot-beta"], root=repo, venv=venv, verbose=False)


def test_install_repairs_a_venv_after_a_component_manifest_changed(
    repo, tmp_path, write_toml, monkeypatch
):
    # The same components with other packages are the same environment, so the
    # command that built it is the command that brings it up to date.
    _fake_uv(tmp_path, monkeypatch)
    venv = tmp_path / "venv"
    install(["robot-alpha"], root=repo, venv=venv, verbose=False)
    write_toml(
        "robots/alpha/component.toml",
        f"""
        schema = {SCHEMA}
        name = "robot-alpha"
        probe = "json"
        dependencies = ["scipy>=1"]
        """,
    )

    fingerprint = install(["robot-alpha"], root=repo, venv=venv, verbose=False)

    assert fingerprint["targets"] == ["robot-alpha"]
    assert "scipy>=1" in requirements_file(venv).read_text(encoding="utf-8")


def test_install_force_replaces_a_venv_holding_another_closure(
    repo, tmp_path, monkeypatch
):
    _fake_uv(tmp_path, monkeypatch)
    venv = tmp_path / "venv"
    install(["robot-alpha"], root=repo, venv=venv, verbose=False)

    fingerprint = install(
        ["robot-beta"], root=repo, venv=venv, force=True, verbose=False
    )

    assert fingerprint["targets"] == ["robot-beta"]
    assert sorted(fingerprint["components"]) == ["robot-beta"]


def test_install_force_refuses_to_replace_a_directory_that_is_not_a_venv(
    repo, tmp_path, monkeypatch
):
    # ``--force`` answers "rebuild the environment at this path"; it does not
    # license deleting whatever happens to be there.
    _fake_uv(tmp_path, monkeypatch)
    venv = tmp_path / "venv"
    venv.mkdir()
    (venv / "notes.txt").write_text("keep me\n", encoding="utf-8")

    with pytest.raises(InstallError, match="is not a virtualenv"):
        install(["robot-alpha"], root=repo, venv=venv, force=True, verbose=False)

    assert (venv / "notes.txt").is_file()


def test_install_force_keeps_the_venv_when_uv_is_missing(repo, tmp_path, monkeypatch):
    # The replacement is the rebuild; without uv there is nothing to rebuild
    # with, so the environment that is there stays.
    _fake_uv(tmp_path, monkeypatch)
    venv = tmp_path / "venv"
    install(["robot-alpha"], root=repo, venv=venv, verbose=False)
    monkeypatch.setattr("shutil.which", lambda name: None)
    monkeypatch.delenv("RPENT_UV", raising=False)

    with pytest.raises(InstallError, match="uv is required"):
        install(["robot-beta"], root=repo, venv=venv, force=True, verbose=False)

    assert read_fingerprint(venv)["targets"] == ["robot-alpha"]


def _plain_venv(tmp_path, *packages):
    """Create a virtualenv directory whose interpreter reports ``packages``."""
    venv = tmp_path / "venv"
    (venv / "bin").mkdir(parents=True)
    listed = "".join(f"    printf '{name}\\n'\n" for name in packages)
    python = venv / "bin" / "python"
    python.write_text(
        "#!/bin/sh\n"
        'case "$*" in\n'
        "  *importlib.metadata*)\n"
        f"{listed}"
        "    ;;\n"
        "  *platform*)\n"
        "    printf '3.11.9\\n'\n"
        "    ;;\n"
        "esac\n"
        "exit 0\n",
        encoding="utf-8",
    )
    python.chmod(0o755)
    return venv


def test_install_refuses_a_venv_with_packages_and_no_fingerprint(
    repo, tmp_path, monkeypatch
):
    # The packages were installed by someone else, so this environment would
    # mix two dependency sets in one interpreter.
    _fake_uv(tmp_path, monkeypatch)
    _plain_venv(tmp_path, "pip", "numpy")

    with pytest.raises(InstallError, match="already contains packages"):
        install(["robot-alpha"], root=repo, venv=tmp_path / "venv", verbose=False)


def test_install_reuses_a_virtualenv_with_nothing_installed(
    repo, tmp_path, monkeypatch
):
    # ``uv venv`` leaves an empty environment; only the packages an install
    # would add belong in it.
    _fake_uv(tmp_path, monkeypatch)
    _plain_venv(tmp_path, "pip")

    fingerprint = install(
        ["robot-alpha"], root=repo, venv=tmp_path / "venv", verbose=False
    )

    assert fingerprint["targets"] == ["robot-alpha"]


def test_probe_command_isolates_the_environment(tmp_path):
    # PYTHONPATH and the caller's directory would let the probe import the
    # source tree, so an editable install missing from the venv would pass.
    command = probe_command(tmp_path / "python", "robots.alpha", "robot")

    assert command[:3] == [str(tmp_path / "python"), "-I", "-c"]
    assert "robots.alpha.robot_spec" in command[3]
    assert "robots.alpha.env_client" in command[3]
    assert "adapter" not in command[3]


def test_probe_checks_the_role_modules_of_each_kind():
    # Importing the package alone passes for a component that installed only
    # its name; every kind names the modules its components must own.
    assert 'importlib.import_module("components.tools.flywheel.cli")' in probe_source(
        "components.tools.flywheel", "tool"
    )
    assert 'importlib.import_module("components.sim.libero.env_server")' in (
        probe_source("components.sim.libero", "sim")
    )
    assert 'importlib.import_module("components.models.openpi.server")' in (
        probe_source("components.models.openpi", "model")
    )
    # Infra components install a third-party package and own no roles.
    assert probe_source("rlinf", "infra") == "import rlinf\n"
    # So does any probe outside the checkout, whatever its kind: the component
    # does not own modules under the upstream distribution's name.
    assert probe_source("openpi", "model") == "import openpi\n"


def test_probe_checks_only_the_roles_the_component_ships(tmp_path):
    # A tool is either a process with a CLI or a capability library a robot
    # slice registers; probing a module the component does not ship would fail
    # an installation that is complete.
    library = tmp_path / "rgbd"
    library.mkdir()
    (library / "segment.py").touch()
    executable = tmp_path / "flywheel"
    executable.mkdir()
    (executable / "cli.py").touch()

    assert probe_source("components.tools.rgbd", "tool", source=library) == (
        "import components.tools.rgbd\n"
    )
    assert 'importlib.import_module("components.tools.flywheel.cli")' in probe_source(
        "components.tools.flywheel", "tool", source=executable
    )


def test_an_upstream_probe_imports_only_the_distribution():
    # `json` owns no role modules, so a probe that names it must not append
    # one: the upstream package the component installs is not named after the
    # component's role files.
    command = probe_command(Path(sys.executable), "json", "robot")

    result = subprocess.run(command, text=True, capture_output=True)

    assert result.returncode == 0, result.stderr


def test_install_repairs_an_environment_left_by_a_failed_install(
    repo, tmp_path, monkeypatch
):
    # A failure after the packages were written leaves the rendered
    # requirements but no fingerprint. Refusing that environment would make the
    # repair command the installer prints fail exactly when it is needed.
    _fake_uv(tmp_path, monkeypatch)
    venv = _plain_venv(tmp_path, "pip", "numpy")
    requirements_file(venv).write_text("numpy>=2\n", encoding="utf-8")

    fingerprint = install(["robot-alpha"], root=repo, venv=venv, verbose=False)

    assert fingerprint["targets"] == ["robot-alpha"]


def test_repairing_a_failed_install_reinstalls_every_requirement(
    repo, tmp_path, monkeypatch
):
    # uv treats package metadata as an installed package even when its files
    # are gone, so a repair that reused the leftovers would record a
    # fingerprint for an environment that does not work.
    _, install_arguments = _fake_uv(tmp_path, monkeypatch)
    fresh = tmp_path / "fresh"

    install(["robot-alpha"], root=repo, venv=fresh, verbose=False)

    assert "--reinstall" not in install_arguments()[0]

    venv = _plain_venv(tmp_path, "pip")
    requirements_file(venv).write_text("numpy>=2\n", encoding="utf-8")

    install(["robot-alpha"], root=repo, venv=venv, verbose=False)

    assert "--reinstall" in install_arguments()[2]


def test_install_reports_an_interpreter_it_cannot_run(repo, tmp_path, monkeypatch):
    _fake_uv(tmp_path, monkeypatch)
    venv = tmp_path / "venv"
    (venv / "bin").mkdir(parents=True)
    (venv / "bin" / "python").write_text("#!/bin/sh\n", encoding="utf-8")

    with pytest.raises(InstallError, match="cannot be run"):
        install(["robot-alpha"], root=repo, venv=venv, verbose=False)


def test_install_fails_before_uv_when_no_interpreter_fits(
    repo, tmp_path, write_toml, monkeypatch
):
    write_toml(
        "robots/beta/component.toml",
        f"""
        schema = {SCHEMA}
        name = "robot-beta"
        probe = "json"
        python = ">=3.10,<3.12"
        """,
    )
    _fake_uv(tmp_path, monkeypatch)
    # An existing virtualenv keeps the interpreter it was created with, so the
    # install checks the components against that interpreter before resolving.
    venv = tmp_path / "venv"
    (venv / "bin").mkdir(parents=True)
    python = venv / "bin" / "python"
    python.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    python.chmod(0o755)
    monkeypatch.setattr(
        "rpent.install.installer._python_version", lambda python: "3.12.1"
    )

    with pytest.raises(InstallError, match=r"requires Python >=3\.10,<3\.12"):
        install(["robot-beta"], root=repo, venv=tmp_path / "venv", verbose=False)

    # The check runs before uv installs anything.
    assert not (tmp_path / "uv-install-args").exists()


def test_install_accepts_an_interpreter_inside_every_component_range(
    repo, tmp_path, write_toml, monkeypatch
):
    write_toml(
        "robots/alpha/component.toml",
        f"""
        schema = {SCHEMA}
        name = "robot-alpha"
        probe = "json"
        python = ">=3.10,<3.14"
        """,
    )
    _fake_uv(tmp_path, monkeypatch)
    monkeypatch.setattr(
        "rpent.install.installer._python_version", lambda python: "3.12.1"
    )

    fingerprint = install(["robot-alpha"], root=repo, venv=tmp_path / "venv")

    assert fingerprint["python"] == "3.12.1"
