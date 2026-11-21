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

"""Doctor reports missing, stale, and unusable installations."""

import pytest

from rpent.install.discovery import discover
from rpent.install.doctor import (
    FAIL,
    OK,
    WARN,
    check_installation,
    inspect_installation,
)
from rpent.install.errors import InstallError
from rpent.install.fingerprint import (
    FINGERPRINT_SCHEMA_FIELD,
    fingerprint_sha256,
    write_fingerprint,
)
from rpent.install.installer import resolve_targets
from rpent.install.manifest import SCHEMA
from rpent.install.requirements import FINGERPRINT_SCHEMA, requirements_sha256


@pytest.fixture
def repo(tmp_path, write_toml):
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
        f"""
        schema = {SCHEMA}
        name = "robot-alpha"
        probe = "json"
        python = ">=3.10,<3.13"
        dependencies = ["numpy>=2"]
        """,
    )
    return tmp_path


def _venv(tmp_path):
    venv = tmp_path / "venv"
    (venv / "bin").mkdir(parents=True)
    python = venv / "bin" / "python"
    python.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    python.chmod(0o755)
    return venv


def _fingerprint(repo, *, targets=("robot-alpha",), python="3.11.9", **extra):
    resolution = resolve_targets(targets, discover(repo), root=repo)
    assert resolution.ok, resolution.errors
    fingerprint = {
        FINGERPRINT_SCHEMA_FIELD: FINGERPRINT_SCHEMA,
        "targets": list(targets),
        "python": python,
        "requirements_sha256": requirements_sha256(resolution, root=repo),
    }
    fingerprint.update(extra)
    return fingerprint


def _missing_uv(monkeypatch):
    def _fail():
        raise InstallError("uv missing")

    monkeypatch.setattr("rpent.install.doctor.find_uv", _fail)


def test_missing_fingerprint_reports_install_command(tmp_path):
    checks = check_installation(tmp_path / "venv", root=tmp_path)

    assert [check.status for check in checks] == [FAIL]
    assert "rpent-install install <components>" in checks[0].message


def test_fingerprint_without_targets_asks_for_a_reinstall(repo):
    venv = _venv(repo)
    write_fingerprint(
        venv, {FINGERPRINT_SCHEMA_FIELD: FINGERPRINT_SCHEMA, "python": "3.11.9"}
    )

    checks = check_installation(venv, root=repo)

    assert [check.status for check in checks] == [FAIL]
    assert "records no target components" in checks[0].message


def test_unresolvable_targets_are_reported(repo, monkeypatch):
    venv = _venv(repo)
    write_fingerprint(
        venv,
        {
            FINGERPRINT_SCHEMA_FIELD: FINGERPRINT_SCHEMA,
            "targets": ["robot-removed"],
            "python": "3.11.9",
        },
    )

    checks = check_installation(venv, root=repo)

    assert (checks[0].status, checks[0].message) == (OK, "targets: robot-removed")
    assert any(
        check.status == FAIL and "robot-removed" in check.message for check in checks
    )


def test_stale_fingerprint_fails_before_running_checks(repo, monkeypatch):
    venv = _venv(repo)
    write_fingerprint(venv, _fingerprint(repo, requirements_sha256="stale"))
    _missing_uv(monkeypatch)

    checks = check_installation(venv, root=repo)

    assert [check.status for check in checks[:3]] == [OK, OK, FAIL]
    assert "out of date" in checks[2].message
    assert "rpent-install install robot-alpha" in checks[2].message
    assert any(
        check.status == WARN and "uv missing" in check.message for check in checks
    )


def test_a_changed_component_closure_repairs_with_force(repo, write_toml, monkeypatch):
    write_toml(
        "robots/beta/component.toml",
        f"""
        schema = {SCHEMA}
        name = "robot-beta"
        probe = "json"
        python = ">=3.10,<3.13"
        """,
    )
    venv = _venv(repo)
    write_fingerprint(
        venv,
        _fingerprint(
            repo,
            targets=("robot-beta",),
            components={"robot-alpha": {}},
        ),
    )
    _missing_uv(monkeypatch)

    checks = check_installation(venv, root=repo)

    message = next(
        check.message for check in checks if "different component set" in check.message
    )
    assert "rpent-install install robot-beta --force" in message


def test_a_package_only_change_repairs_without_force(repo, monkeypatch):
    venv = _venv(repo)
    write_fingerprint(
        venv,
        _fingerprint(
            repo,
            components={"robot-alpha": {}},
            requirements_sha256="stale",
        ),
    )
    _missing_uv(monkeypatch)

    checks = check_installation(venv, root=repo)

    message = next(check.message for check in checks if "out of date" in check.message)
    assert "rpent-install install robot-alpha" in message
    assert "--force" not in message


def test_fingerprint_from_an_older_input_mode_asks_for_a_reinstall(repo, monkeypatch):
    venv = _venv(repo)
    recorded = {
        "targets": ["robot-alpha"],
        "python": "3.11.9",
        "requirements_sha256": "written by an older rpent-install",
    }
    _missing_uv(monkeypatch)

    older = FINGERPRINT_SCHEMA - 1
    for schema, label in ((older, str(older)), (None, "unknown")):
        fingerprint = dict(recorded)
        if schema is not None:
            fingerprint[FINGERPRINT_SCHEMA_FIELD] = schema
        write_fingerprint(venv, fingerprint)

        checks = check_installation(venv, root=repo)

        assert any(
            check.status == FAIL and f"input mode {label}" in check.message
            for check in checks
        )
        # The hash was recorded by a different input mode, so comparing it
        # would misreport drifted inputs instead of the required reinstall.
        assert not any("out of date" in check.message for check in checks)


def test_unchanged_inputs_and_probes_pass(repo, monkeypatch):
    venv = _venv(repo)
    write_fingerprint(venv, _fingerprint(repo))
    _missing_uv(monkeypatch)

    checks = check_installation(venv, root=repo)

    assert checks[0].message == "targets: robot-alpha"
    assert any(
        check.status == OK and check.message == "installation inputs unchanged"
        for check in checks
    )
    assert any(
        check.status == OK and check.message == "probe robot-alpha: import json"
        for check in checks
    )
    assert not any(check.status == FAIL for check in checks)


def test_recorded_interpreter_rejecting_the_components_is_reported(repo, write_toml):
    write_toml(
        "robots/beta/component.toml",
        f"""
        schema = {SCHEMA}
        name = "robot-beta"
        probe = "json"
        python = ">=3.10,<3.12"
        """,
    )
    venv = _venv(repo)
    write_fingerprint(
        venv, _fingerprint(repo, targets=("robot-beta",), python="3.12.1")
    )

    checks = check_installation(venv, root=repo)

    assert any(
        check.status == FAIL
        and "component 'robot-beta' requires Python >=3.10,<3.12" in check.message
        and "3.12.1" in check.message
        for check in checks
    )


def test_a_venv_installed_for_another_platform_is_reported(repo, monkeypatch):
    # The packages were resolved for the recorded platform; this machine may
    # have no wheel for them.
    venv = _venv(repo)
    write_fingerprint(venv, _fingerprint(repo, platform="linux-s390x"))
    _missing_uv(monkeypatch)

    checks = check_installation(venv, root=repo)

    assert any(
        check.status == FAIL
        and "installed for linux-s390x" in check.message
        # A venv another platform built cannot be repaired in place.
        and "--force" in check.message
        for check in checks
    )


def test_any_interpreter_inside_the_component_ranges_is_accepted(repo):
    # CI creates an environment with its own minor version while the recorded
    # interpreter is whatever created the venv; it only has to satisfy the
    # installed components.
    venv = _venv(repo)
    write_fingerprint(venv, _fingerprint(repo, python="3.12.1"))

    checks = check_installation(venv, root=repo)

    assert any(
        check.status == OK and check.message == "python: 3.12.1" for check in checks
    )
    assert not any("requires Python" in check.message for check in checks)


def test_inspect_installation_reports_fingerprint_hash(tmp_path):
    venv = tmp_path / "venv"
    venv.mkdir()
    fingerprint = {"targets": ["robot-alpha"]}
    write_fingerprint(venv, fingerprint)

    inspected = inspect_installation(venv)

    assert inspected["fingerprint"] == fingerprint
    assert inspected["sha256"] == fingerprint_sha256(fingerprint)
    assert inspect_installation(tmp_path / "missing") is None
