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

"""Unit tests for installer lifecycle, destination guards, doctor, and CLI."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

from rpent.cli import install as install_cli
from rpent.install import installer
from rpent.install.errors import InstallError


def _write_repo(tmp_path: Path) -> Path:
    tmp_path.mkdir(parents=True, exist_ok=True)
    (tmp_path / "pyproject.toml").write_text(
        '[project]\nname = "rpent"\ndependencies = ["numpy"]\n'
        '[project.optional-dependencies]\nplanner = ["httpx"]\n',
        encoding="utf-8",
    )
    (tmp_path / "components/infra/a").mkdir(parents=True)
    (tmp_path / "components/infra/a/component.toml").write_text(
        'schema = 2\nname = "infra-a"\nprobe = "json"\ndependencies = []\n',
        encoding="utf-8",
    )
    (tmp_path / "components/infra/b").mkdir(parents=True)
    (tmp_path / "components/infra/b/component.toml").write_text(
        'schema = 2\nname = "infra-b"\nprobe = "json"\ndependencies = ["scipy"]\n',
        encoding="utf-8",
    )
    return tmp_path


def _fake_venv(path: Path) -> Path:
    (path / "bin").mkdir(parents=True, exist_ok=True)
    (path / "pyvenv.cfg").write_text("home = /usr/bin\n", encoding="utf-8")
    (path / "bin/python").write_text("#!/bin/sh\n", encoding="utf-8")
    return path


def _write_valid_fingerprint(
    repo: Path, venv: Path, target: str = "infra-a", **overrides
) -> dict:
    catalog = installer.discover(repo)
    resolution = installer.resolve_targets([target], catalog, root=repo)
    fp = {
        "targets": [target],
        "python": "3.11.9",
        "platform": installer.runtime_platform(),
        "components": {
            name: {"manifest_sha256": comp.manifest_sha256}
            for name, comp in resolution.components.items()
        },
        "requirements_sha256": installer.requirements_sha256(
            resolution,
            root=repo,
            core_dependencies=installer.read_core_dependencies(repo),
            planner_dependencies=installer.read_planner_dependencies(repo),
        ),
        "freeze_sha256": installer.freeze_sha256("numpy==1.26.4\n"),
        "installed_at": "2026-01-01T00:00:00Z",
    }
    fp.update(overrides)
    installer.write_fingerprint(venv, fp)
    return fp


def test_install_rejects_non_empty_non_venv_directory(tmp_path, monkeypatch):
    repo = _write_repo(tmp_path / "repo")
    target = tmp_path / "user-data"
    target.mkdir()
    important = target / "notes.txt"
    important.write_text("keep me", encoding="utf-8")

    monkeypatch.setattr(installer, "find_uv", lambda: "uv")
    for force in (False, True):
        with pytest.raises(InstallError, match="not an empty directory or virtualenv"):
            installer.install(["infra-a"], root=repo, venv=target, force=force)
    assert important.read_text(encoding="utf-8") == "keep me"
    assert not (target / "pyvenv.cfg").exists()


def test_install_preserves_overlays_on_rerun_and_forwards_index_and_torch_backend(
    tmp_path, monkeypatch
):
    repo = _write_repo(tmp_path / "repo")
    venv = tmp_path / ".venv-a"
    commands: list[tuple[list[str], dict | None]] = []

    def fake_run(cmd, *, env=None, capture=False, cwd=None, timeout=3600.0):
        commands.append((list(cmd), env))
        if cmd[:2] == ["uv", "venv"]:
            _fake_venv(Path(cmd[-1]))
        elif "platform.python_version()" in " ".join(cmd):
            return "3.11.9\n"
        elif cmd[:3] == ["uv", "pip", "freeze"]:
            return "numpy==1.26.4\n"
        return ""

    monkeypatch.setattr(installer, "find_uv", lambda: "uv")
    monkeypatch.setattr(installer, "run_command", fake_run)

    fp = installer.install(
        ["infra-a"],
        root=repo,
        venv=venv,
        index_url="https://mirror.example/simple",
        torch_backend="cu126",
        verbose=False,
    )
    assert fp["targets"] == ["infra-a"]
    pip_cmds = [c for c in commands if c[0][:3] == ["uv", "pip", "install"]]
    assert len(pip_cmds) == 2
    for pip_cmd, pip_env in pip_cmds:
        assert "--index-url" in pip_cmd and "https://mirror.example/simple" in pip_cmd
        assert pip_env is not None and pip_env["UV_TORCH_BACKEND"] == "cu126"

    # Re-running the same closure updates in place without nuking overlaid files.
    overlay_marker = venv / "custom-overlay.txt"
    overlay_marker.write_text("kept", encoding="utf-8")
    installer.install(["infra-a"], root=repo, venv=venv, verbose=False)
    assert overlay_marker.read_text(encoding="utf-8") == "kept"

    # Different closure is rejected without --force.
    with pytest.raises(InstallError, match="different environment"):
        installer.install(["infra-b"], root=repo, venv=venv, verbose=False)


def test_doctor_warns_on_overlaid_packages_and_fails_on_unknown_python(
    tmp_path, monkeypatch
):
    repo = _write_repo(tmp_path / "repo")
    venv = _fake_venv(tmp_path / ".venv-a")
    fp = {
        "targets": ["infra-a"],
        "python": "3.11.9",
        "platform": installer.runtime_platform(),
        "components": {
            "infra-a": {
                "manifest_sha256": installer.discover(repo)
                .components["infra-a"]
                .manifest_sha256
            }
        },
        "requirements_sha256": installer.requirements_sha256(
            installer.resolve_targets(["infra-a"], installer.discover(repo), root=repo),
            root=repo,
            core_dependencies=("numpy",),
            planner_dependencies=("httpx",),
        ),
        "freeze_sha256": installer.freeze_sha256("numpy==1.26.4\n"),
        "installed_at": "2026-01-01T00:00:00Z",
    }
    installer.write_fingerprint(venv, fp)
    monkeypatch.setattr(installer, "find_uv", lambda: "uv")

    def fake_diag(cmd):
        if cmd[:3] == ["uv", "pip", "freeze"]:
            return subprocess.CompletedProcess(
                cmd, 0, "numpy==1.26.4\nmodelscope==1.0\n", ""
            )
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(installer, "_run_diag", fake_diag)
    checks = installer.check_installation(venv, root=repo)
    assert not any(c.status == installer.FAIL for c in checks)
    assert any(c.status == installer.WARN and "overlaid" in c.message for c in checks)

    # Unknown Python version fails closed.
    fp["python"] = "unknown"
    installer.write_fingerprint(venv, fp)
    checks = installer.check_installation(venv, root=repo)
    assert any(
        c.status == installer.FAIL and "Python version" in c.message for c in checks
    )


def test_current_venv_prefers_virtual_env_and_skips_bootstrap_cache(
    tmp_path, monkeypatch
):
    active = _fake_venv(tmp_path / "active")
    cache = _fake_venv(tmp_path / "uv-cache")
    monkeypatch.setenv("VIRTUAL_ENV", str(cache))
    monkeypatch.setenv("RPENT_INSTALL_BOOTSTRAP", "1")
    monkeypatch.setenv("RPENT_CALLER_VENV", str(active))
    assert installer.current_venv() == active

    monkeypatch.setenv("RPENT_CALLER_VENV", "")
    assert installer.current_venv() is None


def test_cli_list_and_check(tmp_path, capsys):
    repo = _write_repo(tmp_path / "repo")
    assert install_cli.main(["--root", str(repo), "list"]) == 0
    assert "infra-a" in capsys.readouterr().out
    assert install_cli.main(["--root", str(repo), "check", "--all"]) == 0
    assert "infra-a: ok" in capsys.readouterr().out


def test_install_force_restores_previous_venv_on_failure(tmp_path, monkeypatch):
    repo = _write_repo(tmp_path / "repo")
    venv = _fake_venv(tmp_path / ".venv-a")
    marker = venv / "keep.txt"
    marker.write_text("original", encoding="utf-8")
    _write_valid_fingerprint(repo, venv, "infra-a")

    def fake_run(cmd, *, env=None, capture=False, cwd=None, timeout=3600.0):
        if cmd[:2] == ["uv", "venv"]:
            _fake_venv(Path(cmd[-1]))
            return ""
        if "platform.python_version()" in " ".join(cmd):
            return "3.11.9\n"
        if cmd[:3] == ["uv", "pip", "install"]:
            raise InstallError("network down")
        return ""

    monkeypatch.setattr(installer, "find_uv", lambda: "uv")
    monkeypatch.setattr(installer, "run_command", fake_run)

    with pytest.raises(InstallError, match="network down"):
        installer.install(["infra-b"], root=repo, venv=venv, force=True, verbose=False)

    assert marker.read_text(encoding="utf-8") == "original"
    restored = installer.read_fingerprint(venv)
    assert restored is not None and restored["targets"] == ["infra-a"]
    assert not list(tmp_path.glob(".*.replaced-*"))


def test_install_force_replaces_venv_and_cleans_backup_on_success(
    tmp_path, monkeypatch
):
    repo = _write_repo(tmp_path / "repo")
    venv = _fake_venv(tmp_path / ".venv-a")
    old_marker = venv / "old.txt"
    old_marker.write_text("stale", encoding="utf-8")
    _write_valid_fingerprint(repo, venv, "infra-a")

    def fake_run(cmd, *, env=None, capture=False, cwd=None, timeout=3600.0):
        if cmd[:2] == ["uv", "venv"]:
            _fake_venv(Path(cmd[-1]))
        elif "platform.python_version()" in " ".join(cmd):
            return "3.11.9\n"
        elif cmd[:3] == ["uv", "pip", "freeze"]:
            return "numpy==1.26.4\nscipy==1.14.0\n"
        return ""

    monkeypatch.setattr(installer, "find_uv", lambda: "uv")
    monkeypatch.setattr(installer, "run_command", fake_run)

    fp = installer.install(["infra-b"], root=repo, venv=venv, force=True, verbose=False)
    assert fp["targets"] == ["infra-b"]
    assert not old_marker.exists()
    assert not list(tmp_path.glob(".*.replaced-*"))


def test_install_force_accepts_existing_empty_directory(tmp_path, monkeypatch):
    repo = _write_repo(tmp_path / "repo")
    empty = tmp_path / ".venv-empty"
    empty.mkdir()

    def fake_run(cmd, *, env=None, capture=False, cwd=None, timeout=3600.0):
        if cmd[:2] == ["uv", "venv"]:
            _fake_venv(Path(cmd[-1]))
        elif "platform.python_version()" in " ".join(cmd):
            return "3.11.9\n"
        elif cmd[:3] == ["uv", "pip", "freeze"]:
            return "numpy==1.26.4\n"
        return ""

    monkeypatch.setattr(installer, "find_uv", lambda: "uv")
    monkeypatch.setattr(installer, "run_command", fake_run)

    fp = installer.install(
        ["infra-a"], root=repo, venv=empty, force=True, verbose=False
    )
    assert fp["targets"] == ["infra-a"]
    assert (empty / "pyvenv.cfg").is_file()
    assert not list(tmp_path.glob(".*.replaced-*"))

    failing = tmp_path / ".venv-empty-fail"
    failing.mkdir()

    def fail_pip(cmd, *, env=None, capture=False, cwd=None, timeout=3600.0):
        if cmd[:2] == ["uv", "venv"]:
            _fake_venv(Path(cmd[-1]))
            return ""
        if "platform.python_version()" in " ".join(cmd):
            return "3.11.9\n"
        if cmd[:3] == ["uv", "pip", "install"]:
            raise InstallError("pip failed")
        return ""

    monkeypatch.setattr(installer, "run_command", fail_pip)
    with pytest.raises(InstallError, match="pip failed"):
        installer.install(
            ["infra-a"], root=repo, venv=failing, force=True, verbose=False
        )
    assert not failing.exists()


def test_install_force_rejects_stale_backup_directory(tmp_path, monkeypatch):
    repo = _write_repo(tmp_path / "repo")
    venv = _fake_venv(tmp_path / ".venv-a")
    stale_backup = venv.with_name(f".{venv.name}.replaced-{os.getpid()}")
    stale_backup.mkdir()

    monkeypatch.setattr(installer, "find_uv", lambda: "uv")
    with pytest.raises(InstallError, match="interrupted --force rebuild"):
        installer.install(["infra-a"], root=repo, venv=venv, force=True, verbose=False)
    assert venv.is_dir()


def test_install_removes_newly_created_venv_on_failure(tmp_path, monkeypatch):
    repo = _write_repo(tmp_path / "repo")
    venv = tmp_path / ".venv-fresh"

    def fake_run(cmd, *, env=None, capture=False, cwd=None, timeout=3600.0):
        if cmd[:2] == ["uv", "venv"]:
            _fake_venv(Path(cmd[-1]))
            return ""
        if "platform.python_version()" in " ".join(cmd):
            return "3.11.9\n"
        if cmd[:3] == ["uv", "pip", "install"]:
            raise InstallError("timed out")
        return ""

    monkeypatch.setattr(installer, "find_uv", lambda: "uv")
    monkeypatch.setattr(installer, "run_command", fake_run)

    with pytest.raises(InstallError, match="timed out"):
        installer.install(["infra-a"], root=repo, venv=venv, verbose=False)
    assert not venv.exists()


def test_install_rejects_foreign_unrunnable_or_invalid_venv_and_accepts_base_only(
    tmp_path, monkeypatch
):
    repo = _write_repo(tmp_path / "repo")
    venv = _fake_venv(tmp_path / ".venv-foreign")
    monkeypatch.setattr(installer, "find_uv", lambda: "uv")

    # Foreign packages (>5) are truncated and mention both external & interrupted venvs.
    monkeypatch.setattr(
        installer,
        "_installed_distributions",
        lambda _: [
            "pip",
            "setuptools",
            "wheel",
            "alpha",
            "beta",
            "gamma",
            "delta",
            "epsilon",
            "zeta",
        ],
    )
    with pytest.raises(InstallError) as exc_info:
        installer.install(["infra-a"], root=repo, venv=venv, verbose=False)
    message = str(exc_info.value)
    assert "alpha, beta, delta, epsilon, gamma, and 1 more" in message
    assert "outside rpent-install" in message
    assert "interrupted" in message
    assert "--force" in message

    # Unrunnable Python interpreter fails cleanly.
    monkeypatch.setattr(installer, "_installed_distributions", lambda _: None)
    with pytest.raises(InstallError, match="cannot be run"):
        installer.install(["infra-a"], root=repo, venv=venv, verbose=False)

    # Corrupt/invalid fingerprint without a components dict requires --force.
    installer.write_fingerprint(venv, {"targets": ["infra-a"]})
    with pytest.raises(InstallError, match="invalid fingerprint"):
        installer.install(["infra-a"], root=repo, venv=venv, verbose=False)

    # A pre-created bare venv with only base distributions is accepted without --force.
    bare = _fake_venv(tmp_path / ".venv-bare")
    monkeypatch.setattr(
        installer, "_installed_distributions", lambda _: ["pip", "setuptools", "wheel"]
    )

    def fake_run(cmd, *, env=None, capture=False, cwd=None, timeout=3600.0):
        if "platform.python_version()" in " ".join(cmd):
            return "3.11.9\n"
        if cmd[:3] == ["uv", "pip", "freeze"]:
            return "numpy==1.26.4\n"
        return ""

    monkeypatch.setattr(installer, "run_command", fake_run)
    fp = installer.install(["infra-a"], root=repo, venv=bare, verbose=False)
    assert fp["targets"] == ["infra-a"]


def test_install_dry_run_and_default_venv_path(tmp_path, monkeypatch, capsys):
    repo = _write_repo(tmp_path / "repo")

    def fail_uv():
        pytest.fail("find_uv must not run during --dry-run")

    monkeypatch.setattr(installer, "find_uv", fail_uv)
    result = installer.install(["infra-b", "infra-a"], root=repo, dry_run=True)

    expected_venv = (repo / ".rpent" / "venvs" / "infra-a+infra-b").resolve()
    assert result["targets"] == ["infra-a", "infra-b"]
    assert Path(result["venv"]) == expected_venv
    assert not expected_venv.exists()
    assert "# Generated by rpent-install for: infra-a infra-b" in result["requirements"]
    assert "scipy" in result["requirements"]
    assert capsys.readouterr().out == result["requirements"]


def test_probe_modules_and_command_cover_checkout_roles_and_external_modules(tmp_path):
    assert installer.probe_modules("json", "infra") == ("json",)
    assert installer.probe_command(Path("/py"), "json", "infra") == [
        "/py",
        "-I",
        "-c",
        "import json\n",
    ]

    robot_dir = tmp_path / "robots/alpha"
    robot_dir.mkdir(parents=True)
    (robot_dir / "robot_spec.py").write_text("", encoding="utf-8")
    assert installer.probe_modules("robots.alpha", "robot", source=robot_dir) == (
        "robots.alpha",
        "robots.alpha.robot_spec",
    )
    cmd = installer.probe_command(
        Path("/py"), "robots.alpha", "robot", source=robot_dir
    )
    assert cmd[:3] == ["/py", "-I", "-c"]
    assert "import robots.alpha" in cmd[3]
    assert 'importlib.import_module("robots.alpha.robot_spec")' in cmd[3]


def test_doctor_reports_missing_and_empty_fingerprints(tmp_path):
    repo = _write_repo(tmp_path / "repo")
    venv = _fake_venv(tmp_path / ".venv-empty")

    checks = installer.check_installation(venv, root=repo)
    assert len(checks) == 1
    assert checks[0].status == installer.FAIL
    assert installer.FINGERPRINT_NAME in checks[0].message
    assert installer.inspect_installation(venv) is None

    installer.write_fingerprint(venv, {"targets": []})
    checks = installer.check_installation(venv, root=repo)
    assert len(checks) == 1
    assert checks[0].status == installer.FAIL
    assert "no target components" in checks[0].message


def test_doctor_detects_stale_manifests_and_closure_drift(tmp_path, monkeypatch):
    repo = _write_repo(tmp_path / "repo")
    venv = _fake_venv(tmp_path / ".venv-a")
    _write_valid_fingerprint(repo, venv, "infra-a")
    monkeypatch.setattr(installer, "find_uv", lambda: "uv")

    def fake_diag(cmd):
        if cmd[:3] == ["uv", "pip", "freeze"]:
            return subprocess.CompletedProcess(cmd, 0, "numpy==1.26.4\n", "")
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(installer, "_run_diag", fake_diag)
    clean = installer.check_installation(venv, root=repo)
    assert not any(c.status in {installer.FAIL, installer.WARN} for c in clean)

    # Manifest dependency change triggers requirements_sha256 mismatch.
    manifest = repo / "components/infra/a/component.toml"
    manifest.write_text(
        'schema = 2\nname = "infra-a"\nprobe = "json"\ndependencies = ["pyyaml"]\n',
        encoding="utf-8",
    )
    stale = installer.check_installation(venv, root=repo)
    assert any(c.status == installer.FAIL and "out of date" in c.message for c in stale)

    # Closure change triggers component-set drift with --force repair hint.
    manifest.write_text(
        'schema = 2\nname = "infra-a"\nprobe = "json"\n'
        'requires = ["infra-b"]\ndependencies = []\n',
        encoding="utf-8",
    )
    drift = installer.check_installation(venv, root=repo)
    assert any(
        c.status == installer.FAIL
        and "different component set" in c.message
        and "--force" in c.message
        for c in drift
    )


def test_doctor_detects_platform_mismatch_uv_and_probe_failures(tmp_path, monkeypatch):
    repo = _write_repo(tmp_path / "repo")
    venv = _fake_venv(tmp_path / ".venv-a")
    _write_valid_fingerprint(repo, venv, "infra-a", platform="other-arch")
    monkeypatch.setattr(installer, "find_uv", lambda: "uv")

    def fake_diag(cmd):
        if cmd[:3] == ["uv", "pip", "freeze"]:
            return subprocess.CompletedProcess(cmd, 1, "", "freeze failed")
        if cmd[:3] == ["uv", "pip", "check"]:
            return subprocess.CompletedProcess(cmd, 1, "", "broken dep")
        return subprocess.CompletedProcess(
            cmd, 1, "", "Traceback\nModuleNotFoundError: No module named 'json'"
        )

    monkeypatch.setattr(installer, "_run_diag", fake_diag)
    checks = installer.check_installation(venv, root=repo)
    assert any(c.status == installer.FAIL and "other-arch" in c.message for c in checks)
    assert any(
        c.status == installer.FAIL and "uv pip freeze failed" in c.message
        for c in checks
    )
    assert any(
        c.status == installer.FAIL
        and "uv pip check failed" in c.message
        and "broken dep" in c.message
        for c in checks
    )
    assert any(
        c.status == installer.FAIL
        and "component 'infra-a' is not importable" in c.message
        and "ModuleNotFoundError" in c.message
        for c in checks
    )


def test_doctor_handles_missing_uv_and_missing_python(tmp_path, monkeypatch):
    repo = _write_repo(tmp_path / "repo")
    venv = _fake_venv(tmp_path / ".venv-a")
    _write_valid_fingerprint(repo, venv, "infra-a")

    def missing_uv():
        raise InstallError("uv is required")

    monkeypatch.setattr(installer, "find_uv", missing_uv)
    monkeypatch.setattr(
        installer,
        "_run_diag",
        lambda cmd: subprocess.CompletedProcess(cmd, 0, "", ""),
    )
    checks = installer.check_installation(venv, root=repo)
    assert not any(c.status == installer.FAIL for c in checks)
    assert any(
        c.status == installer.WARN and "uv is required" in c.message for c in checks
    )
    assert any(
        c.status == installer.OK and "probe infra-a" in c.message for c in checks
    )

    (venv / "bin/python").unlink()
    missing_py = installer.check_installation(venv, root=repo)
    assert any(
        c.status == installer.FAIL and "no Python interpreter" in c.message
        for c in missing_py
    )


def test_cli_doctor_fingerprint_and_argument_errors(tmp_path, monkeypatch, capsys):
    repo = _write_repo(tmp_path / "repo")
    venv = _fake_venv(tmp_path / ".venv-a")
    _write_valid_fingerprint(repo, venv, "infra-a")

    assert (
        install_cli.main(
            ["--root", str(repo), "doctor", "--venv", str(venv), "--fingerprint"]
        )
        == 0
    )
    out = capsys.readouterr().out
    assert f"venv: {venv.resolve()}" in out
    assert "sha256: " in out
    assert "targets: ['infra-a']" in out
    assert "components:" in out
    assert "  infra-a:" in out

    empty = _fake_venv(tmp_path / ".venv-empty")
    assert (
        install_cli.main(
            ["--root", str(repo), "doctor", "--venv", str(empty), "--fingerprint"]
        )
        == 1
    )
    assert "no installation fingerprint" in capsys.readouterr().err

    monkeypatch.setattr(install_cli, "current_venv", lambda: None)
    assert install_cli.main(["--root", str(repo), "doctor"]) == 2
    assert "no active virtualenv" in capsys.readouterr().err

    assert install_cli.main(["--root", str(repo), "check", "--all", "infra-a"]) == 2
    assert install_cli.main(["--root", str(repo), "check"]) == 2
    assert install_cli.main(["--root", str(repo), "check", "infra-missing"]) == 1


def test_install_rejects_broken_venv_without_python_and_preserves_directory(
    tmp_path, monkeypatch
):
    repo = _write_repo(tmp_path / "repo")
    broken = tmp_path / ".venv-broken"
    broken.mkdir()
    (broken / "pyvenv.cfg").write_text("home = /usr/bin\n", encoding="utf-8")
    marker = broken / "notes.txt"
    marker.write_text("keep", encoding="utf-8")

    monkeypatch.setattr(installer, "find_uv", lambda: "uv")
    with pytest.raises(InstallError, match="pyvenv.cfg.*no Python interpreter"):
        installer.install(["infra-a"], root=repo, venv=broken, verbose=False)
    assert broken.is_dir()
    assert marker.read_text(encoding="utf-8") == "keep"

    def fake_run(cmd, *, env=None, capture=False, cwd=None, timeout=3600.0):
        if cmd[:2] == ["uv", "venv"]:
            _fake_venv(Path(cmd[-1]))
        elif "platform.python_version()" in " ".join(cmd):
            return "3.11.9\n"
        elif cmd[:3] == ["uv", "pip", "freeze"]:
            return "numpy==1.26.4\n"
        return ""

    monkeypatch.setattr(installer, "run_command", fake_run)
    fp = installer.install(
        ["infra-a"], root=repo, venv=broken, force=True, verbose=False
    )
    assert fp["targets"] == ["infra-a"]
    assert not marker.exists()


def test_install_rechecks_patch_level_python_after_creating_venv(tmp_path, monkeypatch):
    repo = _write_repo(tmp_path / "repo")
    (repo / "components/infra/a/component.toml").write_text(
        'schema = 2\nname = "infra-a"\nprobe = "rpent"\npython = ">=3.11.4,<3.12"\n',
        encoding="utf-8",
    )
    venv = tmp_path / ".venv-patch"
    monkeypatch.setattr(installer, "find_uv", lambda: "uv")

    def fake_run(cmd, *, env=None, capture=False, cwd=None, timeout=3600.0):
        if cmd[:2] == ["uv", "venv"]:
            _fake_venv(Path(cmd[-1]))
        elif "platform.python_version()" in " ".join(cmd):
            return "3.11.2\n"
        return ""

    monkeypatch.setattr(installer, "run_command", fake_run)
    with pytest.raises(InstallError, match="3.11.2"):
        installer.install(["infra-a"], root=repo, venv=venv, verbose=False)
    assert not venv.exists()


def test_install_python_flag_and_doctor_fingerprint_outside_checkout(
    tmp_path, monkeypatch, capsys
):
    repo = _write_repo(tmp_path / "repo")
    venv = tmp_path / ".venv-py311"
    monkeypatch.setattr(installer, "find_uv", lambda: "uv")
    seen_venv_cmd: list[str] = []

    def fake_run(cmd, *, env=None, capture=False, cwd=None, timeout=3600.0):
        if cmd[:2] == ["uv", "venv"]:
            seen_venv_cmd.extend(cmd)
            _fake_venv(Path(cmd[-1]))
        elif "platform.python_version()" in " ".join(cmd):
            return "3.11.9\n"
        elif cmd[:3] == ["uv", "pip", "freeze"]:
            return "numpy==1.26.4\n"
        return ""

    monkeypatch.setattr(installer, "run_command", fake_run)
    assert (
        install_cli.main(
            [
                "--root",
                str(repo),
                "install",
                "infra-a",
                "--python",
                "3.11",
                "--venv",
                str(venv),
            ]
        )
        == 0
    )
    assert "--python" in seen_venv_cmd
    assert seen_venv_cmd[seen_venv_cmd.index("--python") + 1] == "3.11"

    outside = tmp_path / "outside"
    outside.mkdir()
    monkeypatch.chdir(outside)
    monkeypatch.delenv("RPENT_DEFAULT_ROOT", raising=False)
    assert install_cli.main(["doctor", "--venv", str(venv), "--fingerprint"]) == 0
    assert "python: 3.11.9" in capsys.readouterr().out


def test_doctor_fails_on_invalid_components_record(tmp_path, monkeypatch):
    repo = _write_repo(tmp_path / "repo")
    venv = _fake_venv(tmp_path / ".venv-bad-fp")
    fp = _write_valid_fingerprint(repo, venv, "infra-a")
    fp["components"] = []
    installer.write_fingerprint(venv, fp)
    monkeypatch.setattr(installer, "find_uv", lambda: "uv")
    monkeypatch.setattr(
        installer,
        "_run_diag",
        lambda cmd: subprocess.CompletedProcess(
            cmd,
            0,
            "numpy==1.26.4\n" if cmd[:3] == ["uv", "pip", "freeze"] else "",
            "",
        ),
    )

    checks = installer.check_installation(venv, root=repo)
    assert any(
        c.status == installer.FAIL and "invalid components record" in c.message
        for c in checks
    )


def test_install_and_doctor_downgrade_pip_check_when_overrides_declared(
    tmp_path, monkeypatch, capsys
):
    repo = _write_repo(tmp_path / "repo")
    override_file = repo / "components/infra/a/override.txt"
    override_file.write_text("torch==2.7.0\n", encoding="utf-8")
    (repo / "components/infra/a/component.toml").write_text(
        'schema = 2\nname = "infra-a"\nprobe = "json"\n'
        'overrides = ["override.txt"]\ndependencies = []\n',
        encoding="utf-8",
    )
    venv = tmp_path / ".venv-override"
    ran_hard_check = False

    def fake_run(cmd, *, env=None, capture=False, cwd=None, timeout=3600.0):
        nonlocal ran_hard_check
        if cmd[:2] == ["uv", "venv"]:
            _fake_venv(Path(cmd[-1]))
        elif "platform.python_version()" in " ".join(cmd):
            return "3.11.9\n"
        elif cmd[:3] == ["uv", "pip", "check"]:
            ran_hard_check = True
            raise InstallError("uv pip check should not hard-fail with overrides")
        elif cmd[:3] == ["uv", "pip", "freeze"]:
            return "numpy==1.26.4\n"
        return ""

    def fake_diag(cmd):
        if cmd[:3] == ["uv", "pip", "check"]:
            return subprocess.CompletedProcess(
                cmd, 1, "", "pkg requires torch==2.7.1, but torch 2.7.0 is installed"
            )
        if cmd[:3] == ["uv", "pip", "freeze"]:
            return subprocess.CompletedProcess(cmd, 0, "numpy==1.26.4\n", "")
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(installer, "find_uv", lambda: "uv")
    monkeypatch.setattr(installer, "run_command", fake_run)
    monkeypatch.setattr(installer, "_run_diag", fake_diag)

    fp = installer.install(["infra-a"], root=repo, venv=venv, verbose=True)
    assert not ran_hard_check
    assert venv.is_dir()
    assert fp["targets"] == ["infra-a"]
    assert "override-induced mismatches" in capsys.readouterr().err

    checks = installer.check_installation(venv, root=repo)
    assert not any(c.status == installer.FAIL for c in checks)
    assert any(
        c.status == installer.WARN and "override-induced mismatches" in c.message
        for c in checks
    )


def test_install_rejects_broken_venv_with_same_closure_fingerprint_without_deleting(
    tmp_path, monkeypatch
):
    repo = _write_repo(tmp_path / "repo")
    venv = _fake_venv(tmp_path / ".venv-dangling")
    _write_valid_fingerprint(repo, venv, "infra-a")
    marker = venv / "keep.txt"
    marker.write_text("do-not-delete", encoding="utf-8")
    (venv / "bin/python").unlink()

    monkeypatch.setattr(installer, "find_uv", lambda: "uv")
    with pytest.raises(InstallError, match="pyvenv.cfg.*no Python interpreter"):
        installer.install(["infra-a"], root=repo, venv=venv, verbose=False)
    assert venv.is_dir()
    assert marker.read_text(encoding="utf-8") == "do-not-delete"
