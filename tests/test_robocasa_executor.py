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

from __future__ import annotations

import hashlib
import json
import os
import stat
import threading
from dataclasses import replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace

import pytest

from rpent.reproduce.robocasa import (
    executor,
    preflight,
    preliminary_driver,
    secure_script,
)
from rpent.reproduce.robocasa import sandbox as planner_sandbox
from rpent.reproduce.robocasa.deadline_supervisor import (
    CONTRACT_NAME,
    GATE_NAME,
    _open_gate,
)
from rpent.reproduce.robocasa.executor import ExecutorConfig, RuntimePaths, doctor
from rpent.reproduce.robocasa.planner_transport import (
    CHATGPT_BROKER_PROFILE,
    CHATGPT_BROKER_PROTOCOL,
    CODEX_REQUEST_MAX_RETRIES,
    CODEX_STREAM_IDLE_TIMEOUT_MS,
    CODEX_STREAM_MAX_RETRIES,
    PlannerTransport,
)
from rpent.reproduce.robocasa.profiles import get_profile
from rpent.reproduce.robocasa.protocol import cell_for


def _write(path: Path, text: str = "fixture\n", *, mode: int = 0o600) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    path.chmod(mode)
    return path


def _runtime(tmp_path: Path) -> RuntimePaths:
    root = tmp_path / "runtime"
    model = root / "model"
    vlm = root / "vlm"
    model.mkdir(parents=True)
    vlm.mkdir()
    scripts = root / "scripts"
    _write(root / "migration/robocasa_interactive_env.py")
    _write(root / "migration/rldx_skill.py")
    _write(root / "rldx/__init__.py")
    _write(root / "external_dependencies/robocasa365/robocasa/__init__.py")
    _write(root / "external_dependencies/robocasa365/robosuite/robosuite/__init__.py")
    navview = root / "robosuite/robosuite/models/assets/bases/omron_mobile_base.xml"
    _write(
        navview,
        """<mujoco><worldbody><body name="base">
<camera name="navview" mode="fixed" pos="0.2 0 1.6"
xyaxes="0 -1 0 0.643 0 0.766" fovy="75"/>
</body></worldbody></mujoco>\n""",
    )
    for name in executor.CHECKPOINT_SHA256:
        _write(model / name)
    return RuntimePaths(
        root=root,
        sim_python=_write(scripts / "python", mode=0o700),
        driver=_write(scripts / "driver.py"),
        readiness=_write(scripts / "ready.py"),
        deadline=_write(scripts / "deadline.py"),
        isolation_launcher=_write(scripts / "isolate.py"),
        artifact_builder=_write(scripts / "artifacts.py"),
        model=model,
        vlm_metadata=vlm,
        navview_xml=navview,
    )


def _config(tmp_path: Path, *, preliminary: bool = True) -> ExecutorConfig:
    results = tmp_path / "results"
    rollouts = tmp_path / "rollouts"
    memory = tmp_path / "memory"
    for path in (results, memory):
        path.mkdir(mode=0o700)
    rollouts.mkdir(mode=0o711)
    _write(memory / "manifest.json", "{}\n")
    credential = _write(tmp_path / "secret", "secret-value\n")
    return ExecutorConfig(
        runtime=_runtime(tmp_path),
        results_root=results,
        rollout_root=rollouts,
        memory_root=memory,
        planner_profile=get_profile("codex-gpt55-xhigh"),
        planner_transport=PlannerTransport.api_key(
            credential_file=credential,
            base_url="https://planner.example/v1",
        ),
        codex_bin=_write(tmp_path / "codex", mode=0o700),
        preliminary_local_runtime=preliminary,
    )


def _patch_frozen_identities(monkeypatch) -> None:
    monkeypatch.setattr(executor, "_frozen_runtime_problem", lambda *_args: None)
    monkeypatch.setattr(
        executor,
        "_pinned_codex_identity",
        lambda path: (
            {
                "path": str(path),
                "sha256": executor.PINNED_CODEX_SHA256,
                "version": executor.PINNED_CODEX_VERSION,
            },
            None,
        ),
    )
    monkeypatch.setattr(executor, "_managed_codex_config_problem", lambda: None)


def test_doctor_accepts_audited_preliminary_fixture(tmp_path, monkeypatch):
    _patch_frozen_identities(monkeypatch)
    monkeypatch.setattr(executor, "validate_memory_pack", lambda _root: [])
    monkeypatch.setattr(
        executor,
        "_runtime_import_identity",
        lambda _runtime: ({"robocasa": "fixture", "robosuite": "fixture"}, None),
    )
    monkeypatch.setattr(
        executor,
        "_git_snapshot",
        lambda root: {"root": str(root), "commit": "f" * 40, "dirty": False},
    )
    monkeypatch.setattr(executor, "_rollout_directory_problem", lambda _root: None)
    config = _config(tmp_path)
    report = doctor(config)
    assert report["ok"] is True
    assert report["release_ready"] is False
    assert report["runtime_kind"] == "preliminary_external_snapshot"
    assert report["memory"]["manifest_sha256"]
    assert "navview_patch" in report["scripts"]
    serialized = json.dumps(report, sort_keys=True)
    assert str(config.planner_transport.credential_file) not in serialized
    assert "secret-value" not in serialized


def test_doctor_requires_explicit_preliminary_acknowledgement(tmp_path, monkeypatch):
    _patch_frozen_identities(monkeypatch)
    monkeypatch.setattr(executor, "validate_memory_pack", lambda _root: [])
    monkeypatch.setattr(
        executor,
        "_runtime_import_identity",
        lambda _runtime: ({"robocasa": "fixture", "robosuite": "fixture"}, None),
    )
    report = doctor(_config(tmp_path, preliminary=False))
    assert report["ok"] is False
    assert any("preliminary" in problem for problem in report["problems"])


def test_navview_must_be_a_direct_child_of_the_omron_base(tmp_path):
    path = _write(
        tmp_path / "nested-navview.xml",
        """<mujoco><worldbody><body name="base"><body name="nested">
<camera name="navview" mode="fixed" pos="0.2 0 1.6"
xyaxes="0 -1 0 0.643 0 0.766" fovy="75"/>
</body></body></worldbody></mujoco>\n""",
    )

    assert "direct child" in executor._navview_problem(path)


def test_doctor_runs_active_isolation_preflight_when_requested(tmp_path, monkeypatch):
    _patch_frozen_identities(monkeypatch)
    monkeypatch.setattr(executor, "validate_memory_pack", lambda _root: [])
    monkeypatch.setattr(
        executor,
        "_runtime_import_identity",
        lambda _runtime: ({"robocasa": "fixture", "robosuite": "fixture"}, None),
    )
    monkeypatch.setattr(executor, "_rollout_directory_problem", lambda _root: None)
    calls = []

    def active(config):
        calls.append(config)
        return {"attestation_sha256": "a" * 64}

    monkeypatch.setattr(preflight, "run_isolation_preflight", active)
    config = _config(tmp_path)
    report = doctor(config, verify_isolation=True)
    assert report["ok"] is True
    assert calls == [config]
    assert report["isolation_preflight"] == {
        "verified": True,
        "attestation_sha256": "a" * 64,
    }


def test_doctor_reuses_valid_checkpoint_attestation(tmp_path, monkeypatch):
    _patch_frozen_identities(monkeypatch)
    monkeypatch.setattr(executor, "validate_memory_pack", lambda _root: [])
    monkeypatch.setattr(
        executor,
        "_runtime_import_identity",
        lambda _runtime: ({"robocasa": "fixture", "robosuite": "fixture"}, None),
    )
    monkeypatch.setattr(executor, "_rollout_directory_problem", lambda _root: None)
    attestation = {
        "fingerprint": "f" * 64,
        "files": {
            f"model/{name}": {"sha256": digest}
            for name, digest in executor.CHECKPOINT_SHA256.items()
        },
    }
    monkeypatch.setattr(
        executor, "_load_checkpoint_attestation", lambda _config: attestation
    )
    monkeypatch.setattr(
        executor,
        "verify_checkpoint_identity",
        lambda *_args, **_kwargs: pytest.fail("full hash must not run"),
    )
    report = doctor(_config(tmp_path), verify_checkpoint=True)
    assert report["ok"] is True
    assert report["checkpoint"]["verified"] is True
    assert report["checkpoint"]["verification_source"] == "reused_attestation"


def test_checkpoint_attestation_is_kept_out_of_planner_rollouts(tmp_path):
    config = _config(tmp_path)
    path = executor.checkpoint_attestation_path(config)
    assert path == config.results_root / "_preflight/checkpoint-attestation.json"
    assert config.rollout_root not in path.parents


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("https://planner.example", "https://planner.example/v1"),
        ("https://planner.example/v1/", "https://planner.example/v1"),
        ("http://127.0.0.1:4319", "http://127.0.0.1:4319/v1"),
    ],
)
def test_normalize_responses_base_url(raw, expected):
    assert executor.normalize_responses_base_url(raw) == expected


@pytest.mark.parametrize(
    "raw",
    [
        "http://example.com/v1",
        "https://user:secret@example.com/v1",
        "https://example.com/v1/responses",
        "https://example.com/v1?token=secret",
    ],
)
def test_normalize_responses_base_url_rejects_unsafe_values(raw):
    with pytest.raises(ValueError):
        executor.normalize_responses_base_url(raw)


def test_subscription_transport_requires_one_loopback_listener(tmp_path):
    credential = _write(tmp_path / "broker-capability", "capability\n")
    with pytest.raises(ValueError, match="loopback"):
        PlannerTransport.chatgpt_subscription(
            credential_file=credential,
            broker_base_url="https://broker.example/v1",
            broker_health_url="http://127.0.0.1:4319/health",
        )
    with pytest.raises(ValueError, match="same listener"):
        PlannerTransport.chatgpt_subscription(
            credential_file=credential,
            broker_base_url="http://127.0.0.1:4319/v1",
            broker_health_url="http://127.0.0.1:4320/health",
        )


@pytest.mark.parametrize(
    ("health_override", "expected_problem"),
    [({}, None), ({"reasoning_effort": "high"}, "identity differs")],
)
def test_subscription_broker_health_is_bound_to_frozen_profile(
    tmp_path, health_override, expected_problem
):
    payload = {
        "provider_profile": CHATGPT_BROKER_PROFILE,
        "auth_mode": "chatgpt_broker",
        "credential_broker": True,
        "credential_broker_ready": True,
        "credential_broker_protocol": CHATGPT_BROKER_PROTOCOL,
        "model": "gpt-5.5",
        "reasoning_effort": "xhigh",
        **health_override,
    }

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            body = json.dumps(payload).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, _format, *_args):
            return

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        port = int(server.server_address[1])
        config = _config(tmp_path)
        transport = PlannerTransport.chatgpt_subscription(
            credential_file=config.planner_transport.credential_file,
            broker_base_url=f"http://127.0.0.1:{port}/v1",
            broker_health_url=f"http://127.0.0.1:{port}/health",
        )
        problem = executor._broker_health_problem(
            replace(config, planner_transport=transport)
        )
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
    if expected_problem is None:
        assert problem is None
    else:
        assert expected_problem in problem


def test_secret_policy_rejects_group_access_and_symlinks(tmp_path):
    secret = _write(tmp_path / "secret", "secret-value\n", mode=0o640)
    assert "0600" in executor._trusted_credential(secret)
    secret.chmod(0o600)
    alias = tmp_path / "alias"
    alias.symlink_to(secret)
    assert "non-symlink" in executor._trusted_credential(alias)
    assert executor._trusted_credential(secret) is None


def test_secret_is_reopened_once_into_exec_closing_memfd(tmp_path):
    secret = _write(tmp_path / "secret", "fixture-secret\n", mode=0o600)
    descriptor, anonymous_path = planner_sandbox._secret_memfd(secret)
    try:
        assert Path(anonymous_path).read_text(encoding="utf-8") == "fixture-secret\n"
        assert os.get_inheritable(descriptor) is False
    finally:
        os.close(descriptor)
    alias = tmp_path / "alias"
    alias.symlink_to(secret)
    with pytest.raises(SystemExit, match="owned, regular"):
        planner_sandbox._secret_memfd(alias)


def test_secret_memfd_retries_partial_writes(tmp_path, monkeypatch):
    secret = _write(tmp_path / "secret", "fixture-secret\n", mode=0o600)
    real_write = os.write
    write_calls = 0

    def short_write(descriptor, data):
        nonlocal write_calls
        write_calls += 1
        return real_write(descriptor, data[:3])

    monkeypatch.setattr(planner_sandbox.os, "write", short_write)
    descriptor, anonymous_path = planner_sandbox._secret_memfd(secret)
    try:
        assert Path(anonymous_path).read_text(encoding="utf-8") == "fixture-secret\n"
        assert write_calls > 1
    finally:
        os.close(descriptor)


def test_launcher_is_compiled_from_pinned_open_bytes(tmp_path):
    source = b"VALUE = 7\n"
    launcher = tmp_path / "launcher.py"
    launcher.write_bytes(source)
    launcher.chmod(0o600)
    digest = hashlib.sha256(source).hexdigest()
    assert planner_sandbox._load_launcher(launcher, digest).VALUE == 7
    with pytest.raises(SystemExit, match="frozen SHA-256"):
        planner_sandbox._load_launcher(launcher, "0" * 64)
    launcher.chmod(0o620)
    with pytest.raises(SystemExit, match="immutable"):
        planner_sandbox._load_launcher(launcher, digest)


def _stage_deadline_controls(workdir: Path) -> dict[str, Path]:
    controls = {
        GATE_NAME: _write(workdir / GATE_NAME, "", mode=0o600),
        CONTRACT_NAME: _write(workdir / CONTRACT_NAME, "{}\n", mode=0o600),
    }
    assert set(controls) == planner_sandbox.DEADLINE_CONTROL_NAMES
    return controls


def _launcher_module(seal_task_memory=lambda _workdir: None):
    return SimpleNamespace(
        seal_task_memory=seal_task_memory,
        PLANNER_WRITABLE_DIRECTORIES=(),
        PLANNER_WRITABLE_MAILBOXES=(),
        _create_private_directory=lambda *_args: None,
        _create_fixed_mailbox=lambda *_args: None,
        _publish_mailbox_marker=lambda *_args: None,
    )


def test_isolation_prepare_never_shares_deadline_controls(tmp_path):
    workdir = tmp_path / "workdir"
    workdir.mkdir(mode=0o700)
    controls = _stage_deadline_controls(workdir)
    observation = _write(workdir / "state_00.json", "{}\n", mode=0o600)
    identities = {name: path.stat().st_ino for name, path in controls.items()}

    result = planner_sandbox._prepare_isolated_workdir(
        _launcher_module(),
        workdir,
        os.geteuid(),
        os.getegid(),
        owner_uid=os.geteuid(),
        owner_gid=os.getegid(),
    )

    assert result == (workdir / ".codex", workdir / "tmp")
    for name, path in controls.items():
        metadata = path.stat()
        assert metadata.st_ino == identities[name]
        assert metadata.st_uid == os.geteuid()
        assert metadata.st_gid == os.getegid()
        assert stat.S_IMODE(metadata.st_mode) == 0o600
    assert stat.S_IMODE(observation.stat().st_mode) == 0o640
    descriptor = _open_gate(controls[GATE_NAME], create=False)
    os.close(descriptor)


@pytest.mark.parametrize("tamper", ["replace", "hardlink", "permissions"])
def test_isolation_prepare_rejects_deadline_gate_tampering(tmp_path, tamper):
    workdir = tmp_path / "workdir"
    workdir.mkdir(mode=0o700)
    controls = _stage_deadline_controls(workdir)
    gate = controls[GATE_NAME]

    def tamper_with_control(_workdir):
        if tamper == "replace":
            gate.unlink()
            _write(gate, "", mode=0o600)
        elif tamper == "hardlink":
            os.link(gate, workdir / "gate-alias")
        else:
            gate.chmod(0o660)
        return None

    with pytest.raises(SystemExit, match="deadline control is unsafe"):
        planner_sandbox._prepare_isolated_workdir(
            _launcher_module(tamper_with_control),
            workdir,
            os.geteuid(),
            os.getegid(),
            owner_uid=os.geteuid(),
            owner_gid=os.getegid(),
        )


@pytest.mark.parametrize("tamper", ["missing", "extra"])
def test_isolation_prepare_requires_exact_deadline_control_set(tmp_path, tamper):
    workdir = tmp_path / "workdir"
    workdir.mkdir(mode=0o700)
    controls = _stage_deadline_controls(workdir)
    if tamper == "missing":
        controls[CONTRACT_NAME].unlink()
    else:
        _write(workdir / "_deadline_commit_01.json", "{}\n", mode=0o600)

    with pytest.raises(SystemExit, match="deadline control set differs"):
        planner_sandbox._prepare_isolated_workdir(
            _launcher_module(),
            workdir,
            os.geteuid(),
            os.getegid(),
            owner_uid=os.geteuid(),
            owner_gid=os.getegid(),
        )


def test_deadline_gate_stays_root_only_while_isolation_prepare_is_paused(tmp_path):
    workdir = tmp_path / "workdir"
    workdir.mkdir(mode=0o700)
    controls = _stage_deadline_controls(workdir)
    entered = threading.Event()
    release = threading.Event()
    errors = []

    def pause_prepare(_workdir):
        entered.set()
        assert release.wait(timeout=2)
        return None

    def prepare():
        try:
            planner_sandbox._prepare_isolated_workdir(
                _launcher_module(pause_prepare),
                workdir,
                os.geteuid(),
                os.getegid(),
                owner_uid=os.geteuid(),
                owner_gid=os.getegid(),
            )
        except BaseException as exc:
            errors.append(exc)

    thread = threading.Thread(target=prepare)
    thread.start()
    assert entered.wait(timeout=2)
    try:
        for path in controls.values():
            assert stat.S_IMODE(path.stat().st_mode) == 0o600
        descriptor = _open_gate(controls[GATE_NAME], create=False)
        os.close(descriptor)
    finally:
        release.set()
        thread.join(timeout=2)
    assert not thread.is_alive()
    assert errors == []


def test_secure_script_rejects_tampering_and_group_write(tmp_path):
    script = _write(tmp_path / "script.py", "VALUE = 1\n")
    digest = hashlib.sha256(script.read_bytes()).hexdigest()
    assert secure_script._read_source(script, digest) == "VALUE = 1\n"
    with pytest.raises(SystemExit, match="frozen SHA-256"):
        secure_script._read_source(script, "f" * 64)
    script.chmod(0o620)
    with pytest.raises(SystemExit, match="immutable"):
        secure_script._read_source(script, digest)


def test_doctor_runtime_hash_policy_is_fail_closed(tmp_path, monkeypatch):
    source = _write(tmp_path / "driver.py", "frozen\n")
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    monkeypatch.setitem(executor.FROZEN_RUNTIME_SHA256, "driver", digest)
    assert executor._frozen_runtime_problem("driver", source) is None
    source.write_text("changed\n", encoding="utf-8")
    assert "approved SHA-256" in executor._frozen_runtime_problem("driver", source)


def test_codex_pin_rejects_unapproved_binary(tmp_path):
    binary = _write(tmp_path / "codex", "not-codex\n", mode=0o700)
    _identity, problem = executor._pinned_codex_identity(binary)
    assert "pinned 0.147.0 SHA-256" in problem


def test_managed_codex_config_must_be_empty_or_absent(tmp_path):
    managed = tmp_path / "codex"
    assert executor._managed_codex_config_problem(managed) is None
    managed.mkdir()
    assert executor._managed_codex_config_problem(managed) is None
    _write(managed / "managed_config.toml", "model = 'other'\n")
    assert "must be empty" in executor._managed_codex_config_problem(managed)


def test_archive_records_only_bounded_audit_evidence(tmp_path):
    workdir = tmp_path / "workdir"
    destination = tmp_path / "archive"
    workdir.mkdir()
    _write(workdir / "agent.log", "{}\n")
    _write(workdir / "planner_status.json", "{}\n")
    _write(workdir / "image_cam_00.png", "large fixture")
    destination.mkdir()
    descriptor = os.open(destination, os.O_RDONLY | os.O_DIRECTORY)
    try:
        hashes = executor._archive_workdir(workdir, descriptor)
    finally:
        os.close(descriptor)
    assert set(hashes) == {"agent.log", "planner_status.json"}
    assert all(len(value) == 64 for value in hashes.values())
    assert not (destination / "image_cam_00.png").exists()


def test_only_committed_zero_exit_success_is_an_expected_driver_shutdown(tmp_path):
    _write(tmp_path / "done_01.flag")
    _write(tmp_path / "state_01.json", '{"success": true}\n')
    _write(tmp_path / "log_01.json", "{}\n")
    _write(tmp_path / "command_trace.jsonl", '{"action":"release"}\n')

    assert executor._latched_success_exit(tmp_path, 0) is True
    assert executor._latched_success_exit(tmp_path, 1) is False
    (tmp_path / "log_01.json").unlink()
    assert executor._latched_success_exit(tmp_path, 0) is False


def test_timeout_driver_is_reaped_before_fallback_stop():
    waits = []
    process = SimpleNamespace(
        poll=lambda: None,
        wait=lambda timeout: waits.append(timeout) or -9,
    )

    assert (
        executor._observe_driver_exit(
            process,
            termination="planner_timeout",
            timeout_seconds=15,
        )
        == -9
    )
    assert waits == [15]


def test_non_timeout_driver_is_not_waited_for():
    process = SimpleNamespace(
        poll=lambda: None,
        wait=lambda timeout: pytest.fail(f"unexpected wait({timeout})"),
    )

    assert (
        executor._observe_driver_exit(
            process,
            termination="planner_completed",
            timeout_seconds=15,
        )
        is None
    )


def test_private_directory_policy(tmp_path):
    directory = tmp_path / "private"
    directory.mkdir(mode=0o700)
    assert executor._private_directory_problem(directory, "fixture") is None
    directory.chmod(0o750)
    assert "group or other" in executor._private_directory_problem(directory, "fixture")
    directory.chmod(0o700)
    alias = tmp_path / "alias"
    alias.symlink_to(directory, target_is_directory=True)
    assert "real directory" in executor._private_directory_problem(alias, "fixture")


def test_rollout_root_requires_traverse_only_mode(tmp_path):
    root = tmp_path / "rollouts"
    root.mkdir(mode=0o700)
    assert "0711" in executor._rollout_directory_problem(root)
    root.chmod(0o711)
    assert "ancestor" in executor._rollout_directory_problem(root)


def test_rollout_parents_are_explicitly_traverse_only(tmp_path):
    root = tmp_path / "rollouts"
    root.mkdir(mode=0o711)
    target = root / "atomic" / "OpenDrawer_s01"
    executor._prepare_rollout_parent(target, root)
    assert stat.S_IMODE((root / "atomic").stat().st_mode) == 0o711
    assert stat.S_IMODE(target.stat().st_mode) == 0o711


def test_remove_workdir_is_contained(tmp_path):
    root = tmp_path / "rollouts"
    workdir = root / "task" / "attempt"
    workdir.mkdir(parents=True)
    executor._remove_workdir(workdir, root)
    assert not workdir.exists()
    with pytest.raises(RuntimeError, match="rollout root"):
        executor._remove_workdir(root, root)


def test_planner_command_keeps_generated_shell_off_network(tmp_path):
    config = _config(tmp_path)
    command = executor._planner_command(
        config,
        workdir=config.rollout_root,
        run_id="run-1",
        cell=cell_for("atomic", "OpenDrawer", 1),
    )
    assert "--sandbox" not in command
    assert (
        command[command.index("--launcher-sha256") + 1]
        == (executor.FROZEN_RUNTIME_SHA256["isolation_launcher"])
    )
    assert 'default_permissions="rpent_outer_landlock"' in command
    assert 'permissions.rpent_outer_landlock.filesystem={":root"="write"}' in command
    assert "permissions.rpent_outer_landlock.network.enabled=false" in command
    assert 'web_search="disabled"' in command
    assert "allow_login_shell=false" in command
    assert "analytics.enabled=false" in command
    assert "feedback.enabled=false" in command
    assert "check_for_update_on_startup=false" in command
    assert 'otel.trace_exporter="none"' in command
    assert "tools.experimental_request_user_input.enabled=false" in command
    for feature in (
        "browser_use_full_cdp_access",
        "in_app_browser",
        "image_generation",
        "multi_agent",
        "multi_agent_v2",
    ):
        assert feature in command
    assert "use_legacy_landlock" not in command
    assert "--add-dir" not in command
    assert "danger-full-access" not in command
    rendered = "\n".join(command)
    assert "model_provider=rpent_responses_api" in command
    assert "model_providers.rpent_responses_api.base_url=" in rendered
    assert "model_providers.rpent_responses_api.request_max_retries=12" in command
    assert "model_providers.rpent_responses_api.stream_max_retries=120" in command
    assert (
        "model_providers.rpent_responses_api.stream_idle_timeout_ms=330000" in command
    )
    assert "sorux" not in rendered.lower()


def test_formal_planner_adapter_rejects_unaudited_claude_profile(tmp_path):
    config = replace(_config(tmp_path), planner_profile=get_profile("claude-opus48"))
    assert config.planner_profile.model == "claude-opus-4-8"
    with pytest.raises(RuntimeError, match="declared but not audited"):
        executor._planner_command(
            config,
            workdir=config.rollout_root,
            run_id="run-1",
            cell=cell_for("atomic", "OpenDrawer", 1),
        )


def test_subscription_planner_command_uses_broker_without_oauth_in_cell(tmp_path):
    config = _config(tmp_path)
    capability = _write(tmp_path / "broker-capability", "opaque-capability\n")
    transport = PlannerTransport.chatgpt_subscription(
        credential_file=capability,
        broker_base_url="http://127.0.0.1:4319/v1",
        broker_health_url="http://127.0.0.1:4319/health",
    )
    config = replace(config, planner_transport=transport)
    command = executor._planner_command(
        config,
        workdir=config.rollout_root,
        run_id="run-1",
        cell=cell_for("atomic", "OpenDrawer", 1),
    )
    rendered = "\n".join(command)
    assert "model_provider=rpent_chatgpt_broker" in command
    assert "model_providers.rpent_chatgpt_broker.base_url=" in rendered
    assert "model_providers.rpent_chatgpt_broker.request_max_retries=12" in command
    assert "model_providers.rpent_chatgpt_broker.stream_max_retries=120" in command
    assert (
        "model_providers.rpent_chatgpt_broker.stream_idle_timeout_ms=330000" in command
    )
    assert "http://127.0.0.1:4319/v1" in rendered
    assert "gpt-5.5" in rendered
    assert "xhigh" in rendered
    assert "auth.json" not in rendered
    assert "CODEX_HOME" not in rendered
    assert "openai_chatgpt_subscription" not in rendered
    assert "opaque-capability" not in rendered
    assert 'shell_environment_policy.exclude=["RLDX_PLANNER_API_KEY"]' in command
    assert "permissions.rpent_outer_landlock.network.enabled=false" in command


def test_codex_provider_retry_constants_are_frozen():
    assert CODEX_REQUEST_MAX_RETRIES == 12
    assert CODEX_STREAM_MAX_RETRIES == 120
    assert CODEX_STREAM_IDLE_TIMEOUT_MS == 330_000


def test_external_scripts_run_through_pinned_open_bytes_adapter(tmp_path):
    config = _config(tmp_path)
    command = executor._external_script_command(config, "readiness", ["--probe"])
    assert command[1].endswith("/secure_script.py")
    assert command[command.index("--source") + 1] == str(config.runtime.readiness)
    assert (
        command[command.index("--sha256") + 1]
        == (executor.FROZEN_RUNTIME_SHA256["readiness"])
    )
    assert command[-2:] == ["--", "--probe"]


def test_driver_environment_is_allowlisted_and_has_no_reset_override(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("RLDX_RESET_SEED", "123")
    monkeypatch.setenv("PYTHONPATH", "/untrusted")
    monkeypatch.setenv("LD_PRELOAD", "/untrusted/library.so")
    environment = executor._driver_environment(
        _config(tmp_path), 1, cell_for("atomic", "OpenDrawer", 1)
    )
    assert environment["CUDA_VISIBLE_DEVICES"] == "1"
    assert environment["RLDX_ALLOW_RESET"] == "0"
    assert environment["RLDX_PERCEPTION_ISOLATION"] == "1"
    assert "RLDX_RESET_SEED" not in environment
    assert "PYTHONPATH" not in environment
    assert "LD_PRELOAD" not in environment


def test_driver_environment_derives_seen_reset_seed(tmp_path, monkeypatch):
    monkeypatch.setenv("RLDX_RESET_SEED", "999")
    config = _config(tmp_path)
    seen = executor._driver_environment(
        config, 0, cell_for("composite_seen", "PrepareCoffee", 5)
    )
    unseen = executor._driver_environment(
        config, 0, cell_for("composite_unseen", "ArrangeTea", 5)
    )
    assert seen["RLDX_RESET_SEED"] == "4200005"
    assert "RLDX_RESET_SEED" not in unseen


def test_preliminary_driver_rejects_schema_before_motion_and_after_success():
    class Driver:
        calls = 0

        def dump_state(self, _step, publish_done=True):
            return {"success": False}

        def execute(self, _command):
            self.calls += 1
            return {"ok": True}

    module = SimpleNamespace(RoboCasaDriver=Driver)
    preliminary_driver._patch_driver(module)
    driver = Driver()
    driver.env = SimpleNamespace(
        terminated=False,
        current_raw_obs={"language": "open the drawer"},
        env=SimpleNamespace(get_ep_meta=lambda: {"lang": "open the drawer"}),
    )
    invalid = {"action": "move_to", "xyz": [0, 0, 0], "extra": True}
    assert "invalid command schema" in driver.execute(invalid)["error"]
    assert driver.calls == 0
    driver.env.terminated = True
    assert "invalid command schema" in driver.execute(invalid)["error"]
    with pytest.raises(SystemExit):
        driver.execute({"action": "release"})
    assert driver.calls == 0


def test_preliminary_driver_translates_refused_post_success_step():
    class Environment:
        def reset(self):
            return {}

        def step(self, _action):
            raise AssertionError("post-success physics must not run")

    class Driver:
        def dump_state(self, _step, publish_done=True):
            return {"success": False}

        def execute(self, _command):
            return self.env.step(None)

    module = SimpleNamespace(
        RoboCasaDriver=Driver,
        RoboCasaInteractiveEnv=Environment,
    )
    preliminary_driver._patch_driver(module)
    environment = Environment()
    environment._terminated = True
    environment.terminated = False
    environment.current_raw_obs = {"language": "open the drawer"}
    environment.env = SimpleNamespace(get_ep_meta=lambda: {"lang": "open the drawer"})
    driver = Driver()
    driver.env = environment

    assert driver.execute({"action": "release"}) == {
        "status": "task_success",
        "success": True,
    }
