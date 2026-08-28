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

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from rpent.reproduce.robocasa import executor, provenance
from rpent.reproduce.robocasa.executor import ExecutorConfig, RuntimePaths
from rpent.reproduce.robocasa.planner_transport import (
    CODEX_REQUEST_MAX_RETRIES,
    CODEX_STREAM_IDLE_TIMEOUT_MS,
    CODEX_STREAM_MAX_RETRIES,
    PlannerTransport,
)
from rpent.reproduce.robocasa.profiles import PlannerProfile, get_profile

BASE_URL = "https://planner.example/v1"
PROVENANCE_PATH = "rpent/reproduce/robocasa/provenance.py"


def _write(path: Path, text: str = "fixture\n", *, mode: int = 0o600) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    path.chmod(mode)
    return path


def _runtime(tmp_path: Path) -> RuntimePaths:
    root = tmp_path / "runtime"
    migration = root / "migration"
    robocasa = root / "external_dependencies/robocasa365/robocasa/robocasa"
    robosuite = root / "external_dependencies/robocasa365/robosuite/robosuite"
    navview = robosuite / "models/assets/bases/omron_mobile_base.xml"
    _write(root / "rldx/runtime.py", "RLDX = 1\n")
    _write(robocasa / "__init__.py", "ROBOCASA = 1\n")
    _write(robosuite / "__init__.py", "ROBOSUITE = 1\n")
    _write(migration / "robocasa_interactive_env.py")
    _write(migration / "rldx_skill.py")
    _write(
        navview,
        """<mujoco><worldbody><body name="base">
<camera name="navview" mode="fixed" pos="0.2 0 1.6"
xyaxes="0 -1 0 0.643 0 0.766" fovy="75"/>
</body></worldbody></mujoco>
""",
    )
    tools = root / "tools"
    model = root / "checkpoint"
    vlm = root / "vlm"
    model.mkdir(parents=True)
    vlm.mkdir()
    return RuntimePaths(
        root=root,
        sim_python=_write(tools / "python", mode=0o700),
        driver=_write(migration / "driver.py"),
        readiness=_write(migration / "ready.py"),
        deadline=_write(migration / "deadline.py"),
        isolation_launcher=_write(migration / "isolate.py"),
        artifact_builder=_write(migration / "artifacts.py"),
        model=model,
        vlm_metadata=vlm,
        navview_xml=navview,
    )


def _config(tmp_path: Path) -> ExecutorConfig:
    results = tmp_path / "results"
    rollouts = tmp_path / "rollouts"
    memory = tmp_path / "memory"
    results.mkdir(mode=0o700)
    rollouts.mkdir(mode=0o700)
    memory.mkdir(mode=0o700)
    _write(memory / "manifest.json", '{"memory": 1}\n')
    credential = _write(tmp_path / "secret-a", "provenance-test-secret-a\n")
    return ExecutorConfig(
        runtime=_runtime(tmp_path),
        results_root=results,
        rollout_root=rollouts,
        memory_root=memory,
        planner_profile=get_profile("codex-gpt55-xhigh"),
        planner_transport=PlannerTransport.api_key(
            credential_file=credential,
            base_url=BASE_URL,
        ),
        codex_bin=_write(tmp_path / "codex", mode=0o700),
        preliminary_local_runtime=True,
        memory_source="local-migration",
        memory_revision="revision-a",
    )


def _attestation() -> dict[str, str]:
    return {
        "checkpoint_id": "RLDX-1-FT-RC365",
        "authority_manifest_sha256": "a" * 64,
        "fingerprint": "b" * 64,
    }


def _isolation_attestation(
    transport: PlannerTransport | None = None,
) -> dict[str, Any]:
    if transport is None:
        transport = PlannerTransport.api_key(
            credential_file=Path("unused-api-key"),
            base_url=BASE_URL,
        )
    value: dict[str, Any] = {
        "schema_version": 2,
        "passed": True,
        "codex": {
            "version": "codex-cli 0.147.0",
            "sha256": (
                "cb0a15567e9a60a5820d54b0f6ae86d504dc3805c1eab21a47f70e3eb7b73a40"
            ),
        },
        "launcher_sha256": (
            "6f0173f39753a8268b0b73aac2b729066f9798a0679c07e705a7c22f45cf81cb"
        ),
        "sandbox_adapter_sha256": provenance.sha256_file(
            Path(provenance.__file__).with_name("sandbox.py")
        ),
        "profile": {
            "permission_profile": "rpent_outer_landlock",
            "filesystem_authority": (
                "outer_rldx_landlock_abi1_uid_gid_fixed_mailboxes"
            ),
            "codex_filesystem_view": "full_write_fast_path_no_inner_fs_sandbox",
            "command_network_authority": ("codex_0.147_restricted_network_seccomp"),
            "arguments_sha256": "4" * 64,
        },
        "planner_transport": transport.manifest_identity(),
        "kernel_release": "fixture-kernel",
        "landlock_abi": 1,
        "tool_names": [
            "apply_patch",
            "exec_command",
            "update_plan",
            "view_image",
            "write_stdin",
        ],
        "tool_schema_sha256": "5" * 64,
        "observed_item_types": [
            "agent_message",
            "command_execution",
            "file_change",
        ],
        "checks": {
            "scratch_write": True,
            "root_create_blocked": True,
            "rpent_read_blocked": True,
            "runtime_read_blocked": True,
            "proc_environ_read_blocked": True,
            "shell_network_blocked": True,
            "planner_secret_absent": True,
            "usr_write_blocked": True,
            "apply_patch_outside_blocked": True,
            "view_image_outside_blocked": True,
            "exec_escalation_rejected": True,
            "deadline_controls_root_only": True,
        },
    }
    value["attestation_sha256"] = provenance.canonical_sha256(value)
    return value


def _ensure_isolation_attestation(config: ExecutorConfig) -> None:
    directory = config.results_root / "_preflight"
    directory.mkdir(mode=0o700, exist_ok=True)
    path = directory / provenance.ISOLATION_ATTESTATION_NAME
    if path.exists():
        return
    _write(
        path,
        json.dumps(_isolation_attestation(config.planner_transport)) + "\n",
        mode=0o600,
    )


@pytest.fixture
def frozen_implementation(monkeypatch):
    monkeypatch.setattr(
        provenance,
        "_implementation_hashes",
        lambda: {PROVENANCE_PATH: "c" * 64},
    )


def _ensure(
    config: ExecutorConfig,
    attestation: dict[str, str] | None = None,
) -> dict[str, Any]:
    _ensure_isolation_attestation(config)
    return provenance.ensure_run_manifest(
        config,
        attestation or _attestation(),
    )


def test_manifest_creation_is_idempotent(tmp_path, frozen_implementation):
    config = _config(tmp_path)

    first = _ensure(config)
    path = config.results_root / provenance.RUN_MANIFEST_NAME
    first_bytes = path.read_bytes()
    first_stat = path.stat()
    second = _ensure(config)

    assert first["schema_version"] == 3
    assert second == first
    assert path.read_bytes() == first_bytes
    assert path.stat().st_ino == first_stat.st_ino
    assert path.stat().st_mtime_ns == first_stat.st_mtime_ns
    assert path.stat().st_mode & 0o777 == 0o600


def test_nonempty_legacy_results_root_cannot_be_claimed(
    tmp_path, frozen_implementation
):
    config = _config(tmp_path)
    _write(config.results_root / "legacy-result.json", "{}\n")

    with pytest.raises(ValueError, match="non-empty results root"):
        _ensure(config)


def test_checkpoint_preflight_can_precede_run_manifest(tmp_path, frozen_implementation):
    config = _config(tmp_path)
    preflight = config.results_root / "_preflight"
    preflight.mkdir(mode=0o700)
    _write(preflight / "checkpoint-attestation.json", "{}\n", mode=0o600)

    manifest = _ensure(config)

    assert manifest["run_config_sha256"]


def test_runner_lock_and_preflight_can_precede_run_manifest(
    tmp_path, frozen_implementation
):
    config = _config(tmp_path)
    _write(config.results_root / ".rpent-run.lock", "", mode=0o600)
    preflight = config.results_root / "_preflight"
    preflight.mkdir(mode=0o700)
    _write(preflight / "checkpoint-attestation.json", "{}\n", mode=0o600)

    manifest = _ensure(config)

    assert manifest["run_config_sha256"]


@pytest.mark.parametrize("extra_name", ["other.json", "nested"])
def test_checkpoint_preflight_rejects_extra_entries(
    tmp_path, frozen_implementation, extra_name
):
    config = _config(tmp_path)
    preflight = config.results_root / "_preflight"
    preflight.mkdir(mode=0o700)
    _write(preflight / "checkpoint-attestation.json", "{}\n", mode=0o600)
    extra = preflight / extra_name
    if extra_name == "nested":
        extra.mkdir()
    else:
        _write(extra, "{}\n")

    with pytest.raises(ValueError, match="non-empty results root"):
        _ensure(config)


@pytest.mark.parametrize("tamper", ["digest", "config", "extra_field"])
def test_tampered_or_extended_manifest_fails_closed(
    tmp_path, frozen_implementation, tamper
):
    config = _config(tmp_path)
    _ensure(config)
    path = config.results_root / provenance.RUN_MANIFEST_NAME
    value = json.loads(path.read_text(encoding="utf-8"))
    if tamper == "digest":
        value["run_config_sha256"] = "0" * 64
    elif tamper == "config":
        value["config"]["memory"]["source"] = "tampered"
    else:
        value["unfrozen_extension"] = True
    path.write_text(json.dumps(value) + "\n", encoding="utf-8")

    with pytest.raises(ValueError):
        _ensure(config)


def test_symlink_manifest_fails_closed(tmp_path, frozen_implementation):
    config = _config(tmp_path)
    _ensure(config)
    second_root = tmp_path / "results-via-symlink"
    second_root.mkdir(mode=0o700)
    (second_root / provenance.RUN_MANIFEST_NAME).symlink_to(
        config.results_root / provenance.RUN_MANIFEST_NAME
    )

    with pytest.raises(ValueError, match="non-symlink"):
        _ensure(replace(config, results_root=second_root))


def test_manifest_rejects_self_signed_formal_or_incomplete_configuration():
    formal = provenance.make_run_manifest(
        {
            "protocol_id": provenance.PROTOCOL_ID,
            "runtime_kind": "formal_rpent_release",
            "preliminary": False,
        }
    )

    _, problem = provenance.validate_run_manifest_value(formal)

    assert problem is not None


def test_run_manifest_freezes_codex_provider_retry_policy(tmp_path):
    configuration = provenance.build_run_configuration(
        _config(tmp_path),
        _attestation(),
        isolation_attestation=_isolation_attestation(),
    )

    assert configuration["planner"]["provider_retry_policy"] == {
        "request_max_retries": 12,
        "stream_max_retries": 120,
        "stream_idle_timeout_ms": 330_000,
    }
    assert CODEX_REQUEST_MAX_RETRIES == 12
    assert CODEX_STREAM_MAX_RETRIES == 120
    assert CODEX_STREAM_IDLE_TIMEOUT_MS == 330_000

    validated, problem = provenance.validate_run_manifest_value(
        provenance.make_run_manifest(configuration)
    )
    assert problem is None
    assert validated is not None


@pytest.mark.parametrize(
    "policy",
    [
        {},
        {
            "request_max_retries": 11,
            "stream_max_retries": 120,
            "stream_idle_timeout_ms": 330_000,
        },
        {
            "request_max_retries": 12,
            "stream_max_retries": 119,
            "stream_idle_timeout_ms": 330_000,
        },
        {
            "request_max_retries": 12,
            "stream_max_retries": 120,
            "stream_idle_timeout_ms": 329_999,
        },
        {
            "request_max_retries": 12,
            "stream_max_retries": 120,
            "stream_idle_timeout_ms": 330_000,
            "unfrozen_extension": True,
        },
    ],
)
def test_self_signed_manifest_cannot_change_provider_retry_policy(tmp_path, policy):
    configuration = provenance.build_run_configuration(
        _config(tmp_path),
        _attestation(),
        isolation_attestation=_isolation_attestation(),
    )
    configuration["planner"]["provider_retry_policy"] = policy

    validated, problem = provenance.validate_run_manifest_value(
        provenance.make_run_manifest(configuration)
    )

    assert validated is None
    assert problem is not None


def test_execution_rechecks_runtime_identity_for_every_cell(
    tmp_path, frozen_implementation
):
    config = _config(tmp_path)
    _ensure_isolation_attestation(config)
    executor.ensure_execution_run_manifest(config, _attestation())
    _write(config.runtime.driver, "changed between cells\n")

    with pytest.raises(ValueError, match="different RoboCasa run configuration"):
        executor.ensure_execution_run_manifest(config, _attestation())


@pytest.mark.parametrize(
    "change",
    [
        "memory_manifest",
        "memory_source",
        "memory_revision",
        "planner_effort",
        "planner_endpoint",
        "planner_auth_mode",
        "runtime_script",
        "navview",
        "startup_timeout",
        "kill_timeout",
        "checkpoint_fingerprint",
        "isolation_attestation",
    ],
)
def test_scientific_configuration_change_cannot_reuse_results_root(
    tmp_path, frozen_implementation, change
):
    config = _config(tmp_path)
    attestation = _attestation()
    _ensure(config, attestation)

    if change == "memory_manifest":
        _write(config.memory_root / "manifest.json", '{"memory": 2}\n')
    elif change == "memory_source":
        config = replace(config, memory_source="huggingface")
    elif change == "memory_revision":
        config = replace(config, memory_revision="revision-b")
    elif change == "planner_effort":
        profile = replace(config.planner_profile, reasoning_effort="high")
        config = replace(config, planner_profile=profile)
    elif change == "planner_endpoint":
        config = replace(
            config,
            planner_transport=PlannerTransport.api_key(
                credential_file=config.planner_transport.credential_file,
                base_url="https://other-planner.example/v1",
            ),
        )
    elif change == "planner_auth_mode":
        config = replace(
            config,
            planner_transport=PlannerTransport.chatgpt_subscription(
                credential_file=config.planner_transport.credential_file,
                broker_base_url="http://127.0.0.1:8765/v1",
                broker_health_url="http://127.0.0.1:8765/health",
            ),
        )
    elif change == "runtime_script":
        _write(config.runtime.driver, "changed driver\n")
    elif change == "navview":
        _write(config.runtime.navview_xml, "<mujoco changed='true'/>\n")
    elif change == "startup_timeout":
        config = replace(
            config,
            startup_timeout_seconds=config.startup_timeout_seconds + 1,
        )
    elif change == "kill_timeout":
        config = replace(config, kill_after_seconds=config.kill_after_seconds + 1)
    elif change == "isolation_attestation":
        path = (
            config.results_root / "_preflight" / provenance.ISOLATION_ATTESTATION_NAME
        )
        value = json.loads(path.read_text(encoding="utf-8"))
        value["kernel_release"] = "different-kernel"
        payload = {
            key: item for key, item in value.items() if key != "attestation_sha256"
        }
        value["attestation_sha256"] = provenance.canonical_sha256(payload)
        _write(path, json.dumps(value) + "\n")
    else:
        attestation = {**attestation, "fingerprint": "d" * 64}

    with pytest.raises(ValueError, match="different RoboCasa run configuration"):
        if change in {"planner_endpoint", "planner_auth_mode"}:
            path = (
                config.results_root
                / "_preflight"
                / provenance.ISOLATION_ATTESTATION_NAME
            )
            _write(
                path,
                json.dumps(_isolation_attestation(config.planner_transport)) + "\n",
            )
        _ensure(config, attestation)


def test_isolation_attestation_tampering_fails_before_manifest(tmp_path):
    config = _config(tmp_path)
    _ensure_isolation_attestation(config)
    path = config.results_root / "_preflight" / provenance.ISOLATION_ATTESTATION_NAME
    value = json.loads(path.read_text(encoding="utf-8"))
    value["passed"] = False
    _write(path, json.dumps(value) + "\n")

    with pytest.raises(ValueError, match="digest or pass state"):
        provenance.ensure_run_manifest(
            config,
            _attestation(),
        )


def test_implementation_identity_is_nonempty_and_covers_provenance(tmp_path):
    configuration = provenance.build_run_configuration(
        _config(tmp_path),
        _attestation(),
        isolation_attestation=_isolation_attestation(),
    )

    implementation = configuration["implementation_sha256"]
    assert implementation
    assert PROVENANCE_PATH in implementation
    assert len(implementation[PROVENANCE_PATH]) == 64
    assert int(implementation[PROVENANCE_PATH], 16) >= 0


def _keys(value: Any) -> set[str]:
    if isinstance(value, dict):
        return set(value).union(*(_keys(item) for item in value.values()), set())
    if isinstance(value, list):
        return set().union(*(_keys(item) for item in value), set())
    return set()


def test_operational_inputs_do_not_change_configuration_identity(
    tmp_path, frozen_implementation
):
    config = _config(tmp_path)
    first = provenance.build_run_configuration(
        config,
        _attestation(),
        isolation_attestation=_isolation_attestation(config.planner_transport),
    )
    other_secret = _write(tmp_path / "secret-b", "provenance-test-secret-b\n")
    operationally_changed = replace(
        config,
        planner_transport=replace(
            config.planner_transport,
            credential_file=other_secret,
        ),
        results_root=tmp_path / "other-results",
        rollout_root=tmp_path / "other-rollouts",
        keep_workdirs=not config.keep_workdirs,
    )
    second = provenance.build_run_configuration(
        operationally_changed,
        _attestation(),
        isolation_attestation=_isolation_attestation(
            operationally_changed.planner_transport
        ),
    )

    assert second == first
    serialized = json.dumps(first, sort_keys=True)
    assert str(config.planner_transport.credential_file) not in serialized
    assert str(other_secret) not in serialized
    assert "provenance-test-secret-a" not in serialized
    assert _keys(first).isdisjoint(
        {
            "api_key_file",
            "credential_file",
            "gpu",
            "gpus",
            "selection",
            "max_attempts",
            "retry_backoff_seconds",
            "keep_workdirs",
            "results_root",
            "rollout_root",
        }
    )


def test_scheduler_controls_are_not_executor_configuration_fields():
    fields = set(ExecutorConfig.__dataclass_fields__)

    assert fields.isdisjoint(
        {"gpu", "gpus", "selection", "max_attempts", "retry_backoff_seconds"}
    )
    assert set(PlannerProfile.__dataclass_fields__) == {
        "name",
        "backend",
        "model",
        "reasoning_effort",
    }
