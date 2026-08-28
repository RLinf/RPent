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

"""Immutable run identity shared by execution, resume, and validation."""

from __future__ import annotations

import hashlib
import json
import os
import stat
import subprocess
from pathlib import Path
from typing import Any

from .planner_transport import (
    API_KEY_AUTH,
    CHATGPT_BROKER_PROTOCOL,
    CHATGPT_ENDPOINT_IDENTITY,
    CHATGPT_SUBSCRIPTION_AUTH,
    codex_provider_retry_policy,
)
from .protocol import PROTOCOL_ID

RUN_MANIFEST_NAME = "_run_manifest.json"
RUN_MANIFEST_SCHEMA_VERSION = 3
ISOLATION_ATTESTATION_NAME = "isolation-attestation.json"
RUN_CONFIGURATION_FIELDS = frozenset(
    {
        "protocol_id",
        "runtime_kind",
        "preliminary",
        "planner",
        "memory",
        "checkpoint",
        "runtime",
        "implementation_sha256",
    }
)


def sha256_file(path: Path) -> str:
    path = Path(path).resolve(strict=True)
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=True,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def _implementation_hashes() -> dict[str, str]:
    repo = Path(__file__).resolve().parents[3]
    paths = {
        *repo.glob("rpent/reproduce/robocasa/*.py"),
        *repo.glob("robots/robocasa/**/*.py"),
        *repo.glob("robots/robocasa/**/*.json"),
        *repo.glob("robots/robocasa/**/*.patch"),
    }
    return {
        path.relative_to(repo).as_posix(): sha256_file(path)
        for path in sorted(paths)
        if path.is_file() and not path.is_symlink()
    }


def _tree_identity(root: Path, suffixes: frozenset[str]) -> dict[str, Any]:
    if not root.is_dir():
        raise ValueError(f"required runtime source tree is missing: {root}")
    files = {
        path.relative_to(root).as_posix(): sha256_file(path)
        for path in sorted(root.rglob("*"))
        if path.is_file()
        and not path.is_symlink()
        and path.suffix in suffixes
        and ".venv" not in path.parts
        and "__pycache__" not in path.parts
    }
    return {
        "root": str(root.resolve()),
        "file_count": len(files),
        "tree_sha256": canonical_sha256(files),
    }


def _git_identity(root: Path) -> dict[str, Any]:
    try:
        commit = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        diff = subprocess.run(
            ["git", "-C", str(root), "diff", "--binary", "HEAD", "--"],
            check=True,
            capture_output=True,
        ).stdout
        status = subprocess.run(
            ["git", "-C", str(root), "status", "--porcelain=v1"],
            check=True,
            capture_output=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        return {"root": str(root), "available": False}
    return {
        "root": str(root.resolve()),
        "available": True,
        "commit": commit,
        "tracked_diff_sha256": hashlib.sha256(diff).hexdigest(),
        "status_sha256": hashlib.sha256(status).hexdigest(),
    }


def _codex_identity(path: Path) -> dict[str, Any]:
    entrypoint = Path(path).resolve(strict=True)
    files = {"entrypoint": sha256_file(entrypoint)}
    package_root = entrypoint.parent.parent if entrypoint.parent.name == "bin" else None
    if package_root is not None:
        package_json = package_root / "package.json"
        if package_json.is_file() and not package_json.is_symlink():
            files["package.json"] = sha256_file(package_json)
        native = sorted(
            package_root.glob("node_modules/@openai/codex-*/vendor/*/bin/codex")
        )
        if entrypoint.name == "codex.js" and len(native) != 1:
            raise ValueError(
                "Codex JS entrypoint must resolve to exactly one native binary"
            )
        for binary in native:
            if binary.is_file() and not binary.is_symlink():
                files[binary.relative_to(package_root).as_posix()] = sha256_file(binary)
    return {
        "entrypoint": str(entrypoint),
        "files_sha256": files,
    }


def build_run_configuration(
    config: Any,
    checkpoint_attestation: dict[str, Any],
    *,
    isolation_attestation: dict[str, Any],
) -> dict[str, Any]:
    """Build the secret-free scientific identity for one results root."""
    runtime = config.runtime
    runtime_inputs = {
        "sim_python": runtime.sim_python,
        "driver": runtime.driver,
        "readiness": runtime.readiness,
        "deadline": runtime.deadline,
        "isolation_launcher": runtime.isolation_launcher,
        "artifact_builder": runtime.artifact_builder,
        "interactive_env": runtime.root / "migration/robocasa_interactive_env.py",
        "rldx_skill": runtime.root / "migration/rldx_skill.py",
        "preliminary_driver_adapter": Path(__file__).with_name("preliminary_driver.py"),
        "deadline_supervisor_adapter": Path(__file__).with_name(
            "deadline_supervisor.py"
        ),
        "secure_script_adapter": Path(__file__).with_name("secure_script.py"),
        "navview_xml": runtime.navview_xml,
    }
    profile = config.planner_profile
    transport = config.planner_transport
    return {
        "protocol_id": PROTOCOL_ID,
        "runtime_kind": "preliminary_external_snapshot",
        "preliminary": bool(config.preliminary_local_runtime),
        "planner": {
            "profile": profile.name,
            "backend": profile.backend,
            "model": profile.model,
            "reasoning_effort": profile.reasoning_effort,
            **transport.manifest_identity(),
            "wire_api": "responses",
            "provider_retry_policy": codex_provider_retry_policy(),
            "credential_boundary": (
                "root_broker_oauth_outside_cell"
                if transport.uses_broker
                else "cell_codex_only_generated_shell_excluded"
            ),
            "network_policy": {
                "generated_shell_network": False,
                "web_search": False,
                "direct_tool_network": False,
            },
            "isolation": {
                "filesystem_authority": (
                    "outer_rldx_landlock_abi1_uid_gid_fixed_mailboxes"
                ),
                "codex_filesystem_view": "full_write_fast_path_no_inner_fs_sandbox",
                "command_network_authority": ("codex_0.147_restricted_network_seccomp"),
                "permission_profile": "rpent_outer_landlock",
                "managed_config": "etc_codex_empty_or_absent",
                "external_source_loading": "nofollow_fstat_sha256_open_bytes",
            },
            "codex": _codex_identity(config.codex_bin),
            "isolation_preflight": isolation_attestation,
        },
        "memory": {
            "source": config.memory_source,
            "revision": config.memory_revision,
            "manifest_sha256": sha256_file(config.memory_root / "manifest.json"),
            "global_memory": False,
            "task_notes": False,
        },
        "checkpoint": {
            "checkpoint_id": checkpoint_attestation["checkpoint_id"],
            "authority_manifest_sha256": checkpoint_attestation[
                "authority_manifest_sha256"
            ],
            "fingerprint": checkpoint_attestation["fingerprint"],
        },
        "runtime": {
            "root": str(runtime.root.resolve()),
            "inputs_sha256": {
                name: sha256_file(path) for name, path in runtime_inputs.items()
            },
            "navview": {
                "camera": "mobilebase0_navview",
                "mode": "fixed",
                "pos": "0.2 0 1.6",
                "xyaxes": "0 -1 0 0.643 0 0.766",
                "fovy": "75",
            },
            "startup_timeout_seconds": config.startup_timeout_seconds,
            "kill_after_seconds": config.kill_after_seconds,
            "driver_policy": {
                "perception_isolation": True,
                "reset_enabled": False,
                "initial_reset_seed": {
                    "atomic": None,
                    "composite_seen": "4200000 + protocol seed",
                    "composite_unseen": None,
                },
                "max_chunks": 40,
                "settle_patience": 999,
                "attention_implementation": "sdpa",
            },
            "source_trees": {
                "rldx": _tree_identity(
                    runtime.root / "rldx",
                    frozenset({".py", ".json", ".yaml", ".yml", ".jinja"}),
                ),
                "robocasa_python": _tree_identity(
                    runtime.root / "external_dependencies/robocasa365/robocasa",
                    frozenset({".py"}),
                ),
                "robosuite_python": _tree_identity(
                    runtime.root
                    / "external_dependencies/robocasa365/robosuite/robosuite",
                    frozenset({".py"}),
                ),
            },
            "source_git": {
                "robocasa": _git_identity(
                    runtime.root / "external_dependencies/robocasa365/robocasa"
                ),
                "robosuite": _git_identity(
                    runtime.root / "external_dependencies/robocasa365/robosuite"
                ),
            },
        },
        "implementation_sha256": _implementation_hashes(),
    }


def make_run_manifest(configuration: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": RUN_MANIFEST_SCHEMA_VERSION,
        "protocol_id": PROTOCOL_ID,
        "run_config_sha256": canonical_sha256(configuration),
        "config": configuration,
    }


def validate_run_manifest_value(value: Any) -> tuple[dict[str, Any] | None, str | None]:
    if not isinstance(value, dict) or set(value) != {
        "schema_version",
        "protocol_id",
        "run_config_sha256",
        "config",
    }:
        return None, "run manifest must contain exactly the frozen fields"
    if value.get("schema_version") != RUN_MANIFEST_SCHEMA_VERSION:
        return None, "unsupported run manifest schema_version"
    if value.get("protocol_id") != PROTOCOL_ID:
        return None, "run manifest protocol_id mismatch"
    configuration = value.get("config")
    if not isinstance(configuration, dict):
        return None, "run manifest config must be an object"
    if set(configuration) != RUN_CONFIGURATION_FIELDS:
        return None, "run manifest config must contain exactly the preliminary schema"
    if configuration.get("protocol_id") != PROTOCOL_ID:
        return None, "run manifest config protocol_id mismatch"
    if configuration.get("runtime_kind") != "preliminary_external_snapshot":
        return None, "only the preliminary external runtime is currently supported"
    if configuration.get("preliminary") is not True:
        return None, "formal publication runtime is not implemented"
    nested_fields = {
        "planner": {
            "profile",
            "backend",
            "model",
            "reasoning_effort",
            "auth_mode",
            "provider",
            "endpoint_identity",
            "credential_broker",
            "credential_broker_protocol",
            "wire_api",
            "provider_retry_policy",
            "credential_boundary",
            "network_policy",
            "isolation",
            "codex",
            "isolation_preflight",
        },
        "memory": {
            "source",
            "revision",
            "manifest_sha256",
            "global_memory",
            "task_notes",
        },
        "checkpoint": {
            "checkpoint_id",
            "authority_manifest_sha256",
            "fingerprint",
        },
        "runtime": {
            "root",
            "inputs_sha256",
            "navview",
            "startup_timeout_seconds",
            "kill_after_seconds",
            "driver_policy",
            "source_trees",
            "source_git",
        },
    }
    for name, expected in nested_fields.items():
        item = configuration.get(name)
        if not isinstance(item, dict) or set(item) != expected:
            return None, f"run manifest config.{name} does not match the frozen schema"
    if (
        configuration["planner"].get("provider_retry_policy")
        != codex_provider_retry_policy()
    ):
        return None, "run manifest planner retry policy differs"
    implementation = configuration.get("implementation_sha256")
    if (
        not isinstance(implementation, dict)
        or not implementation
        or any(
            not isinstance(path, str)
            or not path
            or not isinstance(file_digest, str)
            or len(file_digest) != 64
            or any(character not in "0123456789abcdef" for character in file_digest)
            for path, file_digest in implementation.items()
        )
    ):
        return None, "run manifest implementation identity is invalid"
    digest = value.get("run_config_sha256")
    if not isinstance(digest, str) or digest != canonical_sha256(configuration):
        return None, "run manifest config digest mismatch"
    return value, None


def load_run_manifest(root: Path) -> tuple[dict[str, Any] | None, str | None]:
    path = Path(root) / RUN_MANIFEST_NAME
    try:
        metadata = path.lstat()
    except FileNotFoundError:
        return None, "missing run manifest"
    except OSError as exc:
        return None, f"cannot inspect run manifest: {exc}"
    if path.is_symlink() or not stat.S_ISREG(metadata.st_mode):
        return None, "run manifest must be a regular non-symlink file"
    if metadata.st_uid != os.geteuid() or stat.S_IMODE(metadata.st_mode) != 0o600:
        return None, "run manifest must be owned by the run user with mode 0600"
    try:
        value = json.loads(path.read_text(encoding="utf-8", errors="strict"))
    except (OSError, UnicodeError, ValueError) as exc:
        return None, f"invalid run manifest JSON: {exc}"
    return validate_run_manifest_value(value)


def _exclusive_write(path: Path, value: dict[str, Any]) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags, 0o600)
    try:
        os.fchmod(descriptor, 0o600)
        payload = (
            json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
        ).encode("utf-8")
        os.write(descriptor, payload)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _load_isolation_attestation(root: Path) -> dict[str, Any]:
    path = root / "_preflight" / ISOLATION_ATTESTATION_NAME
    try:
        metadata = path.lstat()
    except OSError as exc:
        raise ValueError(f"missing or unreadable isolation attestation: {exc}") from exc
    if (
        path.is_symlink()
        or not stat.S_ISREG(metadata.st_mode)
        or metadata.st_uid != os.geteuid()
        or stat.S_IMODE(metadata.st_mode) != 0o600
    ):
        raise ValueError("isolation attestation must be owned, regular, and mode 0600")
    try:
        value = json.loads(path.read_text(encoding="utf-8", errors="strict"))
    except (OSError, UnicodeError, ValueError) as exc:
        raise ValueError(f"invalid isolation attestation JSON: {exc}") from exc
    required = {
        "schema_version",
        "passed",
        "codex",
        "launcher_sha256",
        "sandbox_adapter_sha256",
        "profile",
        "planner_transport",
        "kernel_release",
        "landlock_abi",
        "tool_names",
        "tool_schema_sha256",
        "observed_item_types",
        "checks",
        "attestation_sha256",
    }
    if not isinstance(value, dict) or set(value) != required:
        raise ValueError("isolation attestation fields differ from the frozen schema")
    digest = value.get("attestation_sha256")
    payload = {key: item for key, item in value.items() if key != "attestation_sha256"}
    if (
        value.get("schema_version") != 2
        or value.get("passed") is not True
        or not isinstance(digest, str)
        or digest != canonical_sha256(payload)
    ):
        raise ValueError("isolation attestation digest or pass state is invalid")

    def is_sha256(item: Any) -> bool:
        return (
            isinstance(item, str)
            and len(item) == 64
            and all(character in "0123456789abcdef" for character in item)
        )

    codex = value.get("codex")
    profile = value.get("profile")
    planner_transport = value.get("planner_transport")
    expected_profile = {
        "permission_profile": "rpent_outer_landlock",
        "filesystem_authority": ("outer_rldx_landlock_abi1_uid_gid_fixed_mailboxes"),
        "codex_filesystem_view": "full_write_fast_path_no_inner_fs_sandbox",
        "command_network_authority": "codex_0.147_restricted_network_seccomp",
    }
    expected_checks = {
        "scratch_write",
        "root_create_blocked",
        "rpent_read_blocked",
        "runtime_read_blocked",
        "proc_environ_read_blocked",
        "shell_network_blocked",
        "planner_secret_absent",
        "usr_write_blocked",
        "apply_patch_outside_blocked",
        "view_image_outside_blocked",
        "exec_escalation_rejected",
        "deadline_controls_root_only",
    }
    if codex != {
        "version": "codex-cli 0.147.0",
        "sha256": ("cb0a15567e9a60a5820d54b0f6ae86d504dc3805c1eab21a47f70e3eb7b73a40"),
    }:
        raise ValueError("isolation attestation Codex identity is not approved")
    if value.get("launcher_sha256") != (
        "6f0173f39753a8268b0b73aac2b729066f9798a0679c07e705a7c22f45cf81cb"
    ):
        raise ValueError("isolation attestation launcher identity is not approved")
    if value.get("sandbox_adapter_sha256") != sha256_file(
        Path(__file__).with_name("sandbox.py")
    ):
        raise ValueError("isolation attestation sandbox adapter identity differs")
    if (
        not isinstance(profile, dict)
        or set(profile) != {*expected_profile, "arguments_sha256"}
        or any(profile.get(key) != item for key, item in expected_profile.items())
        or not is_sha256(profile.get("arguments_sha256"))
    ):
        raise ValueError("isolation attestation profile identity differs")
    if not isinstance(planner_transport, dict) or set(planner_transport) != {
        "auth_mode",
        "provider",
        "endpoint_identity",
        "credential_broker",
        "credential_broker_protocol",
    }:
        raise ValueError("isolation attestation planner transport is malformed")
    auth_mode = planner_transport.get("auth_mode")
    if auth_mode == API_KEY_AUTH:
        valid_transport = (
            planner_transport.get("provider") == "rpent_responses_api"
            and isinstance(planner_transport.get("endpoint_identity"), str)
            and planner_transport["endpoint_identity"].startswith("responses_api:")
            and planner_transport.get("credential_broker") is False
            and planner_transport.get("credential_broker_protocol") is None
        )
    elif auth_mode == CHATGPT_SUBSCRIPTION_AUTH:
        valid_transport = planner_transport == {
            "auth_mode": CHATGPT_SUBSCRIPTION_AUTH,
            "provider": "rpent_chatgpt_broker",
            "endpoint_identity": CHATGPT_ENDPOINT_IDENTITY,
            "credential_broker": True,
            "credential_broker_protocol": CHATGPT_BROKER_PROTOCOL,
        }
    else:
        valid_transport = False
    if not valid_transport:
        raise ValueError("isolation attestation planner transport is not approved")
    if (
        not isinstance(value.get("kernel_release"), str)
        or not value["kernel_release"]
        or type(value.get("landlock_abi")) is not int
        or value["landlock_abi"] < 1
    ):
        raise ValueError("isolation attestation kernel identity is invalid")
    if value.get("tool_names") != [
        "apply_patch",
        "exec_command",
        "update_plan",
        "view_image",
        "write_stdin",
    ] or not is_sha256(value.get("tool_schema_sha256")):
        raise ValueError("isolation attestation capability identity differs")
    observed = value.get("observed_item_types")
    if observed != ["agent_message", "command_execution", "file_change"]:
        raise ValueError("isolation attestation item-event identity is invalid")
    checks = value.get("checks")
    if (
        not isinstance(checks, dict)
        or set(checks) != expected_checks
        or any(item is not True for item in checks.values())
    ):
        raise ValueError("isolation attestation checks are incomplete")
    return value


def _only_approved_preflight(root: Path, entries: list[Path]) -> bool:
    by_name = {entry.name: entry for entry in entries if entry.parent == root}
    if (
        len(by_name) != len(entries)
        or not by_name
        or not set(by_name)
        <= {
            "_preflight",
            ".rpent-run.lock",
        }
    ):
        return False
    lock = by_name.get(".rpent-run.lock")
    if lock is not None:
        try:
            lock_metadata = lock.lstat()
        except OSError:
            return False
        if (
            lock.is_symlink()
            or not stat.S_ISREG(lock_metadata.st_mode)
            or lock_metadata.st_uid != os.geteuid()
            or stat.S_IMODE(lock_metadata.st_mode) != 0o600
            or lock_metadata.st_nlink != 1
        ):
            return False
    directory = by_name.get("_preflight")
    if directory is None:
        return lock is not None
    try:
        metadata = directory.lstat()
    except OSError:
        return False
    if (
        directory.is_symlink()
        or not stat.S_ISDIR(metadata.st_mode)
        or metadata.st_uid != os.geteuid()
        or stat.S_IMODE(metadata.st_mode) != 0o700
    ):
        return False
    children = list(directory.iterdir())
    names = {child.name for child in children}
    if len(names) != len(children) or names not in (
        {ISOLATION_ATTESTATION_NAME},
        {"checkpoint-attestation.json", ISOLATION_ATTESTATION_NAME},
    ):
        return False
    for attestation in children:
        try:
            metadata = attestation.lstat()
        except OSError:
            return False
        if (
            attestation.is_symlink()
            or not stat.S_ISREG(metadata.st_mode)
            or metadata.st_uid != os.geteuid()
            or stat.S_IMODE(metadata.st_mode) != 0o600
        ):
            return False
    return True


def ensure_run_manifest(
    config: Any,
    checkpoint_attestation: dict[str, Any],
) -> dict[str, Any]:
    """Create one immutable run identity, or require an exact existing match."""
    root = Path(config.results_root)
    isolation_attestation = _load_isolation_attestation(root)
    if (
        isolation_attestation.get("planner_transport")
        != config.planner_transport.manifest_identity()
    ):
        raise ValueError("isolation preflight belongs to a different planner transport")
    candidate = make_run_manifest(
        build_run_configuration(
            config,
            checkpoint_attestation,
            isolation_attestation=isolation_attestation,
        )
    )
    path = root / RUN_MANIFEST_NAME
    existing, problem = load_run_manifest(root)
    if existing is not None:
        if existing != candidate:
            raise ValueError(
                "results root belongs to a different RoboCasa run configuration"
            )
        return existing
    if problem != "missing run manifest":
        raise ValueError(problem)
    entries = list(root.iterdir())
    if entries and not _only_approved_preflight(root, entries):
        raise ValueError(
            "refusing to attach a run manifest to a non-empty results root"
        )
    try:
        _exclusive_write(path, candidate)
    except FileExistsError:
        existing, problem = load_run_manifest(root)
        if problem is not None or existing != candidate:
            raise ValueError(
                "concurrent process created a different or invalid run manifest"
            )
        return existing
    return candidate
