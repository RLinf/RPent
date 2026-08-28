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

"""Execute one frozen RoboCasa cell through the audited local RLDX harness."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import signal
import stat
import subprocess
import tempfile
import time
import uuid
import xml.etree.ElementTree as ET
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import ProxyHandler, Request, build_opener

from robots.robocasa.checkpoint_identity import (
    expected_fingerprint,
    load_attestation,
)
from robots.robocasa.checkpoint_identity import (
    verify_checkpoint as verify_checkpoint_identity,
)

from .artifacts import (
    CellResult,
    Completion,
    Integrity,
    Outcome,
    atomic_write_artifact_file,
    publish_completed_cell,
    secure_artifact_directory,
    secure_artifact_subdirectory,
)
from .deadline_supervisor import (
    CONTRACT_NAME,
    DEADLINE_PROTOCOL,
    FREEZE_NAME,
    SEAL_NAME,
    _read_trusted_json,
    driver_stop_problem,
    process_identity,
    scan_committed_prefix,
    timeout_driver_exit_problem,
)
from .formal_prompt import render_formal_prompt
from .memory import validate_memory_pack
from .planner_transport import (
    CHATGPT_BROKER_PROFILE,
    CODEX_REQUEST_MAX_RETRIES,
    CODEX_STREAM_IDLE_TIMEOUT_MS,
    CODEX_STREAM_MAX_RETRIES,
    PlannerTransport,
    normalize_responses_base_url,
)
from .profiles import PlannerAdapter, PlannerProfile, is_audited_profile
from .protocol import EMPTY_MEMORY_TASKS, SPLITS, Cell, Split
from .provenance import ensure_run_manifest

EXIT_CANONICAL = 0
EXIT_RETRYABLE_INFRA = 75
EXIT_ARTIFACT = 76
EXIT_CONFIGURATION = 78

PROTOCOL_ID = "robocasa-harness-vla-v1"
RETRYABLE_TERMINATIONS = frozenset(
    {
        "driver_startup_failed",
        "driver_runtime_failed",
        "planner_failed",
        "wrapper_error",
    }
)
BUILDER_TERMINATIONS = frozenset(
    {
        "planner_completed",
        "planner_timeout",
        "planner_failed",
        "planner_protocol_error",
        "driver_startup_failed",
        "driver_runtime_failed",
        "configuration_failed",
        "operator_interrupted",
    }
)

CHECKPOINT_SHA256 = {
    "model-00001-of-00003.safetensors": (
        "2a2f48bd2d2979979700c85c44051c37c3256de528842d82883ba756b070e541"
    ),
    "model-00002-of-00003.safetensors": (
        "4bb91b9038d7825809c09da425d5dbd6a52ba1a1af25de09bc94ff218e1f80fc"
    ),
    "model-00003-of-00003.safetensors": (
        "f348bb0aee031e6fd32cad2ff51aa5e4eaf74fa42299dae051f7e3ca0b8adf53"
    ),
}
PINNED_CODEX_VERSION = "codex-cli 0.147.0"
PINNED_CODEX_SHA256 = "cb0a15567e9a60a5820d54b0f6ae86d504dc3805c1eab21a47f70e3eb7b73a40"
PINNED_CODEX_PATH = Path("/usr/local/libexec/rpent/codex-0.147.0")
FROZEN_RUNTIME_SHA256 = {
    "driver": "e776ad2b09f0060d1926ddf081e427626b1f3db03f99ec96e04b3862ca3f2531",
    "readiness": "22153c6b41946bad1500023361034e35d12601fe293e59355ec62340c67a30c3",
    "deadline": "ae9a7b65b68e6d1a61dace0c15f273aa742f53ce2143b8d6dde92471c2b14c0b",
    "isolation_launcher": (
        "6f0173f39753a8268b0b73aac2b729066f9798a0679c07e705a7c22f45cf81cb"
    ),
    "artifact_builder": (
        "724f53794887c609be94a2c2a7fd8323ee4b9ad480d7d731721e82f633f41232"
    ),
    "interactive_env": (
        "f42908f22ccc0db88610d35d314d14c903829cdd2de8ca53057c35c651ece634"
    ),
    "rldx_skill": ("f1c5b16cff92d938adcdb3929d8c5f8ba595b640de9105c36aa87d7d39e0b56b"),
}


@dataclass(frozen=True)
class RuntimePaths:
    """Paths supplied by the preliminary local RLDX runtime snapshot."""

    root: Path
    sim_python: Path
    driver: Path
    readiness: Path
    deadline: Path
    isolation_launcher: Path
    artifact_builder: Path
    model: Path
    vlm_metadata: Path
    navview_xml: Path

    @classmethod
    def discover(cls, root: Path) -> "RuntimePaths":
        root = Path(root).expanduser().resolve()
        migration = root / "migration"
        robosuite = (
            root / "external_dependencies" / "robocasa365" / "robosuite" / "robosuite"
        )
        return cls(
            root=root,
            sim_python=(
                root / "rldx/eval/sim/robocasa365/robocasa365_uv/.venv/bin/python"
            ),
            driver=migration / "robocasa_interactive_driver.py",
            readiness=migration / "wait_for_driver_ready.py",
            deadline=migration / "run_planner_with_deadline.py",
            isolation_launcher=migration / "launch_isolated_codex.py",
            artifact_builder=migration / "build_rollout_artifacts.py",
            model=root / "checkpoints/RLDX-1-FT-RC365",
            vlm_metadata=root / "checkpoints/RLDX-1-VLM-metadata",
            navview_xml=robosuite / "models/assets/bases/omron_mobile_base.xml",
        )


@dataclass(frozen=True)
class ExecutorConfig:
    """Immutable inputs shared by every cell in one reproduction run."""

    runtime: RuntimePaths
    results_root: Path
    rollout_root: Path
    memory_root: Path
    planner_profile: PlannerProfile
    planner_transport: PlannerTransport
    codex_bin: Path = PINNED_CODEX_PATH
    startup_timeout_seconds: int = 2400
    kill_after_seconds: int = 15
    keep_workdirs: bool = False
    preliminary_local_runtime: bool = False
    memory_source: str = "local"
    memory_revision: str | None = None


@dataclass(frozen=True)
class Execution:
    """One executor attempt before scheduler retry policy is applied."""

    cell: Cell
    return_code: int
    termination_cause: str
    canonical: bool
    workdir: Path
    message: str | None = None


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def checkpoint_attestation_path(config: ExecutorConfig) -> Path:
    return config.results_root / "_preflight" / "checkpoint-attestation.json"


def _write_private_json(path: Path, value: dict[str, Any]) -> None:
    parent = path.parent
    if parent.exists() or parent.is_symlink():
        metadata = parent.lstat()
        if parent.is_symlink() or not stat.S_ISDIR(metadata.st_mode):
            raise ValueError(
                f"private output parent must be a real directory: {parent}"
            )
        if metadata.st_uid != os.geteuid() or metadata.st_mode & 0o077:
            raise ValueError(f"private output parent has unsafe permissions: {parent}")
    else:
        parent.mkdir(parents=True, mode=0o700)
        parent.chmod(0o700)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=parent)
    temporary = Path(temporary_name)
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        path.chmod(0o600)
    finally:
        temporary.unlink(missing_ok=True)


def _load_checkpoint_attestation(config: ExecutorConfig) -> dict[str, Any]:
    return load_attestation(
        checkpoint_attestation_path(config),
        config.runtime.model,
        config.runtime.vlm_metadata,
    )


def _trusted_credential(path: Path) -> str | None:
    try:
        metadata = path.lstat()
    except OSError as exc:
        return f"cannot inspect planner credential file: {exc}"
    if path.is_symlink() or not stat.S_ISREG(metadata.st_mode):
        return "planner credential path must be a regular non-symlink file"
    if stat.S_IMODE(metadata.st_mode) != 0o600:
        return "planner credential file mode must be 0600"
    if metadata.st_uid != os.geteuid():
        return "planner credential file must be owned by the reproduction user"
    if metadata.st_size <= 1:
        return "planner credential file is empty"
    if metadata.st_size > 16 * 1024:
        return "planner credential file is unexpectedly large"
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        return f"cannot securely open planner credential file: {exc}"
    try:
        opened = os.fstat(descriptor)
        if (opened.st_dev, opened.st_ino) != (metadata.st_dev, metadata.st_ino):
            return "planner credential file changed while it was being opened"
        raw = os.read(descriptor, metadata.st_size + 1)
    finally:
        os.close(descriptor)
    try:
        text = raw.decode("utf-8")
    except UnicodeError:
        return "planner credential file must contain UTF-8 text"
    if len(text.splitlines()) != 1 or not text.strip():
        return "planner credential file must contain exactly one non-empty line"
    return None


def _broker_health_problem(config: ExecutorConfig) -> str | None:
    """Verify the provider-neutral subscription broker admission contract."""
    transport = config.planner_transport
    if not transport.uses_broker:
        return None
    if transport.broker_health_url is None or transport.broker_protocol is None:
        return "subscription transport is missing its broker health contract"
    request = Request(
        transport.broker_health_url,
        method="GET",
        headers={
            "Accept": "application/json",
            "User-Agent": "rpent-robocasa-preflight",
        },
    )
    try:
        opener = build_opener(ProxyHandler({}))
        with opener.open(request, timeout=5.0) as response:
            payload = response.read(64 * 1024 + 1)
    except (HTTPError, URLError, OSError, TimeoutError) as exc:
        return f"subscription broker health check failed: {type(exc).__name__}"
    if len(payload) > 64 * 1024:
        return "subscription broker health response is too large"
    try:
        value = json.loads(payload.decode("utf-8", errors="strict"))
    except (UnicodeError, ValueError):
        return "subscription broker health response is not valid JSON"
    expected = {
        "provider_profile": CHATGPT_BROKER_PROFILE,
        "auth_mode": "chatgpt_broker",
        "credential_broker": True,
        "credential_broker_ready": True,
        "credential_broker_protocol": transport.broker_protocol,
        "model": config.planner_profile.model,
        "reasoning_effort": config.planner_profile.reasoning_effort,
    }
    if not isinstance(value, dict) or any(
        value.get(key) != expected_value for key, expected_value in expected.items()
    ):
        return "subscription broker health identity differs from the frozen contract"
    return None


def _private_directory_problem(path: Path, label: str) -> str | None:
    try:
        metadata = path.lstat()
    except OSError as exc:
        return f"cannot inspect {label}: {exc}"
    if path.is_symlink() or not stat.S_ISDIR(metadata.st_mode):
        return f"{label} must be a real directory: {path}"
    if metadata.st_uid != os.geteuid():
        return f"{label} must be owned by the reproduction user: {path}"
    if metadata.st_mode & 0o077:
        return f"{label} must not be accessible by group or other: {path}"
    return None


def _rollout_directory_problem(path: Path) -> str | None:
    """Require a non-listable root that isolated planner UIDs can traverse."""
    try:
        metadata = path.lstat()
    except OSError as exc:
        return f"cannot inspect rollout root: {exc}"
    if path.is_symlink() or not stat.S_ISDIR(metadata.st_mode):
        return f"rollout root must be a real directory: {path}"
    if metadata.st_uid != os.geteuid():
        return f"rollout root must be owned by the reproduction user: {path}"
    if stat.S_IMODE(metadata.st_mode) != 0o711:
        return "rollout root must have exact mode 0711"
    resolved = path.resolve()
    for parent in reversed(resolved.parents):
        try:
            parent_metadata = parent.stat()
        except OSError as exc:
            return f"cannot inspect rollout ancestor {parent}: {exc}"
        if not stat.S_ISDIR(parent_metadata.st_mode):
            return f"rollout ancestor must be a directory: {parent}"
        if parent_metadata.st_mode & stat.S_IXOTH == 0:
            return f"rollout ancestor must be traversable by the planner: {parent}"
    return None


def _roots_overlap(first: Path, second: Path) -> bool:
    left = first.resolve()
    right = second.resolve()
    return left == right or left in right.parents or right in left.parents


def _trusted_codex_problem(path: Path) -> str | None:
    """Accept the standard root-owned /usr/bin symlink to an immutable CLI file."""
    try:
        link = path.lstat()
        resolved = path.resolve(strict=True)
        target = resolved.stat()
    except OSError as exc:
        return f"cannot inspect codex_bin: {exc}"
    if not (stat.S_ISREG(link.st_mode) or stat.S_ISLNK(link.st_mode)):
        return "codex_bin must be a regular file or a controlled symlink"
    if link.st_uid != os.geteuid():
        return "codex_bin entry must be owned by the run user"
    if stat.S_ISREG(link.st_mode) and link.st_mode & 0o022:
        return "codex_bin entry must not be writable by group or other"
    if not stat.S_ISREG(target.st_mode):
        return "codex_bin must resolve to a regular file"
    if target.st_uid != os.geteuid() or target.st_mode & 0o022:
        return "codex_bin target must be owned by the run user and not writable"
    if resolved.name == "codex.js" and resolved.parent.name == "bin":
        package_root = resolved.parent.parent
        package_json = package_root / "package.json"
        native = sorted(
            package_root.glob("node_modules/@openai/codex-*/vendor/*/bin/codex")
        )
        if not package_json.is_file() or package_json.is_symlink() or len(native) != 1:
            return "Codex package must contain one immutable native binary"
        for component in (package_json, native[0]):
            metadata = component.lstat()
            if (
                component.is_symlink()
                or not stat.S_ISREG(metadata.st_mode)
                or metadata.st_uid != os.geteuid()
                or metadata.st_mode & 0o022
            ):
                return "Codex package inputs must be owned, regular, and immutable"
    return None


def _pinned_codex_identity(path: Path) -> tuple[dict[str, str], str | None]:
    identity: dict[str, str] = {}
    try:
        resolved = path.resolve(strict=True)
        digest = sha256_file(resolved)
    except OSError as exc:
        return identity, f"cannot hash codex_bin: {exc}"
    identity["path"] = str(resolved)
    identity["sha256"] = digest
    if digest != PINNED_CODEX_SHA256:
        return identity, "codex_bin does not match the pinned 0.147.0 SHA-256"
    environment = {
        "PATH": "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        "NO_COLOR": "1",
    }
    try:
        result = subprocess.run(
            [str(resolved), "--version"],
            check=False,
            capture_output=True,
            text=True,
            timeout=15,
            env=environment,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return identity, f"cannot query codex_bin version: {type(exc).__name__}"
    version = result.stdout.strip()
    identity["version"] = version
    if result.returncode != 0 or version != PINNED_CODEX_VERSION:
        return identity, f"codex_bin version must be exactly {PINNED_CODEX_VERSION}"
    return identity, None


def _managed_codex_config_problem(path: Path = Path("/etc/codex")) -> str | None:
    """Managed config can override CLI policy, so the frozen run permits none."""
    if not path.exists() and not path.is_symlink():
        return None
    try:
        metadata = path.lstat()
    except OSError as exc:
        return f"cannot inspect managed Codex config root: {exc}"
    if path.is_symlink() or not stat.S_ISDIR(metadata.st_mode):
        return f"managed Codex config root must be absent: {path}"
    try:
        entries = list(path.iterdir())
    except OSError as exc:
        return f"cannot enumerate managed Codex config root: {exc}"
    if entries:
        return f"managed Codex config root must be empty: {path}"
    return None


def _frozen_runtime_problem(label: str, path: Path) -> str | None:
    expected = FROZEN_RUNTIME_SHA256[label]
    try:
        metadata = path.lstat()
    except OSError as exc:
        return f"cannot inspect frozen {label}: {exc}"
    if (
        path.is_symlink()
        or not stat.S_ISREG(metadata.st_mode)
        or metadata.st_uid != os.geteuid()
        or metadata.st_mode & 0o022
    ):
        return f"frozen {label} must be owned, regular, and immutable"
    try:
        actual = sha256_file(path)
    except OSError as exc:
        return f"cannot hash frozen {label}: {exc}"
    if actual != expected:
        return f"frozen {label} does not match its approved SHA-256"
    return None


def _git_snapshot(root: Path) -> dict[str, Any]:
    value: dict[str, Any] = {"root": str(root)}
    try:
        value["commit"] = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        value["dirty"] = bool(
            subprocess.run(
                ["git", "-C", str(root), "status", "--porcelain"],
                check=True,
                capture_output=True,
                text=True,
            ).stdout
        )
    except (OSError, subprocess.CalledProcessError):
        value["commit"] = None
        value["dirty"] = None
    return value


def _navview_problem(path: Path) -> str | None:
    try:
        root = ET.parse(path).getroot()
    except Exception as exc:
        return f"cannot parse robosuite Omron XML: {exc}"
    all_cameras = [
        item for item in root.findall(".//camera") if item.get("name") == "navview"
    ]
    direct_cameras = root.findall(
        "./worldbody/body[@name='base']/camera[@name='navview']"
    )
    if len(all_cameras) != 1 or len(direct_cameras) != 1:
        return (
            "robosuite Omron base must contain exactly one navview camera as a "
            "direct child of worldbody/body[@name='base']"
        )
    camera = direct_cameras[0]
    expected = {
        "mode": "fixed",
        "pos": "0.2 0 1.6",
        "xyaxes": "0 -1 0 0.643 0 0.766",
        "fovy": "75",
    }
    mismatches = {
        key: camera.get(key)
        for key, value in expected.items()
        if camera.get(key) != value
    }
    return f"navview camera attributes differ: {mismatches}" if mismatches else None


def _runtime_import_identity(
    runtime: RuntimePaths,
) -> tuple[dict[str, str], str | None]:
    marker = "RPENT_IMPORT_IDENTITY="
    robocasa_source = runtime.root / "external_dependencies/robocasa365"
    robosuite_source = robocasa_source / "robosuite"
    probe = (
        "import json, sys; "
        f"sys.path[:0] = {json.dumps([str(robocasa_source), str(robosuite_source)])}; "
        "import robocasa, robosuite; "
        f"print('{marker}' + json.dumps({{'robocasa': robocasa.__file__, "
        "'robosuite': robosuite.__file__}, sort_keys=True))"
    )
    environment = {
        "PATH": "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        "MUJOCO_GL": "egl",
        "PYOPENGL_PLATFORM": "egl",
        "PYTHONNOUSERSITE": "1",
        "PYTHONSAFEPATH": "1",
    }
    try:
        result = subprocess.run(
            [str(runtime.sim_python), "-c", probe],
            cwd=runtime.root,
            env=environment,
            check=False,
            capture_output=True,
            text=True,
            timeout=120,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return {}, f"cannot probe simulator imports: {type(exc).__name__}"
    lines = [line for line in result.stdout.splitlines() if line.startswith(marker)]
    if result.returncode != 0 or len(lines) != 1:
        return {}, "simulator import identity probe failed"
    try:
        identity = json.loads(lines[0][len(marker) :])
        expected = {
            "robocasa": runtime.root
            / "external_dependencies/robocasa365/robocasa/__init__.py",
            "robosuite": runtime.root
            / "external_dependencies/robocasa365/robosuite/robosuite/__init__.py",
        }
        for name, expected_path in expected.items():
            imported = Path(identity[name])
            if not imported.samefile(expected_path):
                return identity, f"simulator imports unexpected {name}: {imported}"
    except (KeyError, OSError, TypeError, ValueError):
        return {}, "simulator import identity is invalid"
    return identity, None


def doctor(
    config: ExecutorConfig,
    *,
    verify_checkpoint: bool = False,
    verify_isolation: bool = False,
) -> dict[str, Any]:
    """Return fail-closed runtime diagnostics without starting a simulator."""
    problems: list[str] = []
    runtime = config.runtime
    if not config.preliminary_local_runtime:
        problems.append(
            "the external RLDX snapshot adapter is preliminary; pass the explicit "
            "preliminary-local-runtime acknowledgement"
        )
    if os.geteuid() != 0:
        problems.append("the audited local isolation launcher requires root")
    if problem := _private_directory_problem(config.results_root, "results root"):
        problems.append(problem)
    if problem := _rollout_directory_problem(config.rollout_root):
        problems.append(problem)
    if _roots_overlap(config.results_root, config.rollout_root):
        problems.append("results root and rollout root must not overlap")
    transport = config.planner_transport
    normalized_base_url = normalize_responses_base_url(transport.request_base_url)
    planner_adapter: PlannerAdapter | None = None
    planner_files: dict[str, Path] = {}
    try:
        planner_adapter = _planner_adapter(config.planner_profile)
    except RuntimeError as exc:
        problems.append(str(exc))
    else:
        planner_files = planner_adapter.required_files(config)
    required_files = {
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
        "navview_patch": (
            Path(__file__).resolve().parents[3]
            / "robots/robocasa/patches/robosuite_navview.patch"
        ),
    }
    required_files.update(planner_files)
    trusted_symlink_labels = {"sim_python"}
    if planner_adapter is not None:
        trusted_symlink_labels.update(planner_adapter.trusted_symlink_labels)
    for label, path in required_files.items():
        if path.is_symlink() and label not in trusted_symlink_labels:
            problems.append(f"{label} must not be a symlink: {path}")
        if not path.is_file():
            problems.append(f"missing {label}: {path}")
    for label in FROZEN_RUNTIME_SHA256:
        path = required_files[label]
        if path.is_file() and (problem := _frozen_runtime_problem(label, path)):
            problems.append(problem)
    codex_identity: dict[str, str] = {}
    managed_config = {"path": "/etc/codex", "empty_or_absent": False}
    if planner_adapter is not None and planner_adapter.backend == "codex":
        if problem := _trusted_codex_problem(config.codex_bin):
            problems.append(problem)
        codex_identity, identity_problem = _pinned_codex_identity(config.codex_bin)
        if identity_problem is not None:
            problems.append(identity_problem)
        if problem := _managed_codex_config_problem():
            problems.append(problem)
        else:
            managed_config["empty_or_absent"] = True
    if not runtime.model.is_dir():
        problems.append(f"missing RLDX checkpoint: {runtime.model}")
    if not runtime.vlm_metadata.is_dir():
        problems.append(f"missing VLM metadata: {runtime.vlm_metadata}")
    if problem := _navview_problem(runtime.navview_xml):
        problems.append(problem)
    runtime_imports, import_problem = _runtime_import_identity(runtime)
    if import_problem is not None:
        problems.append(import_problem)
    if problem := _trusted_credential(transport.credential_file):
        problems.append(problem)
    broker_health_problem = _broker_health_problem(config)
    if broker_health_problem is not None:
        problems.append(broker_health_problem)
    problems.extend(validate_memory_pack(config.memory_root))

    shard_rows: dict[str, dict[str, Any]] = {}
    for name, expected in CHECKPOINT_SHA256.items():
        path = runtime.model / name
        row: dict[str, Any] = {"path": str(path), "expected_sha256": expected}
        if not path.is_file():
            problems.append(f"missing checkpoint shard: {path}")
        else:
            row["size"] = path.stat().st_size
        shard_rows[name] = row
    attestation = None
    attestation_source = None
    if verify_checkpoint:
        try:
            attestation = _load_checkpoint_attestation(config)
        except (OSError, UnicodeError, ValueError):
            try:
                attestation = verify_checkpoint_identity(
                    runtime.model,
                    runtime.vlm_metadata,
                )
            except (OSError, UnicodeError, ValueError) as exc:
                problems.append(f"checkpoint verification failed: {exc}")
            else:
                attestation_source = "full_hash"
        else:
            attestation_source = "reused_attestation"
        if attestation is not None:
            for name, row in shard_rows.items():
                row["sha256"] = attestation["files"][f"model/{name}"]["sha256"]
    scripts = {
        label: sha256_file(path)
        for label, path in required_files.items()
        if label not in planner_files and path.is_file()
    }
    if attestation is not None and attestation_source == "full_hash" and not problems:
        try:
            _write_private_json(checkpoint_attestation_path(config), attestation)
        except (OSError, ValueError) as exc:
            problems.append(f"cannot publish checkpoint attestation: {exc}")
    isolation_attestation = None
    if verify_isolation and not problems:
        from .preflight import run_isolation_preflight

        try:
            isolation_attestation = run_isolation_preflight(config)
        except (OSError, RuntimeError, ValueError) as exc:
            problems.append(f"isolation preflight failed: {exc}")
    return {
        "ok": not problems,
        "release_ready": False,
        "runtime_kind": "preliminary_external_snapshot",
        "problems": problems,
        "runtime_root": str(runtime.root),
        "runtime_imports": runtime_imports,
        "planner_profile": config.planner_profile.name,
        "planner_transport": {
            **transport.manifest_identity(),
            "request_base_url": normalized_base_url,
            "broker_health_verified": (
                transport.uses_broker and broker_health_problem is None
            ),
        },
        "codex": codex_identity,
        "managed_codex_config": managed_config,
        "isolation_preflight": {
            "verified": isolation_attestation is not None,
            "attestation_sha256": (
                isolation_attestation.get("attestation_sha256")
                if isolation_attestation is not None
                else None
            ),
        },
        "navview_xml": str(runtime.navview_xml),
        "scripts": scripts,
        "runtime_git": _git_snapshot(runtime.root),
        "robosuite_git": _git_snapshot(runtime.navview_xml.parents[4]),
        "rpent_git": _git_snapshot(Path(__file__).resolve().parents[3]),
        "memory": {
            "source": config.memory_source,
            "revision": config.memory_revision,
            "manifest_sha256": sha256_file(config.memory_root / "manifest.json")
            if (config.memory_root / "manifest.json").is_file()
            else None,
        },
        "checkpoint": {
            "verified": attestation is not None and not problems,
            "verification_source": attestation_source,
            "fingerprint": (
                attestation["fingerprint"]
                if attestation is not None
                else expected_fingerprint()
            ),
            "attestation": str(checkpoint_attestation_path(config)),
            "shards": shard_rows,
        },
    }


def _stage_memory(config: ExecutorConfig, cell: Cell, workdir: Path) -> bool:
    if problems := validate_memory_pack(config.memory_root):
        raise RuntimeError(
            "memory pack changed after preflight: " + "; ".join(problems)
        )
    manifest = _read_json(config.memory_root / "manifest.json")
    entries = manifest.get("entries")
    entry = (
        next(
            (
                item
                for item in entries
                if isinstance(item, dict) and item.get("task") == cell.task
            ),
            None,
        )
        if isinstance(entries, list)
        else None
    )
    if not isinstance(entry, dict) or not isinstance(entry.get("files"), dict):
        raise RuntimeError(f"memory manifest has no valid entry for {cell.task}")
    expected_hashes = entry["files"]
    source = config.memory_root / cell.task
    destination = workdir / "task_memory"
    destination.mkdir(mode=0o700)
    files = sorted(source.iterdir())
    missing = cell.task in EMPTY_MEMORY_TASKS
    if missing and files:
        raise RuntimeError(f"empty-memory task has staged files: {cell.task}")
    if not missing and {path.name for path in files} != {
        f"{cell.task}_s0.json",
        f"{cell.task}_s0.jsonl",
    }:
        raise RuntimeError(f"task memory is not an exact pair: {cell.task}")
    for source_path in files:
        if source_path.is_symlink() or not source_path.is_file():
            raise RuntimeError(f"unsafe task memory source: {source_path}")
        target = destination / source_path.name
        shutil.copyfile(source_path, target)
        if sha256_file(target) != expected_hashes.get(source_path.name):
            raise RuntimeError(f"task memory changed while staging: {cell.task}")
        os.chown(target, 0, 0)
        os.chmod(target, 0o444)
    os.chown(destination, 0, 0)
    os.chmod(destination, 0o555)
    return not missing


def _reset_seed(cell: Cell) -> int | None:
    if cell.split is Split.COMPOSITE_SEEN:
        return 4_200_000 + cell.seed
    return None


def _driver_environment(
    config: ExecutorConfig, gpu: int, cell: Cell | None = None
) -> dict[str, str]:
    environment = {
        "PATH": "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        "LD_LIBRARY_PATH": "/usr/local/nvidia/lib:/usr/local/nvidia/lib64",
        "VK_DRIVER_FILES": "/etc/vulkan/icd.d/nvidia_icd.json",
        "VK_ICD_FILENAMES": "/etc/vulkan/icd.d/nvidia_icd.json",
        "CUDA_VISIBLE_DEVICES": str(gpu),
        "MUJOCO_GL": "egl",
        "PYOPENGL_PLATFORM": "egl",
        "PYTHONHASHSEED": "0",
        "PYTHONNOUSERSITE": "1",
        "PYTHONSAFEPATH": "1",
        "NO_ALBUMENTATIONS_UPDATE": "1",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "TOKENIZERS_PARALLELISM": "false",
        "RLDX_MODEL_PATH": str(config.runtime.model),
        "RLDX_VLM_PATH": str(config.runtime.vlm_metadata),
        "RLDX_CHECKPOINT_ATTESTATION": str(checkpoint_attestation_path(config)),
        "RLDX_CHECKPOINT_FINGERPRINT": expected_fingerprint(),
        "RLDX_ATTN_IMPL": "sdpa",
        "RLDX_PRELOAD_VLA": "1",
        "RLDX_ALLOW_RESET": "0",
        "RLDX_PERCEPTION_ISOLATION": "1",
        "RLDX_MAX_CHUNKS": "40",
        "RLDX_SETTLE_PATIENCE": "999",
    }
    if cell is not None and (reset_seed := _reset_seed(cell)) is not None:
        environment["RLDX_RESET_SEED"] = str(reset_seed)
    return environment


def ensure_execution_run_manifest(
    config: ExecutorConfig,
    checkpoint_attestation: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Re-attest immutable inputs and bind them to this results root."""
    if checkpoint_attestation is None:
        checkpoint_attestation = _load_checkpoint_attestation(config)
    return ensure_run_manifest(
        config,
        checkpoint_attestation,
    )


def _stop_process(process: subprocess.Popen[Any], grace_seconds: float = 5.0) -> None:
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=grace_seconds)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait(timeout=grace_seconds)


def _navview_rollout_problem(workdir: Path) -> str | None:
    done_steps: list[int] = []
    for path in workdir.glob("done_*.flag"):
        try:
            done_steps.append(int(path.stem.split("_", 1)[1]))
        except (IndexError, ValueError):
            return f"unexpected navview commit marker: {path.name}"
    if not done_steps or 0 not in done_steps:
        return "initial navview commit is missing"
    for step in done_steps:
        suffix = f"{step:02d}"
        for name in (f"image_nav_{suffix}.png", f"image_nav_floor_{suffix}.png"):
            path = workdir / name
            try:
                metadata = path.lstat()
            except OSError:
                return f"committed step {step} is missing {name}"
            if (
                path.is_symlink()
                or not stat.S_ISREG(metadata.st_mode)
                or metadata.st_uid != os.geteuid()
                or metadata.st_size <= 0
            ):
                return f"committed step {step} has unsafe {name}"
    latest = workdir / f"world_nav_{max(done_steps):02d}.npy"
    try:
        metadata = latest.lstat()
    except OSError:
        return "latest committed step has no navview world map"
    if (
        latest.is_symlink()
        or not stat.S_ISREG(metadata.st_mode)
        or metadata.st_uid != os.geteuid()
        or metadata.st_size <= 0
    ):
        return "latest navview world map is unsafe"
    return None


def _latched_success_exit(workdir: Path, return_code: int | None) -> bool:
    """Recognize only the adapter's committed, zero-exit success shutdown."""
    if return_code != 0:
        return False
    steps: list[int] = []
    for marker in workdir.glob("done_*.flag"):
        try:
            steps.append(int(marker.stem.split("_", 1)[1]))
        except (IndexError, ValueError):
            return False
    if not steps:
        return False
    suffix = f"{max(steps):02d}"
    state_path = workdir / f"state_{suffix}.json"
    log_path = workdir / f"log_{suffix}.json"
    trace_path = workdir / "command_trace.jsonl"
    for path in (state_path, log_path, trace_path):
        try:
            metadata = path.lstat()
        except OSError:
            return False
        if path.is_symlink() or not stat.S_ISREG(metadata.st_mode):
            return False
    try:
        state = _read_json(state_path)
    except (OSError, ValueError):
        return False
    return state.get("success") is True


def _timeout_driver_exit_expected(
    deadline_attestation: dict[str, Any] | None,
) -> bool:
    """Accept only the adapter deadline exit or an attested supervisor SIGKILL."""
    if deadline_attestation is None:
        return False
    return (
        timeout_driver_exit_problem(
            deadline_attestation.get("driver_stop"),
            deadline_attestation.get("driver_return_code_before_stop"),
            success_latched=deadline_attestation.get("success_at_deadline"),
        )
        is None
    )


def _observe_driver_exit(
    process: subprocess.Popen[Any],
    *,
    termination: str,
    timeout_seconds: float,
) -> int | None:
    """Reap a timeout-killed driver before the fallback stop path can signal it."""
    return_code = process.poll()
    if return_code is not None or termination != "planner_timeout":
        return return_code
    try:
        return process.wait(timeout=timeout_seconds)
    except subprocess.TimeoutExpired:
        return None


@dataclass(frozen=True)
class _CodexPlannerAdapter:
    """Audited Codex CLI implementation of the formal planner boundary."""

    backend: str = "codex"
    audit_id: str = "codex-cli-0.147.0/outer-landlock-v1"
    trusted_symlink_labels: frozenset[str] = frozenset({"codex_bin"})

    def required_files(self, config: ExecutorConfig) -> dict[str, Path]:
        return {"codex_bin": config.codex_bin}

    def build_command(
        self,
        config: ExecutorConfig,
        *,
        workdir: Path,
        run_id: str,
        cell: Cell,
    ) -> list[str]:
        return _codex_planner_command(config, workdir=workdir, run_id=run_id, cell=cell)


_PLANNER_ADAPTERS: dict[str, PlannerAdapter] = {
    "codex": _CodexPlannerAdapter(),
}


def _planner_adapter(profile: PlannerProfile) -> PlannerAdapter:
    if not is_audited_profile(profile):
        raise RuntimeError(
            f"planner profile {profile.name!r} ({profile.backend}) is declared "
            "but not audited for formal cell execution"
        )
    try:
        return _PLANNER_ADAPTERS[profile.backend]
    except KeyError as exc:
        raise RuntimeError(
            f"no audited PlannerAdapter is registered for backend {profile.backend!r}"
        ) from exc


def _planner_command(
    config: ExecutorConfig,
    *,
    workdir: Path,
    run_id: str,
    cell: Cell,
) -> list[str]:
    """Build a secret-free argv through the profile-selected adapter."""
    return _planner_adapter(config.planner_profile).build_command(
        config, workdir=workdir, run_id=run_id, cell=cell
    )


def _codex_planner_command(
    config: ExecutorConfig,
    *,
    workdir: Path,
    run_id: str,
    cell: Cell,
) -> list[str]:
    identity = zlib.crc32(f"{run_id}|{cell.tag}".encode())
    uid = 100000 + identity % 2_000_000_000
    prompt = render_formal_prompt(
        task=cell.task,
        workdir=".",
        task_memory_dir="task_memory",
    )
    transport = config.planner_transport
    base_url = transport.request_base_url
    provider_id = transport.provider_id
    # The launcher applies an irreversible outer Landlock boundary before
    # Codex starts.  A full-write Codex filesystem view avoids stacking bwrap
    # or a second Landlock layer; the explicit network restriction still makes
    # Codex install seccomp on every generated command.
    return [
        str(config.runtime.sim_python),
        str(Path(__file__).with_name("sandbox.py")),
        "--launcher",
        str(config.runtime.isolation_launcher),
        "--launcher-sha256",
        FROZEN_RUNTIME_SHA256["isolation_launcher"],
        "--rollout-root",
        str(config.rollout_root),
        "--",
        "--uid",
        str(uid),
        "--gid",
        str(uid),
        "--workdir",
        str(workdir),
        "--key-file",
        str(transport.credential_file),
        "--",
        str(config.codex_bin),
        "--ask-for-approval",
        "never",
        "exec",
        "--strict-config",
        "--ephemeral",
        "--json",
        "--ignore-user-config",
        "--ignore-rules",
        "--skip-git-repo-check",
        "--disable",
        "browser_use",
        "--disable",
        "browser_use_external",
        "--disable",
        "browser_use_full_cdp_access",
        "--disable",
        "in_app_browser",
        "--disable",
        "image_generation",
        "--disable",
        "multi_agent",
        "--disable",
        "multi_agent_v2",
        "--disable",
        "standalone_web_search",
        "--disable",
        "apps",
        "--disable",
        "plugins",
        "--disable",
        "remote_plugin",
        "--disable",
        "plugin_sharing",
        "--disable",
        "recommended_plugins",
        "--disable",
        "skill_search",
        "--disable",
        "skill_mcp_dependency_install",
        "--disable",
        "tool_suggest",
        "-C",
        ".",
        "-c",
        'default_permissions="rpent_outer_landlock"',
        "-c",
        'permissions.rpent_outer_landlock.filesystem={":root"="write"}',
        "-c",
        "permissions.rpent_outer_landlock.network.enabled=false",
        "-c",
        'web_search="disabled"',
        "-c",
        "allow_login_shell=false",
        "-c",
        "analytics.enabled=false",
        "-c",
        "feedback.enabled=false",
        "-c",
        "check_for_update_on_startup=false",
        "-c",
        'otel.trace_exporter="none"',
        "-c",
        "tools.experimental_request_user_input.enabled=false",
        "-c",
        f"model_provider={provider_id}",
        "-c",
        f"model={config.planner_profile.model}",
        "-c",
        f"model_reasoning_effort={config.planner_profile.reasoning_effort}",
        "-c",
        "memories.use_memories=false",
        "-c",
        "memories.generate_memories=false",
        "-c",
        "skills.bundled.enabled=false",
        "-c",
        "skills.include_instructions=false",
        "-c",
        "shell_environment_policy.inherit=core",
        "-c",
        "shell_environment_policy.ignore_default_excludes=false",
        "-c",
        'shell_environment_policy.exclude=["RLDX_PLANNER_API_KEY"]',
        "-c",
        f"model_providers.{provider_id}.name={json.dumps(transport.provider_name)}",
        "-c",
        f"model_providers.{provider_id}.base_url={json.dumps(base_url)}",
        "-c",
        f"model_providers.{provider_id}.wire_api=responses",
        "-c",
        f"model_providers.{provider_id}.env_key=RLDX_PLANNER_API_KEY",
        "-c",
        f"model_providers.{provider_id}.requires_openai_auth=false",
        "-c",
        (
            f"model_providers.{provider_id}.request_max_retries="
            f"{CODEX_REQUEST_MAX_RETRIES}"
        ),
        "-c",
        (
            f"model_providers.{provider_id}.stream_max_retries="
            f"{CODEX_STREAM_MAX_RETRIES}"
        ),
        "-c",
        (
            f"model_providers.{provider_id}.stream_idle_timeout_ms="
            f"{CODEX_STREAM_IDLE_TIMEOUT_MS}"
        ),
        prompt,
    ]


def _external_script_command(
    config: ExecutorConfig, label: str, arguments: list[str]
) -> list[str]:
    source = {
        "readiness": config.runtime.readiness,
        "deadline": config.runtime.deadline,
        "artifact_builder": config.runtime.artifact_builder,
    }[label]
    return [
        str(config.runtime.sim_python),
        str(Path(__file__).with_name("secure_script.py")),
        "--source",
        str(source),
        "--sha256",
        FROZEN_RUNTIME_SHA256[label],
        "--",
        *arguments,
    ]


def _deadline_script_command(
    config: ExecutorConfig,
    *,
    workdir: Path,
    run_id: str,
    driver_identity: dict[str, int | str],
    arguments: list[str],
) -> list[str]:
    return [
        str(config.runtime.sim_python),
        str(Path(__file__).with_name("deadline_supervisor.py")),
        "--source",
        str(config.runtime.deadline),
        "--sha256",
        FROZEN_RUNTIME_SHA256["deadline"],
        "--workdir",
        str(workdir),
        "--run-id",
        run_id,
        "--driver-pid",
        str(driver_identity["pid"]),
        "--driver-pgid",
        str(driver_identity["pgid"]),
        "--driver-start-time",
        str(driver_identity["start_time_ticks"]),
        "--",
        *arguments,
    ]


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8", errors="strict"))
    except Exception:
        return {}
    return value if isinstance(value, dict) else {}


def _deadline_attestation(
    workdir: Path,
    *,
    planner_status: dict[str, Any],
    termination: str,
    run_id: str,
    driver_identity: dict[str, int | str],
    driver_return_code_before_stop: int | None,
    expected_timeout_seconds: int,
    expected_external_deadline_sha256: str,
) -> tuple[dict[str, Any] | None, str | None]:
    """Validate the commit gate and, for timeout, its frozen journal prefix."""
    try:
        contract, contract_sha256 = _read_trusted_json(workdir / CONTRACT_NAME)
    except Exception as exc:
        return None, f"deadline contract is missing or invalid: {type(exc).__name__}"
    supervisor = planner_status.get("deadline_supervisor")
    expected_driver = {
        key: driver_identity[key] for key in ("pid", "pgid", "start_time_ticks")
    }
    started_ns = contract.get("started_monotonic_ns")
    deadline_ns = contract.get("deadline_monotonic_ns")
    timeout_ns = contract.get("timeout_ns")
    expected_contract_keys = {
        "schema_version",
        "protocol",
        "run_id",
        "nonce",
        "started_monotonic_ns",
        "deadline_monotonic_ns",
        "timeout_ns",
        "driver",
        "external_deadline_sha256",
    }
    expected_supervisor_keys = {
        "protocol",
        "run_id",
        "nonce",
        "fired",
        "contract_sha256",
        "freeze_sha256",
        "error",
    }
    if (
        set(contract) != expected_contract_keys
        or contract.get("schema_version") != 1
        or contract.get("protocol") != DEADLINE_PROTOCOL
        or contract.get("run_id") != run_id
        or not isinstance(contract.get("nonce"), str)
        or len(contract["nonce"]) != 32
        or any(character not in "0123456789abcdef" for character in contract["nonce"])
        or contract.get("driver") != expected_driver
        or contract.get("external_deadline_sha256") != expected_external_deadline_sha256
        or type(started_ns) is not int
        or type(deadline_ns) is not int
        or type(timeout_ns) is not int
        or timeout_ns != expected_timeout_seconds * 1_000_000_000
        or deadline_ns != started_ns + timeout_ns
        or not isinstance(supervisor, dict)
        or set(supervisor) != expected_supervisor_keys
        or supervisor.get("protocol") != DEADLINE_PROTOCOL
        or supervisor.get("run_id") != run_id
        or supervisor.get("nonce") != contract.get("nonce")
        or supervisor.get("contract_sha256") != contract_sha256
        or supervisor.get("error") is not None
    ):
        return None, "deadline contract disagrees with the planner supervisor"

    timed_out = termination == "planner_timeout"
    if supervisor.get("fired") is not timed_out:
        return None, "deadline fired state disagrees with planner termination"
    result = {
        "protocol": DEADLINE_PROTOCOL,
        "contract_sha256": contract_sha256,
        "seal_sha256": None,
        "freeze_sha256": None,
        "deadline_monotonic_ns": deadline_ns,
        "timeout_ns": timeout_ns,
        "driver": expected_driver,
        "driver_stop": None,
        "driver_return_code_before_stop": None,
        "prefix_step": None,
        "success_at_deadline": None,
    }
    if not timed_out:
        if supervisor.get("freeze_sha256") is not None:
            return None, "completed planner unexpectedly has deadline freeze evidence"
        for name in (SEAL_NAME, FREEZE_NAME):
            if (workdir / name).exists() or (workdir / name).is_symlink():
                return None, "completed planner has a sealed deadline gate"
        return result, None

    try:
        seal, seal_sha256 = _read_trusted_json(workdir / SEAL_NAME)
        freeze, freeze_sha256 = _read_trusted_json(workdir / FREEZE_NAME)
        current_prefix = scan_committed_prefix(workdir)
    except Exception as exc:
        return None, f"deadline freeze is missing or invalid: {type(exc).__name__}"
    frozen_ns = freeze.get("frozen_monotonic_ns")
    sealed_ns = seal.get("sealed_monotonic_ns")
    prefix_fields = {
        "prefix_step",
        "success",
        "final_state_sha256",
        "command_trace_sha256",
        "raw_command_trace_sha256",
        "files_sha256",
    }
    expected_seal_keys = {
        "schema_version",
        "protocol",
        "run_id",
        "nonce",
        "deadline_monotonic_ns",
        "sealed_monotonic_ns",
        "contract_sha256",
    }
    expected_freeze_keys = {
        "schema_version",
        "protocol",
        "run_id",
        "nonce",
        "deadline_monotonic_ns",
        "frozen_monotonic_ns",
        "contract_sha256",
        "seal_sha256",
        "driver",
        "driver_stop",
        "dropped_done_steps",
        *prefix_fields,
    }
    driver_stop = freeze.get("driver_stop")
    dropped_done_steps = freeze.get("dropped_done_steps")
    if (
        supervisor.get("freeze_sha256") != freeze_sha256
        or set(seal) != expected_seal_keys
        or seal.get("schema_version") != 1
        or seal.get("protocol") != DEADLINE_PROTOCOL
        or seal.get("run_id") != run_id
        or seal.get("nonce") != contract.get("nonce")
        or seal.get("contract_sha256") != contract_sha256
        or seal.get("deadline_monotonic_ns") != deadline_ns
        or type(sealed_ns) is not int
        or sealed_ns < deadline_ns
        or set(freeze) != expected_freeze_keys
        or freeze.get("schema_version") != 1
        or freeze.get("protocol") != DEADLINE_PROTOCOL
        or freeze.get("run_id") != run_id
        or freeze.get("nonce") != contract.get("nonce")
        or freeze.get("driver") != expected_driver
        or freeze.get("contract_sha256") != contract_sha256
        or freeze.get("seal_sha256") != seal_sha256
        or freeze.get("deadline_monotonic_ns") != deadline_ns
        or type(frozen_ns) is not int
        or frozen_ns < sealed_ns
        or driver_stop_problem(driver_stop) is not None
        or not isinstance(dropped_done_steps, list)
        or any(type(step) is not int for step in dropped_done_steps)
        or dropped_done_steps != sorted(set(dropped_done_steps))
        or any(step <= current_prefix["prefix_step"] for step in dropped_done_steps)
        or any(freeze.get(key) != current_prefix.get(key) for key in prefix_fields)
    ):
        return None, "deadline freeze does not match the sealed committed prefix"
    result.update(
        {
            "freeze_sha256": freeze_sha256,
            "seal_sha256": seal_sha256,
            "driver_stop": driver_stop,
            "driver_return_code_before_stop": driver_return_code_before_stop,
            "prefix_step": freeze["prefix_step"],
            "success_at_deadline": freeze["success"],
        }
    )
    return result, None


def _archive_workdir(workdir: Path, destination_descriptor: int) -> dict[str, str]:
    allowed_prefixes = (
        "state_",
        "log_",
        "done_",
        "camera_meta",
        "agent.",
        "driver.",
        "planner_",
        "driver_ready_",
        "_agent_audit",
        "command_trace",
        "_deadline_",
    )
    archived: dict[str, str] = {}
    for path in workdir.iterdir():
        if (
            path.is_file()
            and not path.is_symlink()
            and path.name.startswith(allowed_prefixes)
        ):
            data = path.read_bytes()
            atomic_write_artifact_file(destination_descriptor, path.name, data)
            archived[path.name] = hashlib.sha256(data).hexdigest()
    return archived


def _remove_workdir(workdir: Path, rollout_root: Path) -> None:
    resolved = workdir.resolve()
    resolved.relative_to(rollout_root.resolve())
    if resolved == rollout_root.resolve():
        raise RuntimeError("refusing to remove the rollout root")
    shutil.rmtree(resolved)


def _prepare_rollout_parent(path: Path, rollout_root: Path) -> None:
    """Create root-owned traverse-only intermediate rollout directories."""
    root = rollout_root.resolve()
    resolved = path.resolve(strict=False)
    try:
        relative = resolved.relative_to(root)
    except ValueError as exc:
        raise RuntimeError(f"rollout parent escapes rollout root: {path}") from exc
    current = root
    for component in relative.parts:
        current = current / component
        try:
            current.mkdir(mode=0o711)
        except FileExistsError:
            pass
        metadata = current.lstat()
        if (
            current.is_symlink()
            or not stat.S_ISDIR(metadata.st_mode)
            or metadata.st_uid != 0
            or stat.S_IMODE(metadata.st_mode) != 0o711
        ):
            raise RuntimeError(f"unsafe rollout parent: {current}")


def execute_cell(config: ExecutorConfig, cell: Cell, *, gpu: int) -> Execution:
    """Run one cell and publish a completion manifest only after validation."""
    runtime = config.runtime
    checkpoint_attestation = _load_checkpoint_attestation(config)
    run_manifest = ensure_execution_run_manifest(
        config, checkpoint_attestation=checkpoint_attestation
    )
    run_config_sha256 = run_manifest["run_config_sha256"]
    with secure_artifact_directory(config.results_root, cell, create=True) as (
        cell_dir,
        _cell_descriptor,
    ):
        pass
    run_id = f"{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}_{uuid.uuid4().hex[:12]}"
    workdir_parent = config.rollout_root / cell.split.value / cell.tag
    _prepare_rollout_parent(workdir_parent, config.rollout_root)
    workdir = workdir_parent / run_id
    workdir.mkdir(mode=0o700)
    os.chown(workdir, 0, 0)
    os.chmod(workdir, 0o700)
    memory_available = _stage_memory(config, cell, workdir)
    prompt = render_formal_prompt(
        task=cell.task, workdir=str(workdir), task_memory_dir="task_memory"
    )
    (workdir / "_prompt.md").write_text(prompt + "\n", encoding="utf-8")
    os.chmod(workdir / "_prompt.md", 0o600)

    driver_log_path = workdir / "driver.log"
    driver_log = driver_log_path.open("wb")
    try:
        driver = subprocess.Popen(
            [
                str(runtime.sim_python),
                str(Path(__file__).with_name("preliminary_driver.py")),
                "--driver-source",
                str(runtime.driver),
                "--driver-sha256",
                FROZEN_RUNTIME_SHA256["driver"],
                "--interactive-env-sha256",
                FROZEN_RUNTIME_SHA256["interactive_env"],
                "--rldx-skill-sha256",
                FROZEN_RUNTIME_SHA256["rldx_skill"],
                "--env",
                cell.task,
                "--split",
                "target",
                "--seed",
                str(cell.seed),
                "--workdir",
                str(workdir),
                "--run-id",
                run_id,
            ],
            cwd=runtime.root,
            env=_driver_environment(config, gpu, cell),
            stdout=driver_log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    except BaseException:
        driver_log.close()
        raise
    termination = "driver_startup_failed"
    planner_rc = EXIT_RETRYABLE_INFRA
    planner_status: dict[str, Any] = {}
    deadline_attestation: dict[str, Any] | None = None
    driver_return_code_before_stop: int | None = None
    observed_driver = process_identity(driver.pid)
    driver_identity = (
        {key: observed_driver[key] for key in ("pid", "pgid", "start_time_ticks")}
        if observed_driver is not None
        else None
    )
    try:
        startup_rc = (
            subprocess.run(
                _external_script_command(
                    config,
                    "readiness",
                    [
                        "--pid",
                        str(driver.pid),
                        "--workdir",
                        str(workdir),
                        "--timeout-seconds",
                        str(config.startup_timeout_seconds),
                        "--status",
                        str(workdir / "driver_ready_status.json"),
                    ],
                ),
                cwd=runtime.root,
                check=False,
            ).returncode
            if driver_identity is not None
            else EXIT_RETRYABLE_INFRA
        )
        if startup_rc == 0 and driver_identity is not None:
            planner = _planner_command(
                config, workdir=workdir, run_id=run_id, cell=cell
            )
            planner_rc = subprocess.run(
                _deadline_script_command(
                    config,
                    workdir=workdir,
                    run_id=run_id,
                    driver_identity=driver_identity,
                    arguments=[
                        "--timeout-seconds",
                        str(SPLITS[cell.split].timeout_seconds),
                        "--kill-after-seconds",
                        str(config.kill_after_seconds),
                        "--stdout",
                        str(workdir / "agent.log"),
                        "--stderr",
                        str(workdir / "agent.stderr.log"),
                        "--status",
                        str(workdir / "planner_status.json"),
                        "--",
                        *planner,
                    ],
                ),
                cwd=runtime.root,
                check=False,
            ).returncode
            planner_status = _read_json(workdir / "planner_status.json")
            termination = str(
                planner_status.get("termination_cause") or "wrapper_error"
            )
            driver_return_code_before_stop = _observe_driver_exit(
                driver,
                termination=termination,
                timeout_seconds=config.kill_after_seconds,
            )
        else:
            planner_rc = startup_rc
    finally:
        _stop_process(driver)
        driver_log.close()

    deadline_problem = None
    if termination in {"planner_completed", "planner_timeout"}:
        if driver_identity is None:
            deadline_problem = "driver identity was unavailable"
        else:
            deadline_attestation, deadline_problem = _deadline_attestation(
                workdir,
                planner_status=planner_status,
                termination=termination,
                run_id=run_id,
                driver_identity=driver_identity,
                driver_return_code_before_stop=driver_return_code_before_stop,
                expected_timeout_seconds=SPLITS[cell.split].timeout_seconds,
                expected_external_deadline_sha256=FROZEN_RUNTIME_SHA256["deadline"],
            )
        if deadline_problem is not None:
            termination = "wrapper_error"
    supervised_timeout = (
        termination == "planner_timeout" and deadline_attestation is not None
    )
    timeout_driver_exit_expected = supervised_timeout and _timeout_driver_exit_expected(
        deadline_attestation
    )
    driver_exit_expected = (
        timeout_driver_exit_expected
        if supervised_timeout
        else _latched_success_exit(workdir, driver_return_code_before_stop)
    )
    unexpected_driver_exit = (
        not driver_exit_expected
        if supervised_timeout
        else driver_return_code_before_stop is not None and not driver_exit_expected
    )
    if unexpected_driver_exit and termination != "planner_protocol_error":
        termination = "driver_runtime_failed"
    if _navview_rollout_problem(workdir) is not None:
        termination = "driver_runtime_failed"

    build_dir = workdir / "_built_artifacts"
    build_dir.mkdir(mode=0o700)
    builder_termination = (
        termination if termination in BUILDER_TERMINATIONS else "planner_failed"
    )
    build = _external_script_command(
        config,
        "artifact_builder",
        [
            "--workdir",
            str(workdir),
            "--output-dir",
            str(build_dir),
            "--task",
            cell.task,
            "--seed",
            str(cell.seed),
            "--agent-rc",
            str(planner_rc),
            "--termination-cause",
            builder_termination,
        ],
    )
    if not memory_available:
        build.append("--task-memory-missing")
    builder_rc = subprocess.run(build, cwd=runtime.root, check=False).returncode
    with secure_artifact_subdirectory(
        config.results_root, cell, "run_logs", cell.tag, run_id
    ) as (log_dir, log_descriptor):
        archived = _archive_workdir(workdir, log_descriptor)
    evidence_names = {
        "agent_log": "agent.log",
        "planner_status": "planner_status.json",
    }
    if termination in {"planner_completed", "planner_timeout"}:
        evidence_names["deadline_contract"] = CONTRACT_NAME
    if termination == "planner_timeout":
        evidence_names["deadline_seal"] = SEAL_NAME
        evidence_names["deadline_freeze"] = FREEZE_NAME
    evidence_ready = all(name in archived for name in evidence_names.values())
    evidence = {
        label: {
            "path": str((log_dir / name).relative_to(cell_dir)),
            "sha256": archived[name],
        }
        for label, name in evidence_names.items()
        if name in archived
    }

    audit_path = build_dir / f"{cell.tag}.json"
    trace_path = build_dir / f"{cell.tag}.jsonl"
    audit = _read_json(audit_path)
    valid = builder_rc == 0 and audit.get("valid") is True and evidence_ready
    if valid and termination == "planner_timeout":
        assert deadline_attestation is not None
        try:
            built_trace_sha256 = sha256_file(trace_path)
        except OSError:
            built_trace_sha256 = None
        valid = (
            audit.get("steps") == deadline_attestation["prefix_step"]
            and audit.get("success") is deadline_attestation["success_at_deadline"]
            and built_trace_sha256
            == _read_json(workdir / FREEZE_NAME).get("command_trace_sha256")
        )
    canonical_termination = termination in {"planner_completed", "planner_timeout"}
    if valid and canonical_termination:
        audit.update(
            {
                "protocol_id": PROTOCOL_ID,
                "split": cell.split.value,
                "planner_profile": config.planner_profile.name,
                "planner_backend": config.planner_profile.backend,
                "planner_model": config.planner_profile.model,
                "planner_reasoning_effort": config.planner_profile.reasoning_effort,
                "planner_auth_mode": config.planner_transport.auth_mode,
                "planner_provider": config.planner_transport.provider_id,
                "planner_endpoint_identity": (
                    config.planner_transport.endpoint_identity
                ),
                "infra_status": "ok",
                "planner_status": (
                    "timeout" if termination == "planner_timeout" else "completed"
                ),
                "environment_success": audit.get("success") is True,
                "preliminary": True,
                "release_ready": False,
                "run_config_sha256": run_config_sha256,
                "run_id": run_id,
                "runtime_root": str(runtime.root),
                "runtime_scripts": {
                    "external_driver": sha256_file(runtime.driver),
                    "external_deadline": sha256_file(runtime.deadline),
                    "driver_adapter": sha256_file(
                        Path(__file__).with_name("preliminary_driver.py")
                    ),
                    "deadline_supervisor": sha256_file(
                        Path(__file__).with_name("deadline_supervisor.py")
                    ),
                    "artifact_builder": sha256_file(runtime.artifact_builder),
                },
                "deadline": deadline_attestation,
                "global_memory": False,
                "task_notes_available": False,
                "reset_seed": _reset_seed(cell),
                "memory_source": {
                    "kind": config.memory_source,
                    "revision": config.memory_revision,
                    "manifest_sha256": sha256_file(
                        config.memory_root / "manifest.json"
                    ),
                },
                "navview": {
                    "xml_sha256": sha256_file(runtime.navview_xml),
                    "camera": "mobilebase0_navview",
                    "mode": "fixed",
                    "pos": "0.2 0 1.6",
                    "xyaxes": "0 -1 0 0.643 0 0.766",
                    "fovy": "75",
                },
                "checkpoint": {
                    "checkpoint_id": checkpoint_attestation["checkpoint_id"],
                    "authority_manifest_sha256": checkpoint_attestation[
                        "authority_manifest_sha256"
                    ],
                    "fingerprint": checkpoint_attestation["fingerprint"],
                    "files": {
                        name: value["sha256"]
                        for name, value in checkpoint_attestation["files"].items()
                    },
                },
                "checkpoint_sha256": {
                    name: checkpoint_attestation["files"][f"model/{name}"]["sha256"]
                    for name in CHECKPOINT_SHA256
                },
                "evidence": evidence,
            }
        )
        commands = [
            json.loads(line)
            for line in trace_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        result = CellResult(
            Completion.COMPLETED,
            Outcome.SUCCESS if audit["success"] else Outcome.FAILURE,
            Integrity.VALID,
        )
        try:
            publish_completed_cell(config.results_root, cell, result, audit, commands)
        except (OSError, ValueError):
            rc = EXIT_ARTIFACT
            message = "artifact publication rejected the completed cell"
        else:
            from .validator import validate_cell

            validation = validate_cell(
                config.results_root,
                cell,
                expected_run_config_sha256=run_config_sha256,
            )
            if validation.result.canonical:
                rc = EXIT_CANONICAL
                message = None
            else:
                rc = EXIT_ARTIFACT
                message = "published cell failed canonical validation"
    elif termination in RETRYABLE_TERMINATIONS:
        rc = EXIT_RETRYABLE_INFRA
        message = (
            termination
            if builder_rc == 0
            else f"{termination}; artifact builder rc={builder_rc}"
        )
    elif termination == "configuration_failed":
        rc = EXIT_CONFIGURATION
        message = termination
    elif builder_rc != 0:
        rc = EXIT_ARTIFACT
        message = f"artifact builder failed with rc={builder_rc}"
    elif not evidence_ready and canonical_termination:
        rc = EXIT_ARTIFACT
        message = "canonical attempt is missing agent/status evidence"
    else:
        rc = EXIT_ARTIFACT
        message = f"non-canonical termination: {termination}"

    if not config.keep_workdirs:
        _remove_workdir(workdir, config.rollout_root)
    return Execution(cell, rc, termination, rc == EXIT_CANONICAL, workdir, message)
