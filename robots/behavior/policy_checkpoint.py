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

"""Identity contract for the shared BEHAVIOR Pi0.5 checkpoint."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

POLICY_CHECKPOINT_BINDING_SCHEMA_VERSION = 1
POLICY_CHECKPOINT_ENV = "PI05_CHECKPOINT_PATH"
PUBLIC_POLICY_REPOSITORY = "RLinf/RLinf-Pi05-BEHAVIOR-1K-PT50-CS32"
SHARED_POLICY_PROFILE_ID = "pi05-b1kpt50-cs32"
SHARED_POLICY_CHECKPOINT_PATH = Path(
    os.environ.get(POLICY_CHECKPOINT_ENV, SHARED_POLICY_PROFILE_ID)
)


class PolicyCheckpointError(ValueError):
    """Raised when a checkpoint violates the shared BEHAVIOR contract."""


@dataclass(frozen=True)
class CheckpointFileRequirement:
    relative_path: str
    size_bytes: int
    sha256: str


@dataclass(frozen=True)
class PolicyCheckpointProfile:
    profile_id: str
    path: Path
    files: tuple[CheckpointFileRequirement, ...]


@dataclass(frozen=True)
class PolicyCheckpointBinding:
    schema_version: int
    profile_id: str
    resolved_path: str
    files: tuple[CheckpointFileRequirement, ...]
    binding_sha256: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "profile_id": self.profile_id,
            "resolved_path": self.resolved_path,
            "files": {
                item.relative_path: {
                    "size_bytes": item.size_bytes,
                    "sha256": item.sha256,
                }
                for item in self.files
            },
            "binding_sha256": self.binding_sha256,
        }


SHARED_POLICY_PROFILE = PolicyCheckpointProfile(
    profile_id=SHARED_POLICY_PROFILE_ID,
    path=SHARED_POLICY_CHECKPOINT_PATH,
    files=(
        CheckpointFileRequirement(
            relative_path="model.safetensors",
            size_bytes=7_233_650_408,
            sha256="7e257666d835f6af701de493676a6c86a0421b2efc737a0f911d782b7a09f635",
        ),
        CheckpointFileRequirement(
            relative_path="config.json",
            size_bytes=149,
            sha256="a4ae208203adfdd64c5fdbd4b0dc257e4ebbc82e464cb146dd0377051b25fc0a",
        ),
        CheckpointFileRequirement(
            relative_path="assets/behavior-1k/2025-challenge-demos/norm_stats.json",
            size_bytes=6_368,
            sha256="d66ed16830a98f90dde8a315058b4a0df59f5e05734c1686d8b3f66787d0a929",
        ),
    ),
)


def _canonical_sha256(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("utf-8")
    ).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _binding_payload(
    profile: PolicyCheckpointProfile,
    resolved_path: Path,
) -> dict[str, Any]:
    return {
        "schema_version": POLICY_CHECKPOINT_BINDING_SCHEMA_VERSION,
        "profile_id": profile.profile_id,
        "resolved_path": str(resolved_path),
        "files": {
            item.relative_path: {
                "size_bytes": item.size_bytes,
                "sha256": item.sha256,
            }
            for item in profile.files
        },
    }


def validate_policy_checkpoint(
    path: str | Path = SHARED_POLICY_CHECKPOINT_PATH,
) -> PolicyCheckpointBinding:
    """Verify and bind the expected Pi05-Behavior checkpoint files."""

    profile = SHARED_POLICY_PROFILE
    requested = Path(path).expanduser()
    try:
        resolved = requested.resolve(strict=True)
    except OSError as error:
        raise PolicyCheckpointError(
            f"your Pi05-Behavior model checkpoint is unavailable: {error}"
        ) from error
    if not resolved.is_dir():
        raise PolicyCheckpointError(
            f"your Pi05-Behavior model checkpoint is not a directory: {resolved}"
        )
    for requirement in profile.files:
        candidate = resolved / requirement.relative_path
        if candidate.is_symlink() or not candidate.is_file():
            raise PolicyCheckpointError(
                "your Pi05-Behavior model checkpoint file is missing or unsafe: "
                f"{candidate}"
            )
        size = candidate.stat().st_size
        if size != requirement.size_bytes:
            raise PolicyCheckpointError(
                "your Pi05-Behavior model checkpoint size mismatch for "
                f"{requirement.relative_path}: expected {requirement.size_bytes}, "
                f"got {size}"
            )
        actual_sha256 = _file_sha256(candidate)
        if actual_sha256 != requirement.sha256:
            raise PolicyCheckpointError(
                "your Pi05-Behavior model checkpoint SHA256 mismatch for "
                f"{requirement.relative_path}: expected {requirement.sha256}, "
                f"got {actual_sha256}"
            )
    payload = _binding_payload(profile, resolved)
    return PolicyCheckpointBinding(
        schema_version=POLICY_CHECKPOINT_BINDING_SCHEMA_VERSION,
        profile_id=profile.profile_id,
        resolved_path=str(resolved),
        files=profile.files,
        binding_sha256=_canonical_sha256(payload),
    )


def assert_matching_policy_checkpoint_binding(
    actual: Mapping[str, Any] | None,
    expected: PolicyCheckpointBinding | Mapping[str, Any],
) -> dict[str, Any]:
    expected_value = (
        expected.as_dict()
        if isinstance(expected, PolicyCheckpointBinding)
        else dict(expected)
    )
    if not isinstance(actual, Mapping):
        raise PolicyCheckpointError("VLA health metadata lacks checkpoint_binding")
    actual_value = dict(actual)
    if actual_value != expected_value:
        raise PolicyCheckpointError(
            "VLA checkpoint binding does not match your Pi05-Behavior model"
        )
    return actual_value


__all__ = [
    "POLICY_CHECKPOINT_BINDING_SCHEMA_VERSION",
    "POLICY_CHECKPOINT_ENV",
    "PUBLIC_POLICY_REPOSITORY",
    "SHARED_POLICY_CHECKPOINT_PATH",
    "SHARED_POLICY_PROFILE",
    "SHARED_POLICY_PROFILE_ID",
    "CheckpointFileRequirement",
    "PolicyCheckpointBinding",
    "PolicyCheckpointError",
    "PolicyCheckpointProfile",
    "assert_matching_policy_checkpoint_binding",
    "validate_policy_checkpoint",
]
