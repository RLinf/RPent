"""Small BEHAVIOR run-manifest helpers for the main RPent contract."""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Iterable, Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from robots.behavior.redaction import redact_command as _redact_command
from robots.behavior.redaction import redact_text as _redact_text
from robots.behavior.schemas import (
    BEHAVIOR_TOOL_NAMES,
    CURRENT_PUBLIC_TOOL_CONTRACT_VERSION,
    PUBLIC_TOOL_CONTRACTS,
)

MANIFEST_FILENAME = "run_manifest.json"
LEGACY_RUN_MANIFEST_SCHEMA_VERSION = 5
RUN_MANIFEST_SCHEMA_VERSION = 6
PI0_NAV_PICK_CALL_ARTIFACT_SCHEMA_VERSION = 5


def utc_timestamp() -> str:
    """Return a stable UTC timestamp suitable for machine artifacts."""

    return (
        datetime.now(timezone.utc)
        .isoformat(timespec="milliseconds")
        .replace("+00:00", "Z")
    )


def redact_text(value: str) -> str:
    return _redact_text(value)


def redact_command(command: Iterable[object] | str | None) -> list[str] | None:
    return _redact_command(command)


def resolve_run_manifest_public_tool_contract(
    manifest: Mapping[str, Any],
) -> tuple[int, tuple[str, ...]]:
    """Resolve and validate the declared BEHAVIOR public tool ABI."""

    schema_version = manifest.get("schema_version")
    protocol = manifest.get("protocol")
    if not isinstance(protocol, Mapping):
        raise ValueError("run manifest protocol is missing")
    declared_version = protocol.get("public_tool_contract_version")
    declared_tools = tuple(protocol.get("public_primitives") or ())

    if schema_version == LEGACY_RUN_MANIFEST_SCHEMA_VERSION:
        if declared_version is not None:
            raise ValueError("legacy schema must not declare public_tool_contract_version")
        version = 1
    elif schema_version == RUN_MANIFEST_SCHEMA_VERSION:
        if (
            isinstance(declared_version, bool)
            or not isinstance(declared_version, int)
            or declared_version not in PUBLIC_TOOL_CONTRACTS
        ):
            raise ValueError("schema-6 manifest must declare a supported contract")
        version = int(declared_version)
    else:
        raise ValueError(f"unsupported run manifest schema: {schema_version!r}")

    expected = PUBLIC_TOOL_CONTRACTS[version]
    if declared_tools != expected:
        raise ValueError(f"run manifest public primitives do not match v{version}")
    return version, expected


def pi0_nav_pick_exact_chunk_contract() -> dict[str, Any]:
    """Return the public ABI for one BEHAVIOR Pi0 invocation."""

    return {
        "call_artifact_schema_version": PI0_NAV_PICK_CALL_ARTIFACT_SCHEMA_VERSION,
        "chunks_argument": {
            "name": "chunks",
            "required": True,
            "minimum": 1,
            "maximum": None,
        },
        "action_shape": [None, 23],
        "normal_completion": "exact_requested_chunks",
        "raw_success_behavior": "stop_after_success_env_step",
        "official_success_completion": {
            "task_success": True,
            "primitive_success": True,
            "stop_reason": "official_task_success",
            "post_success_env_actions": 0,
        },
    }


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True, ensure_ascii=False)
            stream.write("\n")
        os.replace(temporary_name, path)
    finally:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass


class RunManifest:
    """Minimal idempotent JSON manifest writer used by runtime glue."""

    def __init__(
        self,
        output_dir: str | Path,
        *,
        task_desc: Mapping[str, Any] | None = None,
        command: Iterable[object] | str | None = None,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.path = self.output_dir / MANIFEST_FILENAME
        self._payload: dict[str, Any] = {
            "schema_version": RUN_MANIFEST_SCHEMA_VERSION,
            "created_at": utc_timestamp(),
            "updated_at": utc_timestamp(),
            "task": dict(task_desc or {}),
            "command": redact_command(command),
            "protocol": {
                "public_tool_contract_version": CURRENT_PUBLIC_TOOL_CONTRACT_VERSION,
                "public_primitives": list(BEHAVIOR_TOOL_NAMES),
                "official_success_path": ["info", "done", "success"],
                "pi0_nav_pick": pi0_nav_pick_exact_chunk_contract(),
            },
            "events": [],
        }
        self.write()

    @property
    def payload(self) -> dict[str, Any]:
        return json.loads(json.dumps(self._payload, default=str))

    def event(self, name: str, **fields: Any) -> dict[str, Any]:
        entry = {"name": name, "at": utc_timestamp(), **fields}
        self._payload.setdefault("events", []).append(entry)
        self._payload["updated_at"] = entry["at"]
        self.write()
        return entry

    def finish(self, **fields: Any) -> dict[str, Any]:
        return self.event("finish", **fields)

    def write(self) -> Path:
        _atomic_write_json(self.path, self._payload)
        return self.path


__all__ = [
    "LEGACY_RUN_MANIFEST_SCHEMA_VERSION",
    "MANIFEST_FILENAME",
    "PI0_NAV_PICK_CALL_ARTIFACT_SCHEMA_VERSION",
    "RUN_MANIFEST_SCHEMA_VERSION",
    "RunManifest",
    "pi0_nav_pick_exact_chunk_contract",
    "redact_command",
    "redact_text",
    "resolve_run_manifest_public_tool_contract",
    "utc_timestamp",
]
