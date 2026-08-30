"""Small deterministic schema helpers for BEHAVIOR episode memory."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from typing import Any

SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


class MemoryValidationError(ValueError):
    """Fail-closed validation error with a stable code and path."""

    def __init__(self, code: str, path: str, detail: str) -> None:
        super().__init__(f"{code}: {path}: {detail}")
        self.code = code
        self.path = path
        self.detail = detail


def fail(code: str, path: str, detail: str) -> None:
    raise MemoryValidationError(code, path, detail)


def require_sha256(value: Any, *, path: str) -> str:
    if not isinstance(value, str) or SHA256_PATTERN.fullmatch(value) is None:
        fail("MEMORY_SCHEMA_INVALID", path, "expected one lowercase SHA-256 digest")
    return value


def canonical_json_bytes(value: Any, *, path: str = "$") -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        fail("MEMORY_JSON_INVALID", path, f"{type(exc).__name__}: {exc}")


def canonical_json_file_bytes(value: Any, *, path: str = "$") -> bytes:
    return canonical_json_bytes(value, path=path) + b"\n"


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def require_exact_keys(
    value: Mapping[str, Any],
    expected: set[str] | frozenset[str],
    *,
    path: str,
) -> None:
    actual = set(value)
    expected_set = set(expected)
    if actual != expected_set:
        fail(
            "MEMORY_SCHEMA_INVALID",
            path,
            f"expected keys {sorted(expected_set)}, actual {sorted(actual)}",
        )
