"""Credential-safe serialization helpers for BEHAVIOR artifacts."""

from __future__ import annotations

import re
import shlex
from collections.abc import Iterable
from typing import Any

REDACTED = "[REDACTED]"
_SENSITIVE_NAME = re.compile(
    r"(?:^|[-_.])(?:api[-_.]?key|token|secret|password|passwd|credential|"
    r"auth|authorization|proxy[-_.]?authorization)(?:$|[-_.])",
    re.IGNORECASE,
)
_SENSITIVE_ASSIGNMENT = re.compile(
    r"(?P<name>[A-Za-z0-9_.-]*(?:api[-_.]?key|token|secret|password|passwd|"
    r"credential|auth)[A-Za-z0-9_.-]*)=(?P<value>[^\s&]+)",
    re.IGNORECASE,
)
_URL_USERINFO = re.compile(r"(?P<scheme>https?://)[^/@\s]+@", re.IGNORECASE)
_AUTH_HEADER = re.compile(
    r"(?P<name>(?:proxy-)?authorization)\s*:\s*"
    r"(?P<scheme>bearer|basic)\s+[^\s,;]+",
    re.IGNORECASE,
)
_AUTH_SCHEME = re.compile(
    r"\b(?P<scheme>bearer|basic)\s+[A-Za-z0-9._~+/=-]{8,}", re.IGNORECASE
)


def _is_sensitive_flag(value: str) -> bool:
    return bool(_SENSITIVE_NAME.search(value.lstrip("-")))


def redact_text(value: str) -> str:
    """Remove URL userinfo and common credential assignments from text."""

    value = _URL_USERINFO.sub(r"\g<scheme>[REDACTED]@", str(value))
    value = _SENSITIVE_ASSIGNMENT.sub(
        lambda match: f"{match.group('name')}={REDACTED}", value
    )
    value = _AUTH_HEADER.sub(lambda match: f"{match.group('name')}: {REDACTED}", value)
    return _AUTH_SCHEME.sub(lambda match: f"{match.group('scheme')} {REDACTED}", value)


def redact_command(command: Iterable[object] | str | None) -> list[str] | None:
    """Return credential-redacted argv without changing the executed command."""

    if command is None:
        return None
    if isinstance(command, str):
        try:
            argv = shlex.split(command)
        except ValueError:
            argv = [command]
    else:
        argv = [str(value) for value in command]
    redacted: list[str] = []
    redact_next = False
    for argument in argv:
        if redact_next:
            redacted.append(REDACTED)
            redact_next = False
            continue
        if argument.startswith("-") and "=" in argument:
            name, _ = argument.split("=", 1)
            redacted.append(
                f"{name}={REDACTED}"
                if _is_sensitive_flag(name)
                else redact_text(argument)
            )
            continue
        redacted.append(redact_text(argument))
        if argument.startswith("-") and _is_sensitive_flag(argument):
            redact_next = True
    return redacted


def redact_value(value: Any) -> Any:
    """Recursively redact sensitive fields and strings before persistence."""

    if isinstance(value, dict):
        return {
            str(key): REDACTED if _is_sensitive_flag(str(key)) else redact_value(item)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [redact_value(item) for item in value]
    if isinstance(value, tuple):
        return tuple(redact_value(item) for item in value)
    if isinstance(value, str):
        return redact_text(value)
    return value


__all__ = ["REDACTED", "redact_command", "redact_text", "redact_value"]
