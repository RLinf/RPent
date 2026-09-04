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

"""Raw BEHAVIOR success helpers.

Official task success is only the exact boolean at ``info["done"]["success"]``.
Later snapshots, visual state, videos, and planner ``finish`` status do not
create or revoke that bit.
"""

from __future__ import annotations

import copy
import hashlib
import hmac
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class TerminalReceiptValidation:
    """Result of validating one output-bound official-success receipt."""

    valid: bool
    terminal_image_path: Path | None = None
    reason: str | None = None


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def official_success_receipt_sha256(receipt: Mapping[str, Any]) -> str:
    """Return the canonical digest after excluding ``receipt_sha256``."""

    material = {key: value for key, value in receipt.items() if key != "receipt_sha256"}
    return hashlib.sha256(_canonical_json_bytes(material)).hexdigest()


def validate_official_success_receipt(value: Any) -> dict[str, Any] | None:
    """Return a validated official-success receipt copy, or ``None``."""

    if not isinstance(value, Mapping):
        return None
    receipt = dict(value)
    schema_version = receipt.get("schema_version")
    raw_done = receipt.get("raw_done")
    env_step = receipt.get("env_step")
    digest = receipt.get("receipt_sha256")
    if (
        type(schema_version) is not int
        or schema_version != 1
        or receipt.get("source") != 'info["done"]["success"]'
        or not isinstance(raw_done, Mapping)
        or raw_done.get("success") is not True
        or isinstance(env_step, bool)
        or not isinstance(env_step, int)
        or env_step < 0
        or not isinstance(digest, str)
    ):
        return None
    try:
        expected = official_success_receipt_sha256(receipt)
    except (TypeError, ValueError):
        return None
    if not hmac.compare_digest(digest, expected):
        return None
    return copy.deepcopy(receipt)


def official_task_success(info: Any) -> bool:
    """Return only the raw official BEHAVIOR success bit."""

    done = info.get("done") if isinstance(info, dict) else None
    value = done.get("success") if isinstance(done, dict) else None
    return isinstance(value, (bool, np.bool_)) and bool(value)


def official_success_receipt_from_info(info: Any) -> dict[str, Any] | None:
    """Extract a trusted runtime receipt if one is present and source-matched."""

    runtime = info.get("_rpent") if isinstance(info, dict) else None
    if not isinstance(runtime, dict):
        return None
    return validate_official_success_receipt(runtime.get("official_success_receipt"))


def make_raw_success_receipt(
    info: Any, *, env_step: int | None = None
) -> dict[str, Any] | None:
    """Return a deterministic receipt for raw success when the env did not provide one."""

    if not official_task_success(info):
        return None
    runtime = info.get("_rpent") if isinstance(info, dict) else {}
    if isinstance(runtime, dict):
        step_value = runtime.get("total_env_steps", runtime.get("global_env_steps"))
    else:
        step_value = None
    if type(step_value) is not int or step_value < 0:
        if env_step is None:
            step_value = 0
        elif type(env_step) is int and env_step >= 0:
            step_value = env_step
        else:
            return None
    material = {
        "schema_version": 1,
        "source": 'info["done"]["success"]',
        "env_step": step_value,
        "raw_done": {"success": True},
    }
    return {
        **material,
        "receipt_sha256": official_success_receipt_sha256(material),
    }


def _exact_bool_at(record: dict[str, Any], path: tuple[str, ...]) -> bool | None:
    value: Any = record
    for field in path:
        if not isinstance(value, dict) or field not in value:
            return None
        value = value[field]
    return value if type(value) is bool else None


def summarize_action_trace_success(action_trace_bytes: bytes) -> dict[str, Any] | None:
    """Summarize first raw ``info_done.success`` evidence from a JSONL trace."""

    action_trace_sha256 = hashlib.sha256(action_trace_bytes).hexdigest()
    malformed_lines = 0
    observations: list[tuple[int, int | None, bool]] = []
    last_trace_step: int | None = None
    for line_number, line in enumerate(action_trace_bytes.splitlines(), start=1):
        try:
            record = json.loads(line)
        except (UnicodeDecodeError, json.JSONDecodeError):
            malformed_lines += 1
            continue
        if not isinstance(record, dict):
            malformed_lines += 1
            continue
        raw_step = record.get("step")
        step = (
            raw_step
            if isinstance(raw_step, int)
            and not isinstance(raw_step, bool)
            and raw_step >= 0
            else None
        )
        if step is not None:
            last_trace_step = step
        value = _exact_bool_at(record, ("info_done", "success"))
        if value is not None:
            observations.append((line_number, step, value))
    if not any(value is True for _, _, value in observations):
        return None
    first_index = next(
        i for i, (_, _, value) in enumerate(observations) if value is True
    )
    first_line, first_step, _ = observations[first_index]
    success_count = sum(1 for _, _, value in observations if value is True)
    last_success_step = next(
        step for _, step, value in reversed(observations) if value is True
    )
    success_later_reverted = any(
        value is False for _, _, value in observations[first_index + 1 :]
    )
    notes = [f"malformed_json_lines={malformed_lines}"] if malformed_lines else []
    return {
        "source": "behavior_action_trace",
        "field_path": "info_done.success",
        "first_success_line": first_line,
        "first_success_step": first_step,
        "success_count": success_count,
        "success_later_reverted": success_later_reverted,
        "last_success_step": last_success_step,
        "last_trace_step": last_trace_step,
        "action_trace_sha256": action_trace_sha256,
        "receipt_sha256": None,
        "notes": notes,
    }


def validate_terminal_success_receipt(
    *,
    tool_name: str,
    step: Any,
    result: Any,
    output_dir: str | Path,
) -> TerminalReceiptValidation:
    """Validate raw official success without terminal-hold or image gates."""

    del tool_name, output_dir
    if not isinstance(step, int) or isinstance(step, bool) or step < 0:
        return TerminalReceiptValidation(valid=False, reason="invalid trace step")
    if not isinstance(result, Mapping):
        return TerminalReceiptValidation(valid=False, reason="result is not a mapping")
    if result.get("kind") != "behavior_finish_terminal_receipt":
        return TerminalReceiptValidation(valid=False, reason="invalid receipt kind")
    if result.get("_finish") is not True:
        return TerminalReceiptValidation(valid=False, reason="receipt is not terminal")
    if result.get("task_success") is not True:
        return TerminalReceiptValidation(valid=False, reason="task success is not true")
    if result.get("official_success_source") != 'info["done"]["success"]':
        return TerminalReceiptValidation(
            valid=False, reason="invalid official success source"
        )
    if (
        validate_official_success_receipt(result.get("official_success_receipt"))
        is None
    ):
        return TerminalReceiptValidation(
            valid=False, reason="invalid official success receipt"
        )
    return TerminalReceiptValidation(valid=True)


__all__ = [
    "TerminalReceiptValidation",
    "make_raw_success_receipt",
    "official_success_receipt_sha256",
    "official_success_receipt_from_info",
    "official_task_success",
    "summarize_action_trace_success",
    "validate_official_success_receipt",
    "validate_terminal_success_receipt",
]
