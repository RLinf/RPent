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

"""Field access and JSON conversion for SDK events."""

from __future__ import annotations

import dataclasses
import json
from typing import Any


def get_field(value: Any, key: str, default: Any = None) -> Any:
    """Read a field from a mapping or an SDK object."""
    if isinstance(value, dict):
        return value.get(key, default)
    return getattr(value, key, default)


def unwrap(value: Any) -> Any:
    """Unwrap SDK root models."""
    return getattr(value, "root", value)


def jsonable(value: Any, *, by_alias: bool = False) -> Any:
    """Serialize SDK models and dataclasses, summarizing binary payloads."""
    value = unwrap(value)
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json", exclude_none=True, by_alias=by_alias)
    elif dataclasses.is_dataclass(value) and not isinstance(value, type):
        value = dataclasses.asdict(value)
    if isinstance(value, dict):
        return {str(k): jsonable(v, by_alias=by_alias) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [jsonable(v, by_alias=by_alias) for v in value]
    if isinstance(value, bytes):
        return {"type": "bytes", "size": len(value)}
    return value


def json_object(value: Any) -> dict[str, Any] | None:
    """Read a JSON object from a mapping or encoded JSON string."""
    if isinstance(value, dict):
        return value
    if not isinstance(value, str):
        return None
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def int_field(value: Any, key: str) -> int:
    """Read an integer counter, treating absent or null values as zero."""
    return int(get_field(value, key, 0) or 0)
