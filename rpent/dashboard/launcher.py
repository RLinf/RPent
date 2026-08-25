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

"""Small adapter between the dashboard launch form and CLI args."""

from __future__ import annotations

import math
from typing import Any

from rpent.dashboard.spec import LauncherFieldSpec

_PLANNERS = {"api", "claude_code", "codex"}


def _parse_integer(
    raw: Any,
    *,
    label: str,
    minimum: int | None = None,
    required: bool = False,
) -> int | None:
    if raw in (None, ""):
        if required:
            raise ValueError(f"{label} is required")
        return None
    if isinstance(raw, bool):
        raise ValueError(f"{label} must be an integer")
    try:
        value = int(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be an integer") from exc
    if isinstance(raw, float) and not raw.is_integer():
        raise ValueError(f"{label} must be an integer")
    if minimum is not None and value < minimum:
        raise ValueError(f"{label} must be at least {minimum}")
    return value


def _parse_float(
    raw: Any,
    *,
    label: str,
    minimum: float | None = None,
) -> float | None:
    if raw in (None, ""):
        return None
    if isinstance(raw, bool):
        raise ValueError(f"{label} must be a number")
    try:
        value = float(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be a number") from exc
    if not math.isfinite(value):
        raise ValueError(f"{label} must be a finite number")
    if minimum is not None and value < minimum:
        raise ValueError(f"{label} must be at least {minimum:g}")
    return value


def defaults_from_args(
    args: Any,
    launcher_fields: tuple[LauncherFieldSpec, ...],
) -> dict[str, Any]:
    """Build form defaults for the selected planner."""
    defaults = {
        "planner": args.planner,
        "model": args.model,
        "max-turns": args.max_turns,
        "planner-timeout-s": args.planner_timeout_s,
        "reasoning-effort": args.reasoning_effort,
        "claude-code-max-budget-usd": args.claude_code_max_budget_usd,
        "no-images": args.no_images,
    }
    defaults.update(
        {field["name"]: getattr(args, field["name"]) for field in launcher_fields}
    )
    return defaults


def parse_launcher_fields(
    payload: dict[str, Any],
    launcher_fields: tuple[LauncherFieldSpec, ...],
) -> dict[str, Any]:
    """Parse environment launcher values submitted by the user."""
    values: dict[str, Any] = {}
    for field in launcher_fields:
        name = field["name"]
        raw = payload.get(name)
        if field["kind"] == "string":
            if raw in (None, ""):
                if field.get("required", False):
                    raise ValueError(f"{field['label']} is required")
                values[name] = None
                continue
            if not isinstance(raw, str):
                raise ValueError(f"{field['label']} must be a string")
            value = raw
        else:
            value = _parse_integer(
                raw,
                label=field["label"],
                minimum=field.get("minimum"),
                required=field.get("required", False),
            )
        values[name] = value
    return values


def parse_launcher_config(
    payload: dict[str, Any],
    launcher_fields: tuple[LauncherFieldSpec, ...],
) -> dict[str, Any]:
    """Parse common and robot-owned launcher values."""
    config = dict(payload)
    planner = config.get("planner")
    if not isinstance(planner, str) or planner not in _PLANNERS:
        raise ValueError(f"Planner must be one of: {', '.join(sorted(_PLANNERS))}")
    config["planner"] = planner

    model = config.get("model")
    if model in (None, ""):
        model = None
    elif not isinstance(model, str):
        raise ValueError("Model must be a string")
    else:
        model = model.strip() or None
    if planner == "api":
        provider, separator, model_name = (model or "").partition(":")
        if not separator or not provider or not model_name:
            raise ValueError("API model must use provider:model format")
    config["model"] = model

    config["max-turns"] = _parse_integer(
        config.get("max-turns"),
        label="Max turns",
        minimum=1,
        required=True,
    )
    config["planner-timeout-s"] = _parse_integer(
        config.get("planner-timeout-s"),
        label="Planner timeout",
        minimum=1,
    )
    config["claude-code-max-budget-usd"] = _parse_float(
        config.get("claude-code-max-budget-usd"),
        label="Claude Code budget",
        minimum=0,
    )
    no_images = config.get("no-images", False)
    if not isinstance(no_images, bool):
        raise ValueError("No images must be a boolean")
    config["no-images"] = no_images
    config.update(parse_launcher_fields(config, launcher_fields))
    return config


def apply_to_args(
    args: Any,
    payload: dict[str, Any],
    launcher_fields: tuple[LauncherFieldSpec, ...],
) -> None:
    """Apply values submitted by the Dashboard launch form."""
    config = parse_launcher_config(payload, launcher_fields)
    args.planner = config["planner"]
    args.model = config["model"]
    args.max_turns = config["max-turns"]
    args.planner_timeout_s = config["planner-timeout-s"]
    args.reasoning_effort = config.get("reasoning-effort", "none")
    args.no_images = config["no-images"]
    if args.planner == "claude_code":
        args.claude_code_max_budget_usd = config["claude-code-max-budget-usd"]
    for field in launcher_fields:
        setattr(args, field["name"], config[field["name"]])
