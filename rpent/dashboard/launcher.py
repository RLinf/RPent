"""Small adapter between the dashboard launch form and CLI args."""
from __future__ import annotations

from typing import Any


FIELDS = (
    "suite",
    "task",
    "seed",
    "model",
    "cerebrum",
    "cuda_device",
    "max_turns",
    "max_tokens",
    "max_episode_steps",
)

DEFAULTS = {
    "suite": "libero_object_task",
    "task": 6,
    "seed": 0,
    "cerebrum": "claude_code",
    "max_turns": 100,
    "max_tokens": 8192,
    "max_episode_steps": 600,
}

INT_FIELDS = {"task", "seed", "max_turns", "max_tokens", "max_episode_steps"}
BOOL_FIELDS: set[str] = set()
OPTIONAL_STR_FIELDS = {"model", "cuda_device"}


def defaults_from_args(args: Any) -> dict[str, Any]:
    """Build form defaults from parsed CLI args, falling back to UI defaults."""
    defaults: dict[str, Any] = {}
    for key in FIELDS:
        value = getattr(args, key, None)
        defaults[key] = DEFAULTS.get(key) if value is None else value
    return defaults


def apply_to_args(args: Any, payload: dict[str, Any]) -> None:
    """Overlay launch form values onto parsed CLI args."""
    for key in FIELDS:
        if key not in payload:
            continue
        value = payload[key]
        if key in OPTIONAL_STR_FIELDS and value in ("", None):
            value = None
        elif key in INT_FIELDS:
            value = DEFAULTS[key] if value in ("", None) else int(value)
        elif key in BOOL_FIELDS:
            value = _as_bool(value)
        setattr(args, key, value)


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)
