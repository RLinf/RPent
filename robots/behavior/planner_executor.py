"""Import-safe planner executor compatibility for BEHAVIOR.

The real BEHAVIOR motion executor is simulator-owned.  This module intentionally
does not import OmniGibson or CuRobo at module import time; lightweight callers
can build consistent receipts, while live planning must be provided by the env
RPC backend.
"""

from __future__ import annotations

from typing import Any

import numpy as np


class CuroboPlanningError(RuntimeError):
    """Raised when a live CuRobo plan cannot be produced."""


class PlannerExecutionError(RuntimeError):
    """Raised when the env-backed planner executor is unavailable."""


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def primitive_result(
    *,
    name: str,
    primitive_success: bool,
    task_success: bool = False,
    stop_reason: str | None = None,
    info: Any = None,
    **fields: Any,
) -> dict[str, Any]:
    """Build a normalized primitive result without inventing official success."""

    result: dict[str, Any] = {
        "name": str(name),
        "primitive_success": bool(primitive_success),
        "task_success": bool(task_success),
    }
    if stop_reason is not None:
        result["stop_reason"] = str(stop_reason)
    if info is not None:
        result["info"] = _jsonable(info)
    result.update({str(key): _jsonable(value) for key, value in fields.items()})
    return result


def _quat_rotate_vector_xyzw(quaternion_xyzw: Any, vector: Any) -> np.ndarray:
    """Rotate one 3-vector by an xyzw quaternion."""

    q = np.asarray(quaternion_xyzw, dtype=np.float64)
    v = np.asarray(vector, dtype=np.float64)
    if (
        q.shape != (4,)
        or v.shape != (3,)
        or not np.isfinite(q).all()
        or not np.isfinite(v).all()
    ):
        raise ValueError("expected finite quaternion[4] and vector[3]")
    norm = float(np.linalg.norm(q))
    if norm <= 0.0:
        raise ValueError("zero quaternion")
    x, y, z, w = q / norm
    qvec = np.asarray([x, y, z], dtype=np.float64)
    uv = np.cross(qvec, v)
    uuv = np.cross(qvec, uv)
    return v + 2.0 * (w * uv + uuv)


class PlannerExecutor:
    """Placeholder that requires an env-owned live backend for motion."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.args = args
        self.kwargs = kwargs

    def __getattr__(self, name: str) -> Any:
        raise PlannerExecutionError(
            f"PlannerExecutor.{name} requires the BEHAVIOR env RPC backend"
        )


def execute_finish_receipt(
    toolkit: Any,
    *,
    status: str,
    summary: str,
) -> Any:
    """Call the standard main-compatible finish tool on a toolkit."""

    return toolkit.execute_tool("finish", {"status": status, "summary": summary})


def write_recipe_if_supported(toolkit: Any, recipe_tag: str) -> str | None:
    """Idempotently call ``toolkit.write_recipe`` when available."""

    writer = getattr(toolkit, "write_recipe", None)
    if not callable(writer):
        return None
    return writer(recipe_tag)


__all__ = [
    "CuroboPlanningError",
    "PlannerExecutionError",
    "PlannerExecutor",
    "_quat_rotate_vector_xyzw",
    "execute_finish_receipt",
    "primitive_result",
    "write_recipe_if_supported",
]
