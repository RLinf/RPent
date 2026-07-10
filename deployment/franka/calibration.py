"""Franka camera calibration loading helpers.

Calibration records are stored per RealSense serial under
``~/.cache/physical_agent/franka/camera_calibration``. Supported records:

- fixed scene camera: ``{"T_base_cam": [[...]], ...}``
- wrist camera: ``{"T_tcp_cam": [[...]], ...}``

``T_base_cam`` maps camera-frame points into ``panda_link0``. ``T_tcp_cam`` maps
wrist-camera points into the live TCP frame; the env server composes it with the
current ``T_base_tcp`` for each observation.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np

_CALIB_DIR = "~/.cache/physical_agent/franka/camera_calibration"
MAX_ACCEPTABLE_RMSE_M = 0.02


def calib_path(serial: str) -> Path:
    """Return the on-disk calibration path for a RealSense serial."""
    return Path(os.path.expanduser(_CALIB_DIR)) / f"{serial}.json"


def load_record(serial: str) -> dict | None:
    """Load a camera calibration record, converting known transforms to arrays."""
    path = calib_path(serial)
    if not path.is_file():
        return None
    with open(path) as f:
        data = json.load(f)
    for key in ("T_base_cam", "T_tcp_cam"):
        if key in data and data[key] is not None:
            data[key] = np.asarray(data[key], dtype=np.float64)
    data.setdefault("serial", str(serial))
    data["path"] = str(path)
    return data


def _jsonable(value: Any) -> Any:
    """Convert numpy values to JSON-compatible Python values."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: _jsonable(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(val) for val in value]
    return value


def save_record(serial: str, **fields: Any) -> Path:
    """Save a calibration record for ``serial``.

    Known transform fields are ``T_base_cam`` for fixed cameras and
    ``T_tcp_cam`` for wrist cameras. Extra diagnostics are preserved.
    """
    path = calib_path(serial)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {"serial": str(serial), **fields}
    with open(path, "w") as f:
        json.dump(_jsonable(data), f, indent=2)
    return path


def save_scene_extrinsic(
    serial: str,
    T_base_cam,
    *,
    K=None,
    rmse_m: float | None = None,
    num_points: int | None = None,
    **extra: Any,
) -> Path:
    """Persist a fixed-camera ``T_base_cam`` calibration."""
    fields: dict[str, Any] = {"T_base_cam": np.asarray(T_base_cam, dtype=np.float64)}
    if K is not None:
        fields["K"] = np.asarray(K, dtype=np.float64)
    if rmse_m is not None:
        fields["rmse_m"] = float(rmse_m)
    if num_points is not None:
        fields["num_points"] = int(num_points)
    fields.update(extra)
    return save_record(serial, **fields)


def save_wrist_extrinsic(
    serial: str,
    T_tcp_cam,
    *,
    K=None,
    rmse_m: float | None = None,
    num_points: int | None = None,
    **extra: Any,
) -> Path:
    """Persist a wrist-camera ``T_tcp_cam`` hand-eye calibration."""
    fields: dict[str, Any] = {"T_tcp_cam": np.asarray(T_tcp_cam, dtype=np.float64)}
    if K is not None:
        fields["K"] = np.asarray(K, dtype=np.float64)
    if rmse_m is not None:
        fields["rmse_m"] = float(rmse_m)
    if num_points is not None:
        fields["num_points"] = int(num_points)
    fields.update(extra)
    return save_record(serial, **fields)


def is_accepted(record: dict | None) -> bool:
    """Return whether a record exists and passes the saved RMSE gate."""
    if record is None:
        return False
    rmse = record.get("rmse_m")
    return rmse is None or float(rmse) <= MAX_ACCEPTABLE_RMSE_M
