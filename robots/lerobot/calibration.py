"""Load / save the scene-camera → base extrinsic ``T_base_cam``.

``T_base_cam`` is the fixed rigid transform that maps a point in the scene
camera frame into the arm ``base_link`` world frame. It is produced once by the
touch/Kabsch calibration (``toolkits/lerobot/calibrate_scene_cam.py``) and
loaded by the env server so ``back_project`` can return world coordinates.

Stored per camera serial under the LeRobot cache so it survives across runs.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

_CALIB_DIR = "~/.cache/huggingface/lerobot/calibration/scene_cam"

# A scene-cam -> base fit whose RMSE (meters) exceeds this is treated as a
# FAILED calibration: ``back_project`` would map pixels to badly wrong world
# coordinates (the arm then chases unreachable targets), so the driver refuses
# to save it or trust it on load. ~2 cm matches the auto-calibrator's existing
# "rerun" warning; a good touch/Kabsch fit is typically a few mm.
MAX_ACCEPTABLE_RMSE_M = 0.02


def calib_path(serial: str) -> Path:
    """Return the on-disk path of the extrinsic file for a camera ``serial``."""
    return Path(os.path.expanduser(_CALIB_DIR)) / f"{serial}.json"


def save_extrinsic(
    serial: str,
    T_base_cam,
    *,
    K=None,
    rmse_m: float | None = None,
    num_points: int | None = None,
) -> Path:
    """Persist ``T_base_cam`` (and calibration diagnostics) for ``serial``."""
    path = calib_path(serial)
    path.parent.mkdir(parents=True, exist_ok=True)
    data: dict = {
        "serial": serial,
        "T_base_cam": np.asarray(T_base_cam, dtype=np.float64).tolist(),
    }
    if K is not None:
        data["K"] = np.asarray(K, dtype=np.float64).tolist()
    if rmse_m is not None:
        data["rmse_m"] = float(rmse_m)
    if num_points is not None:
        data["num_points"] = int(num_points)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    return path


def load_extrinsic_record(serial: str) -> dict | None:
    """Load the full saved calibration record for ``serial`` (or ``None``).

    Returns the parsed JSON with ``T_base_cam`` as a ``(4, 4)`` ndarray plus any
    saved diagnostics (``rmse_m``, ``num_points``, ``K``). Use this when you need
    to judge fit quality, not just apply the transform.
    """
    path = calib_path(serial)
    if not path.is_file():
        return None
    with open(path) as f:
        data = json.load(f)
    data["T_base_cam"] = np.asarray(data["T_base_cam"], dtype=np.float64)
    return data


def load_extrinsic(serial: str) -> np.ndarray | None:
    """Load ``T_base_cam`` (4x4) for ``serial``, or ``None`` if not calibrated."""
    record = load_extrinsic_record(serial)
    return None if record is None else record["T_base_cam"]
