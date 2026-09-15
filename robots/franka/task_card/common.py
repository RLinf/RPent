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

"""Recorded-state access and calibrated grounding for both Franka variants."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

from robots.franka.runtime_config import get_calibration_path, get_robot_config_path

VLA_ACTIONS = {"vla_grasp", "vla_right_grasp", "vla_handoff", "vla_left_place"}
ACTIONS = {"move_delta", "rotate_delta", "open_gripper", "close_gripper"} | VLA_ACTIONS


def vector(value: Any, length: int = 3) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if array.shape != (length,) or not np.isfinite(array).all():
        raise ValueError(f"expected a finite {length}-vector")
    return array


def fingerprint() -> dict[str, str]:
    return {
        key: hashlib.sha256(Path(path).expanduser().read_bytes()).hexdigest()
        for key, path in {
            "calibration": get_calibration_path(),
            "robot_config": get_robot_config_path(),
        }.items()
    }


class RecordedState:
    """Read a saved EnvState without resetting or rewriting the source run."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.steps = json.loads((self.root / "states.json").read_text())["steps"]
        if not self.steps or [r["step_idx"] for r in self.steps] != list(
            range(len(self.steps))
        ):
            raise ValueError(
                "source states must have contiguous step indices starting at zero"
            )

    def get(self, step=-1):
        row = self.steps[step]
        return SimpleNamespace(step_idx=row["step_idx"], state=row["state"])

    def artifact_path(self, name, *, step=-1):
        if Path(name).name != name:
            raise ValueError("invalid artifact name")
        index = self.get(step).step_idx
        return self.root / name / f"{index:02d}{Path(name).suffix}"

    def exists(self, name, *, step=-1):
        return self.artifact_path(name, step=step).is_file()

    def load(self, name, *, step=-1):
        path = self.artifact_path(name, step=step)
        return np.load(path) if path.suffix == ".npy" else json.loads(path.read_text())

    def save(self, *args, **kwargs):
        # Existing projection helpers optionally write diagnostic overlays.
        return None


def tcp_pose(state: dict, robot: str, arm: str | None = None) -> np.ndarray:
    if robot == "franka":
        pose = state["raw_base_state"]["tcp_pose"]
    else:
        if arm not in {"left", "right"}:
            raise ValueError("dual-Franka action requires arm=left/right")
        arm_state = state[f"{arm}_arm"]
        pose = arm_state["tcp_pose"]
        frame = arm_state.get("tcp_pose_frame", state.get("coordinate_frame"))
        if frame != "right_base":
            raise ValueError(
                "dual-Franka cards require current right_base state records; re-record legacy traces"
            )
    return vector(pose, 7)


def localize(
    state: Any,
    robot: str,
    anchor: dict,
    molmo: Any,
    *,
    step: int = -1,
    arm: str | None = None,
) -> np.ndarray:
    """Return a grounded point in the current robot's motion frame."""
    camera = anchor["camera"]
    cameras = (
        {"wrist": "wrist", "third_person": "camera"}
        if robot == "franka"
        else {"base": "base", "d455": "d455"}
    )
    if camera not in cameras:
        raise ValueError(f"unsupported projection camera {camera!r} for {robot}")
    path = state.artifact_path(f"{cameras[camera]}.png", step=step)
    found = molmo.ground(path.read_bytes(), anchor["phrase"])
    if not found.found or found.point_xy is None:
        raise ValueError(f"anchor not found: {anchor['phrase']}")
    from PIL import Image

    with Image.open(path) as image:
        width, height = image.size
    col, row = vector(found.point_xy, 2)
    if not (0 <= col < width and 0 <= row < height):
        raise ValueError("grounding pixel outside source image")
    kwargs = {"row": int(row), "col": int(col), "state": state, "step": step}
    if robot == "franka":
        from robots.franka.perception import back_project

        result = back_project(camera=camera, **kwargs)
        if result.get("error"):
            raise ValueError(result["error"])
        return vector(result["point_base"])
    from robots.dual_franka.perception import back_project

    result = back_project(camera=camera, target_name=anchor["phrase"], **kwargs)
    if result.get("error") or result.get("selection_valid") is not True:
        raise ValueError(f"invalid projection: {result}")
    point = vector(result["point_xyz"])
    return vector(point)


def checked_result(result: Any) -> None:
    if (
        not isinstance(result, dict)
        or result.get("error")
        or result.get("ok") is False
        or result.get("interrupted")
        or result.get("state_capture_error")
    ):
        raise RuntimeError(f"primitive did not complete successfully: {result}")
    if isinstance(result.get("last_chunk"), dict):
        checked_result(result["last_chunk"])


def validate_card(card: dict) -> dict:
    if card.get("version") != 1 or card.get("robot") not in {"franka", "dual_franka"}:
        raise ValueError("unsupported Franka task card")
    if card.get("human_verdict") != "success":
        raise ValueError("card requires a human-confirmed successful source")
    if not card.get("plan"):
        raise ValueError("empty task card")
    previous = 0
    for entry in card["plan"]:
        step = entry["source_step"]
        if not isinstance(step, int) or step <= previous:
            raise ValueError("source steps must be strictly increasing")
        previous = step
        action, args = entry["action"], entry["arguments"]
        if action not in ACTIONS:
            raise ValueError(f"unsupported action: {action}")
        if action in VLA_ACTIONS and (
            (card["robot"] == "franka") != (action == "vla_grasp")
        ):
            raise ValueError("VLA primitive does not belong to the requested robot")
        allowed = {
            "move_delta": {"delta_xyz"},
            "rotate_delta": {"delta_rpy"},
            "open_gripper": set(),
            "close_gripper": set(),
            "vla_grasp": {"prompt", "max_chunks"},
            "vla_right_grasp": {"prompt", "max_chunks"},
            "vla_handoff": {"prompt", "max_chunks"},
            "vla_left_place": {"prompt", "max_chunks"},
        }[action]
        if card["robot"] == "dual_franka" and action not in VLA_ACTIONS:
            allowed = allowed | {"arm"}
            if args.get("arm") not in {"left", "right"}:
                raise ValueError("missing left/right arm")
        if set(args) - allowed:
            raise ValueError(f"unsupported arguments for {action}")
        if action == "move_delta":
            vector(args["delta_xyz"])
            vector(entry["offset_xyz"])
            anchor = entry["anchor"]
            cameras = (
                {"wrist", "third_person"}
                if card["robot"] == "franka"
                else {"base", "d455"}
            )
            if (
                not anchor.get("phrase", "").strip()
                or anchor.get("camera") not in cameras
            ):
                raise ValueError(
                    "every move requires a semantic anchor and projection camera"
                )
        if action == "rotate_delta":
            vector(args["delta_rpy"])
            if (
                entry.get("rotation_mode") not in {"relative", "fixed"}
                or not entry.get("intent", "").strip()
            ):
                raise ValueError(
                    "rotation requires explicit intent and relative/fixed mode"
                )
            vector(entry["reference_quaternion"], 4)
            vector(entry["target_quaternion"], 4)
        if action in VLA_ACTIONS:
            if (
                not args.get("prompt", "").strip()
                or not 1 <= args.get("max_chunks", 4) <= 20
            ):
                raise ValueError("invalid VLA intent")
    return card
