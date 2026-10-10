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

from robots.franka.runtime_config import (
    get_perception_calibration_mapping,
    get_robot_config_path,
)

VLA_ACTIONS = {
    "vla_grasp",
    "vla_right_grasp",
    "vla_handoff",
    "vla_left_place",
}
ACTIONS = {"move_delta", "rotate_delta", "open_gripper", "close_gripper"} | VLA_ACTIONS


def vector(value: Any, length: int = 3) -> np.ndarray:
    """Validate a finite, fixed-length vector."""
    array = np.asarray(value, dtype=float)
    if array.shape != (length,) or not np.isfinite(array).all():
        raise ValueError(f"expected a finite {length}-vector")
    return array


def fingerprint() -> dict[str, str]:
    """Hash the robot configuration and every configured calibration source."""
    return {
        key: hashlib.sha256(Path(path).expanduser().read_bytes()).hexdigest()
        for key, path in {
            **get_perception_calibration_mapping(),
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

    def get(self, step: int = -1) -> SimpleNamespace:
        """Return a saved state record."""
        row = self.steps[step]
        return SimpleNamespace(step_idx=row["step_idx"], state=row["state"])

    def artifact_path(self, name: str, *, step: int = -1) -> Path:
        """Locate a per-step artifact without writing the source run."""
        if Path(name).name != name:
            raise ValueError("invalid artifact name")
        index = self.get(step).step_idx
        return self.root / name / f"{index:02d}{Path(name).suffix}"

    def exists(self, name: str, *, step: int = -1) -> bool:
        """Check whether an artifact exists."""
        return self.artifact_path(name, step=step).is_file()

    def load(self, name: str, *, step: int = -1) -> Any:
        """Read a recorded array or JSON artifact."""
        path = self.artifact_path(name, step=step)
        return np.load(path) if path.suffix == ".npy" else json.loads(path.read_text())

    def save(self, *args: Any, **kwargs: Any) -> None:
        """Suppress optional diagnostic writes while reading a source run."""
        # Existing projection helpers optionally write diagnostic overlays.
        return None


def tcp_pose(state: dict, robot: str, arm: str | None = None) -> np.ndarray:
    """Read the TCP pose in the motion frame for this robot."""
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
        else {
            "base": "base",
            "d455": "d455",
            "left_wrist": "left_wrist",
            "right_wrist": "right_wrist",
        }
    )
    if camera not in cameras:
        raise ValueError(f"unsupported projection camera {camera!r} for {robot}")
    path = state.artifact_path(f"{cameras[camera]}.png", step=step)
    query = anchor["phrase"]
    found = molmo.ground(path.read_bytes(), query)
    if not found.found or found.point_xy is None:
        raise ValueError(f"anchor not found: {anchor['phrase']}")
    from PIL import Image

    col, row = vector(found.point_xy, 2)
    with Image.open(path) as image:
        width, height = image.size
        if not (0 <= col < width and 0 <= row < height):
            raise ValueError("grounding pixel outside source image")
    kwargs = {"row": int(row), "col": int(col), "state": state, "step": step}
    if robot == "franka":
        from robots.franka.perception import back_project

        result = back_project(camera=camera, **kwargs).to_dict()
        if result.get("error"):
            raise ValueError(result["error"])
        return vector(result["point_base"])
    from robots.dual_franka.perception import back_project

    result = back_project(
        camera=camera, target_name=anchor["phrase"], **kwargs
    ).to_dict()
    if result.get("error") or result.get("selection_valid") is not True:
        raise ValueError(f"invalid projection: {result}")
    point = vector(result["point_xyz"])
    return vector(point)


def checked_result(result: Any) -> None:
    """Reject controller errors, interruptions and incomplete observations."""
    if (
        not isinstance(result, dict)
        or result.get("ok") is False
        or any(
            result.get(key)
            for key in (
                "error",
                "terminated",
                "truncated",
                "motion_refused",
                "interrupted",
                "state_capture_error",
            )
        )
    ):
        raise RuntimeError(f"primitive did not complete successfully: {result}")
    if isinstance(result.get("last_chunk"), dict):
        checked_result(result["last_chunk"])


def validate_card(card: dict) -> dict:
    """Validate a reviewed v1 plan before generation or execution."""
    if not isinstance(card, dict):
        raise ValueError("Flash plan must be a JSON object")
    if not isinstance(card.get("task"), str) or not isinstance(
        card.get("requirements"), dict
    ):
        raise ValueError(
            "Flash plan requires a task identifier and configuration fingerprint"
        )
    if card.get("version") != 1 or card.get("robot") not in {"franka", "dual_franka"}:
        raise ValueError("unsupported Franka Flash plan")
    if card.get("human_verdict") != "success":
        raise ValueError("card requires a human-confirmed successful source")
    if not isinstance(card.get("plan"), list) or not card["plan"]:
        raise ValueError("empty Flash plan")
    previous = 0
    for entry in card["plan"]:
        step = entry["source_step"]
        if type(step) is not int or step <= previous:
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
                else {"base", "d455", "left_wrist", "right_wrist"}
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
            for key in ("reference_quaternion", "target_quaternion"):
                if np.linalg.norm(vector(entry[key], 4)) < 1e-8:
                    raise ValueError("rotation quaternion must have nonzero norm")
        if action in VLA_ACTIONS:
            if (
                not args.get("prompt", "").strip()
                or type(args.get("max_chunks", 4)) is not int
                or not 1 <= args.get("max_chunks", 4) <= 20
            ):
                raise ValueError("invalid VLA intent")
    return card
