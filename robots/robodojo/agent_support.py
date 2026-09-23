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

"""Agent recording, diagnostics and scripted inverse kinematics."""

from __future__ import annotations

import os
import time
from typing import Any

import numpy as np

from rpent.utils.logging import get_logger

logger = get_logger(__name__)


class _VideoRecorder:
    """Write one mp4 per camera (head / left_wrist / right_wrist)."""

    def __init__(self, video_dir: str, fps: int = 25):
        self.video_dir = video_dir
        self.fps = fps
        self.writers: dict[str, Any] = {}
        self.paths: dict[str, str] = {}

    @property
    def enabled(self) -> bool:
        return bool(self.video_dir)

    def _ensure_writer(self, cam: str, frame) -> None:
        if cam in self.writers:
            return
        h, w = frame.shape[:2]
        os.makedirs(self.video_dir, exist_ok=True)
        run_id = os.environ.get("ROBODOJO_RUN_ID", "run")
        path = os.path.join(self.video_dir, f"episode_{run_id}_{cam}.mp4")
        import cv2

        writer = cv2.VideoWriter(
            path,
            cv2.VideoWriter_fourcc(*"mp4v"),
            self.fps,
            (int(w), int(h)),
        )
        if not writer.isOpened():
            raise RuntimeError(f"cannot open video writer: {path}")
        self.writers[cam] = writer
        self.paths[cam] = path

    def record(self, obs: dict) -> None:
        if not self.enabled:
            return
        try:
            import cv2

            vision = obs.get("vision", {})
            for cam in ("cam_head", "cam_left_wrist", "cam_right_wrist"):
                color = vision.get(cam, {}).get("color")
                if color is None:
                    continue
                frame = np.asarray(color)
                if frame.ndim != 3 or frame.shape[2] != 3:
                    continue
                if frame.dtype != np.uint8:
                    frame = frame.astype(np.uint8)
                self._ensure_writer(cam, frame)
                self.writers[cam].write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        except Exception as exc:  # noqa: BLE001 - recording must never break env
            logger.warning("video record warning: %s", exc)
            self.video_dir = None  # disable after first failure

    def close(self) -> list[str]:
        paths = list(self.paths.values())
        for writer in self.writers.values():
            try:
                writer.release()
            except Exception:
                pass
        self.writers.clear()
        self.paths.clear()
        return paths


def _record_obs_frame(recorder, obs: dict) -> None:
    recorder.record(obs)


# ---------------------------------------------------------------------------
# Safety monitor: rolling / off-table bottle alarm (env-internal, GT-based)
#
# TRAINING / EXPLORATION ONLY — must stay off in evaluation runs. Detection
# reads the simulator's ground-truth bottle poses (layout_manager.get_instance_pose)
# and the alarms reach the agent through every tool result, so the policy is
# fed privileged state it cannot obtain from perception. Scores from a run that
# consumed these alarms are not perception-isolated and are not comparable to
# an eval without them.
# ---------------------------------------------------------------------------

_TABLE_X_RANGE = (-0.40, 0.50)
_TABLE_Y_RANGE = (-0.30, 0.06)
_TABLE_TOP_Z = 0.77
_OFF_TABLE_Z = _TABLE_TOP_Z - 0.15
_ROLL_SPEED_MPS = 0.20
# Dustbin region (put_bottles task): bottles inside the bin are scored, not
# lost — exclude from the off-table check.
_BIN_X_RANGE = (-0.90, -0.45)
_BIN_Y_RANGE = (-0.25, 0.05)
_BIN_Z_RANGE = (0.20, 0.70)


def _bottle_labels(env) -> list[str]:
    labels = []
    for i in range(4):
        label = f"bottle{i}"
        try:
            inst = env.scene_manager.layout_manager.get_instance_name(0, label)
            if inst is not None:
                labels.append(label)
        except Exception:
            continue
    return labels


def _bottle_world_pos(env, label: str):
    """Return (pos, rot) of a bottle in world-ish coords (env 0 at origin)."""
    import torch

    pos, rot = env.scene_manager.layout_manager.get_instance_pose(0, label=label)
    if pos is None:
        return None, None
    if isinstance(pos, torch.Tensor):
        pos = pos.detach().cpu().numpy()
    return np.asarray(pos, dtype=np.float64), rot


class _SafetyMonitor:
    """Detect rolling / off-table bottles across steps (env-internal, GT-based).

    TRAINING / EXPLORATION ONLY. The alarm is derived from the simulator's
    ground-truth object poses, and it is surfaced to the agent on every tool
    result — i.e. it hands the policy perception it did not earn. Any run
    scored with these alarms on is therefore not an evaluation.
    """

    def __init__(self) -> None:
        self.last_poses: dict[str, np.ndarray] = {}
        self.last_t: float = 0.0
        self.alarms: dict[str, dict] = {}

    def check(self, env) -> dict:
        """Detect rolling / off-table bottles after a motion step."""
        now = time.time()
        dt = now - self.last_t if self.last_t > 0 else 0.0
        for label in _bottle_labels(env):
            pos, _rot = _bottle_world_pos(env, label)
            if pos is None:
                continue
            x, y, z = float(pos[0]), float(pos[1]), float(pos[2])
            alarm: dict | None = None
            # off-table: outside the table footprint or fell below it
            outside = (
                x < _TABLE_X_RANGE[0]
                or x > _TABLE_X_RANGE[1]
                or y < _TABLE_Y_RANGE[0]
                or y > _TABLE_Y_RANGE[1]
                or z < _OFF_TABLE_Z
            )
            in_bin = (
                _BIN_X_RANGE[0] <= x <= _BIN_X_RANGE[1]
                and _BIN_Y_RANGE[0] <= y <= _BIN_Y_RANGE[1]
                and _BIN_Z_RANGE[0] <= z <= _BIN_Z_RANGE[1]
            )
            if outside and in_bin:
                outside = False
            if outside:
                alarm = {
                    "state": "off_table",
                    "world_xyz": [round(x, 3), round(y, 3), round(z, 3)],
                    "note": "bottle is outside the table footprint / below table",
                }
            elif label in self.last_poses:
                prev = self.last_poses[label]
                speed = float(np.linalg.norm(np.asarray(pos) - prev)) / max(dt, 1e-3)
                if speed > _ROLL_SPEED_MPS:
                    alarm = {
                        "state": "rolling",
                        "world_xyz": [round(x, 3), round(y, 3), round(z, 3)],
                        "speed_mps": round(speed, 3),
                        "note": "bottle moving fast; risk of falling off the table",
                    }
            if alarm is not None:
                self.alarms[label] = alarm
            elif label in self.alarms:
                # clear rolling alarms once the bottle settles
                if (
                    self.alarms[label].get("state") == "rolling"
                    and speed < _ROLL_SPEED_MPS * 0.5
                ):
                    del self.alarms[label]
            self.last_poses[label] = np.asarray(pos, dtype=np.float64)
        self.last_t = now
        return dict(self.alarms)

    def status(self) -> dict:
        return {
            "alarms": dict(self.alarms),
            "alarm_count": len(self.alarms),
        }


def _obs_dict(env, recorder) -> dict[str, Any]:
    from robots.robodojo.language import resolve_instruction

    obs = env.get_obs(env_idx=0)
    obs["instruction"] = resolve_instruction(env)
    _record_obs_frame(recorder, obs)
    return obs


def _status(env, bottle_mon) -> dict[str, Any]:
    status = {
        "step": int(env.take_action_cnt[0]),
        "step_limit": int(env.step_lim),
        "success": bool(env.is_success(env_idx=0)),
    }
    status["safety"] = bottle_mon.status()
    return status


def _reward_details(env, bottle_mon) -> dict[str, Any]:
    """Per-predicate reward/score breakdown for the current episode state."""
    rm = env.reward_manager
    details: dict[str, Any] = _status(env, bottle_mon)
    try:
        reward = float(rm.get_reward(final_check=True)[0])
        score = float(rm.get_score()[0])
        details["reward"] = round(reward, 4)
        details["score"] = round(score, 4)
        details["score_frac"] = round(score / 100.0, 4)
    except Exception as exc:  # noqa: BLE001
        details["score_error"] = str(exc)

    bottles: dict[str, Any] = {}
    for i in range(4):
        label = f"bottle{i}"
        try:
            bottles[label] = bool(
                rm.check_once(
                    rm.is_A_on_B_bottom(
                        label_A=label,
                        label_B="dustbin",
                        min_z_gap=0.0,
                        max_z_gap=0.4,
                    ),
                    0,
                )
            )
        except Exception as exc:  # noqa: BLE001
            bottles[label] = None
            details.setdefault("predicate_errors", {})[label] = str(exc)
    details["bottles_on_bin_bottom"] = bottles
    details["bottles_on_bin_bottom_count"] = sum(1 for v in bottles.values() if v)
    for name, check in (
        ("grippers_open", rm.is_all_gripper_open(open_threshold=0.8)),
        ("arms_home", rm.all_robot_back_to_origin()),
    ):
        try:
            details[name] = bool(rm.check_once(check, 0))
        except Exception as exc:  # noqa: BLE001
            details[name] = None
            details.setdefault("predicate_errors", {})[name] = str(exc)
    return details


def _find_robot(env, arm: str):
    for robot in env.robot_manager.robot_list:
        if robot.type != "target":
            continue
        if str(robot.arm_name).split("_")[0] == arm:
            return robot
    return None


def _solve_ik_position(env, arm: str, xyz: list) -> dict:
    """Position-only IK for scripted motion.

    The fixed-orientation full-pose IK diverges on lateral targets at low z
    (the exact pose is often unreachable/singular, and CuRobo converges to a
    wrong solution that makes the arm swing). Try several candidate
    orientations and return the reachable solution with the smallest joint
    displacement from the current pose.
    """
    from scipy.spatial.transform import Rotation as _R

    robot = _find_robot(env, arm)
    if robot is None:
        return {"status": "Fail", "error": f"robot for arm {arm!r} not found"}
    obs = env.get_obs(env_idx=0)
    ee_pose = np.asarray(obs["state"][f"{arm}_ee_pose"], dtype=np.float64)
    current_q = np.asarray(ee_pose[3:7], dtype=np.float64)  # (w,x,y,z)
    current_joints = np.asarray(
        env.robot_manager.get_joint(robot, env_idx_list=[0])[0],
        dtype=np.float64,
    )
    if len(xyz) != 3:
        return {"status": "Fail", "error": "xyz must have 3 values"}

    candidates = [current_q.copy()]
    # home orientation (known-good reachable pose)
    candidates.append(np.array([0.7071067811865476, 0.0, 0.0, 0.7071067811865476]))
    # small tilts around the current orientation
    r_cur = _R.from_quat([current_q[1], current_q[2], current_q[3], current_q[0]])
    for axis in ((1, 0, 0), (0, 1, 0)):
        for deg in (-25.0, 25.0):
            r_d = _R.from_rotvec(np.deg2rad(deg) * np.asarray(axis, dtype=np.float64))
            q = (r_d * r_cur).as_quat()  # xyzw
            candidates.append(np.array([q[3], q[0], q[1], q[2]], dtype=np.float64))

    best = None
    for q in candidates:
        pose = [float(v) for v in xyz] + [float(v) for v in q]
        try:
            res = env.robot_manager.solve_ik(target_pose=pose, env_idx=0, robot=robot)
        except Exception:  # noqa: BLE001
            continue
        if res.get("status") != "Success":
            continue
        jv = np.asarray(res["joint_value"], dtype=np.float64)
        disp = float(np.abs(jv - current_joints).sum())
        if best is None or disp < best[0]:
            best = (disp, jv)
    if best is None:
        return {
            "status": "Fail",
            "error": "no IK solution across candidate orientations",
        }
    return {
        "status": "Success",
        "arm": arm,
        "joint_value": [float(v) for v in best[1]],
        "joint_displacement": round(best[0], 4),
    }
