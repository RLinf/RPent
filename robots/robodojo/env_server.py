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

"""RPC server wrapping one RoboDojo Isaac Sim environment.

Runs inside the ``robodojo-sim`` conda environment and exposes the unified
``rpent`` env RPC (``BaseEnvFacade``). Isaac Sim is not thread-safe, so the
facade overrides ``serve`` to dispatch every env op on the main thread.

Launch:

    source <workspace>/scripts/activate_runtime.sh sim
    python -u robots/robodojo/env_server.py --task put_bottles_into_dustbin \
        --layout 1 --env-cfg-type arx_x5 --device-id 0 --host 127.0.0.1 \
        --port 0 --transport http --parent-watch
"""

from __future__ import annotations

import argparse
import base64
import os
import queue
import sys
import threading
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

# Support direct execution from an RPent checkout before package imports.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from rpent.robots.components.env_facade_base import BaseEnvFacade  # noqa: E402

# ---------------------------------------------------------------------------


def _random_reset(env_idx: int = 0) -> None:
    """Fresh random layout: random template + large random env seed.

    Mirrors the official eval's random-layout rollout (eval_env.py): the std
    template only stores placement CONFIG and ClutteredGenerator samples actual
    object positions from the ENV SEED, so a random template + fresh seed =
    a fresh layout. The naive collect_env ``random_mode`` (saved_layouts=None)
    leaves the scene EMPTY — do not use it.
    """
    import random as _random

    template = env.seed_manager.get_seed_scene_info(
        _random.randrange(len(env.seed_manager.seed_info))
    )
    fresh = [_random.randrange(0, 1_000_000_000) for _ in range(env.num_envs)]
    env.traj_recorder.reset_all()
    env.env_seeds = fresh
    env.success = [True] * env.num_envs
    env.end_flag = [False] * env.num_envs
    env.take_action_cnt = [0] * env.num_envs
    env.current_env_seed_map = dict(enumerate(fresh))
    for idx in range(env.num_envs):
        env.scene_manager.layout_manager.set_saved_layout(idx, template)
    # Bypass CollectEnv.reset (it would overwrite the template via
    # get_seed_scene_info(env_seed)); call the task-class reset directly.
    for cls in type(env).__mro__[1:]:
        if "reset" in cls.__dict__:
            cls.reset(env, seed=fresh)
            break
    env.obs_manager.reset()
    env.setup_scene()
    env.robot_manager.set_origin_endpose()
    env.robot_manager.set_robot_init_state()
    env.reward_manager.init_state()
    env.run_reward()
    if hasattr(env, "get_score"):
        try:
            env.get_score()
        except Exception as exc:  # noqa: BLE001
            print(f"[robodojo-env] get_score registration failed: {exc}", flush=True)
    return None


def _standard_reset(env_idx: int = 0) -> None:
    env.reset(seed=_args.layout)
    if hasattr(env, "get_score"):
        try:
            env.get_score()
        except Exception as exc:  # noqa: BLE001
            print(f"[robodojo-env] get_score registration failed: {exc}", flush=True)
    return None


# ---------------------------------------------------------------------------
# Video recording (one mp4 per camera)
# ---------------------------------------------------------------------------


class _VideoRecorder:
    """Write one mp4 per camera (head / left_wrist / right_wrist)."""

    def __init__(self, video_dir: str, fps: int = 25):
        self.video_dir = video_dir
        self.fps = fps
        self._writers: dict[str, Any] = {}
        self._paths: dict[str, str] = {}

    @property
    def enabled(self) -> bool:
        return bool(self.video_dir)

    def _ensure_writer(self, cam: str, frame) -> None:
        if cam in self._writers:
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
        self._writers[cam] = writer
        self._paths[cam] = path

    def record(self, obs: dict) -> None:
        if not self.enabled:
            return
        try:
            import cv2
            import numpy as np

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
                self._writers[cam].write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        except Exception as exc:  # noqa: BLE001 - recording must never break env
            print(f"[robodojo-env] video record warning: {exc}", flush=True)
            self.video_dir = None  # disable after first failure

    def close(self) -> list[str]:
        paths = list(self._paths.values())
        for writer in self._writers.values():
            try:
                writer.release()
            except Exception:
                pass
        self._writers.clear()
        self._paths.clear()
        return paths


def _record_obs_frame(obs: dict) -> None:
    video_recorder.record(obs)


# ---------------------------------------------------------------------------
# Safety monitor: rolling / off-table bottle alarm (env-internal, GT-based)
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

_safety_lock = threading.Lock()
_last_bottle_poses: dict[str, np.ndarray] = {}
_last_safety_t: float = 0.0
_safety_alarms: dict[str, dict] = {}


def _bottle_labels() -> list[str]:
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


def _bottle_world_pos(label: str, env_idx: int = 0):
    """Return (pos, rot) of a bottle in world-ish coords (env 0 at origin)."""
    import torch

    pos, rot = env.scene_manager.layout_manager.get_instance_pose(env_idx, label=label)
    if pos is None:
        return None, None
    if isinstance(pos, torch.Tensor):
        pos = pos.detach().cpu().numpy()
    return np.asarray(pos, dtype=np.float64), rot


def _check_safety(env_idx: int = 0) -> dict:
    """Detect rolling / off-table bottles after a motion step."""
    import time as _time

    global _last_bottle_poses, _last_safety_t, _safety_alarms
    now = _time.time()
    dt = now - _last_safety_t if _last_safety_t > 0 else 0.0
    with _safety_lock:
        for label in _bottle_labels():
            pos, _rot = _bottle_world_pos(label, env_idx)
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
            elif label in _last_bottle_poses:
                prev = _last_bottle_poses[label]
                speed = float(np.linalg.norm(np.asarray(pos) - prev)) / max(dt, 1e-3)
                if speed > _ROLL_SPEED_MPS:
                    alarm = {
                        "state": "rolling",
                        "world_xyz": [round(x, 3), round(y, 3), round(z, 3)],
                        "speed_mps": round(speed, 3),
                        "note": "bottle moving fast; risk of falling off the table",
                    }
            if alarm is not None:
                _safety_alarms[label] = alarm
            elif label in _safety_alarms:
                # clear rolling alarms once the bottle settles
                if (
                    _safety_alarms[label].get("state") == "rolling"
                    and speed < _ROLL_SPEED_MPS * 0.5
                ):
                    del _safety_alarms[label]
            _last_bottle_poses[label] = np.asarray(pos, dtype=np.float64)
        _last_safety_t = now
    return dict(_safety_alarms)


def _safety_status(env_idx: int = 0) -> dict:
    with _safety_lock:
        return {
            "alarms": dict(_safety_alarms),
            "alarm_count": len(_safety_alarms),
        }


def _encode(obj: Any) -> Any:
    """Recursively encode numpy arrays for the JSON wire (matches rpent)."""
    if isinstance(obj, np.ndarray):
        return {
            "__ndarray__": base64.b64encode(obj.tobytes()).decode("ascii"),
            "dtype": str(obj.dtype),
            "shape": list(obj.shape),
        }
    if isinstance(obj, dict):
        return {str(k): _encode(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_encode(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    return obj


def _obs_dict(env_idx: int = 0) -> dict[str, Any]:
    obs = env.get_obs(env_idx=env_idx)
    _record_obs_frame(obs)
    return obs


def _status(env_idx: int = 0) -> dict[str, Any]:
    status = {
        "step": int(env.take_action_cnt[env_idx]),
        "step_limit": int(env.step_lim),
        "success": bool(env.is_success(env_idx=env_idx)),
    }
    status["safety"] = _safety_status(env_idx)
    return status


def _reward_details(env_idx: int = 0) -> dict[str, Any]:
    """Per-predicate reward/score breakdown for the current episode state."""
    rm = env.reward_manager
    details: dict[str, Any] = _status(env_idx)
    try:
        reward = float(rm.get_reward(final_check=True)[env_idx])
        score = float(rm.get_score()[env_idx])
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
                    env_idx,
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
            details[name] = bool(rm.check_once(check, env_idx))
        except Exception as exc:  # noqa: BLE001
            details[name] = None
            details.setdefault("predicate_errors", {})[name] = str(exc)
    return details


def _find_robot(arm: str):
    for robot in env.robot_manager.robot_list:
        if robot.type != "target":
            continue
        if str(robot.arm_name).split("_")[0] == arm:
            return robot
    return None


def _solve_ik_position(arm: str, xyz: list, env_idx: int = 0) -> dict:
    """Position-only IK for scripted motion.

    The fixed-orientation full-pose IK diverges on lateral targets at low z
    (the exact pose is often unreachable/singular, and CuRobo converges to a
    wrong solution that makes the arm swing). Try several candidate
    orientations and return the reachable solution with the smallest joint
    displacement from the current pose.
    """
    import numpy as np
    from scipy.spatial.transform import Rotation as _R

    robot = _find_robot(arm)
    if robot is None:
        return {"status": "Fail", "error": f"robot for arm {arm!r} not found"}
    obs = env.get_obs(env_idx=env_idx)
    ee_pose = np.asarray(obs["state"][f"{arm}_ee_pose"], dtype=np.float64)
    current_q = np.asarray(ee_pose[3:7], dtype=np.float64)  # (w,x,y,z)
    current_joints = np.asarray(
        env.robot_manager.get_joint(robot, env_idx_list=[env_idx])[env_idx],
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
            res = env.robot_manager.solve_ik(
                target_pose=pose, env_idx=env_idx, robot=robot
            )
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


def _control_info_from_action(action: dict, action_type: str, env_idx: int = 0) -> dict:
    """Convert a policy action dict to control_info (mirrors eval_env.take_action_batch)."""
    control_info: dict[str, Any] = {}
    for robot in env.robot_manager.robot_list:
        if robot.type != "target":
            continue
        name = robot.arm_name.split("_")[0]
        if action_type == "joint":
            key_name = env.robot_manager.process_name(robot.arm_name)
            control_info[key_name] = {"position": list(action[key_name])}
            gripper_key = env.robot_manager.process_name(robot.gripper_name)
            if robot.ee_type == "gripper":
                val = float(np.clip(action[gripper_key][0], 0, 1))
                if robot.gripper_move["sign"] == 1:
                    val = (
                        val * (robot.gripper_scale[1] - robot.gripper_scale[0])
                        + robot.gripper_scale[0]
                    )
                else:
                    val = (1 - val) * (
                        robot.gripper_scale[1] - robot.gripper_scale[0]
                    ) + robot.gripper_scale[0]
                vals = [
                    val,
                    val * robot.gripper_move["mimic"][1]
                    + robot.gripper_move["mimic"][2],
                ]
                control_info[gripper_key] = {"position": vals}
        elif action_type == "ee":
            key_name = f"{name}_ee_pose"
            obs_name = env.robot_manager.process_name(robot.arm_name)
            target_pose = action[key_name]
            ik_result = env.robot_manager.solve_ik(
                target_pose=target_pose, env_idx=env_idx, robot=robot
            )
            if ik_result["status"] == "Success":
                control_info[obs_name] = {"position": ik_result["joint_value"]}
            # Gripper must be controlled in ee mode too (mirror
            # eval_env.take_action_batch); otherwise set_gripper / move_to's
            # gripper arg are silently ignored.
            gripper_key = env.robot_manager.process_name(robot.gripper_name)
            if robot.ee_type == "gripper" and gripper_key in action:
                val = float(np.clip(action[gripper_key][0], 0, 1))
                if robot.gripper_move["sign"] == 1:
                    val = (
                        val * (robot.gripper_scale[1] - robot.gripper_scale[0])
                        + robot.gripper_scale[0]
                    )
                else:
                    val = (1 - val) * (
                        robot.gripper_scale[1] - robot.gripper_scale[0]
                    ) + robot.gripper_scale[0]
                vals = [
                    val,
                    val * robot.gripper_move["mimic"][1]
                    + robot.gripper_move["mimic"][2],
                ]
                control_info[gripper_key] = {"position": vals}
    return control_info


def _bump_step(env_idx: int) -> None:
    """Mirror eval_env.take_action_batch step accounting."""
    if env.take_action_cnt[env_idx] >= env.step_lim or env.end_flag[env_idx]:
        return
    env.take_action_cnt[env_idx] += 1


def _infer_action_type(action: dict) -> str:
    """Infer joint/ee action type from the action dict keys."""
    for key in action:
        if key.endswith("_arm_joint_state"):
            return "joint"
        if key.endswith("_ee_pose"):
            return "ee"
    return "joint"


class RoboDojoEnvFacade(BaseEnvFacade):
    """Isaac Sim RoboDojo backend exposing the unified env RPC."""

    def __init__(self) -> None:
        super().__init__()

    def _register_rpc(self) -> None:
        super()._register_rpc()
        self._rpc.update(
            {
                "env.get_obs": self.get_obs,
                "env.get_status": self.get_status,
                "env.get_reward_details": self.get_reward_details,
                "env.get_safety_status": self.get_safety_status,
                "env.solve_ik_position": self.solve_ik_position,
                "env.is_success": self.is_success,
                "env.close": self.close,
            }
        )
        self._readonly_methods.update(
            [
                "env.get_env_meta",
                "env.get_task_language",
                "env.get_camera_meta",
                "env.render_camera",
                "env.get_obs",
                "env.get_status",
                "env.get_reward_details",
                "env.get_safety_status",
                "env.solve_ik_position",
            ]
        )

    # ---- unified facade contract ----
    def get_env_meta(self) -> dict[str, Any]:
        return {
            "task": _args.task,
            "layout": _args.layout,
            "env_cfg_type": _args.env_cfg_type,
            "device_id": _args.device_id,
            "num_envs": _args.num_envs,
            "max_episode_steps": _args.max_episode_steps,
            "random": _args.random,
        }

    def get_task_language(self) -> str:
        try:
            return str(env.gen_instruction(0)[0])
        except Exception:  # noqa: BLE001
            return _args.task

    def get_camera_meta(self) -> dict[str, Any]:
        obs = env.get_obs(env_idx=0)
        vision = obs.get("vision") or {}
        return {name: {"width": 640, "height": 480} for name in vision}

    def render_camera(self, camera_name: str | None = None) -> dict[str, Any]:
        vision = _obs_dict().get("vision") or {}
        if camera_name is not None and camera_name not in vision:
            raise ValueError(f"unknown camera: {camera_name!r}")
        return vision

    def reset(self) -> dict[str, Any]:
        if _args.random:
            _random_reset()
        else:
            _standard_reset()
        return _obs_dict()

    def step(self, flat_action):
        """Apply one RoboDojo action and return ``(obs, reward, done, info)``.

        ``flat_action`` is the RoboDojo action dict (joint or ee keys); the
        action type is inferred from the keys. Mirrors the unified
        ``BaseEnvClient.step`` contract used by the other robot backends.
        """
        action_type = _infer_action_type(flat_action)
        control_info = _control_info_from_action(flat_action, action_type, 0)
        _bump_step(0)
        env.apply_target(control_info, 0)
        alarms = _check_safety(0)
        if alarms:
            print(f"[robodojo-env] SAFETY ALARM: {alarms}", flush=True)
        obs = _obs_dict(0)
        reward_details = _reward_details(0)
        reward = float(reward_details.get("reward", 0.0) or 0.0)
        done = bool(env.is_success(env_idx=0))
        info: dict[str, Any] = {
            "status": _status(0),
            "step_limit": int(env.step_lim),
            "safety": alarms,
        }
        return obs, reward, done, info

    def chunk_step(self, flat_actions, *, return_all_frames: bool = False):
        raise NotImplementedError("RoboDojo uses step for action control")

    # ---- RoboDojo-specific RPC ----
    def get_obs(self) -> dict[str, Any]:
        return _obs_dict(0)

    def get_status(self) -> dict[str, Any]:
        return _status(0)

    def get_reward_details(self) -> dict[str, Any]:
        return _reward_details(0)

    def solve_ik_position(self, arm: str, xyz: list) -> dict[str, Any]:
        return _solve_ik_position(arm, xyz, 0)

    def get_safety_status(self) -> dict[str, Any]:
        return _safety_status(0)

    def is_success(self) -> bool:
        return bool(env.is_success(env_idx=0))

    def close(self) -> None:
        try:
            paths = video_recorder.close()
            print(f"[robodojo-env] videos written: {paths}", flush=True)
        finally:

            def _shutdown() -> None:
                time.sleep(0.5)
                try:
                    simulation_app.close()
                except Exception:  # noqa: BLE001
                    pass

            # Respond to the RPC first, then tear down Isaac (the process
            # would otherwise exit before the HTTP response is sent).
            threading.Thread(target=_shutdown, daemon=True).start()

    def serve(
        self,
        *,
        transport: str,
        host: str,
        port: int,
        parent_watch: bool = False,
    ) -> None:
        """Override: dispatch every env op on the main thread (Isaac Sim is not thread-safe)."""
        work_queue: "queue.Queue[tuple[threading.Event, dict[str, Any]] | None]" = (
            queue.Queue()
        )

        def dispatch(method: str, args: tuple, kwargs: dict) -> Any:
            if method == "healthz":
                return {"status": "ok"}
            if method == "shutdown":
                self._shutdown_event.set()
                return {"ok": True}
            event = threading.Event()
            req: dict[str, Any] = {
                "method": method,
                "args": args,
                "kwargs": kwargs,
                "result": None,
                "error": None,
            }
            work_queue.put((event, req))
            event.wait()
            if req["error"]:
                raise RuntimeError(req["error"])
            return req["result"]

        from rpent.utils.daemon import watch_parent_death
        from rpent.utils.rpc.http_rpc import HttpRpcServer
        from rpent.utils.rpc.socket_rpc import SocketRpcServer

        server_cls = HttpRpcServer if transport == "http" else SocketRpcServer
        server = server_cls((host, port), dispatch)
        bound_host, bound_port = server.server_address
        bound_host = "127.0.0.1" if bound_host == "0.0.0.0" else bound_host
        url = f"{transport}://{bound_host}:{bound_port}"
        print(f"RPC server listening on {url}", flush=True)

        if parent_watch:
            watch_parent_death(self._shutdown_event.set)
        # The HTTP server runs on a daemon thread; env operations are executed
        # on THIS thread (the Isaac Sim main thread) via the work queue.
        threading.Thread(target=server.serve_forever, daemon=True).start()
        try:
            while not self._shutdown_event.is_set():
                item = work_queue.get()
                if item is None:
                    break
                event, req = item
                try:
                    req["result"] = self._dispatch(
                        req["method"], req["args"], req["kwargs"]
                    )
                except Exception:  # noqa: BLE001
                    req["error"] = traceback.format_exc()
                event.set()
        finally:
            server.shutdown()
            server.server_close()
            self.close()


def main() -> None:
    global _args, env, simulation_app, video_recorder

    parser = argparse.ArgumentParser(description="RoboDojo Isaac Sim env RPC server")
    parser.add_argument("--task", default="put_bottles_into_dustbin")
    parser.add_argument("--layout", type=int, default=0, help="layout id == seed")
    parser.add_argument("--env-cfg-type", default="arx_x5")
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument("--max-episode-steps", type=int, default=700)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=0)
    parser.add_argument("--parent-pid", type=int, default=None)
    parser.add_argument(
        "--video-dir",
        default=None,
        help="Directory for per-camera episode mp4 files",
    )
    parser.add_argument(
        "--random",
        action="store_true",
        help="Sample a fresh random scene layout per episode "
        "(random template + fresh env seed, official-eval style)",
    )
    parser.add_argument("--transport", choices=["http", "socket"], default="http")
    parser.add_argument("--parent-watch", action="store_true")

    from isaaclab.app import AppLauncher  # noqa: E402

    AppLauncher.add_app_launcher_args(parser)
    _args = parser.parse_args()

    # Camera capture requires the replicator/sensor extensions + camera flag;
    # without them obs_manager.capture_manager.step() blocks forever. The
    # extensions go through ``--kit_args`` as a single string (Kit passthrough).
    _CAMERA_KIT_ARGS = (
        "--enable isaacsim.replicator.behavior --enable isaacsim.sensors.camera"
    )
    _args.enable_cameras = True
    if getattr(_args, "kit_args", None):
        _args.kit_args = f"{_args.kit_args} {_CAMERA_KIT_ARGS}"
    else:
        _args.kit_args = _CAMERA_KIT_ARGS

    if _args.parent_pid is not None:

        def _watch_parent(pid: int) -> None:
            while True:
                try:
                    os.kill(pid, 0)
                except OSError:
                    os._exit(0)
                time.sleep(2)

        threading.Thread(
            target=_watch_parent, args=(_args.parent_pid,), daemon=True
        ).start()

    os.environ.setdefault(
        "ROBODOJO_RUN_ID", datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    )
    os.environ.setdefault("PYTHONUNBUFFERED", "1")

    app_launcher = AppLauncher(_args)
    simulation_app = app_launcher.app

    # Isaac-dependent imports must run after AppLauncher.
    from env.global_configs import BENCHMARK, ENV_CONFIG_PATH, ROOT_DIR  # noqa: E402
    from omegaconf import OmegaConf  # noqa: E402
    from src.collect_client.collect_env import create_collect_env  # noqa: E402
    from utils.load_file import load_yaml  # noqa: E402
    from utils.pipeline_utils import (  # noqa: E402
        process_config,
        process_randomization,
        resolve_random_task_num_envs,
    )

    def _build_env_cfg() -> Any:
        task_registry = __import__(
            f"task.{BENCHMARK}.task_registry", fromlist=["task_config_path"]
        )
        collect_cfg = load_yaml(
            os.path.join(ENV_CONFIG_PATH, _args.env_cfg_type + ".yml")
        )
        vision_cfg = collect_cfg.setdefault("observation", {}).setdefault("vision", {})
        vision_cfg["depth"] = True
        vision_cfg["intrinsic_matrix"] = True
        vision_cfg["extrinsic_matrix"] = True
        collect_cfg["task_name"] = _args.task
        collect_cfg["num_envs"] = _args.num_envs
        collect_cfg["device_id"] = _args.device_id
        collect_cfg["save_dir"] = "/tmp/rpent_robodojo"
        camera_cfg = load_yaml(
            os.path.join(
                ENV_CONFIG_PATH, "camera", collect_cfg["config"]["camera"] + ".yml"
            )
        )
        camera_cfg = OmegaConf.merge(
            camera_cfg,
            OmegaConf.create(
                {
                    "annotator": {
                        "common": {
                            "distance_to_image_plane_capture": {
                                "type": "distance_to_image_plane",
                                "device": "cpu",
                            }
                        },
                        "cam_head": {
                            "distance_to_image_plane_capture": {
                                "type": "distance_to_image_plane",
                                "device": "cpu",
                            }
                        },
                        "cam_left_wrist": {
                            "distance_to_image_plane_capture": {
                                "type": "distance_to_image_plane",
                                "device": "cpu",
                            }
                        },
                        "cam_right_wrist": {
                            "distance_to_image_plane_capture": {
                                "type": "distance_to_image_plane",
                                "device": "cpu",
                            }
                        },
                    }
                }
            ),
        )
        env_cfg = OmegaConf.create(
            {
                "sim": load_yaml(
                    os.path.join(
                        ENV_CONFIG_PATH,
                        "sim",
                        collect_cfg["config"]["sim"] + ".yml",
                    )
                ),
                "scene": load_yaml(
                    os.path.join(
                        ENV_CONFIG_PATH,
                        "scene",
                        collect_cfg["config"]["scene"] + ".yml",
                    )
                ),
                "camera": camera_cfg,
                "robot": load_yaml(
                    os.path.join(
                        ENV_CONFIG_PATH,
                        "robot",
                        collect_cfg["config"]["robot"] + ".yml",
                    )
                ),
                "task_env": load_yaml(
                    task_registry.task_config_path(
                        os.path.join(ROOT_DIR, "task", BENCHMARK, "config"),
                        _args.task,
                    )
                ),
                "collect_cfg": collect_cfg,
                "eval_cfg": collect_cfg,
            }
        )
        capped = resolve_random_task_num_envs(_args.task, _args.num_envs, env_cfg.sim)
        OmegaConf.update(env_cfg, "sim.scene.num_envs", capped, force_add=True)
        OmegaConf.update(env_cfg, "collect_cfg.num_envs", capped, force_add=True)
        env_cfg = process_randomization(env_cfg)
        env_cfg, _ = process_config(env_cfg, task_name=_args.task)
        OmegaConf.update(
            env_cfg,
            "camera.default_frequency",
            collect_cfg["observation"].get("collect_freq", 0),
            force_add=True,
        )
        env_cfg.sim.seed = [0 for _ in range(capped)]
        return env_cfg

    env = create_collect_env(_build_env_cfg(), simulation_app)
    if _args.random:
        _random_reset()
    else:
        _standard_reset()
    video_recorder = _VideoRecorder(_args.video_dir or "")

    facade = RoboDojoEnvFacade()
    facade.serve(
        transport=_args.transport,
        host=_args.host,
        port=_args.port,
        parent_watch=_args.parent_watch,
    )


if __name__ == "__main__":
    main()
