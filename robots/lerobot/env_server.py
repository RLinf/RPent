"""Standalone LeRobot SO101 env host for the LLM-in-the-loop agent (env-only).

Drives a physical SO101 follower arm through LeRobot's synchronous Python
API (:class:`lerobot.robots.so_follower.SO101Follower`) and exposes a minimal
``reset`` / ``step`` gym-style surface over a pickle-framed TCP RPC server
(:class:`rpent.rpc_driver.socket.SocketRpcServer`) — the same wire
protocol the LIBERO driver uses, so the agent side talks to both identically.

Unlike the LIBERO driver, this server does **not** wrap an RLinf env class:
importing ``rlinf.envs.realworld`` runs node-level ROS setup side effects at
import time (it kills ``roscore``/``rosmaster`` processes), which is
inappropriate for a standalone driver. Instead this file talks to LeRobot
directly, reusing the device / action / observation recipe from RLinf's
``rlinf.envs.realworld.so101.SO101Env``.

Run it inside the ``lerobot`` conda env::

    conda activate lerobot
    python robots/lerobot/env_server.py --output-dir /tmp/so101_run

Hardware defaults match the current bench setup: follower on ``/dev/ttyACM1``
(calibration id ``my_awesome_follower_arm``), an OpenCV hand/arm camera on
``/dev/video2``, and an Intel RealSense D405 scene camera (serial
``409122274720``). Every default is overridable from the CLI.

Launched manually for now; wiring into ``cli/main.py`` (per-env client class +
driver script selection) is a separate step.
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import sys
import threading
import time
from pathlib import Path

import numpy as np

# Make ``rpent`` importable when this file is run as a script from
# the ``lerobot`` conda env (which need not have rpent installed).
_PHYSICALAGENT_ROOT = Path(__file__).resolve().parents[2]
if str(_PHYSICALAGENT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PHYSICALAGENT_ROOT))

from rpent.rpc_driver.socket import SocketRpcServer  # noqa: E402
from rpent.utils.logging import get_logger, init_output_dir  # noqa: E402

from robots.lerobot import calibration as scene_calib  # noqa: E402
from robots.lerobot import geometry as geom  # noqa: E402
from robots.lerobot.kinematics import SO101Kinematics  # noqa: E402
from robots.lerobot.scene_camera import SceneCameraD405  # noqa: E402

logger = get_logger("lerobot_driver")


# ---------------------------------------------------------------------------
# SO101 joint / limit constants (mirrors rlinf SO101Env defaults)
# ---------------------------------------------------------------------------

# Arm joints in bus-ID order; LeRobot keys obs/action by ``<name>.pos``.
_ARM_JOINTS = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
)
_GRIPPER = "gripper"
_NUM_ARM_JOINTS = len(_ARM_JOINTS)

# Arm joint limits in degrees (arm only, no gripper), aligned to the SO101 URDF
# (deg = rad * 180/pi). Matching the URDF matters for ``move_to``'s top-down
# mode: its IK solutions use wrist_flex up to ~95 deg and wrist_roll beyond
# +/-150 deg, so a tighter clip here would silently alter the solved pose (and
# re-tilt the gripper). placo already enforces these limits; this clip is a
# secondary safety net.
_JOINT_LIMIT_LOW_DEG = np.array([-110.0, -100.0, -96.8, -95.0, -157.2], dtype=np.float32)
_JOINT_LIMIT_HIGH_DEG = np.array([110.0, 100.0, 96.8, 95.0, 162.8], dtype=np.float32)
_GRIPPER_LIMIT_LOW = 0.0
_GRIPPER_LIMIT_HIGH = 90.0
# Home pose ``[q1..q5, gripper]``. The five arm joints use the all-zero
# calibrated pose (as requested). The gripper homes to a mid opening, NOT 0:
# driving the gripper to 0 stalls it closed against its mechanical stop and
# trips the motor's overload protection (observed: gripper motor id 6 dropped
# off the bus after a 0 command). Keep this strictly between the limits.
_RESET_QPOS = np.array([0.0, -100.0, 90.0, 65.0, 0.0, 50.0], dtype=np.float32)

# Conservative tabletop workspace box in the base (world) frame, meters.
# ``move_to`` clips targets to this so an LLM-supplied coordinate can't drive
# the arm into the table / out of reach. Sized to the SO101's ~0.35 m reach and
# the observed pick region (the base origin sits above the table, so the plate/
# table surface is around z ~ -0.06): x forward, y lateral, z up. move_to now
# closes the loop and actually reaches the commanded z, so the z floor must stop
# the fingertips just ABOVE the plate (grasps straddle the cube's upper body).
# Tune for your bench.
_WORKSPACE_MIN = np.array([0.08, -0.28, -0.055], dtype=np.float64)
_WORKSPACE_MAX = np.array([0.38, 0.28, 0.30], dtype=np.float64)

# --- grasp control point (TCP) ------------------------------------------
# IK/FK use the ``gripper_frame_link`` frame, which actually sits on the FIXED
# jaw's fingertip -- ~2.5 cm ABOVE the fingertip plane and offset toward the
# fixed side, NOT between the fingers. This vector (meters, in the
# gripper_frame_link LOCAL frame) shifts the controlled/reported point to the
# grasp point between the fingertips, so ``move_to`` targets and ``get_ee_pose``
# refer to where an object is actually grasped. Derived from URDF + FK + a
# calibrated-depth fingertip measurement (~2 cm lateral toward the moving jaw,
# ~2.8 cm down to the fingertip plane). TUNE on hardware / via touch calibration
# if grasps land consistently off-centre; set to zeros for the raw frame.
_TCP_OFFSET_GRIPPER = np.array([-0.028, 0.043, 0.0282], dtype=np.float64)

# --- motion speed / smoothness (safety) ---------------------------------
# The arm runs in position mode. Two knobs keep motions slow and gentle:
#  * Servo acceleration: LeRobot's configure() maxes the Feetech "Acceleration"
#    register at 254 (snappy). We override it with a gentler value (0-254;
#    lower = softer ramps), applied to every motor after connect.
#  * Software velocity cap: point-to-point motions (reset, move_to,
#    move_joints_delta) are streamed as interpolated setpoints so no joint
#    exceeds ``_MAX_JOINT_VEL_DEG_S`` (deg/s), instead of snapping to the target
#    at full servo speed. Both are overridable from the CLI.
_MOTOR_ACCELERATION = 40
_MAX_JOINT_VEL_DEG_S = 70.0
_PACE_DT_S = 0.05  # setpoint-streaming period (20 Hz)
# Feetech position gain. LeRobot's configure() lowers P_Coefficient to 16 (from
# the factory default 32) "to avoid shakiness"; that soft gain lets gravity-
# loaded joints (esp. elbow_flex) under-reach their commanded angle, so the
# gripper lands short and low. We raise it so the arm actually holds commanded
# joints. Overridable from the CLI (raise for stiffer tracking; lower if the
# arm oscillates/buzzes). None = leave LeRobot's value untouched.
_POSITION_GAIN = 32

# move_to interpolates the full gripper POSE into fine steps (straight-line
# position + slerped orientation) and solves IK warm-started at each, so the tip
# tracks the Cartesian line and the wrist reorients smoothly -- avoiding wrist
# IK-branch flips that otherwise swing the long gripper through the table.
_CART_STEP_M = 0.01  # max Cartesian move per interpolated IK step
_REORIENT_STEP_DEG = 6.0  # max orientation change per interpolated IK step
# If an interpolated step's IK solution jumps more than this (deg) from the
# previous one, the path crossed an IK discontinuity (near-singular / infeasible
# top-down pose). move_to stops at the last safe pose rather than streaming the
# arm through the swing, which could drive the long gripper into the table.
_MAX_STEP_JOINT_JUMP_DEG = 20.0
# Closed-loop correction: after the open-loop path, the real servos can settle
# short of the commanded pose (gravity sag under load), so move_to re-commands
# with Cartesian feed-forward until the achieved tip is within _CORRECTION_TOL_M
# or the budget runs out. Feed-forward is capped and workspace-clipped for
# safety (it can never drive below the workspace floor / into the plate).
_MAX_POSITION_CORRECTIONS = 3
_CORRECTION_TOL_M = 0.008
_MAX_CORRECTION_M = 0.06


def _build_camera_configs(raw: dict[str, dict]) -> dict:
    """Build LeRobot ``CameraConfig`` instances from user-facing dicts.

    Each value must contain a ``"type"`` key (``"opencv"`` or
    ``"intelrealsense"``); the remaining keys are forwarded to the matching
    ``CameraConfig`` subclass. Mirrors ``SO101Env._build_camera_configs``:
    the config subclasses self-register on import, so the two camera modules
    must be imported before ``get_choice_class`` can resolve the type name.
    """
    import lerobot.cameras.opencv.configuration_opencv  # noqa: F401  registers "opencv"
    import lerobot.cameras.realsense.configuration_realsense  # noqa: F401  registers "intelrealsense"
    from lerobot.cameras.configs import CameraConfig

    result: dict = {}
    for name, cfg in raw.items():
        cfg = dict(cfg)
        cam_type = cfg.pop("type")
        result[name] = CameraConfig.get_choice_class(cam_type)(**cfg)
    return result


def _to_lerobot_action(arm_targets: np.ndarray, gripper_target: float) -> dict:
    """Build the ``{motor}.pos`` dict LeRobot's ``send_action`` expects.

    LeRobot filters incoming keys via ``key.endswith(".pos")``; a missing
    suffix silently drops that motor.
    """
    action = {f"{name}.pos": float(arm_targets[i]) for i, name in enumerate(_ARM_JOINTS)}
    action[f"{_GRIPPER}.pos"] = float(gripper_target)
    return action


def _topdown_rotation(yaw: float) -> np.ndarray:
    """Target gripper rotation for a top-down approach (in the base frame).

    Maps the gripper's local approach axis (local +z, the wrist->fingertips
    direction) to world -z (straight down). ``yaw`` (radians) rotates the
    gripper about the vertical: the gripper's local +x maps to the horizontal
    direction ``(cos yaw, sin yaw, 0)``, which sets the jaw-line heading.
    """
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array(
        [[c, s, 0.0], [s, -c, 0.0], [0.0, 0.0, -1.0]], dtype=np.float64
    )


def _grasp_point(T_base_gripper: np.ndarray) -> np.ndarray:
    """World xyz of the grasp point (between the fingertips) for a
    ``gripper_frame_link`` pose, applying ``_TCP_OFFSET_GRIPPER`` in the gripper
    frame. ``gripper_frame_link`` itself is on the fixed jaw, so this is what
    ``move_to`` should target and ``get_ee_pose`` should report for grasping.
    """
    T = np.asarray(T_base_gripper, dtype=np.float64)
    return T[:3, 3] + T[:3, :3] @ _TCP_OFFSET_GRIPPER


def _approach_tilt_deg(R: np.ndarray) -> float:
    """Angle (deg) between the gripper approach axis (local +z) and world -z.

    0 deg = pointing straight down. Used to verify a top-down ``move_to``.
    """
    z_world = np.asarray(R, dtype=np.float64) @ np.array([0.0, 0.0, 1.0])
    return float(np.degrees(np.arccos(np.clip(-z_world[2], -1.0, 1.0))))


def _rotation_angle_deg(R0: np.ndarray, R1: np.ndarray) -> float:
    """Geodesic angle (deg) between two rotation matrices."""
    cos = (
        np.trace(np.asarray(R0, dtype=np.float64).T @ np.asarray(R1, dtype=np.float64))
        - 1.0
    ) / 2.0
    return float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))


def _slerp_rotation(R0: np.ndarray, R1: np.ndarray, t: float) -> np.ndarray:
    """Interpolate rotation ``R0`` -> ``R1`` by fraction ``t`` in [0, 1].

    Rotates about the fixed axis of the relative rotation (Rodrigues) -- a
    matrix slerp -- so the gripper reorients along one smooth shortest arc.
    """
    R0 = np.asarray(R0, dtype=np.float64)
    R1 = np.asarray(R1, dtype=np.float64)
    R_rel = R0.T @ R1
    ang = np.arccos(np.clip((np.trace(R_rel) - 1.0) / 2.0, -1.0, 1.0))
    if ang < 1e-8:
        return R0.copy()
    axis = np.array(
        [
            R_rel[2, 1] - R_rel[1, 2],
            R_rel[0, 2] - R_rel[2, 0],
            R_rel[1, 0] - R_rel[0, 1],
        ],
        dtype=np.float64,
    ) / (2.0 * np.sin(ang))
    th = ang * float(t)
    K = np.array(
        [
            [0.0, -axis[2], axis[1]],
            [axis[2], 0.0, -axis[0]],
            [-axis[1], axis[0], 0.0],
        ],
        dtype=np.float64,
    )
    return R0 @ (np.eye(3) + np.sin(th) * K + (1.0 - np.cos(th)) * (K @ K))


class SO101LeRobotEnv:
    """Minimal ``reset`` / ``step`` driver for a physical SO101 arm.

    Action: ``(6,)`` float array ``[q1..q5, gripper]`` of absolute joint
    position targets in degrees. Arm targets are clipped to the configured
    joint limits and the gripper to ``[0, 90]`` before being sent to the
    motor bus.

    Observation::

        {"state": {"joint_position":   (5,) float32,   # arm joints, degrees
                   "gripper_position": (1,) float32,   # gripper opening
                   "ee_pose_base":     (3,) float32,   # gripper xyz in base (FK)
                   "ee_quat_base":     (4,) float32},  # gripper quat wxyz (FK)
         "frames": {"arm": (H, W, 3) uint8, "scene": (H, W, 3) uint8},
         "depth":  {"scene": (H, W) float32}}            # metric, aligned to color

    ``ee_pose_base`` / ``ee_quat_base`` are present only when FK is available;
    ``scene`` frames/depth only when the scene camera is configured. All values
    are plain numpy / floats so they pickle across the RPC wire (the agent
    process does not import torch).
    """

    def __init__(
        self,
        *,
        port: str,
        calibration_id: str,
        arm_camera_cfgs: dict[str, dict],
        scene_serial: str | None = None,
        scene_size: tuple[int, int] = (720, 1280),
        scene_fps: int = 30,
        urdf_path: str | None = None,
        max_relative_target: float | None = None,
        max_episode_steps: int = 200,
        step_frequency: float = 30.0,
        motor_acceleration: int | None = _MOTOR_ACCELERATION,
        max_joint_vel_deg_s: float = _MAX_JOINT_VEL_DEG_S,
        position_gain: int | None = _POSITION_GAIN,
        auto_calibrate: bool = False,
    ) -> None:
        self._max_episode_steps = max_episode_steps
        self._step_frequency = step_frequency
        self._max_joint_vel_deg_s = float(max_joint_vel_deg_s)
        self._pace_dt = _PACE_DT_S
        self._num_steps = 0
        self._action_low = np.append(_JOINT_LIMIT_LOW_DEG, _GRIPPER_LIMIT_LOW).astype(np.float32)
        self._action_high = np.append(_JOINT_LIMIT_HIGH_DEG, _GRIPPER_LIMIT_HIGH).astype(np.float32)

        from lerobot.robots.so_follower import SO101Follower
        from lerobot.robots.so_follower.config_so_follower import SO101FollowerConfig

        robot_cfg = SO101FollowerConfig(
            port=port,
            id=calibration_id,
            use_degrees=True,
            max_relative_target=max_relative_target,
            # Keep torque on at disconnect so the arm holds its parked pose.
            disable_torque_on_disconnect=False,
            cameras=_build_camera_configs(arm_camera_cfgs),
        )
        self._robot = SO101Follower(robot_cfg)
        # ``calibrate=False`` loads the on-disk calibration without ever
        # prompting on stdin (a server must never block on input).
        self._robot.connect(calibrate=auto_calibrate)
        self._arm_camera_names = tuple(arm_camera_cfgs.keys())

        # Soften motion: LeRobot's configure() sets the Feetech "Acceleration"
        # register to its max (254). Override with a gentler value so the arm
        # ramps smoothly rather than snapping (safety). Done with torque briefly
        # off, mirroring LeRobot's own register writes.
        self._motor_acceleration = motor_acceleration
        if motor_acceleration is not None:
            try:
                with self._robot.bus.torque_disabled():
                    for motor in self._robot.bus.motors:
                        # Keep the gripper snappy so grasps close promptly; only
                        # slow the (heavier, safety-relevant) arm joints.
                        if motor == _GRIPPER:
                            continue
                        self._robot.bus.write(
                            "Acceleration", motor, int(motor_acceleration)
                        )
            except Exception as e:
                logger.warning(
                    "could not set motor Acceleration=%s (motions stay fast): %s",
                    motor_acceleration, e,
                )

        # Stiffen position holding: LeRobot lowers P_Coefficient to 16, which
        # lets gravity-loaded arm joints under-reach their target. Raise it on
        # the arm joints (leave the gripper as LeRobot set it).
        self._position_gain = position_gain
        if position_gain is not None:
            try:
                with self._robot.bus.torque_disabled():
                    for motor in self._robot.bus.motors:
                        if motor == _GRIPPER:
                            continue
                        self._robot.bus.write(
                            "P_Coefficient", motor, int(position_gain)
                        )
            except Exception as e:
                logger.warning(
                    "could not set motor P_Coefficient=%s (tracking may be soft): %s",
                    position_gain, e,
                )

        # Scene camera (fixed, depth) is managed directly via pyrealsense2 so we
        # get depth aligned to color + intrinsics (LeRobot's wrapper gives
        # neither). The arm camera stays under LeRobot above (color only).
        self._scene_serial = scene_serial or None
        self._scene_cam: SceneCameraD405 | None = None
        if self._scene_serial:
            scene_h, scene_w = scene_size
            self._scene_cam = SceneCameraD405(
                self._scene_serial, width=scene_w, height=scene_h, fps=scene_fps,
            )

        # Forward kinematics for end-effector pose in the base (world) frame.
        self._kin: SO101Kinematics | None = None
        try:
            self._kin = SO101Kinematics(urdf_path=urdf_path)
        except Exception as e:
            logger.warning("FK unavailable (%s); ee_pose will be omitted", e)

        # Scene-cam -> base extrinsic, if it has been calibrated (touch/Kabsch).
        # A fit whose saved RMSE is too large is REJECTED (treated as
        # uncalibrated): a bad extrinsic makes back_project return badly wrong
        # world coordinates, so the arm would chase unreachable targets. The
        # operator must recalibrate rather than have the agent act on garbage.
        self._T_base_cam = None
        self._calib_rmse_m: float | None = None
        if self._scene_serial:
            record = scene_calib.load_extrinsic_record(self._scene_serial)
            if record is not None:
                self._calib_rmse_m = record.get("rmse_m")
                rmse = self._calib_rmse_m
                if rmse is not None and rmse > scene_calib.MAX_ACCEPTABLE_RMSE_M:
                    logger.warning(
                        "scene-cam extrinsic REJECTED: rmse=%.3fm > %.3fm limit; "
                        "back_project would be unreliable. Recalibrate with "
                        "robots/lerobot/auto_calibrate_scene_cam.py "
                        "(or calibrate_scene_cam.py).",
                        rmse, scene_calib.MAX_ACCEPTABLE_RMSE_M,
                    )
                else:
                    self._T_base_cam = record["T_base_cam"]

        if self._T_base_cam is not None:
            extr_status = (
                f"loaded (rmse={self._calib_rmse_m:.3f}m)"
                if self._calib_rmse_m is not None else "loaded"
            )
        elif self._calib_rmse_m is not None:
            extr_status = f"rejected (rmse={self._calib_rmse_m:.3f}m)"
        else:
            extr_status = "uncalibrated"

        cam_list = list(self._arm_camera_names) + (["scene"] if self._scene_cam else [])
        logger.info(
            "SO101 connected on %s (calibration_id=%s); cameras=[%s]; "
            "FK=%s; scene_extrinsic=%s",
            port, calibration_id, ", ".join(cam_list) or "none",
            "on" if self._kin else "off",
            extr_status,
        )
        logger.info(
            "motion pacing: max_joint_vel=%.0f deg/s; motor_acceleration=%s",
            self._max_joint_vel_deg_s,
            self._motor_acceleration if self._motor_acceleration is not None
            else "default (254)",
        )
        logger.info(
            "position gain: P_Coefficient=%s",
            self._position_gain if self._position_gain is not None
            else "default (LeRobot 16)",
        )

    # ------------------------------------------------------------------
    # gym-like surface
    # ------------------------------------------------------------------

    def _stream_joint_path(self, q_from, q_to, gripper_value) -> None:
        """Glide the arm from ``q_from`` to ``q_to`` (arm-joint vectors, degrees).

        Streams interpolated joint setpoints so no joint moves faster than
        ``self._max_joint_vel_deg_s`` (deg/s), instead of commanding the target
        in one shot and letting the servos snap there at full speed. The gripper
        is held at ``gripper_value`` throughout.
        """
        q_from = np.asarray(q_from, dtype=np.float64).reshape(-1)[:_NUM_ARM_JOINTS]
        q_to = np.asarray(q_to, dtype=np.float64).reshape(-1)[:_NUM_ARM_JOINTS]
        max_delta = float(np.max(np.abs(q_to - q_from))) if q_to.size else 0.0
        per_step = max(1e-6, self._max_joint_vel_deg_s * self._pace_dt)
        n = max(1, int(np.ceil(max_delta / per_step)))
        for i in range(1, n + 1):
            q = q_from + (q_to - q_from) * (i / n)
            self._robot.send_action(_to_lerobot_action(q, gripper_value))
            time.sleep(self._pace_dt)

    def reset(self) -> tuple[dict, dict]:
        """Send the arm to its rest pose (paced), reset the counter, return obs."""
        self._num_steps = 0
        obs = self._robot.get_observation()
        cur = np.array(
            [obs.get(f"{n}.pos", 0.0) for n in _ARM_JOINTS], dtype=np.float64
        )
        self._stream_joint_path(
            cur, _RESET_QPOS[:_NUM_ARM_JOINTS], float(_RESET_QPOS[_NUM_ARM_JOINTS])
        )
        time.sleep(0.3)
        return self._get_observation(), {}

    def step(self, action) -> tuple[dict, float, bool, bool, dict]:
        """Command one absolute joint target and return the gym 5-tuple.

        Returns ``(obs, reward, terminated, truncated, info)``. This minimal
        driver computes no task reward (``reward == 0.0``) and never
        auto-terminates; ``truncated`` flips once ``max_episode_steps`` is
        reached. Task success / stopping is decided by the agent.
        """
        t0 = time.time()
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        expected = _NUM_ARM_JOINTS + 1
        if action.shape[0] != expected:
            raise ValueError(
                f"action must have {expected} entries [q1..q5, gripper]; "
                f"got {action.shape[0]}"
            )
        action = np.clip(action, self._action_low, self._action_high)
        self._robot.send_action(
            _to_lerobot_action(action[:_NUM_ARM_JOINTS], float(action[_NUM_ARM_JOINTS]))
        )
        self._num_steps += 1

        obs = self._get_observation()
        truncated = self._num_steps >= self._max_episode_steps

        # Pace the control loop to the requested frequency.
        dt = time.time() - t0
        time.sleep(max(0.0, (1.0 / self._step_frequency) - dt))
        return obs, 0.0, False, truncated, {}

    def get_spec(self) -> dict:
        """Static self-description for the agent side (action bounds, cams)."""
        cam_names = list(self._arm_camera_names) + (["scene"] if self._scene_cam else [])
        return {
            "action_dim": _NUM_ARM_JOINTS + 1,
            "arm_joints": list(_ARM_JOINTS),
            "action_low": self._action_low.tolist(),
            "action_high": self._action_high.tolist(),
            "camera_names": cam_names,
            "scene_camera": "scene" if self._scene_cam else None,
            "has_ee_pose": self._kin is not None,
            "world_frame": "base_link",
            "max_episode_steps": self._max_episode_steps,
        }

    # ------------------------------------------------------------------
    # localization surface (base frame == world)
    # ------------------------------------------------------------------

    def get_ee_pose(self) -> dict:
        """Live FK: gripper pose in the base (world) frame."""
        obs = self._robot.get_observation()
        joints = np.array(
            [obs.get(f"{n}.pos", 0.0) for n in _ARM_JOINTS], dtype=np.float32
        )
        ee = self._compute_ee_pose(joints)
        if ee is None:
            return {"error": "FK unavailable (URDF/placo missing)"}
        return {
            "xyz": _grasp_point(ee["T"]).tolist(),
            "quat_wxyz": ee["quat"].tolist(),
            "joints_deg": joints.tolist(),
            "T_base_gripper": ee["T"].tolist(),
            "frame": "base_link",
        }

    def get_scene_camera_meta(self) -> dict:
        """Scene-camera intrinsics + depth scale + base extrinsic (if any)."""
        if self._scene_cam is None:
            return {"error": "scene camera not configured"}
        meta = self._scene_cam.meta()
        meta["frame"] = "scene_cam"
        meta["calibrated"] = self._T_base_cam is not None
        meta["rmse_m"] = self._calib_rmse_m
        meta["T_base_cam"] = (
            self._T_base_cam.tolist() if self._T_base_cam is not None else None
        )
        return meta

    def get_scene_frame(self) -> dict:
        """Live scene color + metric depth (for calibration / ad-hoc queries)."""
        if self._scene_cam is None:
            return {"error": "scene camera not configured"}
        rgb, depth = self._scene_cam.read()
        return {"color": rgb, "depth": depth, "K": self._scene_cam.K.tolist()}

    def get_obs(self) -> dict:
        """Return the current observation without moving the arm.

        Used to refresh the agent-side cache after a primitive (e.g. move_to)
        that changes the world but does not itself return an observation.
        """
        return self._get_observation()

    def set_torque(self, enabled: bool) -> dict:
        """Enable/disable arm motor torque.

        Disabling lets the operator free-drive the arm by hand (used by the
        touch/Kabsch scene-camera calibration). Joint encoders remain readable
        with torque off, so FK still works. Re-enable to hold position.
        """
        bus = self._robot.bus
        try:
            if enabled:
                bus.enable_torque()
            else:
                bus.disable_torque()
        except Exception as e:
            return {"ok": False, "error": str(e)}
        logger.info("arm torque %s", "enabled" if enabled else "disabled")
        return {"ok": True, "torque_enabled": bool(enabled)}

    def _read_arm_joints(self) -> np.ndarray:
        """Current arm joint angles (deg) from a fresh observation."""
        obs = self._robot.get_observation()
        return np.array(
            [obs.get(f"{n}.pos", 0.0) for n in _ARM_JOINTS], dtype=np.float64
        )

    def _servo_pose_path(
        self, from_joints, R_start, p_start, R_end, p_end, grip_val, orient_w,
        settle_s,
    ) -> bool:
        """Stream the gripper along an interpolated pose (straight-line position
        + slerped orientation), warm-starting IK per fine step so solutions stay
        on ONE branch and the tip tracks the line. Returns ``halted`` (an IK
        discontinuity forced an early stop). Settles ``settle_s`` before
        returning so the caller can measure the achieved pose.
        """
        from_joints = np.asarray(from_joints, dtype=np.float64)
        p_start = np.asarray(p_start, dtype=np.float64)
        p_end = np.asarray(p_end, dtype=np.float64)
        dist = float(np.linalg.norm(p_end - p_start))
        reorient_deg = _rotation_angle_deg(R_start, R_end) if orient_w > 0 else 0.0
        n_steps = max(
            1,
            int(np.ceil(dist / _CART_STEP_M)),
            int(np.ceil(reorient_deg / _REORIENT_STEP_DEG)),
        )
        seed = from_joints.copy()
        prev_q = from_joints.copy()
        halted = False
        for i in range(1, n_steps + 1):
            frac = i / n_steps
            T_des = np.eye(4)
            T_des[:3, :3] = (
                _slerp_rotation(R_start, R_end, frac) if orient_w > 0 else R_end
            )
            T_des[:3, 3] = p_start + (p_end - p_start) * frac
            q = self._kin.ik(
                seed, T_des, position_weight=1.0, orientation_weight=orient_w
            )
            q_arm = np.clip(
                q[:_NUM_ARM_JOINTS], _JOINT_LIMIT_LOW_DEG.astype(np.float64),
                _JOINT_LIMIT_HIGH_DEG.astype(np.float64),
            )
            # A large jump between consecutive fine-step solutions = the path
            # crossed an IK discontinuity (near-singular / infeasible top-down
            # pose); stop at the last safe pose rather than swinging through it.
            if float(np.max(np.abs(q_arm - prev_q))) > _MAX_STEP_JOINT_JUMP_DEG:
                halted = True
                break
            seed = q_arm
            self._stream_joint_path(prev_q, q_arm, grip_val)
            prev_q = q_arm
        time.sleep(settle_s)
        return halted

    def move_to(
        self,
        xyz,
        *,
        gripper: float | None = None,
        approach: str = "free",
        yaw_deg: float | None = None,
        settle_s: float = 0.4,
        pos_tol_m: float = 0.02,
        tilt_tol_deg: float = 15.0,
        max_corrections: int = _MAX_POSITION_CORRECTIONS,
    ) -> dict:
        """Move the gripper to a world-frame (base_link) XYZ via IK.

        ``approach`` selects the wrist-orientation policy:

        * ``"free"`` (default): position-only IK; the wrist settles at whatever
          orientation placo converges to. Maximal reach, but the fingertips'
          location relative to the returned EE point is unpredictable (the TCP
          is ~0.1 m out along the gripper axis), so it is unreliable for
          grasping.
        * ``"down"``: top-down IK -- the gripper approach axis is driven to
          vertical (pointing straight down) so the fingertips descend along
          world -z, which makes grasping predictable. ``yaw_deg`` sets the
          jaw-line heading about the vertical (0 = +x/forward); if ``None`` a
          reachable yaw is searched automatically.

        The target is clipped to the workspace box and approached by
        interpolating the full gripper pose (straight-line position + smoothly
        slerped orientation) into fine, warm-started IK steps. It then CLOSES
        THE LOOP: it measures the achieved tip and re-commands with feed-forward
        (up to ``max_corrections`` times) to cancel the servos' steady-state sag
        under load, so the tip lands on the commanded xyz -- callers should pass
        the true target, NOT a hand-tuned over-shoot. Holds the current gripper
        opening unless ``gripper`` is given.

        Returns a log dict with ``reached`` (position within ``pos_tol_m`` and,
        for ``"down"``, approach tilt within ``tilt_tol_deg``), the
        commanded/achieved xyz, the position error, and the achieved approach
        tilt in degrees.
        """
        if self._kin is None:
            return {"error": "IK unavailable (URDF/placo missing)"}
        approach = str(approach).lower()
        if approach not in ("free", "down"):
            return {"error": f"approach must be 'free' or 'down'; got {approach!r}"}
        target = np.asarray(xyz, dtype=np.float64).reshape(-1)
        if target.shape[0] != 3:
            return {"error": f"xyz must be 3 numbers; got {target.shape[0]}"}

        clipped = np.clip(target, _WORKSPACE_MIN, _WORKSPACE_MAX)
        was_clipped = not np.allclose(clipped, target, atol=1e-6)

        obs = self._robot.get_observation()
        cur_joints = np.array(
            [obs.get(f"{n}.pos", 0.0) for n in _ARM_JOINTS], dtype=np.float64
        )
        cur_gripper = float(obs.get(f"{_GRIPPER}.pos", 0.0))
        grip_val = cur_gripper if gripper is None else float(gripper)

        T_cur = self._kin.fk(cur_joints)
        p0 = T_cur[:3, 3]
        R0 = T_cur[:3, :3]

        # Target orientation for the move: reorient to vertical for "down";
        # leave it to the IK (weight 0) for "free".
        if approach == "down":
            orient_w = 1.0
            yaw = (
                self._search_topdown_yaw(cur_joints, clipped)
                if yaw_deg is None else float(np.radians(yaw_deg))
            )
            R_target = _topdown_rotation(yaw)
        else:
            orient_w = 0.0
            yaw = None
            R_target = R0  # ignored (orientation_weight=0)

        # Held orientation: top-down for "down", the current (ignored) rotation
        # for "free". Move the tip along the interpolated pose, then close the
        # loop to cancel the servos' steady-state sag under load.
        R_hold = R_target if approach == "down" else R0
        # move_to controls the GRASP POINT (between the fingertips). Convert that
        # target into the gripper_frame_link goal the IK/servo actually drives.
        # The offset is only well-defined for the fixed "down" orientation;
        # "free" leaves the tip frame uncorrected (its final orientation, hence
        # the offset direction, is unknown until IK converges).
        tcp_off_world = (
            R_hold @ _TCP_OFFSET_GRIPPER if approach == "down"
            else np.zeros(3, dtype=np.float64)
        )
        halted = self._servo_pose_path(
            cur_joints, R0, p0, R_hold, clipped - tcp_off_world,
            grip_val, orient_w, settle_s,
        )
        final_joints = self._read_arm_joints()
        T_final = self._kin.fk(final_joints)
        grasp_final = _grasp_point(T_final) if approach == "down" else T_final[:3, 3]

        # Closed-loop correction: the real arm can settle short of the commanded
        # pose (gravity sag), even though the IK path is exact. Feed the residual
        # forward and re-command (workspace-clipped + capped) until the achieved
        # GRASP POINT is within tolerance or the correction budget runs out.
        n_corr = 0
        ff = np.zeros(3, dtype=np.float64)
        while (
            not halted
            and n_corr < max_corrections
            and float(np.linalg.norm(clipped - grasp_final)) > _CORRECTION_TOL_M
        ):
            ff = np.clip(
                ff + (clipped - grasp_final), -_MAX_CORRECTION_M, _MAX_CORRECTION_M
            )
            corr_goal = (
                np.clip(clipped + ff, _WORKSPACE_MIN, _WORKSPACE_MAX) - tcp_off_world
            )
            halted = self._servo_pose_path(
                final_joints, R_hold, T_final[:3, 3], R_hold, corr_goal,
                grip_val, orient_w, settle_s,
            )
            final_joints = self._read_arm_joints()
            T_final = self._kin.fk(final_joints)
            grasp_final = _grasp_point(T_final) if approach == "down" else T_final[:3, 3]
            n_corr += 1

        err = float(np.linalg.norm(grasp_final - clipped))
        tilt = _approach_tilt_deg(T_final[:3, :3])
        reached = (
            not halted
            and err <= pos_tol_m
            and (approach != "down" or tilt <= tilt_tol_deg)
        )
        result = {
            "reached": bool(reached),
            "approach": approach,
            "target_xyz": [round(float(v), 4) for v in target],
            "commanded_xyz": [round(float(v), 4) for v in clipped],
            "final_xyz": [round(float(v), 4) for v in grasp_final],
            "pos_error_m": round(err, 4),
            "approach_tilt_deg": round(tilt, 1),
            "yaw_deg": (None if yaw is None else round(float(np.degrees(yaw)), 1)),
            "clipped_to_workspace": was_clipped,
            "pos_corrections": n_corr,
            "halted_early": bool(halted),
            "joints_deg": [round(float(v), 2) for v in final_joints],
            "gripper": round(grip_val, 2),
        }
        if halted:
            result["note"] = (
                "stopped partway: the straight-line top-down path crossed an "
                "unreachable / near-singular pose. Try a nearer target, a "
                "different yaw_deg, or move in smaller hops."
            )
        elif not reached:
            result["note"] = (
                f"settled {round(err * 1000)} mm short after {n_corr} "
                "correction(s); the target may be past the arm's reach here. "
                "Try a nearer / higher target."
            )
        return result

    def _search_topdown_yaw(self, seed_joints, target, *, n_yaw: int = 12) -> float:
        """Pick a top-down wrist yaw (rad) that reaches ``target`` best.

        Whether a vertical approach is reachable depends on the jaw-line yaw
        (wrist_roll/shoulder_pan coupling), so we sweep candidate yaws, run IK
        for each, and keep the one with the smallest FK position error (ties
        broken by approach tilt). Pure CPU -- the arm does not move.
        """
        best_yaw, best_key = 0.0, None
        seed = np.asarray(seed_joints, dtype=np.float64)
        for k in range(n_yaw):
            yaw = 2.0 * np.pi * k / n_yaw
            T_des = np.eye(4)
            T_des[:3, :3] = _topdown_rotation(yaw)
            T_des[:3, 3] = target
            q = self._kin.ik(seed, T_des, position_weight=1.0, orientation_weight=1.0)
            T_q = self._kin.fk(q[:_NUM_ARM_JOINTS])
            perr = float(np.linalg.norm(T_q[:3, 3] - target))
            tilt = _approach_tilt_deg(T_q[:3, :3])
            key = (round(perr, 4), round(tilt, 1))
            if best_key is None or key < best_key:
                best_key, best_yaw = key, yaw
        return best_yaw

    def move_joints_delta(
        self,
        delta_deg,
        *,
        gripper_delta: float | None = None,
        max_step_deg: float = 15.0,
        settle_s: float = 0.4,
    ) -> dict:
        """Nudge each arm joint by a relative amount (degrees).

        ``delta_deg`` is 5 values ``[d_pan, d_lift, d_elbow, d_wrist_flex,
        d_wrist_roll]`` added to the current arm joints. Each entry is capped to
        +/- ``max_step_deg`` and the result is clamped to the joint limits, so a
        single call makes a small, safe adjustment. Optionally nudge the gripper
        by ``gripper_delta`` (clamped to its limits). Use this for fine
        alignment ``move_to`` cannot express -- e.g. tweak wrist_roll to line the
        jaws up with an object, or descend a few millimetres -- reading the new
        EE pose back from the result.

        Returns a log dict with the applied delta, achieved joints, gripper, and
        (when FK is available) the new EE xyz and approach tilt.
        """
        delta = np.asarray(delta_deg, dtype=np.float64).reshape(-1)
        if delta.shape[0] != _NUM_ARM_JOINTS:
            return {
                "error": f"delta_deg must have {_NUM_ARM_JOINTS} entries "
                f"[pan, lift, elbow, wrist_flex, wrist_roll]; got {delta.shape[0]}"
            }
        cap = abs(float(max_step_deg))
        delta = np.clip(delta, -cap, cap)

        obs = self._robot.get_observation()
        cur_joints = np.array(
            [obs.get(f"{n}.pos", 0.0) for n in _ARM_JOINTS], dtype=np.float64
        )
        cur_gripper = float(obs.get(f"{_GRIPPER}.pos", 0.0))

        target_joints = np.clip(
            cur_joints + delta,
            _JOINT_LIMIT_LOW_DEG.astype(np.float64),
            _JOINT_LIMIT_HIGH_DEG.astype(np.float64),
        )
        if gripper_delta is None:
            grip_val = cur_gripper
        else:
            grip_val = float(
                np.clip(cur_gripper + float(gripper_delta),
                        _GRIPPER_LIMIT_LOW, _GRIPPER_LIMIT_HIGH)
            )

        self._stream_joint_path(cur_joints, target_joints, grip_val)
        time.sleep(settle_s)

        final_obs = self._robot.get_observation()
        final_joints = np.array(
            [final_obs.get(f"{n}.pos", 0.0) for n in _ARM_JOINTS], dtype=np.float64
        )
        result: dict = {
            "applied_delta_deg": [round(float(v), 2) for v in delta],
            "joints_deg": [round(float(v), 2) for v in final_joints],
            "gripper": round(grip_val, 2),
        }
        ee = self._compute_ee_pose(final_joints)
        if ee is not None:
            result["ee_xyz"] = [round(float(v), 4) for v in _grasp_point(ee["T"])]
            result["approach_tilt_deg"] = round(_approach_tilt_deg(ee["T"][:3, :3]), 1)
        return result

    # ------------------------------------------------------------------
    # automatic scene-camera calibration (markerless, gripper-motion)
    # ------------------------------------------------------------------

    def _set_gripper_hold(self, gripper_value: float) -> None:
        """Set the gripper opening while freezing the arm at its current joints.

        Used during calibration so that between the two capture frames ONLY the
        gripper fingers move (clean motion segmentation).
        """
        obs = self._robot.get_observation()
        q = np.array([obs.get(f"{n}.pos", 0.0) for n in _ARM_JOINTS], dtype=np.float64)
        self._robot.send_action(_to_lerobot_action(q, float(gripper_value)))

    @staticmethod
    def _calibration_targets() -> list[list[float]]:
        """A spread, non-coplanar grid of tip targets inside the workspace."""
        xs = [0.15, 0.22, 0.28]
        ys = [-0.12, 0.0, 0.12]
        zs = [0.12, 0.19]
        return [[x, y, z] for z in zs for y in ys for x in xs]

    def auto_calibrate_scene_camera(
        self,
        *,
        n_points: int = 10,
        gripper_open: float = 90.0,
        gripper_closed: float = 20.0,
        settle_s: float = 0.8,
        ransac_thresh_m: float = 0.015,
        save: bool = True,
    ) -> dict:
        """Markerless automatic scene-cam -> base calibration.

        Drives the tip to a grid of base-frame positions (move_to needs no
        extrinsic), and at each pose toggles the gripper with the arm frozen and
        segments the motion in the scene image to locate the tip (centroid +
        median depth -> camera point). The achieved FK gives the base point.
        A RANSAC Kabsch fit then yields ``T_base_cam``, which is saved and
        hot-loaded so back_project returns world coords immediately.

        Returns a summary dict (``n_used``, ``rmse_m``, per-pose diagnostics).
        """
        if self._scene_cam is None:
            return {"error": "scene camera not configured"}
        if self._kin is None:
            return {"error": "IK unavailable (URDF/placo missing)"}

        targets = self._calibration_targets()
        cam_pts: list[list[float]] = []
        base_pts: list[list[float]] = []
        poses: list[dict] = []

        for tgt in targets:
            if len(cam_pts) >= n_points:
                break
            mv = self.move_to(tgt, gripper=gripper_open, settle_s=settle_s)
            if "error" in mv or not mv.get("reached"):
                poses.append({"target": tgt, "skipped": "unreachable"})
                continue
            time.sleep(settle_s)

            base = np.asarray(mv["final_xyz"], dtype=np.float64)
            self._scene_cam.read()  # flush a frame
            rgb_open, _ = self._scene_cam.read()
            self._set_gripper_hold(gripper_closed)
            time.sleep(settle_s)
            rgb_closed, depth = self._scene_cam.read()
            self._set_gripper_hold(gripper_open)  # reopen for the next pose

            det = geom.detect_tip_pixel_by_motion(
                rgb_open, rgb_closed, depth, self._scene_cam.K,
            )
            if det is None:
                poses.append({"target": tgt, "skipped": "no_tip_detected"})
                continue
            cam_pts.append(det["xyz_cam"])
            base_pts.append(base.tolist())
            poses.append({"target": tgt, "base_xyz": base.round(4).tolist(),
                          "pixel": [round(v, 1) for v in det["pixel"]],
                          "depth_m": round(det["depth_m"], 4), "area": det["area"]})

        if len(cam_pts) < 4:
            return {"error": f"only {len(cam_pts)} usable points (need >= 4)",
                    "poses": poses}

        T, rmse, inliers = geom.ransac_kabsch(
            cam_pts, base_pts, thresh_m=ransac_thresh_m,
        )
        accepted = bool(rmse <= scene_calib.MAX_ACCEPTABLE_RMSE_M)
        result = {
            "n_targets": len(targets),
            "n_used": len(cam_pts),
            "n_inliers": int(np.asarray(inliers).sum()),
            "rmse_m": round(float(rmse), 4),
            "T_base_cam": T.tolist(),
            "accepted": accepted,
            "saved": False,
            "poses": poses,
        }
        if not accepted:
            # A high RMSE means the cam/base correspondences are inconsistent
            # (poor tip detection, lighting, or occlusion). Saving it would
            # silently corrupt every back_project, so refuse and ask for a rerun.
            result["error"] = (
                f"calibration RMSE {rmse * 1000:.1f} mm exceeds the "
                f"{scene_calib.MAX_ACCEPTABLE_RMSE_M * 1000:.0f} mm limit; not "
                "saved. Clear the workspace, improve gripper visibility/lighting, "
                "and rerun."
            )
            logger.warning(
                "scene-cam calibration REJECTED: rmse=%.4fm (> %.3fm); not saved",
                rmse, scene_calib.MAX_ACCEPTABLE_RMSE_M,
            )
        elif save:
            path = scene_calib.save_extrinsic(
                self._scene_serial, T, K=self._scene_cam.K,
                rmse_m=rmse, num_points=int(np.asarray(inliers).sum()),
            )
            self._T_base_cam = T  # hot-load so back_project works immediately
            self._calib_rmse_m = round(float(rmse), 4)
            result["saved"] = True
            result["path"] = str(path)
            logger.info("scene-cam calibrated: rmse=%.4fm, saved %s", rmse, path)

        # Park the arm at rest after the sweep (paced, gentle).
        try:
            obs = self._robot.get_observation()
            cur = np.array(
                [obs.get(f"{n}.pos", 0.0) for n in _ARM_JOINTS], dtype=np.float64
            )
            self._stream_joint_path(
                cur, _RESET_QPOS[:_NUM_ARM_JOINTS], float(_RESET_QPOS[_NUM_ARM_JOINTS])
            )
        except Exception:
            pass
        return result

    def close(self) -> None:
        """Park the arm at rest (paced + torque held) and disconnect cleanly."""
        try:
            obs = self._robot.get_observation()
            cur = np.array(
                [obs.get(f"{n}.pos", 0.0) for n in _ARM_JOINTS], dtype=np.float64
            )
            self._stream_joint_path(
                cur, _RESET_QPOS[:_NUM_ARM_JOINTS], float(_RESET_QPOS[_NUM_ARM_JOINTS])
            )
            time.sleep(0.3)
        except Exception as e:
            logger.warning("failed to park arm on close: %s", e)
        if self._scene_cam is not None:
            try:
                self._scene_cam.close()
            except Exception as e:
                logger.warning("error closing scene camera: %s", e)
        try:
            self._robot.disconnect()
            logger.info("SO101 disconnected")
        except Exception as e:
            logger.warning("error disconnecting robot: %s", e)

    # ------------------------------------------------------------------
    # internals
    # ------------------------------------------------------------------

    def _get_observation(self) -> dict:
        obs = self._robot.get_observation()
        joint_position = np.array(
            [obs.get(f"{n}.pos", 0.0) for n in _ARM_JOINTS], dtype=np.float32
        )
        gripper_position = np.array([obs.get(f"{_GRIPPER}.pos", 0.0)], dtype=np.float32)

        frames: dict = {}
        for cam in self._arm_camera_names:
            frame = obs.get(cam)
            if frame is not None:
                frames[cam] = np.ascontiguousarray(np.asarray(frame, dtype=np.uint8))

        depth: dict = {}
        if self._scene_cam is not None:
            scene_rgb, scene_depth = self._scene_cam.read()
            frames["scene"] = scene_rgb
            depth["scene"] = scene_depth

        state: dict = {
            "joint_position": joint_position,
            "gripper_position": gripper_position,
        }
        ee = self._compute_ee_pose(joint_position)
        if ee is not None:
            state["ee_pose_base"] = _grasp_point(ee["T"]).astype(np.float32)
            state["ee_quat_base"] = ee["quat"]

        out: dict = {"state": state, "frames": frames}
        if depth:
            out["depth"] = depth
        return out

    def _compute_ee_pose(self, joints_deg) -> dict | None:
        """FK -> gripper pose in the base (world) frame, or None if FK is off."""
        if self._kin is None:
            return None
        try:
            T = self._kin.fk(joints_deg)
        except Exception as e:
            logger.warning("FK failed: %s", e)
            return None
        xyz = T[:3, 3].astype(np.float32)
        quat = geom.rotation_to_quat(T[:3, :3]).astype(np.float32)
        return {"xyz": xyz, "quat": quat, "T": T}


# ---------------------------------------------------------------------------
# RPC plumbing
# ---------------------------------------------------------------------------

_INITIAL_PPID = os.getppid()


def _start_parent_watchdog(
    server: SocketRpcServer,
    shutdown_event: threading.Event,
    poll_s: float = 2.0,
) -> None:
    """Shut the RPC server down if the agent (parent) process dies."""

    def _watch() -> None:
        while not shutdown_event.is_set():
            time.sleep(poll_s)
            ppid = os.getppid()
            if ppid != _INITIAL_PPID or ppid == 1:
                logger.warning(
                    "parent died (ppid %s -> %s); stopping RPC server",
                    _INITIAL_PPID,
                    ppid,
                )
                shutdown_event.set()
                threading.Thread(target=server.shutdown, daemon=True).start()
                return

    threading.Thread(target=_watch, daemon=True).start()


def _build_dispatcher(env: SO101LeRobotEnv, shutdown_event: threading.Event):
    """Route ``env.*`` / ``shutdown`` to the right callable."""

    def dispatch(method: str, args: tuple, kwargs: dict):
        if method.startswith("env."):
            return getattr(env, method[len("env."):])(*args, **kwargs)
        if method == "shutdown":
            shutdown_event.set()
            return {"ok": True}
        raise ValueError(f"unknown RPC method: {method!r}")

    return dispatch


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Standalone LeRobot SO101 env server")
    p.add_argument("--port", default="/dev/ttyACM1",
                   help="Serial port of the SO101 follower arm.")
    p.add_argument("--calibration-id", default="my_awesome_follower_arm",
                   help="LeRobot calibration id (loads <id>.json).")
    p.add_argument("--arm-camera-path", default="/dev/video2",
                   help="OpenCV device path for the arm/hand camera "
                        "(empty string to disable).")
    p.add_argument("--scene-camera-serial", default="409122274720",
                   help="Intel RealSense serial/name for the scene camera "
                        "(empty string to disable).")
    p.add_argument("--camera-width", type=int, default=640)
    p.add_argument("--camera-height", type=int, default=480)
    p.add_argument("--camera-fps", type=int, default=30)
    p.add_argument("--scene-camera-width", type=int, default=1280,
                   help="Scene (RealSense) color/depth width; default 1280 "
                        "(720p) for finer localization. <=0 falls back to "
                        "--camera-width.")
    p.add_argument("--scene-camera-height", type=int, default=720,
                   help="Scene (RealSense) color/depth height; default 720. "
                        "<=0 falls back to --camera-height.")
    p.add_argument("--no-cameras", action="store_true",
                   help="Disable all cameras (state-only observations).")
    p.add_argument("--max-relative-target", type=float, default=None,
                   help="Per-step joint movement cap in degrees (safety). "
                        "Default: no cap (matches RLinf SO101Env).")
    p.add_argument("--max-episode-steps", type=int, default=200)
    p.add_argument("--step-frequency", type=float, default=30.0)
    p.add_argument("--max-joint-vel", type=float, default=_MAX_JOINT_VEL_DEG_S,
                   help="Max joint speed (deg/s) for paced point-to-point moves "
                        "(reset/move_to/move_joints_delta). Lower = slower/safer.")
    p.add_argument("--motor-acceleration", type=int, default=_MOTOR_ACCELERATION,
                   help="Feetech servo Acceleration register (0-254; LeRobot "
                        "default 254). Lower = gentler ramps. Set 254 for the "
                        "original snappy motion.")
    p.add_argument("--position-gain", type=int, default=_POSITION_GAIN,
                   help="Feetech servo P_Coefficient / position gain (LeRobot "
                        "uses 16, factory default 32). Higher = stiffer, holds "
                        "commanded joints under load (fixes gripper landing "
                        "short/low); too high can buzz/oscillate. Applied to arm "
                        "joints only.")
    p.add_argument("--auto-calibrate", action="store_true",
                   help="Allow LeRobot to run interactive calibration if the "
                        "arm is uncalibrated (may block on stdin). Off by default.")
    p.add_argument("--urdf-path", default=None,
                   help="SO101 URDF for FK / EE pose. Default: "
                        "~/.cache/huggingface/lerobot/urdf/so101.urdf")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--transport-port", type=int, default=0,
                   help="Socket transport port. 0 asks the OS for a free port.")
    return p


def _build_arm_camera_cfgs(args: argparse.Namespace) -> dict[str, dict]:
    """Assemble the LeRobot (arm/hand) camera spec from CLI args.

    The scene camera is NOT included here — it is managed directly via
    pyrealsense2 (see :class:`SceneCameraD405`) so we get depth + intrinsics.
    """
    if args.no_cameras or not args.arm_camera_path:
        return {}
    return {
        "arm": {
            "type": "opencv",
            "index_or_path": args.arm_camera_path,
            "width": args.camera_width,
            "height": args.camera_height,
            "fps": args.camera_fps,
        }
    }


def main() -> int:
    args = _build_argparser().parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    init_output_dir(args.output_dir)

    arm_camera_cfgs = _build_arm_camera_cfgs(args)
    scene_serial = None if args.no_cameras else (args.scene_camera_serial or None)
    logger.info(
        "starting SO101 env server: port=%s output_dir=%s arm_cams=%s scene=%s",
        args.port, args.output_dir, list(arm_camera_cfgs), scene_serial,
    )

    env = SO101LeRobotEnv(
        port=args.port,
        calibration_id=args.calibration_id,
        arm_camera_cfgs=arm_camera_cfgs,
        scene_serial=scene_serial,
        scene_size=(
            args.scene_camera_height if args.scene_camera_height > 0 else args.camera_height,
            args.scene_camera_width if args.scene_camera_width > 0 else args.camera_width,
        ),
        scene_fps=args.camera_fps,
        urdf_path=args.urdf_path,
        max_relative_target=args.max_relative_target,
        max_episode_steps=args.max_episode_steps,
        step_frequency=args.step_frequency,
        motor_acceleration=args.motor_acceleration,
        max_joint_vel_deg_s=args.max_joint_vel,
        position_gain=args.position_gain,
        auto_calibrate=args.auto_calibrate,
    )

    shutdown_event = threading.Event()
    dispatch = _build_dispatcher(env, shutdown_event)

    server = SocketRpcServer(("127.0.0.1", args.transport_port), dispatch)
    bound_host, bound_port = server.server_address
    client_host = "127.0.0.1" if bound_host == "0.0.0.0" else bound_host
    print(
        json.dumps({
            "event": "transport_ready",
            "kind": "socket",
            "host": client_host,
            "port": bound_port,
        }),
        flush=True,
    )
    logger.info("RPC server listening on %s:%s", client_host, bound_port)

    # Park the arm on SIGTERM / SIGINT (launcher kill, Ctrl-C). The handler
    # only flags shutdown; the actual parking runs in the ``finally`` below,
    # never inside the signal context. (SIGKILL / kill -9 cannot be caught.)
    def _handle_signal(signum, _frame):
        logger.warning(
            "received %s; parking arm and shutting down",
            signal.Signals(signum).name,
        )
        shutdown_event.set()

    for _sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(_sig, _handle_signal)

    _start_parent_watchdog(server, shutdown_event)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        # Loop with a timeout so a signal landing during the wait is observed
        # promptly even if it doesn't interrupt the blocking call.
        while not shutdown_event.wait(timeout=1.0):
            pass
    finally:
        env.close()
        server.shutdown()
        server.server_close()
    logger.info("driver exited cleanly")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
