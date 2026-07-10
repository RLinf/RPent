"""Standalone Franka (FR3) env host for the LLM-in-the-loop agent (env-only).

Drives a Franka arm through the SERL cartesian-impedance ROS controller and the
``franka_gripper`` action topics, exposing agent-friendly *Cartesian* primitives
(``reset`` / ``get_obs`` / ``get_ee_pose`` / ``move_to`` / ``move_delta`` /
``open_gripper`` / ``close_gripper`` / ``get_spec``) over a pickle-framed TCP RPC
server (:class:`rpent.rpc_driver.socket.SocketRpcServer`) -- the same
wire protocol the LIBERO and LeRobot drivers use, so the agent side talks to all
three identically.

Unlike the LIBERO driver, this server does **not** import any
``rlinf.envs.realworld`` module. Importing that package runs node-level ROS setup
side effects at import time (it kills any running ``roscore`` / ``rosmaster``) and
pulls in the Ray-based ``Worker`` stack. This driver instead talks to ROS directly
with ``rospy``, reusing only the *recipe* from
``rlinf.envs.realworld.franka.{franka_controller,franka_env}``: the impedance
controller channel names, the ``roslaunch`` bring-up, the ``franka_gripper``
action messages, and the safety-box + pose-interpolation logic.

Run it inside the RLinf ``.venv`` with the ``serl_franka_controllers`` catkin
workspace sourced (see ``deployment/franka/run_env_server.sh``)::

    source /home/franka/franka/RLinf/.venv/franka_catkin_ws/devel/setup.bash
    /home/franka/franka/RLinf/.venv/bin/python deployment/franka/env_server.py \
        --output-dir /tmp/franka_run --robot-ip 172.16.0.2

Hardware defaults match the current bench: an FR3 at ``172.16.0.2`` with the
Franka Hand, plus two Intel RealSense D435I cameras. Every default is overridable
from the CLI.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp

# Make ``rpent`` importable when this file is run from the RLinf .venv
# (which need not have rpent installed) -- the source tree is enough.
_PHYSICALAGENT_ROOT = Path(__file__).resolve().parents[2]
if str(_PHYSICALAGENT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PHYSICALAGENT_ROOT))

from rpent.rpc_driver.socket import SocketRpcServer  # noqa: E402
from rpent.utils.logging import get_logger, init_output_dir  # noqa: E402

from deployment.franka import calibration as camera_calib  # noqa: E402

logger = get_logger("franka_driver")


# ---------------------------------------------------------------------------
# Bench defaults (override from the CLI)
# ---------------------------------------------------------------------------

_DEFAULT_ROBOT_IP = os.environ.get("FRANKA_ROBOT_IP", "172.16.0.2")
_DEFAULT_ROS_PKG = "serl_franka_controllers"

# Two Intel RealSense D435I on the current bench. Order matters: the first camera
# is the primary/overview view (maps to the ToolResult ``_image_bytes`` slot), the
# second maps to ``_image_cam_bytes``. Names are what the agent sees.
_DEFAULT_CAMERAS: tuple[tuple[str, str], ...] = (
    ("scene", "142122070838"),
    ("wrist", "141722078696"),
)
_CAM_WIDTH = 640
_CAM_HEIGHT = 480
_CAM_FPS = 30

# Conservative tabletop workspace box in the base (panda_link0) frame, meters.
# ``move_to`` clips targets to this so an LLM-supplied coordinate cannot drive the
# arm into the table / out of reach. Tuned around the collect-data reset pose
# ([0.5, 0, 0.1] with the gripper pointing down); widen/lower on your bench once
# you have confirmed the table height in the base frame.
_WORKSPACE_MIN = np.array([0.30, -0.50, 0.00], dtype=np.float64)
_WORKSPACE_MAX = np.array([1.10, 0.50, 0.50], dtype=np.float64)

# Default "home" pose the agent's reset() drives to: above the table, gripper
# pointing straight down. Orientation is euler xyz (radians); rx = -pi points the
# Franka Hand down (matches the RLinf realworld collect configs).
_RESET_XYZ = np.array([0.50, 0.0, 0.25], dtype=np.float64)
_RESET_EULER = np.array([np.pi, 0.0, 0.0], dtype=np.float64)

# The nominal orientation the arm holds; move_to keeps the current orientation
# unless the caller overrides it, and clips any requested orientation to a window
# around this so the wrist cannot flip into a self-collision.
_TARGET_EULER = _RESET_EULER.copy()
_EULER_WINDOW = np.array([0.6, 0.6, np.pi], dtype=np.float64)  # +/- rad per axis

# Motion smoothness / safety. The impedance controller tracks a streamed sequence
# of equilibrium poses; capping the per-step Cartesian and angular deltas and
# pacing at ``step_frequency`` keeps motions slow and gentle.
_STEP_FREQUENCY = 10.0  # Hz (equilibrium-pose publish rate)
_MAX_STEP_M = 0.01      # max Cartesian move per streamed setpoint (=> ~0.1 m/s)
_MAX_STEP_DEG = 5.0     # max orientation change per streamed setpoint
_MAX_MOVE_M = 0.60      # hard cap on a single move_to path length (safety)
_REACHED_TOL_M = 0.003  # move_to "reached" tolerance
_SETTLE_TIMEOUT_S = 2.0 # max time to hold the final setpoint while settling

# franka_gripper widths (meters).
_GRIPPER_OPEN_WIDTH = 0.09
_GRIPPER_GRASP_WIDTH = 0.01
_GRIPPER_GRASP_FORCE = 130.0
_GRIPPER_SPEED = 0.3


# ---------------------------------------------------------------------------
# small pose helpers (scipy uses scalar-last quaternions: [x, y, z, w])
# ---------------------------------------------------------------------------


def _euler_to_quat(euler_xyz) -> np.ndarray:
    return R.from_euler("xyz", np.asarray(euler_xyz, dtype=np.float64)).as_quat()


def _quat_to_euler(quat_xyzw) -> np.ndarray:
    return R.from_quat(np.asarray(quat_xyzw, dtype=np.float64)).as_euler("xyz")


def _pose_to_matrix(pose7: np.ndarray) -> np.ndarray:
    """Convert ``[x, y, z, qx, qy, qz, qw]`` to ``T_base_tcp``."""
    pose7 = np.asarray(pose7, dtype=np.float64).reshape(-1)[:7]
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R.from_quat(pose7[3:]).as_matrix()
    T[:3, 3] = pose7[:3]
    return T


def _clip_euler_window(euler_xyz: np.ndarray) -> np.ndarray:
    """Clip an euler orientation to +/-``_EULER_WINDOW`` around ``_TARGET_EULER``.

    Wraps each axis into ``[-pi, pi]`` relative to the target first so the clip is
    on the shortest angular distance, not the raw value.
    """
    euler = np.asarray(euler_xyz, dtype=np.float64).copy()
    delta = (euler - _TARGET_EULER + np.pi) % (2 * np.pi) - np.pi
    delta = np.clip(delta, -_EULER_WINDOW, _EULER_WINDOW)
    return _TARGET_EULER + delta


# ---------------------------------------------------------------------------
# RealSense color + depth camera
# ---------------------------------------------------------------------------


class RealSenseDepthCamera:
    """Minimal RealSense RGB-D grabber with depth aligned to color."""

    def __init__(self, name: str, serial: str, *, width: int, height: int, fps: int):
        import pyrealsense2 as rs

        self.name = name
        self.serial = str(serial)
        self._rs = rs
        self._pipeline = rs.pipeline()
        cfg = rs.config()
        cfg.enable_device(self.serial)
        cfg.enable_stream(rs.stream.color, width, height, rs.format.rgb8, fps)
        cfg.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
        self._profile = self._pipeline.start(cfg)
        self._align = rs.align(rs.stream.color)

        depth_sensor = self._profile.get_device().first_depth_sensor()
        self._depth_scale = float(depth_sensor.get_depth_scale())

        color_stream = self._profile.get_stream(
            rs.stream.color
        ).as_video_stream_profile()
        intr = color_stream.get_intrinsics()
        self._K = np.array(
            [[intr.fx, 0.0, intr.ppx], [0.0, intr.fy, intr.ppy], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )
        self._dist_coeffs = np.asarray(intr.coeffs, dtype=np.float64)
        self._distortion_model = str(intr.model)
        self._width = int(width)
        self._height = int(height)
        self._calib = camera_calib.load_record(self.serial)
        self._calib_accepted = camera_calib.is_accepted(self._calib)

        # Drop the first few frames so auto-exposure settles.
        for _ in range(5):
            try:
                self._pipeline.wait_for_frames(2000)
            except Exception:
                break
        calib_kind = "uncalibrated"
        if self._calib_accepted:
            if self._calib and "T_base_cam" in self._calib:
                calib_kind = "T_base_cam"
            elif self._calib and "T_tcp_cam" in self._calib:
                calib_kind = "T_tcp_cam"
        elif self._calib is not None:
            calib_kind = "rejected"
        logger.info(
            "camera '%s' (serial %s) RGB-D started; calibration=%s",
            name,
            self.serial,
            calib_kind,
        )

    def read(self) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(rgb_uint8, depth_m_float32)`` with depth aligned to RGB."""
        frames = self._pipeline.wait_for_frames(2000)
        frames = self._align.process(frames)
        color = frames.get_color_frame()
        depth = frames.get_depth_frame()
        if not color or not depth:
            raise RuntimeError(f"camera '{self.name}' incomplete frameset")
        rgb = np.ascontiguousarray(np.asanyarray(color.get_data()), dtype=np.uint8)
        depth_raw = np.asanyarray(depth.get_data())
        depth_m = np.ascontiguousarray(
            depth_raw.astype(np.float32) * self._depth_scale
        )
        return rgb, depth_m

    def meta(self, *, T_base_tcp: np.ndarray | None = None) -> dict:
        """Return JSON-able camera metadata and any base-frame extrinsic."""
        T_base_cam = None
        calibration_kind = None
        if self._calib_accepted and self._calib is not None:
            if "T_base_cam" in self._calib:
                T_base_cam = self._calib["T_base_cam"]
                calibration_kind = "T_base_cam"
            elif "T_tcp_cam" in self._calib:
                calibration_kind = "T_tcp_cam"
                if T_base_tcp is not None:
                    T_base_cam = np.asarray(T_base_tcp, dtype=np.float64) @ self._calib[
                        "T_tcp_cam"
                    ]

        return {
            "name": self.name,
            "serial": self.serial,
            "frame": f"{self.name}_camera",
            "width": self._width,
            "height": self._height,
            "K": self._K.tolist(),
            "dist_coeffs": self._dist_coeffs.tolist(),
            "distortion_model": self._distortion_model,
            "depth_scale": self._depth_scale,
            "calibrated": T_base_cam is not None,
            "calibration_kind": calibration_kind,
            "calibration_rmse_m": (
                None if self._calib is None else self._calib.get("rmse_m")
            ),
            "calibration_path": (
                None if self._calib is None else self._calib.get("path")
            ),
            "T_base_cam": None if T_base_cam is None else T_base_cam.tolist(),
        }

    def reload_calibration(self) -> dict:
        """Reload this camera's calibration record from disk."""
        self._calib = camera_calib.load_record(self.serial)
        self._calib_accepted = camera_calib.is_accepted(self._calib)
        meta = self.meta()
        logger.info(
            "camera '%s' calibration reloaded: calibrated=%s kind=%s",
            self.name,
            meta["calibrated"],
            meta["calibration_kind"],
        )
        return meta

    def close(self) -> None:
        try:
            self._pipeline.stop()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Franka ROS backend (plain rospy; no rlinf, no Ray)
# ---------------------------------------------------------------------------


class FrankaRobotBackend:
    """Low-level Franka arm + gripper over ROS.

    Reuses the channel names, impedance ``roslaunch`` bring-up, and message
    parsing from ``rlinf.envs.realworld.franka.franka_controller`` /
    ``.common.gripper.franka_gripper`` -- reimplemented on plain ``rospy`` so this
    driver stays free of the Ray ``Worker`` stack and the destructive
    ``rlinf.envs.realworld`` import-time side effects.
    """

    _ARM_EQUILIBRIUM = "/cartesian_impedance_controller/equilibrium_pose"
    _ARM_STATE = "/franka_state_controller/franka_states"
    _ARM_RESET = "/franka_control/error_recovery/goal"
    _GRIPPER_MOVE = "/franka_gripper/move/goal"
    _GRIPPER_GRASP = "/franka_gripper/grasp/goal"
    _GRIPPER_STATE = "/franka_gripper/joint_states"

    def __init__(
        self,
        *,
        robot_ip: str,
        ros_pkg: str = _DEFAULT_ROS_PKG,
        load_gripper: bool = True,
        node_name: str = "franka_agent_driver",
        state_timeout_s: float = 4.0,
        launch_timeout_s: float = 40.0,
    ):
        # Lazy ROS imports so the module can be imported (e.g. for --help) on a
        # host without ROS on the path.
        import geometry_msgs.msg as geom_msg
        import rospy
        from franka_gripper.msg import GraspActionGoal, MoveActionGoal
        from franka_msgs.msg import ErrorRecoveryActionGoal, FrankaState
        from sensor_msgs.msg import JointState

        self._rospy = rospy
        self._geom_msg = geom_msg
        self._FrankaState = FrankaState
        self._ErrorRecoveryActionGoal = ErrorRecoveryActionGoal
        self._MoveActionGoal = MoveActionGoal
        self._GraspActionGoal = GraspActionGoal
        self._JointState = JointState

        self._robot_ip = robot_ip
        self._ros_pkg = ros_pkg
        self._load_gripper = load_gripper
        self._impedance_proc = None  # only set if we launch it ourselves

        # Live state (updated by subscriber callbacks).
        self._tcp_pose = np.zeros(7, dtype=np.float64)
        self._tcp_pose[6] = 1.0  # unit quaternion
        self._tcp_force = np.zeros(3, dtype=np.float64)
        self._tcp_torque = np.zeros(3, dtype=np.float64)
        self._state_seen = threading.Event()
        self._gripper_width = _GRIPPER_OPEN_WIDTH
        self._gripper_open = True
        self._gripper_seen = threading.Event()

        self._ensure_master()
        rospy.init_node(node_name, anonymous=True, disable_signals=True)

        # Publishers.
        self._pub_equilibrium = rospy.Publisher(
            self._ARM_EQUILIBRIUM, geom_msg.PoseStamped, queue_size=10
        )
        self._pub_reset = rospy.Publisher(
            self._ARM_RESET, ErrorRecoveryActionGoal, queue_size=1
        )
        self._pub_gripper_move = rospy.Publisher(
            self._GRIPPER_MOVE, MoveActionGoal, queue_size=1
        )
        self._pub_gripper_grasp = rospy.Publisher(
            self._GRIPPER_GRASP, GraspActionGoal, queue_size=1
        )

        # Subscribers.
        rospy.Subscriber(self._ARM_STATE, FrankaState, self._on_state_msg)
        rospy.Subscriber(self._GRIPPER_STATE, JointState, self._on_gripper_msg)

        self._ensure_impedance(state_timeout_s, launch_timeout_s)

    # -- lifecycle helpers -------------------------------------------------

    def _ensure_master(self) -> None:
        """Make sure a ROS master is reachable; launch ``roscore`` if not.

        Mirrors ``rlinf...common.ros.ros_controller.ROSController``: reuse a
        running master, otherwise start one. ``rosmaster --core`` (what
        ``roscore`` spawns) also counts as a running master.
        """
        import psutil

        try:
            import rosgraph

            if rosgraph.is_master_online():
                logger.info("ROS master already online at %s", os.environ.get(
                    "ROS_MASTER_URI", "http://localhost:11311"))
                return
        except Exception:
            pass

        for proc in psutil.process_iter(["name"]):
            if proc.info.get("name") in ("roscore", "rosmaster"):
                logger.info("found running %s (pid %s)", proc.info["name"], proc.pid)
                return

        logger.info("no ROS master found; starting `roscore`")
        self._roscore_proc = psutil.Popen(
            ["roscore"], stdout=sys.stdout, stderr=sys.stdout
        )
        time.sleep(2.0)

    def _impedance_up(self, timeout_s: float) -> bool:
        """Return True once a franka_states message arrives within ``timeout_s``."""
        return self._state_seen.wait(timeout=timeout_s)

    def _ensure_impedance(self, state_timeout_s: float, launch_timeout_s: float) -> None:
        """Attach to a running cartesian-impedance controller, or launch one.

        We never launch a second controller: if ``franka_states`` is already
        publishing we simply attach. Only when no state arrives do we
        ``roslaunch serl_franka_controllers impedance.launch``.
        """
        if self._impedance_up(state_timeout_s):
            logger.info("attached to live cartesian-impedance controller")
            return

        import psutil

        load_gripper = "true" if self._load_gripper else "false"
        cmd = [
            "roslaunch",
            self._ros_pkg,
            "impedance.launch",
            f"robot_ip:={self._robot_ip}",
            f"load_gripper:={load_gripper}",
        ]
        logger.info("no live controller detected; starting `%s`", " ".join(cmd))
        self._impedance_proc = psutil.Popen(cmd, stdout=sys.stdout, stderr=sys.stdout)

        deadline = time.time() + launch_timeout_s
        while time.time() < deadline:
            if self._impedance_proc.poll() is not None:
                raise RuntimeError(
                    "impedance roslaunch exited before becoming ready; run "
                    f"`{' '.join(cmd)}` manually to see the error (is "
                    f"{self._ros_pkg} on the ROS package path?)."
                )
            if self._impedance_up(1.0):
                logger.info("cartesian-impedance controller is up")
                return
        raise RuntimeError(
            f"cartesian-impedance controller not ready after {launch_timeout_s}s"
        )

    # -- ROS callbacks -----------------------------------------------------

    def _on_state_msg(self, msg) -> None:
        tmatrix = np.array(list(msg.O_T_EE)).reshape(4, 4).T
        quat = R.from_matrix(tmatrix[:3, :3].copy()).as_quat()
        self._tcp_pose = np.concatenate([tmatrix[:3, 3], quat])
        self._tcp_force = np.array(list(msg.K_F_ext_hat_K)[:3])
        self._tcp_torque = np.array(list(msg.K_F_ext_hat_K)[3:])
        self._state_seen.set()

    def _on_gripper_msg(self, msg) -> None:
        # joint_states reports both finger joints; their sum is the opening width.
        self._gripper_width = float(np.sum(msg.position))
        self._gripper_open = self._gripper_width > 0.06
        self._gripper_seen.set()

    # -- arm ---------------------------------------------------------------

    def get_tcp_pose(self) -> np.ndarray:
        """Current TCP pose ``[x, y, z, qx, qy, qz, qw]`` in the base frame."""
        return self._tcp_pose.copy()

    def get_state(self) -> dict:
        return {
            "tcp_pose": self._tcp_pose.copy(),
            "tcp_force": self._tcp_force.copy(),
            "tcp_torque": self._tcp_torque.copy(),
            "gripper_width": self._gripper_width,
            "gripper_open": self._gripper_open,
        }

    def move_arm(self, pose7: np.ndarray) -> None:
        """Publish one equilibrium pose ``[x, y, z, qx, qy, qz, qw]``."""
        pose7 = np.asarray(pose7, dtype=np.float64).reshape(-1)
        assert pose7.shape[0] == 7, f"expected 7-D pose, got {pose7.shape}"
        msg = self._geom_msg.PoseStamped()
        msg.header.frame_id = "0"
        msg.header.stamp = self._rospy.Time.now()
        msg.pose.position = self._geom_msg.Point(pose7[0], pose7[1], pose7[2])
        msg.pose.orientation = self._geom_msg.Quaternion(
            pose7[3], pose7[4], pose7[5], pose7[6]
        )
        self._pub_equilibrium.publish(msg)

    def clear_errors(self) -> None:
        self._pub_reset.publish(self._ErrorRecoveryActionGoal())

    # -- gripper -----------------------------------------------------------

    def open_gripper(self, speed: float = _GRIPPER_SPEED) -> None:
        msg = self._MoveActionGoal()
        msg.goal.width = _GRIPPER_OPEN_WIDTH
        msg.goal.speed = speed
        self._pub_gripper_move.publish(msg)
        self._gripper_open = True

    def close_gripper(
        self, speed: float = _GRIPPER_SPEED, force: float = _GRIPPER_GRASP_FORCE
    ) -> None:
        msg = self._GraspActionGoal()
        msg.goal.width = _GRIPPER_GRASP_WIDTH
        msg.goal.speed = speed
        msg.goal.epsilon.inner = 1.0
        msg.goal.epsilon.outer = 1.0
        msg.goal.force = force
        self._pub_gripper_grasp.publish(msg)
        self._gripper_open = False

    def shutdown(self) -> None:
        """Terminate only the controller we launched (never a pre-existing one)."""
        if self._impedance_proc is not None and self._impedance_proc.poll() is None:
            logger.info("terminating impedance controller we launched")
            self._impedance_proc.terminate()
            try:
                self._impedance_proc.wait(timeout=10)
            except Exception:
                self._impedance_proc.kill()


# ---------------------------------------------------------------------------
# Agent-facing env facade
# ---------------------------------------------------------------------------


class FrankaAgentEnv:
    """Cartesian-primitive facade the agent RPCs into.

    Observation::

        {"state": {"tcp_xyz":   (3,) float,   # meters, base frame
                   "tcp_quat":  (4,) float,   # [x, y, z, w]
                   "tcp_euler": (3,) float,   # radians xyz
                   "gripper_width": float,    # meters
                   "gripper_open":  bool},
         "frames": {<cam>: (H, W, 3) uint8, ...}}

    All values are plain numpy / python scalars so they pickle across the RPC
    wire (the agent process does not import torch or ROS).
    """

    def __init__(
        self,
        backend: FrankaRobotBackend,
        cameras: list[RealSenseDepthCamera],
        *,
        workspace_min: np.ndarray = _WORKSPACE_MIN,
        workspace_max: np.ndarray = _WORKSPACE_MAX,
        reset_xyz: np.ndarray = _RESET_XYZ,
        reset_euler: np.ndarray = _RESET_EULER,
        step_frequency: float = _STEP_FREQUENCY,
        max_step_m: float = _MAX_STEP_M,
        max_step_deg: float = _MAX_STEP_DEG,
        max_move_m: float = _MAX_MOVE_M,
        settle_timeout_s: float = _SETTLE_TIMEOUT_S,
    ):
        self._backend = backend
        self._cameras = cameras
        self._workspace_min = np.asarray(workspace_min, dtype=np.float64)
        self._workspace_max = np.asarray(workspace_max, dtype=np.float64)
        self._reset_xyz = np.asarray(reset_xyz, dtype=np.float64)
        self._reset_quat = _euler_to_quat(reset_euler)
        self._step_frequency = float(step_frequency)
        self._max_step_m = float(max_step_m)
        self._max_step_deg = float(max_step_deg)
        self._max_move_m = float(max_move_m)
        self._settle_timeout_s = float(settle_timeout_s)

    # -- observation -------------------------------------------------------

    def _frames(self) -> dict[str, np.ndarray]:
        out: dict[str, np.ndarray] = {}
        for cam in self._cameras:
            try:
                rgb, _ = cam.read()
                out[cam.name] = rgb
            except Exception as e:
                logger.warning("camera '%s' frame grab failed: %s", cam.name, e)
        return out

    def _camera_observation(
        self, T_base_tcp: np.ndarray
    ) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, dict]]:
        frames: dict[str, np.ndarray] = {}
        depths: dict[str, np.ndarray] = {}
        meta: dict[str, dict] = {}
        for cam in self._cameras:
            try:
                rgb, depth = cam.read()
                frames[cam.name] = rgb
                depths[cam.name] = depth
                meta[cam.name] = cam.meta(T_base_tcp=T_base_tcp)
            except Exception as e:
                logger.warning("camera '%s' RGB-D grab failed: %s", cam.name, e)
        return frames, depths, meta

    def _make_obs(self) -> dict:
        pose = self._backend.get_tcp_pose()
        st = self._backend.get_state()
        frames, depths, camera_meta = self._camera_observation(_pose_to_matrix(pose))
        return {
            "state": {
                "tcp_xyz": pose[:3].astype(np.float64),
                "tcp_quat": pose[3:].astype(np.float64),
                "tcp_euler": _quat_to_euler(pose[3:]).astype(np.float64),
                "gripper_width": float(st["gripper_width"]),
                "gripper_open": bool(st["gripper_open"]),
            },
            "frames": frames,
            "depth": depths,
            "camera_meta": camera_meta,
        }

    def get_obs(self) -> dict:
        """Return the current observation without moving the arm."""
        return self._make_obs()

    def get_ee_pose(self) -> dict:
        """Return the current end-effector pose in the base frame."""
        pose = self._backend.get_tcp_pose()
        return {
            "xyz": [round(float(v), 4) for v in pose[:3]],
            "quat_xyzw": [round(float(v), 4) for v in pose[3:]],
            "euler_xyz": [round(float(v), 4) for v in _quat_to_euler(pose[3:])],
            "frame": "panda_link0",
        }

    def get_spec(self) -> dict:
        """Static self-description for the agent side."""
        return {
            "world_frame": "panda_link0",
            "control": "cartesian_impedance",
            "position_unit": "meters",
            "orientation": "euler_xyz_radians (also quat xyzw)",
            "workspace_min": [round(float(v), 3) for v in self._workspace_min],
            "workspace_max": [round(float(v), 3) for v in self._workspace_max],
            "camera_names": [c.name for c in self._cameras],
            "depth_camera_names": [c.name for c in self._cameras],
            "preferred_backproject_camera": "wrist",
            "gripper": "binary (open_gripper / close_gripper); width in meters",
            "reset_xyz": [round(float(v), 3) for v in self._reset_xyz],
        }

    def get_camera_meta(self) -> dict:
        """Return live per-camera intrinsics and base-frame extrinsics if calibrated."""
        T_base_tcp = _pose_to_matrix(self._backend.get_tcp_pose())
        return {cam.name: cam.meta(T_base_tcp=T_base_tcp) for cam in self._cameras}

    def reload_camera_calibration(self, camera: str | None = None) -> dict:
        """Reload camera calibration records from disk."""
        requested = None if camera is None else str(camera)
        reloaded: dict[str, dict] = {}
        for cam in self._cameras:
            if requested is not None and cam.name != requested:
                continue
            reloaded[cam.name] = cam.reload_calibration()
        if requested is not None and requested not in reloaded:
            return {
                "error": f"unknown camera {requested!r}",
                "available_cameras": [cam.name for cam in self._cameras],
            }
        return reloaded

    # -- motion ------------------------------------------------------------

    def _clip_xyz(self, xyz: np.ndarray) -> tuple[np.ndarray, bool]:
        clipped = np.clip(xyz, self._workspace_min, self._workspace_max)
        return clipped, bool(np.any(clipped != xyz))

    def _stream_to(self, target_xyz: np.ndarray, target_quat: np.ndarray) -> None:
        """Stream interpolated equilibrium poses from the current pose to the
        target, capping per-step Cartesian and angular deltas and pacing at
        ``step_frequency`` so the impedance controller tracks a slow, smooth path.
        """
        cur = self._backend.get_tcp_pose()
        cur_xyz, cur_quat = cur[:3], cur[3:]

        dist = float(np.linalg.norm(target_xyz - cur_xyz))
        ang = float(
            (R.from_quat(cur_quat).inv() * R.from_quat(target_quat)).magnitude()
        )
        n_pos = int(np.ceil(dist / self._max_step_m)) if dist > 0 else 0
        n_ang = int(np.ceil(np.degrees(ang) / self._max_step_deg)) if ang > 0 else 0
        n = max(1, n_pos, n_ang)

        slerp = Slerp([0.0, 1.0], R.concatenate([R.from_quat(cur_quat), R.from_quat(target_quat)]))
        period = 1.0 / self._step_frequency
        for i in range(1, n + 1):
            t = i / n
            pos = cur_xyz + (target_xyz - cur_xyz) * t
            quat = slerp([t])[0].as_quat()
            self._backend.move_arm(np.concatenate([pos, quat]))
            time.sleep(period)

    def _wait_until_reached(
        self, target_xyz: np.ndarray, target_quat: np.ndarray
    ) -> tuple[np.ndarray, float, bool]:
        """Hold the final setpoint until the measured TCP pose is close enough."""
        target_pose = np.concatenate([target_xyz, target_quat])
        deadline = time.time() + self._settle_timeout_s
        period = 1.0 / self._step_frequency
        while True:
            final = self._backend.get_tcp_pose()
            pos_err = float(np.linalg.norm(final[:3] - target_xyz))
            if pos_err <= _REACHED_TOL_M or time.time() >= deadline:
                return final, pos_err, pos_err <= _REACHED_TOL_M
            self._backend.move_arm(target_pose)
            time.sleep(period)

    def _apply_gripper(self, gripper: Optional[str]) -> None:
        if gripper is None:
            return
        g = str(gripper).lower()
        if g in ("open", "release"):
            self._backend.open_gripper()
            time.sleep(0.6)
        elif g in ("close", "grasp"):
            self._backend.close_gripper()
            time.sleep(0.6)
        else:
            raise ValueError(f"gripper must be 'open' or 'close', got {gripper!r}")

    def move_to(
        self,
        xyz,
        *,
        euler_xyz=None,
        quat_xyzw=None,
        gripper: Optional[str] = None,
    ) -> dict:
        """Move the TCP to a base-frame ``xyz`` (meters), holding the current
        orientation unless ``euler_xyz`` / ``quat_xyzw`` is given.

        The target is clipped to the workspace box and the orientation to a safe
        window; the path is streamed as slow capped setpoints. Optionally set the
        gripper ("open"/"close") first. Returns a log dict.
        """
        self._backend.clear_errors()
        cur = self._backend.get_tcp_pose()

        target_xyz = np.asarray(xyz, dtype=np.float64).reshape(-1)[:3]
        target_xyz, clipped = self._clip_xyz(target_xyz)

        path_len = float(np.linalg.norm(target_xyz - cur[:3]))
        if path_len > self._max_move_m:
            return {
                "reached": False,
                "error": (
                    f"requested move of {path_len:.3f} m exceeds the {self._max_move_m} m "
                    "single-move safety cap; issue smaller moves."
                ),
                "current_xyz": [round(float(v), 4) for v in cur[:3]],
            }

        if quat_xyzw is not None:
            target_quat = np.asarray(quat_xyzw, dtype=np.float64).reshape(-1)[:4]
            target_quat = _euler_to_quat(_clip_euler_window(_quat_to_euler(target_quat)))
        elif euler_xyz is not None:
            target_quat = _euler_to_quat(_clip_euler_window(euler_xyz))
        else:
            target_quat = cur[3:]

        self._apply_gripper(gripper)
        self._stream_to(target_xyz, target_quat)
        final, pos_err, reached = self._wait_until_reached(target_xyz, target_quat)
        return {
            "reached": reached,
            "pos_error_m": round(pos_err, 4),
            "clipped_to_workspace": clipped,
            "target_xyz": [round(float(v), 4) for v in target_xyz],
            "final_xyz": [round(float(v), 4) for v in final[:3]],
            "final_euler": [round(float(v), 4) for v in _quat_to_euler(final[3:])],
            "gripper_open": bool(self._backend.get_state()["gripper_open"]),
        }

    def move_delta(
        self,
        *,
        dxyz=None,
        drpy_deg=None,
        gripper: Optional[str] = None,
    ) -> dict:
        """Nudge the TCP by a relative ``dxyz`` (meters) and/or ``drpy_deg``
        (degrees, applied in the base frame), for fine alignment.
        """
        cur = self._backend.get_tcp_pose()
        target_xyz = cur[:3].copy()
        if dxyz is not None:
            target_xyz = target_xyz + np.asarray(dxyz, dtype=np.float64).reshape(-1)[:3]

        euler = _quat_to_euler(cur[3:])
        if drpy_deg is not None:
            euler = euler + np.radians(np.asarray(drpy_deg, dtype=np.float64).reshape(-1)[:3])

        return self.move_to(
            target_xyz, euler_xyz=euler, gripper=gripper
        )

    def open_gripper(self) -> dict:
        self._backend.open_gripper()
        time.sleep(0.6)
        return {"gripper_open": True, "gripper_width": self._backend.get_state()["gripper_width"]}

    def close_gripper(self) -> dict:
        self._backend.close_gripper()
        time.sleep(0.6)
        return {"gripper_open": False, "gripper_width": self._backend.get_state()["gripper_width"]}

    # -- reset / teardown --------------------------------------------------

    def reset(self) -> tuple[dict, dict]:
        """Clear errors and drive the arm to its home pose; return ``(obs, {})``."""
        self._backend.clear_errors()
        self._stream_to(self._reset_xyz, self._reset_quat)
        time.sleep(0.5)
        return self._make_obs(), {}

    def close(self) -> None:
        for cam in self._cameras:
            cam.close()
        self._backend.shutdown()


# ---------------------------------------------------------------------------
# RPC dispatcher + parent watchdog (mirrors the LIBERO / LeRobot drivers)
# ---------------------------------------------------------------------------


_INITIAL_PPID = os.getppid()


def _start_parent_watchdog(
    server: SocketRpcServer, shutdown_event: threading.Event, poll_s: float = 2.0
) -> None:
    """Shut the RPC server down if the agent (parent) process dies."""

    def _watch() -> None:
        while not shutdown_event.is_set():
            time.sleep(poll_s)
            ppid = os.getppid()
            if ppid != _INITIAL_PPID or ppid == 1:
                logger.warning("parent died (ppid %s -> %s); stopping", _INITIAL_PPID, ppid)
                shutdown_event.set()
                threading.Thread(target=server.shutdown, daemon=True).start()
                return

    threading.Thread(target=_watch, daemon=True).start()


def _build_dispatcher(env: FrankaAgentEnv, shutdown_event: threading.Event):
    def dispatch(method: str, args: tuple, kwargs: dict):
        if method.startswith("env."):
            attr = method[len("env."):]
            try:
                return getattr(env, attr)(*args, **kwargs)
            except Exception as e:
                logger.warning("env method %s failed: %s", method, e)
                raise
        if method == "shutdown":
            shutdown_event.set()
            return {"ok": True}
        raise ValueError(f"unknown RPC method: {method!r}")

    return dispatch


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_cameras(specs: list[str] | None) -> list[tuple[str, str]]:
    if not specs:
        return list(_DEFAULT_CAMERAS)
    out: list[tuple[str, str]] = []
    for spec in specs:
        name, _, serial = spec.partition(":")
        if not name or not serial:
            raise ValueError(f"--camera expects name:serial, got {spec!r}")
        out.append((name, serial))
    return out


def _build_cameras(specs: list[tuple[str, str]]) -> list[RealSenseDepthCamera]:
    cams: list[RealSenseDepthCamera] = []
    for name, serial in specs:
        try:
            cams.append(
                RealSenseDepthCamera(
                    name, serial, width=_CAM_WIDTH, height=_CAM_HEIGHT, fps=_CAM_FPS
                )
            )
        except Exception as e:
            logger.warning("could not open camera '%s' (serial %s): %s", name, serial, e)
    return cams


def main() -> int:
    p = argparse.ArgumentParser(description="Standalone Franka env server")
    p.add_argument("--output-dir", type=str, required=True)
    p.add_argument("--robot-ip", type=str, default=_DEFAULT_ROBOT_IP)
    p.add_argument("--ros-pkg", type=str, default=_DEFAULT_ROS_PKG)
    p.add_argument("--no-gripper", action="store_true", help="Do not load the Franka Hand.")
    p.add_argument(
        "--camera", action="append", default=None,
        help="Camera as name:serial (repeatable). Defaults to the two bench D435I.",
    )
    p.add_argument("--transport-host", type=str, default="127.0.0.1")
    p.add_argument("--transport-port", type=int, default=0)
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    init_output_dir(args.output_dir)
    logger.info(
        "starting Franka env server: robot_ip=%s output_dir=%s",
        args.robot_ip, args.output_dir,
    )

    backend = FrankaRobotBackend(
        robot_ip=args.robot_ip,
        ros_pkg=args.ros_pkg,
        load_gripper=not args.no_gripper,
    )
    cameras = _build_cameras(_parse_cameras(args.camera))
    env = FrankaAgentEnv(backend, cameras)

    shutdown_event = threading.Event()
    dispatch = _build_dispatcher(env, shutdown_event)
    server = SocketRpcServer((args.transport_host, args.transport_port), dispatch)
    bound_host, bound_port = server.server_address
    client_host = "127.0.0.1" if bound_host == "0.0.0.0" else bound_host
    print(
        json.dumps({
            "event": "transport_ready", "kind": "socket",
            "host": client_host, "port": bound_port,
        }),
        flush=True,
    )
    logger.info("RPC server listening on %s:%s", client_host, bound_port)

    _start_parent_watchdog(server, shutdown_event)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        shutdown_event.wait()
    finally:
        server.shutdown()
        server.server_close()
        env.close()
    logger.info("driver exited cleanly")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
