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

"""Planner-facing motions and observations over RLinf's R2 environment."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation

from robots.r2.camera_geometry import BASE_FRAME, UrdfCameraChains
from robots.r2.runtime_config import ARMS, R2SiteConfig
from rpent.utils.logging import get_logger

logger = get_logger("r2_env")

#: Bumped when an RPC route or payload changes incompatibly.
PROTOCOL_VERSION = 1


def _refusal(message: str, **details: Any) -> dict[str, Any]:
    return {"ok": False, "error": message, **details}


def _rounded(values: Any) -> list[float]:
    return [round(float(v), 4) for v in np.asarray(values).reshape(-1)]


class R2AgentEnv:
    """Bounded arm, gripper, head, and chassis motion on one RLinf ``R2Env``.

    The RLinf env owns the hardware: it composes and connects the robot,
    returns the arms to their rest pose on reset, and disconnects on close.
    This adapter adds the motions a planner commands, each checked against
    the site bounds before anything moves. Construction never moves the robot.

    TCP poses are ``[x, y, z, roll, pitch, yaw]`` in ``base_link`` (extrinsic
    ``xyz`` Euler angles), computed from the measured joints with the same
    model the IK solver uses, so a relative move composes with what it reports.
    """

    def __init__(
        self,
        site: R2SiteConfig,
        *,
        env: Any,
        connection: Any,
        joint_names: Mapping[str, Sequence[str]],
    ) -> None:
        self._site = site
        self._env = env
        self._closed = False
        self._connection = connection
        self._joint_names = {
            group: tuple(names) for group, names in joint_names.items()
        }
        self._camera_names = tuple(connection.camera_names)
        self._chains = (
            UrdfCameraChains(
                site.urdf,
                {name: site.cameras[name].frame for name in self._camera_names},
            )
            if site.urdf
            else None
        )

    @classmethod
    def from_site(cls, site: R2SiteConfig) -> R2AgentEnv:
        """Connect RLinf's ``R2Env`` to the robot described by ``site``."""
        from rlinf.envs.real.r2 import R2Env, R2EnvConfig
        from rlinf.robotics import R2Config, RobotInfo
        from rlinf.robotics.parts.arms.r2 import (
            R2_LEFT_ARM_JOINTS,
            R2_NECK_JOINTS,
            R2_RIGHT_ARM_JOINTS,
            R2_WAIST_JOINTS,
        )

        hardware = R2Config(**site.robot)
        rest = {
            "reset_joint_speed": site.rest.get("joint_speed"),
            "reset_duration": site.rest.get("duration"),
            "reset_joint_qpos": site.rest.get("joint_qpos"),
        }
        env = R2Env(
            R2EnvConfig(
                enable_camera_player=False,
                reset_on_init=False,
                **{key: value for key, value in rest.items() if value is not None},
            ),
            robot_info=RobotInfo(
                type="R2", model=hardware.hardware_model("R2"), config=hardware
            ),
        )
        return cls(
            site,
            env=env,
            connection=env.robot.child("left").child("arm").owner,
            joint_names={
                "waist": R2_WAIST_JOINTS,
                "neck": R2_NECK_JOINTS,
                "left_arm": R2_LEFT_ARM_JOINTS,
                "right_arm": R2_RIGHT_ARM_JOINTS,
            },
        )

    # Metadata and observation

    def get_env_meta(self) -> dict[str, Any]:
        """Describe the robot variant and the bounds the planner works within."""
        motion = self._site.motion
        base = self._site.base
        return {
            "robot": "r2",
            "protocol": PROTOCOL_VERSION,
            "frame": BASE_FRAME,
            "body_dof": int(self._connection.body_dof),
            "end_effector": str(self._connection.end_effector_type),
            "cameras": list(self._camera_names),
            "depth": bool(self._connection.enable_depth),
            "base_motion": base.allow_motion,
            "limits": {
                "max_joint_speed": motion.max_joint_speed,
                "max_move_m": motion.max_move_m,
                "max_delta_m": motion.max_delta_m,
                "max_rotation_rad": motion.max_rotation_rad,
                "workspace": {
                    arm: {"low": list(box.low), "high": list(box.high)}
                    for arm, box in motion.workspace.items()
                },
                "neck_low": _rounded(self._neck_spec.low),
                "neck_high": _rounded(self._neck_spec.high),
                **(
                    {
                        "base_max_linear_speed": base.max_linear_speed,
                        "base_max_angular_speed": base.max_angular_speed,
                        "base_max_duration": base.max_duration,
                    }
                    if base.allow_motion
                    else {}
                ),
            },
        }

    @property
    def _neck_spec(self) -> Any:
        return self._connection.joint_group_specs["neck"]

    def _measured(self) -> tuple[Any, dict[str, np.ndarray]]:
        """Return the state snapshot and both TCP poses from its joints."""
        state = self._connection.get_state()
        tcp = self._connection.forward_kinematics(
            np.concatenate(
                [
                    state.waist_positions,
                    state.left_joint_positions,
                    state.right_joint_positions,
                ]
            )
        )
        return state, {arm: np.asarray(tcp[f"{arm}_tcp_pose"]) for arm in ARMS}

    def robot_state(self) -> dict[str, Any]:
        """Return the measured robot state and its health."""
        state, tcp = self._measured()
        return self._state_dict(state, tcp)

    def _state_dict(self, state: Any, tcp: Mapping[str, np.ndarray]) -> dict[str, Any]:
        arms = {}
        for arm in ARMS:
            entry = {
                "tcp_xyz": _rounded(tcp[arm][:3]),
                "tcp_rpy": _rounded(tcp[arm][3:]),
                "joint_positions": _rounded(getattr(state, f"{arm}_joint_positions")),
            }
            if self._connection.end_effector_type == "gripper":
                entry["gripper_opening"] = round(
                    float(getattr(state, f"{arm}_gripper_position")[0]), 3
                )
            arms[arm] = entry
        neck = _rounded(state.neck_positions)
        payload = {
            "frame": BASE_FRAME,
            "arms": arms,
            "neck": {"yaw": neck[0], "pitch": neck[1]},
            "waist": _rounded(state.waist_positions),
            "health": self._connection.diagnose().to_dict(),
        }
        if self._site.robot.get("enable_base", False):
            payload["base_pose"] = _rounded(state.base_pose)
        return payload

    def observe(self) -> dict[str, Any]:
        """Read the robot state, every camera, and the matching camera poses."""
        state, tcp = self._measured()
        views: dict[str, dict[str, Any]] = {}
        for name in self._camera_names:
            if not self._connection.check_camera(name, timeout=0.5):
                views[name] = {"error": f"camera {name!r} has delivered no frame"}
                continue
            view: dict[str, Any] = {"rgb": self._connection.get_camera(name)}
            depth = self._connection.get_camera_depth(name)
            if depth is not None:
                # Millimetres keep the sensor's uint16 resolution at half the size.
                view["depth_mm"] = np.clip(np.round(depth * 1000.0), 0, 65535).astype(
                    np.uint16
                )
            views[name] = view
        return {
            "robot_state": self._state_dict(state, tcp),
            "views": views,
            "camera_meta": self._camera_meta(state, views),
        }

    def _camera_meta(
        self, state: Any, views: Mapping[str, Mapping[str, Any]]
    ) -> dict[str, Any]:
        joints: dict[str, float] = {}
        for group, values in (
            ("waist", state.waist_positions),
            ("neck", state.neck_positions),
            ("left_arm", state.left_joint_positions),
            ("right_arm", state.right_joint_positions),
        ):
            joints.update(zip(self._joint_names[group], map(float, values)))
        cameras = {}
        for name in self._camera_names:
            calibration = self._site.cameras.get(name)
            view = views.get(name, {})
            rgb = view.get("rgb")
            cameras[name] = {
                "frame": calibration.frame if calibration else None,
                "image_shape": list(np.asarray(rgb).shape[:2])
                if rgb is not None
                else None,
                "intrinsics": calibration.intrinsics if calibration else None,
                "distortion": (
                    list(calibration.distortion)
                    if calibration and calibration.distortion
                    else None
                ),
                "T_base_camera": (
                    self._chains.pose(name, joints).round(6).tolist()
                    if self._chains is not None
                    else None
                ),
                "depth_shape": (
                    list(view["depth_mm"].shape) if "depth_mm" in view else None
                ),
            }
        return {
            "frame": BASE_FRAME,
            "optical_axes": "x right, y down, z forward",
            "pixel_order": "row_col",
            "depth_unit": "mm",
            "cameras": cameras,
        }

    def get_camera_meta(self) -> dict[str, Any]:
        """Return camera intrinsics and current ``base_link`` poses."""
        state, _ = self._measured()
        return self._camera_meta(state, {})

    # Motion

    def _check_arm(self, arm: str) -> None:
        if arm not in ARMS:
            raise ValueError(f"arm must be one of {ARMS}, got {arm!r}")

    def move_to(
        self, arm: str, xyz: Sequence[float], rpy: Sequence[float] | None = None
    ) -> dict[str, Any]:
        """Move one TCP to an absolute pose, keeping its orientation by default."""
        self._check_arm(arm)
        _, tcp = self._measured()
        start = tcp[arm]
        target_xyz = np.asarray(xyz, dtype=np.float64).reshape(3)
        distance = float(np.linalg.norm(target_xyz - start[:3]))
        if distance > self._site.motion.max_move_m:
            return _refusal(
                f"move of {distance:.3f} m exceeds max_move_m "
                f"{self._site.motion.max_move_m} m; split it into shorter moves",
                start_tcp=_rounded(start),
            )
        target_rpy = start[3:] if rpy is None else np.asarray(rpy, dtype=np.float64)
        return self._move_tcp(arm, start, target_xyz, target_rpy)

    def move_delta(self, arm: str, delta_xyz: Sequence[float]) -> dict[str, Any]:
        """Translate one TCP by a ``base_link`` offset, keeping its orientation."""
        self._check_arm(arm)
        delta = np.asarray(delta_xyz, dtype=np.float64).reshape(3)
        if float(np.linalg.norm(delta)) > self._site.motion.max_delta_m:
            return _refusal(
                f"delta of {np.linalg.norm(delta):.3f} m exceeds max_delta_m "
                f"{self._site.motion.max_delta_m} m"
            )
        _, tcp = self._measured()
        start = tcp[arm]
        return self._move_tcp(arm, start, start[:3] + delta, start[3:])

    def rotate_delta(self, arm: str, delta_rpy: Sequence[float]) -> dict[str, Any]:
        """Rotate one TCP about the ``base_link`` axes, keeping its position."""
        self._check_arm(arm)
        rotation = Rotation.from_euler("xyz", np.asarray(delta_rpy, dtype=np.float64))
        if rotation.magnitude() > self._site.motion.max_rotation_rad:
            return _refusal(
                f"rotation of {rotation.magnitude():.3f} rad exceeds "
                f"max_rotation_rad {self._site.motion.max_rotation_rad} rad"
            )
        _, tcp = self._measured()
        start = tcp[arm]
        target = rotation * Rotation.from_euler("xyz", start[3:])
        return self._move_tcp(arm, start, start[:3], target.as_euler("xyz"))

    def _move_tcp(
        self,
        arm: str,
        start: np.ndarray,
        xyz: np.ndarray,
        rpy: np.ndarray,
    ) -> dict[str, Any]:
        if not (np.all(np.isfinite(xyz)) and np.all(np.isfinite(rpy))):
            raise ValueError(f"TCP target must be finite, got xyz={xyz} rpy={rpy}")
        box = self._site.motion.workspace[arm]
        if not box.contains(xyz):
            return _refusal(
                f"target {_rounded(xyz)} is outside the {arm} workspace",
                workspace={"low": list(box.low), "high": list(box.high)},
                start_tcp=_rounded(start),
            )
        target = np.concatenate([xyz, rpy]).astype(np.float32)
        try:
            solved = self._connection.inverse_kinematics(**{f"{arm}_tcp_pose": target})
        except ValueError as error:
            return _refusal(str(error), start_tcp=_rounded(start))
        completed = self._connection.move_body_smooth(
            **{f"{arm}_arm": solved[f"{arm}_joint_positions"]},
            max_velocity=self._site.motion.max_joint_speed,
        )
        _, tcp = self._measured()
        final = tcp[arm]
        result = {
            "ok": bool(completed),
            "arm": arm,
            "start_tcp": _rounded(start),
            "target_tcp": _rounded(target),
            "final_tcp": _rounded(final),
            "position_error_m": round(float(np.linalg.norm(final[:3] - xyz)), 4),
            "rotation_error_rad": round(
                float(
                    (
                        Rotation.from_euler("xyz", rpy)
                        * Rotation.from_euler("xyz", final[3:]).inv()
                    ).magnitude()
                ),
                4,
            ),
        }
        if not completed:
            result["error"] = "the move did not complete"
            result["health"] = self._connection.diagnose().to_dict()
        return result

    def set_gripper(self, arm: str, opening: float) -> dict[str, Any]:
        """Move one gripper to ``opening`` (0 closed, 1 fully open)."""
        self._check_arm(arm)
        if self._connection.end_effector_type != "gripper":
            return _refusal(
                f"this R2 has end effector {self._connection.end_effector_type!r}, "
                "not a gripper"
            )
        value = float(opening)
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"opening must be within [0, 1], got {value}")
        completed = self._connection.move_body_smooth(**{f"{arm}_gripper": value})
        state, _ = self._measured()
        result = {
            "ok": bool(completed),
            "arm": arm,
            "target_opening": value,
            "measured_opening": round(
                float(getattr(state, f"{arm}_gripper_position")[0]), 3
            ),
        }
        if not completed:
            result["error"] = "the gripper command did not complete"
            result["health"] = self._connection.diagnose().to_dict()
        return result

    def look(self, yaw: float, pitch: float) -> dict[str, Any]:
        """Turn the head to absolute neck angles in radians."""
        target = np.array([float(yaw), float(pitch)])
        low, high = np.asarray(self._neck_spec.low), np.asarray(self._neck_spec.high)
        if np.any(target < low) or np.any(target > high):
            return _refusal(
                f"neck target {_rounded(target)} is outside the limits",
                neck_low=_rounded(low),
                neck_high=_rounded(high),
            )
        completed = self._connection.move_body_smooth(
            neck=target, max_velocity=self._site.motion.max_joint_speed
        )
        state, _ = self._measured()
        result = {"ok": bool(completed), "neck": _rounded(state.neck_positions)}
        if int(self._connection.body_dof) == 20:
            result["notice"] = "This 20-DoF body has no neck yaw; only pitch moved."
        if not completed:
            result["error"] = "the head move did not complete"
            result["health"] = self._connection.diagnose().to_dict()
        return result

    def go_home(self) -> dict[str, Any]:
        """Return both arms to the configured rest pose."""
        try:
            self._env.go_to_rest()
        except RuntimeError as error:
            return _refusal(str(error), health=self._connection.diagnose().to_dict())
        return {"ok": True, "robot_state": self.robot_state()}

    def move_base(
        self, vx: float, vy: float, wz: float, duration: float
    ) -> dict[str, Any]:
        """Drive the chassis at a body-frame velocity for ``duration`` seconds."""
        limits = self._site.base
        if not limits.allow_motion:
            return _refusal("base motion is disabled in this site's config")
        linear = float(np.hypot(vx, vy))
        if linear > limits.max_linear_speed or abs(wz) > limits.max_angular_speed:
            return _refusal(
                f"base velocity exceeds {limits.max_linear_speed} m/s or "
                f"{limits.max_angular_speed} rad/s"
            )
        if not 0.0 < duration <= limits.max_duration:
            return _refusal(f"duration must be within (0, {limits.max_duration}] s")
        state, _ = self._measured()
        start = _rounded(state.base_pose)
        self._connection.move_base_pulse([vx, vy, wz], duration=float(duration))
        state, _ = self._measured()
        return {
            "ok": True,
            "start_base_pose": start,
            "base_pose": _rounded(state.base_pose),
        }

    # Episode lifecycle

    def reset(self) -> dict[str, Any]:
        """Return the arms to their rest pose to start an episode."""
        self._env.reset()
        return {"ok": True, "robot_state": self.robot_state()}

    def close(self) -> None:
        """Disconnect the robot. Safe to call more than once."""
        if not self._closed:
            self._closed = True
            self._env.close()
