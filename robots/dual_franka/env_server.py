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

"""RPC server owning one RLinf dual-Franka ``RealWorldEnv`` worker."""

from __future__ import annotations

import queue
import sys
import threading
import time
from typing import Any

import numpy as np

from robots.dual_franka.runtime_config import load_runtime_config
from robots.franka.env_server import FrankaEnvFacade, _to_numpy_tree, main
from rpent.utils.config import get_repo_root, get_rlinf_repo_path

# Resolve the RLinf checkout before the deferred ``import rlinf`` executes.
RPENT_ROOT = get_repo_root()
RLINF_REPO_PATH = get_rlinf_repo_path() or (RPENT_ROOT.parent / "rlinf").resolve()
if str(RLINF_REPO_PATH) not in sys.path:
    sys.path.insert(0, str(RLINF_REPO_PATH))

_ARM_INDEX = {"left": 0, "right": 1}

FRANKA_JOINT_LIMIT_LOW = np.asarray(
    [-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973],
    dtype=np.float32,
)
FRANKA_JOINT_LIMIT_HIGH = np.asarray(
    [2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973],
    dtype=np.float32,
)


class DualFrankaEnvFacade(FrankaEnvFacade):
    """Expose dual-arm-only recovery in addition to the common Franka RPCs."""

    _METHODS = (*FrankaEnvFacade._METHODS, "recover_joint_posture")


def _batch_raw_obs(raw_obs: dict[str, Any]) -> dict[str, Any]:
    """Add the vector-env batch axis to one raw real-world observation."""
    output: dict[str, Any] = {}
    for section, values in raw_obs.items():
        if isinstance(values, dict):
            output[section] = {
                key: np.expand_dims(np.asarray(value), axis=0)
                for key, value in values.items()
            }
        else:
            output[section] = np.expand_dims(np.asarray(values), axis=0)
    return output


def _matrix_to_rot6d(matrix: np.ndarray) -> np.ndarray:
    """Encode a 3x3 rotation as its first two columns (RLinf rot6d convention)."""
    matrix = np.asarray(matrix, dtype=np.float32)
    return np.concatenate([matrix[:, 0], matrix[:, 1]]).astype(np.float32)


def _pack_dual_action(
    left_xyz: np.ndarray,
    left_rot6d: np.ndarray,
    right_xyz: np.ndarray,
    right_rot6d: np.ndarray,
    *,
    left_grip: float = 0.0,
    right_grip: float = 0.0,
) -> np.ndarray:
    """Assemble a 20-D ``[L_xyz, L_rot6d, L_grip, R_xyz, R_rot6d, R_grip]`` action."""
    left = np.concatenate(
        [
            np.asarray(left_xyz, dtype=np.float32),
            np.asarray(left_rot6d, dtype=np.float32),
            np.array([left_grip], dtype=np.float32),
        ]
    )
    right = np.concatenate(
        [
            np.asarray(right_xyz, dtype=np.float32),
            np.asarray(right_rot6d, dtype=np.float32),
            np.array([right_grip], dtype=np.float32),
        ]
    )
    return np.concatenate([left, right]).astype(np.float32)


def _create_worker_class():
    """Build the Worker subclass only inside the RLinf server environment."""
    from rlinf.envs.realworld.common.camera import CameraInfo, create_camera
    from rlinf.envs.realworld.realworld_env import RealWorldEnv
    from rlinf.scheduler import Worker
    from scipy.spatial.transform import Rotation as Rotation

    from robots.dual_franka.perception import (
        load_calibration_bundle,
        transform_pose_between_base_frames,
    )

    class DualFrankaEnvWorker(Worker):
        """Ray worker that owns both physical Franka arms and camera resources."""

        def __init__(self, cfg: Any, controller_config: dict[str, Any]):
            super().__init__()
            self.cfg = cfg
            self.controller = dict(controller_config)
            self.env = RealWorldEnv(
                cfg.env.eval,
                num_envs=1,
                seed_offset=0,
                total_num_processes=1,
                worker_info=self.worker_info,
            )
            self.action_dim = int(self.env.action_space.shape[-1])
            self.per_arm_dim = int(
                self.env.env.call("get_wrapper_attr", "PER_ARM_ACTION_DIM")[0]
            )
            self.gripper_idx = int(
                self.env.env.call("get_wrapper_attr", "GRIPPER_IDX_IN_ARM")[0]
            )
            if self.per_arm_dim != 10 or self.action_dim != 2 * self.per_arm_dim:
                raise ValueError(
                    "dual-Franka RPent bridge requires the TCP rot6d env "
                    f"(20-D); got action_dim={self.action_dim}, "
                    f"per_arm_dim={self.per_arm_dim}"
                )
            env_config = self.env.env.call("get_wrapper_attr", "config")[0]
            self.action_scale = np.asarray(env_config.action_scale, dtype=np.float32)
            self.last_obs: dict[str, Any] | None = None
            self._perception_cameras: dict[str, Any] = {}
            self._perception_camera_last_frames: dict[str, np.ndarray] = {}
            self._perception_camera_meta: dict[str, dict[str, Any]] = {}
            try:
                self._open_perception_cameras()
            except Exception:
                self.env.close()
                raise

        # ------------------------------------------------------------ lifecycle

        def get_env_meta(self) -> dict[str, Any]:
            return {
                "ok": True,
                "action_dim": self.action_dim,
                "per_arm_dim": self.per_arm_dim,
                "action_scale": self.action_scale.tolist(),
                "arms": ["left", "right"],
                "perception_cameras": sorted(self._perception_cameras),
                "agent_observation": self.controller.get("agent_observation", {}),
                "projection_views": self.controller.get("projection_views", {}),
            }

        def close_env(self) -> None:
            for camera in self._perception_cameras.values():
                try:
                    camera.close()
                except Exception:
                    pass
            self._perception_cameras.clear()
            try:
                self.env.close()
            except Exception:
                pass

        def reset(self) -> dict[str, Any]:
            observation, info = self.env.reset()
            self.last_obs = observation
            return {
                "ok": True,
                "info": _to_numpy_tree(info),
                "robot_state": self.get_robot_state(),
            }

        # --------------------------------------------------------- observation

        def _ensure_obs(self) -> dict[str, Any]:
            if self.last_obs is None:
                raise RuntimeError(
                    "env.reset() has not been called; observation is unavailable. "
                    "Call reset before reading observations."
                )
            return self.last_obs

        @staticmethod
        def _strip_batch(value: Any) -> Any:
            array = _to_numpy_tree(value)
            if isinstance(array, np.ndarray) and array.ndim > 0 and array.shape[0] == 1:
                return array[0]
            if isinstance(array, list) and len(array) == 1:
                return array[0]
            return array

        def get_observation(self) -> dict[str, Any]:
            observation = self._ensure_obs()
            output = {
                key: self._strip_batch(value) for key, value in observation.items()
            }
            value = output.get("extra_view_images")
            if (
                isinstance(value, np.ndarray)
                and value.ndim == 5
                and value.shape[0] == 1
            ):
                output["extra_view_images"] = value[0]
            snapshot_getter = self.env.env.call(
                "get_wrapper_attr", "get_raw_camera_snapshot"
            )[0]
            snapshot = _to_numpy_tree(snapshot_getter())
            output["raw_camera_frames"] = snapshot.get("raw_frames", {})
            output["raw_camera_depths"] = snapshot.get("raw_depths", {})
            perception = self._capture_perception_camera_snapshot()
            for raw_key, frame in perception["raw_frames"].items():
                alias = raw_key.removesuffix("_rgb")
                output[f"{alias}_images"] = frame
            for raw_key, depth in perception["raw_depths"].items():
                alias = raw_key.removesuffix("_rgb")
                output[f"{alias}_depths"] = depth
            return output

        # --------------------------------------------------------- arm state

        def _arm_states(self) -> tuple[Any, Any]:
            left = self.env.env.call("get_wrapper_attr", "_left_state")[0]
            right = self.env.env.call("get_wrapper_attr", "_right_state")[0]
            return left, right

        @property
        def _raw_env(self) -> Any:
            return self.env.env.envs[0]

        def _raw_rlinf_env(self) -> Any:
            # PhysicalAgent alignment note: RPent needs a few real-robot
            # operations that RLinf's public vector-env surface does not expose
            # yet, most importantly gripper-preserving joint recovery and a
            # fresh raw observation after direct controller commands.  These
            # helpers deliberately reach into RLinf internals as a compatibility
            # bridge for the deployed dual-Franka setup; the cleaner long-term
            # shape is to upstream public RLinf methods for these operations.
            env = self._raw_env
            return env.unwrapped if hasattr(env, "unwrapped") else env

        def _refresh_robot_state(self) -> None:
            # See _raw_rlinf_env(): direct controller state refresh keeps RPent
            # snapshots/logs aligned with what the physical arms actually did
            # after reset_joint/open_gripper/close_gripper calls made outside
            # the normal vector-env step path.
            raw = self._raw_rlinf_env()
            if getattr(raw.config, "is_dummy", False):
                return
            raw._left_state = raw._left_ctrl.get_state().wait()[0]
            raw._right_state = raw._right_ctrl.get_state().wait()[0]

        def _refresh_wrapped_observation(self) -> None:
            raw = self._raw_rlinf_env()
            raw_obs = raw._get_observation()
            self.last_obs = self.env._wrap_obs(_batch_raw_obs(raw_obs))

        @staticmethod
        def _wait_if_needed(value: Any) -> Any:
            wait = getattr(value, "wait", None)
            if callable(wait):
                return wait()
            return value

        def _reset_both_joints_no_gripper(self, reset_qpos: Any) -> dict[str, Any]:
            # RLinf's normal env reset may change gripper state and home both
            # arms as one episode-boundary operation.  PhysicalAgent recovery
            # instead reset joints while preserving an object already held in a
            # gripper, so RPent commands the two arm controllers directly here.
            raw = self._raw_rlinf_env()
            results: dict[str, Any] = {}
            errors: dict[str, BaseException] = {}

            def run(arm: str, ctrl: Any, qpos: Any) -> None:
                try:
                    results[arm] = self._wait_if_needed(ctrl.reset_joint(qpos))
                except BaseException as exc:
                    errors[arm] = exc

            threads = [
                threading.Thread(
                    target=run,
                    args=("left", raw._left_ctrl, reset_qpos[0]),
                    daemon=True,
                ),
                threading.Thread(
                    target=run,
                    args=("right", raw._right_ctrl, reset_qpos[1]),
                    daemon=True,
                ),
            ]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()
            if errors:
                arm, exc = next(iter(errors.items()))
                raise RuntimeError(f"{arm} reset_joint failed: {exc}") from exc
            return results

        def _arm_poses(self) -> tuple[np.ndarray, np.ndarray]:
            left, right = self._arm_states()
            return (
                np.asarray(left.tcp_pose, dtype=np.float32),
                np.asarray(right.tcp_pose, dtype=np.float32),
            )

        def _calibration_bundle(self) -> dict[str, Any]:
            bundle = getattr(self, "_dual_franka_calibration_bundle", None)
            if bundle is None:
                bundle = load_calibration_bundle(self.controller.get("calibration_path"))
                self._dual_franka_calibration_bundle = bundle
            return bundle

        def _pose_to_world(self, arm: str, pose: Any) -> np.ndarray:
            source = "left_base" if arm == "left" else "right_base"
            return transform_pose_between_base_frames(
                pose,
                target="right_base",
                source=source,
                calibration=self._calibration_bundle(),
            ).astype(np.float32)

        def _pose_from_world(self, arm: str, pose: Any) -> np.ndarray:
            target = "left_base" if arm == "left" else "right_base"
            return transform_pose_between_base_frames(
                pose,
                target=target,
                source="right_base",
                calibration=self._calibration_bundle(),
            ).astype(np.float32)

        def _world_arm_poses(self) -> tuple[np.ndarray, np.ndarray]:
            left, right = self._arm_poses()
            return self._pose_to_world("left", left), self._pose_to_world(
                "right", right
            )

        def _arm_rot6d(self, pose: np.ndarray) -> np.ndarray:
            return _matrix_to_rot6d(Rotation.from_quat(pose[3:]).as_matrix())

        @staticmethod
        def _gripper_command_from_open(gripper_open: bool | None) -> float:
            if gripper_open is None:
                return 0.0
            return 1.0 if bool(gripper_open) else -1.0

        def _current_gripper_commands(self) -> tuple[float, float]:
            left_state, right_state = self._arm_states()
            return (
                self._gripper_command_from_open(
                    getattr(left_state, "gripper_open", None)
                ),
                self._gripper_command_from_open(
                    getattr(right_state, "gripper_open", None)
                ),
            )

        def _hold_action(
            self,
            left: np.ndarray,
            right: np.ndarray,
            *,
            left_grip: float | None = None,
            right_grip: float | None = None,
        ) -> np.ndarray:
            if left_grip is None or right_grip is None:
                current_left_grip, current_right_grip = self._current_gripper_commands()
                if left_grip is None:
                    left_grip = current_left_grip
                if right_grip is None:
                    right_grip = current_right_grip
            return _pack_dual_action(
                left[:3],
                self._arm_rot6d(left),
                right[:3],
                self._arm_rot6d(right),
                left_grip=left_grip,
                right_grip=right_grip,
            )

        def _joint_health_from_raw(self, raw: dict[str, Any]) -> dict[str, Any]:
            return {
                arm: self._joint_health_for_arm(arm, raw.get(arm))
                for arm in ("left", "right")
            }

        def _joint_health_for_arm(self, arm: str, state: Any) -> dict[str, Any]:
            if not isinstance(state, dict):
                return {"status": "unknown", "reason": "missing arm state"}
            native = self._native_joint_health(state)
            q = state.get("arm_joint_position")
            if not isinstance(q, (list, tuple, np.ndarray)) or len(q) != 7:
                status = (
                    "critical"
                    if native["critical_reasons"]
                    else "warning"
                    if native["warning_reasons"]
                    else "unknown"
                )
                return {
                    "status": status,
                    "reason": "missing arm_joint_position",
                    "native": native["summary"],
                    "reasons": native["critical_reasons"]
                    or native["warning_reasons"],
                }
            q_arr = np.asarray(q, dtype=np.float32)
            raw = self._raw_rlinf_env()
            reset_qpos = getattr(raw.config, "joint_reset_qpos", None)
            if isinstance(reset_qpos, (list, tuple)) and len(reset_qpos) >= 2:
                reset_idx = 0 if arm == "left" else 1
                nominal = np.asarray(reset_qpos[reset_idx], dtype=np.float32)
            else:
                nominal = np.zeros(7, dtype=np.float32)
            drift = q_arr - nominal
            margin = np.minimum(
                q_arr - FRANKA_JOINT_LIMIT_LOW,
                FRANKA_JOINT_LIMIT_HIGH - q_arr,
            )
            jacobian = state.get("arm_jacobian")
            sigma_min = None
            condition_number = None
            if isinstance(jacobian, (list, tuple, np.ndarray)):
                try:
                    sv = np.linalg.svd(
                        np.asarray(jacobian, dtype=np.float32),
                        compute_uv=False,
                    )
                    sigma_min = float(np.min(sv))
                    condition_number = float(np.max(sv) / max(np.min(sv), 1e-12))
                except Exception:
                    pass

            thresholds = self.controller["joint_health_thresholds"].get(arm, {})
            reasons: list[str] = []
            critical_reasons: list[str] = []
            reasons.extend(native["warning_reasons"])
            critical_reasons.extend(native["critical_reasons"])

            def check(name: str, value: float | None, warning_op, critical_op) -> None:
                if value is None:
                    return
                warning = thresholds.get(f"warning_{name}")
                critical = thresholds.get(f"critical_{name}")
                if critical is not None and critical_op(value, float(critical)):
                    critical_reasons.append(f"{name}={value:.4g}")
                elif warning is not None and warning_op(value, float(warning)):
                    reasons.append(f"{name}={value:.4g}")

            drift_l2 = float(np.linalg.norm(drift))
            min_joint_margin = float(np.min(margin))
            check("drift_l2", drift_l2, lambda v, t: v > t, lambda v, t: v > t)
            check("q1", float(q_arr[0]), lambda v, t: v > t, lambda v, t: v > t)
            check("q3", float(q_arr[2]), lambda v, t: v < t, lambda v, t: v < t)
            check(
                "condition_number",
                condition_number,
                lambda v, t: v > t,
                lambda v, t: v > t,
            )
            check("sigma_min", sigma_min, lambda v, t: v < t, lambda v, t: v < t)
            check(
                "min_joint_margin",
                min_joint_margin,
                lambda v, t: v < t,
                lambda v, t: v < t,
            )
            status = "critical" if critical_reasons else "warning" if reasons else "ok"
            return {
                "status": status,
                "reasons": critical_reasons or reasons,
                "q": q_arr.round(5).tolist(),
                "nominal_q": nominal.round(5).tolist(),
                "drift_from_nominal": drift.round(5).tolist(),
                "drift_l2": round(drift_l2, 5),
                "max_abs_drift": round(float(np.max(np.abs(drift))), 5),
                "min_joint_margin": round(min_joint_margin, 5),
                "worst_margin_joint": int(np.argmin(margin)) + 1,
                "condition_number": round(condition_number, 5)
                if condition_number is not None
                else None,
                "sigma_min": round(sigma_min, 5) if sigma_min is not None else None,
                "native": native["summary"],
            }

        @staticmethod
        def _native_joint_health(state: dict[str, Any]) -> dict[str, Any]:
            def active_flags(value: Any) -> list[str]:
                if not isinstance(value, dict):
                    return []
                return [
                    str(name)
                    for name, enabled in value.items()
                    if isinstance(enabled, (bool, np.bool_)) and bool(enabled)
                ]

            def any_array(value: Any) -> bool:
                if not isinstance(value, (list, tuple, np.ndarray)):
                    return False
                return bool(np.asarray(value, dtype=bool).any())

            warning_reasons: list[str] = []
            critical_reasons: list[str] = []
            mode = state.get("robot_mode")
            mode_text = str(mode) if mode is not None else None
            mode_lower = mode_text.lower() if mode_text else ""
            if bool(state.get("has_errors")):
                critical_reasons.append("franka_has_errors")
            for name in active_flags(state.get("current_errors")):
                critical_reasons.append(f"current_error:{name}")
            # PhysicalAgent alignment note: these non-current native signals are
            # surfaced as warnings so the planner/operator can notice deteriorating
            # Franka health during long runs.  They are intentionally conservative
            # and may need demotion to summary-only for contact-rich deployments.
            for name in active_flags(state.get("last_motion_errors")):
                warning_reasons.append(f"last_motion_error:{name}")
            if mode_lower and any(token in mode_lower for token in ("error", "reflex")):
                critical_reasons.append(f"robot_mode:{mode_text}")
            elif mode_lower and "userstopped" in mode_lower:
                warning_reasons.append(f"robot_mode:{mode_text}")
            if any_array(state.get("joint_collision")):
                critical_reasons.append("joint_collision")
            if any_array(state.get("cartesian_collision")):
                critical_reasons.append("cartesian_collision")
            if any_array(state.get("joint_contact")):
                warning_reasons.append("joint_contact")
            if any_array(state.get("cartesian_contact")):
                warning_reasons.append("cartesian_contact")
            return {
                "critical_reasons": critical_reasons,
                "warning_reasons": warning_reasons,
                "summary": {
                    "robot_mode": mode_text,
                    "has_errors": state.get("has_errors"),
                    "current_errors": active_flags(state.get("current_errors")),
                    "last_motion_errors": active_flags(
                        state.get("last_motion_errors")
                    ),
                    "joint_contact": any_array(state.get("joint_contact")),
                    "cartesian_contact": any_array(state.get("cartesian_contact")),
                    "joint_collision": any_array(state.get("joint_collision")),
                    "cartesian_collision": any_array(
                        state.get("cartesian_collision")
                    ),
                },
            }

        def get_robot_state(self) -> dict[str, Any]:
            left, right = self._arm_states()
            left_raw = _to_numpy_tree(left)
            right_raw = _to_numpy_tree(right)
            left_out = dict(left_raw)
            right_out = dict(right_raw)
            if "tcp_pose" in left_raw:
                left_out["raw_tcp_pose"] = left_raw["tcp_pose"]
                left_out["raw_tcp_pose_frame"] = "left_base"
                left_out["tcp_pose"] = self._pose_to_world(
                    "left", left_raw["tcp_pose"]
                )
                left_out["tcp_pose_frame"] = "right_base"
            if "tcp_pose" in right_raw:
                right_out["raw_tcp_pose"] = right_raw["tcp_pose"]
                right_out["raw_tcp_pose_frame"] = "right_base"
                right_out["tcp_pose_frame"] = "right_base"
            return {
                "coordinate_frame": "right_base",
                "left_arm": left_out,
                "right_arm": right_out,
                "joint_health": self._joint_health_from_raw(
                    {"left": left_raw, "right": right_raw}
                ),
                "action_dim": self.action_dim,
                "per_arm_dim": self.per_arm_dim,
                "action_scale": self.action_scale.tolist(),
            }

        def get_camera_meta(self) -> dict[str, Any] | None:
            metadata: dict[str, Any] = {}
            try:
                specs_getter = self.env.env.call(
                    "get_wrapper_attr", "_all_camera_specs"
                )[0]
                specs = list(specs_getter())
            except Exception as exc:
                return metadata or {
                    "error": str(exc),
                    "error_type": type(exc).__name__,
                }
            cameras = {
                name: {"serial": serial, "type": camera_type}
                for name, serial, camera_type in specs
            }
            main_key = self.cfg.env.eval.get("main_image_key")
            extras = [name for name, _, _ in specs if name != main_key]
            metadata.update(
                {
                    name: {"serial": serial, "type": camera_type}
                    for name, serial, camera_type in specs
                }
            )
            metadata_getter = self.env.env.call(
                "get_wrapper_attr", "get_raw_camera_metadata"
            )[0]
            metadata.update(_to_numpy_tree(metadata_getter()))
            metadata.update(self._perception_camera_meta)
            return {
                "cameras": cameras,
                "observation_camera_map": {
                    "main": main_key,
                    **{f"extra_{index}": name for index, name in enumerate(extras)},
                },
                "agent_observation": self.controller.get("agent_observation", {}),
                "projection_views": self.controller.get("projection_views", {}),
                **metadata,
            }

        def _open_perception_cameras(self) -> None:
            cameras = self.controller["perception"].get("cameras")
            if not isinstance(cameras, dict):
                return
            for alias, raw_config in cameras.items():
                if not isinstance(raw_config, dict) or not bool(
                    raw_config.get("enabled", True)
                ):
                    continue
                resolution = tuple(
                    int(value) for value in raw_config.get("resolution", [640, 480])
                )
                if len(resolution) != 2:
                    raise ValueError(
                        f"perception camera {alias!r} resolution must have two values"
                    )
                info = CameraInfo(
                    name=f"{alias}_rgb",
                    serial_number=str(raw_config["serial_number"]),
                    camera_type=str(raw_config.get("camera_type", "realsense")),
                    resolution=resolution,
                    fps=int(raw_config.get("fps", 15)),
                    enable_depth=bool(raw_config.get("enable_depth", True)),
                )
                camera = create_camera(info)
                camera.open()
                try:
                    first_frame = camera.get_frame(timeout=8)
                except Exception:
                    camera.close()
                    raise
                self._perception_cameras[str(alias)] = camera
                self._perception_camera_last_frames[str(alias)] = np.asarray(
                    first_frame
                )

        def _capture_perception_camera_snapshot(self) -> dict[str, dict[str, Any]]:
            output: dict[str, dict[str, Any]] = {
                "raw_frames": {},
                "raw_depths": {},
                "camera_meta": {},
            }
            for alias, camera in self._perception_cameras.items():
                try:
                    frame = camera.get_frame(timeout=2)
                    self._perception_camera_last_frames[alias] = np.asarray(frame)
                except queue.Empty:
                    frame = self._perception_camera_last_frames.get(alias)
                    if frame is None:
                        continue
                frame = np.asarray(frame)
                if frame.ndim != 3 or frame.shape[-1] < 3:
                    continue
                rgb = frame[..., :3][..., ::-1].astype(np.uint8, copy=True)
                raw_key = f"{alias}_rgb"
                output["raw_frames"][raw_key] = rgb
                depth = None
                if frame.shape[-1] >= 4:
                    depth_scale = float(getattr(camera, "depth_scale", 1.0))
                    depth = frame[..., 3].astype(np.float32) * depth_scale
                    output["raw_depths"][raw_key] = depth
                intrinsics_getter = getattr(camera, "get_color_intrinsics", None)
                intrinsics = (
                    intrinsics_getter() if callable(intrinsics_getter) else None
                )
                info = camera._camera_info
                meta = {
                    "name": raw_key,
                    "camera_alias": alias,
                    "camera_type": info.camera_type,
                    "serial_number": info.serial_number,
                    "rgb_shape": list(rgb.shape),
                    "depth_shape": list(depth.shape) if depth is not None else None,
                    "depth_enabled": bool(info.enable_depth),
                    "depth_available": depth is not None,
                    "color_intrinsics": intrinsics,
                }
                output["camera_meta"][raw_key] = meta
                self._perception_camera_meta[raw_key] = meta
            return output

        # --------------------------------------------------------- primitives

        def _arm_index(self, arm: str) -> int:
            index = _ARM_INDEX.get(str(arm).strip().lower())
            if index is None:
                raise ValueError("arm must be 'left' or 'right'")
            return index

        def move_delta(self, arm: str, delta_xyz: Any) -> dict[str, Any]:
            arm_idx = self._arm_index(arm)
            arm_name = ["left", "right"][arm_idx]
            requested = np.asarray(delta_xyz, dtype=np.float32)
            left, right = self._arm_poses()
            start_local = left if arm_idx == 0 else right
            start_world = self._pose_to_world(arm_name, start_local)
            target_world = start_world.copy()
            target_world[:3] = start_world[:3] + requested
            max_step = self.controller["move_max_step_m"]
            deadline = time.time() + self.controller["move_timeout_s"]
            max_iterations = max(
                self.controller["min_iterations"],
                int(np.ceil(np.max(np.abs(requested)) / max_step))
                * self.controller["iteration_multiplier"],
            )
            iterations = 0
            while iterations < max_iterations and time.time() < deadline:
                left, right = self._arm_poses()
                current_local = left if arm_idx == 0 else right
                current_world = self._pose_to_world(arm_name, current_local)
                remaining = target_world[:3] - current_world[:3]
                if np.linalg.norm(remaining) <= self.controller["move_tolerance_m"]:
                    break
                step_xyz = np.clip(remaining, -max_step, max_step)
                next_world = current_world.copy()
                next_world[:3] = current_world[:3] + step_xyz
                next_local = self._pose_from_world(arm_name, next_world)
                action = self._hold_action(left, right)
                base = arm_idx * self.per_arm_dim
                action[base : base + 3] = next_local[:3]
                self.env.step(action[None, :], auto_reset=False)
                iterations += 1
            left, right = self._arm_poses()
            final_local = left if arm_idx == 0 else right
            final_world = self._pose_to_world(arm_name, final_local)
            error = float(np.linalg.norm(target_world[:3] - final_world[:3]))
            return {
                "ok": error <= self.controller["move_tolerance_m"],
                "arm": arm_name,
                "coordinate_frame": "right_base",
                "requested_delta_xyz": requested.tolist(),
                "start_tcp_pose": start_world.tolist(),
                "final_tcp_pose": final_world.tolist(),
                "target_tcp_pose": target_world.tolist(),
                "raw_tcp_pose_frame": "left_base"
                if arm_name == "left"
                else "right_base",
                "start_raw_tcp_pose": start_local.tolist(),
                "final_raw_tcp_pose": final_local.tolist(),
                "final_error_m": error,
                "steps_used": iterations,
            }

        def rotate_delta(self, arm: str, delta_rpy: Any) -> dict[str, Any]:
            arm_idx = self._arm_index(arm)
            arm_name = ["left", "right"][arm_idx]
            requested = np.asarray(delta_rpy, dtype=np.float32)
            left, right = self._arm_poses()
            start_local = left if arm_idx == 0 else right
            start_world = self._pose_to_world(arm_name, start_local)
            start_rot = Rotation.from_quat(start_world[3:])
            target_rot = Rotation.from_euler("xyz", requested) * start_rot
            max_step = self.controller["rotate_max_step_rad"]
            deadline = time.time() + self.controller["rotate_timeout_s"]
            max_iterations = max(
                self.controller["min_iterations"],
                int(np.ceil(np.linalg.norm(requested) / max_step))
                * self.controller["iteration_multiplier"],
            )
            iterations = 0
            error = float("inf")
            while iterations < max_iterations and time.time() < deadline:
                left, right = self._arm_poses()
                current_local = left if arm_idx == 0 else right
                current_world = self._pose_to_world(arm_name, current_local)
                current_rot = Rotation.from_quat(current_world[3:])
                error_rotvec = (target_rot * current_rot.inv()).as_rotvec()
                error = float(np.linalg.norm(error_rotvec))
                if error <= self.controller["rotate_tolerance_rad"]:
                    break
                if error > max_step:
                    error_rotvec = error_rotvec * (max_step / error)
                step_rot = Rotation.from_rotvec(error_rotvec) * current_rot
                next_world = current_world.copy()
                next_world[3:] = step_rot.as_quat()
                next_local = self._pose_from_world(arm_name, next_world)
                action = self._hold_action(left, right)
                base = arm_idx * self.per_arm_dim
                action[base + 3 : base + 9] = _matrix_to_rot6d(
                    Rotation.from_quat(next_local[3:]).as_matrix()
                )
                self.env.step(action[None, :], auto_reset=False)
                iterations += 1
            left, right = self._arm_poses()
            final_local = left if arm_idx == 0 else right
            final_world = self._pose_to_world(arm_name, final_local)
            return {
                "ok": error <= self.controller["rotate_tolerance_rad"],
                "arm": arm_name,
                "coordinate_frame": "right_base",
                "requested_delta_rpy": requested.tolist(),
                "start_tcp_pose": start_world.tolist(),
                "final_tcp_pose": final_world.tolist(),
                "raw_tcp_pose_frame": "left_base"
                if arm_name == "left"
                else "right_base",
                "start_raw_tcp_pose": start_local.tolist(),
                "final_raw_tcp_pose": final_local.tolist(),
                "final_error_rad": error,
                "steps_used": iterations,
            }

        def set_gripper(self, arm: str, *, open: bool) -> dict[str, Any]:
            arm_idx = self._arm_index(arm)
            deadline = time.time() + self.controller["gripper_timeout_s"]
            command = 1.0 if open else -1.0
            iterations = 0
            reached = False
            while (
                iterations < self.controller["gripper_max_iterations"]
                and time.time() < deadline
            ):
                left, right = self._arm_poses()
                action = self._hold_action(left, right)
                action[arm_idx * self.per_arm_dim + self.gripper_idx] = command
                self.env.step(action[None, :], auto_reset=False)
                time.sleep(self.controller["gripper_settle_s"])
                left_state, right_state = self._arm_states()
                state = left_state if arm_idx == 0 else right_state
                reached = bool(getattr(state, "gripper_open")) == bool(open)
                iterations += 1
                if reached:
                    break
            return {
                "ok": reached,
                "arm": ["left", "right"][arm_idx],
                "target_gripper_open": bool(open),
                "steps_used": iterations,
                "robot_state": self.get_robot_state(),
            }

        def recover_joint_posture(
            self,
            reason: str = "",
            return_to_start: bool = True,
        ) -> dict[str, Any]:
            # PhysicalAgent alignment note: this reproduces the live operator
            # recovery sequence used when one arm's joints drift toward a poor
            # posture during multi-stage VLA execution:
            #   1. record both TCP poses in the shared right_base frame;
            #   2. record and re-command gripper open/closed state so held
            #      objects remain clamped;
            #   3. reset both joint postures to the configured healthy qpos;
            #   4. optionally move/rotate both TCPs back near their pre-recovery
            #      world poses.
            # This is deliberately more specialized than RLinf's episode reset.
            self._ensure_obs()
            start_left, start_right = self._arm_poses()
            start_raw = {"left": start_left, "right": start_right}
            start = {
                "left": self._pose_to_world("left", start_left),
                "right": self._pose_to_world("right", start_right),
            }
            before_state = self.get_robot_state()

            def gripper_open_from_state(
                state: dict[str, Any],
            ) -> dict[str, bool | None]:
                gripper_open: dict[str, bool | None] = {}
                for arm in ("left", "right"):
                    arm_state = state.get(f"{arm}_arm", {})
                    value = (
                        arm_state.get("gripper_open")
                        if isinstance(arm_state, dict)
                        else None
                    )
                    gripper_open[arm] = bool(value) if value is not None else None
                return gripper_open

            start_gripper_open = gripper_open_from_state(before_state)

            def restore_grippers(stage: str) -> dict[str, Any]:
                current_state = self.get_robot_state()
                before = gripper_open_from_state(current_state)
                commands: dict[str, Any] = {}
                for arm in ("left", "right"):
                    target_open = start_gripper_open[arm]
                    current_open = before[arm]
                    if target_open is None:
                        commands[arm] = {
                            "ok": False,
                            "skipped": True,
                            "reason": "missing initial gripper_open state",
                        }
                    elif current_open == target_open:
                        commands[arm] = {
                            "ok": True,
                            "already_at_target": True,
                            "target_gripper_open": target_open,
                        }
                    else:
                        commands[arm] = self.set_gripper(arm, open=target_open)

                after_state = self.get_robot_state()
                after = gripper_open_from_state(after_state)
                ok = all(
                    target is not None and after[arm] == target
                    for arm, target in start_gripper_open.items()
                )
                return {
                    "ok": ok,
                    "stage": stage,
                    "target_gripper_open": start_gripper_open,
                    "before_gripper_open": before,
                    "after_gripper_open": after,
                    "commands": _to_numpy_tree(commands),
                }

            raw = self._raw_rlinf_env()

            def command_grippers_directly(stage: str) -> dict[str, Any]:
                if getattr(raw.config, "is_dummy", False):
                    return {
                        "ok": True,
                        "stage": stage,
                        "skipped": True,
                        "reason": "dummy environment",
                    }

                ctrls = {"left": raw._left_ctrl, "right": raw._right_ctrl}
                results: dict[str, Any] = {}
                errors: dict[str, BaseException] = {}

                def run(arm: str) -> None:
                    target_open = start_gripper_open[arm]
                    if target_open is None:
                        results[arm] = {
                            "ok": False,
                            "skipped": True,
                            "reason": "missing initial gripper_open state",
                        }
                        return
                    try:
                        ctrl = ctrls[arm]
                        method = (
                            ctrl.open_gripper if target_open else ctrl.close_gripper
                        )
                        results[arm] = self._wait_if_needed(method())
                    except BaseException as exc:
                        errors[arm] = exc

                threads = [
                    threading.Thread(target=run, args=(arm,), daemon=True)
                    for arm in ("left", "right")
                ]
                for thread in threads:
                    thread.start()
                for thread in threads:
                    thread.join()
                time.sleep(self.controller["gripper_settle_s"])
                self._refresh_robot_state()
                after = gripper_open_from_state(self.get_robot_state())
                return {
                    "ok": (not errors)
                    and all(
                        target is not None and after[arm] == target
                        for arm, target in start_gripper_open.items()
                    ),
                    "stage": stage,
                    "target_gripper_open": start_gripper_open,
                    "after_gripper_open": after,
                    "results": _to_numpy_tree(results),
                    "errors": {arm: str(exc) for arm, exc in errors.items()},
                    "reclamp_closed_grippers": True,
                }

            reset_qpos = getattr(raw.config, "joint_reset_qpos", None)
            if not isinstance(reset_qpos, (list, tuple)) or len(reset_qpos) < 2:
                raise RuntimeError(
                    "raw.config.joint_reset_qpos must contain left/right qpos"
                )

            direct_gripper_command_results: dict[str, Any] = {
                "before_joint_reset": command_grippers_directly("before_joint_reset")
            }
            if getattr(raw.config, "is_dummy", False):
                reset_results = {"left": "dummy", "right": "dummy"}
            else:
                reset_results = self._reset_both_joints_no_gripper(reset_qpos)
                time.sleep(0.5)
                self._refresh_robot_state()
                self._refresh_wrapped_observation()

            after_reset_state = self.get_robot_state()
            direct_gripper_command_results["after_joint_reset"] = (
                command_grippers_directly("after_joint_reset")
            )
            gripper_restore_results: dict[str, Any] = {
                "after_joint_reset": restore_grippers("after_joint_reset")
            }
            return_results: dict[str, Any] = {}
            if return_to_start:
                for arm in ("left", "right"):
                    current_raw = self._arm_poses()[self._arm_index(arm)]
                    current = self._pose_to_world(arm, current_raw)
                    target = start[arm]
                    return_results[f"{arm}_move"] = self.move_delta(
                        arm,
                        target[:3] - current[:3],
                    )
                    current_raw = self._arm_poses()[self._arm_index(arm)]
                    current = self._pose_to_world(arm, current_raw)
                    delta_rot = (
                        Rotation.from_quat(target[3:].copy())
                        * Rotation.from_quat(current[3:].copy()).inv()
                    )
                    return_results[f"{arm}_rotate"] = self.rotate_delta(
                        arm,
                        delta_rot.as_euler("xyz").astype(np.float32),
                    )

            direct_gripper_command_results["after_return_to_start"] = (
                command_grippers_directly("after_return_to_start")
            )
            gripper_restore_results["final_check"] = restore_grippers("final_check")
            final_state = self.get_robot_state()
            final_left_raw, final_right_raw = self._arm_poses()
            final_raw = {"left": final_left_raw, "right": final_right_raw}
            final = {
                "left": self._pose_to_world("left", final_left_raw),
                "right": self._pose_to_world("right", final_right_raw),
            }
            final_gripper_open = gripper_open_from_state(final_state)
            gripper_preserved = all(
                target is not None and final_gripper_open[arm] == target
                for arm, target in start_gripper_open.items()
            )
            pose_error = {}
            for arm in ("left", "right"):
                target = start[arm]
                actual = final[arm]
                pose_error[arm] = {
                    "translation_m": float(np.linalg.norm(actual[:3] - target[:3])),
                    "rotation_rad": float(
                        np.linalg.norm(
                            (
                                Rotation.from_quat(target[3:].copy())
                                * Rotation.from_quat(actual[3:].copy()).inv()
                            ).as_rotvec()
                        )
                    ),
                }

            return_ok = all(
                bool(value.get("ok"))
                for value in return_results.values()
                if isinstance(value, dict)
            )
            gripper_restore_ok = all(
                bool(value.get("ok"))
                for value in gripper_restore_results.values()
                if isinstance(value, dict)
            )
            direct_gripper_command_ok = all(
                bool(value.get("ok"))
                for value in direct_gripper_command_results.values()
                if isinstance(value, dict)
            )
            return {
                "ok": ((not return_to_start) or return_ok)
                and direct_gripper_command_ok
                and gripper_restore_ok
                and gripper_preserved,
                "primitive": "recover_joint_posture",
                "reason": str(reason or ""),
                "return_to_start": bool(return_to_start),
                "preserve_gripper_state": True,
                "behavior": (
                    "record each gripper open/closed state, directly re-command "
                    "that state before joint reset, reset both arms to configured "
                    "joint_reset_qpos, directly re-command grippers again, then "
                    "optionally return TCPs to their pre-recovery poses and verify "
                    "grippers once more"
                ),
                "start": {
                    "coordinate_frame": "right_base",
                    "left_tcp_pose": start["left"].tolist(),
                    "right_tcp_pose": start["right"].tolist(),
                    "left_raw_tcp_pose": start_raw["left"].tolist(),
                    "right_raw_tcp_pose": start_raw["right"].tolist(),
                    "gripper_open": start_gripper_open,
                    "joint_health": before_state.get("joint_health"),
                },
                "after_joint_reset": {
                    "joint_health": after_reset_state.get("joint_health"),
                    "gripper_open": gripper_open_from_state(after_reset_state),
                    "reset_results": _to_numpy_tree(reset_results),
                },
                "direct_gripper_command_results": _to_numpy_tree(
                    direct_gripper_command_results
                ),
                "gripper_restore_results": _to_numpy_tree(gripper_restore_results),
                "return_results": _to_numpy_tree(return_results),
                "final": {
                    "coordinate_frame": "right_base",
                    "joint_health": final_state.get("joint_health"),
                    "gripper_open": final_gripper_open,
                    "gripper_preserved": gripper_preserved,
                    "left_raw_tcp_pose": final_raw["left"].tolist(),
                    "right_raw_tcp_pose": final_raw["right"].tolist(),
                    "pose_error_from_start": pose_error,
                },
                "robot_state": final_state,
            }

        def chunk_step(
            self,
            actions: Any,
            *,
            return_all_frames: bool = False,
        ) -> dict[str, Any]:
            observations = []
            terminated = False
            truncated = False
            last_info: Any = None
            for action in np.asarray(actions, dtype=np.float32):
                observation, _reward, term, trunc, info = self.env.step(
                    action[None, :], auto_reset=False
                )
                self.last_obs = observation
                observations.append(self.get_observation())
                terminated = terminated or bool(np.asarray(_to_numpy_tree(term)).any())
                truncated = truncated or bool(np.asarray(_to_numpy_tree(trunc)).any())
                last_info = _to_numpy_tree(info)
                if terminated or truncated:
                    break
            return {
                "observation": observations if return_all_frames else observations[-1],
                "terminated": terminated,
                "truncated": truncated,
                "info": last_info,
            }

    return DualFrankaEnvWorker


if __name__ == "__main__":
    raise SystemExit(
        main(
            create_worker_class=_create_worker_class,
            load_runtime_config=load_runtime_config,
            facade_class=DualFrankaEnvFacade,
        )
    )
