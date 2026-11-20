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

"""R1Pro native robot state and official CuRobo single-arm inverse kinematics.

The solver reuses the loaded simulator robot and its official USD geometry. It solves
the goal joint configuration with CuRobo IK, then interpolates the arm joints
under the velocity limit. No path-level collision check is made against observed
surfaces; the simulator's physics is the contact authority, so a carried object
does not block a planned correction. CuRobo is imported only by the operations
that need it.
"""

from __future__ import annotations

import tempfile
from collections.abc import Callable
from copy import deepcopy
from pathlib import Path
from typing import Any, Literal

import numpy as np
import yaml

from robots.behavior.observation import (
    CONTROL_DT_S,
    public_proprio,
    valid_observation_token,
)
from robots.behavior.tools import decode_robot_state, extract_public_proprio

Arm = Literal["left", "right"]
TRUNK_JOINT_NAMES = tuple(f"torso_joint{i}" for i in range(1, 5))
ARM_JOINT_NAMES = {
    arm: tuple(f"{arm}_arm_joint{i}" for i in range(1, 8)) for arm in ("left", "right")
}
FINGER_JOINT_NAMES = {
    arm: tuple(f"{arm}_gripper_finger_joint{i}" for i in range(1, 3))
    for arm in ("left", "right")
}
# The base publishes velocity only. Each remaining group has measured qpos and qvel.
_ANGULAR_REST = (0.02, "radian_per_second")
_FINGER_REST = (0.002, "metre_per_second")
_CONTROL_GROUPS = (
    ("base", (), None, slice(0, 3), (0.01, "metre_or_radian_per_second")),
    ("trunk", TRUNK_JOINT_NAMES, slice(53, 57), slice(57, 61), _ANGULAR_REST),
    ("left_arm", ARM_JOINT_NAMES["left"], slice(3, 10), slice(10, 17), _ANGULAR_REST),
    (
        "right_arm",
        ARM_JOINT_NAMES["right"],
        slice(28, 35),
        slice(35, 42),
        _ANGULAR_REST,
    ),
    (
        "left_fingers",
        FINGER_JOINT_NAMES["left"],
        slice(24, 26),
        slice(26, 28),
        _FINGER_REST,
    ),
    (
        "right_fingers",
        FINGER_JOINT_NAMES["right"],
        slice(49, 51),
        slice(51, 53),
        _FINGER_REST,
    ),
)


def _velocity_contract(state: np.ndarray, previous: np.ndarray | None = None) -> dict:
    """Measure rest from a single raw frame or an adjacent confirmed pair."""
    groups = {}
    unsettled = []
    for name, _, qpos, qvel, (tolerance, unit) in _CONTROL_GROUPS:
        velocity = state[qvel]
        if previous is not None and qpos is not None:
            velocity = (state[qpos] - previous[qpos]) / CONTROL_DT_S
        peak = float(np.max(np.abs(velocity)))
        group = {"measured_max_abs": peak, "max_abs_tolerance": tolerance, "unit": unit}
        if previous is not None:
            group["source"] = (
                "public_base_qvel" if qpos is None else "adjacent_confirmed_public_qpos"
            )
        groups[name] = group
        if peak > tolerance:
            unsettled.append(name)
    return {
        "velocity_groups": groups,
        "stationary": not unsettled,
        "unsettled_groups": unsettled,
    }


def _proprio(value: Any) -> np.ndarray:
    return public_proprio(value).astype(np.float64, copy=False)


def _arm(value: str) -> Arm:
    if value not in ARM_JOINT_NAMES:
        raise ValueError("arm must be left or right")
    return value


def _numpy(value: Any) -> np.ndarray:
    """Copy a native CPU/GPU tensor without keeping simulator-owned storage."""
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.array(value, dtype=np.float64, copy=True)


def _fk_pose(position: Any, quaternion: Any) -> np.ndarray:
    """Pack native CuRobo position and wxyz rotation into xyz/xyzw."""
    return np.r_[
        _numpy(position).reshape(3), _numpy(quaternion).reshape(4)[[1, 2, 3, 0]]
    ]


# One reposition bound for every single-arm plan. The physical simulator still
# enforces collision, so this only keeps a single request local.
MAX_SINGLE_ARM_REPOSITION_M = 0.30
# FK agreement is a model-consistency check, not a motion limit.
FK_POSITION_TOLERANCE_M = 0.02
FK_ORIENTATION_TOLERANCE_RAD = 0.10
# How far toward the base an IK refusal suggests pulling the hand back.
RETREAT_SUGGESTION_M = 0.05


class R1ProMotionModel:
    """Borrow the current simulator robot and cache its native single-arm solvers.

    Args:
        get_robot: Owner-thread provider for the current loaded R1Pro. Reset may
            rebuild the scene, so solvers are discarded when robot identity changes.
            This model never sets simulator joint positions or advances physics.
    """

    def __init__(self, get_robot: Callable[[], Any]) -> None:
        if not callable(get_robot):
            raise TypeError("get_robot must provide the current loaded simulator robot")
        self._get_robot = get_robot
        self._robot: Any = None
        self._config: dict | None = None
        self._generators: dict[tuple[str, str], Any] = {}
        self._temp: tempfile.TemporaryDirectory | None = None

    @property
    def robot(self) -> Any:
        """Return the current native robot, invalidating caches after scene rebuild."""
        robot = self._get_robot()
        if robot is not self._robot:
            if robot.model != "r1pro":
                raise ValueError("single-arm motion requires the loaded R1Pro robot")
            self.close()
            self._robot = robot
        return robot

    def close(self) -> None:
        """Release owned solver references and temporary configs, not the robot."""
        self._generators.clear()
        if self._temp is not None:
            self._temp.cleanup()
            self._temp = None
        self._robot = None
        self._config = None

    @staticmethod
    def active_joint_names(arm: Arm) -> tuple[str, ...]:
        """Return the selected arm's native seven-joint order."""
        return ARM_JOINT_NAMES[_arm(arm)]

    @staticmethod
    def joint_positions(proprio: Any) -> dict[str, float]:
        """Map public positions to 22 physical joints, excluding velocities.

        Finger positions are measured meters, not normalized gripper commands.
        The public 61-vector does not expose wheel/steering or virtual base
        positions; they are not invented or inserted into this model.
        """
        state = _proprio(proprio)
        return {
            name: float(value)
            for _, names, section, _, _ in _CONTROL_GROUPS
            if section is not None
            for name, value in zip(names, state[section])
        }

    def joint_limits(self, arm: Arm) -> np.ndarray:
        """Return seven native [lower, upper] position limits in radians."""
        robot = self.robot
        return np.asarray(
            [
                [robot.joints[name].lower_limit, robot.joints[name].upper_limit]
                for name in self.active_joint_names(arm)
            ],
            dtype=np.float64,
        )

    def velocity_limits(self, arm: Arm) -> np.ndarray:
        """Return the selected arm's actual simulator velocity bounds in rad/s."""
        robot = self.robot
        limits = np.asarray(
            [robot.joints[name].max_velocity for name in self.active_joint_names(arm)],
            dtype=np.float64,
        )
        if not np.isfinite(limits).all() or np.any(limits <= 0):
            raise ValueError("native arm velocity limits must be finite and positive")
        return limits

    @staticmethod
    def static_state_contract(proprio: Any) -> dict[str, Any]:
        """Describe whether measured velocities support static single-arm planning.

        CuRobo receives a rest-state start and locks the remaining geometry.
        Both arms, the trunk and both finger pairs must therefore have settled;
        an immobile base alone is insufficient. These are adapter tolerances,
        not native joint velocity limits or proof that a grasp is stable.
        Command-to-opening agreement remains an execution-side requirement.
        """
        state = _proprio(proprio)
        return {
            "source": "current_public_proprio",
            **_velocity_contract(state),
            "held_finger_qpos": {
                "left": state[24:26].tolist(),
                "right": state[49:51].tolist(),
            },
        }

    @staticmethod
    def planning_state_contract(
        proprio: Any, motion_context: dict[str, Any] | None
    ) -> dict[str, Any]:
        """Validate the static CuRobo start using confirmed public control frames.

        Joint/finger velocities are displacement estimates over the official
        1/30 s action interval. They do not establish substep rest. The base has
        no public position pair, so its original public velocity gate remains.
        Raw joint velocities are retained for diagnostics and never rewritten.
        Only the very first frame may bootstrap from those raw velocities.
        """
        state = _proprio(proprio)
        raw = R1ProMotionModel.static_state_contract(state)
        invalid = {
            **raw,
            "source": "confirmed_public_control_observations",
            "valid": False,
            "stationary": False,
            "unsettled_groups": [],
            "raw_velocity_groups": raw["velocity_groups"],
            "code": "motion_history_unavailable",
            "error": "an adjacent confirmed public control-frame pair is required",
        }
        if not isinstance(motion_context, dict):
            return invalid
        context = motion_context

        token = context.get("current_token")
        previous_token = context.get("previous_token")
        counts = [
            context.get(name)
            for name in ("issued_actions", "confirmed_actions", "unconfirmed_actions")
        ]
        dt = context.get("control_dt_s")
        if (
            context.get("source") != "confirmed_public_control_observations"
            or not valid_observation_token(token)
            or any(type(value) is not int or value < 0 for value in counts)
            or counts[0] != counts[1]
            or counts[2] != 0
            or context.get("actions_pending") is not False
            or isinstance(dt, bool)
            or not isinstance(dt, (int, float))
            or not np.isclose(dt, CONTROL_DT_S, atol=1e-12, rtol=0)
        ):
            return invalid
        try:
            current = _proprio(context.get("current_proprio"))
        except (TypeError, ValueError):
            return invalid
        if not np.array_equal(current, state):
            return {
                **invalid,
                "error": "motion history does not match the public frame",
            }
        if (
            context.get("initial_bootstrap") is True
            and context.get("pair_confirmed") is False
            and token["observation_index"] == 0
            and previous_token is None
            and context.get("previous_proprio") is None
            and counts == [0, 0, 0]
        ):
            return {
                **raw,
                "source": "initial_public_velocity_bootstrap",
                "valid": True,
                "observation_token": token,
                "raw_velocity_groups": raw["velocity_groups"],
            }
        if (
            context.get("initial_bootstrap") is not False
            or context.get("pair_confirmed") is not True
            or not valid_observation_token(previous_token)
            or previous_token["generation"] != token["generation"]
            or previous_token["observation_index"] + 1 != token["observation_index"]
            or token["observation_index"] != counts[1]
        ):
            return invalid
        try:
            previous = _proprio(context.get("previous_proprio"))
        except (TypeError, ValueError):
            return invalid
        return {
            **raw,
            **_velocity_contract(current, previous),
            "source": "adjacent_confirmed_public_qpos",
            "valid": True,
            "raw_velocity_groups": raw["velocity_groups"],
            "control_dt_s": CONTROL_DT_S,
            "observation_token": token,
            "previous_observation_token": previous_token,
        }

    def validate_finger_geometry(self, proprio: Any) -> None:
        """Check measured geometry without deriving replacement gripper commands."""
        positions = self.joint_positions(proprio)
        robot = self.robot
        for arm, names in FINGER_JOINT_NAMES.items():
            for name in names:
                joint = robot.joints[name]
                if (
                    not joint.lower_limit - 0.001
                    <= positions[name]
                    <= joint.upper_limit + 0.001
                ):
                    raise ValueError(
                        f"{arm} measured fingers exceed native position limits"
                    )

    def _native_positions(self, proprio: Any) -> np.ndarray:
        robot = self.robot
        positions = _numpy(robot.get_joint_positions())
        if positions.shape != (len(robot.joints),) or not np.isfinite(positions).all():
            raise ValueError("native joint positions must be a finite ordered vector")
        measured = dict(zip(robot.joints, positions))
        if any(
            abs(measured[name] - value) > 1e-5
            for name, value in self.joint_positions(proprio).items()
        ):
            raise ValueError(
                "public motion snapshot differs from the current native joints"
            )
        return positions

    def _native_config(self) -> dict:
        """Read the immutable CuRobo template once for each loaded robot."""
        robot = self.robot
        if self._config is None:
            path = robot.curobo_path
            if isinstance(path, dict):
                path = path["arm"]
            self._config = yaml.safe_load(Path(path).read_text())["robot_cfg"]
        return self._config

    def robot_config(self, arm: Arm, proprio: Any) -> dict[str, Any]:
        """Narrow official USD configuration to one arm; retain all collision geometry.

        Base, trunk, opposite arm and both finger pairs are locked to real native
        positions. The SDK updates their fixed transforms before each solve.
        This configuration describes geometry and never changes hold commands.
        """
        robot = self.robot
        positions = dict(zip(robot.joints, self._native_positions(proprio)))
        config = deepcopy(self._native_config())
        kinematics = config["kinematics"]
        active = self.active_joint_names(arm)
        kinematics.update(
            use_usd_kinematics=True,
            ee_link=robot.eef_link_names[arm],
            link_names=[robot.eef_link_names["right" if arm == "left" else "left"]],
            lock_joints={
                name: float(value)
                for name, value in positions.items()
                if name not in active
            },
        )
        kinematics["cspace"].update(
            joint_names=list(active),
            retract_config=[float(positions[name]) for name in active],
            null_space_weight=[1.0] * 7,
            cspace_distance_weight=[1.0] * 7,
        )
        return config

    def ik_solver(self, arm: Arm, proprio: Any, device: str) -> Any:
        """Reuse official USD loading/locked-joint updates and its CuRobo IK solver."""
        import torch
        from curobo.types.state import JointState
        from omnigibson.action_primitives.curobo import (
            CuRoboEmbodimentSelection,
            CuRoboMotionGenerator,
        )

        robot = self.robot
        positions = self._native_positions(proprio)
        key = (_arm(arm), device)
        embodiment = CuRoboEmbodimentSelection.DEFAULT
        if key not in self._generators:
            if self._temp is None:
                self._temp = tempfile.TemporaryDirectory(prefix="rpent-behavior-")
            path = Path(self._temp.name) / f"{arm}.yaml"
            path.write_text(
                yaml.safe_dump({"robot_cfg": self.robot_config(arm, proprio)})
            )
            self._generators[key] = CuRoboMotionGenerator(
                robot,
                robot_cfg_path=str(path),
                robot_usd_path=robot.usd_path,
                device=device,
                batch_size=1,
                use_cuda_graph=False,
                use_default_embodiment_only=True,
                motion_cfg_kwargs={
                    "num_ik_seeds": 32,
                    "num_batch_ik_seeds": 32,
                    "position_threshold": FK_POSITION_TOLERANCE_M,
                    "rotation_threshold": FK_ORIENTATION_TOLERANCE_RAD,
                    "self_collision_check": True,
                },
            )
        generator = self._generators[key]
        solver = generator.mg[embodiment].ik_solver
        if tuple(solver.joint_names) != self.active_joint_names(arm):
            raise ValueError(
                "native IK configuration must optimize only the selected seven arm joints"
            )
        generator.update_locked_joints(
            JointState(
                position=torch.as_tensor(
                    positions[None], dtype=torch.float32, device=device
                ),
                joint_names=list(robot.joints),
            ),
            embodiment,
        )
        return solver

    def transform_pose(self, pose: Any, *, to_solver: bool) -> np.ndarray:
        """Convert between physical robot base and native CuRobo base, in xyz/xyzw."""
        from scipy.spatial.transform import Rotation

        robot = self.robot
        base = self._native_config()["kinematics"]["base_link"]
        physical = robot.get_position_orientation()
        native = robot.links[base].get_position_orientation()
        source, destination = (physical, native) if to_solver else (native, physical)
        src_pos, src_rot = _numpy(source[0]), Rotation.from_quat(_numpy(source[1]))
        dst_pos, dst_rot = (
            _numpy(destination[0]),
            Rotation.from_quat(_numpy(destination[1])),
        )
        pose = np.asarray(pose, dtype=np.float64)
        return np.concatenate(
            (
                dst_rot.inv().apply(src_rot.apply(pose[:3]) + src_pos - dst_pos),
                (dst_rot.inv() * src_rot * Rotation.from_quat(pose[3:])).as_quat(),
            )
        )

    def compare_proprio(self, proprio: Any, fk_state: Any) -> dict[str, Any]:
        """Compare both native-USD CuRobo FK results with the public snapshot."""
        from scipy.spatial.transform import Rotation

        state = _proprio(proprio)
        result: dict[str, Any] = {
            "frame": "robot_base",
            "source": "native_usd_curobo_fk",
        }
        for arm, start in (("left", 17), ("right", 42)):
            link = self.robot.eef_link_names[arm]
            native = fk_state.link_poses[link]
            pose = self.transform_pose(
                _fk_pose(native.position, native.quaternion),
                to_solver=False,
            )
            observed_q = state[start + 3 : start + 7]
            if abs(np.linalg.norm(observed_q) - 1.0) > 1e-3:
                raise ValueError(f"{arm} public EEF quaternion must have unit norm")
            result[arm] = {
                "position_error_m": float(
                    np.linalg.norm(pose[:3] - state[start : start + 3])
                ),
                "orientation_error_rad": float(
                    (
                        Rotation.from_quat(observed_q).inv()
                        * Rotation.from_quat(pose[3:])
                    ).magnitude()
                ),
                "predicted": {
                    "position": pose[:3].tolist(),
                    "quaternion_xyzw": pose[3:].tolist(),
                    "frame": "robot_base",
                    "eef_link": link,
                },
            }
        return result


class R1ProMotionPlanner:
    """Plan one seven-joint arm while locking the observed remaining joints.

    CuRobo imports and GPU allocation are deferred until the first request.
    The official action clock is 30 Hz; video encoding at 120 FPS is separate.
    Only short single-arm repositioning is supported by this adapter mode.
    The caller owns the native model cache; the simulator owns the robot.
    """

    def __init__(
        self,
        *,
        model: R1ProMotionModel,
        task_id: int,
        device: str = "cuda:0",
    ) -> None:
        self.task_id = task_id
        self.device = device
        self.model = model
        self.control_dt_s = CONTROL_DT_S

    def _ik_solver(self, arm: str, proprio: np.ndarray) -> Any:
        """Get native IK with the current base/trunk/opposite-arm/finger locks."""
        return self.model.ik_solver(arm, proprio, self.device)

    def plan_arm_path(
        self,
        *,
        arm: str,
        target_pose: Any,
        observation: dict[str, Any],
        motion_context: dict[str, Any] | None = None,
        allow_orientation_relax: bool = False,
    ) -> dict[str, Any]:
        """Return native joint waypoints and the exact state contract.

        With ``allow_orientation_relax`` a failed full-pose IK is retried with
        the position constraint only; the result then carries
        ``orientation_relaxed`` and the achieved orientation error.
        """
        if arm not in {"left", "right"}:
            raise ValueError("arm must be left or right")
        goal = np.asarray(target_pose, dtype=np.float64)
        if (
            goal.shape != (7,)
            or not np.isfinite(goal).all()
            or not np.isclose(np.linalg.norm(goal[3:]), 1, atol=1e-4)
        ):
            raise ValueError("target_pose must be robot-base xyz and unit xyzw")
        proprio = extract_public_proprio(observation, task_id=self.task_id)
        state = decode_robot_state(proprio)
        reposition_m = float(np.linalg.norm(goal[:3] - state[arm]["eef_position"]))
        if reposition_m > MAX_SINGLE_ARM_REPOSITION_M:
            return {
                "status": "Unavailable",
                "code": "outside_verified_motion_mode",
                "error": (
                    "single-arm reposition exceeds the per-call bound; "
                    "split it into successive targets"
                ),
                "measured_reposition_m": reposition_m,
                "allowed_reposition_m": MAX_SINGLE_ARM_REPOSITION_M,
            }
        motion_state = self.model.planning_state_contract(proprio, motion_context)
        if not motion_state["valid"]:
            return {
                "status": "Unavailable",
                "code": motion_state["code"],
                "error": motion_state["error"],
                "motion_state_contract": motion_state,
            }
        if not motion_state["stationary"]:
            # CuRobo plans from one snapshot; a moving start makes the path stale
            # immediately. The tool layer attempts to settle first, so this only
            # triggers when that bounded attempt did not reach the tolerance.
            return {
                "status": "Unavailable",
                "code": "base_not_stationary"
                if "base" in motion_state["unsettled_groups"]
                else "motion_state_unsettled",
                "error": "robot motion still exceeds the static arm-planning tolerance",
                "motion_state_contract": motion_state,
                "suggested_next": (
                    "allow motion to settle and retry; the tool already spends a "
                    "bounded number of hold actions trying to reach the tolerance"
                ),
            }
        try:
            self.model.validate_finger_geometry(proprio)
        except ValueError as error:
            return {
                "status": "Unavailable",
                "code": "invalid_finger_geometry",
                "error": str(error),
                "motion_state_contract": motion_state,
            }
        solver = self._ik_solver(arm, proprio)
        names = list(self.model.active_joint_names(arm))
        start_q = np.asarray(state[arm]["arm_qpos"], dtype=np.float64)
        native_fk = solver.fk(solver.tensor_args.to_device([start_q]))
        comparison = self.model.compare_proprio(proprio, native_fk)
        if any(
            item["position_error_m"] > FK_POSITION_TOLERANCE_M
            or item["orientation_error_rad"] > FK_ORIENTATION_TOLERANCE_RAD
            for item in (comparison["left"], comparison["right"])
        ):
            return {
                "status": "Unavailable",
                "code": "curobo_fk_mismatch",
                "fk_comparison": comparison,
            }
        native_pose_error = {
            key: comparison[arm][key]
            for key in ("position_error_m", "orientation_error_rad")
        }
        collision_scope = {
            "source": "none_ik_only",
            "occluded_space_checked": False,
            "held_objects_modeled": False,
            "note": (
                "no path-level collision check; the simulator's physics is the "
                "contact authority"
            ),
        }
        from curobo.types.math import Pose
        from scipy.spatial.transform import Rotation

        native_goal = self.model.transform_pose(goal, to_solver=True)
        target = Pose(
            position=solver.tensor_args.to_device([native_goal[:3]]),
            quaternion=solver.tensor_args.to_device([native_goal[[6, 3, 4, 5]]]),
        )
        # MotionGen warmup may leave goals for additional links in the cached
        # IK buffer. Passing None does not clear them. Explicitly hold each
        # modeled non-primary EEF at this snapshot's FK pose on every solve.
        held_link_poses = {
            name: Pose(
                position=pose.position.clone(), quaternion=pose.quaternion.clone()
            )
            for name, pose in native_fk.link_poses.items()
            if name != self.model.robot.eef_link_names[arm]
        }
        # The current configuration is passed as an IK seed. CuRobo also uses
        # its own seeds, so this does not guarantee a solution near the
        # present posture.
        seed = solver.tensor_args.to_device(
            np.asarray(start_q, dtype=np.float32).reshape(1, 1, 7)
        )
        from curobo.rollout.cost.pose_cost import PoseCostMetric

        orientation_relaxed = False
        try:
            solver.update_pose_cost_metric(PoseCostMetric.reset_metric())
            result = solver.solve_single(
                target, seed_config=seed, return_seeds=1, link_poses=held_link_poses
            )
            solved = bool(result.success.reshape(-1)[0].item())
            if not solved and allow_orientation_relax:
                solver.update_pose_cost_metric(
                    PoseCostMetric(
                        reach_partial_pose=True,
                        reach_vec_weight=solver.tensor_args.to_device(
                            [0, 0, 0, 1, 1, 1]
                        ),
                    )
                )
                relaxed = solver.solve_single(
                    target, seed_config=seed, return_seeds=1, link_poses=held_link_poses
                )
                if bool(relaxed.success.reshape(-1)[0].item()):
                    result = relaxed
                    solved = orientation_relaxed = True
        finally:
            # Solvers are cached; a retry must not relax any subsequent request.
            solver.update_pose_cost_metric(PoseCostMetric.reset_metric())
        goal_q = (
            result.js_solution.get_ordered_joint_state(names)
            .position.detach()
            .cpu()
            .numpy()
            .reshape(7)
        )
        position_error = float(result.position_error.reshape(-1)[0].item())
        if orientation_relaxed:
            relaxed_fk = solver.fk(solver.tensor_args.to_device([goal_q]))
            relaxed_wxyz = relaxed_fk.ee_quaternion.detach().cpu().numpy().reshape(4)
            rotation_error = float(
                (
                    Rotation.from_quat(relaxed_wxyz[[1, 2, 3, 0]]).inv()
                    * Rotation.from_quat(native_goal[3:])
                ).magnitude()
            )
        else:
            rotation_error = float(result.rotation_error.reshape(-1)[0].item())
        if not solved:
            # Report where the arm could actually get to, so the caller can
            # decide between a shorter arm move and a base move. Also suggest
            # a retreat point: the same height, pulled back toward the base,
            # which usually restores a solvable posture.
            best_fk = solver.fk(solver.tensor_args.to_device([goal_q]))
            best_pose = self.model.transform_pose(
                _fk_pose(best_fk.ee_position, best_fk.ee_quaternion),
                to_solver=False,
            )
            best_xyz = best_pose[:3]
            current_xyz = np.asarray(state[arm]["eef_position"], dtype=np.float64)
            planar = current_xyz[:2]
            norm = float(np.linalg.norm(planar))
            retreat = current_xyz.copy()
            if norm > 1e-6:
                retreat[:2] = planar - planar / norm * RETREAT_SUGGESTION_M
            return {
                "status": "Failed",
                "code": "curobo_ik_failed",
                "solver_status": "ik_unreachable",
                "position_error_m": position_error,
                "orientation_error_rad": rotation_error,
                "nearest_reachable_xyz": best_xyz.tolist(),
                "suggested_retreat_xyz": retreat.tolist(),
                "orientation_relax_attempted": allow_orientation_relax,
                "fk_comparison": comparison,
                "collision_scope": collision_scope,
            }
        dt = self.control_dt_s
        # Linear joint-space interpolation under the per-joint velocity limit;
        # at least one waypoint, never more than the native path cap.
        velocity = self.model.velocity_limits(arm)
        steps = int(np.ceil(np.max(np.abs(goal_q - start_q) / (velocity * dt))))
        if steps > 600:
            return {
                "status": "Unavailable",
                "code": "path_too_long",
                "error": "native velocity limits require more than 600 action steps",
            }
        steps = max(steps, 1)
        path = start_q + (goal_q - start_q) * (np.arange(1, steps + 1) / steps)[:, None]
        return {
            "status": "Success",
            "backend": "curobo_ik_joint_interpolation",
            "ik_position_error_m": position_error,
            "ik_orientation_error_rad": rotation_error,
            "orientation_relaxed": orientation_relaxed,
            "frame": "robot_base",
            "quaternion_order": "xyzw",
            "position": path.tolist(),
            "joint_names": names,
            "dt_s": dt,
            "control_dt_s": self.control_dt_s,
            "start_arm_qpos": state[arm]["arm_qpos"],
            "held_trunk_qpos": state["trunk_qpos"],
            "held_other_arm_qpos": state["right" if arm == "left" else "left"][
                "arm_qpos"
            ],
            "held_finger_qpos": motion_state["held_finger_qpos"],
            "motion_state_contract": motion_state,
            "joint_position_limits_rad": self.model.joint_limits(arm).tolist(),
            "joint_velocity_limits_rad_s": velocity.tolist(),
            "native_fk": native_pose_error,
            "fk_comparison": comparison,
            "collision_checked": False,
            "collision_scope": collision_scope,
            "target_pose": goal.tolist(),
            "held_objects_modeled": False,
        }
