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

"""R1Pro static model and native CuRobo single-arm inverse kinematics.

The solver uses bundled robot geometry only. It solves the goal joint
configuration with CuRobo IK, then interpolates the seven arm joints linearly
under the velocity limit. No path-level collision check is made against observed
surfaces; the simulator's physics is the contact authority, so a carried object
does not block a planned correction. CuRobo is imported only by the operations
that need it.
"""

from __future__ import annotations

import copy
import hashlib
import json
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Literal

import numpy as np

from robots.behavior.tools import extract_robot_state, snapshot_observation

Arm = Literal["left", "right"]
TRUNK_JOINT_NAMES = tuple(f"torso_joint{i}" for i in range(1, 5))
ARM_JOINT_NAMES = {
    arm: tuple(f"{arm}_arm_joint{i}" for i in range(1, 8)) for arm in ("left", "right")
}
FINGER_JOINT_NAMES = {
    arm: tuple(f"{arm}_gripper_finger_joint{i}" for i in range(1, 3))
    for arm in ("left", "right")
}
_JOINT_SLICES = (
    (TRUNK_JOINT_NAMES, slice(53, 57)),
    (ARM_JOINT_NAMES["left"], slice(3, 10)),
    (ARM_JOINT_NAMES["right"], slice(28, 35)),
    (FINGER_JOINT_NAMES["left"], slice(24, 26)),
    (FINGER_JOINT_NAMES["right"], slice(49, 51)),
)


def _proprio(value: Any) -> np.ndarray:
    array = np.asarray(value)
    if (
        array.shape != (61,)
        or array.dtype.kind not in "fiu"
        or not np.isfinite(array).all()
    ):
        raise ValueError("proprio must contain 61 finite real numbers")
    return array.astype(np.float64, copy=True)


def _arm(value: str) -> Arm:
    if value not in ARM_JOINT_NAMES:
        raise ValueError("arm must be left or right")
    return value


def _xyz(element: ET.Element | None, attribute: str) -> np.ndarray:
    text = "0 0 0" if element is None else element.get(attribute, "0 0 0")
    return np.asarray([float(x) for x in text.split()], dtype=np.float64)


# One reposition bound for every single-arm plan. The physical simulator still
# enforces collision, so this only keeps a single request local.
MAX_SINGLE_ARM_REPOSITION_M = 0.30
# FK agreement is a model-consistency check, not a motion limit.
FK_POSITION_TOLERANCE_M = 0.02
FK_ORIENTATION_TOLERANCE_RAD = 0.10
# How far toward the base an IK refusal suggests pulling the hand back.
RETREAT_SUGGESTION_M = 0.05


class R1ProMotionModel:
    """Load the reviewed static model without importing CuRobo or a simulator.

    Args:
        asset_dir: Directory containing the derived URDF, the CuRobo robot
            configuration (its self-collision spheres included) and provenance
            hashes. Defaults to bundled assets.
    """

    def __init__(self, asset_dir: str | Path | None = None) -> None:
        self.asset_dir = (
            Path(asset_dir)
            if asset_dir is not None
            else Path(__file__).parent / "assets" / "r1pro"
        ).resolve()
        self.provenance = json.loads((self.asset_dir / "provenance.json").read_text())
        for name in ("r1pro_base.urdf", "collision_config.json"):
            actual = hashlib.sha256((self.asset_dir / name).read_bytes()).hexdigest()
            if actual != self.provenance["derived_files"][name]:
                raise ValueError(f"R1Pro motion asset SHA256 mismatch: {name}")
        self.urdf_path = self.asset_dir / "r1pro_base.urdf"
        self.urdf_sha256 = self.provenance["derived_files"]["r1pro_base.urdf"]
        self._config = json.loads(
            (self.asset_dir / "collision_config.json").read_text()
        )
        root = ET.parse(self.urdf_path).getroot()
        self._joints = {joint.attrib["name"]: joint for joint in root.findall("joint")}
        self._by_child = {
            joint.find("child").attrib["link"]: joint for joint in self._joints.values()
        }
        self._links = tuple(link.attrib["name"] for link in root.findall("link"))

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
            for names, section in _JOINT_SLICES
            for name, value in zip(names, state[section])
        }

    def joint_limits(self, arm: Arm) -> np.ndarray:
        """Return seven [lower, upper] URDF position limits in radians."""
        return np.asarray(
            [
                [
                    float(self._joints[name].find("limit").attrib[key])
                    for key in ("lower", "upper")
                ]
                for name in self.active_joint_names(arm)
            ]
        )

    def velocity_limits(self, arm: Arm) -> np.ndarray:
        """Return the selected arm's sourced URDF velocity bounds in rad/s."""
        return np.asarray(
            [
                float(self._joints[name].find("limit").attrib["velocity"])
                for name in self.active_joint_names(arm)
            ]
        )

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
        sections = {
            "base": (slice(0, 3), 0.01, "metre_or_radian_per_second"),
            "trunk": (slice(57, 61), 0.02, "radian_per_second"),
            "left_arm": (slice(10, 17), 0.02, "radian_per_second"),
            "right_arm": (slice(35, 42), 0.02, "radian_per_second"),
            "left_fingers": (slice(26, 28), 0.002, "metre_per_second"),
            "right_fingers": (slice(51, 53), 0.002, "metre_per_second"),
        }
        groups = {
            name: {
                "measured_max_abs": float(np.max(np.abs(state[section]))),
                "max_abs_tolerance": tolerance,
                "unit": unit,
            }
            for name, (section, tolerance, unit) in sections.items()
        }
        unsettled = [
            name
            for name, group in groups.items()
            if group["measured_max_abs"] > group["max_abs_tolerance"]
        ]
        return {
            "source": "current_public_proprio",
            "stationary": not unsettled,
            "unsettled_groups": unsettled,
            "velocity_groups": groups,
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

        def valid_token(token: Any) -> bool:
            return (
                isinstance(token, dict)
                and set(token) == {"generation", "observation_index"}
                and all(type(value) is int and value >= 0 for value in token.values())
            )

        token = context.get("current_token")
        previous_token = context.get("previous_token")
        counts = [
            context.get(name)
            for name in ("issued_actions", "confirmed_actions", "unconfirmed_actions")
        ]
        dt = context.get("control_dt_s")
        if (
            context.get("source") != "confirmed_public_control_observations"
            or not valid_token(token)
            or any(type(value) is not int or value < 0 for value in counts)
            or counts[0] != counts[1]
            or counts[2] != 0
            or context.get("actions_pending") is not False
            or isinstance(dt, bool)
            or not isinstance(dt, (int, float))
            or not np.isclose(dt, 1 / 30, atol=1e-12, rtol=0)
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
            or not valid_token(previous_token)
            or previous_token["generation"] != token["generation"]
            or previous_token["observation_index"] + 1 != token["observation_index"]
            or token["observation_index"] != counts[1]
        ):
            return invalid
        try:
            previous = _proprio(context.get("previous_proprio"))
        except (TypeError, ValueError):
            return invalid
        position_sections = {
            "trunk": slice(53, 57),
            "left_arm": slice(3, 10),
            "right_arm": slice(28, 35),
            "left_fingers": slice(24, 26),
            "right_fingers": slice(49, 51),
        }
        groups = {
            "base": {**raw["velocity_groups"]["base"], "source": "public_base_qvel"}
        }
        for name, section in position_sections.items():
            velocity = (current[section] - previous[section]) / (1 / 30)
            groups[name] = {
                **raw["velocity_groups"][name],
                "measured_max_abs": float(np.max(np.abs(velocity))),
                "source": "adjacent_confirmed_public_qpos",
            }
        unsettled = [
            name
            for name, group in groups.items()
            if group["measured_max_abs"] > group["max_abs_tolerance"]
        ]
        return {
            **raw,
            "source": "adjacent_confirmed_public_qpos",
            "valid": True,
            "stationary": not unsettled,
            "unsettled_groups": unsettled,
            "velocity_groups": groups,
            "raw_velocity_groups": raw["velocity_groups"],
            "control_dt_s": 1 / 30,
            "observation_token": token,
            "previous_observation_token": previous_token,
        }

    def validate_finger_geometry(self, proprio: Any) -> None:
        """Check measured geometry without deriving replacement gripper commands."""
        positions = self.joint_positions(proprio)
        for arm, names in FINGER_JOINT_NAMES.items():
            for name in names:
                limits = self._joints[name].find("limit").attrib
                if (
                    not float(limits["lower"]) - 0.001
                    <= positions[name]
                    <= float(limits["upper"]) + 0.001
                ):
                    raise ValueError(
                        f"{arm} measured fingers exceed the URDF position limits"
                    )

    def robot_config(self, arm: Arm, proprio: Any) -> dict[str, Any]:
        """Build the inner dictionary accepted by ``RobotConfig.from_dict``.

        All four trunk joints, the opposite arm and both measured finger pairs
        are locked at this observation. Both arms keep their self-collision
        spheres so IK rejects configurations that intersect the robot itself.
        Callers must check the static-state contract and retain matching hold
        targets while executing; this configuration does not command the robot.
        """
        arm = _arm(arm)
        positions = self.joint_positions(proprio)
        active = self.active_joint_names(arm)
        config = copy.deepcopy(self._config)
        dynamic_limits = config.pop("motion_limits")
        config.update(
            urdf_path=str(self.urdf_path),
            asset_root_path=str(self.asset_dir),
            use_usd_kinematics=False,
            ee_link=f"{arm}_eef_link",
            link_names=[f"{'right' if arm == 'left' else 'left'}_eef_link"],
            lock_joints={
                name: value for name, value in positions.items() if name not in active
            },
            cspace={
                "joint_names": list(active),
                "retract_config": [positions[name] for name in active],
                "null_space_weight": [1.0] * 7,
                "cspace_distance_weight": [1.0] * 7,
                **dynamic_limits,
            },
        )
        return {"kinematics": config}

    def link_transforms(self, proprio: Any) -> dict[str, np.ndarray]:
        """Return static FK transforms from each modeled link to base_link."""
        from scipy.spatial.transform import Rotation

        positions = self.joint_positions(proprio)
        transforms = {"base_link": np.eye(4)}

        def transform(link: str) -> np.ndarray:
            if link in transforms:
                return transforms[link]
            joint = self._by_child[link]
            origin = joint.find("origin")
            local = np.eye(4)
            local[:3, :3] = Rotation.from_euler("xyz", _xyz(origin, "rpy")).as_matrix()
            local[:3, 3] = _xyz(origin, "xyz")
            kind = joint.attrib["type"]
            if kind != "fixed":
                axis = _xyz(joint.find("axis"), "xyz")
                motion = np.eye(4)
                q = positions[joint.attrib["name"]]
                if kind == "revolute":
                    motion[:3, :3] = Rotation.from_rotvec(axis * q).as_matrix()
                elif kind == "prismatic":
                    motion[:3, 3] = axis * q
                else:
                    raise ValueError(f"Unsupported joint type in derived model: {kind}")
                local = local @ motion
            result = transform(joint.find("parent").attrib["link"]) @ local
            transforms[link] = result
            return result

        for link in self._links:
            transform(link)
        return transforms

    def forward(self, arm: Arm, proprio: Any) -> dict[str, Any]:
        """Return the EEF pose in meters and xyzw order in the physical base."""
        from scipy.spatial.transform import Rotation

        arm = _arm(arm)
        matrix = self.link_transforms(proprio)[f"{arm}_eef_link"]
        return {
            "position": matrix[:3, 3].tolist(),
            "quaternion_xyzw": Rotation.from_matrix(matrix[:3, :3]).as_quat().tolist(),
            "frame": "robot_base",
            "eef_link": f"{arm}_eef_link",
        }

    def compare_proprio(self, proprio: Any) -> dict[str, Any]:
        """Compare independent URDF FK with public EEF positions/orientations."""
        from scipy.spatial.transform import Rotation

        state = _proprio(proprio)
        result: dict[str, Any] = {
            "frame": "robot_base",
            "urdf_sha256": self.urdf_sha256,
        }
        for arm, start in (("left", 17), ("right", 42)):
            pose = self.forward(arm, state)
            observed_q = state[start + 3 : start + 7]
            norm = np.linalg.norm(observed_q)
            if abs(norm - 1.0) > 1e-3:
                raise ValueError(f"{arm} public EEF quaternion must have unit norm")
            observed = Rotation.from_quat(observed_q)
            predicted = Rotation.from_quat(pose["quaternion_xyzw"])
            result[arm] = {
                "position_error_m": float(
                    np.linalg.norm(
                        np.asarray(pose["position"]) - state[start : start + 3]
                    )
                ),
                "orientation_error_rad": float(
                    (observed.inv() * predicted).magnitude()
                ),
                "predicted": pose,
            }
        return result


class R1ProMotionPlanner:
    """Plan one seven-joint arm while locking the observed remaining joints.

    CuRobo imports and GPU allocation are deferred until the first request.
    The official action clock is 30 Hz; video encoding at 120 FPS is separate.
    Only short single-arm repositioning is supported by this adapter mode.
    """

    def __init__(
        self, *, task_id: int, task_language: str, device: str = "cuda:0"
    ) -> None:
        self.task_id = task_id
        self.task_language = task_language
        self.device = device
        self.model = R1ProMotionModel()
        self.control_dt_s = 1.0 / 30.0

    def _ik_solver(self, arm: str, proprio: np.ndarray) -> Any:
        """Build a solver for this observation's locked joints.

        The locked trunk, opposite arm and finger positions change with every
        request, and CuRobo's IK solver cannot update them in place, so each
        request gets its own solver.
        """
        from curobo.types.base import TensorDeviceType
        from curobo.types.robot import RobotConfig
        from curobo.wrap.reacher.ik_solver import IKSolver, IKSolverConfig

        args = TensorDeviceType(device=self.device)
        robot = RobotConfig.from_dict(self.model.robot_config(arm, proprio), args)
        return IKSolver(
            IKSolverConfig.load_from_robot_config(
                robot_cfg=robot,
                world_model=None,
                tensor_args=args,
                num_seeds=32,
                use_cuda_graph=False,
                self_collision_check=True,
                position_threshold=FK_POSITION_TOLERANCE_M,
                rotation_threshold=FK_ORIENTATION_TOLERANCE_RAD,
            )
        )

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
        snapshot = snapshot_observation(
            observation, task_id=self.task_id, task_language=self.task_language
        )
        proprio = np.asarray(snapshot["proprio"])
        state = extract_robot_state(proprio)
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
        comparison = self.model.compare_proprio(proprio)
        if any(
            item["position_error_m"] > FK_POSITION_TOLERANCE_M
            or item["orientation_error_rad"] > FK_ORIENTATION_TOLERANCE_RAD
            for item in (comparison["left"], comparison["right"])
        ):
            return {
                "status": "Unavailable",
                "code": "robot_fk_mismatch",
                "fk_comparison": comparison,
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

        solver = self._ik_solver(arm, proprio)
        names = list(self.model.active_joint_names(arm))
        start_q = np.asarray(state[arm]["arm_qpos"], dtype=np.float64)
        native_fk = solver.fk(solver.tensor_args.to_device([start_q]))
        native_xyz = native_fk.ee_position.detach().cpu().numpy().reshape(3)
        native_wxyz = native_fk.ee_quaternion.detach().cpu().numpy().reshape(4)
        native_pose_error = {
            "position_error_m": float(
                np.linalg.norm(native_xyz - state[arm]["eef_position"])
            ),
            "orientation_error_rad": float(
                (
                    Rotation.from_quat(native_wxyz[[1, 2, 3, 0]]).inv()
                    * Rotation.from_quat(state[arm]["eef_quaternion_xyzw"])
                ).magnitude()
            ),
        }
        if (
            native_pose_error["position_error_m"] > FK_POSITION_TOLERANCE_M
            or native_pose_error["orientation_error_rad"] > FK_ORIENTATION_TOLERANCE_RAD
        ):
            return {
                "status": "Unavailable",
                "code": "curobo_fk_mismatch",
                "native_fk": native_pose_error,
                "fk_comparison": comparison,
            }
        target = Pose(
            position=solver.tensor_args.to_device([goal[:3]]),
            quaternion=solver.tensor_args.to_device([goal[[6, 3, 4, 5]]]),
        )
        # The current configuration is passed as an IK seed. CuRobo also uses
        # its own seeds, so this does not guarantee a solution near the
        # present posture.
        seed = solver.tensor_args.to_device(
            np.asarray(start_q, dtype=np.float32).reshape(1, 1, 7)
        )
        result = solver.solve_single(target, seed_config=seed, return_seeds=1)
        goal_q = (
            result.js_solution.get_ordered_joint_state(names)
            .position.detach()
            .cpu()
            .numpy()
            .reshape(7)
        )
        position_error = float(result.position_error.reshape(-1)[0].item())
        rotation_error = float(result.rotation_error.reshape(-1)[0].item())
        orientation_relaxed = False
        if not bool(result.success.reshape(-1)[0].item()) and allow_orientation_relax:
            # Near full extension the arm often cannot keep the requested
            # orientation for a small position change. Retry with the position
            # constraint only and report how far the wrist will turn; the
            # caller decides whether that is acceptable.
            from curobo.rollout.cost.pose_cost import PoseCostMetric

            solver.update_pose_cost_metric(
                PoseCostMetric(
                    reach_partial_pose=True,
                    reach_vec_weight=solver.tensor_args.to_device([0, 0, 0, 1, 1, 1]),
                )
            )
            relaxed = solver.solve_single(target, seed_config=seed, return_seeds=1)
            if bool(relaxed.success.reshape(-1)[0].item()):
                result = relaxed
                goal_q = (
                    relaxed.js_solution.get_ordered_joint_state(names)
                    .position.detach()
                    .cpu()
                    .numpy()
                    .reshape(7)
                )
                position_error = float(relaxed.position_error.reshape(-1)[0].item())
                relaxed_fk = solver.fk(solver.tensor_args.to_device([goal_q]))
                relaxed_wxyz = (
                    relaxed_fk.ee_quaternion.detach().cpu().numpy().reshape(4)
                )
                rotation_error = float(
                    (
                        Rotation.from_quat(relaxed_wxyz[[1, 2, 3, 0]]).inv()
                        * Rotation.from_quat(goal[3:])
                    ).magnitude()
                )
                orientation_relaxed = True
        if not bool(result.success.reshape(-1)[0].item()):
            # Report where the arm could actually get to, so the caller can
            # decide between a shorter arm move and a base move. Also suggest
            # a retreat point: the same height, pulled back toward the base,
            # which usually restores a solvable posture.
            best_fk = solver.fk(solver.tensor_args.to_device([goal_q]))
            best_xyz = best_fk.ee_position.detach().cpu().numpy().reshape(3)
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
        steps = int(np.clip(steps, 1, 600))
        path = start_q + (goal_q - start_q) * (np.arange(1, steps + 1) / steps)[:, None]
        if not 1 <= len(path) <= 600:
            return {
                "status": "Failed",
                "code": "motion_path_length",
                "path_points": len(path),
            }
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
            "joint_velocity_limits_rad_s": self.model.velocity_limits(arm).tolist(),
            "native_fk": native_pose_error,
            "fk_comparison": comparison,
            "collision_checked": False,
            "collision_scope": collision_scope,
            "target_pose": goal.tolist(),
            "held_objects_modeled": False,
        }
