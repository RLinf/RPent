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

"""Read kinematics in the existing RLinf simulation actor's owning thread.

Ray's built-in actor call executes this importable function serially alongside
reset/step. No simulator object crosses the process boundary. Actions continue
through OfficialBehaviorBackend.chunk_step, including official-success checks.
"""

from __future__ import annotations

import numpy as np


def get_camera_observation(process, env_index: int = 0) -> dict:
    """Capture synchronized RGB, optical-axis depth and calibration without stepping."""
    import omnigibson as og

    from robots.behavior.rlinf_env import _torch_to_numpy

    env = process.env.envs[env_index]
    robot = env.robots[0]
    cameras = {}
    for camera, link in (
        ("head", "zed_link"),
        ("left_wrist", "left_realsense_link"),
        ("right_wrist", "right_realsense_link"),
    ):
        sensors = [
            sensor for name, sensor in robot.sensors.items() if f":{link}:" in name
        ]
        if len(sensors) != 1:
            raise ValueError(f"expected one {camera} sensor, found {len(sensors)}")
        sensor = sensors[0]
        sensor.add_modality("depth_linear")
        # Initialize the camera-parameter annotator before flushing render latency.
        sensor.camera_parameters
        cameras[camera] = sensor
    env.load_observation_space()
    for _ in range(3):
        og.sim.render()
    result = {}
    for camera, sensor in cameras.items():
        obs, _ = sensor.get_obs()
        params = sensor.camera_parameters
        view = (
            np.asarray(_torch_to_numpy(params["cameraViewTransform"])).reshape(4, 4).T
        )
        result[camera] = {
            "rgb": np.asarray(_torch_to_numpy(obs["rgb"]))[..., :3],
            "depth": np.asarray(_torch_to_numpy(obs["depth_linear"])).squeeze(),
            "intrinsic": np.asarray(_torch_to_numpy(sensor.intrinsic_matrix)),
            "camera_to_world": np.linalg.inv(view),
        }
    return result


def get_planning_state(process, env_index: int = 0) -> dict:
    """Read the actor's current articulation and conservative collision geometry."""
    from pathlib import Path

    from robots.behavior.rlinf_env import _torch_to_numpy

    env = process.env.envs[env_index]
    robot = env.robots[0]
    result = get_motion_state(process, env_index)
    q = np.asarray(_torch_to_numpy(robot.get_joint_positions()))
    result["joint_positions"] = dict(zip(robot.joints, q.tolist()))
    result["urdf_path"] = robot.urdf_path
    model_dir = Path(robot.urdf_path).parent.parent
    result["collision_config_path"] = str(
        model_dir / "curobo" / "r1pro_description_curobo_arm_no_torso.yaml"
    )
    pos, quat = robot.links["base_link"].get_position_orientation()
    result["base_position"] = np.asarray(_torch_to_numpy(pos))
    result["base_quaternion_xyzw"] = np.asarray(_torch_to_numpy(quat))
    result["robot_aabb"] = [np.asarray(_torch_to_numpy(x)) for x in robot.aabb]
    result["obstacles"] = {}
    for obj in env.scene.objects:
        if obj is robot:
            continue
        for name, link in obj.links.items():
            low, high = (np.asarray(_torch_to_numpy(x)) for x in link.aabb)
            if np.all(high > low):
                result["obstacles"][f"{obj.name}_{name}"] = {"low": low, "high": high}
    return result


def navigation_collision(state: dict, goal: np.ndarray) -> str | None:
    """Conservative swept footprint for a short holonomic base segment."""
    low, high = map(np.asarray, state["robot_aabb"])
    position = np.asarray(state["base_position"])
    radius = float(
        np.linalg.norm(np.maximum(high[:2] - position[:2], position[:2] - low[:2]))
    )
    for fraction in np.linspace(
        0, 1, max(2, int(np.linalg.norm(goal[:2] - position[:2]) / 0.01) + 1)
    ):
        point = position[:2] + fraction * (goal[:2] - position[:2])
        for name, obstacle in state["obstacles"].items():
            a, b = np.asarray(obstacle["low"]), np.asarray(obstacle["high"])
            if b[2] <= low[2] or a[2] >= high[2]:
                continue
            nearest = np.clip(point, a[:2], b[:2])
            if np.linalg.norm(nearest - point) <= radius:
                return name
    return None


def build_robot_config(state: dict, hands=("left", "right")) -> dict:
    """Adapt the installed R1Pro collision model to cuRobo 0.8's URDF loader."""
    import xml.etree.ElementTree as ET

    import yaml
    from scipy.spatial.transform import Rotation

    with open(state["collision_config_path"]) as source:
        original = yaml.safe_load(source)["robot_cfg"]["kinematics"]
    xml_joints = ET.parse(state["urdf_path"]).getroot().findall("joint")
    joints = {j.attrib["name"] for j in xml_joints if j.attrib["type"] != "fixed"}
    active = [f"{hand}_arm_joint{i}" for hand in hands for i in range(1, 8)]
    # R1Pro's holonomic-base import fixes the six URDF wheel/steering joints
    # at their zero transform; they are not DOFs in the live articulation.
    fixed_wheels = {
        f"{kind}_motor_joint{i}" for kind in ("wheel", "steer") for i in range(1, 4)
    }
    locked = {
        name: state["joint_positions"][name]
        for name in joints - set(active) - fixed_wheels
    }
    kinematics = {
        name: original[name]
        for name in (
            "collision_link_names",
            "collision_spheres",
            "collision_sphere_buffer",
            "self_collision_buffer",
            "self_collision_ignore",
            "extra_links",
            "extra_collision_spheres",
        )
    }
    # cuRobo 0.8's loader computes padding compensation in a local variable
    # but passes the uncompensated dict to SelfCollisionKinematicsCfg. Restore
    # its intended semantics here: external collision padding remains intact;
    # self-collision uses the original spheres plus the explicit self margins.
    buffer = original["collision_sphere_buffer"]
    kinematics["self_collision_buffer"] = {
        name: original["self_collision_buffer"].get(name, 0.0)
        - (buffer if isinstance(buffer, (float, int)) else buffer.get(name, 0.0))
        for name in original["collision_link_names"]
    }
    for joint in xml_joints:
        if joint.attrib["name"] not in fixed_wheels:
            continue
        name = joint.find("child").attrib["link"]
        origin = joint.find("origin")
        xyz = [float(x) for x in origin.attrib.get("xyz", "0 0 0").split()]
        rpy = [float(x) for x in origin.attrib.get("rpy", "0 0 0").split()]
        quat = Rotation.from_euler("xyz", rpy).as_quat()[[3, 0, 1, 2]].tolist()
        kinematics["extra_links"][name] = {
            "link_name": name,
            "parent_link_name": joint.find("parent").attrib["link"],
            "joint_name": joint.attrib["name"],
            "joint_type": "FIXED",
            "fixed_transform": xyz + quat,
        }
    for hand in ("left", "right"):
        name = f"{hand}_eef_link"
        kinematics["extra_links"][name] = {
            "link_name": name,
            "parent_link_name": f"{hand}_gripper_link",
            "joint_name": name + "_joint",
            "joint_type": "FIXED",
            # r1pro_source_cfg.yaml: xyzw [0,1,0,0], translated to wxyz here.
            "fixed_transform": [0, 0, -0.06, 0, 0, 1, 0],
        }
    kinematics.update(
        base_link="base_link",
        urdf_path=state["urdf_path"],
        tool_frames=[f"{hand}_eef_link" for hand in hands],
        lock_joints=locked,
        cspace={
            "joint_names": active,
            "default_joint_position": [state["joint_positions"][n] for n in active],
            "cspace_distance_weight": [
                original["cspace"]["cspace_distance_weight"][
                    original["cspace"]["joint_names"].index(n)
                ]
                for n in active
            ],
            "null_space_weight": [
                original["cspace"]["null_space_weight"][
                    original["cspace"]["joint_names"].index(n)
                ]
                for n in active
            ],
            "max_acceleration": 5.0,
            "max_jerk": 100.0,
            "velocity_scale": 0.25,
        },
    )
    return {"kinematics": kinematics}


class BehaviorMotionPlanner:
    """cuRobo 0.8 arm planner, constructed lazily on the ENV facade thread.

    The live torso, unselected arm and grippers are locked. The world uses
    conservative link AABBs; a blocked path is a failure, never a teleport.
    """

    def __init__(self):
        self._planner = None
        self._lock_key = None

    def close(self):
        if self._planner is not None:
            self._planner.destroy()
            self._planner = None
        self._lock_key = None

    def plan(self, state: dict, targets: dict) -> dict:
        import torch
        from curobo.motion_planner import MotionPlanner, MotionPlannerCfg
        from curobo.scene import Scene
        from curobo.types import GoalToolPose, JointState, Pose
        from scipy.spatial.transform import Rotation

        config = build_robot_config(state, tuple(targets))
        kinematics = config["kinematics"]
        locks = kinematics["lock_joints"]
        key = (
            tuple(targets),
            tuple((n, round(v, 3)) for n, v in sorted(locks.items())),
        )
        base_rotation = Rotation.from_quat(state["base_quaternion_xyzw"])
        inverse = base_rotation.inv()
        orientation = inverse.as_quat()[[3, 0, 1, 2]].tolist()
        cuboids = {}
        for name, obstacle in state["obstacles"].items():
            low, high = np.asarray(obstacle["low"]), np.asarray(obstacle["high"])
            center = inverse.apply((low + high) / 2 - state["base_position"])
            cuboids[name] = {
                "dims": (high - low).tolist(),
                "pose": center.tolist() + orientation,
            }
        scene = {"cuboid": cuboids}
        if self._planner is None or self._lock_key != key:
            self.close()
            self._planner = MotionPlanner(
                MotionPlannerCfg.create(
                    robot={"robot_cfg": config},
                    scene_model=scene,
                    collision_cache={"cuboid": max(len(cuboids), 1)},
                    self_collision_check=True,
                    max_batch_size=1,
                    max_goalset=1,
                )
            )
            self._lock_key = key
        else:
            self._planner.update_world(Scene.create(scene))
        planner = self._planner
        q = torch.tensor(
            [[state["joint_positions"][n] for n in planner.joint_names]],
            device="cuda",
            dtype=torch.float32,
        )
        start = JointState.from_position(q, joint_names=planner.joint_names)
        fk = planner.compute_kinematics(start)
        goals = {}
        for hand, target in targets.items():
            name = f"{hand}_eef_link"
            actual = state["hands"][hand]
            predicted = fk.tool_poses.get_link_pose(name)
            predicted_world = (
                base_rotation.apply(
                    predicted.position.detach().cpu().numpy().reshape(3)
                )
                + state["base_position"]
            )
            if np.linalg.norm(predicted_world - actual["position"]) > 0.005:
                raise ValueError(f"{hand}: URDF/live kinematics mismatch")
            relative = inverse.apply(
                np.asarray(target["position"]) - state["base_position"]
            )
            quat = (inverse * Rotation.from_quat(target["quaternion_xyzw"])).as_quat()[
                [3, 0, 1, 2]
            ]
            goals[name] = Pose(
                torch.tensor(relative[None], device="cuda", dtype=torch.float32),
                torch.tensor(quat[None], device="cuda", dtype=torch.float32),
            )
        goal = GoalToolPose.from_poses(goals, ordered_tool_frames=planner.tool_frames)
        result = planner.plan_pose(goal, start, max_attempts=1)
        if result is None or not bool(result.success.all()):
            return {
                "success": False,
                "stop_reason": "planning_failed",
                "details": str(getattr(result, "status", "no trajectory")),
            }
        trajectory = result.get_interpolated_plan().reorder(planner.joint_names)
        positions = (
            trajectory.position.detach()
            .cpu()
            .numpy()
            .reshape(-1, len(planner.joint_names))
        )
        dt = float(planner.trajopt_solver.config.interpolation_dt)
        return {
            "success": True,
            "positions": positions,
            "joint_names": planner.joint_names,
            "dt": dt,
        }


def get_motion_state(process, env_index: int = 0) -> dict:
    import omnigibson as og
    from omnigibson.utils.transform_utils import quat2mat

    from robots.behavior.rlinf_env import _torch_to_numpy

    env = process.env.envs[env_index]
    robot = env.robots[0]
    controls = robot.get_control_dict()
    joint_positions = robot.get_joint_positions()
    _, base_quat = robot.get_position_orientation()

    def array(value):
        return np.asarray(_torch_to_numpy(value)).copy()

    rotation = array(quat2mat(base_quat))
    result = {"control_dt": float(og.sim.get_sim_step_dt()), "hands": {}}
    for hand in ("left", "right"):
        controller = robot.controllers[f"arm_{hand}"]
        if controller.motor_type != "position" or controller.use_delta_commands:
            raise RuntimeError("motion requires absolute position arm controllers")
        idx = robot.arm_control_idx[hand]
        pos, quat = robot.get_eef_pose(hand)
        jacobian = array(controls[f"eef_{hand}_jacobian_relative"][:, idx])
        world_jacobian = np.concatenate(
            (rotation @ jacobian[:3], rotation @ jacobian[3:]), axis=0
        )
        palm_pos, _ = robot.links[f"{hand}_gripper_link"].get_position_orientation()
        direction = array(pos - palm_pos)
        length = float(np.linalg.norm(direction))
        if length < 1e-6:
            raise RuntimeError("EEF and gripper origins do not define an approach axis")
        contacts = sorted(
            {
                str(body)
                for link in robot.finger_links[hand]
                for contact in link.contact_list()
                for body in (contact.body0, contact.body1)
                if not str(body).startswith(robot.prim_path + "/")
            }
        )
        result["hands"][hand] = {
            "position": array(pos),
            "quaternion_xyzw": array(quat),
            "joint_positions": array(joint_positions[idx]),
            "joint_lower_limits": array(robot.joint_lower_limits[idx]),
            "joint_upper_limits": array(robot.joint_upper_limits[idx]),
            "jacobian": world_jacobian,
            "approach_direction": direction / length,
            "contacts": contacts,
        }
    return result
