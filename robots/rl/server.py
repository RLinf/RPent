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

"""Start a clean-workflow simulator using the existing RPent facades."""

from __future__ import annotations

import argparse
import os


def planner_command_pose(planner, actual_base_pose, target_pose):
    """Map a world goal through the native planner's nominal base convention.

    The native planner uses the articulation origin and a translation bias;
    cuRobo FK starts at its configured kinematic base link. Fixed mounting
    rotations between these frames must be included for a world-pose command.
    """
    import numpy as np

    pose_type = type(actual_base_pose)
    nominal_base = planner.robot_origion_pose * pose_type(
        -np.asarray(planner.frame_bias)
    )
    goal = (
        nominal_base
        * actual_base_pose.inv()
        * pose_type(target_pose[:3], target_pose[3:])
    )
    return [*goal.p, *goal.q]


def _release_unused_batch_planners(env):
    """Keep the single-pose planners used by this workflow and release warmup caches."""
    import gc

    import torch

    sub_env = env._sub_env(0)
    with sub_env.lock:
        robot = getattr(sub_env.task, "robot", None)
        if robot is not None:
            for arm in ("left", "right"):
                planner = getattr(robot, f"{arm}_planner")
                planner.motion_gen_batch = None
    gc.collect()
    torch.cuda.empty_cache()


def main() -> None:
    """Run one isolated simulator in its own Python environment."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env", choices=("robocasa", "robotwin"), required=True)
    parser.add_argument("--task-name", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--split", default="target")
    parser.add_argument("--task-config", default="demo_randomized")
    parser.add_argument("--assets-path")
    parser.add_argument("--camera-h", type=int, default=256)
    parser.add_argument("--camera-w", type=int, default=256)
    parser.add_argument("--max-episode-steps", type=int, default=10000)
    parser.add_argument("--cuda-device", type=int)
    parser.add_argument("--transport", choices=("socket", "http"), default="http")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=0)
    parser.add_argument("--parent-watch", action="store_true")
    args = parser.parse_args()
    if args.env == "robocasa":
        if args.cuda_device is not None:
            from rpent.utils.egl import configure_egl_device

            os.environ.pop("CUDA_VISIBLE_DEVICES", None)
            configure_egl_device(args.cuda_device)
        from robots.robocasa.env_server import RoboCasaEnvFacade

        class CleanRoboCasaFacade(RoboCasaEnvFacade):
            def _register_rpc(self):
                super()._register_rpc()
                self._rpc["env.control_meta"] = self.control_meta
                self._readonly_methods.add("env.control_meta")
                self._rpc["env.control_frame"] = self.control_frame
                self._readonly_methods.add("env.control_frame")

            def control_meta(self):
                controller = self.env.robots[0].composite_controller
                return {
                    "gripper_max_gap": float(
                        sum(
                            max(abs(bounds))
                            for bounds in self.env.sim.model.jnt_range[
                                [
                                    self.env.sim.model.joint_name2id(name)
                                    for name in self.env.robots[0]
                                    .gripper["right"]
                                    .joints
                                ]
                            ]
                        )
                    ),
                    "parts": {
                        name: {
                            "input_min": part.input_min,
                            "input_max": part.input_max,
                            "output_min": part.output_min,
                            "output_max": part.output_max,
                            "input_ref_frame": getattr(part, "input_ref_frame", "base"),
                            "actuator_min": part.actuator_min,
                            "actuator_max": part.actuator_max,
                        }
                        for name, part in controller.part_controllers.items()
                    },
                    "slices": controller._action_split_indexes,
                    "control_freq": self.env.control_freq,
                }

            def control_frame(self):
                """Read robot-only controller origins used by native delta control."""
                import numpy as np

                controller = self.env.robots[0].composite_controller
                _, arm_rotation = controller.get_controller_base_pose("right")
                base = controller.part_controllers["base"]
                _, rotation = base.get_base_pose()
                theta = np.arctan2(rotation[1, 0], rotation[0, 0]) - np.arctan2(
                    base.init_ori[1, 0], base.init_ori[0, 0]
                )
                return {"arm_origin_rotation": arm_rotation, "base_theta": theta}

        facade = CleanRoboCasaFacade(
            args.task_name,
            split=args.split,
            seed=args.seed,
            camera_h=args.camera_h,
            camera_w=args.camera_w,
        )
    else:
        if args.cuda_device is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(args.cuda_device)
        import torch

        torch.cuda.set_device(0)
        from robots.robotwin.env_server import RoboTwinEnvFacade, make_env
        from robots.robotwin.robot_spec import env_runtime_contract

        if not args.assets_path:
            parser.error("--assets-path is required for RoboTwin")
        env = make_env(
            args.task_name,
            args.task_config,
            args.seed,
            args.assets_path,
            args.max_episode_steps,
        )
        _release_unused_batch_planners(env)

        class CleanRoboTwinFacade(RoboTwinEnvFacade):
            def reset(self):
                result = super().reset()
                _release_unused_batch_planners(self._env)
                self._finger_pad_sites = None
                return result

            def plan_arm_path(self, arm, target_pose):
                sub_env = self._env._sub_env(0)
                with sub_env.lock:
                    robot = sub_env.task.robot
                    planner = getattr(robot, f"{arm}_planner")
                    name = planner.robot_config["robot_cfg"]["kinematics"]["base_link"]
                    link = next(
                        link
                        for link in getattr(robot, f"{arm}_entity").get_links()
                        if link.get_name() == name
                    )
                    target_pose = planner_command_pose(
                        planner, link.get_pose(), target_pose
                    )
                return super().plan_arm_path(arm, target_pose)

            def _register_rpc(self):
                super()._register_rpc()
                self._rpc["env.measured_grippers"] = self.measured_grippers
                self._readonly_methods.add("env.measured_grippers")
                self._rpc["env.coordinate_calibration"] = self.coordinate_calibration
                self._readonly_methods.add("env.coordinate_calibration")

            def coordinate_calibration(self):
                """Measure both Aloha mounting bases and finger geometry."""
                from robots.robotwin.general import robotwin_calibration

                sub_env = self._env._sub_env(0)
                with sub_env.lock:
                    calibration = robotwin_calibration(sub_env.task)
                    return calibration

            def measured_grippers(self):
                """Read physical joint positions; native gripper getters return targets."""
                import numpy as np

                sub_env = self._env._sub_env(0)
                with sub_env.lock:
                    robot = sub_env.task.robot
                    result = {}
                    for side in ("left", "right"):
                        entity = getattr(robot, f"{side}_entity")
                        joint, multiplier, offset = getattr(robot, f"{side}_gripper")[0]
                        index = entity.get_active_joints().index(joint)
                        position = float(entity.get_qpos()[index])
                        low, high = getattr(robot, f"{side}_gripper_scale")
                        opening = ((position - offset) / multiplier - low) / (
                            high - low
                        )
                        result[side] = {
                            "opening": float(np.clip(opening, 0, 1)),
                            "joint_position": position,
                            "commanded_opening": float(
                                getattr(robot, f"{side}_gripper_val")
                            ),
                        }
                    return result

        facade = CleanRoboTwinFacade(
            env,
            metadata=env_runtime_contract(
                task_name=args.task_name,
                task_config=args.task_config,
                seed=args.seed,
                max_episode_steps=args.max_episode_steps,
            ),
        )
    facade.serve(
        transport=args.transport,
        host=args.host,
        port=args.port,
        parent_watch=args.parent_watch,
    )


if __name__ == "__main__":
    main()
