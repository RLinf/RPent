# Copyright 2026 The RPent Authors.
# Licensed under the Apache License, Version 2.0.
"""Read native LIBERO grip-site poses without changing RLinf policy observations."""

from __future__ import annotations

import numpy as np


class ControlObservation:
    """Transparent native-env wrapper exposing a picklable robot-state property."""

    def __init__(self, env):
        self.env = env

    def __getattr__(self, name):
        return getattr(self.env, name)

    @property
    def rpent_control_state(self) -> dict:
        """Read MuJoCo's site frame and the active OSC controller configuration."""
        native = self.env
        while hasattr(native, "env"):
            native = native.env
        robot = native.robots[0]
        arm = robot.arms[0]
        controller = robot.part_controllers[arm]
        if controller.name != "OSC_POSE" or controller.input_type != "delta":
            raise ValueError("LIBERO RL requires the native delta OSC_POSE controller")
        site = robot.eef_site_id[arm]
        body = native.sim.model.body_name2id(robot.robot_model.eef_name[arm])
        site_rotation = native.sim.data.site_xmat[site].reshape(3, 3).copy()
        body_rotation = native.sim.data.body_xmat[body].reshape(3, 3).copy()
        origin = (
            np.eye(3)
            if controller.input_ref_frame == "world"
            else np.asarray(controller.origin_ori)
        )
        if origin.shape != (3, 3):
            raise ValueError("native OSC origin rotation is unavailable")
        jaw_axes = []
        for name in robot.gripper[arm].joints:
            joint = native.sim.model.joint_name2id(name)
            joint_body = native.sim.model.jnt_bodyid[joint]
            axis_world = (
                native.sim.data.body_xmat[joint_body].reshape(3, 3)
                @ native.sim.model.jnt_axis[joint]
            )
            jaw_axes.append(site_rotation.T @ axis_world)
        return {
            "jaw_slide_axes_site": np.asarray(jaw_axes),
            "xyz": native.sim.data.site_xpos[site].copy(),
            "rotation": site_rotation,
            "body_to_site_rotation": body_rotation.T @ site_rotation,
            "control_to_world_rotation": origin.copy(),
            "input_min": np.asarray(controller.input_min).copy(),
            "input_max": np.asarray(controller.input_max).copy(),
            "output_min": np.asarray(controller.output_min).copy(),
            "output_max": np.asarray(controller.output_max).copy(),
            "reference": "grip_site",
            "control_reference": controller.input_ref_frame,
        }


def wrap_factory(factory):
    """Wrap one existing RLinf factory; no simulator import or environment clone."""

    def create():
        return ControlObservation(factory())

    return create


def controlled_env_type():
    """Return a thin LiberoEnv subclass using its supported factory extension."""
    from rlinf.envs.sim.libero.libero_env import LiberoEnv

    class ControlledLiberoEnv(LiberoEnv):
        def get_env_fns(self):
            return [wrap_factory(factory) for factory in super().get_env_fns()]

    return ControlledLiberoEnv
