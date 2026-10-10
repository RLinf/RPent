# Copyright 2026 The RPent Authors.
# Licensed under the Apache License, Version 2.0 (the "License");

"""RoboTwin action layouts for native qpos and end-effector execution."""

from rpent.robots.components.wam_control_spec import WAMControlSpec

# Native joint_action.vector order: left six joint targets and gripper,
# then right six joint targets and gripper. Preserve native gripper values.
ROBOTWIN_QPOS = WAMControlSpec(
    embodiment="robotwin_qpos14",
    action_space="robotwin.joint_action.qpos14.v1",
    action_schema=[f"joint_action_{i}" for i in range(14)],
    action_type="qpos",
)

# Absolute native end-effector poses with wxyz quaternions, left arm first.
ROBOTWIN_EEF = WAMControlSpec(
    embodiment="robotwin_eef16",
    action_space="robotwin.eef_pose.absolute.v1",
    action_schema=[
        f"{arm}_{field}"
        for arm in ("left", "right")
        for field in ("x", "y", "z", "qw", "qx", "qy", "qz", "gripper")
    ],
    action_type="ee",
)

ROBOTWIN_CONTROLLERS = {
    spec["action_space"]: spec for spec in (ROBOTWIN_QPOS, ROBOTWIN_EEF)
}
