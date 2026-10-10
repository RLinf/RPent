# Copyright 2026 The RPent Authors.
# Licensed under the Apache License, Version 2.0 (the "License");

"""LIBERO's native OSC action layout shared by clients and model adapters."""

from rpent.robots.components.wam_contracts import WAMControlSpec

# Values use LIBERO's native OSC scaling and gripper convention. Model
# adapters restore this representation before handing actions to the env.
LIBERO_OSC = WAMControlSpec(
    embodiment="libero_7d",
    action_space="libero.osc_pose.delta.v1",
    action_schema=[
        "delta_x",
        "delta_y",
        "delta_z",
        "delta_axis_angle_x",
        "delta_axis_angle_y",
        "delta_axis_angle_z",
        "gripper",
    ],
    action_type=None,
)
