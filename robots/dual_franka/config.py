"""Developer defaults for a dual-Franka runtime.

Users edit only ``robot_config.yaml`` (machine identity + workspace geometry).
Everything here is developer-tuned or fixed placement, applied over RLinf's own
``DualFrankaRobotConfig`` / ``DualFrankaTCPRobotConfig`` defaults.
"""

from __future__ import annotations

# Fixed two-node placement (the RLinf cluster/placement is a training concern;
# RPent only evaluates). Users do not change these.
NODES = [0, 1]
HARDWARE_NODE = 0
LEFT_CONTROLLER_NODE = 0
RIGHT_CONTROLLER_NODE = 1

# Primitive-control knobs consumed by the RPent dual-Franka env server. RLinf
# has no equivalent fields; ``max_step_*`` bound each interpolation step.
CONTROL = {
    "move": {"timeout_s": 20.0, "tolerance_m": 0.005, "max_step_m": 0.02},
    "rotate": {"timeout_s": 20.0, "tolerance_rad": 0.04, "max_step_rad": 0.1},
    "servo": {"iteration_multiplier": 4, "min_iterations": 8},
    "gripper": {"settle_s": 0.4, "timeout_s": 10.0, "max_iterations": 4},
}

# The d455 perception camera is a depth camera. RLinf's ``CameraInfo`` defaults
# to ``enable_depth=False``; RPent opts into depth (for back-projection), while
# resolution/fps keep ``CameraInfo``'s defaults (640x480 @ 15 fps).
PERCEPTION_DEFAULTS = {
    "enable_depth": True,
}

# Episode length, used for both ``override_cfg.max_num_steps`` (RLinf default
# 100) and ``env.eval.max_episode_steps``.
EPISODE_STEPS = 300
