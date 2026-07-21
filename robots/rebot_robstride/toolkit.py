"""RPent tool surface for the physical reBot DevArm RobStride arm."""

from __future__ import annotations

from typing import Any

from rpent.tools.toolkit import Toolkit

TOOLS_SPEC = [
    {
        "name": "get_robot_state",
        "description": (
            "Read fresh RobStride joint, velocity, gripper, enable, stop, and "
            "disable-failure state. Includes raw fault/warning reports. Call this "
            "before enabling or planning any motion."
        ),
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "enable_arm",
        "description": (
            "Explicitly enable the six arm joints after reading state. The driver "
            "validates startup feedback, clears and rechecks faults, selects MIT mode, "
            "and holds the observed pose before returning."
        ),
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "move_joints",
        "description": (
            "Move six arm joints in raw motor radians through a bounded minimum-jerk "
            "trajectory. Returns final read-back evidence."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "positions": {
                    "type": "array",
                    "items": {"type": "number"},
                    "minItems": 6,
                    "maxItems": 6,
                },
                "duration_s": {
                    "type": "number",
                    "minimum": 0.1,
                    "maximum": 60.0,
                    "default": 2.0,
                },
            },
            "required": ["positions"],
        },
    },
    {
        "name": "set_gripper",
        "description": (
            "Move the calibrated gripper to a normalized position: 0.0=open, "
            "1.0=closed. Refuses to move when endpoints are not configured."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "position": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                "duration_s": {
                    "type": "number",
                    "minimum": 0.1,
                    "maximum": 60.0,
                    "default": 1.0,
                },
            },
            "required": ["position"],
        },
    },
    {
        "name": "open_gripper",
        "description": "Open the calibrated gripper with a bounded trajectory.",
        "input_schema": {
            "type": "object",
            "properties": {
                "duration_s": {
                    "type": "number",
                    "minimum": 0.1,
                    "maximum": 60.0,
                    "default": 1.0,
                }
            },
            "required": [],
        },
    },
    {
        "name": "close_gripper",
        "description": "Close the calibrated gripper with a bounded trajectory.",
        "input_schema": {
            "type": "object",
            "properties": {
                "duration_s": {
                    "type": "number",
                    "minimum": 0.1,
                    "maximum": 60.0,
                    "default": 1.0,
                }
            },
            "required": [],
        },
    },
    {
        "name": "stop_motion",
        "description": (
            "Latch a software stop, hold the current observed pose, and reject new "
            "motions until reset_stop is called."
        ),
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "reset_stop",
        "description": "Clear the software-stop latch. This never enables motors.",
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "emergency_stop",
        "description": (
            "Immediately disable every motor. WARNING: an unsupported arm may fall "
            "under gravity after this torque-off operation."
        ),
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
]


class RebotRobstrideToolkit(Toolkit):
    """Common RPent tools plus guarded physical-arm operations."""

    def __init__(self, *, env: Any, dashboard: Any = None) -> None:
        super().__init__(dashboard=dashboard)
        self._env = env
        specs = {spec["name"]: spec for spec in TOOLS_SPEC}
        self.add_tool("get_robot_state", specs["get_robot_state"], env.state)
        self.add_tool("enable_arm", specs["enable_arm"], env.enable)
        self.add_tool("move_joints", specs["move_joints"], env.move_joints)
        self.add_tool("set_gripper", specs["set_gripper"], env.set_gripper)
        self.add_tool(
            "open_gripper",
            specs["open_gripper"],
            lambda duration_s=1.0: env.set_gripper(0.0, duration_s=duration_s),
        )
        self.add_tool(
            "close_gripper",
            specs["close_gripper"],
            lambda duration_s=1.0: env.set_gripper(1.0, duration_s=duration_s),
        )
        self.add_tool("stop_motion", specs["stop_motion"], env.stop_motion)
        self.add_tool("reset_stop", specs["reset_stop"], env.reset_stop)
        self.add_tool("emergency_stop", specs["emergency_stop"], env.emergency_stop)
