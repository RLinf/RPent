from __future__ import annotations

from robots.rebot_robstride.toolkit import RebotRobstrideToolkit


class FakeEnv:
    def __init__(self) -> None:
        self.calls: list[tuple] = []

    def state(self):
        self.calls.append(("state",))
        return {"enabled": False, "joint_positions": [0.0] * 6}

    def enable(self):
        self.calls.append(("enable",))
        return {"enabled": True}

    def move_joints(self, positions, *, duration_s=2.0):
        self.calls.append(("move_joints", positions, duration_s))
        return {"reached": True, "final_positions": positions}

    def set_gripper(self, position, *, duration_s=1.0):
        self.calls.append(("set_gripper", position, duration_s))
        return {"normalized_position": position}

    def stop_motion(self):
        self.calls.append(("stop_motion",))
        return {"stopped": True}

    def reset_stop(self):
        self.calls.append(("reset_stop",))
        return {"stopped": False}

    def emergency_stop(self):
        self.calls.append(("emergency_stop",))
        return {"enabled": False, "stopped": True}


def test_toolkit_exposes_safe_rebot_tools() -> None:
    toolkit = RebotRobstrideToolkit(env=FakeEnv())

    names = [spec["name"] for spec in toolkit.get_tools_spec()]

    assert {
        "get_robot_state",
        "enable_arm",
        "move_joints",
        "set_gripper",
        "open_gripper",
        "close_gripper",
        "stop_motion",
        "reset_stop",
        "emergency_stop",
    }.issubset(names)


def test_toolkit_dispatches_motion_and_gripper_without_bypassing_client() -> None:
    env = FakeEnv()
    toolkit = RebotRobstrideToolkit(env=env)

    move = toolkit.execute_tool(
        "move_joints", {"positions": [0.0] * 6, "duration_s": 3.0}
    )
    opened = toolkit.execute_tool("open_gripper", {"duration_s": 1.5})
    closed = toolkit.execute_tool("close_gripper", {})

    assert move.result["reached"] is True
    assert opened.result["normalized_position"] == 0.0
    assert closed.result["normalized_position"] == 1.0
    assert ("move_joints", [0.0] * 6, 3.0) in env.calls
    assert ("set_gripper", 0.0, 1.5) in env.calls
    assert ("set_gripper", 1.0, 1.0) in env.calls
