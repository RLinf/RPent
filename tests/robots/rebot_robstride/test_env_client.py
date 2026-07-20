from __future__ import annotations

from robots.rebot_robstride.env_client import RebotRobstrideEnvClient


class FakeRpcClient:
    def __init__(self) -> None:
        self.calls: list[tuple] = []

    def call(self, method, args=(), kwargs=None, timeout_s=None):
        self.calls.append((method, args, kwargs or {}, timeout_s))
        return {"method": method, **(kwargs or {})}


def test_client_uses_stable_robot_rpc_names() -> None:
    rpc = FakeRpcClient()
    client = RebotRobstrideEnvClient(rpc)

    assert client.state()["method"] == "robot.state"
    assert client.enable()["method"] == "robot.enable"
    move = client.move_joints([0.0] * 6, duration_s=3.0)
    assert move == {
        "method": "robot.move_joints",
        "positions": [0.0] * 6,
        "duration_s": 3.0,
    }
    assert client.set_gripper(0.5, duration_s=1.5)["method"] == "robot.set_gripper"
    assert client.stop_motion()["method"] == "robot.stop_motion"
    assert client.reset_stop()["method"] == "robot.reset_stop"
    assert client.emergency_stop()["method"] == "robot.emergency_stop"
    assert client.heartbeat()["method"] == "robot.heartbeat"
