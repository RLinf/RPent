from __future__ import annotations

import threading

import pytest

from robots.rebot_robstride.env_server import make_dispatch


class FakeDriver:
    def state(self):
        return {"op": "state"}

    def enable(self):
        return {"op": "enable"}

    def move_joints(self, positions, *, duration_s=2.0):
        return {"op": "move", "positions": positions, "duration_s": duration_s}

    def set_gripper(self, position, *, duration_s=1.0):
        return {"op": "gripper", "position": position, "duration_s": duration_s}

    def stop_motion(self):
        return {"op": "stop"}

    def reset_stop(self):
        return {"op": "reset"}

    def emergency_stop(self):
        return {"op": "estop"}


def test_dispatch_maps_only_declared_robot_methods() -> None:
    shutdown = threading.Event()
    dispatch = make_dispatch(FakeDriver(), shutdown)

    assert dispatch("robot.state", (), {}) == {"op": "state"}
    assert dispatch("robot.enable", (), {}) == {"op": "enable"}
    assert (
        dispatch("robot.move_joints", (), {"positions": [0.0] * 6, "duration_s": 3.0})[
            "duration_s"
        ]
        == 3.0
    )
    assert (
        dispatch("robot.set_gripper", (), {"position": 0.5, "duration_s": 1.5})[
            "position"
        ]
        == 0.5
    )
    assert dispatch("robot.stop_motion", (), {}) == {"op": "stop"}
    assert dispatch("robot.reset_stop", (), {}) == {"op": "reset"}
    assert dispatch("robot.emergency_stop", (), {}) == {"op": "estop"}

    with pytest.raises(ValueError, match="unknown RPC method"):
        dispatch("robot.set_zero", (), {})


def test_shutdown_sets_event_without_touching_motion() -> None:
    shutdown = threading.Event()
    dispatch = make_dispatch(FakeDriver(), shutdown)

    assert dispatch("shutdown", (), {}) == {"ok": True}
    assert shutdown.is_set()
