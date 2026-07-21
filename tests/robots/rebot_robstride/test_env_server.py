from __future__ import annotations

import threading

import pytest

from robots.rebot_robstride.env_server import make_dispatch, validate_loopback_host


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

    def heartbeat(self):
        return {"op": "heartbeat"}


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
    assert dispatch("robot.heartbeat", (), {}) == {"op": "heartbeat"}

    with pytest.raises(ValueError, match="unknown RPC method"):
        dispatch("robot.set_zero", (), {})


def test_shutdown_sets_event_after_emergency_stop() -> None:
    shutdown = threading.Event()
    dispatch = make_dispatch(FakeDriver(), shutdown)

    assert dispatch("shutdown", (), {}) == {"ok": True}
    assert shutdown.is_set()


def test_shutdown_does_not_set_event_when_emergency_stop_fails() -> None:
    class FailingDriver(FakeDriver):
        def emergency_stop(self):
            raise RuntimeError("disable failed")

    shutdown = threading.Event()
    dispatch = make_dispatch(FailingDriver(), shutdown)

    with pytest.raises(RuntimeError, match="disable failed"):
        dispatch("shutdown", (), {})
    assert not shutdown.is_set()


@pytest.mark.parametrize("host", ["127.0.0.1", "127.0.0.2", "::1", "localhost"])
def test_loopback_hosts_are_accepted(host: str) -> None:
    validate_loopback_host(host)


@pytest.mark.parametrize("host", ["0.0.0.0", "192.168.1.4", "rpent.example"])
def test_non_loopback_hosts_are_rejected(host: str) -> None:
    with pytest.raises(ValueError, match="loopback"):
        validate_loopback_host(host)
