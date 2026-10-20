# Copyright 2026 The RPent Authors.
# Licensed under the Apache License, Version 2.0.

import logging
from types import SimpleNamespace

import numpy as np
import pytest

from robots.yam.followers import YamFollowerPair, YamMoveCancelled


class Arm:
    """An RLinf YAM arm part that tracks every target at once."""

    def __init__(self):
        self.qpos = np.array([0.0, 0.1, 0.2, 0.0, 0.0, 0.0, 1.0])
        self.sent = []
        self.holds = 0
        self.connected = False
        self.release_error = None

    def connect(self):
        self.connected = True

    def disconnect(self):
        self.connected = False
        if self.release_error is not None:
            raise self.release_error

    def get_state(self):
        return SimpleNamespace(
            arm_joint_position=self.qpos[:6].copy(),
            gripper_position=self.qpos[6:].copy(),
            timestamp=1.0,
        )

    def joint_limits(self):
        return np.tile([-3.0, 3.0], (6, 1))

    def move_joints(self, joints):
        self.qpos[:6] = joints
        self.sent.append(self.qpos.copy())

    def move_gripper(self, opening):
        self.qpos[6] = np.asarray(opening).reshape(-1)[0]
        self.sent.append(self.qpos.copy())

    def hold(self):
        self.holds += 1


@pytest.fixture
def arms():
    return Arm(), Arm()


@pytest.fixture
def pair(arms):
    value = YamFollowerPair(
        *arms,
        joint_lower=np.full((2, 6), -2.0),
        joint_upper=np.full((2, 6), 2.0),
        step_frequency=1000.0,
    )
    value.connect_followers()
    return value


def test_non_finite_target_is_held_not_sent(pair, arms):
    target = pair.read_state().as_vector()
    target[3] = np.nan

    result = pair.command(target)

    assert result.rejection_reason == "non_finite_action"
    assert all(not arm.sent and arm.holds == 1 for arm in arms)
    np.testing.assert_array_equal(result.accepted, pair.read_state().as_vector())


def test_cancelled_move_holds_and_stays_connected(pair, arms):
    target = np.tile([0.5, 0.5, 0.5, 0.0, 0.0, 0.0, 0.5], 2)
    sends = []

    with pytest.raises(YamMoveCancelled):
        pair.move_to_qpos(
            target,
            duration_s=0.05,
            max_joint_delta=0.02,
            tolerance=0.01,
            timeout_s=5.0,
            cancelled=lambda: sends.append(None) or len(sends) > 3,
        )

    assert all(arm.connected and arm.holds == 1 for arm in arms)
    assert len(arms[0].sent) == 3 * 2
    assert pair.followers is not None


def test_failed_release_keeps_failing(pair, arms):
    arms[1].release_error = RuntimeError("CAN thread did not stop")

    with pytest.raises(RuntimeError, match="failed to release"):
        pair.close()
    with pytest.raises(RuntimeError, match="state is unknown"):
        pair.close()
    assert not arms[0].connected


@pytest.mark.parametrize("failed_side", [0, 1], ids=["left", "right"])
def test_failed_connect_keeps_release_state_unknown(arms, monkeypatch, failed_side):
    def fail_connect():
        raise RuntimeError("CAN cleanup failed during startup")

    monkeypatch.setattr(arms[failed_side], "connect", fail_connect)
    pair = YamFollowerPair(
        *arms,
        joint_lower=np.full((2, 6), -2.0),
        joint_upper=np.full((2, 6), 2.0),
        step_frequency=1000.0,
    )

    with pytest.raises(RuntimeError, match="CAN cleanup failed during startup"):
        pair.connect_followers()

    assert not any(arm.connected for arm in arms)
    for _ in range(2):
        with pytest.raises(RuntimeError, match="state is unknown"):
            pair.close()
    with pytest.raises(RuntimeError, match="cannot reconnect"):
        pair.connect_followers()


@pytest.mark.parametrize("failed_side", ["left", "right"])
def test_sdk_startup_cleanup_failure_keeps_release_state_unknown(
    monkeypatch, tmp_path, failed_side
):
    rlinf_logging = pytest.importorskip("rlinf.utils.logging")
    monkeypatch.setattr(
        rlinf_logging, "get_logger", lambda *args, **kwargs: logging.getLogger(__name__)
    )
    yam = pytest.importorskip("rlinf.robotics.parts.arms.yam")
    claims = pytest.importorskip("rlinf.robotics.parts.claims")
    sdk = pytest.importorskip("i2rt.robots.get_robot")
    monkeypatch.setattr(claims, "_CLAIM_DIR", str(tmp_path / "claims"))

    class SDKDevice:
        def __init__(self, fails):
            self.fails = fails
            self.closed = False

        def get_joint_pos(self):
            return np.zeros(7)

        def command_joint_pos(self, target):
            if self.fails:
                raise RuntimeError("SDK startup write failed")

        def close(self):
            if self.fails:
                raise RuntimeError("SDK cleanup failed")
            self.closed = True

    devices = {side: SDKDevice(side == failed_side) for side in ("left", "right")}
    monkeypatch.setattr(
        sdk, "get_yam_robot", lambda **settings: devices[settings["channel"]]
    )
    arms = [yam.YamArm(channel=side) for side in ("left", "right")]
    pair = YamFollowerPair(
        *arms,
        joint_lower=np.full((2, 6), -2.0),
        joint_upper=np.full((2, 6), 2.0),
        step_frequency=1000.0,
    )

    with pytest.raises(RuntimeError, match="SDK cleanup failed"):
        pair.connect_followers()

    assert not any(arm.is_connected for arm in arms)
    assert not devices[failed_side].closed
    if failed_side == "right":
        assert devices["left"].closed
    with pytest.raises(RuntimeError, match="state is unknown"):
        pair.close()


def test_limits_wider_than_the_sdk_close_what_opened(arms):
    pair = YamFollowerPair(
        *arms,
        joint_lower=np.full((2, 6), -4.0),
        joint_upper=np.full((2, 6), 2.0),
        step_frequency=1000.0,
    )

    with pytest.raises(ValueError, match="within the i2rt limits"):
        pair.connect_followers()
    assert not any(arm.connected for arm in arms)
    assert pair.followers is None
    pair.close()
