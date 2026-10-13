# Copyright 2026 The RPent Authors.
# Licensed under the Apache License, Version 2.0.

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
        pair.move_to(
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
