from __future__ import annotations

from dataclasses import replace

import pytest

from robots.rebot_robstride.config import default_config
from robots.rebot_robstride.driver import RebotRobstrideDriver

MECH_POS = 0x7019
MECH_VEL = 0x701A


class FakeMotor:
    def __init__(self, motor_id: int, position: float = 0.0) -> None:
        self.motor_id = motor_id
        self.position = position
        self.velocity = 0.0
        self.calls: list[tuple] = []
        self.fail_on_send = False

    def robstride_get_param_f32(self, parameter: int, timeout_ms: int = 1000) -> float:
        self.calls.append(("read", parameter, timeout_ms))
        if parameter == MECH_POS:
            return self.position
        if parameter == MECH_VEL:
            return self.velocity
        raise AssertionError(f"unexpected parameter {parameter:#x}")

    def clear_error(self) -> None:
        self.calls.append(("clear_error",))

    def ensure_mode(self, mode, timeout_ms: int = 1000) -> None:
        self.calls.append(("ensure_mode", int(mode), timeout_ms))

    def send_mit(
        self, pos: float, vel: float, kp: float, kd: float, tau: float
    ) -> None:
        if self.fail_on_send:
            raise RuntimeError("injected send failure")
        self.calls.append(("send_mit", pos, vel, kp, kd, tau))
        self.velocity = pos - self.position
        self.position = pos


class FakeController:
    def __init__(self, channel: str, positions: dict[int, float] | None = None) -> None:
        self.channel = channel
        self.positions = positions or {}
        self.motors: dict[int, FakeMotor] = {}
        self.enabled = False
        self.disabled = False
        self.closed = False

    def add_robstride_motor(
        self, motor_id: int, feedback_id: int, model: str
    ) -> FakeMotor:
        assert feedback_id == 0xFD
        motor = FakeMotor(motor_id, self.positions.get(motor_id, 0.0))
        motor.calls.append(("registered", feedback_id, model))
        self.motors[motor_id] = motor
        return motor

    def enable_all(self) -> None:
        self.enabled = True

    def disable_all(self) -> None:
        self.disabled = True
        self.enabled = False

    def close(self) -> None:
        self.closed = True


def make_driver(positions: dict[int, float] | None = None):
    controllers: list[FakeController] = []

    def factory(channel: str) -> FakeController:
        controller = FakeController(channel, positions)
        controllers.append(controller)
        return controller

    driver = RebotRobstrideDriver(
        default_config(),
        controller_factory=factory,
        sleep=lambda _: None,
    )
    return driver, controllers


def test_connect_is_passive_and_reads_every_motor() -> None:
    expected = {motor_id: motor_id / 100 for motor_id in range(1, 8)}
    driver, controllers = make_driver(expected)

    state = driver.connect()

    controller = controllers[0]
    assert controller.channel == "can0"
    assert sorted(controller.motors) == list(range(1, 8))
    assert not controller.enabled
    assert state["joint_positions"] == pytest.approx([expected[i] for i in range(1, 7)])
    assert state["gripper_position"] == pytest.approx(expected[7])
    for motor in controller.motors.values():
        assert any(call[:2] == ("read", MECH_POS) for call in motor.calls)

    driver.close()
    assert controller.disabled is False
    assert controller.closed is True


def test_enable_holds_observed_pose_before_accepting_motion() -> None:
    positions = {motor_id: -0.01 * motor_id for motor_id in range(1, 8)}
    driver, controllers = make_driver(positions)
    driver.connect()

    result = driver.enable()

    controller = controllers[0]
    assert result["enabled"] is True
    assert controller.enabled
    for motor_id, motor in controller.motors.items():
        names = [call[0] for call in motor.calls]
        assert "clear_error" in names
        assert "ensure_mode" in names
        hold = next(call for call in motor.calls if call[0] == "send_mit")
        assert hold[1] == pytest.approx(positions[motor_id])


def test_enable_failure_disables_all_motors() -> None:
    driver, controllers = make_driver()
    driver.connect()
    controllers[0].motors[1].fail_on_send = True

    with pytest.raises(RuntimeError, match="injected send failure"):
        driver.enable()

    assert controllers[0].disabled is True
    assert driver.state()["enabled"] is False


def test_move_joints_interpolates_and_returns_readback_evidence() -> None:
    driver, controllers = make_driver()
    driver.connect()
    driver.enable()
    target = [0.1, -0.1, -0.1, 0.05, -0.05, 0.1]

    result = driver.move_joints(target, duration_s=0.2)

    assert result["reached"] is True
    assert result["final_positions"] == pytest.approx(target)
    assert result["max_error"] == pytest.approx(0.0)
    for motor_id in range(1, 7):
        commands = [
            call
            for call in controllers[0].motors[motor_id].calls
            if call[0] == "send_mit"
        ]
        assert len(commands) >= 3  # initial hold plus at least two trajectory points


def test_motion_transport_failure_latches_soft_stop() -> None:
    driver, controllers = make_driver()
    driver.connect()
    driver.enable()
    controllers[0].motors[3].fail_on_send = True

    with pytest.raises(RuntimeError, match="injected send failure"):
        driver.move_joints([0.1, -0.1, -0.1, 0.0, 0.0, 0.0], duration_s=1.0)

    state = driver.state()
    assert state["enabled"] is True
    assert state["stopped"] is True
    with pytest.raises(RuntimeError, match="stopped"):
        driver.move_joints([0.0] * 6, duration_s=1.0)


def test_move_joints_rejects_limit_violation() -> None:
    driver, _ = make_driver()
    driver.connect()
    driver.enable()

    with pytest.raises(ValueError, match="joint2"):
        driver.move_joints([0.0, 0.1, -0.1, 0.0, 0.0, 0.0], duration_s=1.0)


def test_motion_requires_enable_and_respects_soft_stop() -> None:
    driver, _ = make_driver()
    driver.connect()

    with pytest.raises(RuntimeError, match="not enabled"):
        driver.move_joints([0.0] * 6, duration_s=1.0)

    driver.enable()
    driver.stop_motion()
    with pytest.raises(RuntimeError, match="stopped"):
        driver.move_joints([0.0] * 6, duration_s=1.0)
    driver.reset_stop()
    assert driver.move_joints([0.0] * 6, duration_s=0.1)["reached"] is True


def test_stop_failure_keeps_soft_stop_latched() -> None:
    driver, controllers = make_driver()
    driver.connect()
    driver.enable()
    controllers[0].motors[3].fail_on_send = True

    with pytest.raises(RuntimeError, match="injected send failure"):
        driver.stop_motion()

    assert driver.state()["stopped"] is True


def test_emergency_stop_disables_all_motors() -> None:
    driver, controllers = make_driver()
    driver.connect()
    driver.enable()

    result = driver.emergency_stop()

    assert result == {"enabled": False, "stopped": True}
    assert controllers[0].disabled


def test_gripper_refuses_motion_until_endpoints_are_calibrated() -> None:
    driver, _ = make_driver()
    driver.connect()
    driver.enable()

    with pytest.raises(RuntimeError, match="not calibrated"):
        driver.set_gripper(0.5)


def test_gripper_maps_normalized_position_to_calibrated_range() -> None:
    config = default_config()
    calibrated = replace(
        config,
        gripper=replace(config.gripper, open_position=-1.0, closed_position=0.0),
    )
    controller = FakeController("can0")
    driver = RebotRobstrideDriver(
        calibrated,
        controller_factory=lambda _: controller,
        sleep=lambda _: None,
    )
    driver.connect()
    driver.enable()

    with pytest.raises(ValueError, match="duration_s"):
        driver.set_gripper(0.25, duration_s=0.0)

    result = driver.set_gripper(0.25)

    assert result["target_position"] == pytest.approx(-0.75)
    assert controller.motors[7].position == pytest.approx(-0.75)


def test_gripper_transport_failure_latches_soft_stop() -> None:
    config = default_config()
    calibrated = replace(
        config,
        gripper=replace(config.gripper, open_position=-1.0, closed_position=0.0),
    )
    controller = FakeController("can0")
    driver = RebotRobstrideDriver(
        calibrated,
        controller_factory=lambda _: controller,
        sleep=lambda _: None,
    )
    driver.connect()
    driver.enable()
    controller.motors[7].fail_on_send = True

    with pytest.raises(RuntimeError, match="injected send failure"):
        driver.set_gripper(0.5)

    assert driver.state()["stopped"] is True
