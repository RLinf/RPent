from __future__ import annotations

import threading
import time
from dataclasses import replace

import pytest

from robots.rebot_robstride.config import default_config
from robots.rebot_robstride.driver import MotionCancelled, RebotRobstrideDriver

MECH_POS = 0x7019


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0
        self.lock = threading.Lock()

    def __call__(self) -> float:
        with self.lock:
            return self.now

    def sleep(self, duration: float) -> None:
        with self.lock:
            self.now += max(0.0, duration)


class FakeMotor:
    def __init__(
        self,
        motor_id: int,
        *,
        position: float = 0.0,
        clock=time.monotonic,
    ) -> None:
        self.motor_id = motor_id
        self.position = position
        self.clock = clock
        self.calls: list[tuple] = []
        self.enabled = False
        self.fail_on_send = False
        self.follow_commands = True
        self.fault_raw = 0
        self.warning_raw = 0
        self.fault_on_send = 0
        self.clear_fault_on_clear = True
        self.send_event = threading.Event()
        self.block_enable = False
        self.enable_entered = threading.Event()
        self.release_enable = threading.Event()
        self.block_send = False
        self.send_entered = threading.Event()
        self.release_send = threading.Event()

    def robstride_get_param_f32(self, parameter: int, timeout_ms: int = 1000) -> float:
        self.calls.append(("read", parameter, timeout_ms))
        if parameter != MECH_POS:
            raise AssertionError(f"unexpected parameter {parameter:#x}")
        return self.position

    def robstride_get_fault_report(self) -> tuple[int, int]:
        self.calls.append(("fault_report",))
        return self.fault_raw, self.warning_raw

    def clear_error(self) -> None:
        self.calls.append(("clear_error",))
        if self.clear_fault_on_clear:
            self.fault_raw = 0
            self.warning_raw = 0

    def ensure_mode(self, mode, timeout_ms: int = 1000) -> None:
        self.calls.append(("ensure_mode", int(mode), timeout_ms))

    def enable(self) -> None:
        self.calls.append(("enable",))
        if self.block_enable:
            self.enable_entered.set()
            if not self.release_enable.wait(timeout=2.0):
                raise TimeoutError("test did not release motor enable")
        self.enabled = True

    def send_mit(
        self, pos: float, vel: float, kp: float, kd: float, tau: float
    ) -> None:
        if self.fail_on_send:
            raise RuntimeError("injected send failure")
        self.calls.append(("send_mit", pos, vel, kp, kd, tau, self.clock()))
        self.send_event.set()
        if self.block_send:
            self.send_entered.set()
            if not self.release_send.wait(timeout=2.0):
                raise TimeoutError("test did not release motor send")
        if self.fault_on_send:
            self.fault_raw = self.fault_on_send
        if self.follow_commands:
            self.position = pos


class FakeController:
    def __init__(
        self,
        channel: str,
        positions: dict[int, float] | None = None,
        *,
        clock=time.monotonic,
    ) -> None:
        self.channel = channel
        self.positions = positions or {}
        self.clock = clock
        self.motors: dict[int, FakeMotor] = {}
        self.disabled = False
        self.closed = False

    @property
    def enabled(self) -> bool:
        return any(motor.enabled for motor in self.motors.values())

    def add_robstride_motor(
        self, motor_id: int, feedback_id: int, model: str
    ) -> FakeMotor:
        assert feedback_id == 0xFD
        motor = FakeMotor(
            motor_id,
            position=self.positions.get(motor_id, 0.0),
            clock=self.clock,
        )
        motor.calls.append(("registered", feedback_id, model))
        self.motors[motor_id] = motor
        return motor

    def disable_all(self) -> None:
        self.disabled = True
        for motor in self.motors.values():
            motor.enabled = False

    def close(self) -> None:
        self.closed = True


def make_driver(
    positions: dict[int, float] | None = None,
    *,
    config=None,
    clock: FakeClock | None = None,
):
    fake_clock = clock or FakeClock()
    controllers: list[FakeController] = []

    def factory(channel: str) -> FakeController:
        controller = FakeController(channel, positions, clock=fake_clock)
        controllers.append(controller)
        return controller

    driver = RebotRobstrideDriver(
        config or default_config(),
        controller_factory=factory,
        clock=fake_clock,
        sleep=fake_clock.sleep,
    )
    return driver, controllers, fake_clock


def test_connect_is_passive_and_reads_every_motor() -> None:
    expected = {motor_id: motor_id / 100 for motor_id in range(1, 8)}
    driver, controllers, _ = make_driver(expected)

    state = driver.connect()

    controller = controllers[0]
    assert controller.channel == "can0"
    assert sorted(controller.motors) == list(range(1, 8))
    assert not controller.enabled
    assert state["joint_positions"] == pytest.approx([expected[i] for i in range(1, 7)])
    assert state["gripper_position"] == pytest.approx(expected[7])
    for motor in controller.motors.values():
        assert any(call[:2] == ("read", MECH_POS) for call in motor.calls)
        assert ("fault_report",) in motor.calls
        assert ("enable",) not in motor.calls

    driver.close()
    assert controller.disabled is False
    assert controller.closed is True


def test_enable_validates_then_holds_only_arm_motors() -> None:
    positions = {motor_id: -0.01 * motor_id for motor_id in range(1, 7)} | {7: 0.0}
    driver, controllers, _ = make_driver(positions)
    driver.connect()

    result = driver.enable()

    controller = controllers[0]
    assert result["enabled"] is True
    assert result["gripper_enabled"] is False
    for motor_id in range(1, 7):
        motor = controller.motors[motor_id]
        assert ("clear_error",) in motor.calls
        assert ("enable",) in motor.calls
        hold = next(call for call in motor.calls if call[0] == "send_mit")
        assert hold[1] == pytest.approx(positions[motor_id])
    assert ("enable",) not in controller.motors[7].calls
    assert not controller.motors[7].enabled


def test_enable_rejects_out_of_limit_startup_before_torque() -> None:
    driver, controllers, _ = make_driver({2: 0.2})
    driver.connect()

    with pytest.raises(ValueError, match="joint2"):
        driver.enable()

    assert not controllers[0].enabled
    assert all(
        ("enable",) not in motor.calls for motor in controllers[0].motors.values()
    )


def test_enable_rejects_persistent_fault_before_torque() -> None:
    driver, controllers, _ = make_driver()
    driver.connect()
    motor = controllers[0].motors[3]
    motor.fault_raw = 4
    motor.clear_fault_on_clear = False

    with pytest.raises(RuntimeError, match="fault"):
        driver.enable()

    assert controllers[0].disabled
    assert not controllers[0].enabled


def test_enable_failure_disables_all_motors() -> None:
    driver, controllers, _ = make_driver()
    driver.connect()
    controllers[0].motors[1].fail_on_send = True

    with pytest.raises(RuntimeError, match="injected send failure"):
        driver.enable()

    assert controllers[0].disabled
    assert driver.state()["enabled"] is False


def test_emergency_stop_cancels_an_enable_in_progress() -> None:
    driver, controllers, _ = make_driver()
    driver.connect()
    blocked_motor = controllers[0].motors[1]
    blocked_motor.block_enable = True
    enable_error: list[BaseException] = []

    def enable() -> None:
        try:
            driver.enable()
        except BaseException as exc:  # captured for the test thread
            enable_error.append(exc)

    enable_thread = threading.Thread(target=enable)
    enable_thread.start()
    assert blocked_motor.enable_entered.wait(timeout=1.0)

    stop_thread = threading.Thread(target=driver.emergency_stop)
    stop_thread.start()
    blocked_motor.release_enable.set()
    enable_thread.join(timeout=1.0)
    stop_thread.join(timeout=1.0)

    assert not enable_thread.is_alive()
    assert not stop_thread.is_alive()
    assert enable_error and isinstance(enable_error[0], MotionCancelled)
    assert controllers[0].disabled
    assert not controllers[0].enabled


def test_emergency_stop_wins_after_the_last_enable_hold() -> None:
    driver, controllers, _ = make_driver()
    driver.connect()
    last_arm_motor = controllers[0].motors[6]
    last_arm_motor.block_send = True
    enable_error: list[BaseException] = []

    def enable() -> None:
        try:
            driver.enable()
        except BaseException as exc:  # captured for the test thread
            enable_error.append(exc)

    enable_thread = threading.Thread(target=enable)
    enable_thread.start()
    assert last_arm_motor.send_entered.wait(timeout=1.0)

    stop_thread = threading.Thread(target=driver.emergency_stop)
    stop_thread.start()
    last_arm_motor.release_send.set()
    enable_thread.join(timeout=1.0)
    stop_thread.join(timeout=1.0)

    assert not enable_thread.is_alive()
    assert not stop_thread.is_alive()
    assert enable_error and isinstance(enable_error[0], MotionCancelled)
    assert controllers[0].disabled
    assert driver.state()["enabled"] is False


def test_move_joints_interpolates_and_returns_settled_evidence() -> None:
    driver, controllers, _ = make_driver()
    driver.connect()
    driver.enable()
    target = [0.1, -0.1, -0.1, 0.05, -0.05, 0.1]

    result = driver.move_joints(target, duration_s=0.2)

    assert result["reached"] is True
    assert result["final_positions"] == pytest.approx(target)
    assert result["final_velocities"] == pytest.approx([0.0] * 6)
    assert result["max_error"] == pytest.approx(0.0)
    for motor_id in range(1, 7):
        commands = [
            call
            for call in controllers[0].motors[motor_id].calls
            if call[0] == "send_mit"
        ]
        assert len(commands) >= 3


def test_minimum_jerk_profile_respects_configured_velocity_cap() -> None:
    driver, controllers, _ = make_driver()
    driver.connect()
    driver.enable()

    result = driver.move_joints([0.5, 0.0, 0.0, 0.0, 0.0, 0.0], duration_s=0.1)

    commands = [
        call for call in controllers[0].motors[1].calls if call[0] == "send_mit"
    ]
    velocities = [
        abs(current[1] - previous[1]) / (current[6] - previous[6])
        for previous, current in zip(commands, commands[1:])
        if current[6] > previous[6]
    ]
    assert max(velocities) <= driver.config.joints[0].max_velocity * 1.02
    assert result["actual_duration_s"] >= 1.875


def test_motion_transport_failure_disables_all_and_latches_stop() -> None:
    driver, controllers, _ = make_driver()
    driver.connect()
    driver.enable()
    controllers[0].motors[3].fail_on_send = True

    with pytest.raises(RuntimeError, match="injected send failure"):
        driver.move_joints([0.1, -0.1, -0.1, 0.0, 0.0, 0.0], duration_s=1.0)

    state = driver.state()
    assert state["enabled"] is False
    assert state["stopped"] is True
    assert controllers[0].disabled


def test_motion_fault_report_disables_all() -> None:
    driver, controllers, _ = make_driver()
    driver.connect()
    driver.enable()
    controllers[0].motors[3].fault_on_send = 8

    with pytest.raises(RuntimeError, match="fault during motion"):
        driver.move_joints([0.1, -0.1, -0.1, 0.0, 0.0, 0.0], duration_s=1.0)

    assert controllers[0].disabled
    assert driver.state()["enabled"] is False


def test_stalled_feedback_aborts_and_disables() -> None:
    driver, controllers, _ = make_driver()
    driver.connect()
    driver.enable()
    controllers[0].motors[1].follow_commands = False

    with pytest.raises(RuntimeError, match="tracking error"):
        driver.move_joints([0.6, 0.0, 0.0, 0.0, 0.0, 0.0], duration_s=0.1)

    assert controllers[0].disabled
    assert driver.state()["enabled"] is False


def test_settlement_timeout_returns_unreached_and_disables() -> None:
    config = replace(
        default_config(),
        max_tracking_error_rad=0.5,
        settle_timeout_s=0.2,
    )
    driver, controllers, _ = make_driver(config=config)
    driver.connect()
    driver.enable()
    controllers[0].motors[1].follow_commands = False

    result = driver.move_joints([0.1, 0.0, 0.0, 0.0, 0.0, 0.0], duration_s=0.2)

    assert result["reached"] is False
    assert result["enabled"] is False
    assert controllers[0].disabled


def test_move_joints_rejects_limit_and_duration_violations() -> None:
    driver, _, _ = make_driver()
    driver.connect()
    driver.enable()

    with pytest.raises(ValueError, match="joint2"):
        driver.move_joints([0.0, 0.1, -0.1, 0.0, 0.0, 0.0], duration_s=1.0)
    with pytest.raises(ValueError, match="must not exceed"):
        driver.move_joints([0.0] * 6, duration_s=61.0)


def test_motion_requires_enable_and_respects_soft_stop() -> None:
    driver, _, _ = make_driver()
    driver.connect()

    with pytest.raises(RuntimeError, match="not enabled"):
        driver.move_joints([0.0] * 6, duration_s=1.0)

    driver.enable()
    driver.stop_motion()
    with pytest.raises(RuntimeError, match="stopped"):
        driver.move_joints([0.0] * 6, duration_s=1.0)
    driver.reset_stop()
    assert driver.move_joints([0.0] * 6, duration_s=0.1)["reached"] is True


def test_emergency_stop_preempts_active_trajectory() -> None:
    controllers: list[FakeController] = []

    def factory(channel: str) -> FakeController:
        controller = FakeController(channel)
        controllers.append(controller)
        return controller

    driver = RebotRobstrideDriver(default_config(), controller_factory=factory)
    driver.connect()
    driver.enable()
    for motor in controllers[0].motors.values():
        motor.send_event.clear()

    motion_error: list[BaseException] = []

    def move() -> None:
        try:
            driver.move_joints([0.5, 0.0, 0.0, 0.0, 0.0, 0.0], duration_s=2.0)
        except BaseException as exc:  # captured for the test thread
            motion_error.append(exc)

    thread = threading.Thread(target=move)
    thread.start()
    assert controllers[0].motors[1].send_event.wait(timeout=1.0)

    started = time.monotonic()
    result = driver.emergency_stop()
    elapsed = time.monotonic() - started
    thread.join(timeout=1.0)

    assert elapsed < 0.25
    assert result == {"enabled": False, "stopped": True}
    assert controllers[0].disabled
    assert not thread.is_alive()
    assert motion_error and isinstance(motion_error[0], MotionCancelled)


def test_expired_agent_heartbeat_disables_enabled_motors() -> None:
    driver, controllers, clock = make_driver()
    driver.connect()
    driver.enable()

    assert driver.enforce_heartbeat_deadman() is False
    clock.sleep(driver.config.heartbeat_timeout_s + 0.01)

    assert driver.enforce_heartbeat_deadman() is True
    assert controllers[0].disabled
    assert driver.state()["enabled"] is False


def test_gripper_refuses_motion_until_endpoints_are_calibrated() -> None:
    driver, controllers, _ = make_driver()
    driver.connect()
    driver.enable()

    with pytest.raises(RuntimeError, match="not calibrated"):
        driver.set_gripper(0.5)

    assert not controllers[0].motors[7].enabled


def test_gripper_rejects_excessive_duration_before_enable() -> None:
    config = default_config()
    calibrated = replace(
        config,
        gripper=replace(
            config.gripper,
            open_position=0.0,
            closed_position=-1.0,
            max_velocity=0.01,
        ),
    )
    driver, controllers, _ = make_driver(config=calibrated)
    driver.connect()
    driver.enable()

    with pytest.raises(ValueError, match="exceeds max_motion_duration_s"):
        driver.set_gripper(1.0)

    gripper = controllers[0].motors[7]
    assert not gripper.enabled
    assert not any(call[0] == "enable" for call in gripper.calls)


def test_gripper_maps_and_returns_settlement_evidence() -> None:
    config = default_config()
    calibrated = replace(
        config,
        gripper=replace(config.gripper, open_position=-1.0, closed_position=0.0),
    )
    driver, controllers, _ = make_driver(config=calibrated)
    driver.connect()
    driver.enable()

    result = driver.set_gripper(0.25)

    assert result["target_position"] == pytest.approx(-0.75)
    assert result["final_position"] == pytest.approx(-0.75)
    assert result["final_velocity"] == pytest.approx(0.0)
    assert result["max_error"] == pytest.approx(0.0)
    assert result["reached"] is True
    assert controllers[0].motors[7].enabled


def test_gripper_transport_failure_disables_all() -> None:
    config = default_config()
    calibrated = replace(
        config,
        gripper=replace(config.gripper, open_position=-1.0, closed_position=0.0),
    )
    driver, controllers, _ = make_driver(config=calibrated)
    driver.connect()
    driver.enable()
    controllers[0].motors[7].fail_on_send = True

    with pytest.raises(RuntimeError, match="injected send failure"):
        driver.set_gripper(0.5)

    assert driver.state()["stopped"] is True
    assert driver.state()["enabled"] is False
    assert controllers[0].disabled


def test_close_always_disables_if_any_motor_was_enabled() -> None:
    driver, controllers, _ = make_driver()
    driver.connect()
    driver.enable()

    driver.close()

    assert controllers[0].disabled
    assert controllers[0].closed
