from __future__ import annotations

import threading
import time
from dataclasses import replace
from types import SimpleNamespace

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


class BlockingSetEvent:
    def __init__(self) -> None:
        self._event = threading.Event()
        self.block_next_set = False
        self.set_entered = threading.Event()
        self.release_set = threading.Event()

    def set(self) -> None:
        self._event.set()
        if self.block_next_set:
            self.block_next_set = False
            self.set_entered.set()
            if not self.release_set.wait(timeout=2.0):
                raise TimeoutError("test did not release cancellation set")

    def clear(self) -> None:
        self._event.clear()

    def is_set(self) -> bool:
        return self._event.is_set()

    def wait(self, timeout: float | None = None) -> bool:
        return self._event.wait(timeout)


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
        self.read_delay_s = 0.0
        self.read_entered = threading.Event()
        self.status_available = True
        self.status_code = 0
        self.active_report = False

    def _delay_read(self) -> None:
        self.read_entered.set()
        if self.read_delay_s <= 0:
            return
        sleep = getattr(self.clock, "sleep", time.sleep)
        sleep(self.read_delay_s)

    def robstride_get_param_f32(self, parameter: int, timeout_ms: int = 1000) -> float:
        self.calls.append(("read", parameter, timeout_ms))
        self._delay_read()
        if parameter != MECH_POS:
            raise AssertionError(f"unexpected parameter {parameter:#x}")
        return self.position

    def robstride_get_fault_report(self) -> tuple[int, int]:
        self.calls.append(("fault_report",))
        self._delay_read()
        return self.fault_raw, self.warning_raw

    def get_state(self):
        self.calls.append(("get_state",))
        if not self.status_available:
            return None
        return SimpleNamespace(status_code=self.status_code)

    def robstride_set_active_report(self, enabled: bool) -> None:
        self.calls.append(("active_report", enabled))
        self.active_report = enabled

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
        self.fail_disable = False
        self.block_disable = False
        self.disable_entered = threading.Event()
        self.release_disable = threading.Event()
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
        if self.fail_disable:
            raise RuntimeError("injected disable failure")
        if self.block_disable:
            self.disable_entered.set()
            if not self.release_disable.wait(timeout=2.0):
                raise TimeoutError("test did not release disable_all")
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
        assert ("active_report", True) in motor.calls
        assert ("enable",) not in motor.calls

    driver.close()
    assert controller.disabled is True
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


def test_repeated_enable_preserves_an_enabled_gripper() -> None:
    config = default_config()
    calibrated = replace(
        config,
        gripper=replace(config.gripper, open_position=-1.0, closed_position=0.0),
    )
    driver, controllers, _ = make_driver(config=calibrated)
    driver.connect()
    driver.enable()
    driver.set_gripper(0.5)

    result = driver.enable()

    assert result["gripper_enabled"] is True
    assert controllers[0].motors[7].enabled
    assert driver.state()["gripper_enabled"] is True


def test_motion_refreshes_holds_for_every_enabled_motor() -> None:
    config = default_config()
    calibrated = replace(
        config,
        gripper=replace(config.gripper, open_position=-0.5, closed_position=0.5),
    )
    driver, controllers, _ = make_driver(config=calibrated)
    driver.connect()
    driver.enable()
    driver.set_gripper(0.5, duration_s=0.1)
    controller = controllers[0]

    gripper_sends_before = sum(
        call[0] == "send_mit" for call in controller.motors[7].calls
    )
    driver.move_joints([0.0] * 6, duration_s=0.1)
    gripper_sends_after = sum(
        call[0] == "send_mit" for call in controller.motors[7].calls
    )
    assert gripper_sends_after > gripper_sends_before

    arm_sends_before = {
        motor_id: sum(
            call[0] == "send_mit" for call in controller.motors[motor_id].calls
        )
        for motor_id in range(1, 7)
    }
    driver.set_gripper(0.25, duration_s=0.1)
    for motor_id in range(1, 7):
        arm_sends_after = sum(
            call[0] == "send_mit" for call in controller.motors[motor_id].calls
        )
        assert arm_sends_after > arm_sends_before[motor_id]


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


def test_enable_rejects_missing_operation_status() -> None:
    driver, controllers, _ = make_driver()
    driver.connect()
    controllers[0].motors[1].status_available = False

    with pytest.raises(RuntimeError, match="operation status"):
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


def test_emergency_stop_invalidates_enable_that_already_passed_admission() -> None:
    driver, controllers, _ = make_driver()
    driver.connect()
    original_sample = driver._sample_feedback
    cancellation = BlockingSetEvent()
    cancellation.block_next_set = True
    driver.__dict__["_cancel_event"] = cancellation
    sample_entered = threading.Event()
    release_sample = threading.Event()
    enable_error: list[BaseException] = []

    def blocked_sample(operation_epoch: int):
        sample_entered.set()
        if not release_sample.wait(timeout=2.0):
            raise TimeoutError("test did not release startup feedback")
        return original_sample(operation_epoch)

    def enable() -> None:
        try:
            driver.enable()
        except BaseException as exc:  # captured for the test thread
            enable_error.append(exc)

    driver._sample_feedback = blocked_sample
    estop_thread = threading.Thread(target=driver.emergency_stop)
    estop_thread.start()
    assert cancellation.set_entered.wait(timeout=1.0)
    enable_thread = threading.Thread(target=enable)
    enable_thread.start()

    sample_entered.wait(timeout=0.2)
    cancellation.release_set.set()
    estop_thread.join(timeout=1.0)
    release_sample.set()
    enable_thread.join(timeout=1.0)

    assert not estop_thread.is_alive()
    assert not enable_thread.is_alive()
    assert enable_error and isinstance(enable_error[0], (MotionCancelled, RuntimeError))
    assert controllers[0].disabled
    assert not controllers[0].enabled
    assert driver.state()["stopped"] is True


def test_reset_stop_rejects_while_emergency_disable_is_in_progress() -> None:
    driver, controllers, _ = make_driver()
    driver.connect()
    driver.enable()
    controller = controllers[0]
    controller.block_disable = True

    estop_thread = threading.Thread(target=driver.emergency_stop)
    estop_thread.start()
    assert controller.disable_entered.wait(timeout=1.0)

    try:
        with pytest.raises(RuntimeError, match="stop.*in progress"):
            driver.reset_stop()
    finally:
        controller.release_disable.set()
        estop_thread.join(timeout=1.0)
    assert not estop_thread.is_alive()


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


def test_feedback_latency_does_not_create_catch_up_bursts() -> None:
    driver, controllers, _ = make_driver()
    driver.connect()
    driver.enable()
    for motor in controllers[0].motors.values():
        motor.read_delay_s = 0.003
    motor = controllers[0].motors[1]
    command_count = sum(call[0] == "send_mit" for call in motor.calls)

    driver.move_joints([0.5, 0.0, 0.0, 0.0, 0.0, 0.0], duration_s=0.1)

    commands = [call for call in motor.calls if call[0] == "send_mit"][command_count:]
    intervals = [
        current[6] - previous[6] for previous, current in zip(commands, commands[1:])
    ]
    velocities = [
        abs(current[1] - previous[1]) / interval
        for previous, current, interval in zip(commands, commands[1:], intervals)
    ]
    assert min(intervals) >= 1.0 / driver.config.control_rate_hz - 1e-9
    assert max(velocities) <= driver.config.joints[0].max_velocity * 1.02


def test_emergency_stop_waits_for_at_most_one_feedback_read() -> None:
    controllers: list[FakeController] = []

    def factory(channel: str) -> FakeController:
        controller = FakeController(channel)
        controllers.append(controller)
        return controller

    driver = RebotRobstrideDriver(default_config(), controller_factory=factory)
    driver.connect()
    driver.enable()
    for motor in controllers[0].motors.values():
        motor.read_delay_s = 0.05
        motor.read_entered.clear()

    state_error: list[BaseException] = []

    def read_state() -> None:
        try:
            driver.state()
        except BaseException as exc:  # captured for the test thread
            state_error.append(exc)

    state_thread = threading.Thread(target=read_state)
    state_thread.start()
    assert controllers[0].motors[1].read_entered.wait(timeout=1.0)

    started = time.monotonic()
    driver.emergency_stop()
    elapsed = time.monotonic() - started
    state_thread.join(timeout=1.0)

    assert elapsed < 0.25
    assert not state_thread.is_alive()
    assert not state_error or isinstance(state_error[0], MotionCancelled)


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


def test_arm_motion_fails_closed_on_enabled_gripper_fault() -> None:
    config = default_config()
    calibrated = replace(
        config,
        gripper=replace(config.gripper, open_position=-1.0, closed_position=0.0),
    )
    driver, controllers, _ = make_driver(config=calibrated)
    driver.connect()
    driver.enable()
    driver.set_gripper(0.5)
    controllers[0].motors[7].fault_raw = 1

    with pytest.raises(RuntimeError, match="fault"):
        driver.move_joints([0.0] * 6, duration_s=0.1)

    assert controllers[0].disabled


def test_gripper_motion_fails_closed_on_enabled_arm_fault() -> None:
    config = default_config()
    calibrated = replace(
        config,
        gripper=replace(config.gripper, open_position=-1.0, closed_position=0.0),
    )
    driver, controllers, _ = make_driver(config=calibrated)
    driver.connect()
    driver.enable()
    controllers[0].motors[1].fault_raw = 1

    with pytest.raises(RuntimeError, match="fault"):
        driver.set_gripper(0.5)

    assert controllers[0].disabled


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


def test_settlement_disable_failure_latches_stop_and_blocks_reset() -> None:
    config = replace(
        default_config(),
        max_tracking_error_rad=0.5,
        settle_timeout_s=0.2,
    )
    driver, controllers, _ = make_driver(config=config)
    driver.connect()
    driver.enable()
    controllers[0].motors[1].follow_commands = False
    controllers[0].fail_disable = True

    with pytest.raises(RuntimeError, match="disable_all also failed"):
        driver.move_joints([0.1, 0.0, 0.0, 0.0, 0.0, 0.0], duration_s=0.2)

    state = driver.state()
    assert state["enabled"] is True
    assert state["stopped"] is True
    assert state["disable_failed"] is True
    with pytest.raises(RuntimeError, match="retry emergency_stop"):
        driver.reset_stop()

    controllers[0].fail_disable = False
    driver.emergency_stop()
    assert driver.state()["disable_failed"] is False


def test_move_joints_rejects_limit_and_duration_violations() -> None:
    driver, _, _ = make_driver()
    driver.connect()
    driver.enable()

    with pytest.raises(ValueError, match="joint2"):
        driver.move_joints([0.0, 0.1, -0.1, 0.0, 0.0, 0.0], duration_s=1.0)
    with pytest.raises(ValueError, match="must not exceed"):
        driver.move_joints([0.0] * 6, duration_s=61.0)


def test_motion_server_deadline_precedes_client_timeout() -> None:
    driver, controllers, clock = make_driver()
    driver.connect()
    driver.enable()
    for motor in controllers[0].motors.values():
        motor.read_delay_s = 0.3

    with pytest.raises(RuntimeError, match="server motion deadline"):
        driver.move_joints([0.0] * 6, duration_s=60.0)

    assert clock() < 75.0
    assert controllers[0].disabled


def test_gripper_server_deadline_precedes_client_timeout() -> None:
    config = default_config()
    calibrated = replace(
        config,
        gripper=replace(config.gripper, open_position=-0.5, closed_position=0.5),
    )
    driver, controllers, clock = make_driver(config=calibrated)
    driver.connect()
    driver.enable()
    for motor in controllers[0].motors.values():
        motor.read_delay_s = 0.3

    with pytest.raises(RuntimeError, match="server motion deadline"):
        driver.set_gripper(0.5, duration_s=60.0)

    assert clock() < 75.0
    assert controllers[0].disabled


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


def test_estop_between_gripper_ready_and_feedback_cannot_reenable() -> None:
    config = default_config()
    calibrated = replace(
        config,
        gripper=replace(config.gripper, open_position=-1.0, closed_position=0.0),
    )
    driver, controllers, _ = make_driver(config=calibrated)
    driver.connect()
    driver.enable()
    original_sample = driver._sample_feedback
    sample_entered = threading.Event()
    release_sample = threading.Event()
    motion_error: list[BaseException] = []

    def blocked_sample(operation_epoch: int, server_deadline: float | None = None):
        sample_entered.set()
        if not release_sample.wait(timeout=2.0):
            raise TimeoutError("test did not release feedback")
        return original_sample(operation_epoch, server_deadline)

    def move_gripper() -> None:
        try:
            driver.set_gripper(0.5)
        except BaseException as exc:  # captured for the test thread
            motion_error.append(exc)

    driver._sample_feedback = blocked_sample
    thread = threading.Thread(target=move_gripper)
    thread.start()
    assert sample_entered.wait(timeout=1.0)

    driver.emergency_stop()
    release_sample.set()
    thread.join(timeout=1.0)

    assert not thread.is_alive()
    assert motion_error and isinstance(motion_error[0], MotionCancelled)
    assert not controllers[0].motors[7].enabled
    assert driver.state()["enabled"] is False


def test_soft_stop_holds_enabled_gripper_at_measured_position() -> None:
    config = default_config()
    calibrated = replace(
        config,
        gripper=replace(config.gripper, open_position=-1.0, closed_position=0.0),
    )
    driver, controllers, _ = make_driver(config=calibrated)
    driver.connect()
    driver.enable()
    driver.set_gripper(0.5)
    gripper = controllers[0].motors[7]
    gripper.position = -0.4

    result = driver.stop_motion()

    gripper_commands = [call for call in gripper.calls if call[0] == "send_mit"]
    assert gripper_commands[-1][1] == pytest.approx(-0.4)
    assert result["gripper_enabled"] is True
    assert result["stopped"] is True


def test_soft_stop_persists_fresh_gripper_hold_after_reset() -> None:
    config = default_config()
    calibrated = replace(
        config,
        gripper=replace(config.gripper, open_position=-1.0, closed_position=0.0),
    )
    driver, controllers, _ = make_driver(config=calibrated)
    driver.connect()
    driver.enable()
    driver.set_gripper(0.5)
    gripper = controllers[0].motors[7]
    gripper.position = -0.4

    driver.stop_motion()
    driver.reset_stop()
    command_count = sum(call[0] == "send_mit" for call in gripper.calls)
    driver.move_joints([0.0] * 6, duration_s=0.1)

    later_commands = [call for call in gripper.calls if call[0] == "send_mit"][
        command_count:
    ]
    assert later_commands
    assert all(call[1] == pytest.approx(-0.4) for call in later_commands)


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


def test_close_passive_connection_still_confirms_disable_and_is_idempotent() -> None:
    driver, controllers, _ = make_driver()
    driver.connect()

    driver.close()
    driver.close()

    assert controllers[0].disabled
    assert controllers[0].closed


def test_close_disable_failure_preserves_connected_uncertain_state() -> None:
    driver, controllers, _ = make_driver()
    driver.connect()
    driver.enable()
    controllers[0].fail_disable = True

    with pytest.raises(RuntimeError, match="injected disable failure"):
        driver.close()

    state = driver.state()
    assert state["connected"] is True
    assert state["enabled"] is True
    assert state["disable_failed"] is True
    assert not controllers[0].closed

    controllers[0].fail_disable = False
    driver.close()
    driver.close()

    assert controllers[0].disabled
    assert controllers[0].closed
