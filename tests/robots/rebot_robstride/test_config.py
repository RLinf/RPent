from __future__ import annotations

from pathlib import Path

import pytest

from robots.rebot_robstride.config import default_config, load_config


def test_default_config_matches_rebot_robstride_bus() -> None:
    config = default_config()
    assert config.channel == "can0"
    assert config.bitrate == 1_000_000
    assert [joint.motor_id for joint in config.joints] == [1, 2, 3, 4, 5, 6]
    assert [joint.model for joint in config.joints] == [
        "rs-06",
        "rs-06",
        "rs-06",
        "rs-00",
        "rs-00",
        "rs-00",
    ]
    assert config.gripper.motor_id == 7
    assert config.gripper.model == "rs-00"
    assert config.gripper.open_position is None
    assert config.gripper.closed_position is None


def test_load_config_rejects_duplicate_motor_ids(tmp_path: Path) -> None:
    path = tmp_path / "duplicate.yaml"
    path.write_text(
        """
channel: can0
joints:
  - {name: joint1, motor_id: 1, model: rs-06, lower: -2.8, upper: 2.8, kp: 20, kd: 1}
  - {name: joint2, motor_id: 1, model: rs-06, lower: -3.14, upper: 0, kp: 20, kd: 1}
  - {name: joint3, motor_id: 3, model: rs-06, lower: -3.14, upper: 0, kp: 20, kd: 1}
  - {name: joint4, motor_id: 4, model: rs-00, lower: -1.57, upper: 1.57, kp: 15, kd: 1}
  - {name: joint5, motor_id: 5, model: rs-00, lower: -1.57, upper: 1.57, kp: 15, kd: 1}
  - {name: joint6, motor_id: 6, model: rs-00, lower: -3.14, upper: 3.14, kp: 15, kd: 1}
gripper: {motor_id: 7, model: rs-00}
""".strip()
    )

    with pytest.raises(ValueError, match="motor IDs must be unique"):
        load_config(path)


def test_load_config_rejects_half_calibrated_gripper(tmp_path: Path) -> None:
    path = tmp_path / "gripper.yaml"
    path.write_text(
        """
gripper:
  open_position: -5.0
  closed_position:
""".strip()
    )

    with pytest.raises(ValueError, match="both be set or both be null"):
        load_config(path)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("control_rate_hz", ".inf"),
        ("feedback_rate_hz", ".nan"),
        ("settle_timeout_s", ".nan"),
        ("max_motion_duration_s", ".inf"),
    ],
)
def test_load_config_rejects_non_finite_global_values(
    tmp_path: Path, field: str, value: str
) -> None:
    path = tmp_path / "non-finite.yaml"
    path.write_text(f"{field}: {value}\n")

    with pytest.raises(ValueError, match="finite and positive"):
        load_config(path)


def test_load_config_rejects_unknown_root_key(tmp_path: Path) -> None:
    path = tmp_path / "unknown.yaml"
    path.write_text("control_rates_hz: 50\n")

    with pytest.raises(ValueError, match="unknown reBot config fields"):
        load_config(path)


def test_load_config_rejects_non_numeric_nested_value(tmp_path: Path) -> None:
    path = tmp_path / "bad-gripper.yaml"
    path.write_text("gripper: {kp: fast}\n")

    with pytest.raises(ValueError, match="gripper contains a non-finite value"):
        load_config(path)


@pytest.mark.parametrize(
    ("yaml_text", "message"),
    [
        ("control_rate_hz: 1\n", "control_rate_hz must be in"),
        ("feedback_rate_hz: 0.0001\n", "feedback_rate_hz must be in"),
        ("settle_timeout_s: 100000\n", "settle_timeout_s must not exceed"),
        ("heartbeat_timeout_s: 100000\n", "heartbeat_timeout_s must not exceed"),
        ("max_motion_duration_s: 100\n", "max_motion_duration_s must not exceed"),
        ("read_timeout_ms: 101\n", "read_timeout_ms must be an integer"),
        ("gripper: {kp: 1000000000}\n", "gripper gains must satisfy"),
        ("gripper: {max_velocity: 1000000000}\n", "gripper max_velocity"),
    ],
)
def test_load_config_rejects_values_above_safety_ceilings(
    tmp_path: Path, yaml_text: str, message: str
) -> None:
    path = tmp_path / "unsafe.yaml"
    path.write_text(yaml_text)

    with pytest.raises(ValueError, match=message):
        load_config(path)
