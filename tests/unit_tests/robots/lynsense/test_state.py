"""Offline contracts for joint data and independent stream freshness."""

from types import SimpleNamespace

import pytest

from robots.lynsense.state import StateCache

STAMP = "2026-09-15T10:00:00+00:00"


def joint_message(**changes):
    fields = dict(name=["j1", "j2"], position=[0.1, -0.2], velocity=[], effort=[])
    return SimpleNamespace(**(fields | changes))


def ready_cache():
    cache = StateCache(stale_after_s=2.0)
    cache.update_joints(joint_message(), 10.0, STAMP)
    cache.update_driver(10.0)
    return cache


def test_requires_both_streams():
    cache = StateCache(stale_after_s=2.0)
    snapshot = cache.snapshot(10.0)
    assert snapshot["status"] == "unavailable"
    assert snapshot["positions"] is None
    assert snapshot["observed_at"] is None
    cache.update_joints(joint_message(), 10.0, STAMP)
    assert cache.snapshot(10.0)["status"] == "unavailable"
    cache.update_driver(10.0)
    snapshot = cache.snapshot(10.5)
    assert snapshot["status"] == "ok"
    assert snapshot["positions"] == [0.1, -0.2]
    assert snapshot["observed_at"] == STAMP
    assert snapshot["robot_state_received"] is True
    assert snapshot["age_s"] == {"joints": 0.5, "driver": 0.5}
    assert "velocities" not in snapshot


@pytest.mark.parametrize("refreshed", ["joints", "driver"])
def test_each_stream_must_remain_fresh(refreshed):
    cache = ready_cache()
    if refreshed == "joints":
        cache.update_joints(joint_message(), 12.1, STAMP)
    else:
        cache.update_driver(12.1)
    snapshot = cache.snapshot(12.1)
    assert snapshot["status"] == "stale"
    assert ("driver" if refreshed == "joints" else "joints") in snapshot["reason"]


def test_freshness_threshold_is_inclusive():
    assert ready_cache().snapshot(12.0)["status"] == "ok"


@pytest.mark.parametrize(
    "changes",
    [
        {"name": [], "position": []},
        {"name": ["j1", "j1"]},
        {"name": ["", "j2"]},
        {"name": [" ", "j2"]},
        {"name": [None, "j2"]},
        {"name": "j1", "position": [1, 2]},
        {"position": []},
        {"position": [1]},
        {"position": [float("nan"), 2]},
        {"position": [float("inf"), 2]},
        {"position": [float("-inf"), 2]},
        {"position": [True, 2]},
        {"position": ["0.1", 2]},
        {"position": [None, 2]},
        {"position": None},
    ],
)
def test_bad_joint_message_invalidates_previous_ok_and_can_recover(changes):
    cache = ready_cache()
    cache.update_joints(joint_message(**changes), 10.2, STAMP)
    assert cache.snapshot(10.3)["status"] == "invalid"
    assert cache.snapshot(10.3)["reason"]
    cache.update_joints(joint_message(), 10.4, STAMP)
    assert cache.snapshot(10.5)["status"] == "ok"


def test_missing_required_fields_never_create_zero_positions():
    cache = StateCache(2.0)
    cache.update_joints(SimpleNamespace(), 10.0, STAMP)
    assert cache.snapshot(10.0)["status"] == "invalid"
    assert cache.snapshot(10.0)["positions"] is None


@pytest.mark.parametrize("field,output", [("velocity", "velocities"), ("effort", "efforts")])
@pytest.mark.parametrize("values", [[1], [1, float("nan")], None, "12"])
def test_invalid_optional_fields_are_dropped_with_warning(field, output, values):
    cache = ready_cache()
    cache.update_joints(joint_message(**{field: values}), 10.0, STAMP)
    snapshot = cache.snapshot(10.0)
    assert snapshot["status"] == "ok"
    assert output not in snapshot
    assert any(output in warning for warning in snapshot["warnings"])


def test_optional_fields_and_snapshots_are_copied():
    cache = StateCache(2.0)
    message = joint_message(velocity=[1, 2], effort=[3, 4])
    cache.update_joints(message, 10.0, STAMP)
    cache.update_driver(10.0)
    message.position[0] = 999
    first = cache.snapshot(10.0)
    assert first["positions"] == [0.1, -0.2]
    assert first["velocities"] == [1.0, 2.0]
    first["positions"][0] = 999
    first["efforts"].clear()
    assert cache.snapshot(10.0)["positions"] == [0.1, -0.2]
    assert cache.snapshot(10.0)["efforts"] == [3.0, 4.0]


@pytest.mark.parametrize("timeout", [0, -1, float("nan"), float("inf"), True])
def test_invalid_staleness_configuration(timeout):
    with pytest.raises(ValueError):
        StateCache(timeout)


@pytest.mark.parametrize("timestamp", [float("nan"), float("inf")])
def test_non_finite_receipt_time_is_rejected(timestamp):
    cache = ready_cache()
    with pytest.raises(ValueError):
        cache.update_driver(timestamp)
    with pytest.raises(ValueError):
        cache.update_joints(joint_message(), timestamp, STAMP)


def test_clock_regression_is_not_fresh_data():
    cache = ready_cache()
    assert cache.snapshot(9.0)["status"] == "failed"
    cache = ready_cache()
    cache.update_driver(9.0)
    assert cache.snapshot(10.0)["status"] == "failed"
