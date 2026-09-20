"""Actual adapter lifecycle against in-memory ROS factories."""

import ast
import os
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from robots.lynsense.ros2_adapter import Ros2StateAdapter


def wait_for(predicate):
    deadline = time.monotonic() + 1.0
    while not predicate() and time.monotonic() < deadline:
        time.sleep(0.002)
    assert predicate()


def test_construction_is_inert_and_close_is_idempotent(fake_ros):
    adapter = Ros2StateAdapter()
    assert fake_ros.events == []
    assert adapter.get_snapshot()["status"] == "unavailable"
    adapter.close()
    adapter.close()
    assert adapter.get_snapshot()["status"] == "closed"
    assert fake_ros.events == []
    with pytest.raises(RuntimeError, match="closed"):
        adapter.connect()


def test_two_subscriptions_no_controls_and_owned_cleanup(fake_ros):
    adapter = Ros2StateAdapter()
    before = os.environ.copy()
    try:
        assert adapter.connect()["status"] == "ok"
        assert adapter.connect()["status"] == "ok"
        assert len(fake_ros.executors) == 1
        assert adapter.get_snapshot()["positions"] == [0.1, -0.2]
        assert set(fake_ros.subscriptions) == {"/right_xarm/joint_states", "/right_xarm/robot_states"}
        assert fake_ros.node_options == dict(enable_rosout=False, start_parameter_services=False, use_global_arguments=False)
        assert fake_ros.init_options == dict(args=[], signal_handler_options="no_signals")
        for _, _, qos in fake_ros.subscriptions.values():
            assert vars(qos) == dict(reliability="best_effort", durability="volatile", history="keep_last", depth=10)
    finally:
        adapter.close()
    adapter.close()
    assert fake_ros.events.count("context_shutdown") == 1
    assert fake_ros.events[-3:] == ["executor_shutdown", "node_destroy", "context_shutdown"]
    assert os.environ == before


@pytest.mark.parametrize("domain", [None, "", "0", "abc", "-1", "233"])
def test_domain_mismatch_never_initializes_ros(fake_ros, monkeypatch, domain):
    if domain is None:
        monkeypatch.delenv("ROS_DOMAIN_ID")
    else:
        monkeypatch.setenv("ROS_DOMAIN_ID", domain)
    adapter = Ros2StateAdapter()
    result = adapter.connect()
    assert result["status"] == "failed"
    assert "ROS_DOMAIN_ID" in result["reason"]
    assert fake_ros.events == []


@pytest.mark.parametrize("kwargs", [
    {"expected_domain_id": -1}, {"expected_domain_id": 233}, {"expected_domain_id": True},
    {"joint_topic": "relative"}, {"joint_topic": "/invalid space"},
    {"joint_topic": "/1bad"}, {"joint_topic": "/a//b"}, {"joint_topic": "/a/"},
    {"joint_topic": "/a__b"}, {"joint_topic": "/a", "robot_topic": "/a"},
    {"connect_timeout_s": 0}, {"connect_timeout_s": float("inf")},
    {"stale_after_s": float("nan")},
])
def test_invalid_configuration_fails_before_ros_import(kwargs):
    with pytest.raises(ValueError):
        Ros2StateAdapter(**kwargs)


def test_external_context_is_not_shutdown(fake_ros):
    fake_ros.active = True
    adapter = Ros2StateAdapter()
    assert adapter.connect()["status"] == "failed"
    adapter.close()
    assert fake_ros.active
    assert fake_ros.events == []


@pytest.mark.parametrize("stage,owns_context", [
    ("init", False), ("after_init", True), ("node", True),
    ("subscription_2", True), ("executor", True), ("add_node", True),
])
def test_partial_initialization_failure_releases_only_owned_resources(fake_ros, stage, owns_context):
    fake_ros.fail_at = stage
    adapter = Ros2StateAdapter()
    result = adapter.connect()
    assert result["status"] == "failed"
    assert stage in result["reason"]
    assert fake_ros.events.count("context_shutdown") == int(owns_context)
    assert not fake_ros.active


def test_missing_message_package_is_an_environment_error(fake_ros, monkeypatch):
    monkeypatch.setitem(sys.modules, "xarm_msgs.msg", None)
    result = Ros2StateAdapter().connect()
    assert result["status"] == "failed"
    assert "xarm_msgs" in result["reason"]
    assert "init" not in fake_ros.events


@pytest.mark.parametrize("only_topic", [None, "/right_xarm/joint_states", "/right_xarm/robot_states"])
def test_discovery_or_missing_first_message_times_out(fake_ros, only_topic):
    fake_ros.auto_messages = False
    if only_topic:
        fake_ros.messages.put((only_topic, SimpleNamespace(name=["j1"], position=[0.0])))
    adapter = Ros2StateAdapter(connect_timeout_s=0.04)
    result = adapter.connect()
    assert result["status"] == "failed"
    assert "timed out" in result["reason"]
    assert fake_ros.events.count("context_shutdown") == 1


def test_stale_and_invalid_samples_are_not_ready(fake_ros):
    adapter = Ros2StateAdapter(stale_after_s=0.025)
    try:
        assert adapter.connect()["status"] == "ok"
        wait_for(lambda: adapter.get_snapshot()["status"] == "stale")
        fake_ros.messages.put(("/right_xarm/joint_states", object()))
        wait_for(lambda: adapter.get_snapshot()["status"] == "invalid")
    finally:
        adapter.close()


def test_executor_failure_is_visible_and_thread_can_be_closed(fake_ros):
    adapter = Ros2StateAdapter()
    assert adapter.connect()["status"] == "ok"
    worker = adapter._thread
    fake_ros.messages.put(RuntimeError("executor failed"))
    try:
        worker.join(1)
        assert not worker.is_alive()
        assert not fake_ros.active
        assert fake_ros.events[-3:] == ["executor_shutdown", "node_destroy", "context_shutdown"]
        wait_for(lambda: adapter.get_snapshot()["status"] == "failed")
        assert "executor failed" in adapter.get_snapshot()["reason"]
    finally:
        adapter.close()
    assert not fake_ros.active


def test_executor_fault_cleanup_racing_owner_close_does_not_deadlock(fake_ros, monkeypatch):
    adapter = Ros2StateAdapter()
    assert adapter.connect()["status"] == "ok"
    worker = adapter._thread
    original_close = adapter.close
    fault_ready = threading.Event()
    allow_cleanup = threading.Event()
    owner_joining = threading.Event()
    errors = []
    original_join = worker.join

    def observed_join(timeout=None):
        owner_joining.set()
        return original_join(timeout)

    def controlled_close():
        if threading.current_thread() is worker:
            fault_ready.set()
            assert allow_cleanup.wait(1)
        original_close()

    def close_from_owner():
        try:
            original_close()
        except Exception as exc:
            errors.append(exc)

    monkeypatch.setattr(adapter, "close", controlled_close)
    monkeypatch.setattr(worker, "join", observed_join)
    fake_ros.messages.put(RuntimeError("executor fault during close"))
    closer = threading.Thread(target=close_from_owner)
    try:
        assert fault_ready.wait(1)
        closer.start()
        assert owner_joining.wait(1)
        allow_cleanup.set()
        closer.join(1)
        assert not closer.is_alive()
        assert not errors
        assert not fake_ros.active
        assert fake_ros.events.count("context_shutdown") == 1
        assert "executor fault" in adapter.get_snapshot()["reason"]
    finally:
        allow_cleanup.set()
        if closer.ident is not None:
            closer.join(3)
        original_close()


def test_automatic_fault_cleanup_failure_is_visible_and_retryable(fake_ros):
    adapter = Ros2StateAdapter()
    assert adapter.connect()["status"] == "ok"
    worker = adapter._thread
    fake_ros.shutdown_result = False
    fake_ros.messages.put(RuntimeError("runtime fault"))
    try:
        worker.join(1)
        assert not worker.is_alive()
        assert "executor_shutdown" in fake_ros.events
        assert "node_destroy" not in fake_ros.events
        snapshot = adapter.get_snapshot()
        assert snapshot["status"] == "failed"
        assert "runtime fault" in snapshot["reason"]
        assert "cleanup failed" in snapshot["reason"]
    finally:
        fake_ros.shutdown_result = True
        adapter.close()
    assert not fake_ros.active
    assert "runtime fault" in adapter.get_snapshot()["reason"]


def test_close_interrupts_connection_wait(fake_ros):
    fake_ros.auto_messages = False
    adapter = Ros2StateAdapter(connect_timeout_s=5)
    result = []
    connector = threading.Thread(target=lambda: result.append(adapter.connect()))
    connector.start()
    try:
        assert fake_ros.spin_entered.wait(1)
        adapter.close()
        connector.join(1)
        assert not connector.is_alive()
        assert result[0]["status"] == "failed"
        assert fake_ros.events.count("context_shutdown") == 1
    finally:
        adapter.close()
        connector.join(1)


def test_close_timeout_does_not_destroy_a_live_node_and_can_be_retried(fake_ros, monkeypatch):
    adapter = Ros2StateAdapter()
    assert adapter.connect()["status"] == "ok"
    fake_ros.spin_entered.clear()
    fake_ros.spin_release = threading.Event()
    assert fake_ros.spin_entered.wait(1)
    monkeypatch.setattr(adapter, "_CLOSE_TIMEOUT_S", 0.03)
    try:
        with pytest.raises(RuntimeError, match="thread"):
            adapter.close()
        assert "node_destroy" not in fake_ros.events
        assert adapter.get_snapshot()["status"] == "failed"
    finally:
        fake_ros.spin_release.set()
        monkeypatch.setattr(adapter, "_CLOSE_TIMEOUT_S", 2.0)
        adapter.close()
    assert not fake_ros.active


def test_executor_shutdown_failure_is_not_reported_as_closed(fake_ros):
    adapter = Ros2StateAdapter()
    assert adapter.connect()["status"] == "ok"
    fake_ros.shutdown_result = False
    try:
        with pytest.raises(RuntimeError, match="executor"):
            adapter.close()
        assert "node_destroy" not in fake_ros.events
    finally:
        fake_ros.shutdown_result = True
        adapter.close()
    assert adapter.get_snapshot()["status"] == "closed"


def test_thread_start_failure_cleans_up_created_ros_resources(fake_ros, monkeypatch):
    def fail_start(self):
        raise RuntimeError("thread start failed")

    monkeypatch.setattr(threading.Thread, "start", fail_start)
    adapter = Ros2StateAdapter()
    result = adapter.connect()
    assert result["status"] == "failed"
    assert "thread start failed" in result["reason"]
    assert adapter.get_snapshot()["status"] == "closed"
    assert not fake_ros.active
    assert "node_destroy" in fake_ros.events


@pytest.mark.parametrize("replace_default", [False, True])
def test_context_shutdown_failure_can_retry_owned_context(fake_ros, replace_default):
    adapter = Ros2StateAdapter()
    assert adapter.connect()["status"] == "ok"
    owned = fake_ros.context
    fake_ros.fail_at = "context_shutdown"
    try:
        with pytest.raises(RuntimeError, match="context_shutdown"):
            adapter.close()
        assert owned.active
        assert adapter.get_snapshot()["status"] == "failed"
        replacement = None
        if replace_default:
            fake_ros.get_default_context(shutting_down=True)
            replacement = fake_ros.get_default_context()
            replacement.active = True
        fake_ros.fail_at = None
        adapter.close()
        assert not owned.active
        assert fake_ros.events.count("context_shutdown") == 2
        assert adapter.get_snapshot()["status"] == "closed"
        if replacement is not None:
            assert fake_ros.get_default_context() is replacement
            assert replacement.active
    finally:
        # Keep even the pre-fix red test entirely in memory and resource-free.
        fake_ros.fail_at = None
        fake_ros.default_context = owned
        adapter.close()
        for context in fake_ros.contexts:
            context.active = False


def test_new_adapter_can_connect_after_previous_context_shutdown(fake_ros):
    first = Ros2StateAdapter()
    assert first.connect()["status"] == "ok"
    original_context = fake_ros.get_default_context()
    first.close()
    second = Ros2StateAdapter()
    try:
        assert second.connect()["status"] == "ok"
        assert fake_ros.get_default_context() is not original_context
    finally:
        second.close()
    assert not fake_ros.active


def test_production_adapter_has_no_control_or_subprocess_factories():
    import robots.lynsense.ros2_adapter as module
    tree = ast.parse(Path(module.__file__).read_text())
    calls = {node.func.attr for node in ast.walk(tree) if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)}
    assert calls.isdisjoint({"create_client", "create_service", "create_publisher", "Popen", "system", "send_goal_async"})
