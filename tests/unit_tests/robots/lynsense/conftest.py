"""In-memory ROS and network guards shared by the Lynsense offline tests."""

import queue
import socket
import sys
import threading
from types import ModuleType, SimpleNamespace

import pytest


def joint_message():
    return SimpleNamespace(name=["j1", "j2"], position=[0.1, -0.2], velocity=[], effort=[])


class FakeRos:
    def __init__(self, monkeypatch):
        self.events = []
        self.subscriptions = {}
        self.messages = queue.Queue()
        self.contexts = []
        self.default_context = None
        self.context = self.get_default_context()
        self.fail_at = None
        self.auto_messages = True
        self.shutdown_result = True
        self.spin_entered = threading.Event()
        self.spin_release = None
        self.node_options = {}
        self.init_options = {}
        self.executors = []
        self.nodes = []

        def module(name, **attrs):
            mod = ModuleType(name)
            mod.__dict__.update(attrs)
            monkeypatch.setitem(sys.modules, name, mod)
            return mod

        qos = module(
            "rclpy.qos",
            QoSProfile=lambda **kwargs: SimpleNamespace(**kwargs),
            ReliabilityPolicy=SimpleNamespace(BEST_EFFORT="best_effort"),
            DurabilityPolicy=SimpleNamespace(VOLATILE="volatile"),
            HistoryPolicy=SimpleNamespace(KEEP_LAST="keep_last"),
        )
        signals = module("rclpy.signals", SignalHandlerOptions=SimpleNamespace(NO="no_signals"))
        executors = module("rclpy.executors", SingleThreadedExecutor=self.new_executor)
        module(
            "rclpy", init=self.init, shutdown=self.shutdown,
            get_default_context=self.get_default_context, create_node=self.new_node,
            qos=qos, executors=executors, signals=signals,
        )
        sensor = module("sensor_msgs.msg", JointState=type("JointState", (), {}))
        module("sensor_msgs", msg=sensor)
        xarm = module("xarm_msgs.msg", RobotMsg=type("RobotMsg", (), {}))
        module("xarm_msgs", msg=xarm)

    def step(self, name):
        self.events.append(name)
        if self.fail_at == name:
            raise RuntimeError(f"injected failure: {name}")

    @property
    def active(self):
        return any(context.active for context in self.contexts)

    @active.setter
    def active(self, value):
        self.get_default_context().active = value

    def get_default_context(self, *, shutting_down=False):
        if self.default_context is None:
            context = SimpleNamespace(active=False, initialized=False)
            context.ok = lambda: context.active
            self.default_context = context
            self.contexts.append(context)
        context = self.default_context
        if shutting_down:
            self.default_context = None
        return context

    def init(self, **kwargs):
        self.init_options = kwargs
        self.step("init")
        context = self.get_default_context()
        if context.initialized:
            raise RuntimeError("Context.init() must only be called once")
        context.initialized = True
        context.active = True
        self.step("after_init")

    def shutdown(self, *, context=None, uninstall_handlers):
        assert uninstall_handlers is False
        if context is None:
            context = self.get_default_context(shutting_down=True)
        self.step("context_shutdown")
        context.active = False

    def new_node(self, name, **kwargs):
        self.node_options = kwargs
        self.step("node")
        node = SimpleNamespace(
            create_subscription=self.subscribe,
            create_client=self.forbidden,
            create_publisher=self.forbidden,
            create_service=self.forbidden,
            destroy_node=lambda: self.step("node_destroy"),
        )
        self.nodes.append(node)
        return node

    def forbidden(self, *args, **kwargs):
        raise AssertionError("attempted a control ROS factory")

    def subscribe(self, msg_type, topic, callback, qos):
        self.step("subscription_" + str(len(self.subscriptions) + 1))
        self.subscriptions[topic] = (msg_type, callback, qos)
        return object()

    def new_executor(self, **kwargs):
        self.step("executor")
        owner = self

        class Executor:
            def add_node(self, node):
                owner.step("add_node")
                if owner.auto_messages:
                    for topic in owner.subscriptions:
                        owner.messages.put((topic, joint_message() if topic.endswith("joint_states") else object()))

            def spin_once(self, *, timeout_sec):
                assert 0 <= timeout_sec <= 0.1
                owner.spin_entered.set()
                if owner.spin_release is not None:
                    owner.spin_release.wait()
                try:
                    item = owner.messages.get(timeout=min(timeout_sec, 0.02))
                except queue.Empty:
                    return
                if isinstance(item, Exception):
                    raise item
                topic, message = item
                if topic is not None:
                    owner.subscriptions[topic][1](message)

            def wake(self):
                owner.messages.put((None, None))

            def shutdown(self, *, timeout_sec):
                assert timeout_sec >= 0
                owner.step("executor_shutdown")
                return owner.shutdown_result

        executor = Executor()
        self.executors.append(executor)
        return executor


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def reject(*args, **kwargs):
        raise AssertionError("Lynsense unit tests must not access the network")

    monkeypatch.setattr(socket.socket, "connect", reject)
    monkeypatch.setattr(socket.socket, "connect_ex", reject)
    monkeypatch.setattr(socket, "getaddrinfo", reject)


@pytest.fixture
def fake_ros(monkeypatch):
    monkeypatch.setenv("ROS_DOMAIN_ID", "3")
    return FakeRos(monkeypatch)
