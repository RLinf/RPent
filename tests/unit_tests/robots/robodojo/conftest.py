# Copyright 2026 The RPent Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Offline agent adapter fixtures, including environments without RLinf."""

import importlib
import sys
from copy import deepcopy
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

from robots.robodojo.access import ARX_X5_JOINT_FIELDS, CAMERAS
from robots.robodojo.env_client import RoboDojoEnvClient
from robots.robodojo.env_facade import RoboDojoEnvFacade


@pytest.fixture
def agent_module(monkeypatch):
    name = "rlinf.envs.sim.robodojo.robodojo_env"
    try:
        importlib.import_module(name)
    except ModuleNotFoundError:
        # The simulator adapter is optional in RPent's base test environment.
        stub = ModuleType(name)

        class RoboDojoEnv:
            def close(self, clear_cache=True):
                if not self._closed:
                    self.venv.close(clear_cache)
                    self._closed = True

        stub.RoboDojoEnv = RoboDojoEnv
        monkeypatch.setitem(sys.modules, name, stub)
    monkeypatch.delitem(sys.modules, "robots.robodojo.rlinf_env", raising=False)
    module = importlib.import_module("robots.robodojo.rlinf_env")
    yield module
    sys.modules.pop("robots.robodojo.rlinf_env", None)


@pytest.fixture
def make_agent(agent_module):
    def create(native, *, meta=None, slot=None, recorder=None):
        agent = object.__new__(agent_module.RoboDojoAgentEnv)
        agent.meta = meta or {}
        agent.eval_fair = agent.meta.get("mode") == "eval-fair"
        agent._closed = False
        agent.recorder = recorder or SimpleNamespace(
            record=lambda obs: None, close=lambda: []
        )
        agent.bottle_mon = agent_module._SafetyMonitor()
        agent.venv = SimpleNamespace(envs=[slot or SimpleNamespace(env=native)])
        return agent

    return create


def _make_native():
    robots = [
        SimpleNamespace(
            type="target",
            arm_name=f"{arm}_arm",
            gripper_name=f"{arm}_ee",
            robot_name=arm,
            arm_joint_indices=list(range(6)),
            entity_origin_pose=[0, 0, 0, 1, 0, 0, 0],
            gripper_scale=[-0.01, 0.044],
            gripper_move={"sign": 1},
        )
        for arm in ("left", "right")
    ]
    previous = {
        f"{arm}_arm_joint_state": {"position": [offset] * 6}
        for arm, offset in (("left", 0.4), ("right", 0.7))
    }
    previous.update(
        {
            f"{arm}_ee_joint_state": {"position": [val, val]}
            for arm, val in (("left", -0.01), ("right", 0.017))
        }
    )
    plans = []

    def plan(**kwargs):
        plans.append(kwargs)
        return {"status": "Success", "position": np.zeros((3, 6))}

    manager = SimpleNamespace(
        robot_list=robots,
        robot_key=[
            SimpleNamespace(
                data=SimpleNamespace(joint_pos_limits=np.tile([-3.14, 3.14], (1, 6, 1)))
            )
            for _ in robots
        ],
        control_manager=SimpleNamespace(prev_control=[previous]),
        scene=SimpleNamespace(env_origins=np.array([[1, 2, 3]])),
        process_name=lambda name: name + "_joint_state",
        get_joint=lambda robot, **kw: {0: np.zeros(6)},
        get_real_endpose=lambda robot, **kw: {0: np.array([1, 2, 3, 1, 0, 0, 0])},
        get_end_effector_real_val=lambda robot, **kw: {0: np.array([0.017, 0.017])},
        planner={arm: SimpleNamespace(plan_path=plan) for arm in ("left", "right")},
    )
    return SimpleNamespace(
        robot_manager=manager, take_action_cnt=[0], step_lim=10, end_flag=[False]
    ), plans


@pytest.fixture
def make_native():
    return _make_native


class FakeEnv(RoboDojoEnvClient):
    def __init__(self, limit=20, done_at=None, *, record_vpp2_history=False):
        self.frame = 0
        self.limit = limit
        self.done_at = done_at
        self.closed = False
        self.after_step = lambda: None
        super().__init__(
            SimpleNamespace(call=self._rpc),
            expected_meta={},
            record_vpp2_history=record_vpp2_history,
        )

    def get_obs(self):
        return {
            "instruction": self.get_task_language(),
            "state": {
                "frame": self.frame,
                **{
                    f"{arm}_ee_pose": [0, 0, 1, 1, 0, 0, 0] for arm in ("left", "right")
                },
                **{
                    key: np.full(size, self.frame, dtype=float)
                    for key, size in ARX_X5_JOINT_FIELDS.items()
                },
            },
            "vision": {
                name: {
                    "color": np.full((2, 2, 3), self.frame % 256, np.uint8),
                    "depth": np.ones((2, 2)),
                    "intrinsic_matrix": np.eye(3),
                    "extrinsic_matrix": np.eye(4),
                }
                for name in CAMERAS
            },
        }

    def get_task_language(self):
        return "stack the blocks"

    def get_status(self):
        return {"step": self.frame, "step_limit": self.limit}

    def _native_step(self, action, *, expected_step=None):
        before = self.frame
        if expected_step is not None:
            assert expected_step == before
        if "expected_step" in action:
            assert action["expected_step"] == self.frame + 1
        if "left_arm_joint_state" in action:
            assert action["left_arm_joint_state"][0] == self.frame + 1
        self.frame += 1
        self.after_step()
        return (
            self.get_obs(),
            0.0,
            self.frame == self.done_at,
            {
                "status": self.get_status(),
                "step_before": before,
                "step_after": self.frame,
                "executed_actions": 1,
                "terminated": self.frame == self.done_at,
                "truncated": self.frame >= self.limit,
            },
        )

    def _native_reset(self):
        self.frame = 0
        return self.get_obs()

    def get_motion_state(self):
        return {
            **{f"{arm}_gripper": 0.5 for arm in ("left", "right")},
            **{f"{arm}_eef_pose": [0, 0, 1, 1, 0, 0, 0] for arm in ("left", "right")},
        }

    def plan_arm_path(self, arm, target):
        return {"status": "Failure", "position": []}

    def is_success(self):
        return self.frame == self.done_at

    def _rpc(self, method, *, kwargs=None, **options):
        kwargs = kwargs or {}
        if method == "env.get_env_meta":
            return {}
        if method == "env.reset":
            return self._native_reset()
        if method == "env.close":
            self.closed = True
            return None
        if method == "env.step":
            return self._native_step(
                kwargs["flat_action"], expected_step=kwargs.get("expected_step")
            )
        if method == "env.step_joint_update":
            obs, _, _, info = self._native_step(
                {}, expected_step=kwargs.get("expected_step")
            )
            return {**info, "episode_status": info["status"], "observation": obs}
        raise AssertionError(method)


class EpisodePolicy:
    """Fixed eight-action chunks; counts in direct execution tests use this fixture."""

    def __init__(self):
        self.events = []
        self.step = None

    def reset_session(self):
        self.events.append(("reset",))

    def update_obs(self, obs, step):
        assert obs["instruction"] == "stack the blocks"
        assert obs["state"]["frame"] == step
        self.step = step
        self.events.append(("observe", step))

    def get_action(self):
        self.events.append(("infer", self.step))
        return [{"expected_step": self.step + i} for i in range(1, 9)]

    def end_episode(self):
        self.events.append(("end",))


@pytest.fixture
def fake_env():
    return FakeEnv


@pytest.fixture
def episode_policy():
    return EpisodePolicy


class VPP2Policy:
    def __init__(self, env):
        self.env = env
        self.events = []

    def infer(self, observation, instruction):
        self.events.append(("infer", self.env.frame))
        return [{"expected_step": self.env.frame + i} for i in range(1, 25)]


@pytest.fixture
def policy_runtime():
    from robots.robodojo.primitives import RoboDojoPrimitives

    instances = []

    def build(backend, limit=100, done_at=None, check_cancelled=lambda: None):
        env = FakeEnv(limit, done_at, record_vpp2_history=backend == "vpp2")
        policy = VPP2Policy(env) if backend == "vpp2" else EpisodePolicy()
        primitive = RoboDojoPrimitives(
            env=env,
            vla_client=policy,
            vla_backend=backend,
            check_cancelled=check_cancelled,
        )
        instances.append(primitive)
        return env, policy, primitive

    yield build
    for primitive in instances:
        primitive.close()


@pytest.fixture
def policy_toolkit(monkeypatch, tmp_path):
    from robots.robodojo.toolkit import RoboDojoToolkit
    from rpent.dashboard.events import NullDashboardEventSink
    from rpent.memory import MemoryManager
    from rpent.utils import logging

    monkeypatch.setattr(logging, "_output_dir", tmp_path)
    instances = []

    def build(backend, limit=100, done_at=None, **kwargs):
        env = FakeEnv(limit, done_at, record_vpp2_history=backend == "vpp2")
        policy = VPP2Policy(env) if backend == "vpp2" else EpisodePolicy()
        kit = RoboDojoToolkit(
            primitives_kwargs={
                "env": env,
                "vla_client": policy,
                "vla_backend": backend,
                **kwargs,
            },
            dashboard_events=NullDashboardEventSink(),
            memory=MemoryManager(tmp_path / "memory"),
            state_output_dir=tmp_path / str(len(instances)),
        )
        instances.append(kit)
        return env, policy, kit

    yield build
    for kit in instances:
        kit.close()


@pytest.fixture
def boundary(make_agent, agent_module, monkeypatch, make_native, fake_env):
    native, _ = make_native()
    observed = fake_env()

    def observation(step):
        observed.frame = step
        return observed.get_obs()

    applied = []
    instruction = ["Move the object."]
    agent = make_agent(native)
    agent.get_status = lambda: {
        "step": native.take_action_cnt[0],
        "step_limit": native.step_lim,
        "success": False,
    }
    agent.get_task_language = lambda: instruction[0]
    agent.get_native_obs = lambda: {
        **observation(native.take_action_cnt[0]),
        "instruction": instruction[0],
        "state": {
            **observation(native.take_action_cnt[0])["state"],
            "counter": native.take_action_cnt[0],
        },
    }
    monkeypatch.setattr(agent_module, "_reward_details", lambda *a: {})

    def apply(action, action_type, **kwargs):
        applied.append(deepcopy(action))
        native.take_action_cnt[0] += 1

    agent.venv.envs[0].apply_action = apply

    def reset():
        native.take_action_cnt[0] = 0
        native.end_flag[0] = False
        return agent.get_native_obs()

    agent.reset_episode = reset
    facade = RoboDojoEnvFacade(agent)
    calls = []

    def call(method, *, args=(), kwargs=None, **options):
        calls.append(method)
        return facade._rpc[method](*args, **(kwargs or {}))

    rpc = SimpleNamespace(call=call)
    client = RoboDojoEnvClient(rpc, expected_meta={})
    return SimpleNamespace(
        env=client,
        native=native,
        agent=agent,
        facade=facade,
        rpc=rpc,
        calls=calls,
        applied=applied,
        instruction=instruction,
    )
