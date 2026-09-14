# Copyright 2026 The RPent Authors.
# Licensed under the Apache License, Version 2.0.

import json
import time
from types import SimpleNamespace

import numpy as np
import pytest

from robots.yam.contracts import env_runtime_contract
from robots.yam.env_client import YamEnvClient
from robots.yam.env_server import YamEnvFacade
from robots.yam.operator_control import write_receipt
from robots.yam.primitives import YamPrimitives
from robots.yam.rlinf_env import YamAgentEnv


class Runtime:
    def __init__(self):
        self.qpos = np.tile([0.1, 0.2, 0.3, 0.0, 0.0, 0.0, 0.5], 2)
        self.commands = []
        self.events = []

    def connect_followers(self):
        self.events.append("connect")

    def hold(self):
        self.events.append("hold")

    def read_state(self):
        return SimpleNamespace(
            as_vector=lambda: self.qpos.copy(),
            left=SimpleNamespace(timestamp_s=10.0),
            right=SimpleNamespace(timestamp_s=11.0),
        )

    def command(self, target):
        self.commands.append(target.copy())
        self.qpos = target.copy()
        return SimpleNamespace(
            accepted=target.copy(), clipped=False, rejection_reason=None
        )

    def close(self):
        self.events.append("close")


class Cameras:
    def open(self):
        pass

    def snapshot(self, *, not_before_monotonic_s=None):
        now = time.monotonic()
        return {
            "snapshot_id": str(now),
            "views": {
                name: SimpleNamespace(
                    rgb=np.full((2, 3, 3), index, dtype=np.uint8),
                    depth=np.ones((2, 3), dtype=np.float32),
                    camera_meta={
                        "name": name,
                        "height": 2,
                        "width": 3,
                        "intrinsic_K": np.eye(3),
                        "cam2world_cv": np.eye(4),
                        "timestamps": {
                            "host_before_monotonic_s": now,
                            "host_after_monotonic_s": now,
                        },
                    },
                )
                for index, name in enumerate(("top", "left", "right"), 1)
            },
        }


class Kinematics:
    def fk(self, joints, gripper):
        transform = np.eye(4)
        transform[0, 3] = joints[0]
        return transform


@pytest.fixture
def env(tmp_path, monkeypatch):
    identity = np.eye(4).tolist()
    extrinsics = tmp_path / "extrinsics.json"
    extrinsics.write_text(
        json.dumps(
            {
                "world_frame": "left_base",
                "top_camera": {
                    "T_base_to_cam": {
                        "via_left_arm": identity,
                        "via_right_arm": identity,
                    }
                },
                "left_wrist": {"T_grasp_to_cam": identity},
                "right_wrist": {"T_grasp_to_cam": identity},
            }
        )
    )
    value = YamAgentEnv(
        {
            "task_name": "place_cube",
            "task_language": "place the cube",
            "seed": 12,
            "max_episode_steps": 100,
            "max_joint_delta_per_step": None,
            "operator_receipt_path": str(tmp_path / "receipt.json"),
            "table_z": -1.0,
            "table_clearance_m": 0.01,
            "extrinsics_path": str(extrinsics),
        },
        runtime=Runtime(),
        cameras=Cameras(),
        kinematics={side: Kinematics() for side in ("left", "right")},
    )
    monkeypatch.setattr(value, "_pace", lambda: None)
    yield value
    value.config.pop("park_on_close", None)
    value.close()
    if value._stop_worker is not None:
        value._stop_worker.join(timeout=2)
        assert not value._stop_worker.is_alive()


@pytest.fixture
def receipt(env):
    def write(event, episode_id=None):
        return write_receipt(
            env.operator_receipt_path,
            episode_id=episode_id or env._episode_id,
            event=event,
            note="test operator",
        )

    return write


@pytest.fixture
def client(env):
    metadata = env_runtime_contract(
        task_name="place_cube", seed=12, max_episode_steps=100
    )
    metadata["execution"]["compact_control"] = True
    facade = YamEnvFacade(env, metadata=metadata)

    class RPC:
        def call(self, method, args=(), kwargs=None, *, timeout_s=None):
            return facade._dispatch(method, args, kwargs or {})

    return YamEnvClient(RPC(), expected_meta=metadata)


@pytest.fixture
def ready_client(client, receipt):
    receipt("ready")
    client.reset()
    return client


@pytest.fixture
def model():
    value = SimpleNamespace(calls=[])

    def predict(obs):
        value.calls.append(obs)
        return np.repeat(obs["states"][:, None, :], 30, axis=1)

    value.predict = predict
    return value


@pytest.fixture
def primitives(ready_client, model):
    return YamPrimitives(env=ready_client, model=model, check_cancelled=lambda: None)


@pytest.fixture
def clock(monkeypatch):
    value = SimpleNamespace(now=100.0)

    def sleep(duration):
        value.now += duration

    monkeypatch.setattr(time, "monotonic", lambda: value.now)
    monkeypatch.setattr(time, "sleep", sleep)
    return value


@pytest.fixture
def toolkit_factory(client, tmp_path):
    from robots.yam.toolkit import YamToolkit
    from rpent.dashboard.events import NullDashboardEventSink
    from rpent.memory.manager import MemoryManager

    created = []

    def make(mode="exploration", model=None):
        memory = MemoryManager(
            tmp_path / "memory",
            memory_access="inbox_write" if mode == "exploration" else "read_only",
            inbox_cell_tag="yam_place_cube_s12" if mode == "exploration" else None,
        )
        toolkit = YamToolkit(
            primitives_kwargs={"env": client, "model": model},
            dashboard_events=NullDashboardEventSink(),
            memory=memory,
            mode=mode,
            attempts_per_session=2,
            state_output_dir=tmp_path / "state",
            run_output_dir=tmp_path / "run",
        )
        created.append(toolkit)
        return toolkit

    yield make
    for toolkit in created:
        toolkit.close()
