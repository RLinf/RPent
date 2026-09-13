# Copyright 2026 The RPent Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0

"""Regression tests for RoboTwin exact-seed language initialization."""

from __future__ import annotations

from threading import Lock
from types import SimpleNamespace

import numpy as np
import pytest
import torch

rlinf_robotwin = pytest.importorskip("rlinf.envs.robotwin.robotwin_env")
rpent_robotwin = pytest.importorskip("robots.robotwin.rlinf_env")
rpent_env_server = pytest.importorskip("robots.robotwin.env_server")
RoboTwinEnv = rlinf_robotwin.RoboTwinEnv
RoboTwinAgentEnv = rpent_robotwin.RoboTwinAgentEnv
RoboTwinEnvFacade = rpent_env_server.RoboTwinEnvFacade


class _FakeEnv:
    def __init__(self, *, actual_seed: int, instruction: str):
        self.actual_seed = actual_seed
        self.instruction = instruction
        self.calls: list[tuple] = []

    def reset(self, *, env_idx, env_seeds):
        self.calls.append(("reset", list(env_idx), list(env_seeds)))
        observation = {"head_camera": np.zeros((1, 2, 2, 3), dtype=np.uint8)}
        info = {
            "episode_status": {
                "eval_success": False,
                "take_action_cnt": 0,
                "step_lim": 10000,
                "actual_seed": self.actual_seed,
            }
        }
        return observation, info

    def get_task_language(self, env_id):
        self.calls.append(("get_task_language", env_id))
        return self.instruction

    def set_task_language(self, instruction, env_id):
        self.calls.append(("set_task_language", instruction, env_id))
        self.instruction = instruction


class _FakeTask:
    def __init__(self, instruction: str):
        self.instruction = instruction

    def set_instruction(self, instruction: str):
        self.instruction = instruction


def _agent_env(*, initial_env_seeds, num_envs=1):
    env = RoboTwinAgentEnv.__new__(RoboTwinAgentEnv)
    env.cfg = {"initial_env_seeds": initial_env_seeds}
    env.num_envs = num_envs
    env.seed = 7
    return env


def test_requested_seeds_initialize_vector_env_and_language_prewalk():
    env = _agent_env(initial_env_seeds=[100003, 100007], num_envs=2)

    env._init_reset_state_ids()

    assert torch.equal(env.reset_state_ids, torch.tensor([100003, 100007]))
    assert env.success_seeds is None
    assert env._current_seed_index == 0


def test_initial_seed_count_must_match_environment_count():
    env = _agent_env(initial_env_seeds=[100003], num_envs=2)

    with pytest.raises(ValueError, match="expected 2, got 1"):
        env._init_reset_state_ids()


def test_initialization_falls_back_to_rlinf_without_requested_seeds(monkeypatch):
    env = _agent_env(initial_env_seeds=None)
    calls = []

    monkeypatch.setattr(
        RoboTwinEnv,
        "_init_reset_state_ids",
        lambda self: calls.append(self),
    )

    env._init_reset_state_ids()

    assert calls == [env]


def test_standard_evaluation_rebinds_language_from_seed_table():
    env = _FakeEnv(actual_seed=100003, instruction="stale prewalk language")
    facade = RoboTwinEnvFacade(
        env,
        metadata={
            "seed": 100003,
            "task_name": "adjust_bottle",
            "task_config": "demo_randomized",
        },
    )

    _, info = facade.reset()

    expected = "Pick the medium green bottle upright from the table"
    assert env.calls == [
        ("reset", [0], [100003]),
        ("set_task_language", expected, 0),
        ("get_task_language", 0),
    ]
    assert info["instruction"] == expected
    assert info["instruction_source"] == "evaluation_seed_table"


def test_seed_outside_evaluation_table_keeps_native_language():
    env = _FakeEnv(actual_seed=999999, instruction="native custom language")
    facade = RoboTwinEnvFacade(
        env,
        metadata={
            "seed": 999999,
            "task_name": "adjust_bottle",
            "task_config": "demo_randomized",
        },
    )

    _, info = facade.reset()

    assert env.calls == [
        ("reset", [0], [999999]),
        ("get_task_language", 0),
    ]
    assert info["instruction"] == "native custom language"
    assert info["instruction_source"] == "native"


def test_rebound_language_updates_all_native_instruction_stores():
    task = _FakeTask("stale task language")
    sub_env = SimpleNamespace(
        lock=Lock(),
        instruction="stale sub-env language",
        args={"instruction": "stale args language"},
        task=task,
    )
    env = RoboTwinAgentEnv.__new__(RoboTwinAgentEnv)
    env._sub_env = lambda env_id: sub_env

    env.set_task_language("published evaluation language")

    assert sub_env.instruction == "published evaluation language"
    assert sub_env.args["instruction"] == "published evaluation language"
    assert task.instruction == "published evaluation language"
