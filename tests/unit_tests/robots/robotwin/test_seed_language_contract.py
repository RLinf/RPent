# Copyright 2026 The RPent Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0

"""Regression tests for RoboTwin exact-seed language initialization."""

from __future__ import annotations

import pytest
import torch

rlinf_robotwin = pytest.importorskip("rlinf.envs.robotwin.robotwin_env")
rpent_robotwin = pytest.importorskip("robots.robotwin.rlinf_env")
RoboTwinEnv = rlinf_robotwin.RoboTwinEnv
RoboTwinAgentEnv = rpent_robotwin.RoboTwinAgentEnv


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
