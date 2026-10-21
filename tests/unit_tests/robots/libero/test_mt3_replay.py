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

from types import SimpleNamespace

import numpy as np

from robots.libero.env_server import _configure_fixture_seed, _seed_fixture_sampler
from robots.libero.mt3.core import POS_SCALE, EnvHooks, MT3Config, MT3Runner


def test_native_rpent_policy_preserves_seed_method(monkeypatch):
    """Benchmark evaluation must leave RPent's native seed behavior intact."""
    monkeypatch.setenv("RPENT_MT3_FIXTURE_SEED_POLICY", "rpent")
    calls = []
    seed = calls.append
    env = SimpleNamespace(seed=seed)
    assert _configure_fixture_seed(env) is env
    assert env.seed is seed
    env.seed(9)
    assert calls == [9]


def test_fixture_placement_repeats_after_reseeding():
    """Both placement samplers and the wrapped environment receive the seed."""
    state = np.random.get_state()
    try:
        received = []
        env = _seed_fixture_sampler(SimpleNamespace(seed=received.append))
        env.seed(17)
        first = np.random.uniform(size=12)
        np.random.uniform(size=29)
        env.seed(17)
        np.testing.assert_array_equal(first, np.random.uniform(size=12))
        assert received == [17, 17]
    finally:
        np.random.set_state(state)


def test_continuous_replay_keeps_transit_frames_and_blends_reference_changes():
    """All source frames run, including transit between independent objects."""
    frames = 24
    poses = np.repeat(np.eye(4)[None], frames, axis=0)
    poses[:, 0, 3] = np.arange(frames) * 0.001
    segments = [
        {"ref": "a", "start": 0, "c": 8, "end": 12},
        {"ref": "b", "start": 12, "c": 21, "end": 24},
    ]
    lib = SimpleNamespace(
        frames=lambda _: slice(0, frames),
        eef=poses,
        target=poses.copy(),
        gripper=np.full(frames, -1.0),
        actions=np.zeros((frames, 7)),
        segments=[segments],
        pose={name: np.repeat(np.eye(4)[None], frames, axis=0) for name in ("a", "b")},
    )
    live = {"a": np.eye(4), "b": np.eye(4)}
    live["b"][1, 3] = 0.02
    current = poses[0].copy()
    achieved = []

    def step(action):
        current[:3, 3] += action[:3] * POS_SCALE
        achieved.append(current.copy())

    hooks = EnvHooks(
        step,
        lambda: current.copy(),
        lambda names: {k: live[k] for k in names},
        lambda: False,
    )
    runner = MT3Runner(lib, 0, MT3Config(mode="continuous", radius=0.004), hooks)
    for k in range(2):
        runner.run_segment(k)
    assert len(achieved) == frames
    np.testing.assert_allclose(np.array(achieved)[:12], poses[:12], atol=1e-12)
    assert achieved[12][1, 3] == 0.0
    assert 0.0 < achieved[14][1, 3] < 0.02
    assert achieved[21][1, 3] == 0.02
    assert all(log["align_steps"] == 0 for log in runner.logs)
