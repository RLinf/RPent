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

import json

import numpy as np

from flywheel.episode import EpisodeWriter, validate_episode
from flywheel.export import export_lerobot


def _obs(value: int) -> dict:
    return {
        "main_images": np.full((256, 256, 3), value, np.uint8),
        "wrist_images": np.full((256, 256, 3), value + 1, np.uint8),
        "states": np.full(8, value, np.float32),
        "task_descriptions": "put the bowl on the plate",
    }


def test_success_episode_keeps_aligned_training_prefix(tmp_path):
    initial = _obs(1)
    writer = EpisodeWriter(
        tmp_path,
        suite="libero_object",
        task_id=2,
        seed=7,
        initial_observation=initial,
    )
    initial["main_images"].fill(99)

    writer.begin_primitive("move_to")
    writer.add_transition(np.zeros(7), _obs(2), 0, False, False)
    writer.end_primitive()

    writer.begin_primitive("pi0_pick")
    proposal = np.ones((2, 7), np.float32)
    vla_id = writer.add_proposal("pick up the bowl", proposal)
    proposal.fill(99)
    writer.add_transition(
        np.ones(7), _obs(3), 1, True, False, vla_id=vla_id, proposal_index=0
    )
    writer.add_transition(
        np.full(7, 2),
        _obs(4),
        0,
        False,
        False,
        vla_id=vla_id,
        proposal_index=1,
    )
    writer.end_primitive()

    path = writer.finalize()
    metadata = validate_episode(path)
    assert metadata["is_success"] is True
    assert metadata["step_count"] == 3
    assert metadata["training_step_count"] == 2
    assert metadata["primitive_names"] == ["move_to", "pi0_pick"]

    with np.load(path / "transitions.npz", allow_pickle=False) as data:
        assert data["main_images"].shape[0] == 4
        assert data["main_images"][0, 0, 0, 0] == 1
        np.testing.assert_array_equal(data["action_source"], [0, 1, 1])
        np.testing.assert_array_equal(data["primitive_id"], [0, 1, 1])
        np.testing.assert_array_equal(data["proposal_index"], [-1, 0, 1])
    with np.load(path / "proposals.npz", allow_pickle=False) as proposals:
        assert proposals["actions"][0, 0, 0] == 1
        assert proposals["created_step"].tolist() == [1]

    assert json.loads((path / "episode.json").read_text())["stop_reason"] == (
        "env_terminated"
    )
    assert not path.with_name(f"{path.name}.partial").exists()
    assert writer.finalize() == path


def test_failed_episode_has_no_training_prefix(tmp_path):
    writer = EpisodeWriter(
        tmp_path,
        suite="libero_object",
        task_id=2,
        seed=8,
        initial_observation=_obs(1),
    )
    writer.begin_primitive("move_to")
    writer.add_transition(np.zeros(7), _obs(2), 0, False, True)

    metadata = validate_episode(writer.finalize())
    assert metadata["is_success"] is False
    assert metadata["training_step_count"] == 0
    assert metadata["stop_reason"] == "env_truncated"


def test_export_uses_only_success_prefix(tmp_path):
    success = EpisodeWriter(
        tmp_path,
        suite="libero_object",
        task_id=2,
        seed=1,
        initial_observation=_obs(1),
    )
    success.begin_primitive("pi0_pick")
    success.add_transition(np.ones(7), _obs(2), 1, True, False)
    success.add_transition(np.full(7, 2), _obs(3), 0, False, False)
    success.finalize()

    failure = EpisodeWriter(
        tmp_path,
        suite="libero_object",
        task_id=2,
        seed=2,
        initial_observation=_obs(4),
    )
    failure.begin_primitive("move_to")
    failure.add_transition(np.zeros(7), _obs(5), 0, False, True)
    failure.finalize()

    report = export_lerobot(
        tmp_path, suite="libero_object", task_id=2, dataset_id="test"
    )
    assert report["episode_count"] == 1
    assert report["frame_count"] == 1

    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    dataset = LeRobotDataset(report["repo_id"], root=report["dataset_path"])
    assert len(dataset) == 1
    assert tuple(dataset[0]["actions"].shape) == (7,)
    assert dataset[0]["task"] == "put the bowl on the plate"
