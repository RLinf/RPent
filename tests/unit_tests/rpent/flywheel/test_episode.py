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
import pytest

from robots.libero.flywheel import LIBERO_SPEC, create_episode_writer, export_options
from rpent.flywheel.episode import EpisodeWriter, validate_episode
from rpent.flywheel.export import export_lerobot


def _obs(value: int) -> dict:
    return {
        "main_images": np.full((256, 256, 3), value, np.uint8),
        "wrist_images": np.full((256, 256, 3), value + 1, np.uint8),
        "states": np.full(8, value, np.float32),
        "task_descriptions": "put the bowl on the plate",
    }


def test_success_episode_keeps_aligned_training_prefix(tmp_path):
    initial = _obs(1)
    writer = create_episode_writer(
        {"root": tmp_path, "suite": "libero_object", "task_id": 2, "seed": 7},
        initial,
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
    metadata = validate_episode(path, spec=LIBERO_SPEC)
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
    writer = create_episode_writer(
        {"root": tmp_path, "suite": "libero_object", "task_id": 2, "seed": 8},
        _obs(1),
    )
    writer.begin_primitive("move_to")
    writer.add_transition(np.zeros(7), _obs(2), 0, False, True)

    metadata = validate_episode(writer.finalize(), spec=LIBERO_SPEC)
    assert metadata["is_success"] is False
    assert metadata["training_step_count"] == 0
    assert metadata["stop_reason"] == "env_truncated"


def test_export_uses_only_success_prefix(tmp_path):
    pytest.importorskip("lerobot")

    success = create_episode_writer(
        {"root": tmp_path, "suite": "libero_object", "task_id": 2, "seed": 1},
        _obs(1),
    )
    success.begin_primitive("pi0_pick")
    success.add_transition(np.ones(7), _obs(2), 1, True, False)
    success.add_transition(np.full(7, 2), _obs(3), 0, False, False)
    success.finalize()

    failure = create_episode_writer(
        {"root": tmp_path, "suite": "libero_object", "task_id": 2, "seed": 2},
        _obs(4),
    )
    failure.begin_primitive("move_to")
    failure.add_transition(np.zeros(7), _obs(5), 0, False, True)
    failure.finalize()

    report = export_lerobot(
        **export_options(tmp_path, suite="libero_object", task_id=2),
        spec=LIBERO_SPEC,
        dataset_id="test",
    )
    assert report["episode_count"] == 1
    assert report["frame_count"] == 1

    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    dataset = LeRobotDataset(report["repo_id"], root=report["dataset_path"])
    assert len(dataset) == 1
    assert tuple(dataset[0]["actions"].shape) == (7,)
    assert dataset[0]["task"] == "put the bowl on the plate"


def test_generic_rules_control_collection_validation_and_export(tmp_path):
    pytest.importorskip("lerobot")
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    spec = {
        "arrays": {
            "camera": {"shape": (4, 5, 3), "dtype": "uint8"},
            "joints": {"shape": (3,), "dtype": "float64"},
            "actions": {"shape": (2,), "dtype": "float32"},
        },
        "export_fields": {"image": "camera", "state": "joints", "actions": "actions"},
        "image_fields": ("camera",),
        "fps": 5,
        "robot_type": "test",
        "success_mask": lambda data: np.asarray(data["rewards"]) > 0.5,
    }
    obs = {"camera": np.zeros((4, 5, 3), np.uint8), "joints": np.ones(3)}
    writer = EpisodeWriter(
        tmp_path / "raw",
        metadata={"experiment": "unit", "task_language": "move"},
        initial_observation=obs,
        spec=spec,
    )
    writer.begin_primitive("policy")
    actions = np.ones((3, 2), np.float32)
    chunk = writer.add_proposal("move", actions)
    for index in range(3):
        writer.add_transition(
            actions[index],
            obs,
            float(index == 1),
            False,
            False,
            vla_id=chunk,
            proposal_index=index,
        )
    path = writer.finalize()
    metadata = validate_episode(path, spec=spec)
    assert metadata["is_success"] is True
    assert metadata["training_step_count"] == 2
    assert metadata["stop_reason"] == "agent_stopped"
    with np.load(path / "transitions.npz", allow_pickle=False) as data:
        assert data["joints"].dtype == np.float64
        assert data["camera"].shape == (4, 4, 5, 3)
        assert not data["terminated"].any()
    with pytest.raises(TypeError, match="spec"):
        validate_episode(path)
    with pytest.raises(ValueError, match="training boundary"):
        validate_episode(
            path, spec={**spec, "success_mask": LIBERO_SPEC["success_mask"]}
        )
    with pytest.raises(ValueError, match="one boolean per action"):
        validate_episode(path, spec={**spec, "success_mask": lambda data: True})

    report = export_lerobot(
        [path],
        spec=spec,
        expected_metadata={"experiment": "unit"},
        repo_id_prefix="rpent/unit",
        output_root=tmp_path / "export",
        dataset_id="test",
    )
    assert report["frame_count"] == 2
    dataset = LeRobotDataset(report["repo_id"], root=report["dataset_path"])
    assert dataset.meta.fps == 5
    assert dataset.meta.info["robot_type"] == "test"
    assert tuple(dataset[0]["actions"].shape) == (2,)
    assert tuple(dataset[0]["state"].shape) == (3,)
    assert tuple(dataset[0]["image"].shape) == (3, 4, 5)


@pytest.mark.parametrize("field", ["states", "actions"])
def test_validator_checks_declared_dtype(tmp_path, field):
    writer = create_episode_writer(
        {"root": tmp_path, "suite": "libero_object", "task_id": 2, "seed": 0},
        _obs(0),
    )
    writer.add_transition(np.zeros(7), _obs(1), 0, False, False)
    path = writer.finalize()
    with np.load(path / "transitions.npz", allow_pickle=False) as stored:
        data = dict(stored)
    data[field] = data[field].astype(np.float64)
    np.savez_compressed(path / "transitions.npz", **data)
    with pytest.raises(ValueError, match=f"invalid {field} dtype"):
        validate_episode(path, spec=LIBERO_SPEC)
