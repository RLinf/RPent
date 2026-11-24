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
import shutil
import threading

import numpy as np
import pytest

from rpent.flywheel.export import export_dataset, load_dataset

pytest.importorskip("lerobot")


@pytest.mark.parametrize("trajectory", [38], indirect=True)
def test_export_saves_declared_rldx_videos_actions_and_metadata(
    robocasa_trajectory, tmp_path
):
    trajectory = robocasa_trajectory
    trajectory["actions"][21]["source"] = "tool_computed"
    (trajectory["root"] / "actions.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in trajectory["actions"])
    )
    output = tmp_path / "robocasa"
    report = export_dataset(
        trajectory["root"],
        output,
        horizon=16,
        image_writer_threads=4,
    )
    assert report["robot"] == "robocasa"
    assert report["frame_counts"] == {"train": 37}
    assert report["episode_counts"] == {"train": 2}
    assert report["window_counts"] == {"train": 7}
    assert len(list((output / "train" / "videos").rglob("*.mp4"))) == 6
    assert not (output / "train" / "images").exists()
    moved = output.rename(tmp_path / "moved")
    shutil.rmtree(trajectory["root"])
    dataset = load_dataset(moved)
    assert dataset.features["observation.state"]["dtype"] == "float64"
    assert dataset.features["action"]["dtype"] == "float64"
    indices = np.r_[1:21, 22:39]
    actions = np.arange(12)[None, :] / 100 + indices[:, None] / 20
    actions[:, 6] = ((-1.0) ** indices + 1) / 2
    actions[:, 11] = ((-1.0) ** (indices + 1) + 1) / 2
    states = np.arange(16)[None, :] + indices[:, None] * 100
    stored = dataset.hf_dataset.with_format(None)
    np.testing.assert_array_equal(stored["action"], actions)
    np.testing.assert_array_equal(stored["observation.state"], states)
    assert dataset.meta.episodes[0]["length"] == 20
    assert dataset.meta.episodes[1]["length"] == 17
    for index in (0, 19, 20, 36):
        frame = dataset[index]
        assert frame["prompt"] == frame["task"] == "pick up the bowl"
        for camera_index, camera in enumerate(
            ("robot0_agentview_left", "robot0_agentview_right", "robot0_eye_in_hand")
        ):
            key = f"observation.images.{camera}"
            assert dataset.features[key]["dtype"] == "video"
            decoded = frame[key].numpy().transpose(1, 2, 0) * 255
            expected = np.full((64, 64, 3), indices[index] * 4, dtype=np.uint8)
            expected[..., camera_index] = 180
            expected[:16, :16] = [220, 40, 80]
            # Video compression is lossy; check camera order, orientation and time.
            assert np.abs(decoded - expected).mean() < 4
    meta = moved / "train" / "meta"
    modality = json.loads((meta / "modality.json").read_text())
    assert modality["annotation"] == {
        "human.task_description": {"original_key": "task_index"}
    }
    assert modality["action"] == {
        "end_effector_position": {"start": 0, "end": 3},
        "end_effector_rotation": {"start": 3, "end": 6},
        "gripper_close": {"start": 6, "end": 7},
        "base_motion": {"start": 7, "end": 11},
        "control_mode": {"start": 11, "end": 12},
    }
    assert modality["state"] == {
        "end_effector_position_relative": {"start": 0, "end": 3},
        "end_effector_rotation_relative": {"start": 3, "end": 7},
        "gripper_qpos": {"start": 7, "end": 9},
        "base_position": {"start": 9, "end": 12},
        "base_rotation": {"start": 12, "end": 16},
    }
    stats = json.loads((meta / "stats.json").read_text())
    for name, values in (("observation.state", states), ("action", actions)):
        for key in ("min", "max", "mean", "std"):
            np.testing.assert_allclose(stats[name][key], getattr(values, key)(axis=0))
        for key, q in (("q01", 0.01), ("q99", 0.99)):
            np.testing.assert_allclose(stats[name][key], np.quantile(values, q, axis=0))


def test_bimanual_export_preserves_native_state_actions_and_camera_order(
    bimanual_trajectory, tmp_path
):
    trajectory = bimanual_trajectory
    robot = trajectory["task"]["config"]["data"]["arguments"]["robot_name"]
    dim = (
        16
        if trajectory["actions"][1]["action"]["format"] == "robotwin.action.ee"
        else 14
    )
    output = tmp_path / "bimanual"
    report = export_dataset(
        trajectory["root"],
        output,
        horizon=50,
    )
    assert report["frame_counts"] == {"train": 6}
    assert report["fps"] == (25 if robot == "robodojo" else 20)
    assert report["window_counts"] == {"train": 0}
    moved = output.rename(tmp_path / "moved")
    shutil.rmtree(trajectory["root"])
    dataset = load_dataset(moved)
    assert dataset.features["action"]["shape"] == (dim,)
    assert dataset.features["action"]["dtype"] == "float64"
    assert dataset.features["observation.state"]["shape"] == (dim,)
    assert dataset.features["observation.state"]["dtype"] == (
        "float32" if robot == "robodojo" else "float64"
    )
    stored = dataset.hf_dataset.with_format(None)
    for index, frame in enumerate(dataset, 1):
        expected_action = np.arange(dim) / 10 + index
        np.testing.assert_array_equal(stored[index - 1]["action"], expected_action)
        # The Hugging Face torch formatter converts stored float64 lists to float32.
        np.testing.assert_array_equal(
            frame["action"], expected_action.astype(np.float32)
        )
        expected_state = np.arange(dim, dtype=np.float64) + index * 100
        if dim == 16:
            expected_state[[7, 15]] += 0.123456789
        np.testing.assert_array_equal(
            stored[index - 1]["observation.state"], expected_state
        )
        assert frame["prompt"] == frame["task"] == "pick up the bowl"
        for camera_index, camera in enumerate(
            ("cam_high", "cam_left_wrist", "cam_right_wrist")
        ):
            expected = np.full((16, 24, 3), camera_index + 1, dtype=np.uint8)
            expected[0, 0] = [10 + camera_index, 20 + camera_index, 30 + camera_index]
            np.testing.assert_array_equal(
                frame[f"observation.images.{camera}"],
                expected.transpose(2, 0, 1).astype(np.float32) / 255,
            )


def test_libero_export_loads_all_attempts_as_individual_frames(trajectory, tmp_path):
    jobs = tmp_path / "jobs"
    for job in ("first/attempt_01", "second"):
        shutil.copytree(trajectory["root"], jobs / job)
    output = tmp_path / "export"
    report = export_dataset(jobs, output, horizon=4)
    assert report["episode_counts"] == {"train": 2}
    assert report["frame_counts"] == {"train": 12}
    assert report["window_counts"] == {"train": 6}
    assert [row["excluded_action_counts"] for row in report["trajectories"]] == [
        {"non_vla": 2, "unselected_vla": 0}
    ] * 2

    moved = output.rename(tmp_path / "moved")
    shutil.rmtree(jobs)
    dataset = load_dataset(moved)
    assert len(dataset) == 12
    assert dataset.features["actions"]["shape"] == (7,)
    assert dataset.features["actions"]["dtype"] == "float32"
    for offset in (0, 5, 6, 11):
        frame = dataset[offset]
        index = offset % 6 + 1
        assert frame["episode_index"].item() == offset // 6
        assert frame["prompt"] == frame["task"] == "pick up the bowl"
        np.testing.assert_array_equal(frame["state"], np.full(8, index))
        np.testing.assert_array_equal(
            frame["actions"],
            np.arange(index * 7, (index + 1) * 7, dtype=np.float32) / 100,
        )
        for key, value in (("image", 0), ("wrist_image", 1)):
            assert frame[key].shape == (3, 256, 256)
            assert (frame[key] == value).all()


@pytest.mark.parametrize("trajectory", [35], indirect=True)
def test_export_keeps_short_segments_and_loader_does_not_bridge_a_script(
    trajectory, tmp_path
):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    path = trajectory["root"] / "actions.jsonl"
    actions = [json.loads(line) for line in path.read_text().splitlines()]
    actions[8]["source"] = "tool_computed"
    path.write_text("".join(json.dumps(action) + "\n" for action in actions))

    output = tmp_path / "split-export"
    report = export_dataset(trajectory["root"], output)
    assert report["episode_counts"] == {"train": 2}
    assert report["frame_counts"] == {"train": 34}
    assert report["window_counts"] == {"train": 18}
    source = report["trajectories"][0]
    assert source["selected_action_count"] == 34
    assert source["exported_action_count"] == 34
    assert source["excluded_action_count"] == 3
    assert source["excluded_action_counts"] == {
        "non_vla": 3,
        "unselected_vla": 0,
    }
    assert [
        (episode["start_index"], episode["stop_index"])
        for episode in source["episodes"]
    ] == [(1, 8), (9, 36)]
    dataset = load_dataset(output)
    assert [sample["state"][0].item() for sample in dataset] == [
        *range(1, 8),
        *range(9, 36),
    ]

    windows = LeRobotDataset(
        report["repo_id"],
        root=output / "train",
        video_backend="pyav",
        delta_timestamps={"actions": [step / report["fps"] for step in range(10)]},
    )
    last_short = windows[6]
    assert last_short["actions"].shape == (10, 7)
    assert last_short["actions_is_pad"].tolist() == [False] + [True] * 9
    np.testing.assert_array_equal(
        last_short["actions"],
        np.repeat(dataset[6]["actions"].numpy()[None, :], 10, axis=0),
    )
    assert not windows[7]["actions_is_pad"].any()
    np.testing.assert_array_equal(
        windows[7]["actions"],
        np.stack([dataset[index]["actions"] for index in range(7, 17)]),
    )
    assert sum(not sample["actions_is_pad"].any() for sample in windows) == 18


def test_export_closes_image_threads_when_a_later_image_is_missing(
    trajectory, tmp_path, monkeypatch
):
    from lerobot.datasets import image_writer

    observation = trajectory["actions"][2]["before_observation"]["payload"]["data"]
    observation["main_images"]["path"] = "missing.npy"
    (trajectory["root"] / "actions.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in trajectory["actions"])
    )
    writers = set()
    write_image = image_writer.write_image

    def record_writer(image, path):
        writers.add(threading.current_thread())
        write_image(image, path)

    monkeypatch.setattr(image_writer, "write_image", record_writer)
    output = tmp_path / "missing-later-image"
    with pytest.raises(FileNotFoundError, match="missing.npy"):
        export_dataset(trajectory["root"], output, image_writer_threads=4)
    assert writers
    assert all(not writer.is_alive() for writer in writers)
    assert not output.exists()
    assert not list(tmp_path.glob(".missing-later-image-*"))


def test_export_does_not_overwrite_an_existing_directory(trajectory, tmp_path):
    output = tmp_path / "existing"
    output.mkdir()
    marker = output / "keep.txt"
    marker.write_text("keep")
    with pytest.raises(FileExistsError):
        export_dataset(trajectory["root"], output)
    assert marker.read_text() == "keep"


def test_failed_trajectories_still_produce_an_exclusion_report(trajectory, tmp_path):
    path = trajectory["root"] / "task.json"
    document = json.loads(path.read_text())
    document["task"]["result"]["data"]["environment_success"] = False
    path.write_text(json.dumps(document))
    output = tmp_path / "failed-only"
    report = export_dataset(trajectory["root"], output)
    assert report["episode_counts"] == {"train": 0}
    assert report["dataset_paths"] == {"train": None}
    assert report["trajectories"][0]["excluded_action_count"] == 8
    assert (
        report["trajectories"][0]["rejected"][0]["reason"]
        == "no_confirmed_environment_success"
    )
    assert json.loads((output / "report.json").read_text())["frame_counts"] == {
        "train": 0,
    }
    assert report["window_counts"] == {"train": 0}
    assert len(load_dataset(output)) == 0
