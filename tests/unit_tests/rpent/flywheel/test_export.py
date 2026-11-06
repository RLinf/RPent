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
import sys
import threading
from dataclasses import replace
from types import ModuleType

import numpy as np
import pytest

from rpent.flywheel.export import export_dataset, load_dataset
from rpent.robots import get_robot_spec

pytest.importorskip("lerobot")


@pytest.mark.parametrize("trajectory", [38], indirect=True)
@pytest.mark.parametrize("robot", ["robocasa", "test_robot"])
@pytest.mark.parametrize("image_writer_threads", [0, 4])
def test_export_saves_declared_rldx_videos_actions_and_metadata(
    robocasa_trajectory, tmp_path, monkeypatch, robot, image_writer_threads
):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    trajectory = robocasa_trajectory
    if robot == "test_robot":
        spec = replace(get_robot_spec("robocasa"), name=robot)
        module = ModuleType(f"robots.{robot}")
        module.get_robot_spec = lambda: spec
        monkeypatch.setitem(sys.modules, module.__name__, module)
    trajectory["task"]["config"]["data"]["arguments"]["robot_name"] = robot
    document = json.loads((trajectory["root"] / "task.json").read_text())
    document["task"] = trajectory["task"]
    (trajectory["root"] / "task.json").write_text(json.dumps(document))
    trajectory["actions"][21]["source"] = "tool_computed"
    (trajectory["root"] / "actions.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in trajectory["actions"])
    )
    output = tmp_path / "robocasa"
    report = export_dataset(
        trajectory["root"],
        output,
        horizon=16,
        image_writer_threads=image_writer_threads,
    )
    assert report["robot"] == robot
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

    windows = LeRobotDataset(
        report["repo_id"],
        root=moved / "train",
        video_backend="pyav",
        delta_timestamps={"action": [step / report["fps"] for step in range(16)]},
    )
    np.testing.assert_array_equal(windows[0]["action"], actions[:16].astype(np.float32))
    np.testing.assert_array_equal(
        windows[19]["action"], np.repeat(actions[19:20], 16, axis=0).astype(np.float32)
    )
    assert windows[19]["action_is_pad"].tolist() == [False] + [True] * 15
    np.testing.assert_array_equal(
        windows[20]["action"], actions[20:36].astype(np.float32)
    )


@pytest.mark.parametrize("image_writer_threads", [0, 4])
def test_bimanual_export_preserves_native_state_actions_and_camera_order(
    bimanual_trajectory, tmp_path, image_writer_threads
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
        image_writer_threads=image_writer_threads,
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
        np.testing.assert_array_equal(
            frame["observation.state"], np.arange(dim) + index * 100
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


@pytest.mark.parametrize("trajectory", [68], indirect=True)
@pytest.mark.parametrize("image_writer_threads", [0, 4])
def test_bimanual_export_keeps_fifty_and_seventeen_step_segments(
    bimanual_trajectory, tmp_path, image_writer_threads
):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    trajectory = bimanual_trajectory
    robot = trajectory["task"]["config"]["data"]["arguments"]["robot_name"]
    script = trajectory["actions"][51]
    script["source"] = "tool_computed"
    script["action"]["format"] = (
        "robodojo.action.ee" if robot == "robodojo" else "robotwin.action.qpos"
    )
    (trajectory["root"] / "actions.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in trajectory["actions"])
    )
    output = tmp_path / "segments"
    report = export_dataset(
        trajectory["root"],
        output,
        horizon=50,
        image_writer_threads=image_writer_threads,
    )
    assert report["frame_counts"] == {"train": 67}
    assert report["episode_counts"] == {"train": 2}
    assert report["window_counts"] == {"train": 1}
    assert [
        (e["start_index"], e["stop_index"])
        for e in report["trajectories"][0]["episodes"]
    ] == [(1, 51), (52, 69)]
    windows = LeRobotDataset(
        report["repo_id"],
        root=output / "train",
        video_backend="pyav",
        delta_timestamps={"action": [i / report["fps"] for i in range(50)]},
    )
    np.testing.assert_array_equal(windows[0]["action"][:, 0], np.arange(1, 51))
    np.testing.assert_array_equal(windows[49]["action"][:, 0], np.full(50, 50))
    np.testing.assert_array_equal(
        windows[50]["action"][:, 0], np.r_[np.arange(52, 69), np.full(33, 68)]
    )
    assert windows[50]["action_is_pad"].tolist() == [False] * 17 + [True] * 33


def test_bimanual_export_does_not_split_by_layout_or_seed(
    bimanual_trajectory, tmp_path
):
    trajectory = bimanual_trajectory
    jobs = tmp_path / "jobs"
    for index, scene_id in enumerate((8, 9, None)):
        root = shutil.copytree(trajectory["root"], jobs / str(index))
        path = root / "task.json"
        doc = json.loads(path.read_text())
        args = doc["task"]["config"]["data"]["arguments"]
        key = "layout" if args["robot_name"] == "robodojo" else "seed"
        if scene_id is None:
            args.pop(key)
        else:
            args[key] = scene_id
        path.write_text(json.dumps(doc))
    output = tmp_path / "training"
    report = export_dataset(jobs, output)
    assert report["frame_counts"] == {"train": 18}
    assert report["episode_counts"] == {"train": 3}
    assert report["trajectory_count"] == report["exported_trajectory_count"] == 3
    dataset = load_dataset(output)
    assert len(dataset) == 18
    np.testing.assert_array_equal(
        dataset.hf_dataset["episode_index"], np.repeat([0, 1, 2], 6)
    )
    assert not (output / "validation").exists()


@pytest.mark.parametrize("trajectory", [35], indirect=True)
@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_export_stores_all_trajectories_as_individual_training_frames(
    trajectory, tmp_path, dtype
):
    action_path = trajectory["root"] / "actions.jsonl"
    actions = [json.loads(line) for line in action_path.read_text().splitlines()]
    for action in actions:
        action["action"]["data"]["dtype"] = dtype
    action_path.write_text("".join(json.dumps(action) + "\n" for action in actions))
    jobs = tmp_path / "jobs"
    for name, seed in (("first/attempt_01", 7), ("second", 8), ("third", None)):
        path = shutil.copytree(trajectory["root"], jobs / name)
        document = json.loads((path / "task.json").read_text())
        arguments = document["task"]["config"]["data"]["arguments"]
        if seed is None:
            arguments.pop("seed")
        else:
            arguments["seed"] = seed
        (path / "task.json").write_text(json.dumps(document))
    destination = tmp_path / "export"
    report = export_dataset(jobs, destination)
    assert report["episode_counts"] == {"train": 3}
    assert report["frame_counts"] == {"train": 105}
    assert report["window_counts"] == {"train": 78}
    assert report["dataset_paths"] == {"train": "train"}
    assert report["schema_version"] == 5
    assert report["trajectory_count"] == report["exported_trajectory_count"] == 3
    assert report["fps"] == 20
    assert not list(destination.rglob("*.png"))
    assert len(list(destination.rglob("*.parquet"))) == 3
    assert {path.name for path in destination.iterdir()} == {
        "train",
        "report.json",
    }
    assert [row["excluded_action_count"] for row in report["trajectories"]] == [2, 2, 2]
    assert [row["frame_counts"] for row in report["trajectories"]] == [
        {"train": 35}
    ] * 3
    assert [row["window_counts"] for row in report["trajectories"]] == [
        {"train": 26}
    ] * 3
    assert all(
        row["excluded_action_counts"] == {"non_vla": 2, "unselected_vla": 0}
        for row in report["trajectories"]
    )

    # Embedded images and relative dataset paths must survive moving the export.
    moved = destination.rename(tmp_path / "moved")
    shutil.rmtree(jobs)
    dataset = load_dataset(moved)
    assert len(dataset) == dataset.meta.total_frames == 105
    assert "observation.state" not in dataset.features
    assert "action" not in dataset.features
    assert dataset.features["state"]["shape"] == (8,)
    assert dataset.features["actions"]["shape"] == (7,)
    assert dataset.features["actions"]["dtype"] == dtype
    for offset in range(105):
        sample = dataset[offset]
        start = offset % 35 + 1
        expected = np.asarray(
            np.arange(start * 7, (start + 1) * 7, dtype=np.float32) / 100,
            dtype=dtype,
        )
        np.testing.assert_array_equal(sample["actions"], expected)
        np.testing.assert_array_equal(
            sample["state"], np.full(8, start, dtype=np.float32)
        )
        assert "actions_is_pad" not in sample
        assert sample["image"].shape == sample["wrist_image"].shape == (3, 256, 256)
        assert sample["prompt"] == sample["task"] == "pick up the bowl"
        assert sample["episode_index"].item() == offset // 35


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


@pytest.mark.parametrize(
    "trajectory,window_count",
    [(0, 0), (1, 0), (9, 0), (10, 1), (11, 2)],
    indirect=["trajectory"],
)
def test_export_preserves_every_step_including_short_tails(
    trajectory, tmp_path, window_count
):
    output = tmp_path / "tails"
    report = export_dataset(trajectory["root"], output)
    expected_steps = len(trajectory["actions"]) - 2
    assert report["frame_counts"] == {"train": expected_steps}
    assert report["window_counts"] == {"train": window_count}
    assert report["trajectories"][0]["excluded_action_count"] == 2
    dataset = load_dataset(output)
    assert len(dataset) == expected_steps
    assert [frame["state"][0].item() for frame in dataset] == list(
        range(1, expected_steps + 1)
    )


def test_horizon_only_changes_window_statistics(trajectory, tmp_path):
    outputs = []
    for horizon, count in ((4, 3), (10, 0)):
        output = tmp_path / f"horizon-{horizon}"
        report = export_dataset(trajectory["root"], output, horizon=horizon)
        assert report["horizon"] == horizon
        assert report["frame_counts"] == {"train": 6}
        assert report["window_counts"] == {"train": count}
        outputs.append(np.stack([frame["actions"] for frame in load_dataset(output)]))
    np.testing.assert_array_equal(outputs[0], outputs[1])


@pytest.mark.parametrize("trajectory", [20], indirect=True)
@pytest.mark.parametrize(
    "boundary,starts",
    [("script", [11]), ("missing_action", [9, 10, 11]), ("instruction", [10, 11])],
)
def test_loader_windows_stay_inside_selected_segments(
    trajectory, tmp_path, boundary, starts
):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    actions = trajectory["actions"]
    predictions = trajectory["predictions"]
    if boundary == "script":
        actions[10]["source"] = "tool_computed"
    elif boundary == "missing_action":
        actions.pop(8)
    else:
        for prediction in predictions[3:]:
            prediction["instruction"] = "different"
    for name, records in (("actions", actions), ("predictions", predictions)):
        (trajectory["root"] / f"{name}.jsonl").write_text(
            "".join(json.dumps(record) + "\n" for record in records)
        )
    output = tmp_path / "boundaries"
    report = export_dataset(trajectory["root"], output)
    assert report["episode_counts"] == {"train": 2}
    assert report["window_counts"] == {"train": len(starts)}
    windows = LeRobotDataset(
        report["repo_id"],
        root=output / "train",
        video_backend="pyav",
        delta_timestamps={"actions": [step / report["fps"] for step in range(10)]},
    )
    assert [
        frame["state"][0].item()
        for frame in windows
        if not frame["actions_is_pad"].any()
    ] == starts


@pytest.mark.parametrize("trajectory", [20], indirect=True)
def test_export_merges_continuous_actions_across_tool_calls(trajectory, tmp_path):
    for prediction in trajectory["predictions"][3:]:
        prediction["tool_call_id"] = "T-next"
    for action in trajectory["actions"][10:21]:
        action["tool_call_id"] = "T-next"
    for name in ("actions", "predictions"):
        (trajectory["root"] / f"{name}.jsonl").write_text(
            "".join(json.dumps(record) + "\n" for record in trajectory[name])
        )
    output = tmp_path / "tool-calls"
    report = export_dataset(trajectory["root"], output)
    assert report["episode_counts"] == {"train": 1}
    assert report["frame_counts"] == {"train": 20}
    assert report["window_counts"] == {"train": 11}
    assert [frame["state"][0].item() for frame in load_dataset(output)] == list(
        range(1, 21)
    )


@pytest.mark.parametrize("horizon", [0, -1, True, 1.5])
def test_export_horizon_must_be_a_positive_integer(trajectory, tmp_path, horizon):
    with pytest.raises(ValueError, match="positive integer"):
        export_dataset(trajectory["root"], tmp_path / "invalid", horizon=horizon)


@pytest.mark.parametrize("image_writer_threads", [-1, True, 1.5, "4"])
def test_export_rejects_invalid_image_writer_threads(tmp_path, image_writer_threads):
    output = tmp_path / "invalid"
    with pytest.raises(ValueError, match="image_writer_threads.*non-negative integer"):
        export_dataset(tmp_path, output, image_writer_threads=image_writer_threads)
    assert not output.exists()


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


@pytest.mark.parametrize("trajectory", [10], indirect=True)
def test_missing_image_aborts_export_without_publishing_partial_data(
    trajectory, tmp_path
):
    for path in trajectory["root"].rglob("*.npy"):
        path.unlink()
    output = tmp_path / "missing-image"
    with pytest.raises(FileNotFoundError):
        export_dataset(trajectory["root"], output)
    assert not output.exists()
    assert not list(tmp_path.glob(".missing-image-*"))


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
