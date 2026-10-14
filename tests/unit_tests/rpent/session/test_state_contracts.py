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

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from rpent.session import EnvState, StepRecord


class _StepFailure(Exception):
    pass


def test_step_record_serializes_optional_fields_and_sorts_artifacts() -> None:
    minimal = StepRecord(
        step_idx=2,
        state={"objects": ["cup"]},
        artifacts={"z.txt", "a.json"},
    )
    complete = StepRecord(
        step_idx=3,
        state={"ready": True},
        terminated=True,
        truncated=True,
        artifacts={"frame.png"},
        command={"action": "move"},
        result={"success": True},
        elapsed_s=1.25,
        extras={"attempt": 4},
    )

    assert minimal.to_blob() == {
        "step_idx": 2,
        "state": {"objects": ["cup"]},
        "terminated": False,
        "truncated": False,
        "artifacts": ["a.json", "z.txt"],
    }
    assert complete.to_blob() == {
        "step_idx": 3,
        "state": {"ready": True},
        "terminated": True,
        "truncated": True,
        "artifacts": ["frame.png"],
        "command": {"action": "move"},
        "result": {"success": True},
        "elapsed_s": 1.25,
        "extras": {"attempt": 4},
    }


def test_env_state_records_copies_and_latest_step_behavior(tmp_path: Path) -> None:
    env_state = EnvState(tmp_path)
    state = {"objects": [{"name": "cup"}]}
    command = {"action": "move", "offset": [1, 2]}
    result = {"success": True}
    extras = {"attempt": 1}

    assert env_state.latest_step is None
    assert env_state.latest_record() is None
    assert env_state.records() == []
    assert env_state.exists("frame.png") is False

    with env_state.record_step(
        state=state,
        command=command,
        result=result,
        elapsed_s=0.5,
        extras=extras,
    ) as step:
        assert step == 0
    state["objects"][0]["name"] = "changed"
    command["offset"].append(3)
    result["success"] = False
    extras["attempt"] = 2

    with env_state.record_step(state={"objects": []}, terminated=True) as step:
        assert step == 1

    assert env_state.latest_step == 1
    assert env_state.latest_record() is not None
    assert env_state.latest_record().step_idx == 1
    first = env_state.get(0)
    assert first.state == {"objects": [{"name": "cup"}]}
    assert first.command == {"action": "move", "offset": [1, 2]}
    assert first.result == {"success": True}
    assert first.extras == {"attempt": 1}
    assert env_state.get().step_idx == 1

    first.state["objects"].clear()
    records = env_state.records()
    records[0].command["offset"].append(99)
    assert env_state.get(0).state == {"objects": [{"name": "cup"}]}
    assert env_state.get(0).command == {"action": "move", "offset": [1, 2]}


@pytest.mark.parametrize(
    "name",
    [
        "",
        None,
        1,
        Path("data.json"),
        ".",
        "..",
        "states.json",
        "without_suffix",
        "unsupported.csv",
        "nested/data.json",
        "../escape.json",
    ],
)
def test_env_state_rejects_invalid_artifact_names(
    tmp_path: Path,
    name: Any,
) -> None:
    env_state = EnvState(tmp_path / "output")

    with pytest.raises(ValueError):
        env_state.artifact_path(name, step=None)
    with pytest.raises(ValueError):
        env_state.save(name, "value", step=None)


def test_env_state_rejects_absolute_artifact_names(tmp_path: Path) -> None:
    env_state = EnvState(tmp_path / "output")
    absolute_name = str(tmp_path / "outside.json")

    with pytest.raises(ValueError, match="base filename"):
        env_state.artifact_path(absolute_name, step=None)
    assert not (tmp_path / "outside.json").exists()


def test_env_state_rejects_invalid_steps(tmp_path: Path) -> None:
    env_state = EnvState(tmp_path)

    with pytest.raises(ValueError, match="-1, None, or nonnegative"):
        env_state.artifact_path("data.json", step=-2)
    with pytest.raises(LookupError, match="no steps available"):
        env_state.load("data.json")
    assert env_state.exists("data.json") is False


def test_env_state_run_level_artifact_round_trips(tmp_path: Path) -> None:
    env_state = EnvState(tmp_path)
    array = np.array([[1, 2], [3, 4]], dtype=np.int16)
    image = np.array(
        [
            [[255, 0, 0], [0, 255, 0]],
            [[0, 0, 255], [255, 255, 255]],
        ],
        dtype=np.uint8,
    )
    cases: list[tuple[str, Any]] = [
        (
            "metadata.json",
            {
                "array": np.array([1, 2]),
                "scalar": np.int64(3),
                "path": Path("relative/file"),
            },
        ),
        ("events.jsonl", [{"step": 1}, {"step": 2}]),
        ("array.npy", array),
        ("compressed.npz", array),
        ("notes.txt", "hello 世界"),
        ("frame.png", image),
        ("payload.bin", b"\x00\x01bytes"),
        ("episode.mp4", b"offline-mp4-bytes"),
    ]

    for name, value in cases:
        assert env_state.save(name, value, step=None) == name
        assert env_state.exists(name, step=None)
        assert env_state.artifact_path(name, step=None) == tmp_path / name

    assert env_state.load("metadata.json", step=None) == {
        "array": [1, 2],
        "scalar": 3,
        "path": "relative/file",
    }
    assert env_state.load("events.jsonl", step=None) == [
        {"step": 1},
        {"step": 2},
    ]
    np.testing.assert_array_equal(env_state.load("array.npy", step=None), array)
    np.testing.assert_array_equal(env_state.load("compressed.npz", step=None), array)
    assert env_state.load("notes.txt", step=None) == "hello 世界"
    np.testing.assert_array_equal(env_state.load("frame.png", step=None), image)
    assert env_state.load("payload.bin", step=None) == b"\x00\x01bytes"
    assert env_state.load_bytes("episode.mp4", step=None) == b"offline-mp4-bytes"


def test_env_state_per_step_paths_save_load_and_exists(tmp_path: Path) -> None:
    env_state = EnvState(tmp_path)

    with env_state.record_step(state={"value": 0}) as first_step:
        assert env_state.save("detail.json", {"step": 0}) == "detail.json"
    with env_state.record_step(state={"value": 1}) as second_step:
        assert env_state.save("detail.json", {"step": 1}) == "detail.json"

    assert (first_step, second_step) == (0, 1)
    assert env_state.artifact_path("detail.json", step=0) == (
        tmp_path / "detail.json" / "00.json"
    )
    assert env_state.artifact_path("detail.json", step=1) == (
        tmp_path / "detail.json" / "01.json"
    )
    assert env_state.load("detail.json", step=0) == {"step": 0}
    assert env_state.load("detail.json") == {"step": 1}
    assert env_state.exists("detail.json", step=0)
    assert env_state.exists("detail.json")
    assert env_state.get(0).artifacts == {"detail.json"}
    assert env_state.get(1).artifacts == {"detail.json"}


def test_env_state_frame_video_supplies_a_supported_encoder(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    frames = [np.full((16, 32, 3), value, dtype=np.uint8) for value in (32, 96)]

    def encode(destination, images, *, fps, codec=None):
        if codec is None:
            raise TypeError("expected bytes, NoneType found")
        assert codec == "libx264"
        assert fps == 12
        for actual, expected in zip(images, frames, strict=True):
            np.testing.assert_array_equal(actual, expected)
        Path(destination).write_bytes(b"encoded-video")

    monkeypatch.setattr("rpent.session.base.imageio.mimwrite", encode)
    state = EnvState(tmp_path)
    assert state.save("episode.mp4", iter(frames), step=None, fps=12) == "episode.mp4"
    assert state.load_bytes("episode.mp4", step=None) == b"encoded-video"
    assert json.loads((tmp_path / "states.json").read_text())["run_artifacts"] == [
        "episode.mp4"
    ]


@pytest.mark.parametrize("backend", ["pyav", "FFMPEG"])
@pytest.mark.parametrize("step", [None, 0])
def test_env_state_frame_video_encodes_and_decodes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, backend: str, step: int | None
) -> None:
    av = pytest.importorskip("av")
    if backend == "FFMPEG":
        pytest.importorskip("imageio_ffmpeg")
    import rpent.session.base as state_module

    writer = state_module.imageio.mimwrite

    def write_video(*args, **kwargs):
        return writer(*args, format=backend, **kwargs)

    monkeypatch.setattr(state_module.imageio, "mimwrite", write_video)
    state = EnvState(tmp_path)
    if step is not None:
        with state.record_step(state={"ready": True}):
            pass
    frames = [np.full((16, 32, 3), value, dtype=np.uint8) for value in (32, 64, 96)]
    assert state.save("clip.mp4", frames, step=step) == "clip.mp4"
    path = state.artifact_path("clip.mp4", step=step)
    assert path.stat().st_size > 0
    with av.open(str(path)) as video:
        assert video.streams.video[0].codec_context.name == "h264"
        assert video.streams.video[0].average_rate == 20
        decoded = [frame.to_ndarray(format="rgb24") for frame in video.decode(video=0)]
    assert len(decoded) == len(frames)
    for actual, expected in zip(decoded, frames, strict=True):
        assert actual.shape == expected.shape
        assert np.abs(actual.astype(np.int16) - expected).max() <= 5
    manifest = json.loads((tmp_path / "states.json").read_text())
    artifacts = (
        manifest["run_artifacts"]
        if step is None
        else manifest["steps"][step]["artifacts"]
    )
    assert artifacts == ["clip.mp4"]
    assert not list(tmp_path.rglob(".*.tmp*"))


@pytest.mark.parametrize(
    "value", [b"encoded", bytearray(b"encoded"), memoryview(b"encoded")]
)
def test_env_state_encoded_video_bypasses_encoder(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    value: bytes | bytearray | memoryview,
) -> None:
    def unexpected_encoding(*args, **kwargs):
        pytest.fail("Already encoded video must not be encoded again")

    monkeypatch.setattr("rpent.session.base.imageio.mimwrite", unexpected_encoding)
    state = EnvState(tmp_path)
    assert state.save("episode.mp4", value, step=None) == "episode.mp4"
    assert state.load_bytes("episode.mp4", step=None) == bytes(value)


def test_env_state_video_encoding_failure_does_not_publish_artifact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    def fail_encoding(destination, *args, **kwargs):
        Path(destination).write_bytes(b"partial-video")
        raise RuntimeError("video encoder failed")

    monkeypatch.setattr("rpent.session.base.imageio.mimwrite", fail_encoding)
    state = EnvState(tmp_path)
    with state.record_step(state={"ready": True}):
        pass
    assert state.save("clip.mp4", [np.zeros((16, 32, 3), dtype=np.uint8)]) is None
    assert not state.exists("clip.mp4")
    assert state.get().artifacts == set()
    assert "video encoder failed" in caplog.text
    assert not list(tmp_path.rglob(".*.tmp*"))


def test_prune_artifacts_keeps_recent_files_and_historical_records(
    tmp_path: Path,
) -> None:
    env_state = EnvState(tmp_path)
    env_state.save("session.txt", "session", step=None)
    names = ("world.npz", "optional_world.npz")

    for value in range(2):
        with env_state.record_step(state={"value": value}) as step:
            env_state.save("world.npz", np.array([value]))
            env_state.save("frame.png", np.zeros((2, 2, 3), dtype=np.uint8))
            env_state.save("metadata.json", {"value": value})
            env_state.prune_artifacts(names, step=step, keep_last=2)

    assert env_state.exists("world.npz", step=0)
    first_record = env_state.get(0)

    with env_state.record_step(state={"value": 2}) as step:
        env_state.save("world.npz", np.array([2]))
        env_state.prune_artifacts(names, step=step, keep_last=2)

    assert not env_state.exists("world.npz", step=0)
    np.testing.assert_array_equal(env_state.load("world.npz", step=1), [1])
    np.testing.assert_array_equal(env_state.load("world.npz", step=2), [2])
    assert env_state.exists("frame.png", step=0)
    assert env_state.load("metadata.json", step=0) == {"value": 0}
    assert env_state.load("session.txt", step=None) == "session"
    assert env_state.get(0) == first_record
    manifest = json.loads((tmp_path / "states.json").read_text())
    assert manifest["steps"][0] == first_record.to_blob()


def test_prune_artifacts_logs_deletion_failure_and_continues(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    env_state = EnvState(tmp_path)
    with env_state.record_step(state={}):
        env_state.save("depth.npz", np.array([1]))
        env_state.save("world.npz", np.array([2]))
    with env_state.record_step(state={}):
        pass

    blocked = env_state.artifact_path("depth.npz", step=0)
    original_unlink = Path.unlink

    def unlink(path: Path, *, missing_ok: bool = False) -> None:
        if path == blocked:
            raise PermissionError("deletion denied")
        original_unlink(path, missing_ok=missing_ok)

    monkeypatch.setattr(Path, "unlink", unlink)
    env_state.prune_artifacts(("depth.npz", "world.npz"), step=1, keep_last=1)

    assert env_state.exists("depth.npz", step=0)
    assert not env_state.exists("world.npz", step=0)
    assert "failed to prune artifact depth.npz at step 0" in caplog.text


def test_manifest_updates_with_deterministic_artifact_order(tmp_path: Path) -> None:
    env_state = EnvState(tmp_path)
    assert env_state.save("z.txt", "last", step=None) == "z.txt"
    assert env_state.save("a.json", {"first": True}, step=None) == "a.json"

    with env_state.record_step(
        state={"array": np.array([2, 1])},
        command={"action": "capture"},
    ):
        assert env_state.save("z.bin", b"z") == "z.bin"
        assert env_state.save("a.txt", "a") == "a.txt"

    manifest = json.loads((tmp_path / "states.json").read_text())

    assert manifest["run_artifacts"] == ["a.json", "z.txt"]
    assert manifest["steps"] == [
        {
            "step_idx": 0,
            "state": {"array": [2, 1]},
            "terminated": False,
            "truncated": False,
            "artifacts": ["a.txt", "z.bin"],
            "command": {"action": "capture"},
        }
    ]
    assert list(tmp_path.glob(".*.tmp*")) == []


def test_failed_artifact_write_removes_temporary_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    env_state = EnvState(tmp_path)

    def fail_after_partial_write(destination: Path, value: Any) -> None:
        Path(destination).write_bytes(b"partial")
        raise RuntimeError("image encoder failed")

    monkeypatch.setattr(
        "rpent.session.base.imageio.imwrite",
        fail_after_partial_write,
    )

    assert env_state.save("broken.png", np.zeros((2, 2, 3)), step=None) is None
    assert not env_state.artifact_path("broken.png", step=None).exists()
    assert list(tmp_path.glob(".*.tmp*")) == []


def test_env_state_reset_clears_memory_but_preserves_disk_artifacts(
    tmp_path: Path,
) -> None:
    env_state = EnvState(tmp_path)
    assert env_state.save("run.txt", "run", step=None) == "run.txt"
    with env_state.record_step(state={"ready": True}):
        assert env_state.save("step.bin", b"step") == "step.bin"

    env_state.reset()

    assert env_state.latest_step is None
    assert env_state.latest_record() is None
    assert env_state.records() == []
    assert env_state.exists("run.txt", step=None)
    assert env_state.exists("step.bin", step=0)
    assert env_state.load("step.bin", step=0) == b"step"
    assert env_state.exists("step.bin") is False


def test_record_step_wraps_caller_failure_after_rolling_back_record(
    tmp_path: Path,
) -> None:
    env_state = EnvState(tmp_path)

    with pytest.raises(
        RuntimeError,
        match="failed to record step 0: step failed",
    ) as exc_info:
        with env_state.record_step(state={"partial": True}):
            raise _StepFailure("step failed")

    assert isinstance(exc_info.value.__cause__, _StepFailure)
    assert env_state.records() == []


def test_record_step_failure_removes_artifacts_written_by_discarded_step(
    tmp_path: Path,
) -> None:
    env_state = EnvState(tmp_path)

    with pytest.raises(RuntimeError) as exc_info:
        with env_state.record_step(state={"partial": True}) as step:
            assert env_state.save("partial.bin", b"partial") == "partial.bin"
            raise _StepFailure("step failed")

    assert isinstance(exc_info.value.__cause__, _StepFailure)
    assert env_state.records() == []
    assert not env_state.artifact_path("partial.bin", step=step).exists()


@pytest.mark.parametrize("error_type", [KeyboardInterrupt, SystemExit])
def test_record_step_preserves_control_flow_exceptions_after_rollback(
    tmp_path: Path,
    error_type: type[BaseException],
) -> None:
    env_state = EnvState(tmp_path)
    error = error_type("stop")

    with pytest.raises(error_type) as exc_info:
        with env_state.record_step(state={"partial": True}) as step:
            assert env_state.save("partial.bin", b"partial") == "partial.bin"
            raise error

    assert exc_info.value is error
    assert env_state.records() == []
    assert not env_state.artifact_path("partial.bin", step=step).exists()
    assert json.loads((tmp_path / "states.json").read_text()) == {
        "run_artifacts": [],
        "steps": [],
    }


def test_env_state_rejects_nested_step_records(tmp_path: Path) -> None:
    env_state = EnvState(tmp_path)

    with pytest.raises(RuntimeError, match="already open"):
        with env_state.record_step(state={"outer": True}):
            with env_state.record_step(state={"inner": True}):
                pass

    assert env_state.records() == []


@pytest.mark.parametrize("second_name", ["first.json", "second.json"])
def test_concurrent_saves_keep_artifacts_and_manifest_consistent(
    tmp_path, monkeypatch, second_name
):
    import os
    import threading
    from concurrent.futures import ThreadPoolExecutor

    state = EnvState(tmp_path)
    with state.record_step(state={}):
        pass
    first_manifest = threading.Event()
    release = threading.Event()
    second_started = threading.Event()
    replace = os.replace

    def delayed_replace(source, destination):
        if Path(destination).name == "states.json" and not first_manifest.is_set():
            first_manifest.set()
            assert release.wait(5)
        replace(source, destination)

    def save_second():
        second_started.set()
        return state.save(second_name, {"value": 2})

    monkeypatch.setattr("rpent.session.base.os.replace", delayed_replace)
    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(state.save, "first.json", {"value": 1})
        try:
            assert first_manifest.wait(3)
            second = pool.submit(save_second)
            assert second_started.wait(3)
            assert not second.done()
        finally:
            release.set()
        assert first.result(timeout=3) == "first.json"
        assert second.result(timeout=3) == second_name
    manifest = json.loads((tmp_path / "states.json").read_text())
    assert set(manifest["steps"][0]["artifacts"]) == {"first.json", second_name}
    assert state.load(second_name) == {"value": 2}
