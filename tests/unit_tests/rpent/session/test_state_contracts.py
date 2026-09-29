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
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from robots.behavior.toolkit import BehaviorToolkit
from rpent.dashboard.events import NullDashboardEventSink
from rpent.memory import MemoryManager
from rpent.robots import RunConfig
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


class _FakeModel:
    def predict(self, observation: dict[str, Any], *, options: dict[str, Any]) -> Any:
        assert observation["task_descriptions"] == "turn on the radio"
        assert options == {"mode": "eval"}
        return np.zeros((32, 23), dtype=np.float32)


class _FakeChunkEnv:
    total_env_steps = 0
    official_success_latched = False
    official_success_receipt = None

    def __init__(self) -> None:
        self.return_all_frames: list[bool] = []

    def chunk_step(
        self,
        actions: Any,
        *,
        return_all_frames: bool = False,
    ) -> tuple[Any, float, bool, bool, dict[str, Any]]:
        array = np.asarray(actions)
        self.return_all_frames.append(bool(return_all_frames))
        self.total_env_steps += int(array.shape[0])
        frames = [
            {
                "main_images": np.full((16, 16, 3), idx, dtype=np.uint8),
                "task_descriptions": "turn on the radio",
            }
            for idx in range(int(array.shape[0]))
        ]
        obs: Any = frames if return_all_frames else frames[-1]
        return (
            obs,
            0.0,
            False,
            False,
            {
                "executed_steps": int(array.shape[0]),
                "_rpent": {"total_env_steps": self.total_env_steps},
            },
        )


def test_finish_writes_terminal_receipt(tmp_path: Path) -> None:
    output_dir = tmp_path / "run"
    toolkit = BehaviorToolkit(
        runtime_kwargs={
            "task_name": "turning_on_radio",
            "output_dir": output_dir,
        },
        dashboard_events=NullDashboardEventSink(),
        memory=MemoryManager(tmp_path / "memory"),
    )

    result = toolkit.execute_tool(
        "finish", {"status": "incomplete", "summary": "bounded test"}
    )
    receipt = json.loads((output_dir / "terminal_receipt.json").read_text())

    assert result.is_finish is True
    assert receipt["_finish"] is True
    assert receipt["kind"] == "behavior_finish_terminal_receipt"
    assert receipt["planner_status"] == "incomplete"
    assert receipt["summary"] == "bounded test"


def test_receipt_is_session_artifact_and_recipe_is_run_artifact(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    session_dir = run_dir / "sessions" / "session_001"
    toolkit = BehaviorToolkit(
        runtime_kwargs={
            "task_name": "turning_on_radio",
            "output_dir": run_dir,
        },
        dashboard_events=NullDashboardEventSink(),
        memory=MemoryManager(tmp_path / "memory"),
        config=RunConfig(
            recipe_tag="turning_on_radio_s0",
            output_dir=run_dir,
            prompt_vars={
                "task_name": "turning_on_radio",
                "public_seed": 0,
                "memory_dir": str(tmp_path / "memory"),
            },
            task_desc={},
        ),
        state_output_dir=session_dir,
    )
    run_audit = run_dir / "turning_on_radio_s0.json"
    run_audit.parent.mkdir(parents=True, exist_ok=True)
    run_audit.write_text('{"audit": true}\n')

    toolkit.execute_tool("finish", {"status": "incomplete", "summary": "done"})
    assert (session_dir / "terminal_receipt.json").is_file()
    assert json.loads((session_dir / "states.json").read_text())["run_artifacts"] == [
        "terminal_receipt.json"
    ]
    assert run_audit.read_text() == '{"audit": true}\n'

    toolkit.primitives._official_success_latched = True
    with toolkit.state.record_step(
        state={"task_success": True},
        terminated=True,
        command={"action": "future_stateful_command", "arg": 1},
        result={"ok": True},
        elapsed_s=0.1,
    ):
        pass
    with toolkit.state.record_step(
        state={"task_success": True},
        terminated=True,
        command={"action": "bad_command"},
        result={"error": "failed"},
        elapsed_s=0.1,
    ):
        pass

    recipe_path = Path(toolkit.write_recipe("turning_on_radio_s0") or "")
    assert recipe_path == run_dir / "turning_on_radio_s0_recipe.jsonl"
    assert not (session_dir / "turning_on_radio_s0_recipe.jsonl").exists()
    commands = [
        json.loads(line)
        for line in recipe_path.read_text().splitlines()
        if line.strip()
    ]
    assert commands == [{"action": "future_stateful_command", "arg": 1}]
    assert "command" not in commands[0]


def test_solved_behavior_session_without_terminal_receipt_refuses_recipe(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    toolkit = BehaviorToolkit(
        runtime_kwargs={"task_name": "turning_on_radio", "output_dir": run_dir},
        memory=MemoryManager(tmp_path / "memory"),
        state_output_dir=run_dir / "sessions" / "session_001",
    )
    toolkit.primitives._official_success_latched = True
    with toolkit.state.record_step(
        state={"task_success": True},
        command={"action": "pi0_nav_pick", "instruction": "turn on radio", "chunks": 1},
        result={"ok": True},
    ):
        pass
    with pytest.raises(RuntimeError, match="missing terminal_receipt.json"):
        toolkit.write_recipe("turning_on_radio_s0")
    assert not (run_dir / "turning_on_radio_s0_recipe.jsonl").exists()


def test_unsolved_behavior_session_does_not_write_recipe(tmp_path: Path) -> None:
    toolkit = BehaviorToolkit(
        runtime_kwargs={
            "task_name": "turning_on_radio",
            "output_dir": tmp_path / "run",
        },
        dashboard_events=NullDashboardEventSink(),
        memory=MemoryManager(tmp_path / "memory"),
    )
    with toolkit.state.record_step(
        state={"task_success": False},
        command={"action": "pi0_nav_pick", "instruction": "turn on the radio"},
        result={"ok": True},
        elapsed_s=0.1,
    ):
        pass

    assert toolkit.write_recipe("turning_on_radio_s0") is None
    assert not (tmp_path / "run" / "turning_on_radio_s0_recipe.jsonl").exists()


@pytest.mark.parametrize("name", ["observe", "open"])
def test_behavior_images_stay_out_of_step_json(tmp_path: Path, name: str) -> None:
    from robots.behavior.tools import _PUBLIC_IMAGE_BYTE_FIELDS

    env = SimpleNamespace(
        total_env_steps=0, official_success_latched=False, official_success_receipt=None
    )
    toolkit = BehaviorToolkit(
        runtime_kwargs={
            "env": env,
            "task_name": "turning_on_radio",
            "output_dir": tmp_path / "run",
            "initial_observation": {
                "main_images": np.zeros((16, 16, 3), dtype=np.uint8),
            },
        },
        memory=MemoryManager(tmp_path / "memory"),
    )
    image = toolkit.state.load_bytes("head_rgb.png", step=0)
    payload = {"status": "ok", **dict.fromkeys(_PUBLIC_IMAGE_BYTE_FIELDS, image)}
    env.observe = lambda **kwargs: payload
    env.open_gripper = lambda **kwargs: payload
    try:
        result = toolkit.execute_tool(
            name, {"camera": "head"} if name == "observe" else {"hand": "left"}
        )
        assert "error" not in result.result
        assert "state_capture_error" not in result.result
        blocks = result.content_blocks
        assert sum(block["type"] == "image" for block in blocks) == 6
        text = next(block["text"] for block in blocks if block["type"] == "text")
        assert not any(field in text for field in _PUBLIC_IMAGE_BYTE_FIELDS)
        saved = json.loads((tmp_path / "run" / "states.json").read_text())
        record = saved["steps"][-1]
        assert not any(field in record["result"] for field in _PUBLIC_IMAGE_BYTE_FIELDS)
        assert record.get("command") == (
            None if name == "observe" else {"action": "open", "hand": "left"}
        )
    finally:
        toolkit.close()


@pytest.mark.parametrize(
    ("payload", "expected"),
    [
        ({"truncated": True}, True),
        ({"stop_reason": "truncated"}, True),
        ({"terminated": True, "truncated": False}, False),
    ],
)
def test_behavior_truncation_reaches_state_and_dashboard(
    tmp_path: Path, payload: dict, expected: bool
) -> None:
    from robots.behavior.robot_spec import BEHAVIOR_DASHBOARD_SPEC
    from rpent.dashboard.state import DashboardState

    toolkit = BehaviorToolkit(
        runtime_kwargs={
            "task_name": "turning_on_radio",
            "output_dir": tmp_path / "run",
        },
        memory=MemoryManager(tmp_path / "memory"),
    )
    try:
        toolkit.get_env_state(
            command={
                "action": "pi0_nav_pick",
                "instruction": "turn on the radio",
                "chunks": 1,
            },
            result=payload,
            elapsed_s=0.1,
        )
        record = toolkit.state.latest_record()
        assert record.truncated is expected
        assert record.terminated is False
        assert toolkit.solved() is False
        stored = json.loads((tmp_path / "run" / "states.json").read_text())["steps"][-1]
        assert stored["truncated"] is expected
        dashboard = DashboardState(
            output_dir=tmp_path / "dashboard", dashboard_spec=BEHAVIOR_DASHBOARD_SPEC
        )
        dashboard.on_step(record)
        snapshot = dashboard.session_detail()
        assert snapshot["truncated"] is expected
        assert snapshot["terminated"] is False
        assert snapshot["timeline"][-1]["result"]["official_success"] is False
        assert snapshot["timeline"][-1]["truncated"] is expected
    finally:
        toolkit.close()


def test_behavior_pi0_chunk_records_streaming_episode_video(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    session_dir = run_dir / "sessions" / "session_001"
    toolkit = BehaviorToolkit(
        runtime_kwargs={
            "env": _FakeChunkEnv(),
            "model": _FakeModel(),
            "task_name": "turning_on_radio",
            "public_seed": 0,
            "max_episode_steps": 64,
            "initial_observation": {
                "main_images": np.zeros((16, 16, 3), dtype=np.uint8),
                "task_descriptions": "turn on the radio",
            },
        },
        dashboard_events=NullDashboardEventSink(),
        memory=MemoryManager(tmp_path / "memory"),
        config=RunConfig(
            recipe_tag="turning_on_radio_s0",
            output_dir=run_dir,
            prompt_vars={
                "mode": "explore",
                "task_name": "turning_on_radio",
                "public_seed": 0,
                "memory_dir": str(tmp_path / "memory"),
            },
            task_desc={},
        ),
        state_output_dir=session_dir,
    )

    result = toolkit.primitives.pi0_nav_pick(
        instruction="turn on the radio",
        chunks=1,
    )
    toolkit.execute_tool("finish", {"status": "incomplete", "summary": "done"})

    assert result["chunks_used"] == 1
    assert result["env_steps_used"] == 32
    assert toolkit.primitives.env.return_all_frames == [True]
    video_path = session_dir / "episode.mp4"
    assert video_path.is_file()
    assert video_path.stat().st_size > 0
    assert (
        "episode.mp4"
        in json.loads((session_dir / "states.json").read_text())["run_artifacts"]
    )
