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

import threading
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest


class Events:
    enabled = True

    def __init__(self):
        self.records = []

    def emit(self, event):
        self.records.append(event.record)


@pytest.fixture
def recording(make_toolkit, monkeypatch):
    toolkit, env, _ = make_toolkit()
    events = Events()
    toolkit._dashboard_events = events
    saved = []
    original_save = toolkit.state.save

    def save(name, value, **kwargs):
        if name.endswith(".mp4"):
            saved.append(
                (name, kwargs["step"], [int(frame[0, 0, 0]) for frame in value])
            )
            value = b"video"
        return original_save(name, value, **kwargs)

    monkeypatch.setattr(toolkit.state, "save", save)
    return toolkit, env, events, saved


def image(env, value):
    env.image = np.full((8, 8, 3), value, dtype=np.uint8)


def test_each_action_saves_its_frames_and_updates_response_and_dashboard(recording):
    toolkit, env, events, saved = recording
    image(env, 1)
    first = toolkit.execute_tool("set_gripper", {"steps": 2})
    image(env, 2)
    second = toolkit.execute_tool("set_gripper", {"steps": 3})
    assert saved == [
        ("action_set_gripper.mp4", 1, [1, 1]),
        ("action_set_gripper.mp4", 2, [2, 2, 2]),
    ]
    for result, event in zip([first, second], events.records):
        assert not result.is_error
        assert "action_set_gripper.mp4" in result.data["artifacts"]
        assert result.data["artifacts"] == sorted(event.artifacts)
        observed = toolkit.execute_tool("view_env_state", {"step": result.data["step"]})
        assert observed.data == {
            key: value for key, value in result.data.items() if key != "agent_elapsed_s"
        }
        assert observed.images == result.images
    assert [int(frame[0, 0, 0]) for frame in toolkit._frames] == [1, 1, 2, 2, 2]


def test_zero_step_and_readonly_calls_do_not_save_clips(recording):
    toolkit, _, _, saved = recording
    for name, args in [
        ("set_gripper", {"steps": 0}),
        ("view_env_state", {}),
        ("finish", {"status": "stuck", "summary": "done"}),
    ]:
        result = toolkit.execute_tool(name, args)
        assert not result.is_error
        assert not any(
            name.endswith(".mp4") for name in result.data.get("artifacts", [])
        )
    assert saved == []


def test_failed_action_keeps_partial_video_and_next_action_starts_fresh(recording):
    toolkit, env, _, saved = recording
    original_step = env.step

    def fail_second(action):
        if len(env.actions) == 1:
            raise RuntimeError("driver failure")
        return original_step(action)

    image(env, 1)
    env.step = fail_second
    failed = toolkit.execute_tool("set_gripper", {"steps": 3})
    assert failed.is_error and failed.error == "driver failure"
    assert "action_set_gripper.mp4" in failed.data["artifacts"]
    assert toolkit.state.get(1).result["error"] == "driver failure"
    env.step = original_step
    image(env, 2)
    assert not toolkit.execute_tool("set_gripper", {"steps": 2}).is_error
    assert saved == [
        ("action_set_gripper.mp4", 1, [1]),
        ("action_set_gripper.mp4", 2, [2, 2]),
    ]


@pytest.mark.parametrize(
    "name,args,completed",
    [
        ("set_gripper", {"steps": 3}, 1),
        ("pi0_doubled", {"prompt": "touch bowl", "max_chunks": 2}, 3),
    ],
)
def test_cancelled_action_saves_completed_frames_before_resume(
    recording, name, args, completed
):
    toolkit, env, _, saved = recording
    stepped, release = threading.Event(), threading.Event()

    def after_step():
        stepped.set()
        assert release.wait(5)

    env.after_step = after_step
    image(env, 1)
    with ThreadPoolExecutor(max_workers=2) as executor:
        action = executor.submit(toolkit.execute_tool, name, args)
        try:
            assert stepped.wait(5)
            cancellation = executor.submit(toolkit.cancel_active_and_wait)
            for call in list(toolkit._scheduler._active_calls):
                assert call.cancel_event.wait(5)
        finally:
            release.set()
        result = action.result(timeout=5)
        cancellation.result(timeout=5)
    assert result.is_error and result.error == "Tool call cancelled."
    assert f"action_{name}.mp4" in result.data["artifacts"]
    toolkit.resume_calls()
    env.after_step = lambda: None
    image(env, 2)
    assert not toolkit.execute_tool("set_gripper", {"steps": 1}).is_error
    assert saved == [
        (f"action_{name}.mp4", 1, [1] * completed),
        ("action_set_gripper.mp4", 2, [2]),
    ]


@pytest.mark.parametrize("record_saved", [False, True])
def test_capture_failure_never_mixes_action_frames(
    recording, monkeypatch, record_saved
):
    toolkit, env, events, saved = recording

    def fail(*args, **kwargs):
        raise RuntimeError("observation failure")

    image(env, 1)
    with monkeypatch.context() as patch:
        if record_saved:
            patch.setattr("robots.libero.toolkit.build_observation", fail)
        else:
            patch.setattr(env, "raw_obs", fail)
        failed = toolkit.execute_tool("set_gripper", {"steps": 2})
    assert failed.is_error and "State capture failed:" in failed.error
    assert "step" not in failed.data
    if record_saved:
        assert saved == [("action_set_gripper.mp4", 1, [1, 1])]
        assert "action_set_gripper.mp4" in events.records[-1].artifacts
    else:
        assert saved == []
        assert events.records == []

    image(env, 2)
    succeeded = toolkit.execute_tool("set_gripper", {"steps": 1})
    assert not succeeded.is_error
    assert saved[-1] == ("action_set_gripper.mp4", 2 if record_saved else 1, [2])


@pytest.mark.parametrize("raises", [False, True])
def test_video_save_failure_keeps_action_success_and_does_not_add_artifact(
    recording, monkeypatch, raises
):
    toolkit, _, events, saved = recording
    original_save = toolkit.state.save

    def fail_video(name, value, **kwargs):
        if name.endswith(".mp4"):
            if raises:
                raise OSError("video failure")
            return None
        return original_save(name, value, **kwargs)

    monkeypatch.setattr(toolkit.state, "save", fail_video)
    result = toolkit.execute_tool("set_gripper", {"steps": 1})
    assert not result.is_error
    assert "action_set_gripper.mp4" not in result.data["artifacts"]
    assert result.data["artifacts"] == sorted(events.records[-1].artifacts)
    assert saved == []


def test_dashboard_disabled_keeps_episode_recording(make_toolkit, monkeypatch):
    toolkit, _, _ = make_toolkit()
    saved = []
    original_save = toolkit.state.save

    def save(name, value, **kwargs):
        if name.endswith(".mp4"):
            saved.append((name, len(value), kwargs))
            return name
        return original_save(name, value, **kwargs)

    monkeypatch.setattr(toolkit.state, "save", save)
    result = toolkit.execute_tool("set_gripper", {"steps": 2})
    assert not result.is_error
    assert not any(name.endswith(".mp4") for name in result.data["artifacts"])
    assert saved == []
    toolkit.close()
    assert saved == [("episode.mp4", 2, {"step": None, "fps": 20})]


def test_video_save_retains_exclusive_admission(recording, monkeypatch):
    toolkit, _, _, _ = recording
    saving, release, queued = threading.Event(), threading.Event(), threading.Event()
    original_save = toolkit.state.save

    def wait_video(name, value, **kwargs):
        if name.endswith(".mp4"):
            saving.set()
            assert release.wait(5)
        return original_save(name, value, **kwargs)

    scheduler = toolkit._scheduler
    can_start = scheduler._can_start

    def observe_queue(call):
        allowed = can_start(call)
        if not allowed:
            queued.set()
        return allowed

    monkeypatch.setattr(scheduler, "_can_start", observe_queue)
    monkeypatch.setattr(toolkit.state, "save", wait_video)
    with ThreadPoolExecutor(max_workers=2) as executor:
        action = executor.submit(toolkit.execute_tool, "set_gripper", {"steps": 1})
        try:
            assert saving.wait(5)
            observation = executor.submit(toolkit.execute_tool, "view_env_state", {})
            assert queued.wait(5)
            assert not observation.done()
        finally:
            release.set()
        assert not action.result(timeout=5).is_error
        result = observation.result(timeout=5)
        assert "action_set_gripper.mp4" in result.data["artifacts"]
