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

"""Standalone capture controls and cleanup without keyboard or robot hardware."""

import threading
from types import SimpleNamespace

import numpy as np
import pytest

from robots.lekiwi import capture as cli
from robots.lekiwi.env_client import LeKiwiEnvClient
from rpent.utils.rpc.http_rpc import HttpRpcClient, HttpRpcServer


class Keyboard:
    def __init__(self, *actions):
        self.actions = iter(actions)
        self.is_connected = True

    def get_action(self):
        keys = next(self.actions)
        if keys is None:
            self.is_connected = False
            return {"w": None}
        return dict.fromkeys(keys)


class Env:
    def __init__(self):
        self.meta = {
            "control_mode": "manual",
            "velocity_limits": [0.2, 0.15, 45.0],
        }
        self.calls = []
        self.state = {"phase": "recording", "frames": 1, "path": "/capture"}
        self.image = np.full((8, 8, 3), 127, dtype=np.uint8)

    def recording(self, operation, **kwargs):
        self.calls.append((operation, kwargs["identity"]))
        return self.state.copy()

    def stop(self):
        self.calls.append(("stop_base",))

    def hold(self, velocity, *, ttl_s):
        self.calls.append(("hold", velocity.tolist(), ttl_s))

    def get_observation(self):
        return {"main_images": self.image}


@pytest.fixture
def clock(monkeypatch):
    clock = SimpleNamespace(now=0.0)

    def advance(seconds):
        clock.now += seconds

    monkeypatch.setattr(cli.time, "monotonic", lambda: clock.now)
    monkeypatch.setattr(cli.time, "sleep", advance)
    return clock


def test_capture_uses_configured_body_velocities_and_stops_on_release(clock):
    env = Env()
    previews = []
    cli.capture(env, Keyboard("waZ", "", "sdx", "q"), preview=previews.append)
    held = [call for call in env.calls if call[0] == "hold"]
    np.testing.assert_allclose(held[0][1], [0.2, 0.15, 45])
    np.testing.assert_allclose(held[1][1], [-0.2, -0.15, -45])
    assert all(call[2] == 1.0 for call in held)
    first_hold = env.calls.index(held[0])
    assert env.calls[first_hold + 1] == ("stop_base",)
    assert env.calls[-2][0] == "stop_base"
    assert env.calls[-1][0] == "stop"
    identities = {
        call[1] for call in env.calls if call[0] in {"start", "state", "stop"}
    }
    assert len(identities) == 1
    np.testing.assert_array_equal(previews[0]["front"], env.image)


@pytest.mark.parametrize("phase", ["starting", "recorded"])
def test_no_motion_before_first_frame_or_after_recording_limit(clock, phase):
    env = Env()
    env.state.update(phase=phase, frames=0)
    cli.capture(env, Keyboard("w", "q"))
    assert not any(call[0] == "hold" for call in env.calls)
    assert [call[0] for call in env.calls[-2:]] == ["stop_base", "stop"]


def test_keyboard_disconnect_cannot_refresh_a_held_command(clock):
    env = Env()
    cli.capture(env, Keyboard("w", None))
    assert sum(call[0] == "hold" for call in env.calls) == 1
    assert [call[0] for call in env.calls[-2:]] == ["stop_base", "stop"]


@pytest.mark.parametrize(
    "failure", ["hold", "get_observation", "recording", "interrupt"]
)
def test_errors_stop_wheels_and_attempt_recording_cleanup(clock, monkeypatch, failure):
    env = Env()

    def fail(*args, **kwargs):
        if failure == "interrupt":
            raise KeyboardInterrupt
        raise RuntimeError("connection lost")

    if failure == "recording":
        original = env.recording

        def recording(operation, **kwargs):
            result = original(operation, **kwargs)
            if operation == "start":
                fail()  # The recorder started but the RPC response was lost.
            return result

        monkeypatch.setattr(env, "recording", recording)
    else:
        monkeypatch.setattr(env, "hold" if failure == "interrupt" else failure, fail)
    with pytest.raises(KeyboardInterrupt if failure == "interrupt" else RuntimeError):
        cli.capture(env, Keyboard("w"), preview=lambda obs: None)
    assert [call[0] for call in env.calls[-2:]] == ["stop_base", "stop"]


@pytest.mark.parametrize("failure", ["recorder", "camera"])
def test_capture_does_not_drive_with_failed_images(clock, failure):
    env = Env()
    if failure == "recorder":
        env.state.update(phase="error", error="Disk full")
    else:
        env.image = None
    with pytest.raises(RuntimeError, match="Disk full|preview is unavailable"):
        cli.capture(env, Keyboard("w"), preview=lambda obs: None)
    assert not any(call[0] == "hold" for call in env.calls)
    assert [call[0] for call in env.calls[-2:]] == ["stop_base", "stop"]


@pytest.mark.parametrize("slow_call", ["get_observation", "hold"])
def test_stalled_loop_exits_instead_of_resuming_motion(clock, monkeypatch, slow_call):
    env = Env()
    original = getattr(env, slow_call)

    def delayed(*args, **kwargs):
        clock.now += 1.1
        return original(*args, **kwargs)

    monkeypatch.setattr(env, slow_call, delayed)
    with pytest.raises(TimeoutError, match="stalled"):
        cli.capture(env, Keyboard("w"), preview=lambda obs: None)
    assert sum(call[0] == "hold" for call in env.calls) == (slow_call == "hold")
    assert [call[0] for call in env.calls[-2:]] == ["stop_base", "stop"]


def test_scheduler_stall_does_not_resume_a_previously_held_key(clock, monkeypatch):
    env = Env()

    def delayed_sleep(seconds):
        clock.now += 1.1

    monkeypatch.setattr(cli.time, "sleep", delayed_sleep)
    with pytest.raises(TimeoutError, match="stalled"):
        cli.capture(env, Keyboard("w", "w"))
    assert sum(call[0] == "hold" for call in env.calls) == 1
    assert [call[0] for call in env.calls[-2:]] == ["stop_base", "stop"]


@pytest.mark.parametrize("invalid", ["navigation", "keyboard", "limits"])
def test_invalid_setup_never_stops_or_records_another_session(clock, invalid):
    env, keyboard = Env(), Keyboard("w")
    if invalid == "navigation":
        env.meta["control_mode"] = "navigation"
    elif invalid == "keyboard":
        keyboard.is_connected = False
    else:
        env.meta["velocity_limits"] = [float("nan"), 0.1, 30]
    with pytest.raises((RuntimeError, ValueError)):
        cli.capture(env, keyboard)
    assert env.calls == []


def test_recording_cleanup_still_runs_when_stop_rpc_fails(clock, monkeypatch):
    env = Env()
    original = env.stop

    def stop():
        if any(call[0] == "hold" for call in env.calls):
            raise RuntimeError("Stop RPC unavailable")
        original()

    monkeypatch.setattr(env, "stop", stop)
    with pytest.raises(RuntimeError, match="Stop RPC"):
        cli.capture(env, Keyboard("w", "q"))
    assert env.calls[-1][0] == "stop"


def test_capture_with_real_client_over_http():
    backend = Env()

    def dispatch(method, args, kwargs, *, session_id=None):
        if method == "env.get_env_meta":
            result = backend.meta
        elif method.startswith("env.recording_"):
            result = backend.recording(method.removeprefix("env.recording_"), **kwargs)
        else:
            result = getattr(backend, method.removeprefix("env."))(*args, **kwargs)
        return result

    server = HttpRpcServer(("127.0.0.1", 0), dispatch)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    rpc = HttpRpcClient(f"http://127.0.0.1:{server.server_address[1]}")
    try:
        cli.capture(LeKiwiEnvClient(rpc), Keyboard("w", "", "q"))
    finally:
        rpc.close()
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
    held = [call for call in backend.calls if call[0] == "hold"]
    np.testing.assert_allclose(held[0][1], [0.2, 0, 0])
    assert [call[0] for call in backend.calls[-2:]] == ["stop_base", "stop"]
