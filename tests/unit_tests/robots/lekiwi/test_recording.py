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

"""Independent recordings, with a real CPU video encoder and no robot hardware."""

import json
import threading
import time
from types import SimpleNamespace

import imageio.v2 as imageio
import numpy as np
import pytest

from robots.lekiwi import record as cli


class CameraEnv:
    def __init__(self):
        self.read = threading.Event()
        self.image = np.zeros((16, 16, 3), dtype=np.uint8)
        self.image[..., 0] = 255  # A red frame in the Env's RGB format.

    def get_camera_meta(self):
        return {"cameras": {"front": {"serial_number": "test"}}}

    def get_observation(self):
        self.read.set()
        return {"main_images": self.image}


@pytest.mark.parametrize(
    "shape", [(16, 16, 3), (18, 30, 3), (16, 19, 3), (17, 16, 3), (17, 19, 3)]
)
def test_stationary_video_is_saved_without_action_log(tmp_path, shape):
    pytest.importorskip("imageio_ffmpeg")
    env = CameraEnv()  # Deliberately has no motion/reset methods.
    env.image = np.broadcast_to(env.image[0, 0], shape).copy()
    cli.record(env, tmp_path, duration_s=0.1)
    assert env.read.is_set()
    (directory,) = tmp_path.iterdir()
    with imageio.get_reader(directory / "video.mp4") as reader:
        frame = reader.get_data(0)
        assert frame.shape == shape
        assert frame[..., 0].mean() > 240
        assert frame[..., 2].mean() < 15
    timestamps = [
        json.loads(line)
        for line in (directory / "frames.jsonl").read_text().splitlines()
    ]
    assert timestamps[0]["frame_index"] == 0
    assert timestamps[0]["video_time_s"] == 0
    assert timestamps[0]["received_at_unix_s"] <= time.time()
    assert timestamps[0]["elapsed_s"] >= 0
    assert timestamps[0]["width"] == shape[1]
    assert timestamps[0]["height"] == shape[0]
    assert {path.name for path in directory.iterdir()} == {
        "video.mp4",
        "camera_meta.json",
        "frames.jsonl",
    }
    assert (
        json.loads((directory / "camera_meta.json").read_text())
        == env.get_camera_meta()
    )


def test_video_finishes_at_duration_limit(tmp_path, monkeypatch):
    closed = threading.Event()
    monkeypatch.setattr(
        imageio,
        "get_writer",
        lambda *a, **kw: SimpleNamespace(
            append_data=lambda frame: None, close=closed.set
        ),
    )
    env = CameraEnv()
    started = time.monotonic()
    cli.record(env, tmp_path, duration_s=0.05)
    assert 0.05 <= time.monotonic() - started < 1.0
    assert env.read.is_set()
    assert closed.is_set()


def test_low_frame_rate_does_not_delay_stop(tmp_path, monkeypatch):
    closed = threading.Event()
    monkeypatch.setattr(
        imageio,
        "get_writer",
        lambda *a, **kw: SimpleNamespace(
            append_data=lambda frame: None, close=closed.set
        ),
    )
    env = CameraEnv()
    started = time.monotonic()
    cli.record(
        env,
        tmp_path,
        fps=0.1,
        stop_requested=lambda: time.monotonic() - started >= 0.05,
    )
    assert time.monotonic() - started < 1.0
    assert env.read.is_set()
    assert closed.is_set()


@pytest.mark.parametrize("failure", ["write", "close"])
def test_failed_video_closes_writer_and_reports_error(tmp_path, monkeypatch, failure):
    closed = threading.Event()

    def append(frame):
        if failure == "write":
            raise OSError("disk full")

    def close():
        closed.set()
        if failure == "close":
            raise OSError("disk full")

    monkeypatch.setattr(
        imageio,
        "get_writer",
        lambda *a, **kw: SimpleNamespace(append_data=append, close=close),
    )
    with pytest.raises(RuntimeError, match="disk full"):
        cli.record(CameraEnv(), tmp_path, duration_s=0.05)
    assert closed.is_set()
    assert not list(tmp_path.rglob("teleop_actions.jsonl"))


@pytest.mark.parametrize(
    "options",
    [
        {"fps": 0},
        {"fps": float("nan")},
        {"duration_s": 0},
        {"duration_s": 3601},
    ],
)
def test_invalid_recording_has_no_filesystem_effects(tmp_path, options):
    root = tmp_path / "absent"
    with pytest.raises(ValueError):
        cli.record(CameraEnv(), root, **options)
    assert not root.exists()


def test_record_cli_connects_without_reset_or_keyboard(monkeypatch, tmp_path):
    calls = []
    rpc = SimpleNamespace(close=lambda: calls.append("close"))
    monkeypatch.setattr(cli, "make_rpc_client", lambda endpoint: rpc)
    env = CameraEnv()

    def connect(client, *, reset_on_connect):
        assert client is rpc
        assert reset_on_connect is False
        return env

    def record(source, directory, **options):
        assert source is env
        assert directory == tmp_path
        calls.append("record")

    monkeypatch.setattr(cli, "LeKiwiEnvClient", connect)
    monkeypatch.setattr(cli, "record", record)
    cli.main(["--env-endpoint", "http://robot:8120", "--recording-dir", str(tmp_path)])
    assert calls == ["record", "close"]


@pytest.mark.parametrize("option", ["--no-video", "--no-actions", "--name"])
def test_record_cli_rejects_removed_options(option):
    with pytest.raises(SystemExit) as error:
        cli.main(["--env-endpoint", "http://robot:8120", option])
    assert error.value.code == 2


@pytest.mark.parametrize("ending", ["duration", "sigterm", "sigint"])
def test_record_process_starts_immediately_and_flushes_video(tmp_path, ending):
    import signal
    import subprocess
    import sys

    from rpent.utils.rpc.http_rpc import HttpRpcClient, HttpRpcServer

    pytest.importorskip("imageio_ffmpeg")
    camera = CameraEnv()
    calls = []

    def dispatch(method, args, kwargs):
        calls.append(method)
        return {
            "env.get_env_meta": lambda: {"action_dim": 3},
            "env.get_camera_meta": camera.get_camera_meta,
            "env.get_observation": camera.get_observation,
        }[method]()

    server = HttpRpcServer(("127.0.0.1", 0), dispatch)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    process = None
    try:
        endpoint = f"http://127.0.0.1:{server.server_address[1]}"
        with (tmp_path / "record.log").open("w") as log:
            process = subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    "robots.lekiwi.record",
                    "--env-endpoint",
                    endpoint,
                    "--recording-dir",
                    str(tmp_path / "videos"),
                    "--duration",
                    "1" if ending == "duration" else "60",
                ],
                stdout=log,
                stderr=log,
            )
            assert camera.read.wait(10)
            if ending != "duration":
                process.send_signal(
                    signal.SIGTERM if ending == "sigterm" else signal.SIGINT
                )
            assert process.wait(timeout=10) == 0
        directories = list((tmp_path / "videos").iterdir())
        assert len(directories) == 1
        directory = directories[0]
        assert {p.name for p in directory.iterdir()} == {
            "video.mp4",
            "camera_meta.json",
            "frames.jsonl",
        }
        with imageio.get_reader(directory / "video.mp4") as reader:
            assert reader.get_data(0).shape == (16, 16, 3)
        assert set(calls) == {
            "env.get_env_meta",
            "env.get_camera_meta",
            "env.get_observation",
        }
        env_rpc = HttpRpcClient(endpoint)
        try:
            assert env_rpc.call("env.get_env_meta")["action_dim"] == 3
        finally:
            env_rpc.close()
    finally:
        if process is not None and process.poll() is None:
            process.kill()
            process.wait(timeout=5)
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_record_reports_video_failure(tmp_path):
    env = SimpleNamespace(get_camera_meta=lambda: {}, get_observation=lambda: {})
    with pytest.raises(RuntimeError, match="Camera frame unavailable"):
        cli.record(env, tmp_path)
