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

"""Exercise native RoboLab ownership without launching Isaac Sim."""

from pathlib import Path

from robots.rl.server import _WorkerSessionHost
from robots.robolab import episode_server


def test_worker_reuses_runtime_and_closes_each_scene(monkeypatch, tmp_path):
    events = []

    class Runtime:
        def __init__(self, *, cuda_device, scratch_dir):
            events.append(("application", cuda_device))

        def open_episode(self, *, task, seed, max_episode_steps, save_dir):
            events.append(("open", task, seed, max_episode_steps, save_dir))
            self.active = True
            return object()

        def close_episode(self):
            if getattr(self, "active", False):
                self.active = False
                events.append(("scene",))

        def close(self):
            events.append(("shutdown",))

    monkeypatch.setattr(episode_server, "RoboLabEpisodeRuntime", Runtime, raising=False)
    host = _WorkerSessionHost("robolab", None, 120.0, None, None, 4)
    host.start()
    host._build_env({"task_name": "BananaInBowlTask", "seed": 0}, 120)
    host.close()
    host._build_env(
        {"task_name": "FruitsOrangesOnPlateTask", "seed": 1},
        180,
        record_artifacts=True,
        output_dir=str(tmp_path),
    )
    host.close()
    host.shutdown()
    assert events == [
        ("application", 4),
        ("open", "BananaInBowlTask", 0, 120, None),
        ("scene",),
        ("open", "FruitsOrangesOnPlateTask", 1, 180, Path(tmp_path) / "native"),
        ("scene",),
        ("shutdown",),
    ]
