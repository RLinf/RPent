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

"""No hardware: real RPC, Toolkit, RGB-D projection, generation and replay.

The grounder locates a red patch deterministically instead of loading Molmo.
The two software arms use a nonidentity inter-base rotation to expose frame bugs.
"""

import base64
import io
import json
import threading
from contextlib import contextmanager

import numpy as np
import pytest
from PIL import Image

from robots.dual_franka.env_client import DualFrankaEnvClient
from robots.dual_franka.toolkit import DualFrankaToolkit
from robots.franka import runtime_config
from robots.franka.task_card.generate import generate
from robots.franka.task_card.replay import replay
from rpent.dashboard.events import NullDashboardEventSink
from rpent.memory import MemoryManager
from rpent.robots.components.molmo_client import MolmoClient
from rpent.utils.daemon import pick_free_port
from rpent.utils.rpc import RpcFacade, make_rpc_client, wait_for_ready


class SoftwareRig(RpcFacade):
    def __init__(self):
        super().__init__()
        self.shift = 0
        self.fail = False
        self.depth_valid = True
        self.calls = []
        self.ground_calls = 0
        self.poses = {}
        for name in (
            "reset",
            "get_env_meta",
            "get_observation",
            "get_robot_state",
            "get_camera_meta",
            "move_delta",
        ):
            self._rpc[f"env.{name}"] = getattr(self, name)
        self._rpc["molmo.ground"] = self.ground

    def reset(self):
        self.poses = {
            "left": np.array([0.4, 0, 0.4, 0, 0, 0, 1.0]),
            "right": np.array([0.4, 0, 0.4, 0, 0, 0, 1.0]),
        }
        return {"states": np.zeros(20)}

    def get_env_meta(self):
        return {"action_dim": 20}

    def get_robot_state(self):
        return {
            f"{arm}_arm": {"tcp_pose": pose.tolist(), "tcp_pose_frame": "right_base"}
            for arm, pose in self.poses.items()
        }

    def get_observation(self):
        rgb = np.zeros((64, 64, 3), dtype=np.uint8)
        rgb[18:23, 18 + self.shift : 23 + self.shift, 0] = 255
        return {
            "raw_camera_frames": {"base_0_rgb": rgb},
            "raw_camera_depths": {
                "base_0_rgb": np.full(
                    (64, 64), 0.3 if self.depth_valid else 0, dtype=np.float32
                )
            },
        }

    def get_camera_meta(self):
        return {
            "projection_views": {
                "base": {"raw_key": "base_0_rgb", "calibration_key": "base_camera"}
            },
            "base_0_rgb": {
                "color_intrinsics": {"fx": 100.0, "fy": 100.0, "ppx": 20.0, "ppy": 20.0}
            },
        }

    def move_delta(self, arm, delta_xyz):
        self.calls.append((arm, np.asarray(delta_xyz).tolist()))
        if not self.fail:
            self.poses[arm][:3] += delta_xyz
        return {"ok": not self.fail, "states": np.zeros(20)}

    def ground(self, image_base64, query):
        self.ground_calls += 1
        assert query == "red target"
        image = np.asarray(Image.open(io.BytesIO(base64.b64decode(image_base64))))
        rows, cols = np.where(image[..., 0] == 255)
        return {
            "point_xy": [float(cols.mean()), float(rows.mean())],
            "image_size": [64, 64],
        }


@contextmanager
def server(transport):
    rig = SoftwareRig()
    port = pick_free_port()
    worker = threading.Thread(
        target=rig.serve,
        kwargs={"transport": transport, "host": "127.0.0.1", "port": port},
        daemon=True,
    )
    worker.start()
    client = make_rpc_client(f"{transport}://127.0.0.1:{port}")
    try:
        wait_for_ready(client, timeout_s=5, poll_interval_s=0.01)
        yield rig, client
    finally:
        try:
            client.call("shutdown", timeout_s=2)
        finally:
            worker.join(timeout=5)
            client.close()
        assert not worker.is_alive()


@pytest.mark.parametrize("transport", ["http", "socket"])
@pytest.mark.parametrize(
    "case", ["success", "human_failure", "arm_failure", "invalid_depth", "abort"]
)
def test_generate_and_replay_through_real_rpc(tmp_path, monkeypatch, transport, case):
    # x_right = .1 - y_left; y_right = x_left.
    transform = [[0, -1, 0, 0.1], [1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
    workspace = {
        "ee_pose_limit_min": [[-1, -1, 0]] * 2,
        "ee_pose_limit_max": [[1, 1, 1]] * 2,
    }
    config = tmp_path / "robot.json"
    config.write_text(
        json.dumps(
            {
                "workspace": workspace,
                "perception": {
                    "base_frames": {"T_right_base_left_base": {"matrix": transform}}
                },
            }
        )
    )
    calibration = tmp_path / "calibration.json"
    calibration.write_text(
        json.dumps(
            {
                "base_camera": {
                    "transformation": {
                        "matrix": [
                            [1, 0, 0, 0.4],
                            [0, 1, 0, 0],
                            [0, 0, 1, 0],
                            [0, 0, 0, 1],
                        ]
                    }
                }
            }
        )
    )
    # Patch only file selection; production projection/calibration math runs unchanged.
    monkeypatch.setattr(runtime_config, "get_robot_config_path", lambda *a: config)
    monkeypatch.setattr(runtime_config, "get_calibration_path", lambda: calibration)
    from robots.dual_franka import perception
    from robots.franka.task_card import common

    for module in (perception, common):
        monkeypatch.setattr(module, "get_robot_config_path", lambda *a: config)
        monkeypatch.setattr(module, "get_calibration_path", lambda: calibration)

    def toolkit(root, rpc):
        monkeypatch.setattr("robots.franka.toolkit.get_output_dir", lambda: root)
        return DualFrankaToolkit(
            primitives_kwargs={
                "env": DualFrankaEnvClient(rpc),
                "model": None,
                "task_description": "test",
            },
            dashboard_events=NullDashboardEventSink(),
            memory=MemoryManager(tmp_path / "memory"),
        )

    with server(transport) as (rig, rpc):
        source = toolkit(tmp_path / "source", rpc)
        for arm in ("left", "right"):
            source.execute_tool("move_delta", {"arm": arm, "delta_xyz": [0.02, 0, 0]})
        molmo = MolmoClient(rpc)
        card = generate(
            tmp_path / "source",
            {str(i): {"phrase": "red target", "camera": "base"} for i in (1, 2)},
            robot="dual_franka",
            task="dual_franka_t0",
            molmo=molmo,
            human_verdict="success",
        )
        assert rig.ground_calls == 2
        (tmp_path / "card.json").write_text(json.dumps(card, indent=2))
        rig.shift = 10  # 10 pixels * 0.3m / 100px = 3cm along right-base X.
        target = toolkit(tmp_path / "replay", rpc)
        target.task_card_options = {"card": card}
        rig.calls.clear()
        rig.fail = case == "arm_failure"
        rig.depth_valid = case != "invalid_depth"
        answers = iter(
            [
                "abort" if case == "abort" else "start",
                "failure" if case == "human_failure" else "success",
            ]
        )

        def run():
            return replay(
                target, card, molmo, workspace, human=lambda *a: next(answers)
            )

        if case in {"arm_failure", "invalid_depth"}:
            with pytest.raises((ValueError, RuntimeError)):
                run()
        else:
            outcome = run()
            assert outcome["done"] == (case == "success")
        saved = json.loads((tmp_path / "replay/task_card_outcome.json").read_text())
        assert saved["done"] == (case == "success")
        assert target.solved() == saved["done"]
        if case in {"invalid_depth", "abort"}:
            assert rig.calls == []
        else:
            assert rig.calls[0][0] == "left"
            assert rig.calls[0][1] == pytest.approx([0.05, 0, 0], abs=1e-6)
            if case == "arm_failure":
                assert len(rig.calls) == 1
            else:
                assert rig.calls[1][0] == "right"
                assert rig.calls[1][1] == pytest.approx([0.05, 0, 0], abs=1e-6)
        assert (tmp_path / "replay/task_card_recipe.jsonl").exists()
        assert (tmp_path / "replay/base.png/00.png").exists()
