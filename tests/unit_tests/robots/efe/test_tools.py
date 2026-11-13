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

"""EFE raw tools report only public service data."""

from __future__ import annotations

import io

import imageio.v2 as imageio
import numpy as np
import pytest

from robots.efe.tools import EfeTools, dump_state, view_env_state
from rpent.session import EnvState
from tests.unit_tests.robots.efe.efe_support import FakeEfeClient


@pytest.fixture
def tools(tmp_path):
    client = FakeEfeClient()
    return EfeTools(client=client, state=EnvState(tmp_path)), client


def test_spec_reports_action_layout_and_kinematics(tools):
    service, _ = tools

    data = service.efe_spec().data

    assert data["robot"] == "ALOHA arm (6 joints, parallel gripper)"
    assert data["action_types"]["joint_position"]["dim"] == 7
    assert data["action_types"]["joint_position"]["limits"]["gripper"] == [0.002, 0.037]
    assert "not_yet" not in data["action_types"]
    assert data["cameras"] == ["table_cam_front", "table_cam_left", "wrist_cam"]
    assert data["tcp_frames"] == ["tcp"]
    assert data["budget"]["reset"] == 0


def test_observe_returns_images_and_named_proprioception(tools):
    service, client = tools

    result = service.efe_observe()
    data, images = result.data, result.images

    assert len(images) == 3
    assert set(data["proprio"]) >= {"q_waist", "tcp_x", "cmd_gripper"}
    assert data["done"] is False
    assert client.observe_calls == 1


def test_observe_depth_is_saved_as_an_artifact(tools, tmp_path):
    service, _ = tools

    result = service.efe_observe(cameras=["wrist_cam"], depth=True)

    saved = result.data["depth_paths"]["wrist_cam"]
    assert saved.endswith("depth_wrist_cam_ep000_s000000.npy")
    assert np.load(saved).shape == (4, 4)


def test_observe_keeps_the_labels_in_response_order(tools):
    service, _ = tools

    result = service.efe_observe(cameras=["wrist_cam", "table_cam_front"])

    assert result.data["cameras"] == ["wrist_cam", "table_cam_front"]
    assert len(result.images) == 2


def test_observe_saves_frames_for_view_env_state(tmp_path):
    client = FakeEfeClient()
    state = EnvState(tmp_path)
    service = EfeTools(client=client, state=state)

    service.efe_observe(cameras=["wrist_cam"])
    dump_state(
        client,
        state,
        command={},
        result={},
        elapsed_s=0.1,
        frames=service.latest_frames(),
    )
    view = view_env_state(-1, state=state)

    assert len(view.images) == 1
    assert "wrist_cam" in view.data["camera_images"]
    assert view.data["camera_images"]["wrist_cam"].endswith(
        "camera_wrist_cam_ep000_s000000.png"
    )


def test_view_state_without_an_observation_has_no_images(tmp_path):
    client = FakeEfeClient()
    state = EnvState(tmp_path)

    dump_state(client, state, command={}, result={}, elapsed_s=0.1)
    view = view_env_state(-1, state=state)

    assert view.images == []
    assert "camera_images" not in view.data


def test_a_reset_keeps_the_frames_of_the_previous_episode(tmp_path):
    class EpisodeFrames(FakeEfeClient):
        value = 0

        def observe(self, cameras=None, *, depth=False, video="frames"):
            result = super().observe(cameras=cameras, depth=depth, video=video)
            result.last = {
                name: np.full_like(frame, self.value)
                for name, frame in result.last.items()
            }
            return result

    client = EpisodeFrames({**FakeEfeClient().spec, "budget": {"reset": 2}})
    state = EnvState(tmp_path)
    service = EfeTools(client=client, state=state)

    service.efe_observe(cameras=["wrist_cam"])
    dump_state(
        client,
        state,
        command={},
        result={},
        elapsed_s=0.1,
        frames=service.latest_frames(),
    )
    client.value = 200
    client.reset()
    service.efe_observe(cameras=["wrist_cam"])
    dump_state(
        client,
        state,
        command={},
        result={},
        elapsed_s=0.1,
        frames=service.latest_frames(),
    )

    first = view_env_state(0, state=state)
    second = view_env_state(1, state=state)

    assert first.data["state"]["episode"] == 0
    assert second.data["state"]["episode"] == 1
    assert first.data["camera_images"]["wrist_cam"].endswith(
        "camera_wrist_cam_ep000_s000000.png"
    )
    assert second.data["camera_images"]["wrist_cam"].endswith(
        "camera_wrist_cam_ep001_s000000.png"
    )
    assert imageio.imread(io.BytesIO(first.images[0]))[0, 0, 0] == 0
    assert imageio.imread(io.BytesIO(second.images[0]))[0, 0, 0] == 200


def test_low_resolution_shrinks_the_returned_frame(tools):
    service, _ = tools

    result = service.efe_observe(cameras=["wrist_cam"], resolution="low")

    frame = imageio.imread(io.BytesIO(result.images[0]))
    assert frame.shape == (1, 1, 3)
    assert result.data["image_downsample"] == 4


def test_submit_reports_execution_and_episode_state(tools):
    service, client = tools
    rows = [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.02]]

    data = service.efe_submit("joint_position", rows, dt=0.02).data

    assert client.submitted[0][0] == "joint_position"
    assert data["executed_rows"] == 1
    assert data["control_steps_chunk"] == 1
    assert data["done"] is False


def test_status_reports_exactly_the_service_fields(tools):
    service, _ = tools

    data = service.efe_status().data

    assert data == {
        "step": 0,
        "calls": 0,
        "control_steps": 0,
        "done": False,
        "reason": None,
        "elapsed_s": 0.0,
    }


def test_reset_reports_the_new_episode(tmp_path):
    spec = {**FakeEfeClient().spec, "budget": {"reset": 5}}
    client = FakeEfeClient(spec)
    service = EfeTools(client=client, state=EnvState(tmp_path))

    data = service.efe_reset(cameras=["wrist_cam"]).data

    assert client.reset_calls == 1
    assert data["episode"] == 1
    assert data["resets_left"] == 4


def test_end_episode_reports_the_service_status(tmp_path):
    spec = {**FakeEfeClient().spec, "success_signal": "hidden"}
    client = FakeEfeClient(spec)
    service = EfeTools(client=client, state=EnvState(tmp_path))

    data = service.efe_end_episode().data

    assert client.finish_calls == 1
    assert data["ok"] is True
    assert data["reason"] == "finished"


def test_a_finished_episode_records_no_extra_observation(tmp_path):
    client = FakeEfeClient({**FakeEfeClient().spec, "success_signal": "hidden"})
    state = EnvState(tmp_path)
    service = EfeTools(client=client, state=state)
    service.efe_end_episode()
    client.last_meta = {}

    dump_state(
        client, state, command={"tool": "efe_end_episode"}, result={}, elapsed_s=0.1
    )

    assert client.observe_calls == 0
