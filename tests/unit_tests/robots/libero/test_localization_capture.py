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

"""LIBERO localization capture RPC and per-step artifacts."""

from __future__ import annotations

import base64
import io
from types import SimpleNamespace

import imageio.v2 as imageio
import numpy as np
import pytest

from robots.libero import tools
from robots.libero.env_server import LiberoEnvFacade
from robots.libero.localization_demo import run_demo
from rpent.robots.components.env_client_base import BaseEnvClient
from rpent.robots.components.sam3_client import Sam3Client
from rpent.session import EnvState
from rpent.tools.localization.localization import LocalizationFrame
from rpent.tools.perception import Sam3Tools


class _RlinfLibero:
    """RLinf LiberoEnv surface used by the facade, rendering a flat floor."""

    def __init__(self):
        self.cfg = SimpleNamespace(
            init_params=SimpleNamespace(camera_heights=4, camera_widths=6)
        )
        self.elapsed_steps = np.array([7])
        self.calls = []

    def render_camera(self, camera_name, height, width, depth):
        self.calls.append(("render", camera_name, height, width))
        rgb = np.zeros((height, width, 3), np.uint8)
        rgb[..., 1] = np.arange(height, dtype=np.uint8)[:, None]  # OpenGL row
        buffer = np.full((height, width), 0.5, np.float32)
        buffer[0, 0] = 1.0  # nothing drawn
        return rgb, buffer

    def get_camera_meta(self, camera_name, height, width):
        self.calls.append(("meta", camera_name, height, width))
        pose = np.eye(4)
        pose[:3, 3] = [1.0, 2.0, 3.0]
        return {
            "camera_name": camera_name,
            "height": height,
            "width": width,
            "intrinsic_K": [[5.0, 0, width / 2], [0, 5.0, height / 2], [0, 0, 1]],
            "extrinsic_cam2world": pose.tolist(),
            "depth_near": 0.1,
            "depth_far": 10.0,
        }

    def step(self, *_args, **_kwargs):
        raise AssertionError("capture must not step")


def test_capture_rpc_normalizes_each_view_from_one_state():
    env = _RlinfLibero()
    facade = LiberoEnvFacade(env, meta={})
    assert "env.capture_localization" not in facade._readonly_methods
    records = facade.capture_localization(["agentview", ("robot0_eye_in_hand", 8, 8)])
    frames = [LocalizationFrame.from_canonical(record) for record in records]
    assert [call[0] for call in env.calls] == ["render", "meta", "render", "meta"]
    assert env.calls[0][2:] == (4, 6) and env.calls[2][2:] == (8, 8)
    agent = frames[0]
    np.testing.assert_array_equal(agent.rgb[:, 0, 1], [3, 2, 1, 0])
    assert not agent.valid_mask[-1, 0] and agent.valid_mask.sum() == 4 * 6 - 1
    np.testing.assert_allclose(agent.intrinsic_K[:2, 2], [2.5, 1.5])
    depth = 0.1 / (1 - 0.5 * (1 - 0.1 / 10))
    np.testing.assert_allclose(agent.depth_m[0, 0], depth, rtol=1e-6)
    assert agent.metadata["env_step"] == 7
    assert agent.metadata["environment"] == "libero"


@pytest.mark.parametrize("high_res_fails", [False, True])
def test_step_artifacts_share_one_capture_and_keep_existing_names(
    tmp_path, high_res_fails
):
    frames = []
    for height, width in ((4, 6), (4, 6), (8, 8), (8, 8)):
        frames.append(
            LocalizationFrame.from_canonical(
                {
                    "camera": "cam",
                    "rgb": np.full((height, width, 3), 9, np.uint8),
                    "depth_m": np.where(
                        np.arange(height * width).reshape(height, width) == 0,
                        np.nan,
                        1.5,
                    ),
                    "intrinsic_K": [[5.0, 0, 2], [0, 5.0, 1], [0, 0, 1]],
                    "T_world_from_cv": np.eye(4),
                }
            )
        )
    requests = []

    def capture_frames(cameras):
        requests.append(cameras)
        if high_res_fails and len(cameras) > 2:
            raise RuntimeError("high-resolution render failed")
        return frames[: len(cameras)]

    primitives = SimpleNamespace(
        env=SimpleNamespace(capture_frames=capture_frames),
        _last_obs={"main_images": np.zeros((4, 6, 3), np.uint8)},
    )
    raw = {
        "agentview_image": np.zeros((4, 6, 3), np.uint8),
        "robot0_eye_in_hand_image": np.zeros((4, 6, 3), np.uint8),
    }
    state = EnvState(tmp_path)
    with state.record_step(state={}) as step:
        tools._save_observation_artifacts(primitives, state, step, raw)
    low = [("agentview", 4, 6), ("robot0_eye_in_hand", 4, 6)]
    high = [("agentview", 1024, 1024), ("robot0_eye_in_hand", 1024, 1024)]
    # A failed high-resolution render keeps the low-resolution localization.
    assert requests == ([low + high, low] if high_res_fails else [low + high])
    record = state.get(0)
    for stem in ("agentview", "wrist"):
        assert {
            f"{stem}.png",
            f"{stem}_depth.npz",
            f"{stem}_world.npz",
            f"{stem}_valid_mask.npz",
            f"{stem}_metadata.json",
        } <= record.artifacts
        world = state.load(f"{stem}_world.npz", step=0)
        assert world.dtype == np.float32 and np.isnan(world[0, 0]).all()
        high_names = {
            f"{stem}_high.png",
            f"{stem}_world_high.npz",
            f"{stem}_valid_mask_high.npz",
            f"{stem}_metadata_high.json",
        }
        if high_res_fails:
            assert high_names.isdisjoint(record.artifacts)
            continue
        assert high_names <= record.artifacts
        assert state.load(f"{stem}_world_high.npz", step=0).dtype == np.float32
        mask = state.load(f"{stem}_valid_mask_high.npz", step=0)
        assert mask.dtype == np.bool_ and not mask[0, 0]
        metadata = state.load(f"{stem}_metadata_high.json", step=0)
        assert metadata["height"] == 8 and metadata["intrinsic_K"][0][2] == 2
    assert "agentview_policy.png" in record.artifacts


@pytest.mark.parametrize("detection", ["found", "not_found", "wrong_grid"])
def test_demo_captures_segments_and_localizes_without_advancing(tmp_path, detection):
    env = _RlinfLibero()
    facade = LiberoEnvFacade(env, meta={})
    env_calls = []
    sam3_calls = []

    def env_call(method, args=(), kwargs=None, **_options):
        env_calls.append(method)
        return facade._dispatch(method, args, kwargs or {})

    def sam3_call(method, args=(), kwargs=None, **_options):
        sam3_calls.append(method)
        assert method == "sam3.segment" and kwargs["text_prompt"] == "cup"
        rgb = imageio.imread(io.BytesIO(base64.b64decode(kwargs["image_base64"])))
        # The model sees the normalized top-down image, not a policy image.
        np.testing.assert_array_equal(rgb[:, 0, 1], [3, 2, 1, 0])
        if detection == "not_found":
            return {"found": False, "reason": "no cup"}
        shape = (3, 6) if detection == "wrong_grid" else (4, 6)
        buffer = io.BytesIO()
        imageio.imwrite(buffer, np.full(shape, 255, np.uint8), format="png")
        return {
            "found": True,
            "score": 0.9,
            "mask_shape": list(shape),
            "mask_png_base64": base64.b64encode(buffer.getvalue()).decode("ascii"),
        }

    client = BaseEnvClient(
        SimpleNamespace(call=env_call), expected_meta={}, reset_on_connect=False
    )
    perception = Sam3Tools(Sam3Client(SimpleNamespace(call=sam3_call)))
    output = tmp_path / "demo"
    result = run_demo(client, perception, output, prompt="cup", height=4, width=6)
    assert env_calls == ["env.get_env_meta", "env.capture_localization"]
    assert sam3_calls == ["sam3.segment"]
    assert env.elapsed_steps.tolist() == [7]
    with np.load(output / "world.npz" / "00.npz") as archive:
        world = archive["array"]
    assert np.isnan(world[-1, 0]).all()
    assert (output / "result.json" / "00.json").is_file()
    if detection == "found":
        assert result["n_pixels"] == 24 and result["n_valid"] == 23
        # K, positive camera Z and the translated pose give this known median.
        assert result["world_xyz"] == [1.0198, 1.9802, 3.198]
        assert (output / "overlay.png" / "00.png").is_file()
    else:
        assert result["world_xyz"] is None
        expected = "shape mismatch" if detection == "wrong_grid" else "no cup"
        assert expected in result["world_error"]
    # Re-running must preserve the previous demo's files.
    with pytest.raises(FileExistsError):
        run_demo(client, perception, output, prompt="cup", height=4, width=6)
    assert len(sam3_calls) == 1
