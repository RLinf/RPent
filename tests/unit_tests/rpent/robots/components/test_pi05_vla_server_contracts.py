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

import numpy as np
import pytest

from rpent.robots.components.pi05_vla_client import Pi05VLAClient
from rpent.robots.components.pi05_vla_server import (
    PI05_EMBODIMENTS,
    Pi05VLAFacade,
    build_model_cfg,
)
from rpent.robots.components.vla_facade_base import BaseVLAFacade


@pytest.mark.parametrize("embodiment", ["dual_franka", "yam"])
def test_vla_prediction_follows_shared_component_rpc_contract(monkeypatch, embodiment):
    import sys
    from types import ModuleType

    import torch

    from rpent.robots.components.pi05_vla_server import Pi05VLAFacade

    calls = []

    class Model:
        def cuda(self):
            return self

        def eval(self):
            return self

        def predict_action_batch(self, obs, mode):
            calls.append(mode)
            shape = (20, 20) if embodiment == "dual_franka" else (1, 30, 14)
            return torch.ones(shape), None

    def get_model(cfg, torch_dtype):
        assert cfg.openpi_data.repo_id == "test/dataset"
        assert cfg.pi05 is True
        assert cfg.openpi.task == "eval"
        assert cfg.openpi.model_action_dim == 32
        assert cfg.openpi.paligemma_variant == "gemma_2b"
        assert cfg.openpi.action_expert_variant == "gemma_300m"
        if embodiment == "dual_franka":
            assert cfg.action_dim == 20 and cfg.openpi.num_images_in_input == 3
            assert cfg.openpi.config_name == "pi05_dualfranka_tcp_rot6d"
            assert cfg.num_action_chunks == cfg.openpi.action_chunk == 20
            assert cfg.openpi.train_expert_only is False
            assert cfg.openpi.detach_critic_input is True
        else:
            assert cfg.action_dim == 14 and cfg.num_action_chunks == 30
            assert cfg.openpi.config_name == "pi05_yam_joint"
            assert cfg.openpi.num_images_in_input == 3
        return Model()

    loader = ModuleType("rlinf.models.embodiment.openpi")
    loader.get_model = get_model
    monkeypatch.setitem(sys.modules, loader.__name__, loader)
    facade = Pi05VLAFacade(
        model_path="/unused/checkpoint",
        embodiment=embodiment,
        repo_id="test/dataset",
    )
    try:
        assert calls == []
        actions = facade._dispatch("vla.predict", ({},), {"options": {"mode": "eval"}})
        expected_shape = (20, 20) if embodiment == "dual_franka" else (1, 30, 14)
        assert actions.shape == expected_shape and actions.dtype == np.float32
        assert calls == ["eval"]
    finally:
        facade.close()


@pytest.mark.parametrize("embodiment", ["libero", "dual_franka", "robodojo", "yam"])
def test_cli_forwards_model_configuration(monkeypatch, embodiment):
    import sys
    from types import SimpleNamespace

    from rpent.robots.components import pi05_vla_server as server

    received = {}
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "pi05_vla_server",
            "--embodiment",
            embodiment,
            "--model-path",
            "/checkpoint",
            "--repo-id",
            "test/dataset",
            "--norm-stats-path",
            "/stats",
            "--port",
            "6000",
        ],
    )

    def facade(**kwargs):
        received.update(kwargs)
        return SimpleNamespace(serve=lambda **options: received.update(options))

    monkeypatch.setattr(server, "Pi05VLAFacade", facade)
    server.main()
    assert received["embodiment"] == embodiment
    assert received["repo_id"] == "test/dataset"
    assert received["norm_stats_path"] == "/stats"
    assert received["port"] == 6000


def test_libero_preset_keeps_existing_defaults():
    from rpent.robots.components.pi05_vla_server import (
        PI05_EMBODIMENTS,
        build_model_cfg,
    )

    cfg = build_model_cfg("/checkpoint", PI05_EMBODIMENTS["libero"])
    assert cfg.pi05 is True
    assert cfg.openpi.task == "eval"
    assert cfg.openpi.config_name == "pi05_libero"
    assert cfg.action_dim == 7
    assert cfg.num_action_chunks == cfg.openpi.action_chunk == 5
    assert cfg.openpi.num_images_in_input == 2
    assert cfg.openpi.model_action_dim == 32
    assert cfg.openpi.paligemma_variant == "gemma_2b"
    assert cfg.openpi.action_expert_variant == "gemma_300m"
    assert cfg.openpi.train_expert_only is True
    assert cfg.openpi.detach_critic_input is None
    assert "openpi_data" not in cfg


def test_robodojo_preset_uses_full_horizon_and_joint_actions():
    from rpent.robots.components.pi05_vla_server import (
        PI05_EMBODIMENTS,
        build_model_cfg,
    )

    cfg = build_model_cfg("/checkpoint", PI05_EMBODIMENTS["robodojo"])
    assert cfg.openpi.config_name == "pi05_robodojo_arx_x5"
    assert cfg.openpi.task == "eval"
    assert cfg.openpi.model_action_dim == 32
    assert cfg.openpi.paligemma_variant == "gemma_2b"
    assert cfg.openpi.action_expert_variant == "gemma_300m"
    assert cfg.openpi.discrete_state_input is True
    assert cfg.openpi.torch_compile is False
    assert cfg.action_dim == cfg.openpi.action_env_dim == 14
    assert cfg.num_action_chunks == cfg.openpi.action_chunk == 50
    assert cfg.num_steps == cfg.openpi.num_steps == 5
    assert cfg.openpi.num_images_in_input == 3
    assert cfg.use_proprio is True
    assert cfg.add_value_head is cfg.openpi.add_value_head is False


def test_robodojo_cli_resolves_checkpoint_from_environment(monkeypatch):
    import sys
    from types import SimpleNamespace

    from rpent.robots.components import pi05_vla_server as server

    received = {}
    monkeypatch.setenv("PI05_CHECKPOINT_PATH", "/test/robodojo-checkpoint")
    monkeypatch.setattr(sys, "argv", ["pi05_vla_server", "--embodiment", "robodojo"])

    def facade(**kwargs):
        received.update(kwargs)
        return SimpleNamespace(serve=lambda **options: None)

    monkeypatch.setattr(server, "Pi05VLAFacade", facade)
    server.main()
    assert received["model_path"] == "/test/robodojo-checkpoint"


def _yam_observation():
    return {
        "main_images": np.full((4, 5, 3), 1, dtype=np.uint8),
        "extra_view_images": np.stack(
            [
                np.full((4, 5, 3), 2, dtype=np.uint8),
                np.full((4, 5, 3), 3, dtype=np.uint8),
            ]
        ),
        "states": np.arange(14, dtype=np.float32),
        "task_descriptions": "put the cube away",
    }


def test_shared_yam_client_sends_three_views_and_qpos14_to_pi05():
    class FakeRpc:
        def call(self, method, *, args, timeout_s):
            assert method == "vla.predict" and timeout_s > 0
            observation, options = args
            assert options is None
            assert observation["main_images"].shape == (1, 4, 5, 3)
            assert observation["extra_view_images"].shape == (1, 2, 4, 5, 3)
            assert observation["wrist_images"] is None
            assert observation["states"].shape == (1, 14)
            assert observation["task_descriptions"] == ["put the cube away"]
            assert [
                int(view.min()) for view in observation["extra_view_images"][0]
            ] == [
                2,
                3,
            ]
            return np.zeros((1, 30, 14), dtype=np.float32)

    actions = Pi05VLAClient(FakeRpc(), embodiment="yam").predict(_yam_observation())
    assert actions.shape == (30, 14)


@pytest.mark.parametrize(
    "change, message",
    [
        ({"main_images": np.zeros((1, 4, 5, 3), dtype=np.uint8)}, "main_images"),
        (
            {"extra_view_images": np.zeros((1, 4, 5, 3), dtype=np.uint8)},
            "extra_view_images",
        ),
        ({"states": np.zeros(13, dtype=np.float32)}, "states"),
        ({"states": np.full(14, np.nan, dtype=np.float32)}, "states"),
        ({"task_descriptions": ""}, "task_descriptions"),
    ],
)
def test_shared_yam_client_rejects_invalid_robot_observation(change, message):
    observation = _yam_observation() | change
    with pytest.raises(ValueError, match=message):
        Pi05VLAClient(None, embodiment="yam").encode_obs(observation)


def test_shared_yam_server_preset_matches_joint_policy():
    cfg = build_model_cfg("/checkpoint", PI05_EMBODIMENTS["yam"])
    assert cfg.model_path == "/checkpoint"
    assert (cfg.action_dim, cfg.num_action_chunks) == (14, 30)
    assert cfg.openpi.config_name == "pi05_yam_joint"
    assert cfg.openpi.num_images_in_input == 3
    assert cfg.openpi.discrete_state_input is True


@pytest.mark.parametrize("embodiment", ["yam", "libero", "franka", "dual_franka"])
def test_shared_pi05_health_and_model_metadata_are_separate(embodiment):
    facade = Pi05VLAFacade.__new__(Pi05VLAFacade)
    facade._embodiment = embodiment
    BaseVLAFacade.__init__(facade)
    assert facade._dispatch("healthz", (), {}) == {"status": "ok"}
    if embodiment == "yam":
        from robots.yam.contracts import vla_runtime_contract

        assert facade._dispatch("vla.get_model_meta", (), {}) == vla_runtime_contract()
        assert "vla.get_model_meta" in facade._readonly_methods
    else:
        with pytest.raises(ValueError, match="unknown RPC method"):
            facade._dispatch("vla.get_model_meta", (), {})
