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


def test_dual_vla_prediction_follows_shared_component_rpc_contract(monkeypatch):
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
            return torch.ones((20, 20)), None

    def get_model(cfg, torch_dtype):
        assert cfg.action_dim == 20 and cfg.openpi.num_images_in_input == 3
        assert cfg.openpi_data.repo_id == "test/dataset"
        assert cfg.openpi.config_name == "pi05_dualfranka_tcp_rot6d"
        assert cfg.num_action_chunks == cfg.openpi.action_chunk == 20
        assert cfg.openpi.train_expert_only is False
        assert cfg.openpi.detach_critic_input is True
        return Model()

    loader = ModuleType("rlinf.models.embodiment.openpi")
    loader.get_model = get_model
    monkeypatch.setitem(sys.modules, loader.__name__, loader)
    facade = Pi05VLAFacade(
        model_path="/unused/checkpoint",
        embodiment="dual_franka",
        repo_id="test/dataset",
    )
    try:
        assert calls == []
        actions = facade._dispatch("vla.predict", ({},), {"options": {"mode": "eval"}})
        assert actions.shape == (20, 20) and actions.dtype == np.float32
        assert calls == ["eval"]
    finally:
        facade.close()


@pytest.mark.parametrize("embodiment", ["libero", "dual_franka"])
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
    assert received["model_backend"] == "openpi_pytorch"
    assert received["port"] == 6000


def test_libero_preset_keeps_existing_defaults():
    from rpent.robots.components.pi05_vla_server import (
        PI05_EMBODIMENTS,
        build_model_cfg,
    )

    cfg = build_model_cfg("/checkpoint", PI05_EMBODIMENTS["libero"])
    assert cfg.openpi.config_name == "pi05_libero"
    assert cfg.action_dim == 7
    assert cfg.num_action_chunks == cfg.openpi.action_chunk == 5
    assert cfg.openpi.num_images_in_input == 2
    assert cfg.openpi.train_expert_only is True
    assert cfg.openpi.detach_critic_input is None
    assert "openpi_data" not in cfg
