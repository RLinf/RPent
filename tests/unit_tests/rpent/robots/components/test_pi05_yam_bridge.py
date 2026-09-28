# Copyright 2026 The RPent Authors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0

import numpy as np
import pytest

from rpent.robots.components.pi05_vla_client import Pi05VLAClient
from rpent.robots.components.pi05_vla_server import (
    PI05_EMBODIMENTS,
    build_model_cfg,
)


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
    assert cfg.openpi.discrete_state_input is True
