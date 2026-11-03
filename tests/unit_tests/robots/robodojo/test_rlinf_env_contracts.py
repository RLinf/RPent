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

"""Agent-only observation, policy shape and CLI configuration contracts."""

from types import SimpleNamespace

import numpy as np
import pytest

from robots.robodojo import env_server
from robots.robodojo.scripted.eval import bridge_call


def test_server_arguments_reach_runtime_config():
    args = env_server.build_parser().parse_args(
        [
            "--task",
            "fill_pen_holder",
            "--layout",
            "8",
            "--random",
            "--env-cfg-type",
            "arx_x5",
            "--source-root",
            "/data/code",
            "--save-dir",
            "/data/output",
            "--cuda-device",
            "2",
            "--headless",
            "--max-episode-steps",
            "1200",
        ]
    )
    cfg = env_server.build_config(args)
    assert cfg.seed == 8 and not cfg.auto_reset
    assert cfg.task_config.task_name == "fill_pen_holder"
    assert cfg.task_config.source_root == "/data/code"
    assert cfg.task_config.save_dir == "/data/output"
    assert cfg.task_config.cuda_device == 2
    assert cfg.task_config.headless and cfg.task_config.random
    assert cfg.task_config.max_episode_steps == 1200
    assert (
        env_server.build_config(
            env_server.build_parser().parse_args([])
        ).task_config.source_root
        is None
    )
    with pytest.raises(ValueError, match="one environment"):
        env_server.build_config(
            env_server.build_parser().parse_args(["--num-envs", "2"])
        )


def test_policy_vectors_map_to_native_keys(agent_module):
    action = agent_module._policy_action(np.arange(14))
    assert list(action) == [
        "left_arm_joint_state",
        "right_arm_joint_state",
        "left_ee_joint_state",
        "right_ee_joint_state",
    ]
    assert [v.tolist() for v in action.values()] == [
        list(range(6)),
        list(range(6, 12)),
        [12],
        [13],
    ]
    assert agent_module._policy_action(action) is action


@pytest.mark.parametrize(
    "action",
    [
        np.zeros((1, 14)),
        np.zeros(13),
        np.zeros((50, 14)),
        np.full(14, np.nan),
    ],
)
def test_policy_shape_fails_before_motion(agent_module, make_agent, action):
    agent = make_agent(SimpleNamespace())
    with pytest.raises(ValueError, match="shape"):
        agent.step_native(action)


def test_camera_and_language_preserve_rpc_contract(make_agent):
    frames = []
    camera = {
        "color": np.zeros((2, 3, 3), dtype=np.uint8),
        "depth": np.ones((2, 3)),
        "intrinsic_matrix": np.eye(3),
        "extrinsic_matrix": np.eye(4),
    }
    env = SimpleNamespace(
        get_obs=lambda env_idx: {"vision": {"cam_head": camera}, "state": {}},
        obs_manager=SimpleNamespace(
            desc_manager=SimpleNamespace(
                get_one_description=lambda: ["Pick the object."],
            )
        ),
    )
    agent = make_agent(env, recorder=SimpleNamespace(record=frames.append))
    facade = env_server.RoboDojoEnvFacade(agent)
    meta = facade.get_camera_meta("cam_head")
    assert (meta["height"], meta["width"]) == (2, 3)
    np.testing.assert_array_equal(meta["intrinsic_matrix"], np.eye(3))
    np.testing.assert_array_equal(meta["extrinsic_matrix"], np.eye(4))
    rgb, depth = facade.render_camera("cam_head", depth=True)
    assert rgb.shape == (2, 3, 3) and depth.shape == (2, 3)
    assert (
        facade.get_obs()["instruction"]
        == facade.get_task_language()
        == "Pick the object."
    )
    assert len(frames) == 1
    with pytest.raises(IndexError):
        agent._sub_env(1)
    with pytest.raises(ValueError, match="unknown camera"):
        facade.get_camera_meta("missing")


def test_scripted_bridge_uses_current_facade_and_filters_native_results(make_agent):
    """Exercise the extracted service closure with only the native scene faked."""
    actions = []
    native = SimpleNamespace(
        take_action_cnt=[0],
        step_lim=200,
        is_success=lambda env_idx: True,
        get_obs=lambda env_idx: {
            "state": {"right_arm_joint_state": np.zeros(6), "reward": 1},
            "vision": {},
        },
        obs_manager=SimpleNamespace(
            desc_manager=SimpleNamespace(get_one_description=lambda: ["Pick it up."])
        ),
        reward_manager=SimpleNamespace(
            get_reward=lambda **kwargs: [1.0],
            get_score=lambda: [100.0],
            is_all_gripper_open=lambda **kwargs: True,
            all_robot_back_to_origin=lambda: False,
            check_once=lambda value, index: value,
        ),
    )

    def apply_action(action, kind, *, eval_fair):
        actions.append((action, kind, eval_fair))
        native.take_action_cnt[0] += 1

    agent = make_agent(
        native, slot=SimpleNamespace(env=native, apply_action=apply_action)
    )
    facade = env_server.RoboDojoEnvFacade(agent)

    class LocalRpc:
        def call(self, method, args=(), kwargs=None, **options):
            return facade._rpc[method](*args, **(kwargs or {}))

    action = {"right_arm_joint_state": [0.1] * 6}
    result = bridge_call(LocalRpc(), {"method": "step", "args": [action]})
    assert actions == [(action, "joint", False)]
    assert result["status"] == {"step": 1, "step_limit": 200}
    assert set(result["obs"]["state"]) == {"right_arm_joint_state"}
    assert result["obs"]["instruction"] == "Pick it up."
    assert facade.is_success() is True
    assert facade.get_reward_details()["reward"] == 1.0
