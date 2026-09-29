# Copyright 2026 The RPent Authors.
# Licensed under the Apache License, Version 2.0 (the "License");

from __future__ import annotations

import sys
import types
from types import SimpleNamespace

import numpy as np
import pytest

from rpent.robots.components.action_model_protocol import ActionModelProtocolError
from scripts.wam.cosmos_policy_rpc_bridge import (
    CosmosPolicyBridge,
    normalize_cosmos_actions,
    resolve_cosmos_checkpoint,
)
from scripts.wam.dreamzero_rpc_bridge import (
    DROID_PROPRIO_SCHEMA,
    DreamZeroBridge,
)


def _install_dreamzero_client(monkeypatch, client_type: type) -> None:
    package = types.ModuleType("eval_utils")
    module = types.ModuleType("eval_utils.policy_client")
    module.WebsocketClientPolicy = client_type
    package.policy_client = module
    monkeypatch.setitem(sys.modules, "eval_utils", package)
    monkeypatch.setitem(sys.modules, "eval_utils.policy_client", module)


def test_cosmos_bridge_resolves_policy_file_inside_checkpoint_directory(
    tmp_path,
) -> None:
    checkpoint_dir = tmp_path / "cosmos-policy"
    checkpoint_dir.mkdir()
    policy_file = checkpoint_dir / "Cosmos-Policy-LIBERO-Predict2-2B.pt"
    policy_file.write_bytes(b"checkpoint")

    assert resolve_cosmos_checkpoint(str(checkpoint_dir)) == str(policy_file)


def test_cosmos_bridge_normalizes_native_action_lists() -> None:
    actions = normalize_cosmos_actions([[0.1] * 7, [0.2] * 7])
    assert actions.shape == (2, 7)
    assert actions.dtype == np.float32


def test_dreamzero_bridge_requires_session_aware_native_server(monkeypatch) -> None:
    class Client:
        def __init__(self, **kwargs):
            del kwargs

        def get_server_metadata(self):
            return {
                "n_external_cameras": 2,
                "needs_wrist_camera": True,
                "needs_session_id": False,
                "action_space": "joint_position",
            }

    _install_dreamzero_client(monkeypatch, Client)

    with pytest.raises(RuntimeError, match="session IDs"):
        DreamZeroBridge("127.0.0.1", 8000, "DreamZero-DROID")


def test_dreamzero_bridge_maps_explicit_droid_camera_and_proprio_schema(
    monkeypatch,
) -> None:
    created = []

    class Client:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.observations = []
            created.append(self)

        def get_server_metadata(self):
            return {
                "n_external_cameras": 2,
                "needs_wrist_camera": True,
                "needs_session_id": True,
                "action_space": "joint_position",
            }

        def infer(self, observation):
            self.observations.append(observation)
            return np.ones((2, 8), np.float32)

        def reset(self, options):
            del options

    _install_dreamzero_client(monkeypatch, Client)
    bridge = DreamZeroBridge("dreamzero", 8000, "DreamZero-DROID")
    result = bridge.predict(
        {
            "images": {
                "primary": np.zeros((4, 4, 3), np.uint8),
                "wrist": np.ones((4, 4, 3), np.uint8),
                "extra": [np.full((4, 4, 3), 2, np.uint8)],
            },
            "proprio": np.arange(14, dtype=np.float32),
            "instruction": "pick the cup",
            "embodiment": "droid_joint_position_8d",
            "metadata": {
                "proprio_schema": list(DROID_PROPRIO_SCHEMA),
                "episode_id": "episode-7",
            },
        }
    )

    assert result["actions"].shape == (2, 8)
    native = created[0].observations[0]
    np.testing.assert_array_equal(native["observation/joint_position"], np.arange(7))
    np.testing.assert_array_equal(
        native["observation/cartesian_position"], np.arange(7, 13)
    )
    np.testing.assert_array_equal(native["observation/gripper_position"], [13])
    assert native["session_id"] == "episode-7"
    assert native["prompt"] == "pick the cup"


def test_dreamzero_bridge_requires_episode_id_and_switches_native_session(
    monkeypatch,
) -> None:
    created = []

    class Client:
        def __init__(self, **kwargs):
            del kwargs
            self.observations = []
            created.append(self)

        def get_server_metadata(self):
            return {
                "n_external_cameras": 2,
                "needs_wrist_camera": True,
                "needs_session_id": True,
                "action_space": "joint_position",
            }

        def infer(self, observation):
            self.observations.append(observation)
            return np.ones((1, 8), np.float32)

        def reset(self, options):
            del options

    _install_dreamzero_client(monkeypatch, Client)
    bridge = DreamZeroBridge("dreamzero", 8000, "DreamZero-DROID")
    request = {
        "images": {
            "primary": np.zeros((4, 4, 3), np.uint8),
            "wrist": np.ones((4, 4, 3), np.uint8),
            "extra": [np.full((4, 4, 3), 2, np.uint8)],
        },
        "proprio": np.arange(14, dtype=np.float32),
        "instruction": "pick the cup",
        "embodiment": "droid_joint_position_8d",
        "metadata": {"proprio_schema": list(DROID_PROPRIO_SCHEMA)},
    }

    with pytest.raises(ActionModelProtocolError, match="episode_id"):
        bridge.predict(request)

    for episode_id in ("episode-1", "episode-2"):
        request["metadata"]["episode_id"] = episode_id
        bridge.predict(request)
    assert [obs["session_id"] for obs in created[0].observations] == [
        "episode-1",
        "episode-2",
    ]


def test_cosmos_bridge_uses_official_preprocessing_and_preserves_aux_outputs(
    monkeypatch,
) -> None:
    calls = []

    class PolicyEvalConfig(SimpleNamespace):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)

    def get_action(cfg, model, stats, observation, instruction, **kwargs):
        calls.append((cfg, model, stats, observation, instruction, kwargs))
        return {
            "actions": np.ones((16, 7), np.float32),
            "future_image_predictions": {"future_image": np.zeros((4, 4, 3))},
            "value_prediction": np.asarray(0.6, np.float32),
        }

    cosmos_utils = types.ModuleType("cosmos_policy.experiments.robot.cosmos_utils")
    cosmos_utils.get_action = get_action
    cosmos_utils.get_model = lambda cfg: ("model", "config")
    cosmos_utils.load_dataset_stats = lambda path: {"path": path}
    cosmos_utils.init_t5_text_embeddings_cache = lambda path: None
    cosmos_utils.t5_text_embeddings_cache = {
        "pick the bowl": np.zeros((1, 512, 1024), np.float32)
    }
    libero_eval = types.ModuleType(
        "cosmos_policy.experiments.robot.libero.run_libero_eval"
    )
    libero_eval.PolicyEvalConfig = PolicyEvalConfig
    monkeypatch.setitem(
        sys.modules,
        "cosmos_policy.experiments.robot.cosmos_utils",
        cosmos_utils,
    )
    monkeypatch.setitem(
        sys.modules,
        "cosmos_policy.experiments.robot.libero.run_libero_eval",
        libero_eval,
    )

    bridge = CosmosPolicyBridge("nvidia/Cosmos-Policy-LIBERO-Predict2-2B")
    primary = np.arange(4 * 4 * 3, dtype=np.uint8).reshape(4, 4, 3)
    wrist = primary + 1
    proprio = np.array(
        [0.03, -0.03, 0.1, 0.2, 0.3, 0.0, 0.0, 0.0, 1.0],
        dtype=np.float32,
    )
    proprio_schema = [
        "gripper_qpos_0",
        "gripper_qpos_1",
        "eef_x",
        "eef_y",
        "eef_z",
        "eef_quat_x",
        "eef_quat_y",
        "eef_quat_z",
        "eef_quat_w",
    ]
    result = bridge.predict(
        {
            "images": {
                "primary": primary,
                "wrist": wrist,
            },
            "proprio": proprio,
            "instruction": "pick the bowl",
            "embodiment": "libero_7d",
            "metadata": {"proprio_schema": proprio_schema},
        }
    )

    assert result["actions"].shape == (16, 7)
    assert result["value"] == pytest.approx(0.6)
    assert "future_image" in result["future_observation"]
    np.testing.assert_array_equal(calls[0][3]["primary_image"], np.flipud(primary))
    np.testing.assert_array_equal(calls[0][3]["wrist_image"], np.flipud(wrist))
    np.testing.assert_array_equal(calls[0][3]["proprio"], proprio)
    assert bridge.get_capabilities()["proprio_schema"] == proprio_schema
    assert calls[0][4] == "pick the bowl"
    assert calls[0][5]["generate_future_state_and_value_in_parallel"] is True


def test_cosmos_bridge_rejects_instruction_missing_from_official_cache(
    monkeypatch,
) -> None:
    class PolicyEvalConfig(SimpleNamespace):
        pass

    cosmos_utils = types.ModuleType("cosmos_policy.experiments.robot.cosmos_utils")
    cosmos_utils.get_action = lambda *args, **kwargs: pytest.fail(
        "cache misses must not invoke the online T5 fallback"
    )
    cosmos_utils.get_model = lambda cfg: ("model", "config")
    cosmos_utils.load_dataset_stats = lambda path: {"path": path}
    cosmos_utils.init_t5_text_embeddings_cache = lambda path: None
    cosmos_utils.t5_text_embeddings_cache = {
        "put the bowl on the plate": np.zeros((1, 512, 1024), np.float32)
    }
    libero_eval = types.ModuleType(
        "cosmos_policy.experiments.robot.libero.run_libero_eval"
    )
    libero_eval.PolicyEvalConfig = PolicyEvalConfig
    monkeypatch.setitem(
        sys.modules,
        "cosmos_policy.experiments.robot.cosmos_utils",
        cosmos_utils,
    )
    monkeypatch.setitem(
        sys.modules,
        "cosmos_policy.experiments.robot.libero.run_libero_eval",
        libero_eval,
    )

    bridge = CosmosPolicyBridge("nvidia/Cosmos-Policy-LIBERO-Predict2-2B")
    request = {
        "images": {
            "primary": np.zeros((4, 4, 3), np.uint8),
            "wrist": np.ones((4, 4, 3), np.uint8),
        },
        "proprio": np.array(
            [0.03, -0.03, 0.1, 0.2, 0.3, 0.0, 0.0, 0.0, 1.0],
            dtype=np.float32,
        ),
        "instruction": "a paraphrase that is not cached",
        "embodiment": "libero_7d",
        "metadata": {
            "proprio_schema": [
                "gripper_qpos_0",
                "gripper_qpos_1",
                "eef_x",
                "eef_y",
                "eef_z",
                "eef_quat_x",
                "eef_quat_y",
                "eef_quat_z",
                "eef_quat_w",
            ]
        },
    }

    with pytest.raises(
        ActionModelProtocolError,
        match="not present in the precomputed T5 embedding cache",
    ):
        bridge.predict(request)
