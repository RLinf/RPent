# Copyright 2026 The RPent Authors.
# Licensed under the Apache License, Version 2.0.

from robots.yam.contracts import (
    MODEL_SPEC,
    YAM_VLA_CONTRACT,
    env_runtime_contract,
    vla_runtime_contract,
)


def test_model_spec_values_come_from_the_vla_contract() -> None:
    assert MODEL_SPEC.policy_name == YAM_VLA_CONTRACT["policy_name"]
    assert list(MODEL_SPEC.camera_order) == YAM_VLA_CONTRACT["camera_order"]
    assert MODEL_SPEC.state_layout == YAM_VLA_CONTRACT["state_layout"]
    assert MODEL_SPEC.action_layout == YAM_VLA_CONTRACT["action_layout"]
    assert MODEL_SPEC.action_horizon == YAM_VLA_CONTRACT["action_horizon"]
    assert MODEL_SPEC.use_length == YAM_VLA_CONTRACT["use_length"]


def test_control_hz_is_local_and_not_negotiated_with_the_server() -> None:
    assert "control_hz" not in YAM_VLA_CONTRACT
    assert "control_hz" not in vla_runtime_contract()
    assert (
        env_runtime_contract(task_id=0)["execution"]["control_hz"]
        == MODEL_SPEC.control_hz
    )
