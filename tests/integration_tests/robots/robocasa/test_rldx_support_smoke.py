# Copyright 2026 The RPent Authors.
# SPDX-License-Identifier: Apache-2.0

"""Opt-in pinned RLDX load, HTTP RPC and first inference, without a planner."""

import json
import os
import threading
from pathlib import Path

import numpy as np
import pytest


class _DirectEnvRpc:
    def __init__(self, facade):
        self.facade = facade

    def call(self, method, args=(), kwargs=None, **options):
        return getattr(self.facade, method.removeprefix("env."))(
            *args, **(kwargs or {})
        )


@pytest.mark.timeout(300)
def test_pinned_rldx_support_load_and_first_inference(monkeypatch):
    if os.environ.get("RPENT_RUN_RLDX_INTEGRATION") != "1":
        pytest.skip("set RPENT_RUN_RLDX_INTEGRATION=1 to load RLDX on a GPU")
    checkpoint = os.environ.get("RPENT_RLDX_CHECKPOINT")
    if not checkpoint or not Path(checkpoint).is_dir():
        pytest.fail("RPENT_RLDX_CHECKPOINT must point to the downloaded FT checkpoint")

    monkeypatch.setenv("MUJOCO_GL", "egl")
    monkeypatch.setenv("ROBOT_PLATFORM", "ROBOCASA")
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
    monkeypatch.setenv("NO_ALBUMENTATIONS_UPDATE", "1")
    monkeypatch.delenv("RLDX_RESET_SEED", raising=False)
    try:
        import flash_attn  # noqa: F401
    except ImportError:
        monkeypatch.setenv("RLDX_ATTN_IMPL", "sdpa")

    from robots.robocasa import robot_spec
    from robots.robocasa.env_client import RoboCasaEnvClient
    from robots.robocasa.env_server import RoboCasaEnvFacade
    from robots.robocasa.rldx_skill import RLDXSkill
    from robots.robocasa.vla_client import RoboCasaVLAClient
    from robots.robocasa.vla_server import RoboCasaVLAFacade
    from rpent.utils.rpc.http_rpc import HttpRpcClient, HttpRpcServer

    manifest = json.loads(
        (Path(robot_spec.__file__).parent / "eval/target50.json").read_text()
    )
    revision = manifest["dependencies"]["rldx_support"]["revision"]
    vla = RoboCasaVLAFacade(checkpoint, backbone_revision=revision)
    server = HttpRpcServer(("127.0.0.1", 0), vla._dispatch)
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    rpc = HttpRpcClient(
        f"http://127.0.0.1:{server.server_address[1]}", enable_sessions=True
    )
    env = None
    try:
        rpc.call("session.register")
        client = RoboCasaVLAClient(rpc)
        assert client.get_modality_config()["hist_maxlen"] > 0
        env = RoboCasaEnvFacade(task_name="OpenDrawer", split="target", seed=1)
        env_client = RoboCasaEnvClient(
            _DirectEnvRpc(env), expected_meta=env.get_env_meta()
        )
        skill = RLDXSkill(env_client, vla_client=client)
        skill._load()
        language = env_client.get_task_language()
        skill._seed_hist(language)
        actions = client.predict(skill._build_obs(language), {"reset_memory": [True]})
        assert "action.base_motion" in actions
        assert "action.gripper_close" in actions
        for value in actions.values():
            array = np.asarray(value)
            assert array.size > 0
            assert array.shape[0] == 1
            assert np.isfinite(array).all()
        assert isinstance(env_client.check_success(), bool)
    finally:
        if env is not None:
            env.close()
        rpc.close()
        server.shutdown()
        server.server_close()
        worker.join(timeout=10)
        vla.close()
    assert not worker.is_alive()
